"""Load official checkpoints and normalize each policy's distinct outputs."""
import logging
import os
from collections import OrderedDict
from threading import Lock

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

import chess
import jax
import numpy as np
import orbax.checkpoint as ocp

from searchless_chess.src import tokenizer, transformer, utils
from searchless_chess.src.engines import engine as engine_module
from searchless_chess.src.engines import neural_engines

from model_specs import DEFAULT_MODEL, MODEL_SPECS


class ModelRuntime:
    def __init__(self, checkpoint_root):
        self.checkpoint_root = checkpoint_root
        self.engines = {}
        self.cache = OrderedDict()
        self.lock = Lock()
        _, self.buckets = utils.get_uniform_buckets_edges_values(128)

    def catalog(self):
        return [
            {'id': key, 'label': spec.label, 'policy': spec.policy,
             'available': (self.checkpoint_root / spec.checkpoint / '_METADATA').is_file(),
             'loaded': key in self.engines}
            for key, spec in MODEL_SPECS.items()
        ]

    def load(self, model):
        # Called under the inference lock to avoid duplicate GPU allocation.
        if model in self.engines:
            return self.engines[model]
        spec = MODEL_SPECS[model]
        path = self.checkpoint_root / spec.checkpoint
        if not (path / '_METADATA').is_file():
            raise FileNotFoundError(path)
        logging.info('Loading model %s on %s', model, jax.devices())
        config = transformer.TransformerConfig(
            vocab_size=utils.NUM_ACTIONS,
            output_size=utils.NUM_ACTIONS if spec.policy == 'behavioral_cloning' else 128,
            pos_encodings=transformer.PositionalEncodings.LEARNED,
            max_sequence_length=tokenizer.SEQUENCE_LENGTH + 2,
            num_heads=8, num_layers=spec.layers, embedding_dim=spec.width,
            apply_post_ln=True, apply_qk_layernorm=False, use_causal_mask=False,
        )
        predictor = transformer.build_transformer_predictor(config=config)
        if spec.policy != 'action_value':
            from legacy_predictor import build_predictor, unbox
            predictor = build_predictor(config.output_size)
        # Shape-only initialization avoids allocating random large model weights.
        abstract = jax.eval_shape(predictor.initial_params,
                                  rng=jax.random.PRNGKey(1),
                                  targets=np.ones((1, 1), dtype=np.uint32))
        sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
        restore_args = jax.tree.map(
            lambda p: ocp.ArrayRestoreArgs(restore_type=jax.Array, dtype=p.dtype,
                                           global_shape=p.shape, sharding=sharding), abstract)
        with ocp.Checkpointer(ocp.PyTreeCheckpointHandler()) as checkpointer:
            if spec.policy == 'action_value':
                params = checkpointer.restore(path, item=abstract, restore_args=restore_args)
            else:
                metadata = checkpointer.metadata(path).item_metadata.tree
                legacy_args = jax.tree.map(
                    lambda p: ocp.ArrayRestoreArgs(restore_type=jax.Array, dtype=p.dtype,
                                                  global_shape=p.shape, sharding=sharding), metadata)
                params = unbox(checkpointer.restore(path, restore_args=legacy_args))
                if jax.tree.structure(params) != jax.tree.structure(abstract):
                    raise ValueError('Legacy checkpoint tree does not match the predictor')
                for actual, expected in zip(jax.tree.leaves(params), jax.tree.leaves(abstract)):
                    if actual.shape != expected.shape:
                        raise ValueError(f'Legacy parameter shape mismatch: {actual.shape} != {expected.shape}')
        engine = neural_engines.ENGINE_FROM_POLICY[spec.policy](
            return_buckets_values=self.buckets,
            predict_fn=neural_engines.wrap_predict_fn(predictor, params, batch_size=16),
        )
        self.engines[model] = engine
        return engine

    def analyse(self, model, board):
        spec = MODEL_SPECS[model]
        if board.is_game_over():
            result = board.result()
            return {'model': model, 'moves': [], 'game_over': True, 'result': result,
                    'score_type': 'win_probability',
                    'white_win_prob': {'1-0': 1.0, '0-1': 0.0}.get(result, 0.5)}
        key = (model, board.fen())
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]
            analysis = self.load(model).analyse(board)
            ordered = engine_module.get_ordered_legal_moves(board)
            if spec.policy == 'behavioral_cloning':
                scores = np.exp(np.asarray(analysis['log_probs']))
                white_win = None
                score_type = 'move_probability'
            else:
                log_probs = analysis['next_log_probs' if spec.policy == 'state_value' else 'log_probs']
                scores = np.inner(np.exp(log_probs), self.buckets)
                for i, move in enumerate(ordered):
                    board.push(move)
                    if board.is_game_over():
                        scores[i] = 1.0 if board.is_checkmate() else 0.5
                    board.pop()
                value = float(np.max(scores))
                white_win = value if board.turn == chess.WHITE else 1.0 - value
                score_type = 'win_probability'
            moves = [dict(uci=m.uci(), **{'from': chess.square_name(m.from_square),
                     'to': chess.square_name(m.to_square)}, prob=float(scores[i]))
                     for i, m in enumerate(ordered)]
            moves.sort(key=lambda m: m['prob'], reverse=True)
            low, high = moves[-1]['prob'], moves[0]['prob']
            for move in moves:
                move['norm_prob'] = (move['prob'] - low) / (high - low) if high > low else 1.0
            result = {'model': model, 'moves': moves, 'white_win_prob': white_win,
                      'score_type': score_type, 'game_over': False}
            self.cache[key] = result
            if len(self.cache) > 256:
                self.cache.popitem(last=False)
            return result
