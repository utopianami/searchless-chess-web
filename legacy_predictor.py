"""Flax-format 9M state-value and behavioral-cloning checkpoint support.

These two official archives contain a Flax variables tree (31 FEN tokens,
78 positions), unlike the Haiku action-value archives. Preserve Flax attention
and LayerNorm semantics instead of renaming weights into the Haiku model.
The residual/SwiGLU architecture follows searchless_chess/src/transformer.py.
"""
from flax import linen as nn
import jax
import jax.numpy as jnp

from searchless_chess.src.constants import Predictor


class LegacyTransformer(nn.Module):
    output_size: int

    @nn.compact
    def __call__(self, targets):
        inputs = jnp.concatenate([jnp.zeros((len(targets), 1), dtype=targets.dtype),
                                  targets[:, :-1]], axis=1)
        h = nn.Embed(31, 256)(inputs)
        h += nn.Embed(78, 256)(jnp.arange(inputs.shape[1]))
        for _ in range(8):
            norm = nn.LayerNorm()(h)
            h += nn.MultiHeadDotProductAttention(num_heads=8, use_bias=False,
                                                dropout_rate=0.0)(norm, norm)
            norm = nn.LayerNorm()(h)
            gate = nn.Dense(1024, use_bias=False)(norm)
            value = nn.Dense(1024, use_bias=False)(norm)
            h += nn.Dense(256, use_bias=False)(jax.nn.silu(gate) * value)
        h = nn.LayerNorm()(h)
        return jax.nn.log_softmax(nn.Dense(self.output_size)(h), axis=-1)


def unbox(tree):
    """Remove serialized Flax partition wrappers, keeping every trained array."""
    if isinstance(tree, dict):
        if set(tree) == {'value'}:
            return tree['value']
        return {key: unbox(value) for key, value in tree.items()}
    return tree


def build_predictor(output_size):
    module = LegacyTransformer(output_size)
    return Predictor(initial_params=lambda rng, targets: module.init(rng, targets),
                     predict=lambda params, targets, rng=None: module.apply(params, targets))
