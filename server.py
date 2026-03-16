"""Chess web server with searchless_chess 9M model."""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax.sharding
if not hasattr(jax.sharding, 'PositionalSharding'):
    jax.sharding.PositionalSharding = type('PositionalSharding', (), {})
import chess
import numpy as np
from jax import random as jrandom
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from searchless_chess.src import tokenizer
from searchless_chess.src import training_utils
from searchless_chess.src import transformer
from searchless_chess.src import utils
from searchless_chess.src.engines import engine as eng_module
from searchless_chess.src.engines import neural_engines

CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'searchless_chess', 'checkpoints', '9M')

print("Loading 9M model...")
predictor_config = transformer.TransformerConfig(
    vocab_size=utils.NUM_ACTIONS,
    output_size=128,
    pos_encodings=transformer.PositionalEncodings.LEARNED,
    max_sequence_length=tokenizer.SEQUENCE_LENGTH + 2,
    num_heads=8,
    num_layers=8,
    embedding_dim=256,
    apply_post_ln=True,
    apply_qk_layernorm=False,
    use_causal_mask=False,
)
predictor = transformer.build_transformer_predictor(config=predictor_config)
params = training_utils.load_parameters(
    checkpoint_dir=CHECKPOINT_DIR,
    params=predictor.initial_params(
        rng=jrandom.PRNGKey(1),
        targets=np.ones((1, 1), dtype=np.uint32),
    ),
    step=6_400_000,
)
_, return_buckets_values = utils.get_uniform_buckets_edges_values(128)
engine = neural_engines.ActionValueEngine(
    return_buckets_values=return_buckets_values,
    predict_fn=neural_engines.wrap_predict_fn(predictor=predictor, params=params, batch_size=1),
)
print("Model loaded!")

app = FastAPI()


class AnalyseRequest(BaseModel):
    fen: str


class MoveRequest(BaseModel):
    fen: str


@app.post("/api/analyse")
def analyse(req: AnalyseRequest):
    """Return move probabilities and position evaluation."""
    board = chess.Board(req.fen)

    if board.is_game_over():
        result = board.result()
        if result == "1-0":
            white_win = 1.0
        elif result == "0-1":
            white_win = 0.0
        else:
            white_win = 0.5
        return {"moves": [], "white_win_prob": white_win, "game_over": True, "result": result}

    analysis = engine.analyse(board)
    log_probs = analysis['log_probs']
    probs = np.exp(log_probs)
    win_probs = np.inner(probs, return_buckets_values)

    sorted_legal_moves = eng_module.get_ordered_legal_moves(board)

    # win_probs is from current player's perspective
    # Convert to white's perspective
    if board.turn == chess.WHITE:
        white_win_prob = float(np.max(win_probs))
    else:
        white_win_prob = 1.0 - float(np.max(win_probs))

    moves = []
    for i, move in enumerate(sorted_legal_moves):
        wp = float(win_probs[i])
        moves.append({
            "uci": move.uci(),
            "from": chess.square_name(move.from_square),
            "to": chess.square_name(move.to_square),
            "prob": wp,
        })

    # Sort by probability descending
    moves.sort(key=lambda m: m["prob"], reverse=True)

    # Normalize probabilities for display (softmax-style relative to best)
    if moves:
        max_p = moves[0]["prob"]
        min_p = moves[-1]["prob"]
        rng = max_p - min_p if max_p != min_p else 1.0
        for m in moves:
            m["norm_prob"] = (m["prob"] - min_p) / rng

    return {"moves": moves, "white_win_prob": white_win_prob, "game_over": False}


@app.post("/api/move")
def ai_move(req: MoveRequest):
    """AI picks and returns its best move."""
    board = chess.Board(req.fen)

    if board.is_game_over():
        return {"error": "game_over"}

    move = engine.play(board)
    return {"uci": move.uci(), "from": chess.square_name(move.from_square), "to": chess.square_name(move.to_square)}


@app.get("/")
def index():
    return FileResponse("static/index.html")


app.mount("/static", StaticFiles(directory="static"), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8888)
