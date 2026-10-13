"""Searchless Chess web API with per-request model selection."""
import logging
import asyncio
from contextlib import asynccontextmanager, suppress
from pathlib import Path

import chess
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from model_specs import DEFAULT_MODEL, MODEL_SPECS
from gpu_manager import GpuManager, GpuUnavailable

ROOT = Path(__file__).resolve().parent
runtime = GpuManager(ROOT / 'searchless_chess' / 'checkpoints')


@asynccontextmanager
async def lifespan(app):
    async def reaper():
        while True:
            await asyncio.sleep(5)
            await asyncio.to_thread(runtime.expire)
    task = asyncio.create_task(reaper())
    try:
        yield
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
        await asyncio.to_thread(runtime.unload)


app = FastAPI(lifespan=lifespan)
logger = logging.getLogger(__name__)


class PositionRequest(BaseModel):
    fen: str = Field(min_length=1, max_length=200)
    model: str = DEFAULT_MODEL


def analyse_position(req):
    if req.model not in MODEL_SPECS:
        raise HTTPException(422, 'Unknown model')
    try:
        board = chess.Board(req.fen)
        if not board.is_valid():
            raise ValueError('Invalid board')
    except ValueError as exc:
        raise HTTPException(422, 'Invalid FEN position') from exc
    if board.is_game_over():
        result = board.result()
        return {'model': req.model, 'moves': [], 'game_over': True, 'result': result,
                'score_type': 'win_probability',
                'white_win_prob': {'1-0': 1.0, '0-1': 0.0}.get(result, 0.5)}
    try:
        return runtime.analyse(req.model, board)
    except FileNotFoundError as exc:
        raise HTTPException(503, 'Model checkpoint is not installed') from exc
    except GpuUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
    except Exception as exc:
        logger.exception('Model inference failed: %s', req.model)
        raise HTTPException(503, 'Model inference failed; please retry') from exc


@app.get('/api/models')
async def models():
    return {'default_model': DEFAULT_MODEL, 'models': runtime.catalog()}


@app.get('/healthz')
async def health():
    return {'status': 'ok', 'models_available': all(m['available'] for m in runtime.catalog())}


@app.post('/api/analyse')
async def analyse(req: PositionRequest):
    return await asyncio.to_thread(analyse_position, req)


@app.post('/api/move')
async def ai_move(req: PositionRequest):
    result = await asyncio.to_thread(analyse_position, req)
    if result['game_over']:
        return {'error': 'game_over', 'model': req.model}
    move = result['moves'][0]
    return {key: move[key] for key in ('uci', 'from', 'to')} | {'model': req.model}


@app.get('/')
async def index():
    return FileResponse(ROOT / 'static' / 'index.html')


app.mount('/static', StaticFiles(directory=ROOT / 'static'), name='static')


class ActivityRequest(BaseModel):
    model: str = DEFAULT_MODEL


@app.get('/api/gpu')
async def gpu_status():
    return runtime.status()


@app.post('/api/gpu/activity')
async def gpu_activity(req: ActivityRequest):
    if req.model not in MODEL_SPECS:
        raise HTTPException(422, 'Unknown model')
    return await asyncio.to_thread(runtime.touch, req.model)


@app.post('/api/gpu/unload')
async def gpu_unload():
    return await asyncio.to_thread(runtime.unload)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8888)
