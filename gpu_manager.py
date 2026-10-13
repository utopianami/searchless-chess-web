"""One on-demand model worker, a renewable idle lease, and VRAM admission."""
import multiprocessing
import logging
import os
import subprocess
import threading
import time

from gpu_worker import serve
from model_specs import MODEL_SPECS

logger = logging.getLogger('uvicorn.error')


def gpu_memory():
    output = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=memory.total,memory.free',
         '--format=csv,noheader,nounits', '--id=0'], text=True, timeout=5)
    total, free = map(int, output.strip().split(','))
    return {'total_mib': total, 'free_mib': free}


class GpuUnavailable(RuntimeError):
    pass


class GpuManager:
    def __init__(self, checkpoint_root, idle_seconds=None, memory_reader=gpu_memory):
        self.checkpoint_root = checkpoint_root
        self.idle_seconds = int(idle_seconds or os.getenv('CHESS_GPU_IDLE_SECONDS', '1800'))
        self.memory_reader = memory_reader
        self.process = None
        self.connection = None
        self.model = None
        self.deadline = 0
        self.loading = False
        self.lock = threading.RLock()
        self.last_error = None

    def catalog(self):
        state = self.status()
        return [{'id': key, 'label': spec.label, 'policy': spec.policy,
                 'available': (self.checkpoint_root / spec.checkpoint / '_METADATA').is_file(),
                 'loaded': key == state['model'] and state['loaded'],
                 'estimated_vram_mib': spec.estimated_vram_mib}
                for key, spec in MODEL_SPECS.items()]

    def status(self):
        # No GPU initialization or lease renewal just to display status.
        process, model = self.process, self.model
        try:
            alive = process is not None and process.is_alive()
        except (ValueError, AssertionError):
            alive = False  # The worker may have been closed by the reaper.
        return {'model': model if alive else None, 'loaded': alive,
                'loading': self.loading, 'idle_seconds': self.idle_seconds,
                'remaining_seconds': max(0, round(self.deadline - time.monotonic())) if self.model else 0,
                'last_error': self.last_error}

    def touch(self, model):
        with self.lock:
            # A stale page must not resurrect a released worker.
            if self.model == model and self.process is not None and self.process.is_alive():
                self.deadline = time.monotonic() + self.idle_seconds
            return self.status()

    def unload(self):
        with self.lock:
            self._stop()
            return self.status()

    def _stop(self):
        process, connection, model = self.process, self.connection, self.model
        self.process = self.connection = self.model = None
        self.deadline = 0
        self.loading = False
        if process is not None:
            if process.is_alive():
                process.terminate()
            process.join(timeout=10)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
            process.close()
            logger.info('GPU worker released: %s', model)
        if connection is not None:
            connection.close()

    def expire(self):
        if not self.lock.acquire(blocking=False):
            return False
        try:
            if self.model and time.monotonic() >= self.deadline:
                logger.info('GPU lease expired after %s idle seconds', self.idle_seconds)
                self._stop()
                return True
            return False
        finally:
            self.lock.release()

    def _start(self, model):
        self._stop()
        spec = MODEL_SPECS[model]
        if not (self.checkpoint_root / spec.checkpoint / '_METADATA').is_file():
            raise FileNotFoundError(spec.checkpoint)
        try:
            memory = self.memory_reader()
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            raise GpuUnavailable('GPU 상태를 확인할 수 없습니다. 잠시 후 다시 시도해 주세요.') from exc
        # Leave at least 1 GiB free and never assign more than half of this GPU.
        budget = min(memory['total_mib'] // 2, memory['free_mib'] - 1024)
        if spec.estimated_vram_mib > budget:
            raise GpuUnavailable(
                f"GPU 메모리가 부족합니다 (여유 {memory['free_mib']} MiB). 더 작은 모델을 선택하거나 잠시 후 다시 시도해 주세요.")
        context = multiprocessing.get_context('spawn')
        self.connection, child = context.Pipe()
        self.process = context.Process(target=serve, args=(child, self.checkpoint_root, model), daemon=True)
        self.process.start()
        child.close()
        self.model = model
        self.loading = True
        self.deadline = time.monotonic() + self.idle_seconds
        logger.info('GPU worker started: model=%s pid=%s available=%s MiB', model, self.process.pid, memory['free_mib'])

    def analyse(self, model, board):
        with self.lock:
            self.last_error = None
            try:
                if self.model != model or self.process is None or not self.process.is_alive():
                    self._start(model)
                self.connection.send({'fen': board.fen()})
                if not self.connection.poll(120):
                    raise GpuUnavailable('모델 준비 시간이 초과되었습니다. 재시도해 주세요.')
                message = self.connection.recv()
                if 'error' in message:
                    raise GpuUnavailable(message['error'])
                self.loading = False
                self.deadline = time.monotonic() + self.idle_seconds
                return message['result']
            except Exception as exc:
                self.last_error = str(exc)
                self._stop()
                raise
