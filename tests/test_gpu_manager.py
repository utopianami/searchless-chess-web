import time
from pathlib import Path

import pytest

from gpu_manager import GpuManager, GpuUnavailable


class Worker:
    def is_alive(self):
        return True


def test_activity_renews_30_minute_lease_but_status_does_not():
    manager = GpuManager(Path('.'))
    manager.process, manager.model = Worker(), '9M'
    manager.deadline = time.monotonic() + 60
    before = manager.deadline
    manager.status()
    assert manager.deadline == before
    manager.touch('9M')
    assert 1799 <= manager.deadline - time.monotonic() <= 1800


def test_stale_activity_does_not_restart_gpu():
    manager = GpuManager(Path('.'))
    assert manager.touch('270M')['loaded'] is False
    assert manager.process is None
    assert manager.deadline == 0


def test_other_model_activity_does_not_extend_lease():
    manager = GpuManager(Path('.'))
    manager.process, manager.model = Worker(), '9M'
    manager.deadline = 123
    manager.touch('270M')
    assert manager.deadline == 123


def test_expiry_stops_worker(monkeypatch):
    manager = GpuManager(Path('.'))
    manager.model = '9M'
    manager.deadline = time.monotonic() - 1
    stopped = []
    monkeypatch.setattr(manager, '_stop', lambda: stopped.append(True))
    assert manager.expire()
    assert stopped == [True]


def test_insufficient_vram_never_starts_worker(tmp_path):
    checkpoint = tmp_path / '270M/6400000/params'
    checkpoint.mkdir(parents=True)
    (checkpoint / '_METADATA').touch()
    manager = GpuManager(tmp_path, memory_reader=lambda: {'total_mib': 32000, 'free_mib': 2000})
    with pytest.raises(GpuUnavailable, match='메모리가 부족'):
        manager._start('270M')
    assert manager.process is None


def test_larger_than_gpu_budget_never_starts_worker(tmp_path):
    checkpoint = tmp_path / '270M/6400000/params'
    checkpoint.mkdir(parents=True)
    (checkpoint / '_METADATA').touch()
    manager = GpuManager(tmp_path, memory_reader=lambda: {'total_mib': 4000, 'free_mib': 4000})
    with pytest.raises(GpuUnavailable):
        manager._start('270M')
    assert manager.process is None


def test_web_lifespan_runs_reaper_without_browser_requests(monkeypatch):
    import asyncio
    import server
    calls = []

    class Runtime:
        def expire(self):
            calls.append('expired')

        def unload(self):
            calls.append('unloaded')

    monkeypatch.setattr(server, 'runtime', Runtime())

    async def check():
        async with server.lifespan(server.app):
            await asyncio.sleep(5.2)
            assert 'expired' in calls
        assert calls[-1] == 'unloaded'

    asyncio.run(check())
