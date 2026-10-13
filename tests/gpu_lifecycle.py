"""Exercise real CUDA release with a shortened lease, without changing production."""
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import chess
from gpu_manager import GpuManager


def gpu_pids():
    text = subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True)
    return {int(line) for line in text.splitlines() if line.strip().isdigit()}


def main():
    root = Path(__file__).resolve().parents[1] / 'searchless_chess/checkpoints'
    manager = GpuManager(root, idle_seconds=2)
    try:
        first = manager.analyse('9M', chess.Board())
        pid = manager.process.pid
        assert pid in gpu_pids()
        assert manager.analyse('9M', chess.Board()) == first
        assert manager.process.pid == pid
        previous = manager.deadline
        manager.touch('9M')
        assert manager.deadline >= previous
        time.sleep(2.1)
        assert manager.expire()
        assert manager.process is None
        assert pid not in gpu_pids(), 'CUDA context survived worker expiration'
        assert not manager.touch('9M')['loaded']
        manager.analyse('9M', chess.Board())
        new_pid = manager.process.pid
        assert new_pid != pid and new_pid in gpu_pids()
        manager.unload()
        assert new_pid not in gpu_pids()
        print('PASS: same worker reused; activity renewed lease; expiry freed CUDA; stale activity did not reload; next analysis reloaded; manual unload freed CUDA.')
    finally:
        manager.unload()


if __name__ == '__main__':
    main()
