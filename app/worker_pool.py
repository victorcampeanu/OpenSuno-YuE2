"""Resident subprocesses and a page-connection lease for their lifetime."""
import json
import os
import signal
import subprocess
import threading
import time
import uuid
import sys
from model_registry import is_cuda
from runtime_platform import environment_python, stop_process

class WorkerPool:
    def __init__(self, root):
        self.root = root
        self.workers = {}

    def submit(self, jobdir, analyze):
        request = json.loads((jobdir / 'request.json').read_text())
        key = request['model'] if not analyze and is_cuda(request.get('model')) else analyze
        if sys.platform == 'win32':
            for old_key, old in list(self.workers.items()):
                if old_key != key:
                    self.discard(old)
                    old.wait(timeout=15)
        proc = self.workers.get(key)
        if proc is None or proc.poll() is not None:
            python = environment_python(self.root, analyze, environment='.cuda-venv' if is_cuda(key) else None)
            env = {**os.environ, 'PYTHONUNBUFFERED': '1', 'PYTORCH_ENABLE_MPS_FALLBACK': '1',
                   'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                   'YUE2_PARENT_PID': str(os.getpid()), 'PYTHONUTF8': '1'}
            proc = subprocess.Popen([str(python), str(self.root / 'app/resident_worker.py')],
                                    stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL, text=True, env=env, start_new_session=True)
            self.workers[key] = proc
        marker = jobdir / ('.resident-' + uuid.uuid4().hex + '.done')
        proc.stdin.write(json.dumps({'jobdir': str(jobdir), 'analyze': analyze, 'marker': marker.name}) + '\n')
        proc.stdin.flush()
        return proc, marker

    @staticmethod
    def wait(proc, marker):
        try:
            while True:
                if marker.exists():
                    return json.loads(marker.read_text())['code']
                if proc.poll() is not None:
                    return proc.returncode or 1
                time.sleep(.1)
        finally:
            marker.unlink(missing_ok=True)

    def discard(self, proc):
        for key, worker in list(self.workers.items()):
            if worker is proc:
                del self.workers[key]
        self.stop(proc)

    @staticmethod
    def stop(proc):
        if proc.poll() is None:
            stop_process(proc)
        def reap():
            try: proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                stop_process(proc, force=True)
                proc.wait()
            if proc.stdin: proc.stdin.close()
        threading.Thread(target=reap, daemon=True).start()

    def close(self):
        workers, self.workers = self.workers, {}
        for proc in workers.values(): self.stop(proc)

class PageLeases:
    """A persistent HTTP connection survives background tabs; refresh gets a grace period."""
    def __init__(self, lock, on_empty, grace=20):
        self.lock, self.on_empty, self.grace = lock, on_empty, grace
        self.pages = set()
        self.timer = None
        self.generation = 0

    def open(self):
        with self.lock:
            key = uuid.uuid4().hex
            self.generation += 1
            self.pages.add(key)
            if self.timer: self.timer.cancel(); self.timer = None
            return key

    def close(self, key):
        with self.lock:
            self.pages.discard(key)
            if not self.pages:
                if self.timer: self.timer.cancel()
                self.generation += 1
                self.timer = threading.Timer(self.grace, self.expire, args=(self.generation,))
                self.timer.daemon = True
                self.timer.start()

    def expire(self, generation):
        with self.lock:
            if generation != self.generation: return
            if not self.pages:
                self.on_empty()
            self.timer = None

    def shutdown(self):
        with self.lock:
            self.generation += 1
            if self.timer: self.timer.cancel(); self.timer = None
            self.pages.clear()
