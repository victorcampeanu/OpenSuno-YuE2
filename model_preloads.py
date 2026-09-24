"""Warm models through the same resident workers used by actual jobs."""
import json
import threading
import time
import uuid

class ModelPreloads:
    def __init__(self, root, pool, lock):
        self.root, self.pool, self.lock = root, pool, lock
        self.states = {}

    def snapshot(self):
        with self.lock:
            result = {}
            for name, item in self.states.items():
                if item['process'].poll() is not None:
                    result[name] = {'status': 'failed', 'detail': 'Runtime stopped. The next job will reload it.'}
                    continue
                state = {k:v for k,v in item.items() if k not in ('process','path')}
                if state['status'] == 'loading':
                    try:
                        progress = json.loads((item['path']/'progress.json').read_text())
                        state['detail'] = progress.get('loading_detail') or 'Preparing model'
                    except (OSError, ValueError): pass
                    state['elapsed'] = round(time.time()-state['started'])
                result[name] = state
            return result

    def start(self, name, request):
        with self.lock:
            previous = self.states.get(name)
            if previous and previous['process'].poll() is None:
                return
            path = self.root/'.preloads'/(name+'-'+uuid.uuid4().hex)
            path.mkdir(parents=True)
            (path/'request.json').write_text(json.dumps({**request, '_preload': True}))
            proc, marker = self.pool.submit(path, name == 'analysis')
            item = {'status':'loading', 'detail':'Preparing runtime', 'started':time.time(), 'process':proc, 'path':path}
            self.states[name] = item
            def finish():
                code = self.pool.wait(proc, marker)
                with self.lock:
                    if self.states.get(name) is item:
                        item.update(status='ready' if code == 0 else 'failed',
                                    detail='Ready in memory' if code == 0 else 'Preload failed; generation will retry loading.')
            threading.Thread(target=finish, daemon=True).start()

    def mark_ready(self, name, process, path):
        with self.lock:
            if process.poll() is None:
                self.states[name] = {'status':'ready', 'detail':'Ready in memory',
                                     'started':time.time(), 'process':process, 'path':path}

    def clear(self):
        with self.lock: self.states.clear()
