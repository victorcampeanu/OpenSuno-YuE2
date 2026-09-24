"""Run batches in one interpreter, retaining imported runtimes and model caches."""
import json
import os
from pathlib import Path
import runpy
import signal
import sys
import threading
import time
import traceback
from runtime_platform import watch_parent

ROOT = Path(__file__).resolve().parent
class JobCancelled(BaseException):
    pass

def main():
    threading.Thread(target=watch_parent, daemon=True).start()
    cache = {}
    for line in sys.stdin:
        command = json.loads(line)
        jobdir = Path(command['jobdir'])
        marker = jobdir / command['marker']
        sys.stdout.flush(); sys.stderr.flush()
        with (jobdir / 'run.log').open('a') as log:
            os.dup2(log.fileno(), 1); os.dup2(log.fileno(), 2)
        sys.argv = [str(ROOT / 'worker.py'), str(jobdir)] + (['--analyze'] if command['analyze'] else [])
        code = 0
        def check_cancel():
            if (jobdir / '.cancel-resident').exists(): raise JobCancelled()
        try:
            runpy.run_path(str(ROOT / 'worker.py'), run_name='__main__',
                          init_globals={'MODEL_CACHE': cache, 'CHECK_CANCEL': check_cancel})
        except JobCancelled:
            print('[cancel] Stopped; loaded models retained',flush=True)
            code = 2
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else (1 if exc.code else 0)
        except BaseException as exc:
            (jobdir/'runtime-error.json').write_text(json.dumps({'error':str(exc)}))
            traceback.print_exc()
            code = 1
        finally:
            sys.stdout.flush(); sys.stderr.flush()
            temp = marker.with_suffix('.tmp')
            temp.write_text(json.dumps({'code': code}))
            temp.replace(marker)
        if code == 1:
            return 1  # A failed runtime is discarded, rather than reused in an unknown state.
    return 0

if __name__ == '__main__':
    sys.exit(main())
