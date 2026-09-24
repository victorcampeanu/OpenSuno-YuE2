"""Start the local studio through launchd, then open its browser interface."""
import fcntl
import json
import os
from pathlib import Path
import plistlib
import re
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parent.parent
URL = 'http://127.0.0.1:7862/'
LABEL = 'local.opensuno.studio'
SUPPORT = Path.home() / 'Library/Application Support/OpenSuno'
LOGS = Path.home() / 'Library/Logs/OpenSuno'


def ready():
    try:
        with urllib.request.urlopen(URL + 'api/config', timeout=1) as response:
            data = json.load(response)
        return isinstance(data.get('models'), list) and 'defaults' in data
    except (OSError, ValueError):
        return False


def command(*args):
    return subprocess.run(args, capture_output=True, text=True)


def start_server():
    SUPPORT.mkdir(parents=True, exist_ok=True)
    with (SUPPORT / 'launcher.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if ready():
            return
        python = ROOT / '.venv/bin/python'
        if not python.is_file():
            raise RuntimeError('Run scripts/Install OpenSuno.command in the project folder first.')
        LOGS.mkdir(parents=True, exist_ok=True)
        plist = SUPPORT / 'server.plist'
        settings = {
            'Label': LABEL,
            'ProgramArguments': [str(python), '-u', '-c',
                'import sys; sys.path.insert(0, "app"); import server, uvicorn; uvicorn.run(server.app, host="127.0.0.1", '
                'port=7862, timeout_graceful_shutdown=2, access_log=False)'],
            'WorkingDirectory': str(ROOT),
            # A source install renders here and finds its Homebrew ffmpeg; the app build leaves that to the node.
            'EnvironmentVariables': {'PATH': '/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin', 'PYTHONUNBUFFERED': '1'},
            'RunAtLoad': True,
            'ProcessType': 'Interactive',
            'StandardOutPath': str(LOGS / 'server.log'),
            'StandardErrorPath': str(LOGS / 'server.log'),
        }
        with plist.open('wb') as output:
            plistlib.dump(settings, output)
        plist.chmod(0o600)
        domain = f'gui/{os.getuid()}'
        service = f'{domain}/{LABEL}'
        existing = command('/bin/launchctl', 'print', service)
        if existing.returncode:
            result = command('/bin/launchctl', 'bootstrap', domain, str(plist))
        elif not re.search(r'\bpid = \d+', existing.stdout):
            result = command('/bin/launchctl', 'kickstart', service)
        else:
            result = existing
        if result.returncode:
            raise RuntimeError('Could not start the background server: ' + result.stderr.strip())
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if ready():
                return
            time.sleep(0.3)
        raise RuntimeError(f'Studio did not become ready. See {LOGS / "server.log"}.')


def main():
    try:
        start_server()
        if '--no-browser' not in sys.argv:
            subprocess.run(['/usr/bin/open', URL], check=True)
        print('OpenSuno is ready at ' + URL)
    except Exception as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
