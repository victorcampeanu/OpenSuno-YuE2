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
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent.parent
PORT = int(os.environ.get('YUE2_PORT', '7862'))
URL = f'http://127.0.0.1:{PORT}/'
LABEL = 'local.opensuno.studio'
NODE_LABEL = 'local.opensuno.rendernode'
SUPPORT = Path.home() / 'Library/Application Support/OpenSuno'
LOGS = Path.home() / 'Library/Logs/OpenSuno'


def studio_config():
    try:
        with urllib.request.urlopen(URL + 'api/config', timeout=1) as response:
            data = json.load(response)
        return data if isinstance(data.get('models'), list) and 'defaults' in data else None
    except (OSError, ValueError):
        return None


def ready():
    return studio_config() is not None


def command(*args):
    return subprocess.run(args, capture_output=True, text=True)


def start_service(label, plist, settings, name):
    domain = f'gui/{os.getuid()}'
    service = f'{domain}/{label}'
    existing = command('/bin/launchctl', 'print', service)
    if not existing.returncode and re.search(r'\bpid = \d+', existing.stdout):
        return
    plist.parent.mkdir(parents=True, exist_ok=True)
    with plist.open('wb') as output:
        plistlib.dump(settings, output)
    plist.chmod(0o600)
    if not existing.returncode:
        result = command('/bin/launchctl', 'bootout', service)
        if result.returncode:
            raise RuntimeError(f'Could not stop the previous {name} service: ' + result.stderr.strip())
    result = command('/bin/launchctl', 'bootstrap', domain, str(plist))
    if result.returncode:
        raise RuntimeError(f'Could not start {name}: ' + result.stderr.strip())
    # Render nodes with Start at login off have RunAtLoad=false.
    result = command('/bin/launchctl', 'kickstart', service)
    if result.returncode:
        raise RuntimeError(f'Could not start {name}: ' + result.stderr.strip())


def start_local_node(config):
    """Start only a loopback renderer explicitly selected by this Studio."""
    if config is not None:
        settings = config.get('render_node') or {}
    elif 'OPENSUNO_RENDER_URL' in os.environ:
        settings = {'url': os.environ['OPENSUNO_RENDER_URL'].strip().rstrip('/'),
                    'token': os.environ.get('OPENSUNO_RENDER_TOKEN', '').strip()}
    else:
        data_root = Path(os.environ.get('OPENSUNO_DATA') or ROOT)
        try: settings = json.loads((data_root / '.render-node.json').read_text())
        except (OSError, ValueError): settings = {}
    address = str(settings.get('url') or '').strip().rstrip('/')
    parsed = urlsplit(address)
    if parsed.scheme != 'http' or parsed.hostname not in {'127.0.0.1', 'localhost'}:
        return
    port = parsed.port or 80
    token = str(settings.get('token') or '')

    def node_ready():
        req = urllib.request.Request(address + '/v1/health',
                                     headers={'Authorization': 'Bearer ' + token} if token else {})
        try:
            with urllib.request.urlopen(req, timeout=1) as response:
                data = json.load(response)
            return data.get('ok') and data.get('node') == 'opensuno-render-node' and data.get('protocol') == 2
        except (OSError, ValueError):
            return False

    if node_ready():
        return
    if not (ROOT / '.venv/bin/python').is_file():
        raise RuntimeError('The configured local render node needs scripts/Install OpenSuno.command first.')
    LOGS.mkdir(parents=True, exist_ok=True)
    plist = Path.home() / 'Library/LaunchAgents' / (NODE_LABEL + '.plist')
    try: previous = plistlib.loads(plist.read_bytes())
    except (OSError, ValueError): previous = {}
    same_root = previous.get('WorkingDirectory') == str(ROOT)
    env = previous.get('EnvironmentVariables', {}) if same_root else {}
    settings = {
        'Label': NODE_LABEL,
        'ProgramArguments': [str(ROOT / '.venv/bin/python'), '-u', str(ROOT / 'app/render_node.py')],
        'WorkingDirectory': str(ROOT),
        'EnvironmentVariables': {**env,
            'PATH': str(ROOT / 'bin') + ':/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin',
            'PYTHONUNBUFFERED': '1', 'OPENSUNO_NODE_PORT': str(port), 'OPENSUNO_NODE_TOKEN': token},
        'RunAtLoad': previous.get('RunAtLoad', False) if same_root else False,
        'KeepAlive': {'SuccessfulExit': False},
        'ProcessType': 'Interactive',
        'StandardOutPath': str(LOGS / 'render-node.log'),
        'StandardErrorPath': str(LOGS / 'render-node.log'),
    }
    start_service(NODE_LABEL, plist, settings, 'the local render node')
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if node_ready():
            return
        time.sleep(0.3)
    raise RuntimeError(f'The local render node did not answer on port {port}. Check its port/token in Settings → Rendering and {LOGS / "render-node.log"}.')


def start_server():
    SUPPORT.mkdir(parents=True, exist_ok=True)
    with (SUPPORT / 'launcher.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        config = studio_config()
        start_local_node(config)
        if config is not None:
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
                f'port={PORT}, timeout_graceful_shutdown=2, access_log=False)'],
            'WorkingDirectory': str(ROOT),
            # A source install renders here and finds its Homebrew ffmpeg; the app build leaves that to the node.
            'EnvironmentVariables': {
                'PATH': str(ROOT / 'bin') + ':/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin',
                'PYTHONUNBUFFERED': '1', 'YUE2_PORT': str(PORT),
                **{key: os.environ[key] for key in ('OPENSUNO_DATA', 'OPENSUNO_RENDER_URL', 'OPENSUNO_RENDER_TOKEN') if key in os.environ}},
            'RunAtLoad': True,
            'ProcessType': 'Interactive',
            'StandardOutPath': str(LOGS / 'server.log'),
            'StandardErrorPath': str(LOGS / 'server.log'),
        }
        start_service(LABEL, plist, settings, 'the background server')
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
