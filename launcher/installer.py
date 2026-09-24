"""Runs inside "Install OpenSuno.app" with the Python runtime the app just unpacked.

    python installer.py [--studio] [--node] [--analysis] [--login]

Installs the chosen parts into ~/OpenSuno (code, Python environments, ffmpeg) and the apps into
/Applications (or ~/Applications when that is not writable), then starts what was installed.
Progress lines go to stdout for the installer window; a failure goes to stderr with exit code 1.
The Studio and the render node share one ~/OpenSuno: the node's environment is a superset of the Studio's.
"""
import argparse
import json
import os
import plistlib
import re
import secrets
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile
from pathlib import Path

PAYLOAD = Path(__file__).resolve().parent / 'payload'
HOME = Path(os.environ.get('OPENSUNO_HOME') or Path.home() / 'OpenSuno')
SUPPORT = Path.home() / 'Library/Application Support/OpenSuno'
NODE_CONFIG = SUPPORT / 'render-node.json'
STUDIO_SERVICE = f'gui/{os.getuid()}/local.opensuno.studio'
NODE_SERVICE = f'gui/{os.getuid()}/local.opensuno.rendernode'
STUDIO_APP = 'OpenSuno.app'
NODE_APP = 'OpenSuno Render Node.app'
NODE_EXECUTABLE = 'OpenSunoRenderNode'


def say(text):
    print(text, flush=True)


def run(*args, **kw):
    return subprocess.run([str(a) for a in args], check=True, **kw)


def launchctl(*args):
    return subprocess.run(['/bin/launchctl', *args], capture_output=True, text=True)


def stop_service(service, what):
    if launchctl('print', service).returncode != 0: return
    say('Stopping the running ' + what + '…')
    launchctl('bootout', service)
    deadline = time.monotonic() + 20
    while launchctl('print', service).returncode == 0 and time.monotonic() < deadline:
        time.sleep(0.2)


def replace_tree(source, target):
    fresh = target.with_name(target.name + '.new')
    if fresh.exists(): shutil.rmtree(fresh)
    shutil.copytree(source, fresh, symlinks=True)
    if target.exists(): shutil.rmtree(target)
    fresh.rename(target)


def extract_runtime(archive, target):
    """python-build-standalone unpacks as ``python/``; it becomes the environment folder itself."""
    if target.exists(): shutil.rmtree(target)
    staging = target.with_name(target.name + '.new')
    if staging.exists(): shutil.rmtree(staging)
    staging.mkdir()
    with tarfile.open(archive) as tar:
        tar.extractall(staging)
    (staging / 'python').rename(target)
    staging.rmdir()


def pip_install(python, requirements):
    """Stream pip's progress as short lines; the packages come from PyPI, so this is the step that needs internet."""
    process = subprocess.Popen([str(python), '-m', 'pip', 'install', '--no-warn-script-location', '--no-compile',
                                '--progress-bar', 'off', '-r', str(requirements)],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        line = line.strip()
        if line.startswith(('Collecting', 'Downloading', 'Installing collected')):
            say(('Downloading ' + line.split()[1].split('==')[0]) if line.startswith('Collecting') else line[:120])
    if process.wait():
        raise SystemExit('Installing Python packages failed. Check the internet connection and run the installer again.')


def copy_code():
    say('Copying OpenSuno into ' + str(HOME) + '…')
    HOME.mkdir(parents=True, exist_ok=True)
    manifest = HOME / '.app-files'
    previous = set(manifest.read_text().splitlines()) if manifest.is_file() else set()
    current = set()
    for path in (PAYLOAD / 'code').rglob('*'):
        if not path.is_file(): continue
        relative = path.relative_to(PAYLOAD / 'code')
        current.add(str(relative))
        target = HOME / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    for stale in previous - current:
        try: (HOME / stale).unlink()
        except OSError: pass
    manifest.write_text('\n'.join(sorted(current)) + '\n')


def install_python(full):
    say('Installing the Python runtime…')
    extract_runtime(PAYLOAD / 'runtime/python312.tar.gz', HOME / '.venv')
    say('Installing Python packages' + (' for rendering (MLX)…' if full else '…'))
    pip_install(HOME / '.venv/bin/python', HOME / ('requirements-installed.txt' if full else 'requirements.txt'))
    run(HOME / '.venv/bin/python', '-c', 'import fastapi, uvicorn, requests, multipart' + (', mlx.core, soundfile, tiktoken' if full else ''))


def install_ffmpeg():
    say('Installing ffmpeg…')
    target = HOME / 'bin'
    fresh = HOME / 'bin.new'
    if fresh.exists(): shutil.rmtree(fresh)
    fresh.mkdir()
    with zipfile.ZipFile(PAYLOAD / 'runtime/ffmpeg.zip') as archive:
        for name in ('ffmpeg', 'ffprobe'):
            (fresh / name).write_bytes(archive.read('darwin_arm64/' + name))
            (fresh / name).chmod(0o755)
    if target.exists(): shutil.rmtree(target)
    fresh.rename(target)


def install_analysis():
    say('Installing the cover analysis environment (Python 3.11)…')
    extract_runtime(PAYLOAD / 'runtime/python311.tar.gz', HOME / '.transcribe-venv')
    say('Installing analysis packages (PyTorch, transformers)…')
    pip_install(HOME / '.transcribe-venv/bin/python', HOME / 'requirements-transcriber.txt')
    run(HOME / '.transcribe-venv/bin/python', '-c', 'import torch, torchaudio, transformers, scipy, pretty_midi')


def applications_folder():
    """/Applications when this user may write there (admin), else the user's own Applications folder."""
    preferred = [Path(os.environ['OPENSUNO_APPLICATIONS'])] if os.environ.get('OPENSUNO_APPLICATIONS') else []
    for folder in preferred + [Path('/Applications'), Path.home() / 'Applications']:
        try:
            folder.mkdir(exist_ok=True)
            probe = folder / ('.opensuno-write-test-' + secrets.token_hex(3))
            probe.touch(); probe.unlink()
            return folder
        except OSError:
            continue
    raise SystemExit('No Applications folder is writable.')


def install_app(name, configure=None):
    source = PAYLOAD / 'apps' / name
    folder = applications_folder()
    target = folder / name
    say('Installing ' + name.removesuffix('.app') + ' into ' + str(folder) + '…')
    staging = folder / ('.' + name + '.new')
    if staging.exists(): shutil.rmtree(staging)
    shutil.copytree(source, staging, symlinks=True)
    if configure: configure(staging)
    # Ad-hoc signing must be redone after Info.plist changes; codesign ships with macOS.
    run('/usr/bin/codesign', '--force', '--sign', '-', staging, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if target.exists(): shutil.rmtree(target)
    staging.rename(target)
    subprocess.run(['/usr/bin/xattr', '-dr', 'com.apple.quarantine', str(target)], capture_output=True)
    return target


def node_config(login):
    """The token Studios pair with; kept when it exists already."""
    SUPPORT.mkdir(parents=True, exist_ok=True)
    try: config = json.loads(NODE_CONFIG.read_text())
    except (OSError, ValueError): config = {}
    if not config.get('token'): config['token'] = secrets.token_urlsafe(24)
    config.setdefault('port', 7863)
    config.setdefault('idleMinutes', 30)
    config['launchAtLogin'] = bool(login)
    NODE_CONFIG.write_text(json.dumps(config, indent=2, sort_keys=True))
    NODE_CONFIG.chmod(0o600)
    return config


def point_node_app_at_home(app):
    info = app / 'Contents/Info.plist'
    with info.open('rb') as f: data = plistlib.load(f)
    data['OpenSunoRoot'] = str(HOME)
    with info.open('wb') as f: plistlib.dump(data, f)


def main():
    parser = argparse.ArgumentParser()
    for flag in ('studio', 'node', 'analysis', 'login'): parser.add_argument('--' + flag, action='store_true')
    args = parser.parse_args()
    if not (args.studio or args.node): raise SystemExit('Choose the Studio, the render node or both.')
    if (HOME / '.git').exists():
        raise SystemExit(str(HOME) + ' is a source checkout. Move it aside so the installer can use that folder.')
    stop_service(STUDIO_SERVICE, 'Studio')
    stop_service(NODE_SERVICE, 'render node')
    subprocess.run(['/usr/bin/pkill', '-x', NODE_EXECUTABLE], capture_output=True)
    copy_code()
    install_python(full=args.node)
    if args.node:
        install_ffmpeg()
        if args.analysis: install_analysis()
    subprocess.run(['/usr/bin/xattr', '-dr', 'com.apple.quarantine', str(HOME)], capture_output=True)
    (HOME / '.app-build').write_text((PAYLOAD / '.app-build').read_text())
    installed = []
    if args.node:
        node_config(args.login)
        installed.append(install_app(NODE_APP, point_node_app_at_home))
    if args.studio:
        installed.append(install_app(STUDIO_APP))
    if os.environ.get('OPENSUNO_INSTALL_START', '1') != '0':
        say('Starting…')
        for app in installed:
            subprocess.run(['/usr/bin/open', str(app)], capture_output=True)
    say('Done')


if __name__ == '__main__':
    try:
        main()
    except SystemExit as error:
        if error.code not in (None, 0):
            print(str(error.code) if not isinstance(error.code, int) else 'The installer stopped.', file=sys.stderr)
            sys.exit(1)
    except subprocess.CalledProcessError as error:
        print('A step failed: ' + ' '.join(str(a) for a in error.cmd[:3]), file=sys.stderr)
        sys.exit(1)
    except Exception as error:
        print(re.sub(r'\s+', ' ', str(error)), file=sys.stderr)
        sys.exit(1)
