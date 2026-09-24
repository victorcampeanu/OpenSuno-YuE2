"""Runs inside OpenSuno.app: install or update ~/OpenSuno from the bundled payload, then start the Studio.

Progress lines go to stdout for the launcher window; a failure message goes to stderr with a non-zero exit.
"""
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

PAYLOAD = Path(__file__).resolve().parent / 'payload'
HOME = Path(os.environ.get('OPENSUNO_HOME') or Path.home() / 'OpenSuno')
RUNTIME = ('.venv', 'bin')
STUDIO_SERVICE = f'gui/{os.getuid()}/local.opensuno.studio'


def say(text):
    print(text, flush=True)


def code_files():
    for path in PAYLOAD.rglob('*'):
        relative = path.relative_to(PAYLOAD)
        if path.is_file() and relative.parts[0] not in RUNTIME and relative.name != '.app-build':
            yield relative


def stop_studio():
    """A Studio still serving the previous version would keep the old code loaded."""
    def loaded():
        return subprocess.run(['/bin/launchctl', 'print', STUDIO_SERVICE], capture_output=True).returncode == 0
    if loaded():
        say('Stopping the previous Studio…')
        subprocess.run(['/bin/launchctl', 'bootout', STUDIO_SERVICE], capture_output=True)
        # bootout returns before the service is torn down; launch.py must not mistake the dying one for a live Studio.
        deadline = time.monotonic() + 20
        while loaded() and time.monotonic() < deadline:
            time.sleep(0.2)


def replace_tree(source, target):
    fresh = target.with_name(target.name + '.new')
    if fresh.exists():
        shutil.rmtree(fresh)
    shutil.copytree(source, fresh, symlinks=True)
    if target.exists():
        shutil.rmtree(target)
    fresh.rename(target)


def install(build):
    if (HOME / '.git').exists():
        raise SystemExit(f'{HOME} is a source checkout. Move it aside (or set OPENSUNO_HOME) so the app can install there.')
    first = not (HOME / '.app-build').exists()
    say(('Installing OpenSuno into ' if first else 'Updating OpenSuno in ') + str(HOME) + '…')
    HOME.mkdir(parents=True, exist_ok=True)
    stop_studio()
    manifest = HOME / '.app-files'
    previous = set(manifest.read_text().splitlines()) if manifest.is_file() else set()
    current = set()
    for relative in code_files():
        current.add(str(relative))
        target = HOME / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PAYLOAD / relative, target)
    for stale in previous - current:
        try:
            (HOME / stale).unlink()
        except OSError:
            pass
    manifest.write_text('\n'.join(sorted(current)) + '\n')
    say('Copying the Python runtime…')
    replace_tree(PAYLOAD / '.venv', HOME / '.venv')
    # Older builds shipped ffmpeg in bin/; the node does that work now.
    shutil.rmtree(HOME / 'bin', ignore_errors=True)
    # Files copied out of a downloaded disk image inherit its quarantine flag, which would stop them from running.
    subprocess.run(['/usr/bin/xattr', '-dr', 'com.apple.quarantine', str(HOME)], capture_output=True)
    (HOME / '.app-build').write_text(build + '\n')


def main():
    if PAYLOAD.is_dir():
        build = (PAYLOAD / '.app-build').read_text().strip()
        installed = (HOME / '.app-build').read_text().strip() if (HOME / '.app-build').is_file() else ''
        if build != installed or not (HOME / '.venv/bin/python').is_file():
            install(build)
    elif not (HOME / '.venv/bin/python').is_file():
        # The app installed by "Install OpenSuno" carries no payload; the installer put everything in place.
        raise SystemExit(f'OpenSuno is not installed in {HOME}. Run Install OpenSuno again.')
    say('Starting the Studio…')
    python = HOME / '.venv/bin/python'
    os.chdir(HOME)
    os.execv(str(python), [str(python), str(HOME / 'launcher/launch.py')])


if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        raise
    except Exception as error:
        print(re.sub(r'\s+', ' ', str(error)), file=sys.stderr)
        sys.exit(1)
