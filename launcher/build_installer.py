"""Build dist/OpenSuno.dmg: one installer for Macs with nothing installed.

The disk image holds "Install OpenSuno.app". It asks whether this Mac should run the Studio (the
interface), the render node (songs rendered on this GPU, with a menu bar app) or both, then installs
the code, a relocatable Python, ffmpeg and the apps. Python packages are downloaded from PyPI during
installation, which keeps the image around 100 MB instead of shipping MLX and PyTorch.
"""
import hashlib
import os
from pathlib import Path
import plistlib
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parent.parent
CACHE = Path.home() / 'Library/Caches/OpenSuno/build'
RELEASE = 'https://github.com/astral-sh/python-build-standalone/releases/download/20260901/'
PYTHON312_URL = RELEASE + 'cpython-3.12.14%2B20260901-aarch64-apple-darwin-install_only.tar.gz'
PYTHON311_URL = RELEASE + 'cpython-3.11.16%2B20260901-aarch64-apple-darwin-install_only.tar.gz'
FFMPEG_URL = 'https://github.com/zackees/ffmpeg_bins/raw/main/v8.0/darwin_arm64.zip'
# Not shipped: tests, hosting config, the dev-only notes.
SKIP = ('tests/', '.vercel', 'vercel.json', 'TODO-CUDA.md', 'dist/', 'site/', 'docs/screenshots/')
SWIFT = ('/usr/bin/xcrun', 'swiftc', '-O', '-swift-version', '5', '-target', 'arm64-apple-macos13.0')

# The /usr/bin/python3 shim exports the Command Line Tools SDK, which Xcode's Swift toolchain refuses.
os.environ.pop('SDKROOT', None)


def run(*args, **kw):
    return subprocess.run([str(a) for a in args], check=True, **kw)


def fetch(url):
    CACHE.mkdir(parents=True, exist_ok=True)
    target = CACHE / (hashlib.sha1(url.encode()).hexdigest()[:10] + '-' + url.rsplit('/', 1)[1].replace('%2B', '+'))
    if not target.is_file():
        print(f'Downloading {url}')
        with urllib.request.urlopen(url, timeout=120) as response, target.open('wb') as out:
            shutil.copyfileobj(response, out)
    return target


def build_id():
    commit = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip() or 'nogit'
    return time.strftime('%Y%m%d-%H%M%S') + '-' + commit


def source_files():
    listed = subprocess.run(['git', 'ls-files', '-co', '--exclude-standard'], cwd=ROOT, capture_output=True, text=True, check=True).stdout.splitlines()
    return [f for f in listed if f and not f.startswith(SKIP) and (ROOT / f).is_file()]


def icon(app):
    with tempfile.TemporaryDirectory() as folder:
        png = Path(folder) / 'icon.png'
        run('/usr/bin/xcrun', 'swift', ROOT / 'launcher/icon.swift', png)
        iconset = Path(folder) / 'OpenSuno.iconset'
        iconset.mkdir()
        for points in (16, 32, 128, 256, 512):
            for scale in (1, 2):
                suffix = '@2x' if scale == 2 else ''
                run('/usr/bin/sips', '-z', points * scale, points * scale, png, '--out', iconset / f'icon_{points}x{points}{suffix}.png',
                    stdout=subprocess.DEVNULL)
        (app / 'Contents/Resources').mkdir(parents=True, exist_ok=True)
        run('/usr/bin/iconutil', '-c', 'icns', iconset, '-o', app / 'Contents/Resources/OpenSuno.icns')


def write_info(app, executable, identifier, name, stamp, **extra):
    info = {
        'CFBundleExecutable': executable, 'CFBundleIdentifier': identifier, 'CFBundleName': name, 'CFBundleDisplayName': name,
        'CFBundleIconFile': 'OpenSuno', 'CFBundlePackageType': 'APPL', 'CFBundleInfoDictionaryVersion': '6.0',
        'CFBundleShortVersionString': stamp.split('-')[0], 'CFBundleVersion': stamp, 'LSMinimumSystemVersion': '13.0',
        'NSHighResolutionCapable': True, 'LSApplicationCategoryType': 'public.app-category.music', 'NSHumanReadableCopyright': 'OpenSuno',
    }
    info.update(extra)
    with (app / 'Contents/Info.plist').open('wb') as f:
        plistlib.dump(info, f)


def sign(app):
    run('/usr/bin/codesign', '--force', '--deep', '--sign', '-', app, stderr=subprocess.DEVNULL)
    run('/usr/bin/codesign', '--verify', '--strict', app)


def build_studio_app(app, stamp):
    """OpenSuno.app without a payload: it starts the Studio that the installer put in ~/OpenSuno."""
    (app / 'Contents/MacOS').mkdir(parents=True)
    run(*SWIFT, '-parse-as-library', '-o', app / 'Contents/MacOS/OpenSuno', ROOT / 'launcher/OpenSunoApp.swift', stdout=subprocess.DEVNULL)
    (app / 'Contents/Resources').mkdir(parents=True)
    shutil.copy2(ROOT / 'launcher/bootstrap.py', app / 'Contents/Resources/bootstrap.py')
    icon(app)
    write_info(app, 'OpenSuno', 'local.opensuno.studio.app', 'OpenSuno', stamp)
    sign(app)


def build_node_app(app, stamp, root):
    """The menu bar app; OpenSunoRoot tells it where the render node's code and Python live."""
    (app / 'Contents/MacOS').mkdir(parents=True)
    run(*SWIFT, '-o', app / 'Contents/MacOS/OpenSunoRenderNode', ROOT / 'launcher/RenderNodeMenu.swift', stdout=subprocess.DEVNULL)
    icon(app)
    write_info(app, 'OpenSunoRenderNode', 'local.opensuno.rendernode.menu', 'OpenSuno Render Node', stamp,
               LSUIElement=True, OpenSunoRoot=str(root))
    sign(app)


def build(output):
    stamp = build_id()
    with tempfile.TemporaryDirectory(prefix='opensuno-installer-') as folder:
        staging = Path(folder) / 'dmg'
        app = staging / 'Install OpenSuno.app'
        resources = app / 'Contents/Resources'
        payload = resources / 'payload'
        (app / 'Contents/MacOS').mkdir(parents=True)
        (payload / 'code').mkdir(parents=True)
        (payload / 'runtime').mkdir()
        (payload / 'apps').mkdir()
        print('Copying OpenSuno code')
        for relative in source_files():
            target = payload / 'code' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, target)
        (payload / '.app-build').write_text(stamp + '\n')
        print('Adding Python runtimes and ffmpeg')
        shutil.copy2(fetch(PYTHON312_URL), payload / 'runtime/python312.tar.gz')
        shutil.copy2(fetch(PYTHON311_URL), payload / 'runtime/python311.tar.gz')
        shutil.copy2(fetch(FFMPEG_URL), payload / 'runtime/ffmpeg.zip')
        print('Compiling the Studio and Render Node apps')
        build_studio_app(payload / 'apps/OpenSuno.app', stamp)
        build_node_app(payload / 'apps/OpenSuno Render Node.app', stamp, Path.home() / 'OpenSuno')
        print('Compiling the installer')
        run(*SWIFT, '-parse-as-library', '-o', app / 'Contents/MacOS/InstallOpenSuno', ROOT / 'launcher/InstallerApp.swift', stdout=subprocess.DEVNULL)
        shutil.copy2(ROOT / 'launcher/installer.py', resources / 'installer.py')
        icon(app)
        write_info(app, 'InstallOpenSuno', 'local.opensuno.installer', 'Install OpenSuno', stamp)
        print('Signing')
        sign(app)
        print('Packing disk image')
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            output.unlink()
        run('/usr/bin/hdiutil', 'create', '-volname', 'OpenSuno', '-srcfolder', staging, '-ov', '-format', 'UDZO', '-quiet', output)
    size = output.stat().st_size / 1e6
    print(f'Built {output}  ({size:.0f} MB, build {stamp})')


if __name__ == '__main__':
    if sys.platform != 'darwin':
        raise SystemExit('The installer is built on a Mac.')
    build(Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else ROOT / 'dist/OpenSuno.dmg')
