"""A Studio server in a temporary folder, shared by the API tests.

`studio_root(case)` copies the server and its two data files beside a fresh library; `load_studio(case, root, name)`
imports that copy under its own module name so several tests can hold separate servers at once.
"""
import importlib.util
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ('server.py', 'model-assets.json', 'lora-assets.json', 'generation-defaults.json')


def studio_root(case):
    tmp = tempfile.TemporaryDirectory()
    case.addCleanup(tmp.cleanup)
    root = Path(tmp.name)
    for name in FILES:
        shutil.copy2(ROOT / name, root / name)
    shutil.copytree(ROOT / 'web', root / 'web')
    return root


def load_studio(case, root, name):
    spec = importlib.util.spec_from_file_location(name, root / 'server.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    case.addCleanup(sys.modules.pop, spec.name, None)
    spec.loader.exec_module(module)
    return module
