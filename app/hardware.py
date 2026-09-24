"""What this machine can render with: the GPU it has and which model packages are installed for it."""
import importlib.util
import os
import platform
import shutil
import subprocess
import sys

import loras
from model_registry import available_models, default_model, is_cuda
from runtime_platform import environment_python


def detect_gpu():
    """Best-effort GPU description without importing torch or MLX into the server process."""
    if sys.platform == 'darwin':
        try:
            chip = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], capture_output=True, text=True, timeout=5).stdout.strip()
            memory = int(subprocess.run(['sysctl', '-n', 'hw.memsize'], capture_output=True, text=True, timeout=5).stdout.strip() or 0)
        except (OSError, ValueError, subprocess.TimeoutExpired):
            chip, memory = platform.processor() or 'Apple Silicon', 0
        return {'kind': 'apple' if platform.machine() == 'arm64' else 'none', 'name': chip or 'Apple Silicon',
                'memory_gb': round(memory / 1024**3, 1) if memory else None, 'unified_memory': True}
    smi = shutil.which('nvidia-smi')
    if smi:
        try:
            out = subprocess.run([smi, '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'],
                                 capture_output=True, text=True, timeout=10,
                                 creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)).stdout.strip().splitlines()
            if out:
                name, memory = [part.strip() for part in out[0].split(',', 1)]
                return {'kind': 'cuda', 'name': name, 'memory_gb': round(int(memory) / 1024, 1), 'count': len(out)}
        except (OSError, ValueError, subprocess.TimeoutExpired):
            pass
    return {'kind': 'none', 'name': platform.processor() or 'CPU only', 'memory_gb': None}


def cuda_environment_ready(root):
    return environment_python(root, environment='.cuda-venv').is_file() and (root / '.cuda-venv/runtime-ready.json').is_file()


def mlx_environment_ready():
    """The MLX worker runs in the Studio's own environment; an interface-only install does not carry it."""
    return sys.platform == 'darwin' and platform.machine() == 'arm64' and importlib.util.find_spec('mlx') is not None


def render_environment_ready(root):
    """Can this machine render at all, or does it only run the interface for a render node?"""
    return mlx_environment_ready() or cuda_environment_ready(root)


def is_notice(path):
    name = path.rsplit('/', 1)[-1]
    return name in ('LICENSE', 'weights_manifest.json') or name.endswith(('.md', '.txt'))


def model_ready(root, downloads, name):
    """Weights present and the runtime that executes them installed."""
    if is_cuda(name):
        # Licence texts and manifests ship with the weights but the engine never reads them; they do not gate readiness.
        needed = [a for a in downloads.selected_assets(name) if not is_notice(a['path'])]
        return bool(needed) and all(downloads.present(a) for a in needed) and cuda_environment_ready(root)
    needed = [a for a in downloads.assets if a['path'].startswith('model/' + name + '/') or a['path'] in {'model/8bit/vae.safetensors', 'model/8bit/qwen.tiktoken'}]
    return mlx_environment_ready() and all(downloads.present(a) for a in needed) and all((root / 'model' / name / f).is_file() for f in ['vae.safetensors', 'qwen.tiktoken', 'config.json', 'vae_config.json', 'yue2_generation_config.json'])


def models(root, downloads, gpu=None):
    """The generation choices this machine offers, with readiness. Linux boxes without NVIDIA hide the CUDA runtime."""
    gpu = gpu or detect_gpu()
    offered = available_models()
    if sys.platform not in ('darwin', 'win32') and gpu['kind'] != 'cuda':
        offered = {k: v for k, v in offered.items() if not is_cuda(k)}
    return [{'id': k, **v, 'ready': model_ready(root, downloads, k)} for k, v in offered.items()]


def capabilities(root, downloads, gpu=None):
    """Everything a Studio needs to know before sending work here."""
    gpu = gpu or detect_gpu()
    analysis_env = environment_python(root, True).is_file()
    return {
        'platform': {'system': platform.system(), 'machine': platform.machine(), 'python': platform.python_version(), 'hostname': platform.node()},
        'gpu': gpu,
        'models': models(root, downloads, gpu),
        'default_model': default_model(),
        'downloads': downloads.snapshot(),
        'render_environment_ready': render_environment_ready(root),
        'transcriber_environment_ready': analysis_env,
        'transcriber_ready': downloads.package_ready('analysis') and analysis_env,
        'tokens_ready': downloads.package_ready('tokens') and downloads.package_ready('analysis') and analysis_env,
        # Community LoRA files in the loras folder beside the models; Studios offer these in the picker.
        'loras': loras.catalog(root, downloads.lora_catalog),
    }
