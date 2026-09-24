"""Generation choices and their shared download packages."""
import sys

MODELS = {
    'bf16': {'label': 'YuE2 BF16', 'description': 'Full precision · Metal / MLX CUDA', 'backend': 'mlx'},
    'cuda-bf16': {'label': 'YuE2 CUDA BF16', 'description': 'Original YuE2 · full precision', 'backend': 'cuda'},
    'cuda-fp8': {'label': 'YuE2 CUDA FP8', 'description': 'Experimental FP8 token generation · BF16 synthesis', 'backend': 'cuda'},
}


def default_model():
    return 'cuda-bf16' if sys.platform == 'win32' else 'bf16'


def is_cuda(model):
    return MODELS.get(model, {}).get('backend') == 'cuda'


def available_models():
    if sys.platform == 'win32':
        return {'cuda-bf16': MODELS['cuda-bf16']}
    return {key:value for key,value in MODELS.items() if sys.platform != 'darwin' or not is_cuda(key)}
