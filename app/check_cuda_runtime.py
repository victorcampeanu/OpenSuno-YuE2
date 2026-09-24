"""Verify the official CUDA environment before advertising it as ready."""
import json
from pathlib import Path
import torch
from yue2.pipeline import YuE2Pipeline

if __name__ == '__main__':
    assert torch.cuda.is_available(), 'An NVIDIA CUDA device is required'
    assert torch.cuda.is_bf16_supported(), 'CUDA BF16 support is required'
    with torch.inference_mode():
        x = torch.ones((64,64), device='cuda', dtype=torch.bfloat16)
        assert torch.all(x @ x == 64).item()
    torch.cuda.synchronize()
    info = dict(torch=torch.__version__, cuda=torch.version.cuda, device=torch.cuda.get_device_name(0))
    marker = Path(__file__).resolve().parents[1]/'.cuda-venv/runtime-ready.json'
    marker.write_text(json.dumps(info))
    print(json.dumps(info))
