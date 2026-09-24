"""Keep MLX's native loader except for its Windows large-file seek limitation."""
import os
from pathlib import Path


def load_weights(path):
    import mlx.core as mx
    path = Path(path)
    if os.name == 'nt' and path.stat().st_size >= 2**31:
        # MLX 0.32.2's native Windows reader reports the wrong size above 2 GiB.
        # Python's reader uses 64-bit offsets. Evaluate before closing it because
        # MLX loads lazily; this changes neither tensor values nor the checkpoint.
        with path.open('rb') as source:
            weights = mx.load(source, format='safetensors')
            mx.eval(weights)
        return weights
    return mx.load(str(path))
