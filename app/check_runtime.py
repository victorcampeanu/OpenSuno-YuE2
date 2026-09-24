"""Verify actual GPU execution, including the operations used by YuE2."""
from runtime_platform import prepare_gpu_libraries
prepare_gpu_libraries()
import mlx.core as mx
import mlx.nn as nn
import sys
from pathlib import Path


def main():
    mx.set_default_device(mx.gpu)
    x = mx.ones((1, 4, 64), dtype=mx.bfloat16)
    projection = nn.Linear(64, 64, bias=False).to_quantized(group_size=64, bits=8)
    mx.eval(projection(x))
    q = x.reshape(1, 1, 4, 64)
    mx.eval(mx.fast.scaled_dot_product_attention(q, q, q, scale=0.125, mask='causal'))
    mx.eval(nn.ConvTranspose1d(4, 2, 4, stride=2)(mx.ones((1, 8, 4))))
    # Exercise the actual shared AR cache and NAR paths with a tiny random
    # backbone. No downloaded weights are needed for this compatibility check.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'model'))
    from yue2_model import Yue2Model, KVCache
    cfg = dict(hidden_size=64, intermediate_size=128, num_attention_heads=1,
               num_key_value_heads=1, head_dim=64, rms_norm_eps=1e-6,
               rope_theta=1000000, vocab_size=128, num_hidden_layers=1,
               latent_dim=64, max_latent_frames=32, timestep_shift=1.0)
    model = Yue2Model(cfg)
    model.set_dtype(mx.bfloat16)
    cache = [KVCache()]
    mx.eval(model.ar_step(mx.array([[1, 2, 3]]), cache))
    mx.eval(model.ar_step(mx.array([[4]]), cache))
    prefills = model.nar_prefill([1, 2, 3])
    velocity = model.nar_velocity(mx.zeros((4, 64), mx.bfloat16), 0.0, prefills, 3)
    assert velocity.shape == (4, 64) and mx.all(mx.isfinite(velocity)).item()
    print('MLX GPU checks passed:', mx.device_info())


if __name__ == '__main__':
    main()
    # Release CUDA work before Python starts finalizing DLL-backed modules.
    mx.synchronize()
    mx.clear_streams()
    mx.clear_cache()
