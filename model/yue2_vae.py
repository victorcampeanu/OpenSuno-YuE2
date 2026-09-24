"""YuE2 Oobleck VAE decoder in MLX (channels-last; weight norm folded by convert.py)."""
from __future__ import annotations

import json
import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

FRAME = 1920  # samples per latent frame at 48 kHz


class Snake(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.alpha = mx.zeros((ch,))
        self.beta = mx.zeros((ch,))

    def __call__(self, x):
        alpha, beta = mx.exp(self.alpha), mx.exp(self.beta)
        return x + (1.0 / (beta + 1e-9)) * mx.sin(x * alpha) ** 2


class ResidualUnit(nn.Module):
    def __init__(self, ch, dilation):
        super().__init__()
        self.layers = [Snake(ch), nn.Conv1d(ch, ch, 7, dilation=dilation, padding=3 * dilation),
                       Snake(ch), nn.Conv1d(ch, ch, 1)]

    def __call__(self, x):
        y = x
        for layer in self.layers:
            y = layer(y)
        return x + y


class DecoderBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.layers = [Snake(cin),
                       nn.ConvTranspose1d(cin, cout, 2 * stride, stride=stride, padding=math.ceil(stride / 2)),
                       ResidualUnit(cout, 1), ResidualUnit(cout, 3), ResidualUnit(cout, 9)]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class OobleckDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        c, mults, strides = cfg["channels"], [1] + list(cfg["c_mults"]), cfg["strides"]
        layers = [nn.Conv1d(cfg["latent_dim"], mults[-1] * c, 7, padding=3)]
        for i in range(len(mults) - 1, 0, -1):
            layers.append(DecoderBlock(mults[i] * c, mults[i - 1] * c, strides[i - 1]))
        layers += [Snake(mults[0] * c), nn.Conv1d(mults[0] * c, cfg["out_channels"], 7, padding=3, bias=False)]
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def decode_tiled(self, z, core=1024, halo=16):
        """z [T,64] float32 → audio [1920*T-64, 2] float32, exact tile cores, no crossfade."""
        frames, total = z.shape[0], FRAME * z.shape[0] - 64
        out = []
        for start in range(0, frames, core):
            end = min(frames, start + core)
            left, right = max(0, start - halo), min(frames, end + halo)
            tile = self(z[None, left:right])[0]
            crop = (start - left) * FRAME
            out.append(tile[crop:crop + min(end * FRAME, total) - start * FRAME])
            mx.eval(out[-1])
        return mx.concatenate(out)


def load_vae(path: Path, on_load=None) -> OobleckDecoder:
    path = Path(path)
    on_load = on_load or (lambda detail: None)
    on_load('Building audio decoder')
    cfg = json.loads((path / "vae_config.json").read_text())
    decoder = OobleckDecoder(cfg["decoder_config"])
    on_load('Reading audio decoder weights')
    decoder.load_weights(str(path / "vae.safetensors"))
    on_load('Preparing audio decoder on GPU')
    mx.eval(decoder.parameters())
    return decoder
