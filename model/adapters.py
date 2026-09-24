"""Switchable LoRA adapters on the resident MLX model: any community LoRA, chosen per song.

The model stays in memory between songs, so the adapters are not folded into the weights. Every linear a
YuE2 LoRA can touch (AR ``self_attn`` / ``mlp``, NAR ``nar_self_attn`` / ``nar_mlp``, and the ``vae2llm`` /
``llm2vae`` projections) is wrapped once; ``apply`` loads one file's deltas into the wrappers and switches
them on, ``clear`` switches everything off again so the next song hears the stock model. Inactive wrappers
cost one Python branch per call. The LoRA path runs in float32 like its training and works on the 8-bit
model too, since it only reads the base layer's output.
"""
from pathlib import Path
import sys

import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'app'))
import loras  # noqa: E402

ATTENTION = ('q_proj', 'k_proj', 'v_proj', 'o_proj')
MLP = ('gate_proj', 'up_proj', 'down_proj')


class Adapter(nn.Module):
    """``base(x) + scale * (x A^T) B^T`` while ``active``."""

    def __init__(self, base):
        super().__init__()
        self.base = base
        self.A = mx.zeros((1, 1))
        self.B = mx.zeros((1, 1))
        self.scale = 0.0
        self.active = False

    @property
    def weight(self):
        return self.base.weight

    def set(self, A, B, scale):
        self.A, self.B, self.scale = mx.array(A, dtype=mx.float32), mx.array(B, dtype=mx.float32), float(scale)
        self.active = True

    def __call__(self, x):
        y = self.base(x)
        if not self.active:
            return y
        return y + (((x.astype(mx.float32) @ self.A.T) @ self.B.T) * self.scale).astype(y.dtype)


class Replacement(nn.Module):
    """A complete replacement linear (``vae2llm`` / ``llm2vae``) used instead of ``stock`` while ``active``."""

    def __init__(self, stock):
        super().__init__()
        self.stock = stock
        self.real = None
        self.active = False

    @property
    def weight(self):
        return self.stock.weight

    def set(self, weight, bias):
        stock = self.stock
        layer = nn.Linear(weight.shape[1], weight.shape[0], bias=bias is not None)
        layer.weight = mx.array(weight).astype(stock.weight.dtype)
        if bias is not None:
            layer.bias = mx.array(bias).astype(stock.weight.dtype)
        self.real = layer
        self.active = True

    def __call__(self, x):
        return (self.real if self.active and self.real is not None else self.stock)(x)


def innermost(module):
    """The stock linear under any adapter wrappers (these, or the real-audio decoder's)."""
    while not isinstance(module, (nn.Linear, nn.QuantizedLinear)):
        if isinstance(module, Adapter) or 'base' in module:
            module = module.base
        elif isinstance(module, Replacement) or 'stock' in module:
            module = module.stock
        else:
            break
    return module


def attach(model):
    """Wrap every LoRA-able linear once; returns ``{'linears': {(layer, module, proj): Adapter}, 'io': {...}}``."""
    wrappers = model.__dict__.get('_adapters')
    if wrappers is not None:
        return wrappers
    wrappers = {'linears': {}, 'io': {}, 'merged': {}, 'weights': None, 'applied': None}
    for i, layer in enumerate(model.model.layers):
        for module_name in loras.AR_MODULES + loras.NAR_MODULES:
            module = getattr(layer, module_name)
            for name in (ATTENTION if 'attn' in module_name else MLP):
                wrapped = Adapter(getattr(module, name))
                setattr(module, name, wrapped)
                wrappers['linears'][(i, module_name, name)] = wrapped
    for name in loras.IO_NAMES:
        wrapped = Replacement(getattr(model, name))
        setattr(model, name, wrapped)
        wrappers['io'][name] = wrapped
    object.__setattr__(model, '_adapters', wrappers)
    return wrappers


def checkpoint_name(key):
    return f'model.layers.{key[0]}.{key[1]}.{key[2]}.weight'


def apply(model, delta, strength=1.0, weights=None, label=None):
    """Put one file's deltas (from ``loras.deltas``) on the model and return how many layers changed.

    With ``weights`` (the model's ``model.safetensors``), deltas on plain bf16 linears are folded into the
    weights (``W += strength * scale * B @ A``), which costs nothing per token; ``clear`` reads the original
    tensors back from that file. Quantized linears and the io replacements use the switchable wrappers.
    ``label`` names the choice so applying the same LoRA at the same strength twice in a row is a no-op."""
    wrappers = attach(model)
    if label is not None and wrappers.get('applied') == label:
        return wrappers.get('applied_count', 0)
    clear(model)
    if not strength:
        return 0
    if weights is not None:
        wrappers['weights'] = str(weights)
    count = 0
    for key, (A, B, scale) in delta['linears'].items():
        wrapper = wrappers['linears'].get(key)
        if wrapper is None:
            raise ValueError(f'The model has no layers.{key[0]}.{key[1]}.{key[2]}')
        base = innermost(wrapper)
        out_features, in_features = (base.weight.shape if not isinstance(base, nn.QuantizedLinear)
                                     else (base.weight.shape[0], base.weight.shape[1] * 32 // base.bits))
        if A.shape[1] != in_features or B.shape[0] != out_features:
            raise ValueError(f'LoRA shape mismatch at layers.{key[0]}.{key[1]}.{key[2]}: {tuple(A.shape)} x {tuple(B.shape)} for {out_features}x{in_features}')
        if weights is not None and isinstance(base, nn.Linear):
            change = (mx.array(B, dtype=mx.float32) @ mx.array(A, dtype=mx.float32)) * (scale * strength)
            base.weight = (base.weight.astype(mx.float32) + change).astype(base.weight.dtype)
            wrappers['merged'][key] = base
        else:
            wrapper.set(A, B, scale * strength)
        count += 1
    for name, parts in delta['io'].items():
        wrappers['io'][name].set(parts['weight'], parts.get('bias'))
        count += 1
    mx.eval(model.parameters())
    wrappers['applied'], wrappers['applied_count'] = label, count
    return count


def clear(model):
    """Back to the stock model: wrappers off, merged weights read again from the checkpoint."""
    wrappers = model.__dict__.get('_adapters')
    if not wrappers:
        return
    for wrapper in list(wrappers['linears'].values()) + list(wrappers['io'].values()):
        wrapper.active = False
    if wrappers['merged']:
        from weight_io import load_weights
        stock = load_weights(wrappers['weights'])
        for key, base in wrappers['merged'].items():
            base.weight = stock[checkpoint_name(key)]
        mx.eval(model.parameters())
        wrappers['merged'] = {}
    wrappers['applied'] = None
