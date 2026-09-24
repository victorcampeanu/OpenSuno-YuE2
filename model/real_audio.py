"""Render real-audio music tokens: the community NAR LoRA on YuE2's acoustic path, switchable per song.

YuE2's flow-matching decoder (the NAR branch) was trained only on tokens YuE2 wrote itself.
Tokens that ``audio_tokens`` predicts from a recording are close but not identical, and the
stock decoder renders them muddy. ``nar_lora_joint_v9.safetensors`` from
``Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4`` was trained jointly with the
v9 tokenizer head: a rank-32 LoRA on ``nar_self_attn.{q,k,v,o}_proj`` and
``nar_mlp.{gate,up,down}_proj`` of every layer plus replacement ``vae2llm``/``llm2vae``.

The adapters stay separate from the base weights and are switched on only while a song
that contains recording tokens is synthesized, so ordinary songs sound exactly as before.
A rank-32 path adds a negligible amount of work to each ODE step.
"""
from contextlib import contextmanager
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

import adapters
import torch_checkpoint

FILE = 'tokens/nar_lora_joint_v9.safetensors'
ATTENTION = ('q_proj', 'k_proj', 'v_proj', 'o_proj')
MLP = ('gate_proj', 'up_proj', 'down_proj')


class LoRALinear(nn.Module):
    """``base(x) + x A^T B^T`` while ``active``; the LoRA path runs in float32 like its training."""

    def __init__(self, base, A, B):
        super().__init__()
        self.base, self.A, self.B = base, A, B
        self.active = False

    def __call__(self, x):
        y = self.base(x)
        if not self.active:
            return y
        return y + ((x.astype(mx.float32) @ self.A.T) @ self.B.T).astype(y.dtype)


class SwitchLinear(nn.Module):
    """Two complete linear layers; ``real`` replaces ``stock`` while ``active``."""

    def __init__(self, stock, real):
        super().__init__()
        self.stock, self.real = stock, real
        self.active = False

    def __call__(self, x):
        return (self.real if self.active else self.stock)(x)


def pairs(checkpoint, layers):
    """(A, B) per NAR linear in checkpoint order: per layer q, k, v, o, gate, up, down."""
    tensors = checkpoint['lora']
    per_layer = 2 * (len(ATTENTION) + len(MLP))
    if len(tensors) != layers * per_layer:
        raise ValueError(f'The NAR LoRA has {len(tensors)} tensors; {layers} layers need {layers * per_layer}')
    return [(tensors[i], tensors[i + 1]) for i in range(0, len(tensors), 2)]


def _linear(state, like):
    layer = nn.Linear(state['weight'].shape[1], state['weight'].shape[0], bias='bias' in state)
    layer.weight = mx.array(state['weight']).astype(like.weight.dtype)
    if 'bias' in state:
        layer.bias = mx.array(state['bias']).astype(like.weight.dtype)
    return layer


def available(root):
    return (Path(root) / FILE).is_file()


def checkpoint_from_safetensors(tensors):
    """The ``.pt`` dict layout (``lora``, ``io``, ``rank``) from a Mothersuperior NAR safetensors file."""
    prefix = 'nar_' if any('.nar_self_attn.' in key for key in tensors) else ''
    modules = [(f'{prefix}self_attn', name) for name in ATTENTION] + [(f'{prefix}mlp', name) for name in MLP]
    layers = sorted({int(key.split('.')[1]) for key in tensors if key.startswith('layers.')})
    lora = []
    for layer in layers:
        for block, proj in modules:
            lora.append(tensors[f'layers.{layer}.{block}.{proj}.lora_A'])
            lora.append(tensors[f'layers.{layer}.{block}.{proj}.lora_B'])
    io = {name: {key.split('.', 1)[1]: value for key, value in tensors.items() if key.startswith(name + '.')}
          for name in ('vae2llm', 'llm2vae')}
    return {'lora': lora, 'io': io, 'rank': int(lora[0].shape[0])}


def load_checkpoint(path):
    """A NAR adapter checkpoint. ``.pt`` is the original torch archive; safetensors is the same weights."""
    path = Path(path)
    if path.suffix == '.safetensors':
        import loras
        return checkpoint_from_safetensors(loras.read_tensors(path))
    return torch_checkpoint.load(path)


def attach(model, root):
    """Wrap the NAR linears of ``model`` with the adapters (once); returns the list of switches."""
    switches = model.__dict__.get('_real_audio')
    if switches is not None:
        return switches
    return attach_checkpoint(model, load_checkpoint(Path(root) / FILE))


def attach_checkpoint(model, checkpoint):
    switches = model.__dict__.get('_real_audio')
    if switches is not None:
        return switches
    switches = []
    it = iter(pairs(checkpoint, len(model.model.layers)))
    for layer in model.model.layers:
        for module, names in ((layer.nar_self_attn, ATTENTION), (layer.nar_mlp, MLP)):
            for name in names:
                A, B = next(it)
                base = getattr(module, name)
                # A song-level LoRA adapter (adapters.py) may already wrap this linear; measure the stock layer.
                stock = adapters.innermost(base)
                out_features, in_features = stock.weight.shape if not isinstance(stock, nn.QuantizedLinear) else (
                    stock.weight.shape[0], stock.weight.shape[1] * 32 // stock.bits)
                if A.shape != (checkpoint['rank'], in_features) or B.shape != (out_features, checkpoint['rank']):
                    raise ValueError(f'NAR LoRA shape mismatch at {name}: {A.shape} x {B.shape} for {out_features}x{in_features}')
                wrapped = LoRALinear(base, mx.array(A), mx.array(B))
                setattr(module, name, wrapped)
                switches.append(wrapped)
    for name in ('vae2llm', 'llm2vae'):
        stock = getattr(model, name)
        wrapped = SwitchLinear(stock, _linear(checkpoint['io'][name], stock))
        setattr(model, name, wrapped)
        switches.append(wrapped)
    mx.eval(model.parameters())
    object.__setattr__(model, '_real_audio', switches)
    return switches


def set_active(model, active):
    for switch in model.__dict__.get('_real_audio', ()):
        switch.active = bool(active)


@contextmanager
def enabled(model, active=True):
    """Run synthesis with the real-audio decoder switched on (or explicitly off)."""
    set_active(model, active)
    try:
        yield
    finally:
        set_active(model, False)
