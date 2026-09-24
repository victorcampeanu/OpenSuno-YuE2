"""Community LoRA adapters for YuE2: the ``loras/`` folder, its catalogue, and the maths that turns a file into deltas.

A LoRA is a ``.safetensors`` file of low-rank pairs for YuE2's linear layers. Four layouts circulate:

* **HF / upstream** — ``layers.{i}.self_attn.q_proj.lora_A`` + ``lora_B``; apply ``W += B @ A``.
* **PEFT** — the same with ``base_model.model.model.`` in front and ``.weight`` behind.
* **ComfyUI** — ``text_encoders.model.layers.{i}.self_attn.qkv_proj.lora_down.weight`` + ``lora_up.weight``, where
  q/k/v (and gate/up) are fused: ``down`` stacks the A matrices, ``up`` places the B matrices block-diagonally, so
  ``up @ down`` equals the stacked deltas. ``unfuse`` splits them back into the per-projection pairs.
* **Sound & Vision native** — ``sound-vision-yue2-native-export-v1``. AR lives under ``text_encoders.model`` and
  NAR under ``diffusion_model.model``, both using AR module names (``self_attn`` / ``mlp``). Fused ``qkv_proj`` /
  ``gate_up_proj`` keep one ``lora_A`` (rank × in) and a stacked ``lora_B`` (sum of outs × rank); slicing ``B``
  recovers the per-projection pairs. Training step and trigger word are read from the file metadata when present.
  Artist packs (CNZN, MLTNT, CHNSN, QWWL) use the same packing under ComfyUI ``lora_down`` / ``lora_up`` names;
  those are unfused by tensor shape, not by key.

Files may also carry an ``alpha`` per projection (scale ``alpha / rank``) and full replacement weights for the
``vae2llm`` / ``llm2vae`` projections, as the real-audio NAR adapter does. Everything here is NumPy + the standard
library so the Studio can catalogue files without torch or MLX; the engines convert the arrays they need.
"""
import json
import re
import struct
from pathlib import Path


def _numpy():
    """Tensor maths only. The hosted site catalogues files from headers, so import must not require NumPy."""
    import numpy as np
    return np

FOLDER = 'loras'
FILE_NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9 ._()\[\]+-]{0,150}\.safetensors')
ATTENTION = ('q_proj', 'k_proj', 'v_proj', 'o_proj')
MLP = ('gate_proj', 'up_proj', 'down_proj')
AR_MODULES = ('self_attn', 'mlp')
NAR_MODULES = ('nar_self_attn', 'nar_mlp')
IO_NAMES = ('vae2llm', 'llm2vae')
# YuE2-3B projection sizes, needed to split ComfyUI's fused tensors. ``dims_from_config`` reads other sizes.
DIMS = {'hidden': 2048, 'q': 2048, 'kv': 1024, 'intermediate': 6144, 'layers': 28}

KEY = re.compile(r'(?:^|\.)layers\.(\d+)\.(self_attn|nar_self_attn|mlp|nar_mlp)\.'
                 r'(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj|qkv_proj|gate_up_proj)\.'
                 r'(lora_A|lora_B|lora_up|lora_down|alpha)(?:\.weight)?$')
IO_KEY = re.compile(r'(?:^|\.)(vae2llm|llm2vae)\.(weight|bias)$')
def _dtypes(np):
    return {'F32': np.float32, 'F16': np.float16, 'BF16': np.uint16, 'F64': np.float64, 'I64': np.int64, 'I32': np.int32}
NATIVE_FORMAT = 'sound-vision-yue2-native-export-v1'


def _native_module(key, module):
    """Sound & Vision native exports store NAR tensors under ``diffusion_model`` with AR module names."""
    if key.startswith('diffusion_model.') or '.diffusion_model.' in key:
        if module == 'self_attn':
            return 'nar_self_attn'
        if module == 'mlp':
            return 'nar_mlp'
    return module


def _layout_of(key, proj, part, metadata=None):
    metadata = metadata or {}
    if metadata.get('format') == NATIVE_FORMAT or key.startswith('diffusion_model.'):
        return 'sound-vision'
    if part in ('lora_up', 'lora_down'):
        return 'comfyui'
    if proj in ('qkv_proj', 'gate_up_proj'):
        return 'sound-vision' if part in ('lora_A', 'lora_B', 'alpha') else 'comfyui'
    if key.startswith('base_model.'):
        return 'peft'
    return 'hf'


def _prefer_layout(current, incoming):
    """Keep the most specific layout when a file mixes prefixes (artist packs: ComfyUI names + diffusion_model)."""
    if current is None:
        return incoming
    order = {'sound-vision': 0, 'comfyui': 1, 'peft': 2, 'hf': 3}
    return incoming if order.get(incoming, 9) < order.get(current, 9) else current


def _training_notes(metadata):
    """Trigger word and training step from Sound & Vision sidecars, plus any short string notes."""
    skip = {'format', 'ss_tag_frequency', 'source_sha256', 'training_info'}
    notes = {k: v for k, v in metadata.items() if isinstance(v, str) and len(v) <= 400 and k not in skip}
    steps = trigger = None
    raw = metadata.get('training_info')
    if isinstance(raw, str):
        try:
            info = json.loads(raw)
            if isinstance(info, dict) and info.get('step') is not None:
                steps = int(info['step'])
        except (TypeError, ValueError):
            pass
    raw = metadata.get('ss_tag_frequency')
    if isinstance(raw, str):
        try:
            tags = json.loads(raw)
            training = tags.get('training') if isinstance(tags, dict) else None
            if isinstance(training, dict) and training:
                trigger = next(iter(training))
        except (TypeError, ValueError, StopIteration):
            pass
    return notes, steps, trigger


def _fused_outs(proj, dims):
    if proj == 'qkv_proj':
        return ('q_proj', 'k_proj', 'v_proj'), (dims['q'], dims['kv'], dims['kv'])
    return ('gate_proj', 'up_proj'), (dims['intermediate'], dims['intermediate'])


def _block_diagonal(up, outs):
    """True when ``up`` is ComfyUI's fused B: each projection's B on the diagonal, zeros elsewhere."""
    n = len(outs)
    if up.shape[1] % n:
        return False
    r = up.shape[1] // n
    row = 0
    for i, out in enumerate(outs):
        left, right = up[row:row + out, :i * r], up[row:row + out, (i + 1) * r:]
        if left.size and float(abs(left).max()) > 1e-6:
            return False
        if right.size and float(abs(right).max()) > 1e-6:
            return False
        row += out
    return True


def _native_fused(down, up, outs, named_native=False):
    """True when fused tensors are one A (rank × in) and stacked B (sum of outs × rank).

    ComfyUI uses the same 2-D ranks when ``rank * n`` is the stacked height, so an ambiguous
    ``lora_down`` / ``lora_up`` pair is native unless ``up`` is block-diagonal.
    """
    n = len(outs)
    if getattr(down, 'ndim', len(down)) != 2 or getattr(up, 'ndim', len(up)) != 2:
        return False
    down_shape, up_shape = tuple(down.shape) if hasattr(down, 'shape') else tuple(down), tuple(up.shape) if hasattr(up, 'shape') else tuple(up)
    if up_shape[0] != sum(outs) or up_shape[1] != down_shape[0]:
        return False
    if named_native or down_shape[0] % n:
        return True
    return hasattr(up, 'ndim') and not _block_diagonal(up, outs)


def dims_from_config(config):
    """Projection sizes from a YuE2 ``config.json`` dictionary."""
    head = config.get('head_dim', 128)
    return {'hidden': config['hidden_size'], 'q': config['num_attention_heads'] * head,
            'kv': config['num_key_value_heads'] * head, 'intermediate': config['intermediate_size'],
            'layers': config['num_hidden_layers']}


def folder(root):
    return Path(root) / FOLDER


def valid_name(name):
    return bool(name) and FILE_NAME.fullmatch(name) is not None and '/' not in name and '\\' not in name and '..' not in name


def path_of(root, name):
    if not valid_name(name):
        raise ValueError('Not a LoRA file name: ' + repr(name))
    return folder(root) / name


# ----------------------------------------------------------------------------------------------- safetensors ----

def read_header(path):
    """(header dict without ``__metadata__``, metadata dict, byte offset of the data block)."""
    with open(path, 'rb') as f:
        raw = f.read(8)
        if len(raw) < 8:
            raise ValueError('Not a safetensors file')
        n = struct.unpack('<Q', raw)[0]
        if n > 100_000_000:
            raise ValueError('Not a safetensors file')
        header = json.loads(f.read(n).decode('utf-8'))
    metadata = header.pop('__metadata__', None) or {}
    return header, metadata, 8 + n


def read_tensors(path, names=None):
    """Tensors as float32 NumPy arrays (bf16 widened exactly); ``names`` limits which are read."""
    np = _numpy()
    dtypes = _dtypes(np)
    header, _, base = read_header(path)
    out = {}
    with open(path, 'rb') as f:
        for name, info in header.items():
            if names is not None and name not in names:
                continue
            dtype = info['dtype']
            if dtype not in dtypes:
                raise ValueError(f'Unsupported tensor type {dtype} in {name}')
            start, end = info['data_offsets']
            f.seek(base + start)
            buffer = f.read(end - start)
            array = np.frombuffer(buffer, dtype=dtypes[dtype])
            if dtype == 'BF16':
                array = (array.astype(np.uint32) << 16).view(np.float32)
            out[name] = array.astype(np.float32).reshape(info['shape'])
    return out


# ----------------------------------------------------------------------------------------------- catalogue ------

def describe(path, header=None, metadata=None):
    """What a file contains, from its header alone: layout, branches, rank, layers, and any embedded notes."""
    path = Path(path)
    if header is None:
        header, metadata, _ = read_header(path)
    metadata = metadata or {}
    pairs = {}
    layout = None
    io = set()
    for key, info in header.items():
        m = KEY.search(key)
        if m:
            layer, module, proj, part = m.groups()
            layout = _prefer_layout(layout, _layout_of(key, proj, part, metadata))
            module = _native_module(key, module)
            pairs.setdefault((int(layer), module, proj), {})[part] = tuple(info['shape'])
            continue
        m = IO_KEY.search(key)
        if m:
            io.add(m.group(1))
    if not pairs and not io:
        raise ValueError('No YuE2 LoRA tensors found (expected layers.N.self_attn.q_proj.lora_A / lora_B or ComfyUI lora_up / lora_down)')
    ranks = set()
    for (_, _, proj), parts in pairs.items():
        down = parts.get('lora_A') or parts.get('lora_down')
        up = parts.get('lora_B') or parts.get('lora_up')
        if down:
            n = 3 if proj == 'qkv_proj' else 2 if proj == 'gate_up_proj' else 1
            # Native fused qkv/gate_up keep one A of shape (rank, in); ComfyUI stacks n ranks into down.
            native = n > 1 and up and _native_fused(down, up, _fused_outs(proj, DIMS)[1], 'lora_A' in parts)
            ranks.add(down[0] if n == 1 or native or layout == 'sound-vision' or 'lora_A' in parts else down[0] // n)
    branches = []
    if any(module in AR_MODULES for _, module, _ in pairs):
        branches.append('ar')
    if any(module in NAR_MODULES for _, module, _ in pairs) or io:
        branches.append('nar')
    layers = sorted({layer for layer, _, _ in pairs})
    fused = any(proj in ('qkv_proj', 'gate_up_proj') for _, _, proj in pairs)
    rank = min(ranks) if ranks else None
    notes, steps, trigger = _training_notes(metadata)
    info = {'id': path.name, 'name': path.stem.replace('_', ' '), 'layout': layout or 'io', 'branches': branches,
            'rank': rank, 'layers': len(layers), 'io': sorted(io), 'fused': fused,
            'size_mb': round(path.stat().st_size / 1024 / 1024, 1), 'metadata': notes}
    if steps is not None:
        info['steps'] = steps
    if trigger:
        info['trigger'] = trigger
    return info


def notes_from_assets(assets):
    """Download-catalog fields keyed by the file name they install as."""
    return {Path(item['path']).name: item for item in (assets or []) if item.get('path')}


def overlay(info, item):
    """Prefer documented trigger, family and settings from ``lora-assets.json`` when the file is one of those downloads."""
    if not item:
        return info
    if item.get('name'):
        info['name'] = item['name']
    for key in ('trigger', 'steps', 'family', 'description', 'notes', 'prompt', 'settings'):
        value = item.get(key)
        if value in (None, '', [], {}):
            continue
        info[key] = value
    return info


def catalog(root, assets=None):
    """Every ``.safetensors`` in ``loras/``; files that are not YuE2 LoRAs are listed with an ``error``.

    ``assets`` is the ``lora-assets.json`` list: matching files pick up the documented name, trigger
    word, family and recommended settings, and those downloads are listed first, in catalog order.
    """
    out = []
    directory = folder(root)
    if not directory.is_dir():
        return out
    notes = notes_from_assets(assets)
    for path in sorted(directory.iterdir(), key=lambda p: p.name.lower()):
        if path.suffix.lower() != '.safetensors' or not path.is_file() or not valid_name(path.name):
            continue
        try:
            info = overlay(describe(path), notes.get(path.name))
        except (ValueError, KeyError, json.JSONDecodeError, UnicodeDecodeError, OSError) as e:
            info = overlay({'id': path.name, 'name': path.stem.replace('_', ' '), 'error': str(e),
                            'size_mb': round(path.stat().st_size / 1024 / 1024, 1)}, notes.get(path.name))
        out.append(info)
    if notes:
        rank = {Path(item['path']).name: i for i, item in enumerate(assets or []) if item.get('path')}
        out.sort(key=lambda item: (0, rank[item['id']]) if item['id'] in rank else (1, item['name'].lower()))
    return out


# ----------------------------------------------------------------------------------------------- deltas ---------

def _unfuse(proj, down, up, dims):
    """Split ComfyUI's fused qkv / gate_up pair into per-projection (A, B) pairs."""
    names, outs = _fused_outs(proj, dims)
    if down.shape[0] % len(names) or up.shape[0] != sum(outs):
        raise ValueError(f'{proj}: fused tensors {tuple(down.shape)} x {tuple(up.shape)} do not match projections {outs}')
    r = down.shape[0] // len(names)
    result = {}
    row = 0
    for i, (name, out) in enumerate(zip(names, outs)):
        result[name] = (down[i * r:(i + 1) * r], up[row:row + out, i * r:(i + 1) * r])
        row += out
    return result


def _unfuse_native(proj, down, up, dims):
    """Split a Sound & Vision native fused pair: one A (rank × in) and stacked B (sum of outs × rank)."""
    names, outs = _fused_outs(proj, dims)
    if down.ndim != 2 or up.ndim != 2 or up.shape[1] != down.shape[0] or up.shape[0] != sum(outs):
        raise ValueError(f'{proj}: native fused tensors {tuple(down.shape)} x {tuple(up.shape)} do not match projections {outs}')
    result = {}
    row = 0
    for name, out in zip(names, outs):
        result[name] = (down, up[row:row + out])
        row += out
    return result


def deltas(tensors, dims=None):
    """Per-projection ``{(layer, module, proj): (A, B, scale)}`` plus ``io`` replacement weights, from any layout.

    ``A`` is ``rank x in``, ``B`` is ``out x rank``; the delta to add to a weight is ``scale * B @ A``."""
    np = _numpy()
    dims = dims or DIMS
    grouped = {}
    io = {}
    for key, value in tensors.items():
        m = KEY.search(key)
        if m:
            layer, module, proj, part = m.groups()
            module = _native_module(key, module)
            grouped.setdefault((int(layer), module, proj), {})[part] = value
            continue
        m = IO_KEY.search(key)
        if m:
            io.setdefault(m.group(1), {})[m.group(2)] = value
    result = {}
    for (layer, module, proj), parts in grouped.items():
        down = parts.get('lora_A', parts.get('lora_down'))
        up = parts.get('lora_B', parts.get('lora_up'))
        if down is None or up is None:
            raise ValueError(f'layers.{layer}.{module}.{proj}: the LoRA has only half of its pair')
        alpha = parts.get('alpha')
        if alpha is not None:
            alpha = float(np.asarray(alpha, dtype=np.float64).reshape(-1)[0])
        if proj in ('qkv_proj', 'gate_up_proj'):
            named = 'lora_A' in parts or 'lora_B' in parts
            native = _native_fused(down, up, _fused_outs(proj, dims)[1], named)
            split = _unfuse_native(proj, down, up, dims) if native else _unfuse(proj, down, up, dims)
            for name, (A, B) in split.items():
                scale = alpha / A.shape[0] if alpha is not None else 1.0
                result[(layer, module, name)] = (A, B, scale)
        else:
            if down.ndim != 2 or up.ndim != 2 or down.shape[0] != up.shape[1]:
                raise ValueError(f'layers.{layer}.{module}.{proj}: {tuple(down.shape)} and {tuple(up.shape)} are not a LoRA pair')
            scale = alpha / down.shape[0] if alpha is not None else 1.0
            result[(layer, module, proj)] = (down, up, scale)
    for name, parts in io.items():
        if 'weight' not in parts:
            raise ValueError(f'{name}: replacement weights need a weight tensor')
    return {'linears': result, 'io': io}


def drop_ar(delta):
    """Decoder-only view of a delta: NAR linears and io replacements, no planner / music-token adapters.

    Covers already have a transcribed score. Applying a writing LoRA to the AR path on that score
    collapses the music tokens into a short repeating palette; the decoder half can still colour the
    render without fighting the melody.
    """
    return {'linears': {k: v for k, v in delta['linears'].items() if k[1] in NAR_MODULES},
            'io': dict(delta.get('io') or {})}


def load(root, name, dims=None):
    """Deltas of the named file in ``loras/``."""
    path = path_of(root, name)
    if not path.is_file():
        raise FileNotFoundError('The LoRA ' + repr(name) + ' is not in the loras folder')
    return deltas(read_tensors(path), dims)


def expected_shape(key, dims):
    """(out, in) of the base weight a delta applies to; used to check a file against the model."""
    _, module, proj = key
    h, i = dims['hidden'], dims['intermediate']
    if proj == 'q_proj':
        return dims['q'], h
    if proj in ('k_proj', 'v_proj'):
        return dims['kv'], h
    if proj == 'o_proj':
        return h, dims['q']
    if proj in ('gate_proj', 'up_proj'):
        return i, h
    return h, i


def check(delta, dims=None):
    """Raise if any delta does not fit YuE2's layer sizes."""
    dims = dims or DIMS
    for key, (A, B, _) in delta['linears'].items():
        out, inner = expected_shape(key, dims)
        if key[0] >= dims['layers'] or A.shape[1] != inner or B.shape[0] != out:
            raise ValueError(f'layers.{key[0]}.{key[1]}.{key[2]}: LoRA {tuple(A.shape)} x {tuple(B.shape)} does not fit a {out}x{inner} weight')
    for name, parts in delta['io'].items():
        shape = (dims['hidden'], 64) if name == 'vae2llm' else (64, dims['hidden'])
        if tuple(parts['weight'].shape) != shape:
            raise ValueError(f'{name}: replacement weight {tuple(parts["weight"].shape)} should be {shape}')
    return delta


def chosen(request):
    """``[(file, strength), ...]`` the request asks for: the LoRA (writing) and the Sound LoRA (decoder), each
    only when named with a non-zero strength."""
    out = []
    for name_key, strength_key in (('lora', 'lora_strength'), ('sound_lora', 'sound_lora_strength')):
        name, strength = request.get(name_key) or '', float(request.get(strength_key, 1.0) or 0)
        if name and strength:
            out.append((name, strength))
    return out


def combine(parts):
    """One delta (strength 1) standing for several ``(delta, strength)`` applied together, e.g. a writing
    adapter on the planner and a sound adapter on the decoder.

    Deltas on the same projection add: their A rows are stacked and their B columns, each carrying its own
    ``strength * scale``, sit beside each other, so ``B @ A`` is the sum of the two products. Two files that
    both replace ``vae2llm`` / ``llm2vae`` cannot be summed, so that combination is refused."""
    np = _numpy()
    linears, io = {}, {}
    for delta, strength in parts:
        if not strength:
            continue
        for key, (A, B, scale) in delta['linears'].items():
            A = np.asarray(A, dtype=np.float32)
            B = np.asarray(B, dtype=np.float32) * float(scale * strength)
            if key in linears:
                A0, B0, _ = linears[key]
                A, B = np.concatenate([A0, A]), np.concatenate([B0, B], axis=1)
            linears[key] = (A, B, 1.0)
        for name, replacement in delta['io'].items():
            if name in io:
                raise ValueError('Both LoRAs replace ' + name + '; choose adapters that do not both carry decoder projections')
            io[name] = replacement
    return {'linears': linears, 'io': io}


def merge_torch(model, delta, strength):
    """Fold the deltas into a torch YuE2 model's weights (``W += strength * scale * B @ A``); io weights replaced."""
    import torch
    np = _numpy()
    if not strength:
        return 0
    touched = 0
    with torch.no_grad():
        for (layer, module, proj), (A, B, scale) in delta['linears'].items():
            linear = getattr(getattr(model.model.layers[layer], module), proj)
            change = torch.from_numpy(np.ascontiguousarray(B @ A)) * (strength * scale)
            linear.weight.add_(change.to(linear.weight.device, linear.weight.dtype))
            touched += 1
        for name, parts in delta['io'].items():
            linear = getattr(model, name)
            linear.weight.copy_(torch.from_numpy(np.ascontiguousarray(parts['weight'])).to(linear.weight.dtype))
            if 'bias' in parts and linear.bias is not None:
                linear.bias.copy_(torch.from_numpy(np.ascontiguousarray(parts['bias'])).to(linear.bias.dtype))
            touched += 1
    return touched
