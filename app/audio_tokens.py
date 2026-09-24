"""Turn a recording into YuE2 music tokens, so uploaded audio can prime new songs.

YuE2's own audio-to-token encoder was never released. This uses the community
``yue2-mothersuperior-realaudio-tokenizer`` v9 head: MERT-v2-FullSong layer-20 features
(25 frames per second, normalised per recording) → a small transformer that predicts one
of YuE2's 32,768 semantic codes per frame. The codes are approximate (about one in five is
the exact code YuE2 would have written, most others render nearly the same), which is good
enough for the model to continue it.

Runs inside the audio-analysis environment (torch + transformers), next to SheetSage2.
The stock MERT encoder is loaded separately from SheetSage2's LoRA-merged copy because the
head was trained on the unmodified layer outputs. Window arithmetic is standard library
plus NumPy so it can be tested without torch.
"""
import subprocess
from pathlib import Path

import numpy as np

NAME = 'mothersuperior-realaudio-tokenizer-v9'
HEAD_FILE = 'tokens/tokenizer_head_joint_v9.safetensors'
MERT_FILES = ('model.safetensors', 'config.json', 'modeling_mert2.py', 'configuration_mert2.py')
SAMPLE_RATE = 24000
FRAMES_PER_SECOND = 25
LAYER = 20            # hidden_states index: the output of the 21st Conformer block
CHUNK_SECONDS = 30
MIN_CHUNK_SECONDS = 1
WINDOW = 512          # head context in frames (20.48 s)
VOCAB = 32768
FEATURE_DIM = 1024


def available(root):
    root = Path(root)
    return (root / HEAD_FILE).is_file() and all((root / 'mert' / name).is_file() for name in MERT_FILES)


def decode_audio(path, max_seconds=None):
    """Mono float32 samples at MERT's rate, decoded by FFmpeg like the other analysis inputs."""
    command = ['ffmpeg', '-v', 'error', '-nostdin', '-i', str(path), '-vn']
    if max_seconds:
        command += ['-t', str(float(max_seconds))]
    command += ['-ac', '1', '-ar', str(SAMPLE_RATE), '-f', 'f32le', 'pipe:1']
    result = subprocess.run(command, capture_output=True, timeout=600, check=False)
    if result.returncode:
        raise ValueError('Cannot decode audio: ' + result.stderr.decode(errors='replace')[-600:])
    audio = np.frombuffer(result.stdout, dtype='<f4').copy()
    if len(audio) < MIN_CHUNK_SECONDS * SAMPLE_RATE:
        raise ValueError('The recording must be at least one second long.')
    if not np.isfinite(audio).all():
        raise ValueError('The recording contains invalid samples.')
    return audio


def chunk_bounds(samples, chunk=CHUNK_SECONDS * SAMPLE_RATE, minimum=MIN_CHUNK_SECONDS * SAMPLE_RATE):
    """MERT is run on 30-second pieces; a trailing piece shorter than a second is dropped."""
    return [(start, min(samples, start + chunk)) for start in range(0, samples, chunk)
            if min(samples, start + chunk) - start >= minimum]


def normalize(features):
    """Per-recording instance normalisation, matching the head's training (``cfg.instnorm``)."""
    features = np.asarray(features, dtype=np.float32)
    return (features - features.mean(0)) / (features.std(0) + 1e-5)


def window_plan(total, window=WINDOW):
    """Sliding head windows at half-window stride.

    Returns ``(start, count, lo, hi)`` per window: the frames fed to the head and the span of
    predictions kept from it. A quarter window is trimmed from each side that has a
    neighbour, so every frame is predicted with context on both sides where possible.
    """
    starts = list(range(0, max(1, total - window + 1), window // 2))
    if starts[-1] + window < total:
        starts.append(max(0, total - window))
    plan = []
    for start in starts:
        count = min(window, total - start)
        lo = start + (0 if start == 0 else window // 4)
        hi = start + count - (0 if start + count >= total else window // 4)
        plan.append((start, count, lo, hi))
    return plan


def frame_count(samples):
    return int(round(samples / SAMPLE_RATE * FRAMES_PER_SECOND))


def load_head_state(path):
    """Weights of the tokenizer head. A ``.pt`` checkpoint stores them under ``model``; safetensors is the state dict."""
    import torch
    path = Path(path)
    if path.suffix == '.safetensors':
        from safetensors.torch import load_file
        return load_file(str(path))
    checkpoint = torch.load(str(path), map_location='cpu', weights_only=True)
    return checkpoint['model']


class SemanticTokenizer:
    def __init__(self, root, device):
        import torch
        from torch import nn
        from transformers import AutoModel
        root = Path(root)
        if not available(root):
            raise FileNotFoundError('Download Audio input under Models first.')
        self.device = device
        self.mert = AutoModel.from_pretrained(str(root / 'mert'), trust_remote_code=True, local_files_only=True,
                                              torch_dtype=torch.float32).to(device).eval()
        for parameter in self.mert.parameters():
            parameter.requires_grad_(False)

        class TokenHead(nn.Module):
            def __init__(self, width=512, layers=8, heads=8, window=WINDOW, input_dim=FEATURE_DIM, vocab=VOCAB):
                super().__init__()
                self.inp = nn.Linear(input_dim, width)
                self.pos = nn.Parameter(torch.empty(1, window, width))
                layer = nn.TransformerEncoderLayer(width, heads, 4 * width, dropout=0.1, batch_first=True,
                                                   norm_first=True, activation='gelu')
                self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
                self.norm = nn.LayerNorm(width)
                self.head = nn.Linear(width, vocab)

            def forward(self, x):
                return self.head(self.norm(self.enc(self.inp(x) + self.pos[:, :x.shape[1]])))

        head = TokenHead()
        head.load_state_dict(load_head_state(root / HEAD_FILE), strict=True)
        self.head = head.to(device).eval()
        for parameter in self.head.parameters():
            parameter.requires_grad_(False)

    def features(self, audio, progress=None):
        """MERT layer-20 features at 25 Hz for the whole recording, as float16 like the head's training data."""
        import torch
        import torch.nn.functional as F
        bounds = chunk_bounds(len(audio))
        if not bounds:
            raise ValueError('The recording must be at least one second long.')
        pieces = []
        for index, (start, end) in enumerate(bounds):
            if progress:
                progress({'stage': 'tokens_encoding', 'window': index + 1, 'windows': len(bounds)})
            chunk = torch.from_numpy(audio[start:end])[None].to(self.device)
            states = self.mert(chunk, output_hidden_states=True).hidden_states[LAYER][0]
            pieces.append(states.detach().float().cpu())
        joined = torch.cat(pieces)
        resampled = F.interpolate(joined.T[None], size=frame_count(len(audio)), mode='linear', align_corners=False)[0].T
        return resampled.half().numpy()

    def predict(self, features, progress=None):
        import torch
        features = normalize(features)
        total = len(features)
        out = np.zeros(total, dtype=np.int32)
        plan = window_plan(total)
        for index, (start, count, lo, hi) in enumerate(plan):
            if progress:
                progress({'stage': 'tokens_predicting', 'window': index + 1, 'windows': len(plan)})
            value = np.pad(features[start:start + count], ((0, WINDOW - count), (0, 0)))
            logits = self.head(torch.tensor(value[None], device=self.device))[0, :count]
            ids = logits.detach().float().argmax(-1).cpu().numpy()
            out[lo:hi] = ids[lo - start:hi - start]
        return out

    def tokenize(self, path, max_seconds=None, progress=None):
        if progress:
            progress({'stage': 'tokens_audio'})
        audio = decode_audio(path, max_seconds)
        codec = self.predict(self.features(audio, progress), progress)
        return {'codec': [int(t) for t in codec], 'seconds': round(len(audio) / SAMPLE_RATE, 3), 'tokenizer': NAME}
