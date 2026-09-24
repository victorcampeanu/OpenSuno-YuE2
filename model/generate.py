#!/usr/bin/env python3
"""YuE2 song generation on Apple Silicon with MLX.

    python generate.py --model ~/Desktop/Yue2-3B-MLX --style "indie pop, warm vocal" \
        --lyrics-file lyrics.txt --cot off --seed 831001 --out song.wav

Ports the yue2_infer protocol: symbolic ABC planning (cot=melody/full), semantic
codec AR decode with CFG, 32-step midpoint flow matching, tiled VAE decode.
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import re
import sys
import time
import unicodedata
import wave
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np

from yue2_model import KVCache, Yue2Model, load_model, pack_cfg_caches
from yue2_vae import load_vae

EOD = 151643
ABC_START, ABC_END = 151847, 151848
MUSIC_START, MUSIC_END = 151851, 151852
CODEC_OFFSET, CODEC_SIZE = 151853, 32768
VOCAB, CONTEXT = 184704, 24576
SAMPLE_RATE = 48000
INSTRUCTIONS = {
    "off": "Generate music with codec tokens from the given conditions.",
    "melody": "Generate a melody-only ABC transcription without chord symbols, then generate music with codec tokens from the given conditions.",
    "full": "Generate a chord-annotated ABC transcription, then generate music with codec tokens from the given conditions.",
}


class Tokenizer:
    """The frozen Qwen text/ABC BPE (not the audio codec)."""

    def __init__(self, merge_file: Path):
        import tiktoken
        ranks = {base64.b64decode(t): int(r) for t, r in
                 (line.split() for line in Path(merge_file).read_bytes().splitlines() if line)}
        specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<R>", "<S>", "<X>", "<mask>", "<sep>"]
        specials += [f"<extra_{i}>" for i in range(200)]
        specials[204:206] = ["<abc>", "</abc>"]
        pattern = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
        self._enc = tiktoken.Encoding("YuE2", pat_str=pattern, mergeable_ranks=ranks,
                                      special_tokens={s: i + len(ranks) for i, s in enumerate(specials)})

    def encode(self, text):
        return self._enc.encode_ordinary(unicodedata.normalize("NFC", text))

    def decode(self, ids):
        return self._enc.decode([int(i) for i in ids if 0 <= i < self._enc.n_vocab], errors="replace")


@dataclass(frozen=True)
class Sampling:
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 100
    repetition_penalty: float = 1.2
    penalty_window: int = 50
    min_tokens: int = 200
    max_tokens: int = 9000


def request_text(style, lyrics, cot):
    return f"{INSTRUCTIONS[cot]}\n[Tags]\n{style}\n[Lyrics]\n{lyrics}\n"


def token_prefix(tok, style, lyrics, cot, abc_ids=None):
    base = [EOD] + tok.encode(request_text(style, lyrics, cot))
    if cot == "off":
        return base + [ABC_START, ABC_END, MUSIC_START]
    if abc_ids is None:
        return base + [ABC_START]
    return base + [ABC_START] + list(abc_ids) + [ABC_END, MUSIC_START]


def negative_prefix(tok, cot, abc_ids):
    base = [EOD] + tok.encode(INSTRUCTIONS[cot])
    if cot == "off":
        return base + [MUSIC_START]
    return base + [ABC_START] + list(abc_ids) + [ABC_END, MUSIC_START]


# ── AR sampling ──────────────────────────────────────────────────────────────

def phase_window(phase):
    """Contiguous vocab range [lo, hi) that covers every id ``phase`` may emit.

    The semantic stage only needs MUSIC_END plus the codec block that starts
    right after it, so the output head can skip the other 82% of the vocabulary.
    ABC text spans most of the vocabulary and keeps the full head.
    """
    if phase == "semantic":
        assert MUSIC_END == CODEC_OFFSET - 1
        return MUSIC_END, CODEC_OFFSET + CODEC_SIZE
    return 0, VOCAB


def allowed_mask(phase, dtype, window=(0, VOCAB)):
    """Additive mask (0 / -inf) over vocab ids [lo, hi)."""
    idx = mx.arange(*window)
    end = ABC_END if phase == "abc" else MUSIC_END
    ok = (idx < EOD) if phase == "abc" else ((idx >= CODEC_OFFSET) & (idx < CODEC_OFFSET + CODEC_SIZE))
    return mx.where(ok | (idx == end), 0.0, -mx.inf).astype(dtype)


def _recent_tokens(history, window):
    if history is None:
        return None
    recent = history[-window:]
    if isinstance(recent, list):
        return mx.array(recent) if recent else None
    return recent if recent.size else None


def _sample_token(scores, key, temperature, lo=0):
    """Return (token array, rng key). Stays on GPU until the caller copies it.

    ``scores`` may cover only vocab ids [lo, lo + len). Sampling then draws the
    Gumbel noise for the whole vocabulary and slices it, which reproduces
    ``mx.random.categorical`` over full-vocab logits exactly: a seed yields the
    same song whether or not the head was sliced.
    """
    if temperature == 0:
        token = mx.argmax(scores)
        return (token + lo if lo else token), key
    key, sub = mx.random.split(key)
    scores = scores.astype(mx.float32)
    n = scores.shape[-1]
    if n == VOCAB:
        return mx.random.categorical(scores, key=sub), key
    noise = mx.random.gumbel((VOCAB,), key=sub)[lo:lo + n]
    return mx.argmax(scores + noise) + lo, key


def distribution(logits, s: Sampling, history, step, phase, allowed, legacy_off, lo=0):
    """Masked, penalised, temperature/top-k/top-p filtered scores.

    ``logits`` and ``allowed`` cover vocab ids [lo, lo + n); ``history`` holds
    full-vocab ids. Restricting to a window changes no value that survives the
    mask, so results match the full-vocab computation.
    """
    # Historical cot=off arithmetic stays in BF16; symbolic/CFG paths upcast.
    scores = logits if legacy_off else logits.astype(mx.float32)
    n = scores.shape[-1]
    end = (ABC_END if phase == "abc" else MUSIC_END) - lo
    scores = scores + allowed
    if step < s.min_tokens:
        scores[end] = -mx.inf
    recent = _recent_tokens(history, s.penalty_window)
    if s.repetition_penalty != 1.0 and recent is not None:
        if lo:
            recent = recent - lo
        freq = mx.zeros((n,), scores.dtype).at[recent].add(1)
        alpha = mx.power(mx.array(s.repetition_penalty, scores.dtype), freq)
        scores = mx.where(scores < 0, scores * alpha, scores / alpha)
    if s.temperature == 0:
        return scores
    if s.temperature != 1:
        scores = scores / s.temperature
    kth = n - min(s.top_k, n)
    threshold = mx.partition(scores, kth)[kth]
    scores = mx.where(scores < threshold, -mx.inf, scores)
    if s.top_p < 1:
        order = mx.argsort(-scores)
        values = scores[order]
        probs = mx.softmax(values.astype(mx.float32), axis=-1)
        removed = (mx.cumsum(probs) - probs) > s.top_p
        removed = removed & (mx.arange(n) >= (3 if legacy_off else 1))
        scores = mx.put_along_axis(scores, order, mx.where(removed, -mx.inf, values), axis=0)
    return scores


def generate_tokens(model: Yue2Model, prefix, s: Sampling, seed, phase, negative=None, cfg_scale=1.0,
                    legacy_off=False, on_token=None):
    """Return (ids, truncated). CFG prefills separately, then decodes both branches together.

    Sampling stays on Metal. The CPU only copies tokens to stop on EOS and to
    report progress (first token, then every 20), while the next AR step is
    already running.
    """
    if len(prefix) + s.max_tokens > CONTEXT or (negative and len(negative) + s.max_tokens > CONTEXT):
        raise ValueError("Prefix + generation budget exceeds the 24576 context")
    if cfg_scale != 1 and negative is None:
        raise ValueError("CFG requires a negative prefix")
    n_layers = len(model.model.layers)
    window = phase_window(phase)
    lo = window[0]
    if cfg_scale != 1:
        cond_caches = [KVCache() for _ in range(n_layers)]
        uncond_caches = [KVCache() for _ in range(n_layers)]
        cond = model.ar_step(mx.array([prefix]), cond_caches, window=window)[0]
        uncond = model.ar_step(mx.array([negative]), uncond_caches, window=window)[0]
        caches = pack_cfg_caches(cond_caches, uncond_caches)
        mx.async_eval(cond, uncond)
    else:
        caches = [KVCache() for _ in range(n_layers)]
        cond = model.ar_step(mx.array([prefix]), caches, window=window)[0]
        uncond = None
        mx.async_eval(cond)
    end = ABC_END if phase == "abc" else MUSIC_END
    allowed = allowed_mask(phase, cond.dtype if legacy_off else mx.float32, window)
    key = mx.random.key(seed)
    chosen = mx.zeros((s.max_tokens,), dtype=mx.int32)
    history, eos, copied = [], False, 0

    def take_tokens(upto):
        nonlocal eos, copied
        if upto <= copied:
            return
        chunk = chosen[copied:upto]
        mx.eval(chunk)
        for token in (int(t) for t in chunk.tolist()):
            if on_token is not None:
                on_token(phase, token)
            if token == end:
                eos = True
                break
            history.append(token)
        copied = upto

    for step in range(s.max_tokens):
        logits = cond if uncond is None else uncond + cfg_scale * (cond - uncond)
        scores = distribution(logits, s, chosen[:step], step, phase, allowed, legacy_off, lo)
        token, key = _sample_token(scores, key, s.temperature, lo)
        chosen = chosen.at[step].add(token.astype(mx.int32))
        if step + 1 < s.max_tokens:
            nxt = token.reshape((1, 1))
            if uncond is None:
                cond = model.ar_step(nxt, caches, window=window)[0]
                mx.async_eval(cond, token)
            else:
                both = model.ar_step(mx.concatenate([nxt, nxt], axis=0), caches, window=window)
                cond, uncond = both[0], both[1]
                mx.async_eval(cond, uncond, token)
        if step == 0 or (step + 1) % 20 == 0 or step + 1 == s.max_tokens:
            take_tokens(step + 1)
            if eos:
                break
    return history, not eos


class ScoreWriter:
    """Let the model continue a score whose text is partly dictated, token by token.

    ``commit(text)`` appends forced text (a transcribed bar); ``propose(max_tokens, stop)``
    samples a continuation of everything committed so far and returns it as text without
    committing it. The committed text is always re-encoded with the model's own BPE, so a
    ``|"`` barline-and-quote merged token is fed exactly as the model would have written
    it; the KV cache is trimmed back to the last token that still agrees and only the new
    tail is prefilled, which keeps a bar-by-bar arrangement at a few model steps per bar.
    """

    def __init__(self, model: Yue2Model, tokenizer, request_ids, s: Sampling, seed):
        self.model, self.tok, self.s = model, tokenizer, s
        self.base = list(request_ids)
        self.caches = [KVCache() for _ in model.model.layers]
        self.fed, self.text, self.logits = [], "", None
        self.key = mx.random.key(seed)
        self.allowed = allowed_mask("abc", mx.float32)
        self._first_masks = {}

    def commit(self, text):
        self.text += text

    def first_mask(self, pattern):
        """Additive mask keeping only text tokens whose decoded text fully matches ``pattern``."""
        if pattern not in self._first_masks:
            regex = re.compile(pattern)
            ids = [i for i in range(EOD) if regex.fullmatch(self.tok.decode([i]))]
            if not ids:
                raise ValueError(f"No token matches {pattern!r}")
            mask = np.full(VOCAB, -np.inf, np.float32)
            mask[ids] = 0
            self._first_masks[pattern] = mx.array(mask)
        return self._first_masks[pattern]

    def _sync(self):
        target = self.base + self.tok.encode(self.text)
        if len(target) + 64 > CONTEXT:
            raise ValueError("The arranged score does not fit the model context")
        common = 0
        for a, b in zip(self.fed, target):
            if a != b:
                break
            common += 1
        if common < len(self.fed):
            for cache in self.caches:
                cache.offset = common
            del self.fed[common:]
        pending = target[common:]
        if pending:
            self.logits = self.model.ar_step(mx.array([pending]), self.caches)[0]
            self.fed.extend(pending)

    def propose(self, max_tokens, stop, first=None):
        self._sync()
        sampled = []
        for step in range(max_tokens):
            recent = self.fed[-self.s.penalty_window:]
            logits = self.logits + self.first_mask(first) if first and step == 0 else self.logits
            scores = distribution(logits, self.s, recent, 0, "abc", self.allowed, False)
            token, self.key = _sample_token(scores, self.key, self.s.temperature)
            token = int(token.item())
            if token == ABC_END:
                break
            self.logits = self.model.ar_step(mx.array([[token]]), self.caches)[0]
            self.fed.append(token)
            sampled.append(token)
            if stop(self.tok.decode(sampled)):
                break
        return self.tok.decode(sampled)


# ── NAR flow matching ────────────────────────────────────────────────────────

def chunk_ranges(frames, prefix_tokens, context=CONTEXT):
    size = min((context - prefix_tokens - 3) // 2, CONTEXT)
    if frames < 1 or size < 1:
        raise ValueError("Empty codec or prefix leaves no acoustic context")
    return [(a, min(a + size, frames)) for a in range(0, frames, size)]


def _logit(t):
    return max(-20.0, min(20.0, math.log(t / (1 - t)))) if 0 < t < 1 else (20.0 if t >= 1 else -20.0)


def synthesize(model: Yue2Model, prefix, codec, seed, steps=32, noise=None, on_progress=None):
    """Return latents [frames,64] float32 via midpoint ODE from t=1 → 0, solved per context chunk."""
    if noise is None:
        noise = mx.random.normal((len(codec), 64), key=mx.random.key(seed))
    dt = 1.0 / steps
    out = []
    for a, b in chunk_ranges(len(codec), len(prefix)):
        ar_tokens = prefix + [c + CODEC_OFFSET for c in codec[a:b]] + [MUSIC_END]
        ar_cache = model.nar_prefill(ar_tokens)
        state = noise[a:b].astype(mx.bfloat16)
        for step in range(steps):
            t = 1.0 - step * dt
            v1 = model.nar_velocity(state, _logit(t), ar_cache, len(ar_tokens))
            mid = state - v1 * (dt / 2)
            state = state - model.nar_velocity(mid, _logit(t - dt / 2), ar_cache, len(ar_tokens)) * dt
            mx.eval(state)
            if on_progress is not None:
                on_progress(step + 1, steps)
        out.append(state.astype(mx.float32))
    return mx.concatenate(out)


# ── Pipeline ─────────────────────────────────────────────────────────────────

class Yue2Pipeline:
    def __init__(self, path, log=print, on_load=None):
        self.path, self.log = Path(path), log
        on_load = on_load or (lambda detail: None)
        on_load('Reading model configuration')
        gen = json.loads((self.path / "yue2_generation_config.json").read_text())
        self.abc_sampling, self.semantic_sampling = Sampling(**gen["abc"]), Sampling(**gen["semantic"])
        self.ode_steps = gen.get("ode_steps", 32)
        on_load('Loading text tokenizer')
        self.tokenizer = Tokenizer(self.path / "qwen.tiktoken")
        self.vae = load_vae(self.path, on_load=on_load)
        self.model = load_model(self.path, on_load=on_load)

    def _progress(self, label):
        count, start = [0], [None]

        def on_token(_phase, _token):
            count[0] += 1
            if start[0] is None:
                start[0] = time.perf_counter()  # decode rate excludes the prefill
            if count[0] % 200 == 0:
                self.log(f"[{label}] {count[0]} tokens, {(count[0] - 1) / (time.perf_counter() - start[0]):.1f} tok/s")
        return on_token

    def __call__(self, style, lyrics, cot="full", seed=831001, abc=None, cfg_scale=None,
                 semantic_sampling=None, steps=None, on_latents=None):
        if cot not in INSTRUCTIONS:
            raise ValueError("cot must be off, melody or full")
        if abc is not None and cot == "off":
            raise ValueError("External ABC requires cot=melody/full")
        tok = self.tokenizer
        # 1. symbolic plan
        abc_ids, abc_text = [], None
        if cot != "off":
            if abc is not None:
                abc_ids, abc_text = tok.encode(abc), abc
            else:
                self.log("[plan] generating ABC score")
                abc_ids, truncated = generate_tokens(self.model, token_prefix(tok, style, lyrics, cot),
                                                     self.abc_sampling, seed, "abc", on_token=self._progress("abc"))
                abc_text = tok.decode(abc_ids)
                if truncated:
                    self.log("[plan] ABC hit max_tokens")
        prefix = token_prefix(tok, style, lyrics, cot, abc_ids)
        # 2. semantic codec tokens
        guidance = (1.01 if cot == "off" else 1.0) if cfg_scale is None else cfg_scale
        negative = negative_prefix(tok, cot, abc_ids) if guidance != 1 else None
        self.log(f"[semantic] prefix {len(prefix)} tokens, cfg {guidance}")
        ids, truncated = generate_tokens(self.model, prefix, semantic_sampling or self.semantic_sampling, seed,
                                         "semantic", negative, guidance, legacy_off=cot == "off",
                                         on_token=self._progress("semantic"))
        if truncated:
            self.log("[semantic] hit max_tokens")
        codec = [t - CODEC_OFFSET for t in ids]
        if not codec:
            raise RuntimeError("Semantic stage produced no codec tokens")
        # 3. acoustic latents
        self.log(f"[nar] {len(codec)} frames ({len(codec) * 1920 / SAMPLE_RATE:.1f}s), {steps or self.ode_steps} midpoint steps")
        latents = synthesize(self.model, prefix, codec, seed, steps or self.ode_steps,
                             on_progress=lambda i, n: i % 8 == 0 and self.log(f"[nar] step {i}/{n}"))
        if on_latents is not None:
            on_latents(latents)
        # 4. waveform
        self.log("[vae] decoding")
        return self.decode(latents), {"abc": abc_text, "codec": codec, "latents": latents, "prefix": prefix}

    def decode(self, latents):
        return mx.clip(self.vae.decode_tiled(latents), -1, 1)


def write_wav(path, audio, sample_rate=SAMPLE_RATE):
    pcm = (np.clip(np.array(audio, dtype=np.float32), -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as f:
        f.setnchannels(pcm.shape[1])
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(pcm.tobytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True, help="MLX package dir from convert.py")
    parser.add_argument("--style", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--lyrics")
    group.add_argument("--lyrics-file", type=Path)
    parser.add_argument("--cot", default="full", choices=list(INSTRUCTIONS))
    parser.add_argument("--abc-file", type=Path, help="external ABC score (cot=melody/full)")
    parser.add_argument("--seed", type=int, default=831001)
    parser.add_argument("--cfg-scale", type=float)
    parser.add_argument("--steps", type=int, help="NAR midpoint steps (default from generation config)")
    parser.add_argument("--max-semantic-tokens", type=int, help="cap song length in codec frames (40 ms each)")
    parser.add_argument("--out", type=Path, default=Path("yue2.wav"))
    parser.add_argument("--decode-latents", type=Path, help="skip generation; decode a saved <out>.latents.npy")
    args = parser.parse_args()

    lyrics = args.lyrics if args.lyrics is not None else args.lyrics_file.read_text()
    abc = args.abc_file.read_text() if args.abc_file else None
    log = lambda msg: print(msg, file=sys.stderr, flush=True)
    pipe = Yue2Pipeline(args.model, log=log)
    sampling = None
    if args.max_semantic_tokens:
        sampling = Sampling(**{**pipe.semantic_sampling.__dict__, "max_tokens": args.max_semantic_tokens,
                               "min_tokens": min(pipe.semantic_sampling.min_tokens, args.max_semantic_tokens)})
    start = time.perf_counter()
    latents_path = args.out.with_suffix(".latents.npy")
    if args.decode_latents:
        audio, info = pipe.decode(mx.array(np.load(args.decode_latents))), {"abc": None}
    else:
        audio, info = pipe(args.style, lyrics, cot=args.cot, seed=args.seed, abc=abc, cfg_scale=args.cfg_scale,
                           semantic_sampling=sampling, steps=args.steps,
                           on_latents=lambda z: np.save(latents_path, np.array(z)))
    write_wav(args.out, audio)
    if info["abc"] is not None:
        args.out.with_suffix(".abc").write_text(info["abc"])
    log(f"[done] {args.out} {audio.shape[0] / SAMPLE_RATE:.1f}s in {time.perf_counter() - start:.0f}s")


if __name__ == "__main__":
    main()
