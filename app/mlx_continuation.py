"""Extend on the MLX engine.

The same prompt assembly as cuda_engine.continue_plan / continue_semantic, built on the
prefix-based generate.py functions. ``engine`` is the imported ``model/generate`` module so
this file stays importable (and testable) without MLX.
"""
from dataclasses import asdict

import song_continuation


def guidance(cot, cfg_scale):
    """The pipeline's default CFG when the request leaves it unset."""
    return (1.01 if cot == 'off' else 1.0) if cfg_scale is None else cfg_scale


def continue_plan(engine, pipe, style, lyrics, cot, abc_prefix, sampling, seed, on_token=None):
    """Let the model write the rest of a score that was cut at a bar.

    Returns ``(abc_text, abc_ids, truncated)``; without a plan (cot=off) the score is empty.
    """
    if cot == 'off' or not abc_prefix:
        return None, [], False
    tok = pipe.tokenizer
    kept = list(tok.encode(abc_prefix))
    prefix = engine.token_prefix(tok, style, lyrics, cot) + kept
    limits = song_continuation.budget(asdict(sampling), len(prefix))
    ids, truncated = engine.generate_tokens(pipe.model, prefix, engine.Sampling(**limits), seed, 'abc', on_token=on_token)
    abc_ids = kept + [int(t) for t in ids]
    return tok.decode(abc_ids), abc_ids, truncated


def continue_semantic(engine, pipe, style, lyrics, cot, abc_ids, codec_prompt, sampling, seed, cfg_scale,
                      on_token=None):
    """Generate music tokens after a prompt of saved codec tokens; the prompt is kept so the whole song is synthesized.

    Returns ``(prefix, codec, truncated)`` where ``prefix`` is the text/score prefix synthesis needs.
    """
    tok = pipe.tokenizer
    prefix = engine.token_prefix(tok, style, lyrics, cot, abc_ids)
    prompt = [int(c) + engine.CODEC_OFFSET for c in codec_prompt]
    scale = guidance(cot, cfg_scale)
    negative = (engine.negative_prefix(tok, cot, abc_ids) + prompt) if scale != 1 else None
    limits = song_continuation.budget(asdict(sampling), len(prefix) + len(prompt), len(negative or []))
    ids, truncated = engine.generate_tokens(pipe.model, prefix + prompt, engine.Sampling(**limits), seed, 'semantic',
                                            negative, scale, legacy_off=cot == 'off', on_token=on_token)
    new = [int(t) - engine.CODEC_OFFSET for t in ids]
    if not new:
        raise RuntimeError('The model ended the song right away. Cut it earlier or try a different seed.')
    return prefix, list(codec_prompt) + new, truncated
