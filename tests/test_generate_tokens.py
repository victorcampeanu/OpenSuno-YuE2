import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'model'))

try:
    import mlx.core as mx
    from generate import CODEC_OFFSET, MUSIC_END, MUSIC_START, Sampling, VOCAB, generate_tokens
    from yue2_model import KVCache, Yue2Model, pack_cfg_caches
except ImportError:
    mx = None


def _legacy_generate(model, prefix, s, seed, phase, negative=None, cfg_scale=1.0, on_token=None):
    from generate import allowed_mask, distribution
    caches = [[KVCache() for _ in model.model.layers] for _ in range(2 if cfg_scale != 1 else 1)]
    cond = model.ar_step(mx.array([prefix]), caches[0])[0]
    uncond = model.ar_step(mx.array([negative]), caches[1])[0] if cfg_scale != 1 else None
    end = MUSIC_END
    allowed = allowed_mask(phase, mx.float32)
    key = mx.random.key(seed)
    history, eos = [], False
    for step in range(s.max_tokens):
        logits = cond if uncond is None else uncond + cfg_scale * (cond - uncond)
        scores = distribution(logits, s, history, step, phase, allowed, False)
        if s.temperature == 0:
            token = int(mx.argmax(scores).item())
        else:
            key, sub = mx.random.split(key)
            token = int(mx.random.categorical(scores.astype(mx.float32), key=sub).item())
        if on_token is not None:
            on_token(phase, token)
        if token == end:
            eos = True
            break
        history.append(token)
        if step + 1 < s.max_tokens:
            nxt = mx.array([[token]])
            cond = model.ar_step(nxt, caches[0])[0]
            if uncond is not None:
                uncond = model.ar_step(nxt, caches[1])[0]
    return history, not eos


class SequenceModel:
    """Deterministic next-token head used to compare host-sync batching with the old loop."""

    def __init__(self, run=6):
        self.run = run
        self.model = SimpleNamespace(layers=[object()])

    def ar_step(self, tokens, caches, window=None):
        batch, steps = tokens.shape
        dummy = mx.zeros((batch, 1, steps, 4), mx.float32)
        for cache in caches:
            cache.update(dummy, dummy)
        logits = mx.full((int(batch), VOCAB), -1e4, dtype=mx.float32)
        for row in range(int(batch)):
            last = int(tokens[row, -1].item())
            nxt = MUSIC_END if last >= CODEC_OFFSET + self.run - 1 else (
                CODEC_OFFSET if last < CODEC_OFFSET else last + 1)
            logits[row, nxt] = 0
        return logits if window is None else logits[:, window[0]:window[1]]


def tiny_model():
    cfg = dict(hidden_size=64, intermediate_size=128, num_attention_heads=1,
               num_key_value_heads=1, head_dim=64, rms_norm_eps=1e-6,
               rope_theta=1000000, vocab_size=128, num_hidden_layers=2,
               latent_dim=64, max_latent_frames=32, timestep_shift=1.0)
    model = Yue2Model(cfg)
    model.set_dtype(mx.bfloat16)
    mx.eval(model.parameters())
    return model


@unittest.skipIf(mx is None, 'MLX is required for token-generation tests')
class GenerateTokensTest(unittest.TestCase):
    def _sampling(self, **kwargs):
        values = dict(temperature=0, top_p=1, top_k=VOCAB, repetition_penalty=1.2,
                      penalty_window=50, min_tokens=0, max_tokens=40)
        values.update(kwargs)
        return Sampling(**values)

    def test_greedy_matches_per_token_host_sync(self):
        prefix = [MUSIC_START]
        sampling = self._sampling()
        seen, legacy_seen = [], []
        ids, truncated = generate_tokens(SequenceModel(), prefix, sampling, 1, 'semantic',
                                         on_token=lambda phase, token: seen.append(token))
        legacy, legacy_truncated = _legacy_generate(SequenceModel(), prefix, sampling, 1, 'semantic',
                                                    on_token=lambda phase, token: legacy_seen.append(token))
        self.assertEqual(ids, list(range(CODEC_OFFSET, CODEC_OFFSET + 6)))
        self.assertEqual(ids, legacy)
        self.assertEqual(seen, legacy_seen)
        self.assertFalse(truncated)
        self.assertFalse(legacy_truncated)
        self.assertEqual(seen[-1], MUSIC_END)

    def test_cfg_matches_per_token_host_sync(self):
        prefix, negative = [MUSIC_START], [MUSIC_START]
        sampling = self._sampling(temperature=0.8, max_tokens=24)
        ids, _ = generate_tokens(SequenceModel(8), prefix, sampling, 831001, 'semantic',
                                 negative=negative, cfg_scale=1.2)
        legacy, _ = _legacy_generate(SequenceModel(8), prefix, sampling, 831001, 'semantic',
                                     negative=negative, cfg_scale=1.2)
        self.assertEqual(ids, legacy)
        self.assertGreaterEqual(len(ids), 8)

    def test_cfg_unequal_prefixes_match_sequential(self):
        prefix = [MUSIC_START, CODEC_OFFSET, CODEC_OFFSET + 1]
        negative = [MUSIC_START]
        sampling = self._sampling(temperature=0.7, max_tokens=20)
        ids, _ = generate_tokens(SequenceModel(10), prefix, sampling, 42, 'semantic',
                                 negative=negative, cfg_scale=1.5)
        legacy, _ = _legacy_generate(SequenceModel(10), prefix, sampling, 42, 'semantic',
                                     negative=negative, cfg_scale=1.5)
        self.assertEqual(ids, legacy)

    def test_hits_max_tokens_without_eos(self):
        sampling = self._sampling(max_tokens=4, min_tokens=4)
        ids, truncated = generate_tokens(SequenceModel(20), [MUSIC_START], sampling, 1, 'semantic')
        self.assertEqual(ids, list(range(CODEC_OFFSET, CODEC_OFFSET + 4)))
        self.assertTrue(truncated)

    def test_batched_cfg_logits_match_sequential(self):
        model = tiny_model()
        prefix, negative = [1, 2, 3, 4, 5, 6], [7, 8]
        cond_caches = [KVCache() for _ in model.model.layers]
        uncond_caches = [KVCache() for _ in model.model.layers]
        seq_cond = [KVCache() for _ in model.model.layers]
        seq_uncond = [KVCache() for _ in model.model.layers]
        cond = model.ar_step(mx.array([prefix]), cond_caches)
        uncond = model.ar_step(mx.array([negative]), uncond_caches)
        mx.eval(cond, uncond)
        model.ar_step(mx.array([prefix]), seq_cond)
        model.ar_step(mx.array([negative]), seq_uncond)
        packed = pack_cfg_caches(cond_caches, uncond_caches)
        for token in (9, 10, 11, 3, 4):
            nxt = mx.array([[token]])
            both = model.ar_step(mx.concatenate([nxt, nxt], axis=0), packed)
            one = model.ar_step(nxt, seq_cond)
            two = model.ar_step(nxt, seq_uncond)
            mx.eval(both, one, two)
            cond_err = float(mx.abs(both[0].astype(mx.float32) - one[0].astype(mx.float32)).max())
            uncond_err = float(mx.abs(both[1].astype(mx.float32) - two[0].astype(mx.float32)).max())
            self.assertLess(cond_err, 2e-2, msg=f'cond drift {cond_err} at token {token}')
            self.assertLess(uncond_err, 2e-2, msg=f'uncond drift {uncond_err} at token {token}')

    def _assert_sliced_head_matches(self, model, atol):
        import mlx.nn as nn
        lo, hi = 40, 100
        caches = [KVCache() for _ in model.model.layers]
        full = model.ar_step(mx.array([[1, 2, 3]]), caches)
        caches = [KVCache() for _ in model.model.layers]
        part = model.ar_step(mx.array([[1, 2, 3]]), caches, window=(lo, hi))
        mx.eval(full, part)
        self.assertEqual(part.shape, (1, hi - lo))
        err = float(mx.abs(full[:, lo:hi].astype(mx.float32) - part.astype(mx.float32)).max())
        self.assertLess(err, atol, msg=f'sliced head drift {err}')
        self.assertIs(model.logits_head(lo, hi), model.logits_head(lo, hi))
        self.assertIs(model.logits_head(0, model.config['vocab_size']), model.lm_head)
        self.assertNotIn('_sliced_heads', dict(model.parameters()))
        self.assertIsInstance(model.lm_head, (nn.Linear, nn.QuantizedLinear))

    def test_sliced_head_matches_full_bf16(self):
        self._assert_sliced_head_matches(tiny_model(), 1e-6)

    def test_sliced_head_matches_full_quantized(self):
        import mlx.nn as nn
        model = tiny_model()
        nn.quantize(model, 64, 8)
        mx.eval(model.parameters())
        self._assert_sliced_head_matches(model, 1e-6)

    def test_windowed_sampling_matches_full_vocab(self):
        from generate import _sample_token, allowed_mask, distribution, phase_window
        lo, hi = phase_window('semantic')
        s = Sampling(temperature=1.0, top_p=0.95, top_k=100, repetition_penalty=1.2, penalty_window=50,
                     min_tokens=1, max_tokens=10)
        logits = mx.random.normal((VOCAB,), key=mx.random.key(3)) * 4
        history = mx.array([CODEC_OFFSET + 5, CODEC_OFFSET + 5, CODEC_OFFSET + 9])
        full = distribution(logits, s, history, 5, 'semantic', allowed_mask('semantic', mx.float32), False)
        part = distribution(logits[lo:hi], s, history, 5, 'semantic',
                            allowed_mask('semantic', mx.float32, (lo, hi)), False, lo)
        mx.eval(full, part)
        self.assertTrue(mx.array_equal(full[lo:hi], part, equal_nan=True).item())
        self.assertTrue(mx.all(full[:lo] == -mx.inf).item() and mx.all(full[hi:] == -mx.inf).item())
        for seed in range(20):
            key = mx.random.key(seed)
            a, _ = _sample_token(full, key, 1.0)
            b, _ = _sample_token(part, key, 1.0, lo)
            self.assertEqual(int(a.item()), int(b.item()))
        a, _ = _sample_token(full, key, 0)
        b, _ = _sample_token(part, key, 0, lo)
        self.assertEqual(int(a.item()), int(b.item()))


if __name__ == '__main__':
    unittest.main()
