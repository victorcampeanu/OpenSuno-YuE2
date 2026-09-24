"""Switchable LoRA adapters on the MLX model: wrapped, merged, cleared, and coexisting with the real-audio decoder."""
import sys, tempfile, unittest
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'model'))
try:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    import adapters
except ImportError:   # the Windows / CUDA checkouts have no MLX
    mx = None
import loras

H, KV, I = 32, 16, 48   # big enough for MLX's smallest quantization group
DIMS = {'hidden': H, 'q': H, 'kv': KV, 'intermediate': I, 'layers': 2}


def tiny_model():
    class Attn(nn.Module):
        def __init__(s):
            super().__init__(); s.q_proj = nn.Linear(H, H, bias=False); s.k_proj = nn.Linear(H, KV, bias=False)
            s.v_proj = nn.Linear(H, KV, bias=False); s.o_proj = nn.Linear(H, H, bias=False)
    class MLP(nn.Module):
        def __init__(s):
            super().__init__(); s.gate_proj = nn.Linear(H, I, bias=False); s.up_proj = nn.Linear(H, I, bias=False); s.down_proj = nn.Linear(I, H, bias=False)
    class Layer(nn.Module):
        def __init__(s):
            super().__init__(); s.self_attn = Attn(); s.mlp = MLP(); s.nar_self_attn = Attn(); s.nar_mlp = MLP()
    class Backbone(nn.Module):
        def __init__(s):
            super().__init__(); s.layers = [Layer() for _ in range(2)]
    class Model(nn.Module):
        def __init__(s):
            super().__init__(); s.model = Backbone(); s.vae2llm = nn.Linear(64, H); s.llm2vae = nn.Linear(H, 64)
    m = Model(); mx.eval(m.parameters())
    return m


@unittest.skipIf(mx is None, 'MLX is not installed')
class AdapterTest(unittest.TestCase):
    def setUp(self):
        self.m = tiny_model()
        rng = np.random.default_rng(0)
        self.A = rng.standard_normal((3, H)).astype(np.float32); self.B = rng.standard_normal((H, 3)).astype(np.float32)
        self.A2 = rng.standard_normal((3, I)).astype(np.float32); self.B2 = rng.standard_normal((H, 3)).astype(np.float32)
        self.delta = loras.check({'linears': {(0, 'self_attn', 'q_proj'): (self.A, self.B, 0.5), (1, 'nar_mlp', 'down_proj'): (self.A2, self.B2, 1.0)},
                                  'io': {'vae2llm': {'weight': np.full((H, 64), 2.0, np.float32), 'bias': np.zeros(H, np.float32)}}}, DIMS)
        self.x = mx.random.normal((3, H)); self.x12 = mx.random.normal((3, I))
        self.q0 = self.m.model.layers[0].self_attn.q_proj(self.x); self.d0 = self.m.model.layers[1].nar_mlp.down_proj(self.x12)

    def expected(self):
        return (self.q0 + (self.x @ mx.array(self.A).T @ mx.array(self.B).T) * 1.0,        # scale .5 x strength 2
                self.d0 + (self.x12 @ mx.array(self.A2).T @ mx.array(self.B2).T) * 2.0)

    def close(self, a, b, atol=5e-2):
        # MLX 0.32's float32 GPU matmul is only good to ~1e-3 relative on Apple silicon; the checks are about structure, not bits.
        self.assertTrue(mx.allclose(a, b, rtol=5e-3, atol=atol).item(), (a, b))

    def test_wrapped_adapters_switch_on_and_off(self):
        m = self.m
        self.assertEqual(adapters.apply(m, self.delta, strength=2.0), 3)
        q, d = self.expected()
        self.close(m.model.layers[0].self_attn.q_proj(self.x), q); self.close(m.model.layers[1].nar_mlp.down_proj(self.x12), d)
        z = mx.random.normal((3, 64)); self.close(m.vae2llm(z), z @ mx.full((H, 64), 2.0).T, atol=1e-3)
        untouched = m.model.layers[1].self_attn.k_proj
        self.close(untouched(self.x), adapters.innermost(untouched)(self.x))
        adapters.clear(m)
        self.close(m.model.layers[0].self_attn.q_proj(self.x), self.q0); self.close(m.model.layers[1].nar_mlp.down_proj(self.x12), self.d0)
        # A second file reuses the wrappers and replaces the set.
        adapters.apply(m, {'linears': {(0, 'self_attn', 'q_proj'): (self.A, self.B, 1.0)}, 'io': {}}, 1.0)
        self.close(m.model.layers[1].nar_mlp.down_proj(self.x12), self.d0)
        self.assertIs(m.__dict__['_adapters'], adapters.attach(m))

    def test_merged_weights_cost_nothing_per_token_and_come_back_from_the_checkpoint(self):
        m = self.m
        with tempfile.TemporaryDirectory() as d:
            weights = Path(d) / 'model.safetensors'
            mx.save_safetensors(str(weights), dict(tree_flatten(m.parameters())))
            before = np.array(m.model.layers[0].self_attn.q_proj.weight)
            self.assertEqual(adapters.apply(m, self.delta, strength=2.0, weights=weights, label=('a', 2.0)), 3)
            wrappers = m.__dict__['_adapters']
            q_wrapper = wrappers['linears'][(0, 'self_attn', 'q_proj')]
            self.assertFalse(q_wrapper.active); self.assertEqual(set(wrappers['merged']), {(0, 'self_attn', 'q_proj'), (1, 'nar_mlp', 'down_proj')})
            self.assertFalse(np.allclose(np.array(q_wrapper.base.weight), before))
            q, dd = self.expected()
            self.close(m.model.layers[0].self_attn.q_proj(self.x), q); self.close(m.model.layers[1].nar_mlp.down_proj(self.x12), dd)
            # The same choice again is a no-op; a different strength re-merges from the stock weights, not on top.
            self.assertEqual(adapters.apply(m, self.delta, strength=2.0, weights=weights, label=('a', 2.0)), 3)
            self.close(m.model.layers[0].self_attn.q_proj(self.x), q)
            adapters.apply(m, self.delta, strength=1.0, weights=weights, label=('a', 1.0))
            self.close(m.model.layers[0].self_attn.q_proj(self.x), self.q0 + (self.x @ mx.array(self.A).T @ mx.array(self.B).T) * 0.5)
            adapters.clear(m)
            np.testing.assert_array_equal(np.array(m.model.layers[0].self_attn.q_proj.weight), before)
            self.close(m.model.layers[1].nar_mlp.down_proj(self.x12), self.d0)
            self.assertEqual(wrappers['merged'], {}); self.assertIsNone(wrappers['applied'])

    def test_quantized_linears_fall_back_to_wrappers(self):
        m = self.m
        target = m.model.layers[0].self_attn
        target.q_proj = nn.QuantizedLinear.from_linear(target.q_proj, group_size=32, bits=8)
        base_q = target.q_proj(self.x)
        with tempfile.TemporaryDirectory() as d:
            weights = Path(d) / 'model.safetensors'
            mx.save_safetensors(str(weights), dict(tree_flatten(m.parameters())))
            adapters.apply(m, {'linears': {(0, 'self_attn', 'q_proj'): (self.A, self.B, 1.0)}, 'io': {}}, 1.0, weights=weights)
            wrappers = m.__dict__['_adapters']
            self.assertTrue(wrappers['linears'][(0, 'self_attn', 'q_proj')].active); self.assertEqual(wrappers['merged'], {})
            self.close(target.q_proj(self.x), base_q + self.x @ mx.array(self.A).T @ mx.array(self.B).T, atol=2e-2)

    def test_a_writing_and_a_sound_adapter_apply_together(self):
        # The LoRA (AR) and the Sound LoRA (NAR + io) go on as one combined delta; each keeps its own strength.
        m = self.m
        rng = np.random.default_rng(1)
        writing = {'linears': {(0, 'self_attn', 'q_proj'): (self.A, self.B, 0.5)}, 'io': {}}
        A3 = rng.standard_normal((4, H)).astype(np.float32); B3 = rng.standard_normal((H, 4)).astype(np.float32)
        sound = {'linears': {(1, 'nar_mlp', 'down_proj'): (self.A2, self.B2, 1.0), (0, 'self_attn', 'q_proj'): (A3, B3, 1.0)},
                 'io': {'vae2llm': {'weight': np.full((H, 64), 2.0, np.float32), 'bias': np.zeros(H, np.float32)}}}
        combined = loras.check(loras.combine([(writing, 0.7), (sound, 1.0)]), DIMS)
        with tempfile.TemporaryDirectory() as d:
            weights = Path(d) / 'model.safetensors'
            mx.save_safetensors(str(weights), dict(tree_flatten(m.parameters())))
            self.assertEqual(adapters.apply(m, combined, 1.0, weights=weights, label=(('w', 0.7), ('s', 1.0))), 3)
            q = self.q0 + (self.x @ mx.array(self.A).T @ mx.array(self.B).T) * 0.35 + self.x @ mx.array(A3).T @ mx.array(B3).T
            self.close(m.model.layers[0].self_attn.q_proj(self.x), q)
            self.close(m.model.layers[1].nar_mlp.down_proj(self.x12), self.d0 + self.x12 @ mx.array(self.A2).T @ mx.array(self.B2).T)
            z = mx.random.normal((3, 64)); self.close(m.vae2llm(z), z @ mx.full((H, 64), 2.0).T, atol=1e-3)
            adapters.clear(m)
            self.close(m.model.layers[0].self_attn.q_proj(self.x), self.q0)

    def test_real_audio_measures_the_stock_layer_through_the_wrapper(self):
        import real_audio
        adapters.attach(self.m)
        stock = adapters.innermost(self.m.model.layers[0].nar_self_attn.q_proj)
        self.assertIsInstance(stock, nn.Linear); self.assertEqual(tuple(stock.weight.shape), (H, H))
        self.assertTrue(hasattr(real_audio, 'attach'))


if __name__ == '__main__':
    unittest.main()
