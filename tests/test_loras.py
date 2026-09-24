"""Community LoRAs: reading the three file layouts, un-fusing ComfyUI tensors, the catalogue and the API around it."""
import json, struct, sys, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from fastapi.testclient import TestClient
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import loras
from studio_fixture import studio_root, load_studio

# A miniature YuE2: 2 layers, hidden 8, 2 query heads x 2 kv heads of size 2, mlp 12.
DIMS = {'hidden': 8, 'q': 8, 'kv': 4, 'intermediate': 12, 'layers': 2}
R = 3
OUTS = {'q_proj': 8, 'k_proj': 4, 'v_proj': 4, 'o_proj': 8, 'gate_proj': 12, 'up_proj': 12, 'down_proj': 8}
INS = {'q_proj': 8, 'k_proj': 8, 'v_proj': 8, 'o_proj': 8, 'gate_proj': 8, 'up_proj': 8, 'down_proj': 12}


def write_safetensors(path, tensors, metadata=None):
    header, blobs, offset = {}, [], 0
    for name, array in tensors.items():
        array = np.ascontiguousarray(array)
        if array.dtype == np.float32: dtype, raw = 'F32', array.tobytes()
        elif array.dtype == np.float16: dtype, raw = 'F16', array.tobytes()
        else: raise TypeError(array.dtype)
        header[name] = {'dtype': dtype, 'shape': list(array.shape), 'data_offsets': [offset, offset + len(raw)]}
        blobs.append(raw); offset += len(raw)
    if metadata: header['__metadata__'] = metadata
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack('<Q', len(encoded)) + encoded + b''.join(blobs))


def bf16_bytes(array):
    """Round-to-nearest-even bf16 encoding of a float32 array, as safetensors stores it."""
    bits = np.ascontiguousarray(array, dtype=np.float32).view(np.uint32)
    rounded = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)
    return rounded.tobytes()


def pairs(rng, modules):
    """Per-projection (A, B) for every layer and module: the ground truth both layouts are checked against."""
    out = {}
    for layer in range(DIMS['layers']):
        for module in modules:
            for proj in (loras.ATTENTION if 'attn' in module else loras.MLP):
                out[(layer, module, proj)] = (rng.standard_normal((R, INS[proj])).astype(np.float32),
                                              rng.standard_normal((OUTS[proj], R)).astype(np.float32))
    return out


def hf_tensors(truth, prefix='', suffix=''):
    tensors = {}
    for (layer, module, proj), (A, B) in truth.items():
        tensors[f'{prefix}layers.{layer}.{module}.{proj}.lora_A{suffix}'] = A
        tensors[f'{prefix}layers.{layer}.{module}.{proj}.lora_B{suffix}'] = B
    return tensors


def native_pairs(rng, modules):
    """Per-projection pairs that share one A across fused q/k/v and gate/up, as native YuE2 LoRAs do."""
    out = {}
    for layer in range(DIMS['layers']):
        for module in modules:
            if 'attn' in module:
                shared = rng.standard_normal((R, INS['q_proj'])).astype(np.float32)
                for proj in ('q_proj', 'k_proj', 'v_proj'):
                    out[(layer, module, proj)] = (shared, rng.standard_normal((OUTS[proj], R)).astype(np.float32))
                out[(layer, module, 'o_proj')] = (rng.standard_normal((R, INS['o_proj'])).astype(np.float32),
                                                  rng.standard_normal((OUTS['o_proj'], R)).astype(np.float32))
            else:
                shared = rng.standard_normal((R, INS['gate_proj'])).astype(np.float32)
                for proj in ('gate_proj', 'up_proj'):
                    out[(layer, module, proj)] = (shared, rng.standard_normal((OUTS[proj], R)).astype(np.float32))
                out[(layer, module, 'down_proj')] = (rng.standard_normal((R, INS['down_proj'])).astype(np.float32),
                                                     rng.standard_normal((OUTS['down_proj'], R)).astype(np.float32))
    return out


def native_tensors(truth, alpha=None, comfy_names=False):
    """Sound & Vision native export: AR under text_encoders, NAR under diffusion_model, fused qkv/gate_up."""
    tensors = {}
    groups = {}
    a_name, b_name = ('.lora_down.weight', '.lora_up.weight') if comfy_names else ('.lora_A.weight', '.lora_B.weight')
    for (layer, module, proj), pair in truth.items():
        if module in loras.AR_MODULES:
            prefix, short = 'text_encoders.model.', module
        else:
            prefix, short = 'diffusion_model.model.', ('self_attn' if 'attn' in module else 'mlp')
        groups.setdefault((layer, prefix, short), {})[proj] = pair
    for (layer, prefix, short), parts in groups.items():
        head = f'{prefix}layers.{layer}.{short}.'

        def put(proj, A, B, a_name=a_name, b_name=b_name):
            tensors[head + proj + a_name] = A
            tensors[head + proj + b_name] = B
            if alpha is not None:
                tensors[head + proj + '.alpha'] = np.array([alpha], np.float32)

        if 'attn' in short:
            put('qkv_proj', parts['q_proj'][0], np.concatenate([parts[p][1] for p in ('q_proj', 'k_proj', 'v_proj')]))
            put('o_proj', *parts['o_proj'])
        else:
            put('gate_up_proj', parts['gate_proj'][0], np.concatenate([parts[p][1] for p in ('gate_proj', 'up_proj')]))
            put('down_proj', *parts['down_proj'])
    return tensors


def comfy_tensors(truth):
    """ComfyUI's fused layout: down stacks the A rows, up places the B blocks on the diagonal."""
    tensors = {}
    for layer in range(DIMS['layers']):
        for module in {m for _, m, _ in truth}:
            head = f'text_encoders.model.layers.{layer}.{module}.'
            if 'attn' in module:
                A = [truth[(layer, module, p)][0] for p in ('q_proj', 'k_proj', 'v_proj')]
                B = [truth[(layer, module, p)][1] for p in ('q_proj', 'k_proj', 'v_proj')]
                tensors[head + 'qkv_proj.lora_down.weight'] = np.concatenate(A)
                up = np.zeros((sum(b.shape[0] for b in B), 3 * R), np.float32)
                row = 0
                for i, b in enumerate(B):
                    up[row:row + b.shape[0], i * R:(i + 1) * R] = b; row += b.shape[0]
                tensors[head + 'qkv_proj.lora_up.weight'] = up
                tensors[head + 'o_proj.lora_down.weight'], tensors[head + 'o_proj.lora_up.weight'] = truth[(layer, module, 'o_proj')]
            else:
                A = [truth[(layer, module, p)][0] for p in ('gate_proj', 'up_proj')]
                B = [truth[(layer, module, p)][1] for p in ('gate_proj', 'up_proj')]
                tensors[head + 'gate_up_proj.lora_down.weight'] = np.concatenate(A)
                up = np.zeros((2 * DIMS['intermediate'], 2 * R), np.float32)
                up[:DIMS['intermediate'], :R] = B[0]; up[DIMS['intermediate']:, R:] = B[1]
                tensors[head + 'gate_up_proj.lora_up.weight'] = up
                tensors[head + 'down_proj.lora_down.weight'], tensors[head + 'down_proj.lora_up.weight'] = truth[(layer, module, 'down_proj')]
    return tensors


class LayoutTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup); self.root = Path(self.tmp.name)
        (self.root / 'loras').mkdir()
        self.rng = np.random.default_rng(7)

    def assert_same_deltas(self, delta, truth, scale=1.0):
        self.assertEqual(set(delta['linears']), set(truth))
        for key, (A, B) in truth.items():
            got_A, got_B, got_scale = delta['linears'][key]
            np.testing.assert_allclose(got_B @ got_A, B @ A, rtol=1e-5, atol=1e-5, err_msg=str(key))
            self.assertAlmostEqual(got_scale, scale, msg=str(key))

    def test_hf_layout_reads_straight_through(self):
        truth = pairs(self.rng, loras.AR_MODULES)
        path = self.root / 'loras' / 'instrumental_v1.safetensors'
        write_safetensors(path, hf_tensors(truth), {'notes': 'trained on 40 instrumentals', 'format': 'pt'})
        info = loras.describe(path)
        self.assertEqual((info['layout'], info['branches'], info['rank'], info['layers'], info['fused']), ('hf', ['ar'], R, 2, False))
        self.assertEqual(info['name'], 'instrumental v1'); self.assertEqual(info['metadata'], {'notes': 'trained on 40 instrumentals'})
        delta = loras.check(loras.load(self.root, path.name, DIMS), DIMS)
        self.assert_same_deltas(delta, truth); self.assertEqual(delta['io'], {})

    def test_peft_layout_with_alpha_scales_by_alpha_over_rank(self):
        truth = pairs(self.rng, ('nar_self_attn',))
        tensors = hf_tensors(truth, 'base_model.model.model.', '.weight')
        for key in truth: tensors[f'base_model.model.model.layers.{key[0]}.{key[1]}.{key[2]}.alpha'] = np.array(6.0, np.float32)
        path = self.root / 'loras' / 'warm-nar.safetensors'; write_safetensors(path, tensors)
        info = loras.describe(path)
        self.assertEqual((info['layout'], info['branches']), ('peft', ['nar']))
        self.assert_same_deltas(loras.deltas(loras.read_tensors(path), DIMS), truth, scale=6.0 / R)

    def test_comfyui_fused_tensors_are_split_back_into_projections(self):
        truth = pairs(self.rng, loras.AR_MODULES + loras.NAR_MODULES)
        tensors = comfy_tensors(truth)
        path = self.root / 'loras' / 'comfy pack.safetensors'; write_safetensors(path, tensors)
        info = loras.describe(path)
        self.assertEqual((info['layout'], info['branches'], info['rank'], info['fused']), ('comfyui', ['ar', 'nar'], R, True))
        delta = loras.check(loras.deltas(loras.read_tensors(path), DIMS), DIMS)
        self.assert_same_deltas(delta, truth)
        # The fused pair reproduces the stacked deltas exactly: up @ down == [Bq Aq; Bk Ak; Bv Av].
        down, up = tensors['text_encoders.model.layers.0.self_attn.qkv_proj.lora_down.weight'], tensors['text_encoders.model.layers.0.self_attn.qkv_proj.lora_up.weight']
        stacked = np.concatenate([truth[(0, 'self_attn', p)][1] @ truth[(0, 'self_attn', p)][0] for p in ('q_proj', 'k_proj', 'v_proj')])
        np.testing.assert_allclose(up @ down, stacked, rtol=1e-5, atol=1e-5)

    def test_sound_vision_native_export_maps_nar_and_unfuses_shared_a(self):
        truth = native_pairs(self.rng, loras.AR_MODULES + loras.NAR_MODULES)
        path = self.root / 'loras' / 'native-joint.safetensors'
        write_safetensors(path, native_tensors(truth, alpha=float(R)), {
            'format': 'sound-vision-yue2-native-export-v1',
            'training_info': '{"step": 1000}',
            'ss_tag_frequency': '{"training": {"sv_example": 1}}',
            'training_recipe': 'native_joint_v1',
        })
        info = loras.describe(path)
        self.assertEqual((info['layout'], info['branches'], info['rank'], info['fused']),
                         ('sound-vision', ['ar', 'nar'], R, True))
        self.assertEqual((info['steps'], info['trigger'], info['name']), (1000, 'sv_example', 'native-joint'))
        self.assertEqual(info['metadata'].get('training_recipe'), 'native_joint_v1')
        delta = loras.check(loras.deltas(loras.read_tensors(path), DIMS), DIMS)
        self.assert_same_deltas(delta, truth, scale=1.0)
        # Shared A and stacked B reproduce the fused NAR qkv delta.
        A = native_tensors(truth)['diffusion_model.model.layers.0.self_attn.qkv_proj.lora_A.weight']
        B = native_tensors(truth)['diffusion_model.model.layers.0.self_attn.qkv_proj.lora_B.weight']
        stacked = np.concatenate([truth[(0, 'nar_self_attn', p)][1] @ truth[(0, 'nar_self_attn', p)][0]
                                  for p in ('q_proj', 'k_proj', 'v_proj')])
        np.testing.assert_allclose(B @ A, stacked, rtol=1e-5, atol=1e-5)

    def test_comfy_named_native_fused_artist_lora_unfuses_by_shape(self):
        # becausereasons CNZN / MLTNT packs: native one-A / stacked-B, ComfyUI lora_down / lora_up names.
        truth = native_pairs(self.rng, loras.AR_MODULES + loras.NAR_MODULES)
        path = self.root / 'loras' / 'CNZN Sanremo.safetensors'
        write_safetensors(path, native_tensors(truth, comfy_names=True), {'format': 'pt', 'fs_audio': 'artist LoRA'})
        info = loras.describe(path)
        self.assertEqual((info['layout'], info['branches'], info['rank'], info['fused']),
                         ('sound-vision', ['ar', 'nar'], R, True))
        delta = loras.check(loras.deltas(loras.read_tensors(path), DIMS), DIMS)
        self.assert_same_deltas(delta, truth)
        down = native_tensors(truth, comfy_names=True)['text_encoders.model.layers.0.self_attn.qkv_proj.lora_down.weight']
        up = native_tensors(truth, comfy_names=True)['text_encoders.model.layers.0.self_attn.qkv_proj.lora_up.weight']
        stacked = np.concatenate([truth[(0, 'self_attn', p)][1] @ truth[(0, 'self_attn', p)][0]
                                  for p in ('q_proj', 'k_proj', 'v_proj')])
        np.testing.assert_allclose(up @ down, stacked, rtol=1e-5, atol=1e-5)

    def test_bf16_tensors_and_io_replacements_are_read(self):
        A = self.rng.standard_normal((R, 12)).astype(np.float32); B = self.rng.standard_normal((8, R)).astype(np.float32)
        io_w = self.rng.standard_normal((8, 64)).astype(np.float32); io_b = np.zeros(8, np.float32)
        a, b, w = R * 12 * 2, 8 * R * 4, 8 * 64 * 4
        header = {'layers.0.nar_mlp.down_proj.lora_A': {'dtype': 'BF16', 'shape': [R, 12], 'data_offsets': [0, a]},
                  'layers.0.nar_mlp.down_proj.lora_B': {'dtype': 'F32', 'shape': [8, R], 'data_offsets': [a, a + b]},
                  'vae2llm.weight': {'dtype': 'F32', 'shape': [8, 64], 'data_offsets': [a + b, a + b + w]},
                  'vae2llm.bias': {'dtype': 'F32', 'shape': [8], 'data_offsets': [a + b + w, a + b + w + 32]}}
        encoded = json.dumps(header).encode()
        path = self.root / 'loras' / 'mixed.safetensors'
        path.write_bytes(struct.pack('<Q', len(encoded)) + encoded + bf16_bytes(A) + B.tobytes() + io_w.tobytes() + io_b.tobytes())
        info = loras.describe(path); self.assertEqual((info['branches'], info['io']), (['nar'], ['vae2llm']))
        delta = loras.check(loras.deltas(loras.read_tensors(path), DIMS), DIMS)
        got_A, got_B, _ = delta['linears'][(0, 'nar_mlp', 'down_proj')]
        np.testing.assert_allclose(got_A, A, rtol=1e-2, atol=1e-2)   # bf16 keeps 8 bits of mantissa
        np.testing.assert_array_equal(got_B, B)
        np.testing.assert_array_equal(delta['io']['vae2llm']['weight'], io_w)

    def test_files_that_do_not_fit_the_model_are_refused(self):
        truth = pairs(self.rng, ('mlp',))
        tensors = hf_tensors(truth)
        tensors['layers.1.mlp.up_proj.lora_B'] = np.zeros((13, R), np.float32)   # one row too many
        with self.assertRaisesRegex(ValueError, 'layers.1.mlp.up_proj'):
            loras.check(loras.deltas(tensors, DIMS), DIMS)
        half = {k: v for k, v in hf_tensors(truth).items() if not k.endswith('lora_B') or 'layers.0' in k}
        with self.assertRaisesRegex(ValueError, 'half of its pair'):
            loras.deltas(half, DIMS)
        with self.assertRaisesRegex(ValueError, 'do not match projections'):
            loras.deltas({'layers.0.self_attn.qkv_proj.lora_down.weight': np.zeros((3 * R, 8), np.float32),
                          'layers.0.self_attn.qkv_proj.lora_up.weight': np.zeros((15, 3 * R), np.float32)}, DIMS)
        with self.assertRaisesRegex(ValueError, 'No YuE2 LoRA tensors'):
            write_safetensors(self.root / 'loras' / 'sd.safetensors', {'unet.down.weight': np.zeros((2, 2), np.float32)})
            loras.describe(self.root / 'loras' / 'sd.safetensors')

    def test_catalogue_lists_every_file_and_flags_strangers(self):
        self.assertEqual(loras.catalog(self.root / 'nowhere'), [])
        write_safetensors(self.root / 'loras' / 'b_lora.safetensors', hf_tensors(pairs(self.rng, ('mlp',))))
        write_safetensors(self.root / 'loras' / 'A stranger.safetensors', {'unet.down.weight': np.zeros((2, 2), np.float32)})
        (self.root / 'loras' / 'notes.txt').write_text('ignored')
        (self.root / 'loras' / 'broken.safetensors').write_bytes(b'\xff' * 20)
        listing = loras.catalog(self.root)
        self.assertEqual([l['id'] for l in listing], ['A stranger.safetensors', 'b_lora.safetensors', 'broken.safetensors'])
        self.assertIn('error', listing[0]); self.assertNotIn('error', listing[1]); self.assertIn('error', listing[2])
        self.assertEqual(listing[1]['name'], 'b lora')

    def test_catalogue_overlays_download_notes_and_lists_those_files_first(self):
        write_safetensors(self.root / 'loras' / 'QWWL Mehfil.safetensors', hf_tensors(pairs(self.rng, ('mlp',))))
        write_safetensors(self.root / 'loras' / 'zebra.safetensors', hf_tensors(pairs(self.rng, ('mlp',))))
        assets = [{'path': 'loras/QWWL Mehfil.safetensors', 'name': 'QWWL Mehfil', 'family': 'QWWL / DRKSF - Qawwali',
                   'trigger': 'qwwl', 'prompt': 'qwwl, Urdu, dark traditional Sufi qawwali',
                   'settings': {'lora_strength': 1.0, 'cot': 'off', 'cfg_scale': 1.0, 'duration': 170}}]
        listing = loras.catalog(self.root, assets)
        self.assertEqual([l['id'] for l in listing], ['QWWL Mehfil.safetensors', 'zebra.safetensors'])
        mehfil = listing[0]
        self.assertEqual((mehfil['name'], mehfil['trigger'], mehfil['family']),
                         ('QWWL Mehfil', 'qwwl', 'QWWL / DRKSF - Qawwali'))
        self.assertEqual(mehfil['settings']['cot'], 'off')
        self.assertEqual(mehfil['prompt'], 'qwwl, Urdu, dark traditional Sufi qawwali')

    def test_file_names_stay_inside_the_folder(self):
        for bad in ('', '../model.safetensors', 'x/y.safetensors', 'model.pt', '.hidden.safetensors', 'a\\b.safetensors'):
            self.assertFalse(loras.valid_name(bad), bad)
        self.assertTrue(loras.valid_name('YuE2 instrumental (v2).safetensors'))
        with self.assertRaises(ValueError): loras.path_of(self.root, '../x.safetensors')
        with self.assertRaises(FileNotFoundError): loras.load(self.root, 'missing.safetensors', DIMS)

    def test_dims_come_from_the_model_configuration(self):
        config = json.loads((ROOT / 'model' / 'bf16' / 'config.json').read_text()) if (ROOT / 'model' / 'bf16' / 'config.json').is_file() else \
            {'hidden_size': 2048, 'num_attention_heads': 16, 'num_key_value_heads': 8, 'head_dim': 128, 'intermediate_size': 6144, 'num_hidden_layers': 28}
        self.assertEqual(loras.dims_from_config(config), loras.DIMS)
        self.assertEqual(loras.expected_shape((0, 'self_attn', 'k_proj'), loras.DIMS), (1024, 2048))
        self.assertEqual(loras.expected_shape((0, 'mlp', 'down_proj'), loras.DIMS), (2048, 6144))


class CombineTest(unittest.TestCase):
    """The LoRA and the Sound LoRA are folded in together as one delta."""

    def test_two_adapters_sum_on_shared_projections_and_keep_their_own_strengths(self):
        rng = np.random.default_rng(3)
        writing = {'linears': dict(pairs(rng, ('self_attn',)).items()), 'io': {}}
        writing['linears'] = {k: (A, B, 0.5) for k, (A, B) in writing['linears'].items()}
        sound = {'linears': {k: (A, B, 1.0) for k, (A, B) in pairs(rng, ('nar_self_attn',)).items()}, 'io': {'vae2llm': {'weight': np.ones((8, 64), np.float32)}}}
        shared_key = (0, 'self_attn', 'q_proj')
        sound['linears'][shared_key] = (rng.standard_normal((R, 8)).astype(np.float32), rng.standard_normal((8, R)).astype(np.float32), 2.0)
        combined = loras.combine([(writing, 0.7), (sound, 1.0)])
        self.assertEqual(set(combined['linears']), set(writing['linears']) | set(sound['linears']))
        self.assertEqual(list(combined['io']), ['vae2llm'])
        for key, (A, B, scale) in combined['linears'].items():
            self.assertEqual(scale, 1.0)
            expected = np.zeros((OUTS[key[2]], INS[key[2]]), np.float32)
            for delta, strength in ((writing, 0.7), (sound, 1.0)):
                if key in delta['linears']:
                    a, b, s = delta['linears'][key]
                    expected += (b @ a) * (s * strength)
            np.testing.assert_allclose(B @ A, expected, rtol=1e-5, atol=1e-5)
        A, B, _ = combined['linears'][shared_key]
        self.assertEqual((A.shape[0], B.shape[1]), (2 * R, 2 * R))

    def test_zero_strength_drops_an_adapter_and_two_decoder_replacements_are_refused(self):
        rng = np.random.default_rng(4)
        one = {'linears': {k: (A, B, 1.0) for k, (A, B) in pairs(rng, ('mlp',)).items()}, 'io': {'llm2vae': {'weight': np.ones((64, 8), np.float32)}}}
        two = {'linears': {}, 'io': {'llm2vae': {'weight': np.zeros((64, 8), np.float32)}}}
        self.assertEqual(loras.combine([(one, 0), (two, 1.0)])['linears'], {})
        with self.assertRaises(ValueError):
            loras.combine([(one, 1.0), (two, 1.0)])

    def test_the_request_names_the_slots(self):
        self.assertEqual(loras.chosen({}), [])
        self.assertEqual(loras.chosen({'lora': 'a.safetensors'}), [('a.safetensors', 1.0)])
        self.assertEqual(loras.chosen({'lora': 'a.safetensors', 'lora_strength': 0, 'sound_lora': 'b.safetensors', 'sound_lora_strength': 0.8}),
                         [('b.safetensors', 0.8)])
        self.assertEqual(loras.chosen({'lora': 'a.safetensors', 'lora_strength': 0.7, 'sound_lora': 'b.safetensors'}),
                         [('a.safetensors', 0.7), ('b.safetensors', 1.0)])

    def test_drop_ar_keeps_decoder_adapters_for_covers(self):
        rng = np.random.default_rng(5)
        writing = {'linears': {**{k: (A, B, 1.0) for k, (A, B) in pairs(rng, loras.AR_MODULES).items()},
                               **{k: (A, B, 1.0) for k, (A, B) in pairs(rng, loras.NAR_MODULES).items()}},
                   'io': {'vae2llm': {'weight': np.ones((8, 64), np.float32)}}}
        out = loras.drop_ar(writing)
        self.assertTrue(out['linears'])
        self.assertTrue(all(k[1] in loras.NAR_MODULES for k in out['linears']))
        self.assertEqual(len(out['linears']), len(writing['linears']) // 2)
        self.assertEqual(list(out['io']), ['vae2llm'])
        self.assertEqual(loras.drop_ar({'linears': {k: (A, B, 1.0) for k, (A, B) in pairs(rng, ('mlp',)).items()},
                                        'io': {}})['linears'], {})


class TorchMergeTest(unittest.TestCase):
    def test_merge_adds_scaled_deltas_and_replaces_io(self):
        try: import torch
        except ImportError: self.skipTest('torch is not installed in this environment')
        torch.manual_seed(0)

        class Linear(torch.nn.Module):
            def __init__(self, out, inner, bias=False):
                super().__init__(); self.weight = torch.nn.Parameter(torch.randn(out, inner)); self.bias = torch.nn.Parameter(torch.zeros(out)) if bias else None
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                layer = torch.nn.Module(); layer.mlp = torch.nn.Module(); layer.mlp.down_proj = Linear(8, 12)
                layer.self_attn = torch.nn.Module(); layer.self_attn.q_proj = Linear(8, 8)
                self.model = torch.nn.Module(); self.model.layers = torch.nn.ModuleList([layer])
                self.vae2llm = Linear(8, 64, bias=True)
        model = Model(); before = model.model.layers[0].mlp.down_proj.weight.detach().clone()
        A = np.ones((R, 12), np.float32); B = np.ones((8, R), np.float32) * 0.5
        io_w = np.full((8, 64), 2.0, np.float32)
        delta = {'linears': {(0, 'mlp', 'down_proj'): (A, B, 0.5)}, 'io': {'vae2llm': {'weight': io_w, 'bias': np.ones(8, np.float32)}}}
        self.assertEqual(loras.merge_torch(model, delta, 2.0), 2)
        expected = before + torch.from_numpy(B @ A) * 1.0   # strength 2 x scale .5
        torch.testing.assert_close(model.model.layers[0].mlp.down_proj.weight.detach(), expected)
        torch.testing.assert_close(model.vae2llm.weight.detach(), torch.full((8, 64), 2.0))
        torch.testing.assert_close(model.vae2llm.bias.detach(), torch.ones(8))
        self.assertEqual(loras.merge_torch(model, delta, 0), 0)


class LoraAPITest(unittest.TestCase):
    def setUp(self):
        self.root = studio_root(self); self.s = load_studio(self, self.root, 'loras_test_server')
        self.client = TestClient(self.s.app); self.headers = {'X-Studio-Token': self.s.TOKEN}
        (self.root / 'loras').mkdir()
        write_safetensors(self.root / 'loras' / 'strings.safetensors', hf_tensors(pairs(np.random.default_rng(1), ('mlp',))), {'notes': 'lush'})

    def test_listing_comes_from_the_folder_beside_the_models(self):
        r = self.client.get('/api/loras'); self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual([l['id'] for l in r.json()['loras']], ['strings.safetensors']); self.assertFalse(r.json()['remote'])
        self.assertEqual([l['id'] for l in self.client.get('/api/config').json()['loras']], ['strings.safetensors'])

    def test_jobs_name_a_file_in_the_folder(self):
        base = {'kind': 'plan', 'style': 'pop', 'lyrics': 'la', 'model': 'bf16'}
        with patch.object(self.s, 'launch'), patch.object(self.s, 'model_ready', return_value=True):
            r = self.client.post('/api/jobs', json={**base, 'lora': 'missing.safetensors'}, headers=self.headers)
            self.assertEqual(r.status_code, 422, r.text); self.assertIn('not in the loras folder', r.text)
            r = self.client.post('/api/jobs', json={**base, 'lora': '../strings.safetensors'}, headers=self.headers)
            self.assertEqual(r.status_code, 422, r.text)
            r = self.client.post('/api/jobs', json={**base, 'lora': 'strings.safetensors', 'lora_strength': 2.5}, headers=self.headers)
            self.assertEqual(r.status_code, 422, r.text)
            r = self.client.post('/api/jobs', json={**base, 'lora': 'strings.safetensors', 'lora_strength': 0.8}, headers=self.headers)
            self.assertEqual(r.status_code, 200, r.text)
            saved = json.loads((self.s.LIB / r.json()['id'] / 'request.json').read_text())
            self.assertEqual((saved['lora'], saved['lora_strength']), ('strings.safetensors', 0.8))
            self.assertEqual((saved['sound_lora'], saved['sound_lora_strength']), ('', 1.0))

    def test_the_sound_slot_takes_a_second_file(self):
        write_safetensors(self.root / 'loras' / 'room.safetensors', hf_tensors(pairs(np.random.default_rng(2), ('nar_mlp',))))
        base = {'kind': 'plan', 'style': 'pop', 'lyrics': 'la', 'model': 'bf16'}
        with patch.object(self.s, 'launch'), patch.object(self.s, 'model_ready', return_value=True):
            r = self.client.post('/api/jobs', json={**base, 'sound_lora': 'missing.safetensors'}, headers=self.headers)
            self.assertEqual(r.status_code, 422, r.text); self.assertIn('not in the loras folder', r.text)
            r = self.client.post('/api/jobs', json={**base, 'lora': 'strings.safetensors', 'sound_lora': 'strings.safetensors'}, headers=self.headers)
            self.assertEqual(r.status_code, 422, r.text); self.assertIn('same file', r.text)
            r = self.client.post('/api/jobs', json={**base, 'lora': 'strings.safetensors', 'lora_strength': 0.7, 'sound_lora': 'room.safetensors'}, headers=self.headers)
            self.assertEqual(r.status_code, 200, r.text)
            saved = json.loads((self.s.LIB / r.json()['id'] / 'request.json').read_text())
            self.assertEqual((saved['lora'], saved['lora_strength'], saved['sound_lora'], saved['sound_lora_strength']),
                             ('strings.safetensors', 0.7, 'room.safetensors', 1.0))
            self.assertEqual(loras.chosen(saved), [('strings.safetensors', 0.7), ('room.safetensors', 1.0)])

    def test_a_rerender_may_swap_the_sound_lora_over_the_same_tokens(self):
        from test_song_continuation import studio_server
        p = studio_server(self, 'loras_rerender_server')
        (self.root / 'loras').mkdir(exist_ok=True)
        write_safetensors(self.root / 'loras' / 'room.safetensors', hf_tensors(pairs(np.random.default_rng(2), ('nar_mlp',))))
        write_safetensors(self.root / 'loras' / 'strings.safetensors', hf_tensors(pairs(np.random.default_rng(1), ('mlp',))))
        source = {**self.request, 'lora': 'strings.safetensors', 'lora_strength': 0.7}
        (p / 'request.json').write_text(json.dumps(source))
        with patch.object(self.s, 'launch'), patch.object(self.s, 'model_ready', return_value=True):
            for sound, strength in (('', 1.0), ('room.safetensors', 0.8)):
                r = self.client.post('/api/jobs', json={**source, 'render_source': self.job, 'render_candidate': 1, 'steps': 8,
                                                        'sound_lora': sound, 'sound_lora_strength': strength}, headers=self.headers)
                self.assertEqual(r.status_code, 200, r.text)
                saved = json.loads((self.s.LIB / r.json()['id'] / 'request.json').read_text())
                self.assertEqual((saved['seed'], saved['random_seed'], saved['lora'], saved['lora_strength']), (7, False, 'strings.safetensors', 0.7),
                                 'the writing adapter, seed and tokens come from the source song')
                self.assertEqual((saved['sound_lora'], saved['sound_lora_strength']), (sound, strength))
                self.assertTrue(saved['title'].endswith(' · 8 steps' + (' · room' if sound else '')), saved['title'])
            r = self.client.post('/api/jobs', json={**source, 'render_source': self.job, 'render_candidate': 1, 'steps': 8, 'sound_lora': 'strings.safetensors'}, headers=self.headers)
            self.assertEqual(r.status_code, 422, r.text); self.assertIn('same file', r.text)
            r = self.client.post('/api/jobs', json={**self.request, 'render_source': self.job, 'render_candidate': 1, 'steps': 8, 'sound_lora': 'strings.safetensors'}, headers=self.headers)
            self.assertEqual(r.status_code, 400, r.text); self.assertIn('already uses', r.text, 'the source song\'s own LoRA is checked too')
            r = self.client.post('/api/jobs', json={**source, 'render_source': self.job, 'render_candidate': 1, 'steps': 8, 'render_without_lora': True}, headers=self.headers)
            self.assertEqual(r.status_code, 200, r.text)
            saved = json.loads((self.s.LIB / r.json()['id'] / 'request.json').read_text())
            self.assertEqual((saved['lora'], saved['seed'], saved['render_without_lora']), ('', 7, True), 'the tokens stay, the adapter is left out of the render')
            self.assertTrue(saved['title'].endswith(' · 8 steps · without strings'), saved['title'])

    def test_a_render_node_decides_what_it_offers(self):
        with patch.object(self.s.remote, 'configured', return_value=True), \
             patch.object(self.s.remote, 'snapshot', return_value={'models': [{'id': 'cuda-bf16', 'ready': True, 'label': 'x', 'backend': 'cuda'}],
                                                                    'loras': [{'id': 'node.safetensors', 'name': 'node', 'branches': ['ar']}]}):
            self.assertEqual([l['id'] for l in self.client.get('/api/loras').json()['loras']], ['node.safetensors'])
            self.assertTrue(self.s.lora_known('node.safetensors')); self.assertFalse(self.s.lora_known('strings.safetensors'))


if __name__ == '__main__':
    unittest.main()
