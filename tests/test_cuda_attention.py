"""Run with .cuda-venv to verify the Windows attention memory fallback."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0,str(_Path(__file__).resolve().parents[1]/'app'))
import importlib.util
import sys
import unittest


@unittest.skipUnless(sys.platform == 'win32' and importlib.util.find_spec('yue2'),
                     'Requires the Windows CUDA environment')
class WindowsAttentionTest(unittest.TestCase):
    def test_query_tiles_preserve_grouped_attention_and_causal_positions(self):
        import torch
        import cuda_engine
        from yue2 import nar
        if not torch.cuda.is_available() or not hasattr(cuda_engine, '_original_nar_attention'):
            self.skipTest('Requires the Windows SDPA fallback')
        with torch.inference_mode():
            torch.manual_seed(12)
            q = torch.randn(521, 8, 64, device='cuda', dtype=torch.bfloat16)
            k = torch.randn(521, 2, 64, device='cuda', dtype=torch.bfloat16)
            v = torch.randn_like(k)
            for causal in (False, True):
                with self.subTest(causal=causal):
                    expected = cuda_engine._original_nar_attention(
                        q, k, v, causal=causal, backend='math', query_chunk_size=521)
                    actual = nar.attention(q, k, v, causal=causal, backend='math')
                    torch.testing.assert_close(actual, expected, atol=.016, rtol=.016)

    def test_default_dispatch_uses_cudnn_and_preserves_attention(self):
        import torch
        import cuda_engine
        from yue2 import nar
        if not torch.cuda.is_available() or not hasattr(cuda_engine, '_original_nar_attention'):
            self.skipTest('Requires the Windows SDPA fallback')
        with torch.inference_mode():
            for causal in (False, True):
                q = torch.randn(521, 16, 128, device='cuda', dtype=torch.bfloat16)
                k = torch.randn(521 if causal else 777, 8, 128, device='cuda', dtype=torch.bfloat16)
                v = torch.randn_like(k)
                expected = nar.attention(q, k, v, causal=causal, backend='math')
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
                    actual = nar.attention(q, k, v, causal=causal)
                torch.testing.assert_close(actual, expected, atol=.016, rtol=.016)
                self.assertTrue(any('_scaled_dot_product_cudnn_attention' in event.key
                                    for event in profile.key_averages()))
