from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from model.weight_io import load_weights


class WeightReaderTest(unittest.TestCase):
    def test_large_windows_file_is_evaluated_before_closing(self):
        core = ModuleType('mlx.core')
        core.load = Mock(return_value={'weight': object()})
        core.eval = Mock(side_effect=lambda weights: self.assertFalse(core.load.call_args.args[0].closed))
        package = ModuleType('mlx')
        package.core = core
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'weights.safetensors'
            path.write_bytes(b'fixture')
            with patch.dict('sys.modules', {'mlx': package, 'mlx.core': core}), \
                 patch('model.weight_io.os', SimpleNamespace(name='nt')), \
                 patch.object(Path, 'stat', return_value=SimpleNamespace(st_size=2**31)):
                self.assertIs(load_weights(path), core.load.return_value)
            self.assertTrue(core.load.call_args.args[0].closed)
            self.assertEqual(core.load.call_args.kwargs, {'format': 'safetensors'})
            core.eval.assert_called_once_with(core.load.return_value)

    def test_mac_and_small_windows_files_keep_native_loader(self):
        core = ModuleType('mlx.core')
        core.load = Mock()
        core.eval = Mock()
        package = ModuleType('mlx')
        package.core = core
        path = Path('weights.safetensors')
        for platform, size in [('posix', 8 * 1024**3), ('nt', 100)]:
            with self.subTest(platform=platform), \
                 patch.dict('sys.modules', {'mlx': package, 'mlx.core': core}), \
                 patch('model.weight_io.os', SimpleNamespace(name=platform)), \
                 patch.object(Path, 'stat', return_value=SimpleNamespace(st_size=size)):
                load_weights(path)
                core.load.assert_called_with(str(path))
                core.eval.assert_not_called()
