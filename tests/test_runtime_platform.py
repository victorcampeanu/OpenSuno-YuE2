import os
from pathlib import Path
from types import SimpleNamespace
import unittest
import subprocess
import sys
from unittest.mock import patch

from runtime_platform import environment_python, stop_process, replace_file


class PlatformTest(unittest.TestCase):
    def test_windows_retries_transient_sharing_violation(self):
        from unittest.mock import Mock
        source = Mock()
        source.replace.side_effect = [PermissionError('file is being read'), Path('done')]
        with patch('runtime_platform.os', SimpleNamespace(name='nt')), patch('runtime_platform.time.sleep') as sleep:
            self.assertEqual(replace_file(source, Path('done')), Path('done'))
            self.assertEqual(source.replace.call_count, 2)
            sleep.assert_called_once_with(.01)

    def test_replace_errors_are_not_hidden(self):
        from unittest.mock import Mock
        for platform, calls in [('nt', 50), ('posix', 1)]:
            source = Mock()
            source.replace.side_effect = PermissionError('permanently denied')
            with patch('runtime_platform.os', SimpleNamespace(name=platform)), patch('runtime_platform.time.sleep'):
                with self.assertRaises(PermissionError):
                    replace_file(source, Path('done'))
            self.assertEqual(source.replace.call_count, calls)

    @unittest.skipUnless(os.name == 'nt', 'Windows parent handle lifetime')
    def test_worker_exits_when_explicit_server_parent_exits(self):
        parent = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
        watcher = subprocess.Popen([sys.executable, '-c',
            'from runtime_platform import watch_parent; watch_parent()'],
            cwd=Path(__file__).resolve().parents[1],
            env={**os.environ, 'YUE2_PARENT_PID': str(parent.pid)})
        try:
            parent.terminate()
            parent.wait(timeout=5)
            self.assertIn(watcher.wait(timeout=5), (0, 1))
        finally:
            for proc in (parent, watcher):
                if proc.poll() is None:
                    proc.kill()
                proc.wait(timeout=5)

    def test_environment_paths_for_both_platforms(self):
        root = Path.cwd()
        for name, suffix in [('nt', 'Scripts/python.exe'), ('posix', 'bin/python')]:
            with self.subTest(platform=name), patch('runtime_platform.os', SimpleNamespace(name=name)):
                self.assertEqual(environment_python(root), root/'.venv'/suffix)
                self.assertEqual(environment_python(root, True), root/'.transcribe-venv'/suffix)

    def test_posix_stop_preserves_process_group_signals(self):
        import signal
        from unittest.mock import Mock
        proc = Mock(pid=123)
        proc.poll.return_value = None
        fake_os = Mock(name='os')
        fake_os.name = 'posix'
        signals = SimpleNamespace(SIGTERM=15, SIGKILL=9)
        with patch('runtime_platform.os', fake_os), patch('runtime_platform.signal', signals):
            stop_process(proc)
            fake_os.killpg.assert_called_with(123, signals.SIGTERM)
            stop_process(proc, force=True)
            fake_os.killpg.assert_called_with(123, signals.SIGKILL)
