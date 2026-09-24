import asyncio
import hashlib
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import requests
from fastapi.testclient import TestClient
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model_downloads import ModelDownloads
from studio_fixture import studio_root, load_studio
PAYLOAD = b'fixture-model-data' * 1000

class Handler(BaseHTTPRequestHandler):
    ignore_range = False
    ranges = []
    def do_GET(self):
        value = self.headers.get('Range')
        type(self).ranges.append(value)
        offset = int(value.split('=')[1].split('-')[0]) if value and not self.ignore_range else 0
        end = int(value.split('-')[1]) if value and value.split('-')[1] and not self.ignore_range else len(PAYLOAD)-1
        self.send_response(206 if value and not self.ignore_range else 200)
        if value and not self.ignore_range:
            self.send_header('Content-Range', f'bytes {offset}-{end}/{len(PAYLOAD)}')
        self.send_header('Content-Length', str(end-offset+1))
        self.end_headers()
        self.wfile.write(PAYLOAD[offset:end+1])
    def log_message(self, *args):
        pass

class DownloadsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.http = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        cls.thread = threading.Thread(target=cls.http.serve_forever, daemon=True)
        cls.thread.start()
    @classmethod
    def tearDownClass(cls):
        cls.http.shutdown()
        cls.http.server_close()
        cls.thread.join()
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.asset = dict(repo='fixture/model', revision='abc', file='model.safetensors',
                          path='model/8bit/model.safetensors', size=len(PAYLOAD),
                          sha256=hashlib.sha256(PAYLOAD).hexdigest())
        (self.root/'model-assets.json').write_text(json.dumps([self.asset]))
        self.manager = ModelDownloads(self.root)
        self.target = self.root/self.asset['path']
        self.target.parent.mkdir(parents=True)
        Handler.ignore_range = False
        Handler.ranges = []
        real_get = requests.get
        self.get = patch('model_downloads.requests.get', side_effect=lambda url, **kw: real_get(f'http://127.0.0.1:{self.http.server_port}/asset', **kw)).start()
        self.addCleanup(patch.stopall)
    def test_download_and_skip_completed(self):
        self.manager.run()
        self.assertEqual(self.target.read_bytes(), PAYLOAD)
        self.assertEqual(self.manager.snapshot()['status'], 'complete')
        self.manager.run()
        self.assertEqual(self.get.call_count, 1)
        self.assertFalse(self.manager.snapshot()['missing'])
    def test_large_download_uses_one_connection_fresh_and_resumed(self):
        payload = b'x' * (64 * 1024 * 1024 + 1)
        self.asset.update(size=len(payload), sha256=hashlib.sha256(payload).hexdigest())
        self.manager.assets = [self.asset]
        for offset in (0, 1000):
            with self.subTest(offset=offset), patch.dict(globals(), PAYLOAD=payload):
                self.target.unlink(missing_ok=True)
                if offset:
                    self.target.with_suffix('.part').write_bytes(payload[:offset])
                Handler.ranges = []
                self.get.reset_mock()
                self.manager.run()
                self.assertEqual(self.manager.snapshot()['status'], 'complete')
                self.assertEqual(self.target.read_bytes(), payload)
                self.assertEqual(self.get.call_count, 1)
                self.assertEqual(Handler.ranges, [f'bytes={offset}-' if offset else None])
                self.assertEqual(self.manager.snapshot()['connections'], 1)
                self.assertEqual(self.manager.snapshot()['transfer_mode'], 'sequential')

    def test_resume(self):
        self.target.with_suffix('.part').write_bytes(PAYLOAD[:1000])
        self.manager.run()
        self.assertEqual(Handler.ranges, ['bytes=1000-'])
        self.assertEqual(self.target.read_bytes(), PAYLOAD)
    def test_range_ignored_restarts_safely(self):
        Handler.ignore_range = True
        self.target.with_suffix('.part').write_bytes(PAYLOAD[:1000])
        self.manager.run()
        self.assertEqual(self.target.read_bytes(), PAYLOAD)
    def test_bad_checksum_then_retry(self):
        self.manager.assets[0]['sha256'] = '0'*64
        self.manager.run()
        self.assertEqual(self.manager.snapshot()['status'], 'failed')
        self.assertFalse(self.target.exists())
        self.assertFalse(self.target.with_suffix('.part').exists())
        self.manager.assets[0]['sha256'] = hashlib.sha256(PAYLOAD).hexdigest()
        self.manager.run()
        self.assertTrue(self.target.exists())
    def test_interrupted_download_retains_partial(self):
        part = self.target.with_suffix('.part')
        part.write_bytes(PAYLOAD[:1000])
        with patch('model_downloads.requests.get', side_effect=requests.ConnectionError('Connection lost')):
            self.manager.run()
        self.assertEqual(self.manager.snapshot()['status'], 'failed')
        self.assertEqual(part.stat().st_size, 1000)
        self.manager.run()
        self.assertEqual(self.target.read_bytes(), PAYLOAD)
    def test_disk_space_error(self):
        with patch('model_downloads.shutil.disk_usage', return_value=shutil._ntuple_diskusage(0, 0, 0)):
            self.manager.run()
        self.assertIn('disk space', self.manager.snapshot()['message'])
        self.get.assert_not_called()
    def test_bf16_download_excludes_optional_model_and_analysis(self):
        self.manager.assets = json.loads((ROOT/'model-assets.json').read_text())
        with patch.object(self.manager, 'download') as download, patch('model_downloads.shutil.disk_usage', return_value=shutil._ntuple_diskusage(10**12, 0, 10**12)):
            self.manager.run('bf16')
        paths = [call.args[0]['path'] for call in download.call_args_list]
        self.assertTrue(paths and all(p.startswith(('model/bf16/', 'model/8bit/')) for p in paths), paths)
        self.assertFalse(any(p.startswith(('transcriber/', 'mert/', 'tokens/', 'model/cuda/')) for p in paths))
        self.assertTrue(all(not a['path'].startswith(('model/', 'tokens/')) for a in self.manager.selected_assets('analysis')))
        cuda=self.manager.selected_assets('cuda-bf16')
        self.assertTrue(cuda)
        self.assertTrue(all(a['path'].startswith('model/cuda/') for a in cuda))
        self.assertEqual(cuda,self.manager.selected_assets('cuda-fp8'))
        with self.assertRaises(ValueError):
            self.manager.start('invalid')

    def test_missing_download_unions_offered_packages(self):
        self.manager.assets = json.loads((ROOT/'model-assets.json').read_text())
        missing = self.manager.selected_assets('missing')
        paths = [a['path'] for a in missing]
        self.assertEqual(len(paths), len(set(paths)))
        offered = set()
        for pkg in self.manager.offered_packages():
            offered.update(a['path'] for a in self.manager.selected_assets(pkg))
        self.assertEqual(set(paths), offered)
        self.assertTrue(any(p.startswith('tokens/') for p in paths))
        if sys.platform == 'darwin':
            self.assertIn('bf16', self.manager.offered_packages())
            self.assertNotIn('cuda-bf16', self.manager.offered_packages())
            self.assertTrue(any(p.startswith('model/bf16/') for p in paths))
            self.assertFalse(any(p.startswith('model/cuda/') for p in paths))
        snap = self.manager.snapshot()
        self.assertFalse(snap['packages']['missing']['ready'])
        self.assertEqual(snap['packages']['missing']['missing_bytes'], sum(a['size'] for a in missing))
        with patch.object(self.manager, 'download') as download, patch('model_downloads.shutil.disk_usage', return_value=shutil._ntuple_diskusage(10**12, 0, 10**12)):
            self.manager.run('missing')
        self.assertEqual([call.args[0]['path'] for call in download.call_args_list], paths)

    def test_lora_packages_download_separately_from_models(self):
        shutil.copy2(ROOT/'lora-assets.json', self.root/'lora-assets.json')
        shutil.copy2(ROOT/'model-assets.json', self.root/'model-assets.json')
        manager = ModelDownloads(self.root)
        catalog = json.loads((ROOT / 'lora-assets.json').read_text())
        self.assertEqual([item['id'] for item in manager.lora_catalog], [item['id'] for item in catalog])
        self.assertIn('lora-mltnt-frontline', manager.lora_ids)
        self.assertIn('lora-qwwl-mehfil', manager.lora_ids)
        missing = [a['path'] for a in manager.selected_assets('missing')]
        self.assertFalse(any(p.startswith('loras/') for p in missing))
        hiphop = manager.selected_assets('lora-oldschoolhiphop')
        self.assertEqual([a['path'] for a in hiphop], ['loras/Old School Hip-Hop.safetensors'])
        mother = manager.selected_assets('lora-mothersuperior-v3')
        self.assertEqual([a['path'] for a in mother], ['loras/YuE2 instrumental (Mothersuperior v3).safetensors'])
        mehfil = manager.selected_assets('lora-qwwl-mehfil')
        self.assertEqual([a['path'] for a in mehfil], ['loras/QWWL Mehfil.safetensors'])
        snap = manager.snapshot()
        self.assertEqual([p['name'] for p in snap['lora_packages']], [item['name'] for item in catalog])
        self.assertEqual((snap['lora_packages'][1]['steps'], snap['lora_packages'][1]['trigger']), (800, 'sv_oldschoolhiphop'))
        self.assertEqual(snap['lora_packages'][0]['settings'],
                         {'instrumental': True, 'lora_strength': 0.7, 'cot': 'full', 'cfg_scale': 1.0})
        self.assertEqual(snap['lora_packages'][1]['settings']['cot'], 'off')
        # Writing adapters set the plan; the sound-only decoder leaves Plan, Style and style influence alone.
        decoder = next(p for p in snap['lora_packages'] if p['id'] == 'lora-realaudio-decoder-v9')
        self.assertEqual(decoder['settings'], {'lora_strength': 1.0})
        self.assertEqual((decoder['trigger'], decoder['steps']), ('', 3000))
        self.assertTrue(all(p.get('settings') and p['settings'].get('cot') for p in snap['lora_packages']
                            if p['id'] != 'lora-realaudio-decoder-v9'))
        self.assertEqual(snap['lora_packages'][0]['steps'], 5000)
        qwwl = next(p for p in snap['lora_packages'] if p['id'] == 'lora-qwwl-mehfil')
        self.assertEqual((qwwl['family'], qwwl['trigger'], qwwl['settings']['cot']),
                         ('QWWL / DRKSF - Qawwali', 'qwwl', 'off'))
        self.assertEqual(qwwl['repo'], 'becausereasons/yue2-qwwl-qawwali-sufi-tabla')
        chnsn = next(p for p in snap['lora_packages'] if p['id'] == 'lora-chnsn-rive-gauche')
        self.assertEqual(chnsn['repo'], 'becausereasons/yue2-chnsn-chanson-francaise')
        self.assertFalse(snap['packages']['lora-oldschoolhiphop']['ready'])
        self.assertEqual(snap['packages']['lora-oldschoolhiphop']['bytes'], hiphop[0]['size'])
        with self.assertRaises(ValueError):
            manager.selected_assets('lora-unknown')
        with patch.object(manager, 'download') as download, patch('model_downloads.shutil.disk_usage', return_value=shutil._ntuple_diskusage(10**12, 0, 10**12)):
            manager.run('lora-oldschoolhiphop')
        self.assertEqual([call.args[0]['path'] for call in download.call_args_list], ['loras/Old School Hip-Hop.safetensors'])

    def test_packages_download_at_the_same_time(self):
        shutil.copy2(ROOT/'lora-assets.json', self.root/'lora-assets.json')
        shutil.copy2(ROOT/'model-assets.json', self.root/'model-assets.json')
        manager = ModelDownloads(self.root)
        barrier = threading.Barrier(3)
        seen = []
        def fake_download(asset, model=None):
            seen.append(model)
            barrier.wait(timeout=2)
            path = manager.root / asset['path']
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'\0' * asset['size'])
        with patch.object(manager, 'download', side_effect=fake_download), patch('model_downloads.shutil.disk_usage', return_value=shutil._ntuple_diskusage(10**12, 0, 10**12)):
            first = manager.start('lora-qwwl-mehfil')
            second = manager.start('lora-oldschoolhiphop')
            self.assertEqual(first['packages']['lora-qwwl-mehfil']['status'], 'downloading')
            self.assertEqual(second['packages']['lora-oldschoolhiphop']['status'], 'downloading')
            barrier.wait(timeout=2)
            snap = manager.snapshot()
            self.assertEqual(snap['status'], 'downloading')
            self.assertEqual(snap['model'], 'multiple')
            until = time.monotonic() + 2
            while any(job.get('status') == 'downloading' for job in manager.jobs.values()) and time.monotonic() < until:
                time.sleep(0.02)
        snap = manager.snapshot()
        self.assertEqual(sorted(seen), ['lora-oldschoolhiphop', 'lora-qwwl-mehfil'])
        self.assertEqual(snap['status'], 'complete')
        self.assertTrue(snap['packages']['lora-qwwl-mehfil']['ready'])
        self.assertTrue(snap['packages']['lora-oldschoolhiphop']['ready'])
        started = threading.Event()
        release = threading.Event()
        calls = []
        def hold(asset, model=None):
            calls.append(model)
            started.set()
            release.wait(timeout=2)
            path = manager.root / asset['path']
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'\0' * asset['size'])
        with patch.object(manager, 'download', side_effect=hold), patch('model_downloads.shutil.disk_usage', return_value=shutil._ntuple_diskusage(10**12, 0, 10**12)):
            manager.start('lora-mothersuperior-v3')
            self.assertTrue(started.wait(2))
            manager.start('lora-mothersuperior-v3')
            release.set()
            until = time.monotonic() + 2
            while manager.jobs.get('lora-mothersuperior-v3', {}).get('status') == 'downloading' and time.monotonic() < until:
                time.sleep(0.02)
        self.assertEqual(calls, ['lora-mothersuperior-v3'])

    def test_download_all_loras_starts_every_missing_adapter(self):
        shutil.copy2(ROOT/'lora-assets.json', self.root/'lora-assets.json')
        shutil.copy2(ROOT/'model-assets.json', self.root/'model-assets.json')
        manager = ModelDownloads(self.root)
        lora_paths = [a['path'] for a in manager.selected_assets('loras')]
        self.assertTrue(lora_paths)
        self.assertTrue(all(p.startswith('loras/') for p in lora_paths))
        self.assertEqual(len(lora_paths), len(manager.lora_ids))
        self.assertFalse(any(p.startswith('loras/') for p in [a['path'] for a in manager.selected_assets('missing')]))
        self.assertEqual(manager.snapshot()['packages']['loras']['missing_bytes'], sum(a['size'] for a in manager.selected_assets('loras')))
        ready = manager.lora_ids[0]
        first = manager.selected_assets(ready)[0]
        dest = manager.root / first['path']
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b'\0' * first['size'])
        started = []
        def fake_download(asset, model=None):
            started.append(model)
            path = manager.root / asset['path']
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'\0' * asset['size'])
        with patch.object(manager, 'download', side_effect=fake_download), patch('model_downloads.shutil.disk_usage', return_value=shutil._ntuple_diskusage(10**12, 0, 10**12)):
            snap = manager.start('loras')
            self.assertEqual(snap['packages'][ready]['status'], '')
            self.assertTrue(all(snap['packages'][pkg]['status'] == 'downloading' for pkg in manager.lora_ids if pkg != ready))
            until = time.monotonic() + 4
            while any(job.get('status') == 'downloading' for job in manager.jobs.values()) and time.monotonic() < until:
                time.sleep(0.02)
        self.assertEqual(sorted(started), sorted(pkg for pkg in manager.lora_ids if pkg != ready))
        snap = manager.snapshot()
        self.assertTrue(snap['packages']['loras']['ready'])
        with patch.object(manager, 'download') as download:
            manager.start('loras')
            download.assert_not_called()

    def test_shared_links_are_relative(self):
        for name in ('qwen.tiktoken', 'vae.safetensors'):
            (self.target.parent/name).write_bytes(b'fixture')
        self.manager.link_shared()
        for name in ('qwen.tiktoken', 'vae.safetensors'):
            link = self.root/'model/bf16'/name
            if os.name != 'nt':
                self.assertTrue(link.is_symlink())
                self.assertFalse(link.readlink().is_absolute())
            self.assertTrue(link.samefile(self.target.parent/name))
            self.assertEqual(link.read_bytes(), b'fixture')

    @unittest.skipUnless(os.name == 'nt', 'Windows link fallback')
    def test_shared_files_without_symlink_privileges(self):
        for name in ('qwen.tiktoken', 'vae.safetensors'):
            (self.target.parent/name).write_bytes(b'fixture')
        with patch.object(Path, 'symlink_to', side_effect=OSError('privilege required')):
            self.manager.link_shared()
            self.manager.link_shared()  # Rerunning setup preserves existing links.
        for name in ('qwen.tiktoken', 'vae.safetensors'):
            link = self.root/'model/bf16'/name
            self.assertTrue(link.samefile(self.target.parent/name))
            self.assertEqual(link.read_bytes(), b'fixture')

class FirstRunTest(unittest.TestCase):
    def test_server_without_any_models(self):
        root = studio_root(self)
        module = load_studio(self, root, 'first_run_server')
        with TestClient(module.app) as client:
            self.assertEqual(client.get('/').status_code, 200)
            config = client.get('/api/config').json()
            self.assertEqual(config['default_model'], 'cuda-bf16' if sys.platform=='win32' else 'bf16')
            self.assertEqual(client.get('/api/session?token=invalid').status_code,403)
            async def page_lifetime():
                first=await module.page_session(None,config['token'])
                second=await module.page_session(None,config['token'])
                await first.body_iterator.__anext__();await second.body_iterator.__anext__()
                self.assertEqual(len(module.leases.pages),2)
                await first.body_iterator.aclose()
                self.assertEqual(len(module.leases.pages),1)
                await second.body_iterator.aclose()
                self.assertFalse(module.leases.pages)
                self.assertIsNotNone(module.leases.timer)
                module.leases.shutdown()
            asyncio.run(page_lifetime())
            self.assertTrue(all(not model['ready'] for model in config['models']))
            self.assertTrue(config['downloads']['missing'], 'nothing is installed, so everything is missing')
            self.assertFalse(any(pkg['ready'] for pkg in config['downloads']['packages'].values()))
            self.assertFalse(config['transcriber_ready'])
            headers = {'X-Studio-Token': config['token']}
            self.assertEqual(client.post('/api/models/preload').status_code,403)
            self.assertEqual(client.post('/api/models/preload',headers=headers).json(),{'status':'waiting'})
            page=module.leases.open()
            self.assertEqual(client.post('/api/models/preload',headers=headers).json(),{'status':'missing'})
            with patch.object(module,'model_ready',return_value=True), patch.object(module.preloads,'start') as warm:
                module.active='generation'
                self.assertEqual(client.post('/api/models/preload',headers=headers).json(),{'status':'waiting'})
                warm.assert_not_called()
                module.active=None
                client.post('/api/models/preload',headers=headers)
                self.assertEqual(warm.call_args.args[0],'bf16')
                self.assertEqual(warm.call_args.args[1]['kind'],'plan')
            module.leases.close(page);module.leases.shutdown()
            self.assertEqual(client.post('/api/models/download').status_code, 403)
            headers = {'X-Studio-Token': config['token']}
            self.assertEqual(client.post('/api/jobs', headers=headers, json={'style':'Orchestral','instrumental':True}).status_code, 503)
            with patch.object(module.downloads, 'start', return_value={'status':'downloading'}) as start:
                self.assertEqual(client.post('/api/models/download', headers=headers).status_code, 200)
                start.assert_called_once_with('bf16')
                module.active = 'busy'
                self.assertEqual(client.post('/api/models/download', headers=headers).status_code, 409)
                start.assert_called_once_with('bf16')
            module.active = None
            with patch.object(module.downloads, 'start', return_value={'status':'downloading'}) as start:
                self.assertEqual(client.post('/api/models/download?model=missing', headers=headers).status_code, 200)
                start.assert_called_once_with('missing')
            with patch.object(module.downloads, 'start', return_value={'status':'downloading'}) as start:
                self.assertEqual(client.post('/api/models/download?model=lora-oldschoolhiphop', headers=headers).status_code, 200)
                start.assert_called_once_with('lora-oldschoolhiphop')
            with patch.object(module.downloads, 'start', return_value={'status':'downloading'}) as start:
                self.assertEqual(client.post('/api/models/download?model=loras', headers=headers).status_code, 200)
                start.assert_called_once_with('loras')
            self.assertEqual(client.post('/api/models/download?model=nope', headers=headers).status_code, 400)
            self.assertFalse(config['downloads']['packages']['missing']['ready'])
            catalog = json.loads((ROOT / 'lora-assets.json').read_text())
            self.assertEqual([p['name'] for p in config['downloads']['lora_packages']],
                             [item['name'] for item in catalog])
            self.assertTrue(any(p['id'] == 'lora-mltnt-frontline' for p in config['downloads']['lora_packages']))

if __name__ == '__main__':
    unittest.main()
