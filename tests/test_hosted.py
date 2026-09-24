import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

from fastapi.testclient import TestClient
from studio_fixture import studio_root, load_studio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class HostedStudioTest(unittest.TestCase):
    def setUp(self):
        self.root = studio_root(self)
        self.previous = {key: os.environ.get(key) for key in ('OPENSUNO_HOSTED', 'OPENSUNO_DATA', 'VERCEL')}
        os.environ['OPENSUNO_HOSTED'] = '1'
        os.environ['OPENSUNO_DATA'] = str(self.root)
        os.environ.pop('VERCEL', None)
        def restore():
            for key, value in self.previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        self.addCleanup(restore)
        self.s = load_studio(self, self.root, 'hosted_test_server')
        self.client = TestClient(self.s.app)

    def test_hosted_config_hides_downloads_and_marks_the_site_ready(self):
        self.assertTrue(self.s.HOSTED)
        page = self.client.get('/', headers={'host': 'opensuno.vercel.app'})
        self.assertEqual(page.status_code, 200)
        self.assertIn(b'OpenSuno', page.content)
        config = self.client.get('/api/config', headers={'host': 'opensuno.vercel.app'}).json()
        self.assertTrue(config['hosted'])
        self.assertEqual(config['default_model'], 'bf16')
        self.assertEqual([m['id'] for m in config['models']], ['bf16'])
        self.assertTrue(config['models'][0]['ready'])
        self.assertEqual(config['downloads']['status'], 'idle')
        self.assertFalse(config['downloads']['packages'])
        self.assertFalse(config['transcriber_ready'])

    def test_hosted_rejects_model_downloads_and_generation(self):
        headers = {'host': 'opensuno.vercel.app'}
        download = self.client.post('/api/models/download', headers=headers)
        self.assertEqual(download.status_code, 404)
        self.assertIn('local Studio', download.json()['detail'])
        job = self.client.post('/api/jobs', headers=headers, json={'style': 'Indie pop', 'instrumental': True})
        self.assertEqual(job.status_code, 503)
        self.assertIn('website only', job.json()['detail'])
        self.assertEqual(self.client.post('/api/models/preload', headers=headers).json(), {'status': 'ready'})
        self.assertEqual(self.client.get('/api/jobs', headers=headers).json(), [])
        workspaces = self.client.get('/api/workspaces', headers=headers).json()
        self.assertEqual(workspaces['workspaces'][0]['id'], 'default')
        prompts = self.client.get('/api/prompts', headers=headers).json()
        self.assertEqual(prompts, {'prompts': []})
        saved = self.client.post('/api/prompts', json={'title': 'Cloud Hook', 'style': 'indie pop'}, headers=headers)
        self.assertEqual(saved.status_code, 200, saved.text)
        self.assertEqual(saved.json()['title'], 'Cloud Hook')

    def test_hosted_skips_studio_token_and_allows_style_key_test_shape(self):
        response = self.client.post('/api/workspaces', json={'name': 'Cloud'}, headers={'host': 'opensuno.vercel.app'})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()['name'], 'Cloud')

    def test_hosted_imports_without_numpy(self):
        """Vercel installs requirements.txt only; a top-level NumPy import 500s the whole site."""
        script = r'''
import os, sys, tempfile
sys.path.insert(0, sys.argv[1])
class Block:
    def find_spec(self, name, path, target=None):
        if name == "numpy" or (name and name.startswith("numpy.")):
            raise ModuleNotFoundError(name)
        return None
sys.meta_path.insert(0, Block())
os.environ["OPENSUNO_HOSTED"] = "1"
os.environ["OPENSUNO_DATA"] = tempfile.mkdtemp()
os.environ.pop("VERCEL", None)
import server
from fastapi.testclient import TestClient
page = TestClient(server.app).get("/", headers={"host": "opensuno.vercel.app"})
assert page.status_code == 200, page.status_code
assert b"OpenSuno" in page.content
'''
        result = subprocess.run([sys.executable, '-c', script, str(ROOT)], capture_output=True, text=True, cwd=str(ROOT))
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
