import json
from pathlib import Path
import sys
import unittest

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'app'))
from prompts import Prompts, PROMPT_LIMIT
from studio_fixture import studio_root, load_studio


class PromptAPITest(unittest.TestCase):
    def setUp(self):
        self.root = studio_root(self)
        self.s = load_studio(self, self.root, 'prompt_test_server')
        self.client = TestClient(self.s.app)
        self.headers = {'X-Studio-Token': self.s.TOKEN}

    def snapshot(self):
        return self.client.get('/api/prompts').json()

    def create(self, title='Indie Glow', style='English, indie pop, warm vocal', settings=None):
        body = {'title': title, 'style': style, 'settings': settings or {'voice': 'female', 'cot': 'full', 'steps': 8, 'cfg_scale': 1.4}}
        r = self.client.post('/api/prompts', json=body, headers=self.headers)
        self.assertEqual(r.status_code, 200, r.text)
        return r.json()

    def test_create_list_rename_delete_and_persistence(self):
        self.assertEqual(self.snapshot(), {'prompts': []})
        saved = self.create('  Midnight Radio  ', 'dream pop, analog synths')
        self.assertEqual(saved['title'], 'Midnight Radio')
        self.assertEqual(saved['style'], 'dream pop, analog synths')
        self.assertEqual(saved['settings']['voice'], 'female')
        self.assertEqual(saved['settings']['steps'], 8)
        self.assertRegex(saved['id'], r'^[a-f0-9]{16}$')
        listed = self.snapshot()['prompts']
        self.assertEqual(len(listed), 1)
        self.assertEqual(listed[0]['id'], saved['id'])
        renamed = self.client.patch('/api/prompts/' + saved['id'], json={'title': 'Night Drive'}, headers=self.headers)
        self.assertEqual(renamed.status_code, 200, renamed.text)
        self.assertEqual(renamed.json()['title'], 'Night Drive')
        self.assertEqual(Prompts(self.root / '.prompts.json').listing()[0]['title'], 'Night Drive')
        gone = self.client.delete('/api/prompts/' + saved['id'], headers=self.headers)
        self.assertEqual(gone.status_code, 200)
        self.assertEqual(self.snapshot(), {'prompts': []})

    def test_rejects_blank_titles_and_unknown_ids(self):
        r = self.client.post('/api/prompts', json={'title': '   ', 'style': 'pop'}, headers=self.headers)
        self.assertEqual(r.status_code, 400, r.text)
        missing = self.client.patch('/api/prompts/' + 'a' * 16, json={'title': 'Nope'}, headers=self.headers)
        self.assertEqual(missing.status_code, 400)
        self.assertEqual(self.client.delete('/api/prompts/not-an-id', headers=self.headers).status_code, 400)

    def test_newest_first_and_keeps_sampling(self):
        first = self.create('One', 'piano')
        second = self.create('Two', 'guitar', {'voice': 'male', 'semantic_sampling': {
            'temperature': 0.9, 'top_p': 0.9, 'top_k': 50, 'repetition_penalty': 1.2,
            'penalty_window': 64, 'min_tokens': 10, 'max_tokens': 1000
        }})
        titles = [p['title'] for p in self.snapshot()['prompts']]
        self.assertEqual(titles, ['Two', 'One'])
        self.assertEqual(second['settings']['semantic_sampling']['max_tokens'], 1000)
        self.assertEqual(first['id'] in {p['id'] for p in self.snapshot()['prompts']}, True)

    def test_prompt_cap(self):
        store = Prompts(self.root / '.prompts.json')
        for i in range(PROMPT_LIMIT):
            store.create(f'Prompt {i}', 'style', {})
        r = self.client.post('/api/prompts', json={'title': 'Overflow', 'style': 'x'}, headers=self.headers)
        self.assertEqual(r.status_code, 400)
        self.assertIn('200', r.json()['detail'])


if __name__ == '__main__':
    unittest.main()
