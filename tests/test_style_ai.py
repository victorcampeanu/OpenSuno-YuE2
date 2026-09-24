import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0,str(_Path(__file__).resolve().parents[1]/'app'))
import base64
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from style_ai import ArtworkAsk, StyleAsk, adapt_chat_body, artwork_prompt, chat_completion_body, filter_chat_models, generate_artwork, generate_style, list_models, openai_configured, parse_style, resolve_key, resolve_model, router, stored_openai_key, user_message, uses_completion_tokens, validate_key

class StyleAITest(unittest.TestCase):
    def test_parse_style_strips_fences_and_quotes(self):
        self.assertEqual(parse_style('  "English, jazz-funk, Rhodes piano"  '), 'English, jazz-funk, Rhodes piano')
        self.assertEqual(parse_style('```\nEnglish, soul R&B, warm vocal\n```'), 'English, soul R&B, warm vocal')

    def test_user_message_includes_instrumental_and_current_style(self):
        text = user_message(StyleAsk(prompt='club energy', instrumental=True, current_style='English, pop'))
        self.assertIn('club energy', text)
        self.assertIn('instrumental', text.lower())
        self.assertIn('English, pop', text)

    def test_resolve_key_prefers_submitted_then_env_then_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / '.openai-key').write_text('sk-' + 'f' * 24, encoding='utf-8')
            with patch.dict(os.environ, {'OPENAI_API_KEY': 'sk-' + 'e' * 24}, clear=False):
                self.assertEqual(resolve_key('sk-' + 's' * 24, root), 'sk-' + 's' * 24)
                self.assertEqual((root / '.openai-key').read_text(encoding='utf-8').strip(), 'sk-' + 's' * 24)
            (root / '.openai-key').write_text('sk-' + 'f' * 24, encoding='utf-8')
            with patch.dict(os.environ, {'OPENAI_API_KEY': 'sk-' + 'e' * 24}, clear=False):
                self.assertEqual(resolve_key('', root), 'sk-' + 'e' * 24)
            env = os.environ.pop('OPENAI_API_KEY', None)
            try:
                self.assertEqual(resolve_key('', root), 'sk-' + 'f' * 24)
                self.assertTrue(openai_configured(root))
            finally:
                if env is not None:
                    os.environ['OPENAI_API_KEY'] = env
            empty = Path(tmp) / 'empty'
            empty.mkdir()
            with self.assertRaises(HTTPException) as missing:
                resolve_key('', empty)
            self.assertEqual(missing.exception.status_code, 400)
            with self.assertRaises(HTTPException) as invalid:
                resolve_key('not-a-key', empty)
            self.assertEqual(invalid.exception.status_code, 400)

    def test_stored_key_ignores_blank_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / '.openai-key').write_text('  \n', encoding='utf-8')
            env = os.environ.pop('OPENAI_API_KEY', None)
            try:
                self.assertEqual(stored_openai_key(root), '')
                self.assertFalse(openai_configured(root))
            finally:
                if env is not None:
                    os.environ['OPENAI_API_KEY'] = env

    def test_resolve_model_allows_api_ids(self):
        self.assertEqual(resolve_model('gpt-4o'), 'gpt-4o')
        self.assertEqual(resolve_model('gpt-4o-mini-2024-07-18'), 'gpt-4o-mini-2024-07-18')
        self.assertEqual(resolve_model('bad model!'), 'gpt-4o-mini')

    def test_filter_chat_models_omits_non_chat(self):
        models = filter_chat_models(['whisper-1', 'gpt-4o', 'dall-e-3', 'text-embedding-3-large', 'o3-mini', 'gpt-4o-mini-2024-07-18', 'tts-1'])
        self.assertEqual([item['id'] for item in models], ['gpt-4o', 'o3-mini', 'gpt-4o-mini-2024-07-18'])

    def test_generate_style_reads_openai_payload(self):
        payload = json.dumps({'choices': [{'message': {'content': 'English, nu-disco, funky bass'}}], 'usage': {'prompt_tokens': 120, 'completion_tokens': 18}}).encode()
        class Fake:
            def __enter__(self):
                return io.BytesIO(payload)
            def __exit__(self, *args):
                return False
        captured = {}
        def fake_urlopen(request, timeout=0):
            captured['body'] = json.loads(request.data.decode())
            return Fake()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch('style_ai.urlopen', side_effect=fake_urlopen):
                self.assertEqual(generate_style(StyleAsk(prompt='disco night', api_key='sk-' + 'a' * 24, model='gpt-4o'), root), 'English, nu-disco, funky bass')
        self.assertEqual(captured['body']['model'], 'gpt-4o')
        self.assertEqual(captured['body']['max_tokens'], 180)
        self.assertNotIn('max_completion_tokens', captured['body'])
        self.assertEqual(captured['body']['temperature'], 0.8)

    def test_chat_body_uses_completion_tokens_for_gpt5_and_o_series(self):
        self.assertTrue(uses_completion_tokens('gpt-5'))
        self.assertTrue(uses_completion_tokens('gpt-5.2'))
        self.assertTrue(uses_completion_tokens('gpt-5-mini'))
        self.assertTrue(uses_completion_tokens('o3-mini'))
        self.assertFalse(uses_completion_tokens('gpt-4o'))
        self.assertFalse(uses_completion_tokens('gpt-4.1'))
        gpt5 = chat_completion_body('gpt-5', [{'role': 'user', 'content': 'x'}])
        self.assertEqual(gpt5['max_completion_tokens'], 180)
        self.assertNotIn('max_tokens', gpt5)
        self.assertNotIn('temperature', gpt5)
        o3 = chat_completion_body('o3', [{'role': 'user', 'content': 'x'}])
        self.assertEqual(o3['max_completion_tokens'], 180)
        self.assertNotIn('temperature', o3)

    def test_adapt_chat_body_renames_or_drops_unsupported_params(self):
        body = {'model': 'gpt-5', 'max_tokens': 180, 'temperature': 0.8}
        renamed = adapt_chat_body(body, "Unsupported parameter: 'max_tokens' is not supported with this model. Use 'max_completion_tokens' instead.")
        self.assertEqual(renamed['max_completion_tokens'], 180)
        self.assertNotIn('max_tokens', renamed)
        dropped = adapt_chat_body(renamed, "Unsupported value: 'temperature' does not support 0.8 with this model. Only the default (1) value is supported.")
        self.assertNotIn('temperature', dropped)
        self.assertIsNone(adapt_chat_body(dropped, 'some other 400'))

    def test_generate_style_retries_with_completion_tokens(self):
        payload = json.dumps({'choices': [{'message': {'content': 'English, house, four-on-the-floor'}}]}).encode()
        class Fake:
            def __enter__(self):
                return io.BytesIO(payload)
            def __exit__(self, *args):
                return False
        err = json.dumps({'error': {'message': "Unsupported parameter: 'max_tokens' is not supported with this model. Use 'max_completion_tokens' instead."}}).encode()
        calls = []
        def fake_urlopen(request, timeout=0):
            calls.append(json.loads(request.data.decode()))
            if len(calls) == 1:
                raise HTTPError('https://api.openai.com/v1/chat/completions', 400, 'Bad Request', hdrs=None, fp=io.BytesIO(err))
            return Fake()
        with tempfile.TemporaryDirectory() as tmp:
            with patch('style_ai.urlopen', side_effect=fake_urlopen):
                style = generate_style(StyleAsk(prompt='club night', api_key='sk-' + 'a' * 24, model='gpt-4o'), Path(tmp))
        self.assertEqual(style, 'English, house, four-on-the-floor')
        self.assertEqual(calls[0]['max_tokens'], 180)
        self.assertEqual(calls[1]['max_completion_tokens'], 180)
        self.assertNotIn('max_tokens', calls[1])

    def test_generate_style_retries_without_locked_temperature(self):
        payload = json.dumps({'choices': [{'message': {'content': 'English, club, funky bass'}}]}).encode()
        class Fake:
            def __enter__(self):
                return io.BytesIO(payload)
            def __exit__(self, *args):
                return False
        err = json.dumps({'error': {'message': "Unsupported value: 'temperature' does not support 0.8 with this model. Only the default (1) value is supported."}}).encode()
        calls = []
        def fake_urlopen(request, timeout=0):
            calls.append(json.loads(request.data.decode()))
            if len(calls) == 1:
                raise HTTPError('https://api.openai.com/v1/chat/completions', 400, 'Bad Request', hdrs=None, fp=io.BytesIO(err))
            return Fake()
        with tempfile.TemporaryDirectory() as tmp:
            with patch('style_ai.urlopen', side_effect=fake_urlopen):
                style = generate_style(StyleAsk(prompt='club night', api_key='sk-' + 'a' * 24, model='gpt-4o'), Path(tmp))
        self.assertEqual(style, 'English, club, funky bass')
        self.assertEqual(calls[0]['temperature'], 0.8)
        self.assertNotIn('temperature', calls[1])

    def test_generate_style_maps_unauthorized(self):
        error = HTTPError('https://api.openai.com/v1/chat/completions', 401, 'Unauthorized', hdrs=None, fp=io.BytesIO(b'{}'))
        with patch('style_ai.urlopen', side_effect=error):
            with self.assertRaises(HTTPException) as unauthorized:
                generate_style(StyleAsk(prompt='disco night', api_key='sk-' + 'a' * 24))
        self.assertEqual(unauthorized.exception.status_code, 401)

    def test_list_models_reads_openai_catalog(self):
        payload = json.dumps({'data': [{'id': 'gpt-4o'}, {'id': 'whisper-1'}, {'id': 'o3-mini'}]}).encode()
        class Fake:
            def __enter__(self):
                return io.BytesIO(payload)
            def __exit__(self, *args):
                return False
        with patch('style_ai.urlopen', return_value=Fake()):
            models = list_models('sk-' + 'a' * 24)
        self.assertEqual([item['id'] for item in models], ['gpt-4o', 'o3-mini'])
        with patch('style_ai.urlopen', return_value=Fake()):
            self.assertEqual(validate_key('sk-' + 'a' * 24), {'valid': True})

    def test_endpoint_returns_generated_style(self):
        app = FastAPI()
        app.include_router(router, prefix='/api')
        client = TestClient(app)
        with patch('style_ai.generate_style', return_value='English, jazz, upright bass'):
            response = client.post('/api/style/generate', json={'prompt': 'warm jazz club'})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json(), {'style': 'English, jazz, upright bass'})
        self.assertEqual(client.post('/api/style/generate', json={'prompt': 'x'}).status_code, 422)

    def test_artwork_prompt_keeps_cover_rules(self):
        instrumental = artwork_prompt('La Isla Bonita', 'funky style or prince and parliament funkedelics', instrumental=True)
        self.assertIn('No text', instrumental)
        self.assertIn('fair-skinned, East Asian, or Hispanic', instrumental)
        self.assertIn('La Isla Bonita', instrumental)
        self.assertIn('funky style', instrumental)
        self.assertIn('Translate this music style into the image', instrumental)
        self.assertNotIn('Lyric mood', instrumental)
        vocal = artwork_prompt('Night Drive', 'synthwave, analog synthesizers', lyrics='[Verse]\nLast night I dreamt of San Pedro\nJust like I had never gone', variation='job-a')
        self.assertIn('Night Drive', vocal)
        self.assertIn('Last night I dreamt of San Pedro', vocal)
        self.assertNotIn('[Verse]', vocal)
        self.assertIn('synthwave', vocal)
        self.assertIn('Translate this music style into the image', vocal)
        self.assertIn('Visual take:', vocal)
        other = artwork_prompt('Night Drive', 'synthwave, analog synthesizers', lyrics='Last night I dreamt of San Pedro', variation='job-b')
        self.assertIn('Visual take:', other)
        self.assertNotEqual(vocal, other)
        self.assertIn('Untitled song', artwork_prompt('', ''))

    def test_generate_artwork_writes_png_and_elapsed(self):
        png = b'\x89PNG\r\n\x1a\n' + b'x' * 40
        payload = json.dumps({
            'data': [{'b64_json': base64.b64encode(png).decode()}],
            'usage': {'input_tokens': 90, 'output_tokens': 124, 'input_tokens_details': {'text_tokens': 90, 'image_tokens': 0}},
        }).encode()
        class Fake:
            def __enter__(self):
                return io.BytesIO(payload)
            def __exit__(self, *args):
                return False
        captured = {}
        def fake_urlopen(request, timeout=0):
            captured['body'] = json.loads(request.data.decode())
            captured['timeout'] = timeout
            return Fake()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dest = root / 'artwork.png'
            with patch('style_ai.urlopen', side_effect=fake_urlopen):
                meta = generate_artwork(ArtworkAsk(title='Night Drive', style='synthwave', api_key='sk-' + 'a' * 24), dest, root)
            self.assertEqual(captured['body']['model'], 'gpt-image-2.5-flare')
            self.assertEqual(captured['body']['quality'], 'low')
            self.assertEqual(captured['body']['size'], '816x816')
            self.assertEqual(captured['body']['output_format'], 'webp')
            self.assertEqual(captured['body']['output_compression'], 80)
            self.assertIn('Night Drive', captured['body']['prompt'])
            self.assertIn('synthwave', captured['body']['prompt'])
            self.assertIn('Visual take:', captured['body']['prompt'])
            self.assertGreaterEqual(captured['timeout'], 90)
            self.assertEqual(dest.read_bytes(), png)
            self.assertEqual(meta['status'], 'complete')
            self.assertGreaterEqual(meta['elapsed'], 0)

if __name__ == '__main__':
    unittest.main()
