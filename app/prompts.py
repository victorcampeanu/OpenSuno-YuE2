"""Saved style prompts and the generation settings that go with them."""
import json
import re
import secrets
import time

from song_library import write_json

TITLE_LIMIT = 80
STYLE_LIMIT = 6000
PROMPT_LIMIT = 200


class PromptError(ValueError):
    pass


class Prompts:
    def __init__(self, path):
        self.path = path

    def read(self):
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text())
            except (OSError, ValueError):
                return {'prompts': []}
            items = data.get('prompts')
            return {'prompts': items} if isinstance(items, list) else {'prompts': []}
        return {'prompts': []}

    def listing(self):
        return sorted(self.read()['prompts'], key=lambda p: p.get('updated') or p.get('created') or 0, reverse=True)

    def require(self, data, prompt_id):
        prompt = next((p for p in data['prompts'] if p.get('id') == prompt_id), None)
        if prompt is None:
            raise PromptError('That saved prompt no longer exists.')
        return prompt

    def title(self, value):
        value = re.sub(r'\s+', ' ', str(value or '')).strip()[:TITLE_LIMIT]
        if not value:
            raise PromptError('Enter a prompt title between 1 and 80 characters.')
        return value

    def style(self, value):
        value = str(value or '')
        if len(value) > STYLE_LIMIT:
            raise PromptError('Style text is too long.')
        return value

    def create(self, title, style, settings):
        data = self.read()
        if len(data['prompts']) >= PROMPT_LIMIT:
            raise PromptError('You already have 200 saved prompts. Delete one to save another.')
        now = time.time()
        prompt = {
            'id': secrets.token_hex(8),
            'title': self.title(title),
            'style': self.style(style),
            'settings': settings or {},
            'created': now,
            'updated': now,
        }
        data['prompts'].append(prompt)
        write_json(self.path, data)
        return prompt

    def rename(self, prompt_id, title):
        data = self.read()
        prompt = self.require(data, self.valid_id(prompt_id))
        prompt['title'] = self.title(title)
        prompt['updated'] = time.time()
        write_json(self.path, data)
        return prompt

    def delete(self, prompt_id):
        data = self.read()
        self.require(data, self.valid_id(prompt_id))
        data['prompts'] = [p for p in data['prompts'] if p.get('id') != prompt_id]
        write_json(self.path, data)

    def valid_id(self, key):
        if not isinstance(key, str) or not re.fullmatch(r'[a-f0-9]{16}', key):
            raise PromptError('Unknown prompt.')
        return key
