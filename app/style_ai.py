"""Generate YuE2 style prompts with an OpenAI key kept on this computer."""
import base64
import hashlib
import json
import os
import re
import tempfile
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
router = APIRouter()
SYSTEM = '''You write style prompts for YuE2, a text-to-music model.
Return ONLY one style prompt. No quotes, markdown, titles, or explanation.
Use a single line of comma-separated musical cues.
When vocals are present, start with the lyric language (English, Spanish, …).
Then genre, key instruments, rhythm or production, mood, and vocal character (timbre and delivery).
Never state the singer's gender or "male/female vocals": the app adds the chosen voice separately.
Do not write lyrics. Never name an artist or band in the prompt; if the request mentions one, translate their signature sound into concrete cues (genre, instruments, production, tempo feel, vocal timbre) instead.
Keep the prompt under 400 characters.
If the request is instrumental, omit every vocal cue; the app adds the word instrumental itself.'''

MODEL_ID = re.compile(r'^[A-Za-z0-9._:-]{1,80}$')
CHAT_MODEL = re.compile(r'^(gpt-|o[1-9]|chatgpt-)')
SKIP_MODEL = re.compile(r'(embedding|whisper|tts|dall-e|dalle|image|audio|transcribe|realtime|moderation|search|babbage|davinci|ada-|curie|computer-use|sora)')
PREFERRED_MODELS = (
    'gpt-5.2', 'gpt-5.1', 'gpt-5', 'gpt-5-mini', 'gpt-5-nano',
    'gpt-4.1', 'gpt-4.1-mini', 'gpt-4.1-nano', 'gpt-4o', 'gpt-4o-mini',
    'o4-mini', 'o3', 'o3-mini', 'o1', 'o1-mini',
)
class StyleAsk(BaseModel):
    prompt: str = Field(min_length=2, max_length=2000)
    instrumental: bool = False
    current_style: str = Field(default='', max_length=6000)
    api_key: str = Field(default='', max_length=256)
    model: str = Field(default='gpt-4o-mini', max_length=80)

class StyleKey(BaseModel):
    api_key: str = Field(default='', max_length=256)

class ArtworkAsk(BaseModel):
    api_key: str = Field(default='', max_length=256)
    title: str = Field(default='', max_length=200)
    style: str = Field(default='', max_length=6000)
    lyrics: str = Field(default='', max_length=4000)
    instrumental: bool = False
    variation: str = Field(default='', max_length=64)  # the song id; picks the visual take. Empty: the folder name.

IMAGE_MODEL = 'gpt-image-2.5-flare'
IMAGE_QUALITY = 'low'
# Art is shown at most ~400px wide; 816x816 is the smallest square the API accepts (>= 655,360 pixels, edges divisible by 16).
IMAGE_SIZE = '816x816'
IMAGE_FORMAT = 'webp'
IMAGE_COMPRESSION = 80

def resolve_model(name):
    name = (name or '').strip()
    return name if MODEL_ID.fullmatch(name) else 'gpt-4o-mini'

def is_chat_model(name):
    name = (name or '').lower()
    return bool(CHAT_MODEL.match(name) and not SKIP_MODEL.search(name))

def model_sort_key(name):
    dated = 1 if re.search(r'\d{4}-\d{2}-\d{2}', name) else 0
    try:
        rank = PREFERRED_MODELS.index(name)
    except ValueError:
        rank = 80
    return (dated, rank, name)

def filter_chat_models(ids):
    names = sorted({name for name in ids if is_chat_model(name)}, key=model_sort_key)
    return [{'id': name, 'label': name} for name in names]

def validate_key(submitted='', root=None):
    list_models(submitted, root)
    return {'valid': True}

def stored_openai_key(root=None):
    env = os.environ.get('OPENAI_API_KEY', '').strip()
    if env:
        return env
    path = (root or ROOT) / '.openai-key'
    if path.is_file():
        return path.read_text(encoding='utf-8').strip()
    return ''

def openai_configured(root=None):
    return bool(stored_openai_key(root))

def remember_openai_key(key, root=None):
    key = (key or '').strip()
    if not re.fullmatch(r'sk-[A-Za-z0-9_-]{20,}', key):
        return
    path = (root or ROOT) / '.openai-key'
    try:
        if path.is_file() and path.read_text(encoding='utf-8').strip() == key:
            return
        path.write_text(key, encoding='utf-8')
    except OSError:
        pass

def resolve_key(submitted, root=None):
    key = (submitted or '').strip() or stored_openai_key(root)
    if not key:
        raise HTTPException(400, 'Add an OpenAI API key in Settings, or set OPENAI_API_KEY for Studio.')
    if not re.fullmatch(r'sk-[A-Za-z0-9_-]{20,}', key):
        raise HTTPException(400, 'That OpenAI API key does not look valid.')
    if submitted:
        remember_openai_key(key, root)
    return key

def user_message(data):
    parts = [data.prompt.strip()]
    if data.instrumental:
        parts.append('The song is instrumental. Do not include vocal cues.')
    current = data.current_style.strip()
    if current:
        parts.append('Current style prompt (refine or replace as asked):\n' + current)
    return '\n\n'.join(parts)

def parse_style(text):
    style = (text or '').strip().strip('"').strip("'")
    style = re.sub(r'^```(?:\w+)?\s*|\s*```$', '', style).strip()
    style = re.sub(r'\s+', ' ', style)
    if not style:
        raise ValueError('empty')
    if len(style) > 6000:
        style = style[:6000].rsplit(',', 1)[0].strip() or style[:6000]
    return style

RENAME_PARAM = re.compile(
    r"Unsupported parameter: '([^']+)' is not supported with this model\. Use '([^']+)' instead",
    re.I,
)
DROP_PARAM = re.compile(r"Unsupported (?:parameter|value): '([^']+)'", re.I)

def uses_completion_tokens(name):
    # GPT-5 and o-series reject max_tokens and custom temperature.
    return bool(re.match(r'^(gpt-5|o[1-9])', (name or '').lower()))

def chat_completion_body(model, messages, limit=180, temperature=0.8):
    body = {'model': model, 'messages': messages}
    if uses_completion_tokens(model):
        body['max_completion_tokens'] = limit
    else:
        body['max_tokens'] = limit
        body['temperature'] = temperature
    return body

def adapt_chat_body(body, message):
    renamed = RENAME_PARAM.search(message or '')
    if renamed:
        old, new = renamed.group(1), renamed.group(2)
        if old in body and new not in body:
            adapted = dict(body)
            adapted[new] = adapted.pop(old)
            return adapted
    dropped = DROP_PARAM.search(message or '')
    if dropped and dropped.group(1) in body:
        adapted = dict(body)
        del adapted[dropped.group(1)]
        return adapted
    return None

def openai_error_body(error):
    raw = getattr(error, '_openai_body', None)
    if raw is None:
        raw = error.read() or b''
        error._openai_body = raw
    return raw

def openai_error(error, action):
    raw = openai_error_body(error)
    message = ''
    try:
        message = str((json.loads(raw).get('error') or {}).get('message') or '').strip()
    except (TypeError, ValueError, AttributeError):
        message = ''
    if error.code == 400:
        raise HTTPException(400, message or 'OpenAI rejected this request. Try again with a simpler prompt.') from error
    if error.code == 401:
        raise HTTPException(401, 'OpenAI rejected this API key. Check it in Settings and try again.') from error
    if error.code == 429:
        raise HTTPException(429, 'OpenAI is rate-limiting this key. Wait a moment and try again.') from error
    raise HTTPException(502, 'OpenAI could not '+action+'. Try again shortly.') from error

def artwork_snippet(text, limit, lyrics=False):
    text = (text or '').strip()
    if lyrics:
        text = re.sub(r'\[[^\]]+\]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    if len(text) <= limit:
        return text
    clipped = text[:limit].rsplit(' ', 1)[0].strip()
    return clipped or text[:limit]

ARTWORK_TAKES = (
    'cool dusk palette, wide landscape, low horizon, grainy print',
    'warm close crop, high contrast, flat graphic shapes',
    'night scene, saturated neon edges, off-center subject',
    'sun-bleached midday, sparse composition, film still',
    'deep shadows, jewel tones, theatrical lighting',
    'pastel haze, soft focus, floating objects',
    'high-key pop colors, bold geometry, poster crop',
    'muted earth tones, documentary framing, quiet space',
    'stormy contrast, motion blur, diagonal energy',
    'monochrome with one accent color, stark silhouette',
    'retro print texture, overprint colors, 1970s poster',
    'wet streets, reflections, cinematic widescreen crop',
)

def artwork_take(key):
    digest = hashlib.sha256((key or '').encode()).digest()
    return ARTWORK_TAKES[digest[0] % len(ARTWORK_TAKES)]

def artwork_prompt(title='', style='', lyrics='', instrumental=False, variation=''):
    title = artwork_snippet(title, 80) or 'Untitled song'
    style = artwork_snippet(style, 220)
    parts = [
        'Square album cover, graphic illustration, bold color, centered composition.',
        'No text, letters, logos, watermarks, or frames.',
        'If people appear: fair-skinned, East Asian, or Hispanic.',
        'Title: ' + title + '.',
    ]
    if style:
        parts.append('Translate this music style into the image — color, lighting, texture, and era: ' + style + '.')
    if instrumental:
        parts.append('No singer portrait.')
    else:
        excerpt = artwork_snippet(lyrics, 240, lyrics=True)
        if excerpt:
            parts.append('Lyric mood: ' + excerpt + '.')
    parts.append('Invent a new composition; do not copy another cover of this title.')
    if variation:
        parts.append('Visual take: ' + artwork_take(variation) + '.')
    return ' '.join(parts)

def generate_artwork(data, dest, root=None):
    key = resolve_key(data.api_key, root)
    dest = Path(dest)
    started = time.monotonic()
    prompt = artwork_prompt(data.title, data.style, data.lyrics, data.instrumental, data.variation or dest.parent.name)
    body = json.dumps({
        'model': IMAGE_MODEL,
        'prompt': prompt,
        'n': 1,
        'size': IMAGE_SIZE,
        'quality': IMAGE_QUALITY,
        'output_format': IMAGE_FORMAT,
        'output_compression': IMAGE_COMPRESSION,
    }).encode()
    request = Request(
        'https://api.openai.com/v1/images/generations',
        data=body,
        method='POST',
        headers={
            'Authorization': 'Bearer ' + key,
            'Content-Type': 'application/json',
            'User-Agent': 'OpenSuno/1.0',
        },
    )
    try:
        with urlopen(request, timeout=90) as response:
            payload = json.load(response)
        raw = base64.b64decode(payload['data'][0]['b64_json'], validate=True)
        if len(raw) < 32:
            raise ValueError('empty')
        tmp = dest.with_name(dest.name + '.tmp')
        tmp.write_bytes(raw)
        tmp.replace(dest)
        return {
            'status': 'complete',
            'model': IMAGE_MODEL,
            'quality': IMAGE_QUALITY,
            'elapsed': round(time.monotonic() - started, 1),
        }
    except HTTPError as error:
        openai_error(error, 'generate album art')
    except (URLError, TimeoutError, ValueError, KeyError, TypeError, IndexError, json.JSONDecodeError) as error:
        raise HTTPException(502, 'Could not generate album art. Check your internet connection and try again.') from error

def list_models(submitted='', root=None):
    key = resolve_key(submitted, root)
    request = Request(
        'https://api.openai.com/v1/models',
        method='GET',
        headers={'Authorization': 'Bearer ' + key, 'User-Agent': 'OpenSuno/1.0'},
    )
    try:
        with urlopen(request, timeout=20) as response:
            payload = json.load(response)
        ids = [item['id'] for item in payload.get('data', []) if isinstance(item, dict) and item.get('id')]
        models = filter_chat_models(ids)
        if not models:
            raise HTTPException(502, 'OpenAI returned no chat models for this key.')
        return models
    except HTTPError as error:
        openai_error(error, 'list models')
    except (URLError, TimeoutError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise HTTPException(502, 'Could not load OpenAI models. Check your internet connection and try again.') from error

def generate_style(data, root=None):
    key = resolve_key(data.api_key, root)
    body = chat_completion_body(resolve_model(data.model), [
        {'role': 'system', 'content': SYSTEM},
        {'role': 'user', 'content': user_message(data)},
    ])
    tried = {frozenset(body)}
    while True:
        request = Request(
            'https://api.openai.com/v1/chat/completions',
            data=json.dumps(body).encode(),
            method='POST',
            headers={
                'Authorization': 'Bearer ' + key,
                'Content-Type': 'application/json',
                'User-Agent': 'OpenSuno/1.0',
            },
        )
        try:
            with urlopen(request, timeout=45) as response:
                payload = json.load(response)
            style = parse_style(payload['choices'][0]['message']['content'])
            return style
        except HTTPError as error:
            raw = openai_error_body(error)
            message = ''
            try:
                message = str((json.loads(raw).get('error') or {}).get('message') or '').strip()
            except (TypeError, ValueError, AttributeError):
                message = ''
            adapted = adapt_chat_body(body, message) if error.code == 400 else None
            if adapted and frozenset(adapted) not in tried:
                tried.add(frozenset(adapted))
                body = adapted
                continue
            openai_error(error, 'generate a style')
        except (URLError, TimeoutError, ValueError, KeyError, TypeError, IndexError, json.JSONDecodeError) as error:
            raise HTTPException(502, 'Could not generate a style. Check your internet connection and try again.') from error

@router.post('/style/key')
def style_key(data: StyleKey):
    return validate_key(data.api_key)

@router.post('/style/models')
def style_models(data: StyleKey):
    return {'models': list_models(data.api_key)}

@router.post('/style/generate')
def style_generate(data: StyleAsk):
    return {'style': generate_style(data)}

@router.post('/artwork')
def artwork_endpoint(data: ArtworkAsk):
    """Album art for a machine without the key: the image travels back base64-encoded."""
    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / ('artwork.' + IMAGE_FORMAT)
        meta = generate_artwork(data, dest)
        return {**meta, 'format': IMAGE_FORMAT, 'image': base64.b64encode(dest.read_bytes()).decode()}
