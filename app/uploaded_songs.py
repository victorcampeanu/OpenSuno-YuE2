"""Uploaded recordings as library entries, like any other song, badged "Uploaded" instead of a model name.

An entry is a job folder with no generation behind it: ``request.json`` (``kind: upload``), a
complete ``state.json``, a one-version ``result.json`` and ``audio.wav`` (the recording decoded to
the studio's 48 kHz stereo PCM), so playback, downloads, favorites, workspaces, rename and delete
work unchanged. The original upload stays in ``uploads/`` for covers and continuation.

With a render node the decoding happens there; the entry then has no local ``audio.wav`` and the
Studio streams or fetches it like a rendered song (``remote.json``).
"""
import json
import re
import secrets
import time
from pathlib import Path

from audio_edit import duration
from recordings import decode_to_wav, embedded_cover, embedded_tags

KIND = 'upload'
MODEL = 'upload'
TITLE_LIMIT = 100


def clean_title(name, fallback='Uploaded recording'):
    title = re.sub(r'\s+', ' ', re.sub(r'\.[^.]+$', '', str(name or ''))).strip()[:TITLE_LIMIT]
    return title or fallback


def write_request(folder, request):
    tmp = Path(folder) / 'request.tmp'
    tmp.write_text(json.dumps(request, indent=2))
    tmp.replace(Path(folder) / 'request.json')


def write_entry(folder, audio_id, title, seconds, workspace_id='default', lyrics=''):
    title = clean_title(title)
    request = {'kind': KIND, 'model': MODEL, 'title': title, 'audio_id': audio_id, 'workspace_id': workspace_id,
               'style': '', 'lyrics': (lyrics or '').strip()[:20000], 'instrumental': False, 'voice': 'any', 'cot': 'off',
               'seed': 0, 'candidates': 1, 'origin': None, 'created': time.time(), 'embedded_applied': True}
    write_request(folder, request)
    result = {'model': MODEL, 'seconds': seconds, 'elapsed': 0, 'uploaded': True,
              'candidates': [{'index': 1, 'prefix': '', 'seed': 0, 'model': MODEL, 'title': title, 'seconds': seconds, 'elapsed': 0}]}
    (folder / 'result.json').write_text(json.dumps(result, indent=2))
    (folder / 'state.json').write_text(json.dumps({'status': 'complete', 'updated': time.time()}))


def new_folder(library):
    library = Path(library)
    job = time.strftime('%Y%m%d-%H%M%S') + '-' + secrets.token_hex(4)
    (library / job).mkdir()
    return job


def discard(folder):
    for child in folder.iterdir():
        child.unlink()
    folder.rmdir()


def has_artwork(folder):
    folder = Path(folder)
    return any((folder / name).is_file() for name in ('artwork.webp', 'artwork.png'))


def write_artwork_meta(folder, present):
    data = {'status': 'complete' if present else 'none', 'source': 'embedded'}
    tmp = Path(folder) / 'artwork.tmp'
    tmp.write_text(json.dumps(data))
    tmp.replace(Path(folder) / 'artwork.json')


def attach_cover(folder, source):
    """Copy a picture stored inside the recording into the job folder so the library can show it."""
    folder = Path(folder)
    if has_artwork(folder):
        return True
    if embedded_cover(source, folder / 'artwork.png'):
        write_artwork_meta(folder, True)
        return True
    write_artwork_meta(folder, False)
    return False


def apply_lyrics(folder, lyrics):
    """Store tagged lyrics on an upload that was listed before they were kept. Leaves existing words alone."""
    lyrics = (lyrics or '').strip()[:20000]
    if not lyrics:
        return False
    request = json.loads((Path(folder) / 'request.json').read_text())
    if (request.get('lyrics') or '').strip():
        return False
    request['lyrics'] = lyrics
    request['instrumental'] = False
    write_request(folder, request)
    return True


def mark_embedded(folder):
    request = json.loads((Path(folder) / 'request.json').read_text())
    if request.get('embedded_applied'):
        return
    request['embedded_applied'] = True
    write_request(folder, request)


def create(library, upload, audio_id, title, workspace_id='default'):
    """Create the library entry for an uploaded recording on this computer; returns its job id."""
    seconds = round(duration(upload), 3)
    job = new_folder(library)
    folder = Path(library) / job
    try:
        decode_to_wav(upload, folder / 'audio.wav')
        write_entry(folder, audio_id, title, seconds, workspace_id, lyrics=embedded_tags(upload).get('lyrics', ''))
        attach_cover(folder, upload)
    except Exception:
        discard(folder)
        raise
    return job


def create_remote(library, audio_id, title, seconds, workspace_id='default', lyrics=''):
    """The entry for a recording a render node decoded; the caller records where the audio lives."""
    job = new_folder(library)
    folder = Path(library) / job
    try:
        write_entry(folder, audio_id, title, round(float(seconds), 3), workspace_id, lyrics=lyrics)
    except Exception:
        discard(folder)
        raise
    return job
