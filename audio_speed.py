"""Faster or slower copies of a finished song, made from its audio the way Suno's Adjust speed works.

The result is a new library entry with the source's settings and artwork; nothing is generated again.
``keep_pitch`` time-stretches (ffmpeg ``atempo``), otherwise the song is resampled like a tape and the
pitch moves with the speed.
"""
import hashlib
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import audio_edit
import song_library
import uploaded_songs

MIN_FACTOR, MAX_FACTOR = 0.25, 4.0
RATE = 48000
TITLE_LIMIT = 100


def check_factor(factor):
    factor = float(factor)
    if not MIN_FACTOR <= factor <= MAX_FACTOR:
        raise ValueError(f'Choose a speed between {MIN_FACTOR:.2f}x and {MAX_FACTOR:.2f}x.')
    if abs(factor - 1) < .005:
        raise ValueError('Choose a speed other than 1.00x.')
    return round(factor, 2)


def atempo_chain(factor):
    """atempo sounds best between 0.5 and 2 per stage; larger changes are stacked."""
    stages = []
    while factor > 2 + 1e-9:
        stages.append(2.0); factor /= 2
    while factor < .5 - 1e-9:
        stages.append(.5); factor /= .5
    stages.append(factor)
    return ','.join(f'atempo={s:.6g}' for s in stages)


def audio_filter(factor, keep_pitch):
    if keep_pitch:
        return atempo_chain(factor)
    return f'aresample={RATE},asetrate={round(RATE * factor)},aresample={RATE}'


def label(factor):
    return f'{factor:.2f}x'


def render(source, target, factor, keep_pitch):
    """Write the stretched 48 kHz stereo PCM WAV and return its length in seconds."""
    factor = check_factor(factor)
    source, target = Path(source), Path(target)
    expected = audio_edit.duration(source) / factor
    if expected > 1800:
        raise ValueError('The slowed song would be longer than 30 minutes.')
    with tempfile.NamedTemporaryFile(dir=target.parent, suffix='.wav', delete=False) as f:
        temporary = Path(f.name)
    try:
        subprocess.run([audio_edit.binary('ffmpeg'), '-v', 'error', '-nostdin', '-y', '-i', str(source), '-map', '0:a:0', '-vn',
                        '-map_metadata', '-1', '-filter:a', audio_filter(factor, keep_pitch), '-ac', '2', '-ar', str(RATE),
                        '-c:a', 'pcm_s16le', str(temporary)], capture_output=True, check=True, timeout=900)
        seconds = audio_edit.duration(temporary)
        if abs(seconds - expected) > max(.5, expected * .02):
            raise ValueError('The speed change did not produce the expected length.')
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return round(seconds, 3)


def origin(source_folder, index, title, factor, keep_pitch, seconds):
    return {'kind': 'speed', 'job': Path(source_folder).name, 'candidate': index, 'title': title,
            'factor': factor, 'keep_pitch': bool(keep_pitch), 'source_seconds': seconds}


def copy_artwork(source_folder, folder):
    for name in ('artwork.webp', 'artwork.png', 'artwork.json'):
        if (source_folder / name).is_file():
            shutil.copyfile(source_folder / name, folder / name)


def create(library, uploads, source_folder, index, wav, factor, keep_pitch, workspace_id='default'):
    """A new library entry whose audio is version ``index`` of ``source_folder`` played at ``factor`` speed.

    Songs keep their kind, model, style and lyrics so they read like the original in the library; an
    uploaded recording becomes a new upload as well, so covers and continuation can use the stretched audio.
    """
    factor = check_factor(factor)
    source_folder, wav = Path(source_folder), Path(wav)
    request = json.loads((source_folder / 'request.json').read_text())
    _, candidate = song_library.locate(source_folder, index)
    source_title = (candidate or {}).get('title') or request.get('title') or 'Untitled song'
    title = (source_title + ' · ' + label(factor))[:TITLE_LIMIT]
    source_seconds = (candidate or {}).get('seconds')
    if request.get('kind') == uploaded_songs.KIND:
        identity = json.dumps(['speed-v1', request.get('audio_id'), source_folder.name, index, factor, bool(keep_pitch)]).encode()
        upload = Path(uploads) / (hashlib.sha256(identity).hexdigest()[:24] + '.wav')
        if not upload.is_file():
            render(wav, upload, factor, keep_pitch)
        job = uploaded_songs.create(library, upload, upload.name, title, workspace_id)
        folder = Path(library) / job
        entry = json.loads((folder / 'request.json').read_text())
        entry['lyrics'] = request.get('lyrics', '')
        entry['origin'] = origin(source_folder, index, source_title, factor, keep_pitch, source_seconds)
        uploaded_songs.write_request(folder, entry)
        copy_artwork(source_folder, folder)
        return job
    job = uploaded_songs.new_folder(library)
    folder = Path(library) / job
    try:
        seconds = render(wav, folder / 'audio.wav', factor, keep_pitch)
        entry = dict(request, title=title, candidates=1, created=time.time(), seed=(candidate or {}).get('seed', request.get('seed', 0)),
                     random_seed=False, edit_id='', render_source='', continue_source='', workspace_id=workspace_id,
                     origin=origin(source_folder, index, source_title, factor, keep_pitch, source_seconds))
        uploaded_songs.write_request(folder, entry)
        model = (candidate or {}).get('model') or request.get('model', 'bf16')
        # No ``elapsed``: nothing was generated, so the library shows no "Took" badge.
        result = {'model': model, 'seconds': seconds, 'seed': entry['seed'], 'speed': factor,
                  'candidates': [{'index': 1, 'prefix': '', 'seed': entry['seed'], 'model': model, 'title': title, 'seconds': seconds}]}
        (folder / 'result.json').write_text(json.dumps(result, indent=2))
        (folder / 'state.json').write_text(json.dumps({'status': 'complete', 'updated': time.time()}))
        copy_artwork(source_folder, folder)
    except Exception:
        shutil.rmtree(folder, ignore_errors=True)
        raise
    return job
