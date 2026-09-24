"""Uploaded recordings: saving, checking and cutting them, on whichever machine has ffmpeg.

The Studio uses these directly when it renders on its own computer; a render node exposes them over
its API so an interface-only Studio never needs ffmpeg or numpy.
"""
import json
import re
import secrets
import shutil
import subprocess
from pathlib import Path

from fastapi import HTTPException

import audio_edit

EXTENSIONS = {'.wav', '.mp3', '.flac', '.m4a', '.aiff', '.aif', '.ogg', '.aac', '.npy'}
NAME = re.compile(r'[a-f0-9]{24}\.[a-z0-9]+')
EDITABLE = re.compile(r'[a-f0-9]{24}\.(wav|mp3|flac|m4a|aiff|aif|ogg|aac)')
MAX_BYTES = 200 * 1024 * 1024
MAX_LATENT_FRAMES = 18000  # 25 frames per second: 12 minutes


async def save(uploads, file):
    """Store a browser upload under a random name after checking it is audio (or YuE latents) of a sane length."""
    ext = Path(file.filename or '').suffix.lower()
    if ext not in EXTENSIONS:
        raise HTTPException(400, 'Use WAV, MP3, FLAC, M4A, AIFF, OGG, AAC or saved .npy latents')
    uploads = Path(uploads)
    name = secrets.token_hex(12) + ext
    path = uploads / name
    size = 0
    try:
        with path.open('wb') as out:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_BYTES:
                    raise HTTPException(413, 'Maximum upload size is 200 MB')
                out.write(chunk)
        if not size:
            raise HTTPException(400, 'The file is empty')
        if ext == '.npy':
            import numpy as np
            latents = np.load(path, allow_pickle=False, mmap_mode='r')
            if latents.ndim != 2 or latents.shape[1] != 64 or not 1 <= len(latents) <= MAX_LATENT_FRAMES or not np.isfinite(latents).all():
                raise ValueError('Invalid latents shape or values')
            seconds = len(latents) / 25
        else:
            seconds = probe_seconds(path)
        tags = embedded_tags(path) if ext != '.npy' else {}
        return {'id': name, 'name': file.filename, 'bytes': size, 'seconds': seconds, **tags, 'cover': False if ext == '.npy' else has_cover(path)}
    except HTTPException:
        path.unlink(missing_ok=True)
        raise
    except Exception as e:
        path.unlink(missing_ok=True)
        raise HTTPException(400, 'Could not read this file: ' + str(e)[:200])
    finally:
        await file.close()


def probe_seconds(path):
    probe = subprocess.run([audio_edit.binary('ffprobe'), '-v', 'error', '-show_entries', 'format=duration', '-of', 'json', str(path)],
                           capture_output=True, text=True, timeout=15, check=True)
    seconds = float(json.loads(probe.stdout)['format']['duration'])
    if not 0 < seconds <= 1800:
        raise ValueError('Recordings must be between 0 and 30 minutes')
    return seconds


LRC_STAMP = re.compile(r'\[\d{1,3}:\d{2}(?:[.:]\d{1,3})?\]')


def embedded_tags(path):
    """Title, artist and lyrics stored inside the file (ID3, MP4, Vorbis, FLAC), so the form can start filled in.

    Lyrics with LRC time stamps are returned twice: plain for the lyrics box and synced for aligning a cover.
    Nothing here is required: a file without tags gives empty strings.
    """
    if Path(path).suffix.lower() == '.npy':
        return {}
    try:
        probe = subprocess.run([audio_edit.binary('ffprobe'), '-v', 'error', '-show_entries', 'format_tags:stream_tags', '-of', 'json', str(path)],
                               capture_output=True, text=True, timeout=15, check=True)
        data = json.loads(probe.stdout)
        tags = {}
        for stream in data.get('streams', []):
            tags.update(stream.get('tags') or {})
        tags.update(data.get('format', {}).get('tags') or {})
    except Exception:
        return {}
    lowered = {key.lower(): str(value).strip() for key, value in tags.items() if str(value).strip()}

    def first(*names):
        for name in names:
            for key, value in lowered.items():
                if key == name or key.startswith(name + '-'):
                    return value
        return ''
    lyrics = first('lyrics', 'unsyncedlyrics', 'uslt', '©lyr', 'syncedlyrics').replace('\r\n', '\n').replace('\r', '\n')
    synced = lyrics if LRC_STAMP.search(lyrics) else ''
    plain = '\n'.join(LRC_STAMP.sub('', line).strip() for line in lyrics.split('\n')) if synced else lyrics
    plain = re.sub(r'\n{3,}', '\n\n', plain).strip()
    return {'title': first('title')[:100], 'artist': first('artist', 'album_artist')[:100], 'lyrics': plain[:20000], 'synced_lyrics': synced[:40000]}


PNG_MAGIC = b'\x89PNG\r\n\x1a\n'


def has_cover(path):
    """True when the file has a video/picture stream (ID3 attached pic, MP4 cover, FLAC picture)."""
    if Path(path).suffix.lower() == '.npy':
        return False
    try:
        probe = subprocess.run([audio_edit.binary('ffprobe'), '-v', 'error', '-select_streams', 'v',
                                 '-show_entries', 'stream=codec_type', '-of', 'json', str(path)],
                                capture_output=True, text=True, timeout=15, check=True)
        return bool(json.loads(probe.stdout).get('streams'))
    except Exception:
        return False


def cover_png(path):
    """Cached PNG of the first attached picture, extracted once. None if the recording has none."""
    source = Path(path)
    dest = source.with_name(source.name + '.cover.png')
    if dest.is_file():
        return dest
    if embedded_cover(source, dest):
        return dest
    return None


def embedded_cover(path, dest):
    """Pull the first attached picture (ID3, MP4, FLAC) into dest as PNG. Returns True if dest was written."""
    dest = Path(dest)
    if Path(path).suffix.lower() == '.npy':
        return False
    try:
        ffmpeg = audio_edit.binary('ffmpeg')
    except Exception:
        return False
    # ffmpeg's image muxer needs a real image suffix; .part / .tmp are rejected.
    tmp = dest.with_name(dest.stem + '.extract.png')
    try:
        subprocess.run([ffmpeg, '-v', 'error', '-nostdin', '-y', '-i', str(path), '-an', '-map', '0:v:0',
                         '-frames:v', '1', '-update', '1', '-c:v', 'png', str(tmp)],
                       capture_output=True, check=True, timeout=30)
        if tmp.is_file() and tmp.read_bytes()[:8] == PNG_MAGIC:
            tmp.replace(dest)
            return True
    except Exception:
        pass
    tmp.unlink(missing_ok=True)
    dest.unlink(missing_ok=True)
    return False


def editable(uploads, name):
    """The stored recording behind a browser's upload id, if it is audio (latents cannot be trimmed or drawn)."""
    if not EDITABLE.fullmatch(name) or not (Path(uploads) / name).is_file():
        raise HTTPException(404, 'Recording not found')
    return Path(uploads) / name


def waveform(path):
    try:
        return audio_edit.waveform(path)
    except Exception as e:
        raise HTTPException(400, 'Could not read the waveform. ' + (str(e) if isinstance(e, ValueError) else 'Try uploading the recording again.')) from e


def trim(path, start, end):
    try:
        return audio_edit.trim(path, start, end)
    except Exception as e:
        raise HTTPException(400, 'Could not trim the recording. ' + (str(e) if isinstance(e, ValueError) else 'Try again.')) from e


def decode_to_wav(source, target):
    """48 kHz stereo 16-bit PCM like generated songs, so every audio path in the Studio applies."""
    subprocess.run([audio_edit.binary('ffmpeg'), '-v', 'error', '-nostdin', '-y', '-i', str(source), '-map', '0:a:0', '-vn',
                    '-map_metadata', '-1', '-ac', '2', '-ar', '48000', '-c:a', 'pcm_s16le', str(target)],
                   capture_output=True, check=True, timeout=600)
    if not target.is_file() or not target.stat().st_size:
        raise ValueError('The recording could not be decoded.')
