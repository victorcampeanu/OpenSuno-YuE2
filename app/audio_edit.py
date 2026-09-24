"""Local waveform previews and non-destructive recording trims."""
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


def binary(name):
    path = shutil.which(name)
    if not path:
        installer = r'scripts\Install OpenSuno.ps1' if os.name == 'nt' else 'scripts/Install OpenSuno.command'
        raise ValueError(f'{name} is not installed. Run {installer}.')
    return path


def duration(path):
    result = subprocess.run([binary('ffprobe'), '-v', 'error', '-show_entries',
                             'format=duration', '-of', 'json', str(path)],
                            capture_output=True, text=True, check=True, timeout=15)
    seconds = float(json.loads(result.stdout)['format']['duration'])
    if not math.isfinite(seconds) or not 0 < seconds <= 1800:
        raise ValueError('Recordings must be between 0 and 30 minutes.')
    return seconds


def waveform(path):
    import numpy as np
    seconds = duration(path)
    result = subprocess.run([binary('ffmpeg'), '-v', 'error', '-nostdin', '-i', str(path),
                             '-map', '0:a:0', '-vn', '-ac', '1', '-ar', '2000',
                             '-f', 'f32le', 'pipe:1'], capture_output=True, check=True, timeout=60)
    samples = np.frombuffer(result.stdout, dtype='<f4')
    if not len(samples):
        raise ValueError('No audio was found in this recording.')
    peaks = [round(float(np.max(np.abs(chunk))), 4)
             for chunk in np.array_split(samples, min(1000, len(samples)))]
    return {'seconds': seconds, 'peaks': peaks}


def trim(path, start, end):
    """Return a reusable PCM clip, keeping the original file untouched."""
    seconds = duration(path)
    if not all(math.isfinite(v) for v in (start, end)) or not 0 <= start < end <= seconds + .001:
        raise ValueError('Choose a start and end inside the recording.')
    start, end = round(start, 3), min(round(end, 3), seconds)
    if end - start < min(1, seconds) - .001:
        raise ValueError('Select at least one second of audio.')
    if start == 0 and abs(end - seconds) <= .001:
        return {'id': path.name, 'seconds': seconds, 'start': 0, 'end': seconds,
                'bytes': path.stat().st_size}
    identity = json.dumps(['pcm-trim-v1', path.name, start, end]).encode()
    target = path.parent / (hashlib.sha256(identity).hexdigest()[:24] + '.wav')
    if not target.exists():
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.wav', delete=False) as f:
            temporary = Path(f.name)
        try:
            subprocess.run([binary('ffmpeg'), '-v', 'error', '-nostdin', '-y',
                            '-ss', str(start), '-i', str(path), '-t', str(end - start),
                            '-map', '0:a:0', '-vn', '-c:a', 'pcm_s24le', str(temporary)],
                           capture_output=True, check=True, timeout=90)
            actual = duration(temporary)
            if abs(actual - (end - start)) > .1:
                raise ValueError('The selected audio could not be cut accurately.')
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    return {'id': target.name, 'seconds': duration(target), 'start': start, 'end': end,
            'bytes': target.stat().st_size}
