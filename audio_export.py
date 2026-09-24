"""Create reusable MP3 downloads without modifying the generated WAV."""
from pathlib import Path
import subprocess
import tempfile
import threading

from audio_edit import binary

_conversion_lock=threading.Lock()


def mp3_ready(wav):
    mp3=wav.with_suffix('.mp3')
    return mp3.is_file() and mp3.stat().st_size>0 and mp3.stat().st_mtime_ns>=wav.stat().st_mtime_ns


def to_mp3(wav):
    with _conversion_lock:
        target=wav.with_suffix('.mp3')
        if mp3_ready(wav):return target
        with tempfile.NamedTemporaryFile(dir=wav.parent,prefix='.mp3-',suffix='.mp3',delete=False) as f:
            temporary=Path(f.name)
        try:
            subprocess.run([binary('ffmpeg'),'-v','error','-nostdin','-y','-i',str(wav),
                            '-map','0:a:0','-vn','-map_metadata','-1','-codec:a','libmp3lame',
                            '-b:a','320k','-threads','1',str(temporary)],
                           capture_output=True,check=True,timeout=180)
            if not temporary.stat().st_size:raise ValueError('MP3 conversion produced an empty file.')
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        return target
