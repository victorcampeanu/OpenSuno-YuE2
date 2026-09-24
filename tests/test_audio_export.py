import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import unquote
import wave

import numpy as np
from fastapi.testclient import TestClient

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import audio_export
from audio_edit import binary
from studio_fixture import studio_root, load_studio


def wav_bytes(frequency=440):
    output=io.BytesIO()
    with wave.open(output,'wb') as wav:
        wav.setnchannels(2);wav.setsampwidth(2);wav.setframerate(48000)
        signal=(np.sin(np.arange(96000)*2*np.pi*frequency/48000)*12000).astype('<i2')
        wav.writeframes(np.column_stack([signal,signal]).tobytes())
    return output.getvalue()


class Mp3ConversionTest(unittest.TestCase):
    def test_real_conversion_preserves_wav_and_reuses_cached_mp3(self):
        with tempfile.TemporaryDirectory() as directory:
            wav=Path(directory)/'audio.wav';wav.write_bytes(wav_bytes())
            original=hashlib.sha256(wav.read_bytes()).hexdigest()
            with patch.object(audio_export.subprocess,'run',wraps=subprocess.run) as run:
                mp3=audio_export.to_mp3(wav)
                self.assertEqual(audio_export.to_mp3(wav),mp3)
                self.assertEqual(run.call_count,1)
            probe=json.loads(subprocess.check_output([binary('ffprobe'),'-v','error','-show_streams','-show_format','-of','json',str(mp3)]))
            self.assertEqual(probe['streams'][0]['codec_name'],'mp3')
            self.assertEqual(probe['streams'][0]['channels'],2)
            self.assertEqual(probe['streams'][0]['bit_rate'],'320000')
            self.assertAlmostEqual(float(probe['format']['duration']),2,delta=.1)
            self.assertEqual(hashlib.sha256(wav.read_bytes()).hexdigest(),original)

    def test_failed_conversion_never_exposes_partial_download(self):
        with tempfile.TemporaryDirectory() as directory:
            wav=Path(directory)/'audio.wav';wav.write_bytes(wav_bytes())
            with patch.object(audio_export.subprocess,'run',side_effect=subprocess.CalledProcessError(1,'ffmpeg')):
                with self.assertRaises(subprocess.CalledProcessError):audio_export.to_mp3(wav)
            self.assertEqual(list(Path(directory).iterdir()),[wav])


class SongDownloadTest(unittest.TestCase):
    def setUp(self):
        root=studio_root(self);self.server=load_studio(self,root,'export_test_server')
        self.client=TestClient(self.server.app);self.client.__enter__();self.addCleanup(self.client.__exit__,None,None,None)
        self.headers={'X-Studio-Token':self.server.TOKEN}
        self.job=self.server.LIB/'20260912-120000-aabbccdd';self.job.mkdir()
        (self.job/'request.json').write_text(json.dumps({'title':'Original'}))
        (self.job/'result.json').write_text(json.dumps({'candidates':[
            {'index':1,'prefix':'candidate-01/','title':'İstanbul / Remix'},
            {'index':2,'prefix':'candidate-02/','title':'Another version'}]}))
        for index in (1,2):
            p=self.job/f'candidate-{index:02d}';p.mkdir();(p/'audio.wav').write_bytes(wav_bytes(220*index))
        self.base='/api/jobs/'+self.job.name+'/songs/1'

    def test_downloads_selected_version_with_friendly_filename(self):
        response=self.client.get(self.base+'/download?format=wav')
        self.assertEqual(response.status_code,200)
        self.assertEqual(response.content,(self.job/'candidate-01/audio.wav').read_bytes())
        self.assertIn('İstanbul _ Remix.wav',unquote(response.headers['content-disposition']))
        self.assertEqual(self.client.get(self.base+'/download?format=mp3').status_code,404)
        export=self.client.post(self.base+'/export?format=mp3',headers=self.headers)
        self.assertEqual(export.status_code,200,export.text)
        downloaded=self.client.get(export.json()['url'])
        self.assertEqual(downloaded.status_code,200)
        self.assertEqual(downloaded.headers['content-type'],'audio/mpeg')
        self.assertIn('İstanbul _ Remix.mp3',unquote(downloaded.headers['content-disposition']))
        self.assertGreater(len(downloaded.content),1000)
        self.assertFalse((self.job/'candidate-02/audio.mp3').exists())
        self.server.song_library.rename(self.job,1,'New name')
        self.assertIn('New name.mp3',unquote(self.client.get(export.json()['url']).headers['content-disposition']))

    def test_export_validation_and_missing_audio(self):
        self.assertEqual(self.client.post(self.base+'/export?format=mp3').status_code,403)
        self.assertEqual(self.client.post(self.base+'/export?format=flac',headers=self.headers).status_code,422)
        self.assertEqual(self.client.get(self.base.replace('/songs/1','/songs/99')+'/download').status_code,404)
        (self.job/'candidate-01/audio.wav').unlink()
        self.assertEqual(self.client.post(self.base+'/export?format=mp3',headers=self.headers).status_code,404)


if __name__=='__main__':unittest.main()
