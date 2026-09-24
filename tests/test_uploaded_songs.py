"""Uploaded recordings appear in the library as "Uploaded" entries that play, download, rename and delete."""
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import wave

import numpy as np
from fastapi.testclient import TestClient

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'app'))
import uploaded_songs
from studio_fixture import studio_root, load_studio


def recording(seconds=2,rate=44100):
    data=io.BytesIO()
    with wave.open(data,'wb') as wav:
        wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(rate)
        t=np.arange(rate*seconds)/rate
        wav.writeframes((np.sin(2*np.pi*440*t)*12000).astype('<i2').tobytes())
    return data.getvalue()


class UploadedSongsTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);(self.root/'library').mkdir()
        self.upload=self.root/('b'*24+'.wav');self.upload.write_bytes(recording())

    def test_entry_is_a_complete_one_version_song_decoded_to_studio_pcm(self):
        job=uploaded_songs.create(self.root/'library',self.upload,self.upload.name,'My demo.wav','default')
        folder=self.root/'library'/job
        request=json.loads((folder/'request.json').read_text());result=json.loads((folder/'result.json').read_text())
        self.assertEqual(request['kind'],'upload');self.assertEqual(request['model'],'upload');self.assertEqual(request['title'],'My demo')
        self.assertEqual(request['audio_id'],self.upload.name);self.assertEqual(request['workspace_id'],'default')
        self.assertEqual(json.loads((folder/'state.json').read_text())['status'],'complete')
        self.assertEqual(len(result['candidates']),1);self.assertEqual(result['candidates'][0]['title'],'My demo')
        self.assertAlmostEqual(result['seconds'],2.0,places=1)
        with wave.open(str(folder/'audio.wav')) as wav:
            self.assertEqual((wav.getnchannels(),wav.getframerate(),wav.getsampwidth()),(2,48000,2))
            self.assertAlmostEqual(wav.getnframes()/48000,2.0,places=1)
        self.assertTrue(self.upload.is_file(),'the original upload stays for covers and continuation')

    def test_embedded_lyrics_and_cover_are_kept_on_the_entry(self):
        import audio_edit, recordings
        try:
            ffmpeg=audio_edit.binary('ffmpeg')
        except Exception:
            self.skipTest('ffmpeg not available')
        picture=self.root/'cover.png'
        mp3=self.root/'tagged.mp3'
        if subprocess.run([ffmpeg,'-v','error','-y','-f','lavfi','-i','color=c=red:s=32x16:d=1','-frames:v','1',str(picture)],capture_output=True).returncode:
            self.skipTest('ffmpeg could not write a cover picture')
        done=subprocess.run([ffmpeg,'-v','error','-y','-f','lavfi','-i','sine=frequency=440:duration=1','-i',str(picture),
                             '-map','0:a','-map','1:v','-c:a','libmp3lame','-c:v','mjpeg','-disposition:v','attached_pic',
                             '-id3v2_version','3','-metadata','title=Return to Sender','-metadata','lyrics=Return to sender\nReturn to sender',
                             str(mp3)],capture_output=True)
        if done.returncode:
            self.skipTest('ffmpeg could not write a tagged mp3')
        upload=self.root/('a'*24+'.mp3'); shutil.copy2(mp3,upload)
        job=uploaded_songs.create(self.root/'library',upload,upload.name,'Return to Sender — Elvis Presley','default')
        folder=self.root/'library'/job
        request=json.loads((folder/'request.json').read_text())
        self.assertEqual(request['lyrics'],'Return to sender\nReturn to sender')
        self.assertTrue(request['embedded_applied'])
        self.assertTrue((folder/'artwork.png').is_file())
        self.assertTrue((folder/'artwork.png').read_bytes().startswith(recordings.PNG_MAGIC))
        self.assertEqual(json.loads((folder/'artwork.json').read_text())['source'],'embedded')

    def test_title_falls_back_and_is_bounded(self):
        self.assertEqual(uploaded_songs.clean_title('   '),'Uploaded recording')
        self.assertEqual(uploaded_songs.clean_title('a.b.c.mp3'),'a.b.c')
        self.assertEqual(len(uploaded_songs.clean_title('x'*300)),uploaded_songs.TITLE_LIMIT)

    def test_failed_decode_leaves_no_half_entry(self):
        broken=self.root/('c'*24+'.mp3');broken.write_bytes(b'not audio')
        with self.assertRaises(Exception):
            uploaded_songs.create(self.root/'library',broken,broken.name,'Broken')
        self.assertEqual(list((self.root/'library').iterdir()),[])


class UploadedSongAPITest(unittest.TestCase):
    def setUp(self):
        self.root=studio_root(self);self.s=load_studio(self,self.root,'uploaded_song_test_server')
        self.client=TestClient(self.s.app);self.headers={'X-Studio-Token':self.s.TOKEN}
        self.name='d'*24+'.wav';(self.root/'uploads'/self.name).write_bytes(recording())

    def test_upload_becomes_library_entry_that_plays_downloads_renames_and_deletes(self):
        r=self.client.post(f'/api/uploads/{self.name}/song',json={'title':'Take one.wav','workspace_id':'default'},headers=self.headers)
        self.assertEqual(r.status_code,200,r.text);job=r.json()
        self.assertEqual((job['kind'],job['status'],job['title']),('upload','complete','Take one'))
        self.assertIn('audio.wav',job['files']);self.assertEqual(job['result']['candidates'][0]['model'],'upload')
        listed=self.client.get('/api/jobs',headers=self.headers).json()
        self.assertEqual([j['id'] for j in listed],[job['id']])
        self.assertEqual(self.client.get(f"/api/files/{job['id']}/audio.wav",headers=self.headers).status_code,200)
        download=self.client.get(f"/api/jobs/{job['id']}/songs/1/download",headers=self.headers)
        self.assertEqual(download.status_code,200);self.assertIn('Take%20one.wav',download.headers['content-disposition'])
        self.assertEqual(self.client.patch(f"/api/jobs/{job['id']}/songs/1",json={'title':'Renamed'},headers=self.headers).status_code,200)
        self.assertEqual(self.client.get('/api/jobs',headers=self.headers).json()[0]['title'],'Renamed')
        self.assertEqual(self.client.delete(f"/api/jobs/{job['id']}/songs/1",headers=self.headers).status_code,200)
        self.assertEqual(self.client.get('/api/jobs',headers=self.headers).json(),[])
        self.assertTrue((self.root/'uploads'/self.name).is_file())

    def test_recording_timeline_has_waveform_no_bars_and_follows_the_analysis_length(self):
        full=self.client.get(f'/api/uploads/{self.name}/timeline',headers=self.headers).json()
        self.assertAlmostEqual(full['seconds'],2.0,places=1);self.assertEqual((full['bars'],full['sections'],full['cot']),([],[],'off'))
        self.assertTrue(full['peaks'])
        part=self.client.get(f'/api/uploads/{self.name}/timeline?source_seconds=1',headers=self.headers).json()
        self.assertAlmostEqual(part['seconds'],1.0,places=2);self.assertAlmostEqual(len(part['peaks'])/len(full['peaks']),0.5,places=1)
        self.assertEqual(self.client.get('/api/uploads/'+'e'*24+'.wav/timeline',headers=self.headers).status_code,404)

    def test_listed_upload_picks_up_lyrics_and_cover_from_the_original(self):
        import audio_edit, recordings
        try:
            ffmpeg=audio_edit.binary('ffmpeg')
        except Exception:
            self.skipTest('ffmpeg not available')
        picture=self.root/'cover.png'
        mp3=self.root/'tagged.mp3'
        if subprocess.run([ffmpeg,'-v','error','-y','-f','lavfi','-i','color=c=red:s=32x16:d=1','-frames:v','1',str(picture)],capture_output=True).returncode:
            self.skipTest('ffmpeg could not write a cover picture')
        done=subprocess.run([ffmpeg,'-v','error','-y','-f','lavfi','-i','sine=frequency=440:duration=1','-i',str(picture),
                             '-map','0:a','-map','1:v','-c:a','libmp3lame','-c:v','mjpeg','-disposition:v','attached_pic',
                             '-id3v2_version','3','-metadata','lyrics=I gave a letter to the postman',str(mp3)],capture_output=True)
        if done.returncode:
            self.skipTest('ffmpeg could not write a tagged mp3')
        name='f'*24+'.mp3'; shutil.copy2(mp3,self.root/'uploads'/name)
        job='20200101-000000-'+'a'*8
        folder=self.root/'library'/job; folder.mkdir()
        (folder/'request.json').write_text(json.dumps({'kind':'upload','model':'upload','title':'Return to Sender','audio_id':name,
            'workspace_id':'default','style':'','lyrics':'','instrumental':False,'voice':'any','cot':'off','seed':0,'candidates':1,'origin':None}))
        (folder/'result.json').write_text(json.dumps({'model':'upload','seconds':1,'elapsed':0,'uploaded':True,'candidates':[{'index':1,'prefix':'','seed':0,'model':'upload','title':'Return to Sender','seconds':1,'elapsed':0}]}))
        (folder/'state.json').write_text(json.dumps({'status':'complete','updated':1}))
        listed=self.client.get('/api/jobs',headers=self.headers).json()
        self.assertEqual(listed[0]['request']['lyrics'],'I gave a letter to the postman')
        self.assertTrue(listed[0]['request']['embedded_applied'])
        self.assertTrue(listed[0]['artwork']['url'].endswith('/artwork'))
        self.assertEqual(self.client.get('/api/jobs/'+job+'/artwork',headers=self.headers).status_code,200)

    def test_unknown_upload_and_workspace_are_rejected(self):
        self.assertEqual(self.client.post('/api/uploads/'+'e'*24+'.wav/song',json={},headers=self.headers).status_code,404)
        r=self.client.post(f'/api/uploads/{self.name}/song',json={'workspace_id':'nope'},headers=self.headers)
        self.assertEqual(r.status_code,400);self.assertEqual(list((self.root/'library').iterdir()),[])

    def test_cover_endpoint_serves_embedded_picture(self):
        self.assertEqual(self.client.get(f'/api/uploads/{self.name}/cover',headers=self.headers).status_code,404)
        import audio_edit, recordings
        try:
            ffmpeg=audio_edit.binary('ffmpeg')
        except Exception:
            self.skipTest('ffmpeg not available')
        picture=self.root/'cover.png';mp3=self.root/'tagged.mp3'
        if subprocess.run([ffmpeg,'-v','error','-y','-f','lavfi','-i','color=c=red:s=32x16:d=1','-frames:v','1',str(picture)],capture_output=True).returncode:
            self.skipTest('ffmpeg could not write a cover picture')
        done=subprocess.run([ffmpeg,'-v','error','-y','-f','lavfi','-i','sine=frequency=440:duration=1','-i',str(picture),
                             '-map','0:a','-map','1:v','-c:a','libmp3lame','-c:v','mjpeg','-disposition:v','attached_pic',
                             '-id3v2_version','3','-metadata','title=Hello',str(mp3)],capture_output=True)
        if done.returncode:
            self.skipTest('ffmpeg could not write a tagged mp3')
        name='c'*24+'.mp3';shutil.copy2(mp3,self.root/'uploads'/name)
        cover=self.client.get(f'/api/uploads/{name}/cover',headers=self.headers)
        self.assertEqual(cover.status_code,200,cover.text)
        self.assertTrue(cover.content.startswith(recordings.PNG_MAGIC))
        posted=self.client.post('/api/upload',headers=self.headers,files={'file':('hello.mp3',mp3.read_bytes(),'audio/mpeg')})
        self.assertEqual(posted.status_code,200,posted.text)
        self.assertTrue(posted.json()['cover'])
        wav=self.client.post('/api/upload',headers=self.headers,files={'file':('plain.wav',recording(),'audio/wav')})
        self.assertEqual(wav.status_code,200,wav.text)
        self.assertFalse(wav.json()['cover'])


if __name__=='__main__':
    unittest.main()
