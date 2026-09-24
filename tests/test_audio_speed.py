import io
import json
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import Mock

import numpy as np
from fastapi.testclient import TestClient

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import audio_speed
from studio_fixture import studio_root, load_studio


def tone(seconds=2,hz=440):
    t=np.arange(int(48000*seconds))/48000
    samples=(np.sin(2*np.pi*hz*t)*12000).astype('<i2')
    data=io.BytesIO()
    with wave.open(data,'wb') as wav:
        wav.setnchannels(2);wav.setsampwidth(2);wav.setframerate(48000)
        wav.writeframes(np.column_stack([samples,samples]).tobytes())
    return data.getvalue()


def analyse(path):
    with wave.open(str(path),'rb') as wav:
        rate=wav.getframerate();frames=np.frombuffer(wav.readframes(wav.getnframes()),dtype='<i2').reshape(-1,wav.getnchannels())[:,0]
    spectrum=np.abs(np.fft.rfft(frames.astype(float)));peak=np.fft.rfftfreq(len(frames),1/rate)[int(np.argmax(spectrum))]
    return len(frames)/rate,peak


def song_folder(library,kind='generate',audio=True):
    folder=Path(library)/'20260101-120000-0123abcd';folder.mkdir(parents=True)
    request={'kind':kind,'model':'bf16','title':'Return to Sender','style':'Rockabilly','lyrics':'[Verse]\nHello','instrumental':False,
             'voice':'male','cot':'full','seed':7,'candidates':1,'steps':2,'cfg_scale':1.2,'workspace_id':'default','audio_id':'a'*24+'.wav' if kind=='upload' else ''}
    (folder/'request.json').write_text(json.dumps(request))
    (folder/'result.json').write_text(json.dumps({'model':'bf16','seconds':2,'elapsed':30,'candidates':[{'index':1,'prefix':'','seed':7,'model':'bf16','title':'Return to Sender','seconds':2,'elapsed':30}]}))
    (folder/'state.json').write_text(json.dumps({'status':'complete','updated':0}))
    (folder/'artwork.webp').write_bytes(b'RIFF');(folder/'artwork.json').write_text('{"status":"complete"}')
    if audio:(folder/'audio.wav').write_bytes(tone())
    return folder


class SpeedFiltersTest(unittest.TestCase):
    def test_stages_stay_inside_atempo_sweet_spot(self):
        self.assertEqual(audio_speed.atempo_chain(1.5),'atempo=1.5')
        self.assertEqual(audio_speed.atempo_chain(4),'atempo=2,atempo=2')
        self.assertEqual(audio_speed.atempo_chain(3),'atempo=2,atempo=1.5')
        self.assertEqual(audio_speed.atempo_chain(.25),'atempo=0.5,atempo=0.5')
        self.assertEqual(audio_speed.atempo_chain(.3),'atempo=0.5,atempo=0.6')
        self.assertEqual(audio_speed.audio_filter(2,False),'aresample=48000,asetrate=96000,aresample=48000')

    def test_factor_limits(self):
        for bad in (0,.2,4.5,1,1.004,float('nan')):
            with self.subTest(bad=bad),self.assertRaises(ValueError):audio_speed.check_factor(bad)
        self.assertEqual(audio_speed.check_factor(1.256),1.26)


class SpeedRenderTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.source=self.root/'source.wav';self.source.write_bytes(tone())

    def test_keep_pitch_changes_length_only(self):
        seconds=audio_speed.render(self.source,self.root/'fast.wav',2,True)
        length,peak=analyse(self.root/'fast.wav')
        self.assertAlmostEqual(seconds,1,delta=.05);self.assertAlmostEqual(length,1,delta=.05);self.assertAlmostEqual(peak,440,delta=5)

    def test_tape_speed_moves_pitch_with_length(self):
        audio_speed.render(self.source,self.root/'fast.wav',2,False);audio_speed.render(self.source,self.root/'slow.wav',.5,False)
        length,peak=analyse(self.root/'fast.wav');self.assertAlmostEqual(length,1,delta=.02);self.assertAlmostEqual(peak,880,delta=5)
        length,peak=analyse(self.root/'slow.wav');self.assertAlmostEqual(length,4,delta=.02);self.assertAlmostEqual(peak,220,delta=5)
        self.assertEqual(sorted(p.name for p in self.root.iterdir()),['fast.wav','slow.wav','source.wav'])

    def test_generated_song_copy_keeps_settings_and_artwork(self):
        library=self.root/'library';uploads=self.root/'uploads';uploads.mkdir();folder=song_folder(library)
        job=audio_speed.create(library,uploads,folder,1,folder/'audio.wav',1.5,True)
        made=library/job;request=json.loads((made/'request.json').read_text());result=json.loads((made/'result.json').read_text())
        self.assertEqual(request['title'],'Return to Sender · 1.50x');self.assertEqual(request['style'],'Rockabilly');self.assertEqual(request['kind'],'generate')
        self.assertEqual(request['origin'],{'kind':'speed','job':folder.name,'candidate':1,'title':'Return to Sender','factor':1.5,'keep_pitch':True,'source_seconds':2})
        self.assertAlmostEqual(result['candidates'][0]['seconds'],2/1.5,delta=.05);self.assertNotIn('elapsed',result)
        self.assertTrue((made/'artwork.webp').is_file());self.assertTrue((made/'audio.wav').is_file());self.assertFalse(list(uploads.iterdir()))
        self.assertEqual(json.loads((made/'state.json').read_text())['status'],'complete')

    def test_uploaded_recording_copy_is_a_new_upload(self):
        library=self.root/'library';uploads=self.root/'uploads';uploads.mkdir();folder=song_folder(library,kind='upload')
        job=audio_speed.create(library,uploads,folder,1,folder/'audio.wav',.5,False)
        request=json.loads((library/job/'request.json').read_text())
        self.assertEqual(request['kind'],'upload');self.assertEqual(request['origin']['kind'],'speed');self.assertEqual(request['lyrics'],'[Verse]\nHello')
        self.assertTrue((uploads/request['audio_id']).is_file());self.assertNotEqual(request['audio_id'],'a'*24+'.wav')
        self.assertAlmostEqual(analyse(library/job/'audio.wav')[0],4,delta=.05)

    def test_failed_render_leaves_no_entry(self):
        library=self.root/'library';uploads=self.root/'uploads';uploads.mkdir();folder=song_folder(library)
        (folder/'audio.wav').write_bytes(b'not audio')
        with self.assertRaises(Exception):audio_speed.create(library,uploads,folder,1,folder/'audio.wav',2,True)
        self.assertEqual([p.name for p in library.iterdir()],[folder.name])


class SpeedFlowTest(unittest.TestCase):
    def setUp(self):
        root=studio_root(self);self.server=load_studio(self,root,'speed_test_server')
        self.server.run=Mock()
        self.client=TestClient(self.server.app);self.client.__enter__()
        self.addCleanup(self.client.__exit__,None,None,None)
        self.headers={'X-Studio-Token':self.server.TOKEN}

    def test_endpoint_lists_the_copy_and_validates_input(self):
        folder=song_folder(self.server.LIB)
        endpoint='/api/jobs/'+folder.name+'/songs/1/speed'
        self.assertEqual(self.client.post(endpoint,json={'factor':2}).status_code,403)
        for body in ({'factor':0.1},{'factor':5},{'factor':1},{'factor':'NaN'},{'factor':2,'extra':1}):
            self.assertIn(self.client.post(endpoint,json=body,headers=self.headers).status_code,[400,422],body)
        response=self.client.post(endpoint,json={'factor':2,'keep_pitch':False},headers=self.headers)
        self.assertEqual(response.status_code,200,response.text)
        made=response.json()
        self.assertEqual(made['title'],'Return to Sender · 2.00x');self.assertEqual(made['status'],'complete');self.assertEqual(made['request']['origin']['keep_pitch'],False)
        self.assertIn(made['id'],[j['id'] for j in self.client.get('/api/jobs',headers=self.headers).json()])
        self.assertEqual(self.client.post('/api/jobs/'+folder.name+'/songs/2/speed',json={'factor':2},headers=self.headers).status_code,404)


if __name__=='__main__':unittest.main()
