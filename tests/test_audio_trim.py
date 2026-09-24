import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock
import wave

import numpy as np
from fastapi.testclient import TestClient

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import audio_edit
from studio_fixture import studio_root, load_studio


def recording():
    data=io.BytesIO()
    with wave.open(data,'wb') as wav:
        wav.setnchannels(2);wav.setsampwidth(2);wav.setframerate(48000)
        # Four distinct one-second sections make wrong offsets detectable.
        samples=np.repeat(np.array([1000,12000,-9000,3000],dtype='<i2'),48000)
        wav.writeframes(np.column_stack([samples,-samples]).tobytes())
    return data.getvalue()


class AudioTrimTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.path=Path(self.tmp.name)/('a'*24+'.wav');self.path.write_bytes(recording())

    def test_cut_contains_only_selected_samples_and_preserves_original(self):
        original=hashlib.sha256(self.path.read_bytes()).hexdigest()
        result=audio_edit.trim(self.path,1.25,2.75)
        target=self.path.parent/result['id']
        decoded=audio_edit.subprocess.run([audio_edit.binary('ffmpeg'),'-v','error','-i',str(target),
            '-f','s16le','pipe:1'],capture_output=True,check=True).stdout
        samples=np.frombuffer(decoded,dtype='<i2').reshape(-1,2)
        self.assertEqual(samples.shape,(72000,2))
        np.testing.assert_array_equal(samples[:36000,0],12000)
        np.testing.assert_array_equal(samples[36000:,0],-9000)
        np.testing.assert_array_equal(samples[:,1],-samples[:,0])
        self.assertEqual(result['seconds'],1.5)
        self.assertEqual(original,hashlib.sha256(self.path.read_bytes()).hexdigest())
        self.assertEqual(audio_edit.trim(self.path,1.25,2.75)['id'],result['id'])
        self.assertNotEqual(audio_edit.trim(self.path,0,2)['id'],result['id'])
        self.assertEqual(audio_edit.trim(self.path,0,4)['id'],self.path.name)

    def test_waveform_and_invalid_boundaries(self):
        result=audio_edit.waveform(self.path)
        self.assertEqual(result['seconds'],4)
        self.assertEqual(len(result['peaks']),1000)
        for start,end in [(-1,3),(3,2),(1,1),(0,5),(0,.2),(float('nan'),3),(0,float('inf'))]:
            with self.subTest(start=start,end=end),self.assertRaises(ValueError):
                audio_edit.trim(self.path,start,end)
        self.assertEqual(list(self.path.parent.iterdir()),[self.path])


class TrimFlowTest(unittest.TestCase):
    def setUp(self):
        root=studio_root(self);self.server=load_studio(self,root,'trim_test_server')
        self.server.run=Mock();self.server.config=lambda:{'transcriber_ready':True}
        self.client=TestClient(self.server.app);self.client.__enter__()
        self.addCleanup(self.client.__exit__,None,None,None)
        self.headers={'X-Studio-Token':self.server.TOKEN}

    def test_upload_and_trim_do_not_start_analysis_and_selected_clip_is_analyzed(self):
        uploaded=self.client.post('/api/upload',headers=self.headers,files={'file':('test.wav',recording())})
        self.assertEqual(uploaded.status_code,200,uploaded.text)
        original=uploaded.json()['id']
        self.assertEqual(self.client.get('/api/uploads/'+original+'/waveform').status_code,200)
        response=self.client.post('/api/uploads/'+original+'/trim',json={'start':1,'end':3},headers=self.headers)
        self.assertEqual(response.status_code,200,response.text)
        clip=response.json()
        self.assertEqual(clip['seconds'],2)
        self.assertIsNone(self.server.active)
        self.server.run.assert_not_called()
        options={'audio_id':clip['id'],'melody_only':True,'source_seconds':0}
        analysis=self.client.post('/api/analysis',json=options,headers=self.headers)
        self.assertEqual(analysis.status_code,200,analysis.text)
        request=json.loads((self.server.LIB/analysis.json()['job']/'request.json').read_text())
        self.assertEqual(request['audio_id'],clip['id'])
        self.assertNotEqual(request['audio_id'],original)
        self.assertEqual(request['source_seconds'],0)

    def test_trim_rejects_invalid_paths_ranges_and_unauthorized_requests(self):
        name='a'*24+'.wav';(self.server.UP/name).write_bytes(recording())
        endpoint='/api/uploads/'+name+'/trim'
        self.assertEqual(self.client.post(endpoint,json={'start':0,'end':2}).status_code,403)
        for body in ({'start':-1,'end':2},{'start':2,'end':1},{'start':0,'end':100},{'start':'NaN','end':3}):
            self.assertIn(self.client.post(endpoint,json=body,headers=self.headers).status_code,[400,422])
        self.assertEqual(self.client.get('/api/uploads/invalid.wav/waveform').status_code,404)


if __name__=='__main__':unittest.main()
