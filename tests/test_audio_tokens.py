"""Recording → music tokens: window arithmetic, the token cache, packages and the API."""
import json,unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import audio_tokens as at
from analysis_cache import AnalysisCache,TokenCache
from model_downloads import ModelDownloads
from test_song_continuation import studio_server,ROOT

class WindowTest(unittest.TestCase):
 def covered(self,total,window=at.WINDOW):
  plan=at.window_plan(total,window);written=np.zeros(total,dtype=int)
  for start,count,lo,hi in plan:
   self.assertGreaterEqual(start,0);self.assertLessEqual(start+count,total);self.assertLessEqual(count,window)
   self.assertGreaterEqual(lo,start);self.assertLessEqual(hi,start+count);written[lo:hi]+=1
  return plan,written
 def test_every_frame_is_predicted_once_from_a_window_with_context(self):
  for total in (1,100,511,512,513,700,1024,1025,1279,1280,3000,7501):
   plan,written=self.covered(total)
   self.assertTrue((written>=1).all(),total);self.assertEqual(plan[0][0],0)
   # Later windows overwrite the trimmed quarter of the previous one, so frames are written at most twice.
   self.assertLessEqual(written.max(),2,total)
 def test_short_recordings_use_one_window_without_trimming(self):
  self.assertEqual(at.window_plan(300),[(0,300,0,300)]);self.assertEqual(at.window_plan(512),[(0,512,0,512)])
 def test_half_window_stride_trims_a_quarter_on_inner_edges(self):
  plan=at.window_plan(1024)
  self.assertEqual(plan[0],(0,512,0,384));self.assertEqual(plan[1],(256,512,384,640));self.assertEqual(plan[-1],(512,512,640,1024))
 def test_chunks_are_thirty_seconds_and_drop_a_sub_second_tail(self):
  rate=at.SAMPLE_RATE
  self.assertEqual(at.chunk_bounds(75*rate),[(0,30*rate),(30*rate,60*rate),(60*rate,75*rate)])
  self.assertEqual(at.chunk_bounds(30*rate+rate//2),[(0,30*rate)])
  self.assertEqual(at.chunk_bounds(rate),[(0,rate)]);self.assertEqual(at.chunk_bounds(rate//2),[])
 def test_frames_run_at_twenty_five_per_second_and_features_are_normalised_per_recording(self):
  self.assertEqual(at.frame_count(10*at.SAMPLE_RATE),250);self.assertEqual(at.frame_count(at.SAMPLE_RATE//2),12)
  features=np.random.default_rng(0).normal(3,2,(400,8)).astype(np.float16)
  normal=at.normalize(features);self.assertEqual(normal.dtype,np.float32)
  np.testing.assert_allclose(normal.mean(0),0,atol=1e-3);np.testing.assert_allclose(normal.std(0),1,atol=1e-2)
 def test_availability_needs_the_head_and_the_stock_mert(self):
  import tempfile;from pathlib import Path
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);self.assertFalse(at.available(root))
   (root/'tokens').mkdir();(root/at.HEAD_FILE).write_bytes(b'x');self.assertFalse(at.available(root))
   (root/'mert').mkdir()
   for name in at.MERT_FILES:(root/'mert'/name).write_bytes(b'x')
   self.assertTrue(at.available(root))

class TokenCacheTest(unittest.TestCase):
 def test_tokens_and_analyses_of_one_recording_never_collide(self):
  import tempfile;from pathlib import Path
  with tempfile.TemporaryDirectory() as tmp:
   analyses=AnalysisCache(Path(tmp));tokens=TokenCache(Path(tmp))
   r=SimpleNamespace(audio_id='a'*24+'.wav',melody_only=True,source_seconds=0.0)
   self.assertNotEqual(analyses.key(r),tokens.key(r))
   tokens.save(r,{'codec':[1,2,3],'audio_id':r.audio_id,'source_seconds':0.0,'seconds':0.12})
   self.assertEqual(tokens.read(r)['codec'],[1,2,3]);self.assertIsNone(analyses.read(r))
   self.assertIsNone(tokens.read(SimpleNamespace(audio_id=r.audio_id,source_seconds=30.0)))
   tokens.save(r,{'codec':[],'audio_id':r.audio_id,'source_seconds':0.0});self.assertIsNone(tokens.read(r),'empty tokens are not a result')
   tokens.remember(r,'job-1');self.assertEqual(tokens.job(r),'job-1')

class PackageTest(unittest.TestCase):
 def test_the_tokenizer_head_and_decoder_adapter_form_the_audio_input_package(self):
  d=ModelDownloads(ROOT)
  files=[a['path'] for a in d.selected_assets('tokens')];self.assertEqual(files,[at.HEAD_FILE,'tokens/nar_lora_joint_v9.safetensors'])
  self.assertNotIn(at.HEAD_FILE,[a['path'] for a in d.selected_assets('analysis')],'existing Cover analysis installs stay complete')
  self.assertIn('mert/model.safetensors',[a['path'] for a in d.selected_assets('analysis')])
  for asset in (a for a in d.assets if a['path'].startswith('tokens/')):
   self.assertEqual(asset['repo'],'Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4');self.assertEqual(len(asset['sha256']),64)
  snap=d.snapshot()['packages']['tokens']
  self.assertIn('tokens',d.snapshot()['packages'])
  self.assertEqual(snap['bytes'],sum(a['size'] for a in d.selected_assets('tokens')))
  self.assertGreater(snap['bytes'],0)

CODEC=[(i*37)%32768 for i in range(25*40)]  # 40 seconds of fake tokens

class RecordingTokensApiTest(unittest.TestCase):
 def setUp(self):
  studio_server(self,'audio_tokens_test_server')
  self.audio='c'*24+'.mp3';(self.s.UP/self.audio).write_bytes(b'fixture')
  self.options={'audio_id':self.audio,'source_seconds':0}
  self.s.run=lambda p,r:None;self.s.tokens_ready=lambda:True
 def post(self,path,body):return self.client.post(path,json=body,headers=self.headers)
 def get(self,path,params):return self.client.get(path,params=params,headers=self.headers)
 def tokenized(self):
  self.s.recording_tokens.save(self.s.Job(kind='tokenize',**self.options),{**self.options,'codec':CODEC,'seconds':40.0,'tokenizer':at.NAME,'elapsed':3})
 def test_tokenize_jobs_run_in_the_analysis_environment_and_stay_out_of_the_library(self):
  self.assertEqual(self.get('/api/tokens',self.options).json(),{'status':'missing'})
  r=self.post('/api/tokens',self.options);self.assertEqual(r.status_code,200,r.text);job=r.json()['job']
  request=json.loads((self.s.LIB/job/'request.json').read_text())
  self.assertEqual(request['kind'],'tokenize');self.assertTrue(request['background_analysis']);self.assertEqual(request['audio_id'],self.audio)
  self.assertNotIn(job,[j['id'] for j in self.client.get('/api/jobs',headers=self.headers).json()])
  self.assertEqual(self.get('/api/tokens',self.options).json()['job'],job)
  self.assertEqual(self.post('/api/tokens',self.options).json()['job'],job,'a running job is reused')
  self.assertIn(self.s.Job(kind='tokenize',**self.options).kind,self.s.ANALYSIS_KINDS)
  with self.assertRaises(ValueError):self.s.Job(kind='tokenize')
 def test_tokenizing_needs_the_installed_head(self):
  self.s.tokens_ready=lambda:False
  self.assertEqual(self.post('/api/tokens',self.options).status_code,503)
  self.assertEqual(self.post('/api/tokens',{'audio_id':'z'*24+'.mp3'}).status_code,400)
 def test_cached_tokens_are_reported_ready_and_not_recomputed(self):
  self.tokenized()
  self.assertEqual(self.get('/api/tokens',self.options).json(),{'status':'ready','seconds':40.0,'frames':1000})
  self.assertEqual(self.post('/api/tokens',self.options).json()['status'],'ready','cached tokens are not recomputed')

if __name__=='__main__':unittest.main()
