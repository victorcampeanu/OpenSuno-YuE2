import json,sys,unittest
from pathlib import Path
from unittest.mock import patch
from fastapi.testclient import TestClient
from test_composition_edit import SCORE
from studio_fixture import studio_root, load_studio
ROOT=Path(__file__).resolve().parents[1]
class EditAPITest(unittest.TestCase):
 def setUp(self):
  self.root=studio_root(self);self.s=load_studio(self,self.root,'edit_test_server')
  self.client=TestClient(self.s.app);self.headers={'X-Studio-Token':self.s.TOKEN};self.job='20260101-120000-abcdef12';p=self.root/'library'/self.job;p.mkdir()
  self.request=self.s.Job(style='Piano',lyrics='[Verse]\nOne two\n[Chorus]\nStay with me').model_dump()
  (p/'request.json').write_text(json.dumps(self.request));(p/'state.json').write_text(json.dumps({'status':'complete'}));(p/'score.abc').write_text(SCORE);(p/'audio.wav').write_bytes(b'audio');(p/'result.json').write_text(json.dumps({'candidates':[{'index':1,'prefix':''}]}))
  self.base='/api/jobs/'+self.job+'/songs/1';self.options={'style':'Jazz ensemble','lyrics':self.request['lyrics'],'bpm':110,'harmony':'free'}
 def test_prepare_and_generate_uses_saved_composition(self):
  r=self.client.post(self.base+'/edit',json=self.options,headers=self.headers);self.assertEqual(r.status_code,200,r.text);edit=r.json()
  self.assertEqual((self.root/'library'/self.job/'score.abc').read_text(),SCORE)
  with patch.object(self.s,'model_ready',return_value=True),patch.object(self.s,'run'):
   response=self.client.post('/api/jobs',json={**self.request,'edit_id':edit['id'],'style':'Orchestra'},headers=self.headers)
  self.assertEqual(response.status_code,200,response.text)
  generated=json.loads((self.root/'library'/response.json()['id']/'request.json').read_text())
  self.assertIn('Q:1/4=110',generated['abc']);self.assertEqual(generated['cot'],'melody');self.assertEqual(generated['style'],'Orchestra')
 def test_unknown_edits_and_invalid_structure_fail(self):
  r=self.client.post('/api/jobs',json={**self.request,'edit_id':'../x'},headers=self.headers);self.assertEqual(r.status_code,400)
  for data in ({**self.options,'sections':[{'id':'bad'}]},{**self.options,'bpm':0}):self.assertIn(self.client.post(self.base+'/edit',json=data,headers=self.headers).status_code,(400,422))
