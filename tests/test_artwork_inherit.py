import json,unittest
from unittest.mock import patch
from test_song_continuation import studio_server

class InheritedArtworkTest(unittest.TestCase):
 def setUp(self):
  self.p=studio_server(self,'artwork_inherit_server')
  (self.p/'artwork.webp').write_bytes(b'cover');(self.p/'artwork.json').write_text(json.dumps({'status':'complete','model':'gpt-image-2.5-flare'}))
 def submit(self,**extra):
  with patch.object(self.s,'model_ready',return_value=True),patch.object(self.s,'run'),patch.object(self.s,'start_artwork_job') as fresh:
   r=self.client.post('/api/jobs',json={**self.request,'kind':'generate',**extra},headers=self.headers)
  self.assertEqual(r.status_code,200,r.text);return self.root/'library'/r.json()['id'],fresh
 def assert_inherited(self,jobdir):
  self.assertEqual((jobdir/'artwork.webp').read_bytes(),b'cover')
  meta=json.loads((jobdir/'artwork.json').read_text());self.assertEqual(meta['status'],'complete');self.assertEqual(meta['from'],self.job)
  self.assertEqual(self.client.get('/api/jobs/'+jobdir.name).json()['artwork']['url'],'/api/jobs/'+jobdir.name+'/artwork')
 def test_rerender_keeps_the_source_cover(self):
  jobdir,fresh=self.submit(render_source=self.job,render_candidate=1,steps=8)
  self.assert_inherited(jobdir);fresh.assert_not_called()
 def test_extension_keeps_the_source_cover(self):
  jobdir,fresh=self.submit(continue_source=self.job,continue_candidate=1,continue_seconds=12.0)
  self.assert_inherited(jobdir);fresh.assert_not_called()
 def test_edited_composition_keeps_the_source_cover(self):
  r=self.client.post('/api/jobs/'+self.job+'/songs/1/edit',json={'style':'Piano','lyrics':self.request['lyrics'],'bpm':120},headers=self.headers)
  self.assertEqual(r.status_code,200,r.text)
  jobdir,fresh=self.submit(edit_id=r.json()['id'])
  self.assert_inherited(jobdir);fresh.assert_not_called()
 def test_new_songs_and_sources_without_a_cover_still_ask_for_one(self):
  jobdir,fresh=self.submit()
  self.assertFalse((jobdir/'artwork.webp').exists())
  (self.p/'artwork.webp').unlink();(self.p/'artwork.json').unlink()
  jobdir,fresh=self.submit(render_source=self.job,render_candidate=1,steps=8)
  self.assertFalse((jobdir/'artwork.json').exists(),'nothing is copied when the source has no picture; the usual request path decides')

if __name__=='__main__':unittest.main()
