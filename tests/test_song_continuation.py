import json,sys,unittest
from pathlib import Path
from unittest.mock import patch
from fastapi.testclient import TestClient
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'app'))
import song_continuation as sc
from vendor import yue2_abc as abc
from studio_fixture import studio_root, load_studio

HEADER='X:1\nT:\nM:4/4\nL:1/16\nQ:1/4=120\nV: Vocal clef=treble name="Vocal Melody" snm="Vocal"\nV: Ins clef=treble name="Ins Melody" snm="Inst."\nK:C\n'
def group(label,vocal,ins):return (f'% {label}\n' if label else '')+'V: Vocal\n'+vocal+'\nV: Ins\n'+ins+'\n'
BAR='C4D4E4F4|'
INTRO=group('intro','Z2|','C4D4E4F4|G4A4B4c4|')                       # bars 1-2: 0-4 s at 120 BPM
VERSE=group('verse',BAR*4,'Z4|')                                        # bars 3-6: 4-12 s
CHORUS=group('chorus','C2D2E2F2G2A2B2c2|'*2+'A8G8-|G16|','Z4|')         # bars 7-10: 12-20 s, tie across the last barline
SCORE=HEADER+INTRO+VERSE+CHORUS                                         # 10 bars, 20 s nominal
CODEC=[i%1000 for i in range(20*25)]                                    # 20 s of frames

def studio_server(test,name):
 test.root=studio_root(test);test.s=load_studio(test,test.root,name)
 test.client=TestClient(test.s.app);test.headers={'X-Studio-Token':test.s.TOKEN}
 test.job='20260101-120000-abcdef12';p=test.root/'library'/test.job;(p/'candidate-01').mkdir(parents=True)
 test.request=test.s.Job(style='Piano',lyrics='[Verse]\nOne two\n\n[Chorus]\nStay with me',cot='full',model='cuda-bf16').model_dump()
 (p/'request.json').write_text(json.dumps(test.request));(p/'state.json').write_text(json.dumps({'status':'complete'}))
 (p/'result.json').write_text(json.dumps({'candidates':[{'index':1,'prefix':'candidate-01/','seed':7,'title':'Source song','seconds':20.0}]}))
 c=p/'candidate-01';(c/'score.abc').write_text(SCORE);(c/'tokens.json').write_text(json.dumps(CODEC));(c/'audio.wav').write_bytes(b'audio')
 (c/'settings.json').write_text(json.dumps({**test.request,'title':'Source song'}))
 return p

class CutTest(unittest.TestCase):
 def test_timeline_lists_bars_and_sections(self):
  t=sc.timeline(SCORE,20.0)
  self.assertEqual(t['bpm'],120);self.assertEqual(len(t['bars']),10);self.assertEqual(t['bars'][:3],[0.0,2.0,4.0])
  self.assertEqual([(s['tag'],s['start'],s['end'],s['sung']) for s in t['sections']],[('Intro',0.0,4.0,False),('Verse',4.0,12.0,True),('Chorus',12.0,20.0,True)])
  self.assertEqual(sc.timeline('',20.0)['bars'],[])
  self.assertEqual(sc.timeline('not a score',20.0)['sections'],[])
 def test_timeline_scales_nominal_times_onto_the_audio(self):
  t=sc.timeline(SCORE,22.0);self.assertAlmostEqual(t['scale'],1.1);self.assertAlmostEqual(t['bars'][1],2.2)
  self.assertEqual(sc.timeline(SCORE,60.0)['scale'],1.0,'implausible drift is ignored')
 def test_cut_at_a_section_boundary_keeps_whole_groups(self):
  text,kept,at=sc.cut_score(SCORE,12.0)
  self.assertEqual((kept,at),(6,12.0))
  self.assertEqual(text,HEADER+INTRO+VERSE)
  self.assertEqual(len(abc.parse(text).voices['Vocal'].bars),6)
 def test_cut_inside_a_section_truncates_both_voices(self):
  text,kept,at=sc.cut_score(SCORE,7.0)
  self.assertEqual((kept,at),(3,6.0),'the bar that starts at or before the time is regenerated')
  self.assertTrue(text.endswith('% verse\nV: Vocal\nC4D4E4F4|\nV: Ins\nZ|\n'),text)
  abc.parse(text)
 def test_cut_before_a_tied_bar_closes_the_tie(self):
  text,kept,_=sc.cut_score(SCORE,18.0)
  self.assertEqual(kept,9);self.assertTrue(text.endswith('A8G8|\nV: Ins\nZ3|\n'),text)
  abc.parse(text)
 def test_cut_at_the_end_or_beyond_keeps_the_whole_score(self):
  for t in (20.0,25.0):
   text,kept,at=sc.cut_score(SCORE,t);self.assertEqual((kept,at),(10,20.0));self.assertEqual(text,SCORE)
  self.assertEqual(sc.cut_score(SCORE,19.0)[1:],(9,18.0),'inside the last bar regenerates that bar')
 def test_cut_at_zero_keeps_one_bar(self):
  text,kept,_=sc.cut_score(SCORE,0);self.assertEqual(kept,1);abc.parse(text)
 def test_codec_cut_rounds_to_frames(self):
  self.assertEqual(len(sc.cut_codec(CODEC,12.0)),300);self.assertEqual(len(sc.cut_codec(CODEC,12.06)),302)
  self.assertEqual(len(sc.cut_codec(CODEC,0)),1);self.assertEqual(len(sc.cut_codec(CODEC,99)),len(CODEC))
 def test_budget_fits_the_context(self):
  s={'max_tokens':9000,'min_tokens':200}
  self.assertEqual(sc.budget(s,1000)['max_tokens'],9000)
  self.assertEqual(sc.budget(s,20000,20500)['max_tokens'],sc.CONTEXT-20500)
  self.assertEqual(sc.budget({'max_tokens':9000,'min_tokens':5000},20000)['min_tokens'],4576)
  with self.assertRaises(ValueError):sc.budget(s,sc.CONTEXT-10)

class ContinuationAPITest(unittest.TestCase):
 def setUp(self):self.p=studio_server(self,'continuation_test_server')
 def body(self,**extra):
  return {**self.request,'kind':'generate','model':'cuda-bf16','continue_source':self.job,'continue_candidate':1,**extra}
 def test_continue_input_is_cut_at_a_bar(self):
  with patch.object(self.s,'model_ready',return_value=True),patch.object(self.s,'run'):
   r=self.client.post('/api/jobs',json=self.body(continue_seconds=7.0,cot='off',lyrics='[Verse]\nOne two\n\n[Chorus]\nNew words'),headers=self.headers)
  self.assertEqual(r.status_code,200,r.text);jobdir=self.root/'library'/r.json()['id']
  saved=json.loads((jobdir/'continue-input.json').read_text())
  self.assertEqual(saved['kept_bars'],3);self.assertEqual(len(saved['codec']),150);self.assertEqual(saved['cut_seconds'],6.0)
  self.assertEqual(saved['codec'],CODEC[:150]);self.assertTrue(saved['abc_prefix'].endswith('|\n'));self.assertNotIn('chorus',saved['abc_prefix'])
  request=json.loads((jobdir/'request.json').read_text())
  self.assertEqual(request['cot'],'full','the plan mode comes from the source, not the form');self.assertEqual(request['abc'],'')
  self.assertEqual(request['origin']['kind'],'continue');self.assertEqual(request['origin']['job'],self.job);self.assertEqual(request['origin']['cut_seconds'],6.0)
  self.assertEqual(request['lyrics'],'[Verse]\nOne two\n\n[Chorus]\nNew words')
 def test_continue_from_the_end_keeps_everything(self):
  with patch.object(self.s,'model_ready',return_value=True),patch.object(self.s,'run'):
   r=self.client.post('/api/jobs',json=self.body(continue_seconds=None),headers=self.headers)
  self.assertEqual(r.status_code,200,r.text)
  saved=json.loads((self.root/'library'/r.json()['id']/'continue-input.json').read_text())
  self.assertEqual(len(saved['codec']),len(CODEC));self.assertEqual(saved['kept_bars'],10);self.assertEqual(saved['abc_prefix'],SCORE)
 def test_continuation_runs_on_any_engine_and_needs_saved_tokens(self):
  with patch.object(self.s,'model_ready',return_value=True),patch.object(self.s,'run'):
   self.assertEqual(self.client.post('/api/jobs',json=self.body(model='bf16'),headers=self.headers).status_code,200)
   # Re-rendering an extended song sends its saved settings along: the re-render wins and no continuation is made.
   again=self.client.post('/api/jobs',json=self.body(render_source=self.job,render_candidate=1),headers=self.headers)
   self.assertEqual(again.status_code,200,again.text);self.assertFalse((self.root/'library'/again.json()['id']/'continue-input.json').exists())
   self.assertEqual(self.client.post('/api/jobs',json=self.body(continue_candidate=2),headers=self.headers).status_code,400)
   (self.p/'candidate-01'/'tokens.json').unlink()
   r=self.client.post('/api/jobs',json=self.body(),headers=self.headers)
  self.assertEqual(r.status_code,400);self.assertIn('no saved music tokens',r.json()['detail'])
 def test_rerender_of_an_extended_song_does_not_continue_again(self):
  with patch.object(self.s,'model_ready',return_value=True),patch.object(self.s,'run'):
   first=self.client.post('/api/jobs',json=self.body(continue_seconds=7.0),headers=self.headers).json()['id']
   p=self.root/'library'/first;(p/'state.json').write_text(json.dumps({'status':'complete'}))
   (p/'result.json').write_text(json.dumps({'candidates':[{'index':1,'prefix':'','seed':7}]}));(p/'tokens.json').write_text(json.dumps(CODEC[:10]))
   r=self.client.post('/api/jobs',json={**self.request,'render_source':first,'render_candidate':1,'steps':8},headers=self.headers)
  self.assertEqual(r.status_code,200,r.text);again=self.root/'library'/r.json()['id']
  self.assertFalse((again/'continue-input.json').exists());self.assertTrue((again/'render-input.json').exists())
 def test_timeline_route(self):
  r=self.client.get('/api/jobs/'+self.job+'/songs/1/timeline');self.assertEqual(r.status_code,200,r.text);t=r.json()
  self.assertEqual(len(t['bars']),10);self.assertEqual(t['title'],'Source song');self.assertEqual(t['cot'],'full');self.assertEqual(t['seconds'],20.0)
  self.assertIn('peaks',t)
  self.assertEqual(self.client.get('/api/jobs/'+self.job+'/songs/2/timeline').status_code,400)
