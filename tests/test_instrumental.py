import json,sys,unittest
from pathlib import Path
from unittest.mock import patch
from fastapi.testclient import TestClient
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import instrumental
from vendor import yue2_abc as abc
from studio_fixture import studio_root, load_studio

HEADER='X:1\nT:\nM:4/4\nL:1/16\nQ:1/4=100\nV: Vocal clef=treble name="Vocal Melody" snm="Vocal"\nV: Ins clef=treble name="Ins Melody" snm="Inst."\nK:C\n'
def group(label,vocal,ins):return (f'% {label}\n' if label else '')+'V: Vocal\n'+vocal+'\nV: Ins\n'+ins+'\n'
INTRO=group('intro','"C"z16|Z|','C4D4E4F4|G4A4B4c4-|')       # instrument phrase tied into the verse
VERSE=group('verse','"C"C4D4E4F4|"G"G4A4B4c4|','c16|Z|')
CHORUS=group('chorus','"Am"A8G8-|G16|','C8E8|G8E8|')          # sung tie across a barline
SCORE=HEADER+INTRO+VERSE+CHORUS

class InstrumentalScoreTest(unittest.TestCase):
 def test_vocal_is_silenced_and_melody_moves_to_the_instrument(self):
  before=abc.parse(SCORE);after_text,moved,_=instrumental.instrumental_score(SCORE);after=abc.parse(after_text)
  self.assertEqual(after.voices['Vocal'].notes,[])
  self.assertEqual(moved,4)
  self.assertEqual(after.voices['Vocal'].bars,before.voices['Vocal'].bars)
  self.assertEqual(after.voices['Vocal'].chords,before.voices['Vocal'].chords,'chord symbols stay on the silent vocal voice')
  self.assertEqual(after.voices['Ins'].chords,[])
  intro_notes=[n[:2] for n in before.voices['Ins'].notes if n[0]<8]
  self.assertEqual([n[:2] for n in after.voices['Ins'].notes if n[0]<8],intro_notes,'an already silent group keeps its instrument part')
  self.assertEqual([n[1:] for n in after.voices['Ins'].notes if 8<=n[0]<16],[n[1:] for n in before.voices['Vocal'].notes if 8<=n[0]<16],'the verse melody is now played')
  self.assertEqual(len([n for n in after.voices['Ins'].notes if n[0]>=16]),2,'the tied chorus note stays one sounding note')
 def test_tie_between_kept_and_moved_groups_is_closed(self):
  after_text,_,_=instrumental.instrumental_score(SCORE)
  lines=after_text.splitlines()
  self.assertIn('C4D4E4F4|G4A4B4c4|',lines,'the instrument phrase no longer ties into the moved verse melody')
 def test_idempotent(self):
  once,_,_=instrumental.instrumental_score(SCORE);twice,moved,_=instrumental.instrumental_score(once)
  self.assertEqual(once,twice);self.assertEqual(moved,0)
 def test_score_cut_short_by_the_token_cap_loses_only_the_partial_group(self):
  truncated=SCORE+'% outro\nV: Vocal\n"C"z16|"C"'
  out,moved,_=instrumental.instrumental_score(truncated)
  whole,_,_=instrumental.instrumental_score(SCORE)
  self.assertEqual(out,whole);self.assertEqual(moved,4)
 def test_library_style_rest_lines_are_preserved(self):
  text=HEADER+group('verse','Z3|C4D4E4F4|','Z4|')
  out,moved,_=instrumental.instrumental_score(text)
  self.assertEqual(moved,4);self.assertEqual(abc.parse(out).voices['Vocal'].notes,[])
 def test_hook_only_keeps_the_melody_in_the_chorus_and_frees_the_verses(self):
  before=abc.parse(SCORE);out,moved,freed=instrumental.instrumental_score(SCORE,hook_only=True);after=abc.parse(out)
  self.assertEqual((moved,freed),(2,2));self.assertEqual(after.voices['Vocal'].notes,[],'the vocal voice is silent everywhere')
  self.assertEqual(after.voices['Vocal'].chords,before.voices['Vocal'].chords,'the verse keeps its chords for the model to arrange on')
  self.assertEqual([n[1:] for n in after.voices['Ins'].notes if 8<=n[0]<16],[n[1:] for n in before.voices['Ins'].notes if 8<=n[0]<16],'the verse instrument line is left as it was, not given the melody')
  self.assertEqual([n[1:] for n in after.voices['Ins'].notes if n[0]>=16],[n[1:] for n in before.voices['Vocal'].notes if n[0]>=16],'the chorus melody is played')
  self.assertEqual(after.voices['Vocal'].bars,before.voices['Vocal'].bars)
  self.assertTrue(instrumental.is_hook('Chorus') and instrumental.is_hook('hook') and instrumental.is_hook('Refrain'))
  self.assertFalse(instrumental.is_hook('Pre-Chorus') or instrumental.is_hook('verse') or instrumental.is_hook(''))
 def test_hook_only_without_section_labels_plays_the_melody_everywhere(self):
  text=HEADER+group('','"C"C4D4E4F4|"G"G4A4B4c4|','Z2|')+group('','"Am"A8G8|G16|','Z2|')
  out,moved,freed=instrumental.instrumental_score(text,hook_only=True)
  self.assertEqual((moved,freed),(4,0));self.assertEqual(len(abc.parse(out).voices['Ins'].notes),len(abc.parse(text).voices['Vocal'].notes))

class StructureTest(unittest.TestCase):
 def test_only_section_tags_survive(self):
  self.assertEqual(instrumental.structure_only('[Verse]\nla la la\n\n[Chorus]\nwoo\n[Outro]'),'[Verse]\n\n[Chorus]\n\n[Outro]\n')
 def test_default_structure_without_tags(self):
  self.assertEqual(instrumental.structure_only('just some words'),instrumental.DEFAULT_STRUCTURE)
  self.assertEqual(instrumental.structure_only(''),instrumental.DEFAULT_STRUCTURE)
  self.assertTrue(instrumental.DEFAULT_STRUCTURE.startswith('[Intro]\n\n[Verse]'))
 def test_an_instrumental_cover_takes_its_sections_from_the_score(self):
  abc='X:1\nK:C\n% intro\nV:Vocal\nz4|\n% verse\nV:Vocal\nCDEF|\n% pre-chorus\nV:Vocal\nGABc|\n% chorus\nV:Vocal\ncBAG|\n'
  self.assertEqual(instrumental.score_sections(abc),['Intro','Verse','Pre-Chorus','Chorus'])
  self.assertEqual(instrumental.cover_structure(abc,'[Bridge]\nwords the user typed'),'[Intro]\n\n[Verse]\n\n[Pre-Chorus]\n\n[Chorus]\n','the recording decides, not the typed tags')
  self.assertEqual(instrumental.cover_structure('X:1\nK:C\nCDEF|',''),instrumental.DEFAULT_STRUCTURE,'no labels: the usual fallback')
  self.assertEqual(instrumental.cover_structure('X:1\nK:C\nCDEF|','[Verse]\nla\n[Outro]'),'[Verse]\n\n[Outro]\n')
 def test_style_suffix(self):
  self.assertTrue(instrumental.instrumental_style('Warm jazz trio ').endswith('Warm jazz trio\nInstrumental.'))

class InstrumentalAPITest(unittest.TestCase):
 def setUp(self):
  self.root=studio_root(self);self.s=load_studio(self,self.root,'instrumental_test_server')
  self.client=TestClient(self.s.app);self.headers={'X-Studio-Token':self.s.TOKEN}
 def test_instrumental_requires_a_plan(self):
  body=self.s.Job(style='Piano',instrumental=True,lyrics='').model_dump()
  with patch.object(self.s,'model_ready',return_value=True),patch.object(self.s,'run'):
   self.assertEqual(self.client.post('/api/jobs',json={**body,'cot':'off'},headers=self.headers).status_code,422)
   r=self.client.post('/api/jobs',json={**body,'cot':'melody'},headers=self.headers)
  self.assertEqual(r.status_code,200,r.text)
  saved=json.loads((self.root/'library'/r.json()['id']/'request.json').read_text())
  self.assertTrue(saved['instrumental']);self.assertEqual(saved['lyrics'],'')
 def test_an_instrumental_cover_needs_no_tags_and_follows_the_recording(self):
  audio='e'*24+'.mp3';(self.s.UP/audio).write_bytes(b'fixture');options={'audio_id':audio,'melody_only':True,'source_seconds':0}
  abc='X:1\nK:C\n% intro\nV:Vocal\nz4|\n% verse\nV:Vocal\nCDEF|\n% chorus\nV:Vocal\nGABc|\n'
  self.s.analyses.save(self.s.Job(kind='transcribe',**options),{**options,'abc':abc,'elapsed':10})
  status=self.client.get('/api/analysis',params=options,headers=self.headers).json()
  self.assertEqual((status['status'],status['sections']),('ready',['Intro','Verse','Chorus']))
  body={**self.s.Job(style='Piano',instrumental=True,lyrics='',kind='cover',model='bf16',**options).model_dump(),'cot':'melody'}
  with patch.object(self.s,'model_ready',return_value=True),patch.object(self.s,'run'):
   r=self.client.post('/api/jobs',json=body,headers=self.headers)
  self.assertEqual(r.status_code,200,r.text)
  saved=json.loads((self.root/'library'/r.json()['id']/'request.json').read_text())
  self.assertEqual(saved['lyrics'],'[Intro]\n\n[Verse]\n\n[Chorus]\n')
  self.assertFalse(saved['hook_melody'],'off by default: the melody is played throughout')
  with patch.object(self.s,'model_ready',return_value=True),patch.object(self.s,'run'):
   r=self.client.post('/api/jobs',json={**body,'hook_melody':True},headers=self.headers)
  self.assertEqual(r.status_code,200,r.text);self.assertTrue(json.loads((self.root/'library'/r.json()['id']/'request.json').read_text())['hook_melody'])
