import copy,json,tempfile,unittest
from pathlib import Path
from composition_edit import split_score,lyric_sections,apply,save_draft,load_draft,source_song
from vendor import yue2_abc as abc

HEADER='X:1\nT:\nM:4/4\nL:1/16\nQ:1/4=100\nV: Vocal clef=treble name="Vocal Melody" snm="Vocal"\nV: Ins clef=treble name="Ins Melody" snm="Inst."\nK:C\n'
VERSE='% verse\nV: Vocal\n"C"C4D4E4F4|\nV: Ins\nZ|\nV: Vocal\nG4A4B4c4|\nV: Ins\nZ|\n'
CHORUS='% chorus\nV: Vocal\n"Am"A8G8|\nV: Ins\nC8E8|\n'
SCORE=HEADER+VERSE+CHORUS

def fixture():
 header,parts,parsed=split_score(SCORE);lyrics='[Verse]\nOne two three four\nFive six seven eight\n\n[Chorus]\nStay with me'
 assert lyric_sections(lyrics,parts)
 return dict(abc=SCORE,header=header,sections=parts,bpm=100,style='Piano pop',lyrics=lyrics,instrumental=False,origin={'job':'source','candidate':1})

class CompositionTest(unittest.TestCase):
 def test_arrangement_and_tempo_preserve_every_note(self):
  s=fixture();r=apply(s,{'style':'Symphonic orchestra','bpm':120})
  self.assertTrue(abc.compare(abc.parse(SCORE),abc.parse(r['abc']),allow_tempo_change=True)['match'])
  self.assertEqual(r['style'],'Symphonic orchestra');self.assertEqual(r['bpm'],120)
 def test_free_and_seventh_harmony_keep_melody(self):
  for mode in ('free','sevenths'):
   r=apply(fixture(),{'harmony':mode});a=abc.parse(r['abc'])
   self.assertTrue(abc.compare(abc.parse(SCORE),a)['match'])
   if mode=='free':self.assertFalse(a.voices['Vocal'].chords)
   else:self.assertEqual([c[1] for c in a.voices['Vocal'].chords],['Cmaj7','Am7'])
 def test_repeat_chorus_repeats_lyrics_and_music(self):
  r=apply(fixture(),{'sections':[{'id':'0'},{'id':'1'},{'id':'1'}]})
  self.assertEqual(r['lyrics'].count('Stay with me'),2)
  self.assertEqual(float(abc.parse(r['abc']).voices['Vocal'].time),16)
 def test_remove_reorder_shorten(self):
  r=apply(fixture(),{'sections':[{'id':'1'},{'id':'0','fraction':.5}]})
  self.assertTrue(r['lyrics'].startswith('[Chorus]'));self.assertNotIn('Five six',r['lyrics'])
  self.assertEqual(float(abc.parse(r['abc']).voices['Vocal'].time),8)
 def test_unmatched_tags_keep_lyrics_verbatim_on_structure_change(self):
  s=fixture();s['lyrics']='[Chorus]\nStay with me\n\n[Verse]\nOne two three four'
  r=apply(s,{'sections':[{'id':'1'},{'id':'0'}]})
  self.assertEqual(r['lyrics'],s['lyrics']);self.assertIn('exactly as written',r['summary'])
  self.assertTrue(r['abc'].index('% chorus')<r['abc'].index('% verse'))
  self.assertEqual(apply(s,{'style':'Jazz'})['style'],'Jazz')
 def test_edited_lyrics_are_matched_against_the_score_not_the_source(self):
  s=fixture();s['lyrics']='No tags at all'
  r=apply(s,{'lyrics':'[Verse]\nNew verse words\n\n[Chorus]\nNew chorus','sections':[{'id':'1'},{'id':'1'},{'id':'0'}]})
  self.assertEqual(r['lyrics'],'[Chorus]\nNew chorus\n\n[Chorus]\nNew chorus\n\n[Verse]\nNew verse words');self.assertIn('re-sectioned',r['summary'])
 def test_tag_split_leaves_no_partial_assignment_on_failure(self):
  _,parts,_=split_score(SCORE)
  self.assertFalse(lyric_sections('[Verse]\nA\n\n[Bridge]\nB',parts));self.assertEqual([p['lyrics'] for p in parts],['',''])
  self.assertFalse(lyric_sections('Untagged first line\n[Verse]\nA',parts))
  self.assertTrue(lyric_sections('[Verse 1]\nA\n\n[Chorus]\nB',parts));self.assertEqual([p['lyrics'] for p in parts],['A','B'])
 def test_client_section_lyrics_are_ignored(self):
  r=apply(fixture(),{'sections':[{'id':'1','lyrics':'Injected'},{'id':'0'}]})
  self.assertNotIn('Injected',r['lyrics']);self.assertTrue(r['lyrics'].startswith('[Chorus]\nStay with me'))
 def test_solo_uses_motif_in_instrumental_voice(self):
  r=apply(fixture(),{'sections':[{'id':'0'},{'id':'1','solo':True},{'id':'1'}]});p=abc.parse(r['abc'])
  self.assertIn('% interlude',r['abc']);self.assertEqual(r['lyrics'].count('Stay with me'),1)
  self.assertEqual(len(p.voices['Ins'].notes),4)
 def test_invalid_edits_are_rejected(self):
  for opt in ({'bpm':0},{'bpm':301},{'sections':[]},{'sections':[{'id':'bad'}]},{'harmony':'invalid'},{'sections':[{'id':'1','fraction':.5}]}):
   with self.assertRaises((ValueError,KeyError)):apply(fixture(),opt)
 def test_source_and_draft_are_immutable(self):
  s=fixture();before=copy.deepcopy(s)
  with tempfile.TemporaryDirectory() as tmp:
   edit=apply(s,{'harmony':'sevenths'});d=save_draft(tmp,edit)
   self.assertEqual(load_draft(tmp,d['id']),edit)
   with self.assertRaises(ValueError):load_draft(tmp,'../../etc/passwd')
  self.assertEqual(s,before)
 def test_candidate_specific_score(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);c=root/'candidate-02';c.mkdir();(c/'score.abc').write_text(SCORE)
   (root/'request.json').write_text(json.dumps({'title':'Original','style':'Piano','lyrics':'[Verse]\nA\n[Chorus]\nB','instrumental':False}))
   (root/'result.json').write_text(json.dumps({'candidates':[{'index':2,'prefix':'candidate-02/','title':'Second'}]}))
   s=source_song(root,2);self.assertEqual(s['title'],'Second');self.assertEqual(s['origin']['candidate'],2)
   with self.assertRaises(ValueError):source_song(root,1)
 def test_melody_only_sevenths_requires_harmony_choice(self):
  s=fixture();s['abc']=SCORE.replace('"C"','').replace('"Am"','');s['header'],s['sections'],_=split_score(s['abc'])
  with self.assertRaisesRegex(ValueError,'melody only'):apply(s,{'harmony':'sevenths'})
 def test_solo_on_instrumental_uses_instrumental_motif(self):
  s=fixture();s['instrumental']=True;s['sections'][0]['lines']=['V: Vocal','Z|','V: Ins','C4D4E4F4|'];s['sections']=s['sections'][:1]
  r=apply(s,{'sections':[{'id':'0','solo':True}]});self.assertEqual(len(abc.parse(r['abc']).voices['Ins'].notes),4)
 def test_solo_insertion_preserves_untagged_lyrics(self):
  s=fixture();s['lyrics']='One two three four\nStay with me'
  result=apply(s,{'sections':[{'id':'0'},{'id':'1','solo':True},{'id':'1'}]})
  self.assertEqual(result['lyrics'],s['lyrics'])
 def test_inserted_solo_closes_ties_at_new_boundaries(self):
  s=fixture();text=HEADER+'% verse\nV: Vocal\nC16-|\nV: Ins\nZ|\n% chorus\nV: Vocal\nC16|\nV: Ins\nZ|\n'
  s['abc']=text;s['header'],s['sections'],_=split_score(text)
  result=apply(s,{'sections':[{'id':'0'},{'id':'1','solo':True},{'id':'1'}]})
  self.assertEqual(float(abc.parse(result['abc']).voices['Vocal'].time),12)
