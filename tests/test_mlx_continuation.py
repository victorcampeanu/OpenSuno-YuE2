"""Prefix assembly for Extend on the MLX engine, with generate.py replaced by a fake."""
import sys,types,unittest
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import mlx_continuation
from song_continuation import CONTEXT

EOD,ABC_START,ABC_END,MUSIC_START,MUSIC_END=151643,151849,151850,151851,151852
CODEC_OFFSET=151853

@dataclass
class Sampling:
 temperature:float=1.0;top_p:float=0.95;top_k:int=100;repetition_penalty:float=1.2;penalty_window:int=50;min_tokens:int=200;max_tokens:int=9000

class Tokenizer:
 def encode(self,text):return [ord(c) for c in text]
 def decode(self,ids):return ''.join(chr(i) for i in ids)

class FakeEngine(types.SimpleNamespace):
 def __init__(self,reply):
  super().__init__(CODEC_OFFSET=CODEC_OFFSET,Sampling=Sampling,calls=[],reply=reply)
 def token_prefix(self,tok,style,lyrics,cot,abc_ids=None):
  base=[EOD]+tok.encode(f'{cot}|{style}|{lyrics}')
  if cot=='off':return base+[ABC_START,ABC_END,MUSIC_START]
  if abc_ids is None:return base+[ABC_START]
  return base+[ABC_START]+list(abc_ids)+[ABC_END,MUSIC_START]
 def negative_prefix(self,tok,cot,abc_ids):
  base=[EOD]+tok.encode(cot)
  return base+[MUSIC_START] if cot=='off' else base+[ABC_START]+list(abc_ids)+[ABC_END,MUSIC_START]
 def generate_tokens(self,model,prefix,s,seed,phase,negative=None,cfg_scale=1.0,legacy_off=False,on_token=None):
  self.calls.append(dict(prefix=list(prefix),sampling=s,seed=seed,phase=phase,negative=negative,cfg_scale=cfg_scale,legacy_off=legacy_off))
  return list(self.reply[phase]),False

class Pipe:
 tokenizer=Tokenizer();model=None

class MlxContinuationTest(unittest.TestCase):
 def test_plan_continues_after_the_kept_bars(self):
  engine=FakeEngine({'abc':Tokenizer().encode('|c4|')})
  abc,ids,truncated=mlx_continuation.continue_plan(engine,Pipe(),'Pop','[Verse]\nla','full','X:1\nV:1\n|C4',Sampling(),7)
  call=engine.calls[0]
  self.assertEqual(call['phase'],'abc');self.assertEqual(call['seed'],7)
  self.assertEqual(call['prefix'],engine.token_prefix(Pipe.tokenizer,'Pop','[Verse]\nla','full')+Tokenizer().encode('X:1\nV:1\n|C4'))
  self.assertEqual(abc,'X:1\nV:1\n|C4|c4|');self.assertEqual(ids,Tokenizer().encode(abc));self.assertFalse(truncated)
 def test_no_plan_when_cot_is_off_or_nothing_is_kept(self):
  engine=FakeEngine({})
  self.assertEqual(mlx_continuation.continue_plan(engine,Pipe(),'Pop','la','off','X:1',Sampling(),1),(None,[],False))
  self.assertEqual(mlx_continuation.continue_plan(engine,Pipe(),'Pop','la','full','',Sampling(),1),(None,[],False))
  self.assertEqual(engine.calls,[])
 def test_extend_keeps_the_prompt_and_primes_both_cfg_branches(self):
  engine=FakeEngine({'semantic':[CODEC_OFFSET+9,CODEC_OFFSET+8]})
  abc_ids=Tokenizer().encode('X:1')
  prefix,codec,_=mlx_continuation.continue_semantic(engine,Pipe(),'Pop','la','full',abc_ids,[1,2,3],Sampling(),5,1.5)
  call=engine.calls[0]
  self.assertEqual(prefix,engine.token_prefix(Pipe.tokenizer,'Pop','la','full',abc_ids))
  self.assertEqual(call['prefix'],prefix+[CODEC_OFFSET+1,CODEC_OFFSET+2,CODEC_OFFSET+3])
  self.assertEqual(call['negative'],engine.negative_prefix(Pipe.tokenizer,'full',abc_ids)+[CODEC_OFFSET+1,CODEC_OFFSET+2,CODEC_OFFSET+3])
  self.assertEqual(call['cfg_scale'],1.5);self.assertFalse(call['legacy_off'])
  self.assertEqual(codec,[1,2,3,9,8])
 def test_no_plan_uses_pipeline_default_guidance(self):
  engine=FakeEngine({'semantic':[CODEC_OFFSET+4]})
  prefix,codec,_=mlx_continuation.continue_semantic(engine,Pipe(),'Pop','la','off',[],[7,7],Sampling(),5,None)
  call=engine.calls[0]
  self.assertEqual(codec,[7,7,4]);self.assertEqual(call['cfg_scale'],1.01);self.assertTrue(call['legacy_off'])
  self.assertEqual(prefix[-3:],[ABC_START,ABC_END,MUSIC_START])
 def test_budget_fits_the_remaining_context(self):
  engine=FakeEngine({'semantic':[CODEC_OFFSET]})
  prompt=list(range(CONTEXT-1000))
  mlx_continuation.continue_semantic(engine,Pipe(),'','','off',[],prompt,Sampling(),1,1.0)
  s=engine.calls[0]['sampling']
  self.assertLessEqual(len(engine.calls[0]['prefix'])+s.max_tokens,CONTEXT);self.assertLessEqual(s.min_tokens,s.max_tokens)
 def test_an_immediate_end_is_reported(self):
  engine=FakeEngine({'semantic':[]})
  with self.assertRaises(RuntimeError):mlx_continuation.continue_semantic(engine,Pipe(),'','','off',[],[1],Sampling(),1,1.0)

if __name__=='__main__':unittest.main()
