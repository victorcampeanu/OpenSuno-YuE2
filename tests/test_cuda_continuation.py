"""Prefix assembly of cuda_engine's Extend path, with the upstream runtime stubbed (no GPU).

The stubs mirror yue2.protocol 0.1.5: the constants, SongRequest.guidance, token_prefixes and
negative_prefix, and the positional SymbolicPlan / SemanticResult dataclasses.
"""
import importlib
import numpy as np,sys,types,unittest
from dataclasses import dataclass,field,asdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'app'))

EOD=151643;ABC_START,ABC_END=151847,151848;MUSIC_START,MUSIC_END=151851,151852;CODEC_OFFSET=151853;CONTEXT=24576;VOCAB_SIZE=184704
INSTRUCTIONS={'off':'Generate music.','melody':'Generate a melody, then music.','full':'Generate chords, then music.'}

def install_stubs():
 protocol=types.ModuleType('yue2.protocol')
 @dataclass(frozen=True)
 class Sampling:
  temperature:float=1.0;top_p:float=0.95;top_k:int=100;repetition_penalty:float=1.2;penalty_window:int=50;min_tokens:int=200;max_tokens:int=9000
  def __post_init__(self):
   if not 0<=self.min_tokens<=self.max_tokens:raise ValueError('Invalid sampling min/max tokens')
 @dataclass
 class SongRequest:
  style:str;lyrics:str;cot:str='full';seed:int=831001;abc:str|None=None;cfg_scale:float|None=None;id:str='song'
  @property
  def guidance(self):return (1.01 if self.cot=='off' else 1.0) if self.cfg_scale is None else self.cfg_scale
  def text(self):return f"{INSTRUCTIONS[self.cot]}\n[Tags]\n{self.style}\n[Lyrics]\n{self.lyrics}\n"
  def to_dict(self):return asdict(self)
 def token_prefixes(request,tokenizer,abc_ids=None):
  base=[EOD]+tokenizer.encode(request.text())
  if request.cot=='off':return base+[ABC_START,ABC_END,MUSIC_START]
  if abc_ids is None:
   if request.abc is None:return base+[ABC_START]
   abc_ids=tokenizer.encode(request.abc)
  return base+[ABC_START]+list(abc_ids)+[ABC_END,MUSIC_START]
 def negative_prefix(request,tokenizer,abc_ids=None):
  base=[EOD]+tokenizer.encode(INSTRUCTIONS[request.cot])
  if request.cot=='off':return base+[MUSIC_START]
  if abc_ids is None:raise ValueError('Symbolic CFG must retain the exact positive-branch ABC IDs')
  return base+[ABC_START]+list(abc_ids)+[ABC_END,MUSIC_START]
 protocol.__dict__.update(Sampling=Sampling,SongRequest=SongRequest,token_prefixes=token_prefixes,negative_prefix=negative_prefix,
  EOD=EOD,ABC_START=ABC_START,ABC_END=ABC_END,MUSIC_START=MUSIC_START,MUSIC_END=MUSIC_END,CODEC_OFFSET=CODEC_OFFSET,CONTEXT=CONTEXT,VOCAB_SIZE=VOCAB_SIZE)
 pipeline=types.ModuleType('yue2.pipeline')
 @dataclass
 class SymbolicPlan:
  request:object;abc:str|None;abc_ids:list;prefix:list;timing:dict=field(default_factory=dict);truncated:bool=False
 @dataclass
 class SemanticResult:
  plan:object;tokens:list;timing:dict;truncated:bool
 @dataclass
 class SongResult:
  audio:object;sample_rate:int;semantic:object;latents:object;config:dict;weights:dict;timing:dict;request_identity:str
 class YuE2Pipeline:pass
 pipeline.__dict__.update(SymbolicPlan=SymbolicPlan,SemanticResult=SemanticResult,SongResult=SongResult,YuE2Pipeline=YuE2Pipeline)
 storage=types.ModuleType('yue2.storage');storage.identity=lambda payload:'identity'
 sampling=types.ModuleType('yue2.sampling');sampling.distribution=lambda logits,sampling,history,step,phase,legacy_off=False:logits
 yue2=types.ModuleType('yue2');yue2.protocol=protocol;yue2.pipeline=pipeline;yue2.storage=storage;yue2.sampling=sampling
 torch=types.ModuleType('torch');torch.backends=types.SimpleNamespace(cuda=types.SimpleNamespace(is_flash_attention_available=lambda:True))
 torch.full=lambda shape,value:np.full(shape,value,dtype=np.float32)
 installed=[]
 for name,module in {'yue2':yue2,'yue2.protocol':protocol,'yue2.pipeline':pipeline,'yue2.storage':storage,'yue2.sampling':sampling,'torch':torch,'soundfile':types.ModuleType('soundfile')}.items():
  if name not in sys.modules:sys.modules[name]=module;installed.append(name)
 return protocol,pipeline,installed

class Tokenizer:
 """Characters as token ids, well inside the text vocabulary."""
 def encode(self,text):return [ord(c) for c in text]
 def decode(self,ids):return ''.join(chr(i) for i in ids)

class FakePipe:
 def __init__(self,abc_reply='',semantic_reply=()):
  self.tokenizer=Tokenizer();self.calls=[];self.abc_reply=abc_reply;self.semantic_reply=list(semantic_reply)
 def _generate(self,prefix,sampling,seed,phase,**kwargs):
  self.calls.append({'prefix':list(prefix),'sampling':sampling,'seed':seed,'phase':phase,**kwargs,'first_mask':getattr(sys.modules.get('cuda_engine'),'_FirstToken',None) and sys.modules['cuda_engine']._FirstToken.mask})
  if phase=='abc':
   ids=self.tokenizer.encode(self.abc_reply)[:sampling.max_tokens];emitted=[]
   for t in ids+[ABC_END]:  # upstream reports every token, the end token included, before it stops
    if kwargs.get('on_token'):kwargs['on_token'](phase,t)
    if t!=ABC_END:emitted.append(t)
   return emitted,{'seconds':1.0},False
  return [t+CODEC_OFFSET for t in self.semantic_reply],{'seconds':2.0},False

class CudaContinuationTest(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.protocol,cls.pipeline,cls.installed=install_stubs()
  sys.modules.pop('cuda_engine',None);cls.engine=importlib.import_module('cuda_engine')
 @classmethod
 def tearDownClass(cls):
  # Other tests may need the real torch/yue2 modules (or their absence).
  for name in cls.installed+['cuda_engine']:sys.modules.pop(name,None)
 def request(self,**kw):return self.protocol.SongRequest(style='Piano',lyrics='[Verse]\nla',**kw)
 def test_continue_plan_prompts_with_the_kept_bars_and_rebuilds_the_prefix(self):
  pipe=FakePipe(abc_reply='|REST|');request=self.request(seed=5)
  plan=self.engine.continue_plan(pipe,request,'X:1\n|KEPT|',self.protocol.Sampling(min_tokens=32,max_tokens=4096),{'cancelled':None,'on_token':None})
  call=pipe.calls[0];tok=pipe.tokenizer
  self.assertEqual(call['phase'],'abc');self.assertEqual(call['seed'],5)
  self.assertEqual(call['prefix'],[EOD]+tok.encode(request.text())+[ABC_START]+tok.encode('X:1\n|KEPT|'),'the model continues after ABC_START and the kept bars')
  self.assertEqual(plan.abc,'X:1\n|KEPT||REST|');self.assertEqual(plan.abc_ids,tok.encode(plan.abc))
  self.assertEqual(plan.prefix,self.protocol.token_prefixes(request,tok,plan.abc_ids),'synthesize() accepts the plan')
  self.assertEqual(plan.prefix[-2:],[ABC_END,MUSIC_START]);self.assertFalse(plan.truncated)
 def test_continue_plan_without_a_plan_mode_skips_generation(self):
  pipe=FakePipe();request=self.request(cot='off')
  plan=self.engine.continue_plan(pipe,request,'',self.protocol.Sampling(),{})
  self.assertEqual(pipe.calls,[]);self.assertIsNone(plan.abc);self.assertEqual(plan.prefix[-3:],[ABC_START,ABC_END,MUSIC_START])
 def test_continue_semantic_appends_the_codec_prompt_and_keeps_it(self):
  pipe=FakePipe(semantic_reply=[7,8,9]);request=self.request(cfg_scale=1.2);tok=pipe.tokenizer
  abc_ids=tok.encode('X:1\n|A|');plan=self.pipeline.SymbolicPlan(request,'X:1\n|A|',abc_ids,self.protocol.token_prefixes(request,tok,abc_ids))
  prompt=[1,2,3,4]
  result=self.engine.continue_semantic(pipe,plan,prompt,self.protocol.Sampling(),{'cancelled':None,'on_token':None})
  call=pipe.calls[0]
  self.assertEqual(call['phase'],'semantic');self.assertEqual(call['cfg_scale'],1.2);self.assertFalse(call['legacy_off'])
  self.assertEqual(call['prefix'],plan.prefix+[c+CODEC_OFFSET for c in prompt],'prefix = ... ABC_END MUSIC_START codec+OFFSET ...')
  self.assertEqual(call['negative'],self.protocol.negative_prefix(request,tok,abc_ids)+[c+CODEC_OFFSET for c in prompt],'the negative branch hears the same music')
  self.assertEqual(result.tokens,prompt+[7,8,9]);self.assertIs(result.plan,plan)
  self.assertEqual(result.timing['prompt_frames'],4)
 def test_cfg_off_has_no_negative(self):
  pipe=FakePipe(semantic_reply=[5,6]);request=self.request(cot='off',cfg_scale=1.0);tok=pipe.tokenizer
  plan=self.pipeline.SymbolicPlan(request,None,[],self.protocol.token_prefixes(request,tok))
  result=self.engine.continue_semantic(pipe,plan,[42]*10,self.protocol.Sampling(),{})
  call=pipe.calls[0]
  self.assertIsNone(call['negative']);self.assertTrue(call['legacy_off']);self.assertEqual(call['cfg_scale'],1.0)
  self.assertEqual(result.tokens,[42]*10+[5,6])
 def test_semantic_budget_fits_the_context_and_reports_when_it_cannot(self):
  pipe=FakePipe(semantic_reply=[1]);request=self.request(cfg_scale=1.0);tok=pipe.tokenizer
  abc_ids=tok.encode('X');plan=self.pipeline.SymbolicPlan(request,'X',abc_ids,self.protocol.token_prefixes(request,tok,abc_ids))
  long_prompt=[0]*(CONTEXT-len(plan.prefix)-1000)
  self.engine.continue_semantic(pipe,plan,long_prompt,self.protocol.Sampling(min_tokens=200,max_tokens=9000),{})
  self.assertEqual(pipe.calls[0]['sampling'].max_tokens,1000);self.assertEqual(pipe.calls[0]['sampling'].min_tokens,200)
  with self.assertRaises(ValueError):
   self.engine.continue_semantic(pipe,plan,[0]*(CONTEXT-len(plan.prefix)-50),self.protocol.Sampling(),{})
 def test_score_writer_proposes_after_the_committed_text_with_a_short_budget(self):
  pipe=FakePipe(abc_reply='|"G"z');tok=pipe.tokenizer
  writer=self.engine.ScoreWriter(pipe,'Piano','[Verse]\nla',self.protocol.Sampling(min_tokens=200,max_tokens=9000),40,{'cancelled':None})
  writer.commit('X:1\n|c4')
  self.assertEqual(writer.propose(8,lambda t:False),'|"G"z')
  call=pipe.calls[0];request=self.request(cot='full',seed=40)
  self.assertEqual(call['prefix'],self.protocol.token_prefixes(request,tok)+tok.encode('X:1\n|c4'),'request text, ABC_START, then the committed score')
  self.assertEqual((call['sampling'].min_tokens,call['sampling'].max_tokens,call['phase']),(0,8,'abc'))
  self.assertEqual(call['cancelled'],None);self.assertIsNone(call['first_mask'])
  writer.commit('|"F"d4');writer.propose(8,lambda t:False)
  self.assertNotEqual(pipe.calls[0]['seed'],pipe.calls[1]['seed'],'each proposal draws differently')
  self.assertEqual(pipe.calls[1]['prefix'][-len(tok.encode('|"F"d4')):],tok.encode('|"F"d4'))
 def test_score_writer_stops_where_the_caller_says_and_constrains_the_first_token(self):
  pipe=FakePipe(abc_reply='"G"z4|"C"c4|');tok=pipe.tokenizer
  writer=self.engine.ScoreWriter(pipe,'Piano','[Verse]\nla',self.protocol.Sampling(),40,{'cancelled':None,'on_token':lambda phase,t:seen.append(t)})
  seen=[]
  # stop is honoured token by token, like the MLX writer: nothing after the closing quote is kept.
  self.assertEqual(writer.propose(64,lambda t:t.count('"')>=2),'"G"')
  self.assertEqual(tok.decode(seen),'"G"','the progress callback still hears every token that was sampled')
  # first limits the first sampled token to text tokens matching the pattern, through the wrapped distribution.
  writer.propose(8,lambda t:True,first='"')
  mask=pipe.calls[1]['first_mask'];self.assertEqual(mask.shape,(VOCAB_SIZE,))
  self.assertEqual([i for i in range(EOD) if mask[i]==0],[ord('"')]);self.assertEqual(mask[ord('z')],-np.inf)
  self.assertIsNone(self.engine._FirstToken.mask,'the constraint is lifted after the proposal')
  wrapped=sys.modules['yue2.sampling'].distribution
  class T:
   device=dtype=None
   def __init__(self,v):self.v=v
   def to(self,*a):return self
   def __add__(self,o):return T(self.v+o.v)
  self.engine._FirstToken.mask=T(np.array([0,-np.inf]))
  try:
   self.assertEqual(list(wrapped(T(np.array([1.,1.])),None,[],0,'abc').v),[1.,-np.inf],'step 0 is masked')
   self.assertEqual(list(wrapped(T(np.array([1.,1.])),None,[],1,'abc').v),[1.,1.],'later steps are not')
  finally:self.engine._FirstToken.mask=None
  with self.assertRaises(ValueError):writer.propose(8,lambda t:True,first='no token spells this')
 def test_replan_swaps_the_score_and_keeps_the_prefix_consistent(self):
  pipe=FakePipe();request=self.request();tok=pipe.tokenizer
  plan=self.pipeline.SymbolicPlan(request,'X:1\n|A|',tok.encode('X:1\n|A|'),self.protocol.token_prefixes(request,tok,tok.encode('X:1\n|A|')),{'seconds':1},True)
  same=self.engine.replan(pipe,plan,'X:1\n|A|');self.assertIs(same,plan)
  new=self.engine.replan(pipe,plan,'X:1\n|z|')
  self.assertEqual(new.abc,'X:1\n|z|');self.assertEqual(new.prefix,self.protocol.token_prefixes(request,tok,new.abc_ids));self.assertTrue(new.truncated)
