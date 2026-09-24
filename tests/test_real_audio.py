"""Extending a recording: the torch-free checkpoint reader, the switchable NAR adapter, and the API."""
import io,json,pickle,sys,types,unittest,zipfile
from collections import OrderedDict
from pathlib import Path
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'model'));sys.path.insert(0,str(ROOT/'tests'))
import torch_checkpoint
import song_continuation as sc
from test_song_continuation import studio_server
try:
 import mlx.core as mx
except ImportError:
 mx=None

def torch_archive(obj,storages):
 """A ``torch.save``-style zip written without torch: fake ``torch`` globals stand in for the real ones."""
 class FloatStorage:pass
 class LongStorage:pass
 def _rebuild_tensor_v2(*a):return a
 FloatStorage.__module__=LongStorage.__module__='torch';_rebuild_tensor_v2.__module__='torch._utils'
 FloatStorage.__qualname__='FloatStorage';LongStorage.__qualname__='LongStorage';_rebuild_tensor_v2.__qualname__='_rebuild_tensor_v2'
 torch=types.ModuleType('torch');utils=types.ModuleType('torch._utils')
 torch.FloatStorage,torch.LongStorage,utils._rebuild_tensor_v2=FloatStorage,LongStorage,_rebuild_tensor_v2
 class Ref:
  def __init__(self,key,array):self.key,self.array=key,array
 class Tensor:
  def __init__(self,key,array,offset=0,stride=None):self.key,self.array,self.offset,self.stride=key,array,offset,stride
  def __reduce_ex__(self,protocol):
   a=self.array;stride=self.stride or tuple(int(s//a.itemsize) for s in a.strides)
   return (_rebuild_tensor_v2,(Ref(self.key,a),self.offset,tuple(a.shape),stride,False,OrderedDict()))
 class Pickler(pickle.Pickler):
  def persistent_id(self,o):
   if isinstance(o,Ref):return ('storage',FloatStorage if o.array.dtype==np.float32 else LongStorage,o.key,'cpu',o.array.size)
   return None
 saved={k:sys.modules.get(k) for k in ('torch','torch._utils')}
 sys.modules['torch'],sys.modules['torch._utils']=torch,utils
 try:
  buffer=io.BytesIO();Pickler(buffer,protocol=2).dump(obj(Tensor))
 finally:
  for k,v in saved.items():
   if v is None:sys.modules.pop(k,None)
   else:sys.modules[k]=v
 out=io.BytesIO()
 with zipfile.ZipFile(out,'w') as z:
  z.writestr('ck/data.pkl',buffer.getvalue());z.writestr('ck/byteorder','little');z.writestr('ck/version','3')
  for key,array in storages.items():z.writestr(f'ck/data/{key}',np.ascontiguousarray(array).tobytes())
 out.seek(0);return out

class CheckpointReaderTest(unittest.TestCase):
 def test_reads_tensors_dicts_and_scalars_without_torch(self):
  a=np.arange(12,dtype=np.float32).reshape(3,4);b=np.array([5,6,7],dtype=np.int64)
  archive=torch_archive(lambda T:{'lora':[T(0,a),T(1,b)],'io':{'w':OrderedDict(weight=T(0,a))},'rank':32},{0:a,1:b})
  ck=torch_checkpoint.load(archive)
  self.assertEqual(ck['rank'],32);np.testing.assert_array_equal(ck['lora'][0],a);np.testing.assert_array_equal(ck['lora'][1],b)
  self.assertEqual(ck['lora'][0].dtype,np.float32);self.assertEqual(ck['lora'][1].dtype,np.int64)
  np.testing.assert_array_equal(ck['io']['w']['weight'],a)
 def test_offsets_and_transposed_strides_are_honoured(self):
  a=np.arange(12,dtype=np.float32).reshape(3,4)
  archive=torch_archive(lambda T:{'tail':T(0,a[1:],offset=4),'t':T(0,a.T,stride=(1,4))},{0:a})
  ck=torch_checkpoint.load(archive)
  np.testing.assert_array_equal(ck['tail'],a[1:]);np.testing.assert_array_equal(ck['t'],a.T)
 def test_unknown_globals_are_refused(self):
  bad=io.BytesIO()
  with zipfile.ZipFile(bad,'w') as z:
   z.writestr('ck/data.pkl',b'\x80\x02cos\nsystem\nq\x00.')  # GLOBAL os.system
  bad.seek(0)
  with self.assertRaises(pickle.UnpicklingError):torch_checkpoint.load(bad)
  empty=io.BytesIO()
  with zipfile.ZipFile(empty,'w') as z:z.writestr('readme.txt','not a checkpoint')
  empty.seek(0)
  with self.assertRaises(ValueError):torch_checkpoint.load(empty)
 def test_pairs_follow_the_training_order(self):
  import real_audio
  tensors=[np.zeros((2,)) for _ in range(2*7*2)]
  self.assertEqual(len(real_audio.pairs({'lora':tensors},2)),14)
  with self.assertRaises(ValueError):real_audio.pairs({'lora':tensors},3)
 def test_safetensors_nar_uses_the_same_training_order(self):
  import real_audio
  from test_loras import write_safetensors
  rank,layers=4,2
  tensors={}
  expected=[]
  for layer in range(layers):
   for block in ('nar_self_attn','nar_mlp'):
    names=('q_proj','k_proj','v_proj','o_proj') if 'attn' in block else ('gate_proj','up_proj','down_proj')
    for index,name in enumerate(names):
     A=np.full((rank,8),layer*10+index,np.float32);B=np.full((16,rank),100+index,np.float32)
     tensors[f'layers.{layer}.{block}.{name}.lora_A']=A;tensors[f'layers.{layer}.{block}.{name}.lora_B']=B
     expected+=[A,B]
  tensors['vae2llm.weight']=np.ones((4,2),np.float32);tensors['vae2llm.bias']=np.zeros(4,np.float32)
  tensors['llm2vae.weight']=np.ones((2,4),np.float32);tensors['llm2vae.bias']=np.arange(2,dtype=np.float32)
  import tempfile
  with tempfile.TemporaryDirectory() as tmp:
   path=Path(tmp)/'nar.safetensors';write_safetensors(path,tensors)
   ck=real_audio.load_checkpoint(path)
  self.assertEqual(ck['rank'],rank);self.assertEqual(len(ck['lora']),layers*7*2)
  for got,want in zip(ck['lora'],expected):np.testing.assert_array_equal(got,want)
  np.testing.assert_array_equal(ck['io']['vae2llm']['weight'],tensors['vae2llm.weight'])
  np.testing.assert_array_equal(ck['io']['llm2vae']['bias'],tensors['llm2vae.bias'])

@unittest.skipIf(mx is None,'MLX is not installed')
class AdapterTest(unittest.TestCase):
 CFG=dict(vocab_size=200,hidden_size=64,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=16,intermediate_size=96,rms_norm_eps=1e-6,rope_theta=10000.0,latent_dim=8,max_latent_frames=64,timestep_shift=3.0)
 def checkpoint(self,rank=4,seed=0):
  rng=np.random.default_rng(seed);lora=[]
  for _ in range(2):
   for o,i in [(64,64),(32,64),(32,64),(64,64),(96,64),(96,64),(64,96)]:
    lora+=[(rng.normal(size=(rank,i))*.1).astype(np.float32),(rng.normal(size=(o,rank))*.1).astype(np.float32)]
  io_={'vae2llm':{'weight':rng.normal(size=(64,8)).astype(np.float32),'bias':np.zeros(64,np.float32)},
       'llm2vae':{'weight':rng.normal(size=(8,64)).astype(np.float32),'bias':np.zeros(8,np.float32)}}
  return {'lora':lora,'io':io_,'rank':rank}
 def test_adapters_change_nothing_until_enabled_and_match_folded_weights(self):
  from yue2_model import Yue2Model;import real_audio
  model=Yue2Model(self.CFG);mx.eval(model.parameters())
  tokens=list(range(10));state=mx.random.normal((6,8),key=mx.random.key(1)).astype(mx.bfloat16)
  before=model.nar_velocity(state,0.3,model.nar_prefill(tokens),len(tokens));mx.eval(before)
  switches=real_audio.attach_checkpoint(model,self.checkpoint())
  self.assertEqual(len(switches),2*7+2);self.assertIs(real_audio.attach_checkpoint(model,self.checkpoint()),switches,'attached once')
  cache=model.nar_prefill(tokens)
  off=model.nar_velocity(state,0.3,cache,len(tokens));mx.eval(off);self.assertTrue(mx.allclose(before,off))
  with real_audio.enabled(model):
   on=model.nar_velocity(state,0.3,cache,len(tokens));mx.eval(on)
  self.assertFalse(mx.allclose(before,on));self.assertTrue(mx.isfinite(on).all())
  after=model.nar_velocity(state,0.3,cache,len(tokens));mx.eval(after);self.assertTrue(mx.allclose(before,after),'switched back off')
  linear=model.model.layers[0].nar_mlp.gate_proj;ck=self.checkpoint();A,B=ck['lora'][8],ck['lora'][9]
  x=mx.random.normal((3,64),key=mx.random.key(2)).astype(mx.bfloat16)
  linear.active=True;y=linear(x).astype(mx.float32);linear.active=False
  folded=x.astype(mx.float32)@(linear.base.weight.astype(mx.float32)+mx.array(B)@mx.array(A)).T
  self.assertLess(float(mx.abs(y-folded).max()),0.05)
 def test_shape_mismatches_are_reported(self):
  from yue2_model import Yue2Model;import real_audio
  model=Yue2Model(self.CFG);ck=self.checkpoint();ck['lora'][1]=np.zeros((65,4),np.float32)
  with self.assertRaises(ValueError):real_audio.attach_checkpoint(model,ck)

class RecordingContinuationTest(unittest.TestCase):
 TOKENS={'codec':[(i*37)%32768 for i in range(25*40)],'audio_id':'c'*24+'.mp3','source_seconds':0,'seconds':40.0,'tokenizer':'v4'}
 def test_cut_is_by_time_score_free_and_marked_real_audio(self):
  c=sc.recording_continuation(self.TOKENS,12.0,title='take.mp3')
  self.assertEqual(len(c['codec']),300);self.assertEqual(c['cut_seconds'],12.0);self.assertEqual(c['cot'],'off');self.assertEqual(c['abc_prefix'],'')
  self.assertTrue(c['real_audio']);self.assertEqual(c['source'],{'recording':self.TOKENS['audio_id'],'seconds':40.0,'title':'take.mp3','tokenizer':'v4'})
  whole=sc.recording_continuation(self.TOKENS,None);self.assertEqual(len(whole['codec']),1000)
  self.assertEqual(len(sc.recording_continuation(self.TOKENS,99)['codec']),1000);self.assertEqual(len(sc.recording_continuation(self.TOKENS,0)['codec']),1)
  with self.assertRaises(ValueError):sc.recording_continuation({**self.TOKENS,'codec':[]},None)

class ContinueRecordingApiTest(unittest.TestCase):
 def setUp(self):
  studio_server(self,'real_audio_test_server')
  self.audio='c'*24+'.mp3';(self.s.UP/self.audio).write_bytes(b'fixture')
  self.options={'audio_id':self.audio,'source_seconds':0}
  self.s.run=lambda p,r:None;self.s.tokens_ready=lambda:True
 def body(self,**extra):
  return {**self.request,'kind':'generate','model':'bf16','continue_recording':True,'cot':'full',**self.options,**extra}
 def post(self,body):
  with patch.object(self.s,'model_ready',return_value=True):return self.client.post('/api/jobs',json=body,headers=self.headers)
 def tokenized(self):
  self.s.recording_tokens.save(self.s.Job(kind='tokenize',**self.options),{**self.options,'codec':RecordingContinuationTest.TOKENS['codec'],'seconds':40.0,'tokenizer':'v4','elapsed':3})
 def test_needs_the_recording_tokens_and_the_installed_package(self):
  self.assertEqual(self.post(self.body()).status_code,409)
  self.s.tokens_ready=lambda:False;self.assertEqual(self.post(self.body()).status_code,503)
 def test_job_keeps_the_recording_and_renders_it_with_the_real_audio_decoder(self):
  self.tokenized()
  r=self.post(self.body(continue_seconds=12.0,title='take.mp3 · extended'));self.assertEqual(r.status_code,200,r.text);jobdir=self.root/'library'/r.json()['id']
  saved=json.loads((jobdir/'continue-input.json').read_text())
  self.assertEqual(len(saved['codec']),300);self.assertTrue(saved['real_audio']);self.assertEqual(saved['cot'],'off');self.assertEqual(saved['source']['title'],'take.mp3')
  request=json.loads((jobdir/'request.json').read_text())
  self.assertEqual(request['cot'],'off','recordings have no plan');self.assertEqual(request['abc'],'');self.assertTrue(request['continue_recording'])
  self.assertEqual(request['origin']['kind'],'continue');self.assertEqual(request['origin']['recording'],self.audio);self.assertEqual(request['origin']['cut_seconds'],12.0);self.assertEqual(request['origin']['seconds'],40.0)
  whole=self.post(self.body(continue_seconds=None)).json()['id']
  self.assertEqual(len(json.loads((self.root/'library'/whole/'continue-input.json').read_text())['codec']),1000)
 def test_runs_on_mlx_only_and_excludes_other_sources(self):
  self.tokenized()
  self.assertEqual(self.post(self.body(model='cuda-bf16')).status_code,422)
  self.assertEqual(self.post(self.body(continue_source=self.job)).status_code,422)
  self.assertEqual(self.post(self.body(instrumental=True,lyrics='')).status_code,422)
  self.assertEqual(self.post(self.body(audio_id='')).status_code,422)
  self.assertEqual(self.post(self.body(kind='cover')).status_code,422)
 def test_rerender_of_an_extended_recording_keeps_the_real_audio_decoder(self):
  self.tokenized()
  first=self.post(self.body()).json()['id'];p=self.root/'library'/first
  (p/'state.json').write_text(json.dumps({'status':'complete'}));(p/'result.json').write_text(json.dumps({'candidates':[{'index':1,'prefix':'','seed':7}]}))
  (p/'tokens.json').write_text(json.dumps([1,2,3]))
  # The Re-render button sends the source song's saved settings, including its continue_recording flag.
  saved=json.loads((p/'request.json').read_text())
  r=self.post({**saved,'render_source':first,'render_candidate':1,'steps':8});self.assertEqual(r.status_code,200,r.text)
  again=self.root/'library'/r.json()['id'];render=json.loads((again/'render-input.json').read_text())
  self.assertTrue(render['real_audio']);self.assertEqual(render['codec'],[1,2,3]);self.assertFalse((again/'continue-input.json').exists())
  self.assertFalse(json.loads((again/'request.json').read_text())['continue_recording'])
  plain=self.post({**self.request,'model':'bf16','render_source':self.job,'render_candidate':1,'steps':8});self.assertEqual(plain.status_code,200,plain.text)
  self.assertFalse(json.loads((self.root/'library'/plain.json()['id']/'render-input.json').read_text())['real_audio'])

if __name__=='__main__':unittest.main()
