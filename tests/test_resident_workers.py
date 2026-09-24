import json
import shutil
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from worker_pool import WorkerPool,PageLeases
from runtime_platform import environment_python

class ResidentWorkersTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        shutil.copy(ROOT/'resident_worker.py',self.root/'resident_worker.py')
        shutil.copy(ROOT/'runtime_platform.py',self.root/'runtime_platform.py')
        self.python_patch=patch('worker_pool.environment_python',return_value=Path(sys.executable))
        self.python_patch.start()
        self.addCleanup(self.python_patch.stop)
        (self.root/'worker.py').write_text('''import json,sys,time,os
from pathlib import Path
p=Path(sys.argv[1]);r=json.loads((p/'request.json').read_text())
MODEL_CACHE['calls']=MODEL_CACHE.get('calls',0)+1
print('job '+p.name,flush=True)
if r.get('fail'): raise RuntimeError('test failure')
if r.get('_preload'): raise SystemExit(0)
until=time.monotonic()+r.get('sleep',0)
while time.monotonic()<until:
    CHECK_CANCEL();time.sleep(.01)
(p/'result.json').write_text(json.dumps({'pid':os.getpid(),'calls':MODEL_CACHE['calls']}))
''')
        self.pool=WorkerPool(self.root)
    def tearDown(self):
        procs=list(self.pool.workers.values());self.pool.close()
        for p in procs:p.wait(timeout=6)
        self.tmp.cleanup()
    def job(self,name,**request):
        p=self.root/name;p.mkdir();(p/'request.json').write_text(json.dumps(request));return p
    def runjob(self,p,analysis=False):
        proc,marker=self.pool.submit(p,analysis);code=self.pool.wait(proc,marker)
        return proc,code
    def test_reuse_and_separate_analysis_runtime(self):
        first=self.job('first');proc,code=self.runjob(first);self.assertEqual(code,0)
        second=self.job('second');again,code=self.runjob(second)
        self.assertIs(proc,again);self.assertEqual(json.loads((second/'result.json').read_text())['calls'],2)
        self.assertNotIn('second',(first/'run.log').read_text())
        self.assertIn('second',(second/'run.log').read_text())
        analysis=self.job('analysis');other,code=self.runjob(analysis,True)
        self.assertNotEqual(proc.pid,other.pid);self.assertEqual(json.loads((analysis/'result.json').read_text())['calls'],1)
    def test_preload_reuses_worker_and_cache_without_a_song(self):
        from model_preloads import ModelPreloads
        preloads=ModelPreloads(self.root,self.pool,threading.RLock())
        preloads.start('bf16',{})
        proc=self.pool.workers[False]
        preloads.start('bf16',{})
        until=time.monotonic()+4
        while preloads.snapshot()['bf16']['status']=='loading' and time.monotonic()<until:
            time.sleep(.01)
        self.assertEqual(preloads.snapshot()['bf16']['status'],'ready')
        following=self.job('after_preload');again,code=self.runjob(following)
        self.assertEqual(code,0);self.assertIs(proc,again)
        self.assertEqual(json.loads((following/'result.json').read_text())['calls'],2)
        self.assertEqual(len(list((self.root/'.preloads').iterdir())),1)
        preloads.clear();self.assertEqual(preloads.snapshot(),{})
        preloads.mark_ready('analysis',proc,following)
        self.assertEqual(preloads.snapshot()['analysis']['status'],'ready')

    def test_cancel_and_restart(self):
        p=self.job('slow',sleep=30);proc,marker=self.pool.submit(p,False)
        self.pool.discard(proc)
        self.assertNotEqual(self.pool.wait(proc,marker),0)
        replacement,code=self.runjob(self.job('next'))
        self.assertEqual(code,0);self.assertNotEqual(proc.pid,replacement.pid)
    def test_cooperative_cancel_preserves_process_and_cache(self):
        p=self.job('cooperative',sleep=30);proc,marker=self.pool.submit(p,False)
        (p/'.cancel-resident').touch()
        self.assertEqual(self.pool.wait(proc,marker),2)
        following=self.job('following');again,code=self.runjob(following)
        self.assertEqual(code,0);self.assertIs(proc,again)
        self.assertEqual(json.loads((following/'result.json').read_text())['calls'],2)

    def test_failure_discards_cache(self):
        proc,code=self.runjob(self.job('bad',fail=True));self.assertEqual(code,1)
        self.pool.discard(proc)
        nextproc,code=self.runjob(self.job('good'));self.assertEqual(code,0)
        self.assertNotEqual(proc.pid,nextproc.pid)
    def test_close_releases_both_workers(self):
        a,_=self.runjob(self.job('music'));b,_=self.runjob(self.job('cover'),True)
        self.pool.close()
        a.wait(timeout=6);b.wait(timeout=6);self.assertFalse(self.pool.workers)

    def test_cuda_uses_separate_environment_and_releases_windows_memory(self):
        first,_=self.runjob(self.job('mlx',model='bf16'))
        with patch('worker_pool.sys.platform','win32'):
            cuda,code=self.runjob(self.job('cuda',model='cuda-bf16'))
        self.assertEqual(code,0)
        self.assertIsNotNone(first.poll())
        self.assertEqual(list(self.pool.workers),['cuda-bf16'])
        again,code=self.runjob(self.job('cuda_again',model='cuda-bf16'))
        self.assertEqual(code,0);self.assertIs(cuda,again)

class RegistryTest(unittest.TestCase):
    def test_windows_uses_cuda(self):
        from model_registry import default_model,available_models
        with patch('model_registry.sys.platform','win32'):
            self.assertEqual(default_model(),'cuda-bf16')
            self.assertEqual(set(available_models()),{'cuda-bf16'})

    def test_mac_retains_mlx_default_and_choices(self):
        from model_registry import default_model,available_models
        with patch('model_registry.sys.platform','darwin'):
            self.assertEqual(default_model(),'bf16')
            self.assertEqual(set(available_models()),{'bf16'})

class PageLeaseTest(unittest.TestCase):
    def test_last_tab_and_refresh_grace(self):
        emptied=threading.Event();leases=PageLeases(threading.RLock(),emptied.set,grace=.08)
        a=leases.open();b=leases.open();leases.close(a)
        self.assertFalse(emptied.wait(.12))
        leases.close(b);replacement=leases.open()
        self.assertFalse(emptied.wait(.12))
        leases.close(replacement);self.assertTrue(emptied.wait(.5));leases.shutdown()
    def test_shutdown_cancels_pending_release(self):
        emptied=threading.Event();leases=PageLeases(threading.RLock(),emptied.set,grace=.08)
        leases.close(leases.open());leases.shutdown();self.assertFalse(emptied.wait(.12))


class ModelCacheTest(unittest.TestCase):
    def test_pipeline_reused_with_fresh_job_callbacks(self):
        import runpy
        import types
        from unittest.mock import patch
        cache={};constructed=[]
        class Pipeline:
            def __init__(self,*args,**kwargs):
                constructed.append(self);self.log=kwargs['log']
        engine=types.ModuleType('generate');engine.Yue2Pipeline=Pipeline
        engine.Sampling=lambda **kw:kw
        engine.write_wav=engine.generate_tokens=engine.token_prefix=lambda *a,**kw:None
        original=lambda *a,**kw:None
        engine.synthesize=original
        mlx=types.ModuleType('mlx');mlx.__path__=[]
        core=types.ModuleType('mlx.core');mlx.core=core
        modules={'mlx':mlx,'mlx.core':core,'numpy':types.ModuleType('numpy'),'generate':engine}
        with tempfile.TemporaryDirectory() as d,patch.dict(sys.modules,modules):
            for index in range(3):
                p=Path(d)/str(index);p.mkdir()
                request={'kind':'plan','model':'bf16','style':'Piano','lyrics':'', 'abc':'X:1\nK:C\nC', 'abc_sampling':{}}
                if index==0: request['_preload']=True
                (p/'request.json').write_text(json.dumps(request))
                with patch.object(sys,'argv',[str(ROOT/'worker.py'),str(p)]):
                    if index==0:
                        with self.assertRaises(SystemExit) as stopped:
                            runpy.run_path(str(ROOT/'worker.py'),init_globals={'MODEL_CACHE':cache})
                        self.assertEqual(stopped.exception.code,0)
                        self.assertFalse((p/'result.json').exists())
                        continue
                    runpy.run_path(str(ROOT/'worker.py'),init_globals={'MODEL_CACHE':cache})
                self.assertTrue((p/'result.json').exists())
                self.assertEqual(cache['bf16'].log.__globals__['jobdir'],p.resolve())
                self.assertIs(engine._resident_synthesize,original)
            self.assertEqual(len(constructed),1)

if __name__=='__main__':unittest.main()
