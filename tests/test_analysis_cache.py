import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from analysis_cache import AnalysisCache
from studio_fixture import studio_root, load_studio

class AnalysisCacheTest(unittest.TestCase):
    def test_persists_and_distinguishes_recording_and_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            cache=AnalysisCache(directory)
            r=SimpleNamespace(audio_id='a.wav',melody_only=True,source_seconds=0)
            data={**vars(r),'abc':'X:1\nK:C\nCDEF|','elapsed':50}
            cache.save(r,data)
            self.assertEqual(AnalysisCache(directory).read(r),data)
            for changes in ({'audio_id':'b.wav'},{'melody_only':False},{'source_seconds':30}):
                self.assertIsNone(cache.read(SimpleNamespace(**{**vars(r),**changes})))
            for corrupt in ('broken','[]','{}','null'):
                cache.path(r,'.json').write_text(corrupt)
                self.assertIsNone(cache.read(r))

class AnalysisFlowTest(unittest.TestCase):
    def setUp(self):
        root=studio_root(self)
        self.server=load_studio(self,root,'analysis_test_server')
        self.real_run=self.server.run
        self.server.run=Mock()
        self.server.pool=Mock()
        self.server.pool.wait.return_value=0
        self.server.pool.submit.side_effect=self.submit
        self.server.model_ready=lambda name:True
        self.server.config=lambda:{'transcriber_ready':True}
        self.client=TestClient(self.server.app)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__,None,None,None)
        self.audio='a'*24+'.wav'
        (self.server.UP/self.audio).write_bytes(b'fixture')
        self.options={'audio_id':self.audio,'melody_only':True,'source_seconds':0}
        self.headers={'X-Studio-Token':self.server.TOKEN}
        self.submitted=[]

    def submit(self,p,analyze):
        self.submitted.append(analyze)
        r=json.loads((p/'request.json').read_text())
        if analyze:
            result={k:r[k] for k in self.options}
            result.update(abc='X:1\nK:C\nCDEF|',elapsed=50)
            (p/'result.json').write_text(json.dumps(result))
        elif r['kind']=='cover':
            self.assertTrue(json.loads((p/'analysis.json').read_text())['cached'])
        return Mock(),p/'done'

    def status(self,j):
        return json.loads((self.server.LIB/j/'state.json').read_text())['status']

    def post(self,path,body):
        return self.client.post(path,json=body,headers=self.headers)

    def finish(self,j):
        p=self.server.LIB/j
        r=self.server.Job(**json.loads((p/'request.json').read_text()))
        self.real_run(p,r)

    def test_analysis_once_then_two_covers_without_analysis(self):
        first=self.post('/api/analysis',self.options)
        self.assertEqual(first.status_code,200)
        j=first.json()['job']
        self.assertEqual(self.post('/api/analysis',self.options).json()['job'],j)
        self.assertEqual(self.client.get('/api/jobs').json(),[])
        self.finish(j)
        self.assertEqual(self.post('/api/analysis',self.options).json(),{'status':'ready','sections':[]})
        self.assertEqual(self.client.get('/api/analysis',params=self.options).json(),{'status':'ready','sections':[]})
        for _ in range(2):
            result=self.post('/api/jobs',{**self.options,'kind':'cover','style':'Orchestral','instrumental':True})
            self.assertEqual(result.status_code,200,result.text)
            self.finish(result.json()['id'])
        self.assertEqual(self.submitted,[True,False,False])
        self.assertEqual(len(self.client.get('/api/jobs').json()),2)
        self.assertEqual(self.post('/api/analysis',{**self.options,'melody_only':False}).json()['status'],'starting')

    def test_analysis_while_bf16_downloads(self):
        self.server.downloads.update(status='downloading', model='bf16')
        self.server.model_ready=lambda name:False
        # Missing analysis components must still prevent analysis.
        self.server.config=lambda:{'transcriber_ready':False}
        self.assertEqual(self.post('/api/analysis',self.options).status_code,503)
        self.server.config=lambda:{'transcriber_ready':True}
        result=self.post('/api/analysis',self.options)
        self.assertEqual(result.status_code,200,result.text)
        self.finish(result.json()['job'])
        self.assertEqual(self.submitted,[True])
        self.assertEqual(self.post('/api/analysis',self.options).json(),{'status':'ready','sections':[]})
        self.assertEqual(self.server.downloads.snapshot()['status'],'downloading')
        self.assertEqual(self.post('/api/jobs',{'style':'Orchestral','instrumental':True}).status_code,503)

    def test_installed_bf16_can_generate_and_cover_while_another_package_downloads(self):
        self.server.downloads.update(status='downloading',model='tokens')
        self.server.model_ready=lambda name:name=='bf16'
        request={'style':'Orchestral','instrumental':True}
        self.assertEqual(self.post('/api/jobs',{**request,'model':'8bit'}).status_code,422)
        generated=self.post('/api/jobs',{**request,'model':'bf16'})
        self.assertEqual(generated.status_code,200,generated.text)
        queued=self.post('/api/jobs',{**request,'model':'bf16'})
        self.assertEqual(queued.status_code,200,queued.text);self.assertEqual(queued.json()['status'],'queued')
        self.assertEqual(self.post('/api/jobs/'+queued.json()['id']+'/cancel',{}).status_code,200)
        self.server.active=None
        self.server.analyses.save(self.server.Job(kind='transcribe',**self.options),{**self.options,'abc':'X:1\nK:C\nCDEF|','elapsed':10})
        covered=self.post('/api/jobs',{**request,**self.options,'model':'bf16','kind':'cover'})
        self.assertEqual(covered.status_code,200,covered.text)
        self.assertEqual(self.server.downloads.snapshot()['status'],'downloading')

    def test_jobs_queue_behind_the_running_one(self):
        request={'style':'Orchestral','instrumental':True}
        first=self.post('/api/jobs',request).json()
        self.assertEqual((first['status'],first['position']),('starting',0));self.assertEqual(self.server.active,first['id'])
        second=self.post('/api/jobs',request).json();third=self.post('/api/jobs',request).json()
        self.assertEqual((second['status'],second['position'],third['position']),('queued',1,2))
        self.assertEqual(self.client.get('/api/config').json()['queue'],[second['id'],third['id']])
        self.assertEqual(self.status(second['id']),'queued')
        # Analysis for a cover jumps ahead of queued songs because someone is waiting on it.
        analysis=self.post('/api/analysis',self.options).json()
        self.assertEqual(analysis['status'],'queued');self.assertEqual(self.server.queue[0],analysis['job'])
        self.assertEqual(self.post('/api/jobs/'+analysis['job']+'/cancel',{}).status_code,200)
        # Removing a queued job cancels it without touching the running one.
        self.assertEqual(self.post('/api/jobs/'+second['id']+'/cancel',{}).status_code,200)
        self.assertEqual(self.status(second['id']),'cancelled');self.assertEqual(self.server.active,first['id'])
        self.assertEqual(self.server.queue,[third['id']])
        # Renaming a queued song is allowed; deleting it drops it from the queue.
        self.assertEqual(self.client.patch('/api/jobs/'+third['id']+'/songs/0',json={'title':'Later'},headers=self.headers).status_code,200)
        fourth=self.post('/api/jobs',request).json()
        self.assertEqual(self.client.delete('/api/jobs/'+fourth['id']+'/songs/0',headers=self.headers).status_code,200)
        self.assertEqual(self.server.queue,[third['id']])
        # When the running job finishes, the next queued job starts by itself.
        self.finish(first['id'])
        self.assertEqual(self.server.active,third['id']);self.assertEqual(self.server.queue,[])
        self.assertEqual(self.status(third['id']),'starting')
        self.assertEqual(self.post('/api/jobs/'+second['id']+'/cancel',{}).status_code,409)

    def test_not_ready_validation_queue_and_explicit_retry(self):
        cover={**self.options,'kind':'cover','style':'Orchestral','instrumental':True}
        self.assertEqual(self.post('/api/jobs',cover).status_code,409)
        self.assertEqual(self.post('/api/analysis',{**self.options,'audio_id':'../invalid.wav'}).status_code,400)
        self.assertEqual(self.post('/api/analysis',{**self.options,'source_seconds':-1}).status_code,422)
        self.assertEqual(self.post('/api/jobs',{**cover,'background_analysis':True}).status_code,422)
        self.assertEqual(self.client.post('/api/analysis',json=self.options).status_code,403)
        self.server.active='another-job'
        queued=self.post('/api/analysis',self.options).json()
        self.assertEqual(queued['status'],'queued');self.assertEqual(self.server.queue,[queued['job']])
        self.assertEqual(self.post('/api/analysis',self.options).json(),self.client.get('/api/analysis',params=self.options).json())
        self.server.active=None;self.server.start_next()
        first=queued['job']
        self.assertEqual(self.server.active,first);self.assertEqual(self.status(first),'starting')
        self.server.state(self.server.LIB/first,'failed')
        self.server.active=None
        self.assertEqual(self.client.get('/api/analysis',params=self.options).json()['status'],'failed')
        retry=self.post('/api/analysis',self.options).json()['job']
        self.assertNotEqual(first,retry)

if __name__=='__main__':unittest.main()
