"""The render node API on its own, and a Studio rendering through it."""
import importlib.util
import io
import json
import os
import wave
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'app'))
from studio_fixture import studio_root, load_studio

FAKE_WORKER='''import json,sys,time
from pathlib import Path
p=Path(sys.argv[1]);r=json.loads((p/'request.json').read_text())
print('[load] fake model',flush=True)
(p/'progress.json').write_text(json.dumps({'stage':'planning','tokens':3}))
until=time.monotonic()+(30 if 'SLOW' in r.get('title','') else r.get('_sleep',0))
while time.monotonic()<until:
    CHECK_CANCEL();time.sleep(.01)
if r.get('_fail') or 'FAIL' in r.get('title',''): raise RuntimeError('fake failure')
if r['kind']=='transcribe':
    (p/'result.json').write_text(json.dumps({'abc':'X:1\\nK:C\\nCDEF|','elapsed':1,'audio_id':r['audio_id'],'melody_only':r['melody_only'],'source_seconds':r['source_seconds'],'vocal_register':None}))
else:
    (p/'audio.wav').write_bytes(b'RIFFfake')
    (p/'result.json').write_text(json.dumps({'candidates':[{'index':1,'prefix':'','seed':r.get('seed',1)}],'style':r.get('style')}))
print('[done] Finished',flush=True)
'''

def load(module_name,path):
    spec=importlib.util.spec_from_file_location(module_name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    return module

def tiny_wav(seconds=1,rate=8000):
    """A real (silent) WAV so ffmpeg on the node has something to decode."""
    buffer=io.BytesIO()
    with wave.open(buffer,'wb') as out:
        out.setnchannels(1);out.setsampwidth(2);out.setframerate(rate);out.writeframes(b'\0\0'*rate*seconds)
    return buffer.getvalue()

def wait_for(check,timeout=15):
    until=time.monotonic()+timeout
    while time.monotonic()<until:
        value=check()
        if value: return value
        time.sleep(.05)
    raise AssertionError('Timed out waiting')


class NodeFixture:
    """A render node in a temporary root with a fake worker standing in for the model."""
    def __init__(self,case,token='secret'):
        self.tmp=tempfile.TemporaryDirectory();case.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        for name in ('app/render_node.py','app/resident_worker.py','app/runtime_platform.py','config/model-assets.json','config/lora-assets.json'):
            (self.root/name).parent.mkdir(exist_ok=True)
            shutil.copy2(ROOT/name,self.root/name)
        (self.root/'app/worker.py').write_text(FAKE_WORKER)
        python=patch('worker_pool.environment_python',return_value=Path(sys.executable));python.start();case.addCleanup(python.stop)
        # Everything is "installed" on the fake node: weights, the CUDA runtime and the analysis environment.
        for target,value in (('hardware.model_ready',True),('model_downloads.ModelDownloads.package_ready',True)):
            ready=patch(target,return_value=value);ready.start();case.addCleanup(ready.stop)
        env=patch('hardware.environment_python',return_value=Path(sys.executable));env.start();case.addCleanup(env.stop)
        previous=os.environ.get('OPENSUNO_NODE_TOKEN')
        os.environ['OPENSUNO_NODE_TOKEN']=token
        case.addCleanup(lambda:os.environ.update({'OPENSUNO_NODE_TOKEN':previous}) if previous is not None else os.environ.pop('OPENSUNO_NODE_TOKEN',None))
        self.module=load('render_node_under_test_'+os.urandom(3).hex(),self.root/'app/render_node.py')
        case.addCleanup(sys.modules.pop,self.module.__name__,None)
        case.addCleanup(self.stop)
        self.client=TestClient(self.module.app,base_url='http://node')
        self.headers={'Authorization':'Bearer '+token}

    def stop(self):
        procs=list(self.module.runner.pool.workers.values());self.module.runner.pool.close()
        for proc in procs: proc.wait(timeout=6)


class RenderNodeApiTest(unittest.TestCase):
    def setUp(self):
        self.node=NodeFixture(self)
        self.client=self.node.client;self.headers=self.node.headers

    def test_token_is_required(self):
        self.assertEqual(self.client.get('/v1/health').status_code,401)
        self.assertEqual(self.client.get('/v1/health',headers={'Authorization':'Bearer wrong'}).status_code,401)
        health=self.client.get('/v1/health',headers=self.headers).json()
        self.assertTrue(health['ok']);self.assertEqual(health['protocol'],2);self.assertEqual(health['node'],'opensuno-render-node')
        self.assertIn('gpu',health);self.assertTrue(all(m['ready'] for m in health['models']))

    def test_job_runs_and_its_files_can_be_collected(self):
        request={'kind':'plan','model':'bf16','style':'Indie','seed':7}
        created=self.client.post('/v1/jobs',data={'request':json.dumps(request)},headers=self.headers).json()
        self.assertEqual(created['status'],'starting')
        view=wait_for(lambda:(v:=self.client.get('/v1/jobs/'+created['id'],headers=self.headers).json()) if self.client.get('/v1/jobs/'+created['id'],headers=self.headers).json()['status']=='complete' else None)
        self.assertEqual(view['status'],'complete')
        self.assertIn('[load] fake model',view['log']);self.assertEqual(view['progress']['stage'],'planning')
        self.assertEqual(set(view['files']),{'result.json','audio.wav'})  # the fake WAV cannot be encoded to MP3
        self.assertGreater(view['kept_until'],time.time()+29*86400)
        self.assertEqual(self.client.get('/v1/jobs/'+created['id']+'/files/audio.wav',headers=self.headers).content,b'RIFFfake')
        partial=self.client.get('/v1/jobs/'+created['id']+'/files/audio.wav',headers={**self.headers,'Range':'bytes=4-7'})
        self.assertEqual((partial.status_code,partial.content),(206,b'fake'))
        self.assertEqual(self.client.get('/v1/jobs/'+created['id']+'/files/state.json',headers=self.headers).status_code,404)
        # Incremental log reads return only what is new.
        again=self.client.get('/v1/jobs/'+created['id'],params={'log_offset':view['log_offset']},headers=self.headers).json()
        self.assertEqual(again['log'],'')
        self.assertEqual(self.client.delete('/v1/jobs/'+created['id'],headers=self.headers).json(),{'ok':True})
        self.assertEqual(self.client.get('/v1/jobs/'+created['id'],headers=self.headers).status_code,404)

    def test_recordings_are_decoded_drawn_and_cut_on_the_node(self):
        created=self.client.post('/v1/uploads',files={'file':('demo.wav',tiny_wav(2),'audio/wav')},headers=self.headers).json()
        self.assertTrue(created['id'].endswith('.wav'));self.assertAlmostEqual(created['seconds'],2,delta=.1)
        self.assertTrue((self.node.root/'uploads'/created['id']).is_file())
        wave=self.client.get('/v1/uploads/'+created['id']+'/waveform',headers=self.headers).json()
        self.assertTrue(wave['peaks']);self.assertAlmostEqual(wave['seconds'],2,delta=.1)
        clip=self.client.post('/v1/uploads/'+created['id']+'/trim',json={'start':0,'end':1},headers=self.headers).json()
        self.assertAlmostEqual(clip['seconds'],1,delta=.1);self.assertNotEqual(clip['id'],created['id'])
        song=self.client.post('/v1/uploads/'+created['id']+'/song',headers=self.headers).json()
        self.assertEqual(song['status'],'complete');        self.assertEqual(set(song['files']),{'result.json','audio.wav','audio.mp3'})
        self.assertEqual(song.get('lyrics'),'')
        self.assertEqual(self.client.get('/v1/uploads/'+created['id']+'/tags',headers=self.headers).json()['lyrics'],'')
        self.assertEqual(self.client.get('/v1/uploads/'+created['id']+'/cover',headers=self.headers).status_code,404)
        self.assertEqual(self.client.get('/v1/uploads/'+created['id'],headers=self.headers).status_code,200)
        # Embedded tags always come back, empty when the file carries none.
        self.assertEqual({k:created[k] for k in ('title','artist','lyrics','synced_lyrics')},{'title':'','artist':'','lyrics':'','synced_lyrics':''})

    def test_rejects_unknown_models_and_missing_recordings(self):
        bad=self.client.post('/v1/jobs',data={'request':json.dumps({'kind':'plan','model':'nope'})},headers=self.headers)
        self.assertEqual(bad.status_code,400);self.assertIn('does not offer',bad.json()['detail'])
        missing=self.client.post('/v1/jobs',data={'request':json.dumps({'kind':'transcribe','audio_id':'a'*24+'.wav'})},headers=self.headers)
        self.assertIn(missing.status_code,{400,503})
        self.assertEqual(list(self.node.module.JOBS.iterdir()),[])

    def test_cancel_stops_a_running_job(self):
        created=self.client.post('/v1/jobs',data={'request':json.dumps({'kind':'plan','model':'bf16','_sleep':30})},headers=self.headers).json()
        wait_for(lambda:self.client.get('/v1/jobs/'+created['id'],headers=self.headers).json()['status']=='running')
        self.assertEqual(self.client.post('/v1/jobs/'+created['id']+'/cancel',headers=self.headers).json(),{'ok':True})
        view=wait_for(lambda:(v:=self.client.get('/v1/jobs/'+created['id'],headers=self.headers).json()) if self.client.get('/v1/jobs/'+created['id'],headers=self.headers).json()['status']=='cancelled' else None)
        self.assertEqual(view['status'],'cancelled')
        self.assertFalse(self.client.get('/v1/health',headers=self.headers).json()['busy'])


class StudioThroughNodeTest(unittest.TestCase):
    """A Studio in one temporary root sends its jobs to a node in another."""
    def setUp(self):
        self.node=NodeFixture(self)
        root=studio_root(self)
        (root/'.render-node.json').write_text(json.dumps({'url':'http://node','token':'secret'}))
        self.server=load_studio(self,root,'studio_under_test_'+os.urandom(3).hex())
        # The Studio's HTTP client talks to the node's test app instead of a socket.
        self.server.remote.node.session=self.node.client
        self.client=TestClient(self.server.app);self.client.__enter__();self.addCleanup(self.client.__exit__,None,None,None)
        self.headers={'X-Studio-Token':self.server.TOKEN}

    def job(self,j):
        return self.client.get('/api/jobs/'+j).json()

    def test_config_describes_the_node_and_its_models(self):
        config=self.client.get('/api/config').json()
        self.assertTrue(config['render_node']['connected']);self.assertEqual(config['render_node']['url'],'http://node')
        self.assertTrue(all(m['ready'] for m in config['models']))
        self.assertEqual(config['preloads'],{})
        self.assertEqual(self.client.post('/api/models/preload?model=bf16',headers=self.headers).json()['status'],'ready')

    def test_song_renders_on_the_node_and_lands_in_the_library(self):
        created=self.client.post('/api/jobs',json={'kind':'plan','style':'Indie pop','instrumental':True,'model':'bf16'},headers=self.headers)
        self.assertEqual(created.status_code,200,created.text)
        j=created.json()['id']
        done=wait_for(lambda:(v:=self.job(j)) if self.job(j)['status']=='complete' else None)
        p=self.server.LIB/j
        # Light files come home at once; the audio stays on the node and is streamed from there.
        self.assertTrue((p/'result.json').is_file());self.assertFalse((p/'audio.wav').is_file())
        self.assertEqual(done['result']['style'],'Indie pop')
        self.assertIn('[remote] Sending the job to http://node',done['log']);self.assertIn('[done] Finished',done['log'])
        self.assertEqual(done['progress']['stage'],'planning')
        self.assertEqual(done['remote']['pending'],['audio.wav']);self.assertIn('audio.wav',done['files'])
        self.assertEqual(len(list(self.node.module.JOBS.iterdir())),1)  # kept on the node for streaming and later download
        streamed=self.client.get('/api/files/'+j+'/audio.wav',headers={'Range':'bytes=0-3'})
        self.assertEqual((streamed.status_code,streamed.content),(206,b'RIFF'))
        # Download brings the WAV into the library; from then on it is local.
        fetched=self.client.post('/api/jobs/'+j+'/fetch',headers=self.headers).json()
        self.assertTrue((p/'audio.wav').read_bytes()==b'RIFFfake');self.assertNotIn('remote',fetched)
        self.assertEqual(self.client.get('/api/jobs/'+j+'/songs/1/download').content,b'RIFFfake')

    def test_recordings_and_openai_helpers_go_to_the_node(self):
        created=self.client.post('/api/upload',files={'file':('demo.wav',tiny_wav(2),'audio/wav')},headers=self.headers)
        self.assertEqual(created.status_code,200,created.text);audio=created.json()['id']
        # The interface keeps no copy: the node holds the recording and does the ffmpeg work.
        self.assertFalse((self.server.UP/audio).exists());self.assertTrue((self.node.root/'uploads'/audio).is_file())
        self.assertTrue(self.client.get('/api/uploads/'+audio+'/waveform').json()['peaks'])
        self.assertEqual(self.client.get('/api/uploads/'+audio+'/timeline').json()['recording'],audio)
        self.assertAlmostEqual(self.client.post('/api/uploads/'+audio+'/trim',json={'start':0,'end':1},headers=self.headers).json()['seconds'],1,delta=.1)
        self.assertEqual(self.client.get('/api/uploads/'+audio).status_code,200)
        song=self.client.post('/api/uploads/'+audio+'/song',json={'title':'Demo'},headers=self.headers).json()
        self.assertEqual(song['kind'],'upload');self.assertEqual(set(song['remote']['pending']),{'audio.wav','audio.mp3'})
        self.assertEqual(song['request'].get('lyrics'),'')
        self.assertFalse((self.server.LIB/song['id']/'audio.wav').exists())
        self.assertEqual(self.client.get('/api/files/'+song['id']+'/audio.mp3').status_code,200)
        # OpenAI helpers answer from the node, which keeps the key.
        with patch.object(self.node.module.style_ai,'openai_configured',return_value=True):
            self.server.remote.checked=0  # drop the cached health snapshot
            self.assertTrue(self.client.get('/api/config').json()['openai_configured'])

    def test_failure_on_the_node_is_reported(self):
        j=self.client.post('/api/jobs',json={'kind':'plan','style':'x','instrumental':True,'model':'bf16','title':'FAIL please'},headers=self.headers).json()['id']
        done=wait_for(lambda:(v:=self.job(j)) if self.job(j)['status'] in {'complete','failed'} else None)
        self.assertEqual(done['status'],'failed');self.assertIn('fake failure',done['error']);self.assertIn('[error]',done['log'])
        self.assertIsNone(self.server.active)

    def test_cancel_reaches_the_node(self):
        j=self.client.post('/api/jobs',json={'kind':'plan','style':'x','instrumental':True,'model':'bf16','title':'SLOW song'},headers=self.headers).json()['id']
        wait_for(lambda:self.job(j)['status']=='running')
        self.assertEqual(self.client.post('/api/jobs/'+j+'/cancel',headers=self.headers).json(),{'ok':True})
        self.assertEqual(self.job(j)['status'],'cancelling')
        wait_for(lambda:self.job(j)['status']=='cancelled')
        wait_for(lambda:not self.node.client.get('/v1/health',headers=self.node.headers).json()['busy'])
        self.assertIsNone(self.server.active)

    def test_a_recording_kept_only_on_the_node_is_a_valid_job_source(self):
        # The Studio has no copy of a recording uploaded through the node; the node checks for the file itself.
        audio='b'*24+'.wav'
        self.assertFalse((self.server.UP/audio).exists())
        self.assertEqual(self.server.Job(kind='transcribe',audio_id=audio).audio_id,audio)
        with patch.object(self.server.remote,'configured',return_value=False):
            with self.assertRaisesRegex(ValueError,'valid source file'):self.server.Job(kind='transcribe',audio_id=audio)

    def test_settings_of_removed_features_in_old_requests_are_dropped(self):
        old={'kind':'generate','style':'Piano','lyrics':'[Verse]\nla','persona_id':'abc','keep_sound':True,'align_lyrics':True,'created':1}
        job=self.server.Job(**old)
        self.assertEqual(job.style,'Piano');self.assertFalse(hasattr(job,'persona_id'))
        with self.assertRaises(ValueError):self.server.Job(kind='generate',stylee='typo')

    def test_recording_analysis_is_cached_from_the_node(self):
        audio='a'*24+'.wav';(self.server.UP/audio).write_bytes(b'fixture')
        started=self.client.post('/api/analysis',json={'audio_id':audio,'melody_only':True,'source_seconds':0},headers=self.headers).json()
        self.assertIn(started.get('status'),{'starting','queued'},started)
        wait_for(lambda:self.client.get('/api/analysis',params={'audio_id':audio}).json()['status']=='ready')
        self.assertTrue((self.node.root/'uploads'/audio).is_file())  # the recording travelled with the job

    def test_restart_reattaches_to_a_job_still_on_the_node(self):
        j=self.client.post('/api/jobs',json={'kind':'plan','style':'x','instrumental':True,'model':'bf16'},headers=self.headers).json()['id']
        p=self.server.LIB/j
        wait_for(lambda:(p/'remote.json').is_file())
        remote_id=json.loads((p/'remote.json').read_text())['id']
        wait_for(lambda:self.job(j)['status']=='complete')
        # Pretend the Studio died mid-render: the node still has the job (put it back), the library says running.
        node_job=self.node.module.JOBS/remote_id;node_job.mkdir(exist_ok=True)
        (node_job/'request.json').write_text((p/'request.json').read_text())
        (node_job/'result.json').write_text(json.dumps({'candidates':[],'reattached':True}))
        (node_job/'run.log').write_text('[node] old log\n')
        self.node.module.write_state(node_job,'complete',started=time.time())
        self.server.state(p,'running');(p/'result.json').unlink()
        self.server.reconcile_jobs()
        done=wait_for(lambda:(v:=self.job(j)) if self.job(j)['status']=='complete' else None)
        self.assertTrue(done['result']['reattached']);self.assertIn('[remote] Reconnected',done['log'])


if __name__=='__main__':
    unittest.main()
