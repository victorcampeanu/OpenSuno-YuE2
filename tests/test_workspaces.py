import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from workspaces import Workspaces
from studio_fixture import studio_root, load_studio


class WorkspaceAPITest(unittest.TestCase):
    def setUp(self):
        self.root=studio_root(self);self.s=load_studio(self,self.root,'workspace_test_server')
        self.client=TestClient(self.s.app);self.headers={'X-Studio-Token':self.s.TOKEN}
        self.job='20260101-120000-abcdef12';self.legacy='20260101-120001-abcdef12';self.failed='20260101-120002-abcdef12'
        for job in (self.job,self.legacy,self.failed):
            p=self.root/'library'/job;p.mkdir()
            (p/'request.json').write_text(json.dumps({'title':'Original','kind':'generate','style':'Piano'}))
            (p/'state.json').write_text(json.dumps({'status':'failed' if job==self.failed else 'complete'}))
            if job==self.job:
                (p/'result.json').write_text(json.dumps({'candidates':[{'index':i,'prefix':f'candidate-{i:02d}/'} for i in (1,2)]}))
                for i in (1,2):
                    q=p/f'candidate-{i:02d}';q.mkdir();(q/'audio.wav').write_bytes(bytes([i]))
                (p/'favorite.json').write_text('{"candidate":2}')
            elif job==self.legacy:(p/'audio.wav').write_bytes(b'legacy')
        self.originals={str(p):p.read_bytes() for p in (self.root/'library').rglob('*') if p.is_file()}

    def test_incomplete_jobs_do_not_hide_existing_library(self):
        for suffix,request in (('03',None),('04','{')):
            p=self.root/'library'/('20260101-1200'+suffix+'-abcdef12');p.mkdir()
            (p/'state.json').write_text('{"status":"failed"}')
            if request is not None:(p/'request.json').write_text(request)
        response=self.client.get('/api/jobs')
        self.assertEqual(response.status_code,200,response.text)
        self.assertEqual({j['id'] for j in response.json()},{self.job,self.legacy,self.failed})

    def create(self,name='Jazz'):
        r=self.client.post('/api/workspaces',json={'name':name},headers=self.headers)
        self.assertEqual(r.status_code,200,r.text);return r.json()['id']

    def move(self,workspace,songs):
        return self.client.post('/api/workspaces/move',json={'workspace_id':workspace,'songs':songs},headers=self.headers)

    def snapshot(self):return self.client.get('/api/workspaces').json()

    def test_default_create_rename_and_persistence(self):
        self.assertEqual(self.snapshot(),{'workspaces':[{'id':'default','name':'My Workspace'}],'memberships':{}})
        wid=self.create('  Jazz  ')
        r=self.client.patch('/api/workspaces/'+wid,json={'name':'Orchestra'},headers=self.headers)
        self.assertEqual(r.status_code,200)
        self.assertEqual(Workspaces(self.root/'.workspaces.json').read(),self.snapshot())
        self.assertEqual(self.snapshot()['workspaces'][1],{'id':wid,'name':'Orchestra'})
        for name in ('ORCHESTRA','   '):
            r=self.client.post('/api/workspaces',json={'name':name},headers=self.headers)
            self.assertEqual(r.status_code,400)
        self.assertEqual(self.client.delete('/api/workspaces/default',headers=self.headers).status_code,400)

    def test_bulk_move_keeps_versions_audio_favorite_and_request_intact(self):
        wid=self.create()
        songs=[{'job':self.job,'candidate':2},{'job':self.legacy,'candidate':1},{'job':self.job,'candidate':2}]
        r=self.move(wid,songs);self.assertEqual(r.status_code,200,r.text);self.assertEqual(r.json()['moved'],2)
        self.assertEqual(self.snapshot()['memberships'],{self.job+':2':wid,self.legacy+':1':wid})
        for path,original in self.originals.items():self.assertEqual(Path(path).read_bytes(),original)
        r=self.move('default',[songs[0]]);self.assertEqual(r.status_code,200)
        self.assertEqual(self.snapshot()['memberships'][self.job+':2'],'default')

    def test_bulk_validation_is_atomic_and_rejects_stale_versions(self):
        wid=self.create();good={'job':self.job,'candidate':1}
        for bad in ({'job':self.legacy,'candidate':2},{'job':'20260101-120003-aaaaaaaa','candidate':1}):
            self.assertEqual(self.move(wid,[good,bad]).status_code,404)
            self.assertEqual(self.snapshot()['memberships'],{})
        self.assertEqual(self.move('missing',[good]).status_code,400)
        self.s.active=self.job
        self.assertEqual(self.move(wid,[good]).status_code,409)
        self.s.active=None
        self.assertEqual(self.snapshot()['memberships'],{})

    def test_delete_workspace_returns_songs_to_default_without_deleting_files(self):
        wid=self.create()
        self.move(wid,[{'job':self.job,'candidate':1},{'job':self.failed,'candidate':0}])
        self.assertEqual(self.client.delete('/api/workspaces/'+wid,headers=self.headers).status_code,200)
        self.assertTrue(all(v=='default' for v in self.snapshot()['memberships'].values()))
        self.assertEqual(len(self.snapshot()['workspaces']),1)
        for path,original in self.originals.items():self.assertEqual(Path(path).read_bytes(),original)

    def test_generation_saves_workspace_and_rejects_deleted_destination(self):
        wid=self.create()
        request={'style':'Piano','instrumental':True,'workspace_id':wid,'candidates':2}
        with patch.object(self.s,'model_ready',return_value=True),patch.object(self.s,'run'):
            r=self.client.post('/api/jobs',json=request,headers=self.headers)
        self.assertEqual(r.status_code,200,r.text);self.s.active=None
        generated=self.root/'library'/r.json()['id']
        saved=json.loads((generated/'request.json').read_text())
        self.assertEqual(saved['workspace_id'],wid)
        self.client.delete('/api/workspaces/'+wid,headers=self.headers)
        r=self.client.post('/api/jobs',json=request,headers=self.headers)
        self.assertEqual(r.status_code,400,r.text)
        self.assertTrue((generated/'request.json').is_file())

    def test_create_queues_artwork_and_omits_key(self):
        wid=self.create()
        request={'style':'Piano','instrumental':True,'workspace_id':wid,'api_key':'sk-'+'a'*24}
        with patch.object(self.s,'model_ready',return_value=True),patch.object(self.s,'run'),patch.object(self.s,'generate_artwork',return_value={'status':'complete','model':'gpt-image-2.5-flare','quality':'low','elapsed':1}):
            r=self.client.post('/api/jobs',json=request,headers=self.headers)
        self.assertEqual(r.status_code,200,r.text)
        generated=self.root/'library'/r.json()['id']
        saved=json.loads((generated/'request.json').read_text())
        self.assertNotIn('api_key',saved)
        art=json.loads((generated/'artwork.json').read_text())
        self.assertIn(art['status'],{'running','complete'})

    def test_csrf_bounds_and_song_deletion_cleanup(self):
        self.assertEqual(self.client.post('/api/workspaces',json={'name':'No'}).status_code,403)
        wid=self.create()
        for songs in ([],[{'job':'../../elsewhere','candidate':1}],[{'job':self.job,'candidate':9}]):
            self.assertEqual(self.move(wid,songs).status_code,422)
        self.move(wid,[{'job':self.job,'candidate':1},{'job':self.job,'candidate':2}])
        self.assertEqual(self.client.delete('/api/jobs/'+self.job+'/songs/1',headers=self.headers).status_code,200)
        self.assertEqual(self.snapshot()['memberships'],{self.job+':2':wid})
        self.assertEqual(self.client.delete('/api/jobs/'+self.job+'/songs/2',headers=self.headers).status_code,200)
        self.assertEqual(self.snapshot()['memberships'],{})

if __name__=='__main__':unittest.main()
