"""Studio-side client for a Render Node: sends a job folder, mirrors its progress back, collects the files."""
import json
import time
from pathlib import Path

import requests

from runtime_platform import replace_file

PROTOCOL=2  # must match render_node.PROTOCOL; bumped when a node keeps audio and encodes MP3
INPUT_FILES=('analysis.json','render-input.json','continue-input.json')
TERMINAL={'complete','failed','cancelled','interrupted'}
# Audio stays on the node until someone asks for it; the Studio plays the MP3 from there.
HEAVY_SUFFIXES=('.wav','.mp3')


def is_heavy(name):
    return name.endswith(HEAVY_SUFFIXES)


def chunks(response, size=256*1024):
    """Body of a node response in pieces, whether it came from requests (streamed) or a test client."""
    if hasattr(response,'iter_content'): return response.iter_content(size)
    return iter([response.content])


class RenderNodeError(Exception):
    """A node that answered with an error (``status`` is its HTTP code) or could not be reached (502)."""
    def __init__(self, message, status=502):
        super().__init__(message); self.status=status


class RenderNode:
    def __init__(self, url, token='', session=None, timeout=15):
        self.url=url.rstrip('/'); self.token=token or ''
        self.session=session or requests.Session(); self.timeout=timeout

    @property
    def headers(self):
        return {'Authorization':'Bearer '+self.token} if self.token else {}

    def call(self, method, path, timeout=None, stream=False, **kw):
        # Streaming is a requests feature; a test double (Starlette's TestClient) simply returns the whole body.
        if stream and isinstance(self.session,requests.Session): kw['stream']=True
        try:
            response=self.session.request(method,self.url+path,headers={**self.headers,**kw.pop('headers',{})},timeout=timeout or self.timeout,**kw)
        except (requests.RequestException,OSError) as e:
            raise RenderNodeError('Cannot reach the render node at '+self.url+': '+(type(e).__name__ if isinstance(e,requests.RequestException) else str(e))) from e
        if response.status_code>=400:
            try: detail=response.json().get('detail')
            except ValueError: detail=None
            raise RenderNodeError(detail if isinstance(detail,str) else 'The render node answered '+str(response.status_code),response.status_code)
        return response

    def forward(self, method, path, timeout=120, **kw):
        """Run a Studio request on the node instead (uploads, waveforms, OpenAI helpers) and return its JSON."""
        return self.call(method,path,timeout=timeout,**kw).json()

    def health(self, timeout=5):
        data=self.call('GET','/v1/health',timeout=timeout).json()
        if not data.get('ok') or data.get('node')!='opensuno-render-node': raise RenderNodeError('That address is not an OpenSuno render node.')
        return data

    def download_models(self, model):
        return self.call('POST','/v1/models/download',params={'model':model}).json()

    def submit(self, jobdir, uploads_dir):
        """Send request.json, the worker's side files and any recording the request refers to."""
        jobdir=Path(jobdir); r=json.loads((jobdir/'request.json').read_text())
        files=[]
        handles=[]
        try:
            for name in INPUT_FILES:
                if (jobdir/name).is_file():
                    h=(jobdir/name).open('rb');handles.append(h);files.append(('inputs',(name,h,'application/json')))
            name=r.get('audio_id')
            if name and (Path(uploads_dir)/name).is_file():
                h=(Path(uploads_dir)/name).open('rb');handles.append(h);files.append(('uploads',(name,h,'application/octet-stream')))
            data=self.call('POST','/v1/jobs',data={'request':json.dumps(r)},files=files or None,timeout=600).json()
        finally:
            for h in handles: h.close()
        return data['id']

    def status(self, remote_id, log_offset=0):
        return self.call('GET','/v1/jobs/'+remote_id,params={'log_offset':log_offset}).json()

    def fetch(self, remote_id, name, target):
        target=Path(target); target.parent.mkdir(parents=True,exist_ok=True)
        response=self.call('GET','/v1/jobs/'+remote_id+'/files/'+name,timeout=(15,600),stream=True)
        tmp=target.with_name(target.name+'.part')
        try:
            with tmp.open('wb') as out:
                for chunk in chunks(response): out.write(chunk)
            replace_file(tmp,target)
        finally:
            tmp.unlink(missing_ok=True)

    def stream(self, remote_id, name, range_header=None):
        """Open a job file on the node for pass-through to a browser; the Range header lets players seek."""
        return self.stream_path('/v1/jobs/'+remote_id+'/files/'+name,range_header)

    def stream_path(self, path, range_header=None):
        headers={'Range':range_header} if range_header else {}
        return self.call('GET',path,headers=headers,timeout=(15,600),stream=True)

    def cancel(self, remote_id):
        return self.call('POST','/v1/jobs/'+remote_id+'/cancel').json()

    def delete(self, remote_id):
        try: return self.call('DELETE','/v1/jobs/'+remote_id).json()
        except RenderNodeError: return None

    def mirror(self, jobdir, remote_id, on_running=None, interval=1.0, patience=600):
        """Follow a remote job until it ends, writing progress.json and run.log as a local worker would.

        Network hiccups are tolerated for ``patience`` seconds; the node keeps rendering meanwhile.
        Returns the node's final job view with the light files (tokens, plan, result) downloaded into
        ``jobdir``; the audio files are listed under ``pending`` and stay on the node until fetched.
        """
        jobdir=Path(jobdir); offset=0; running_reported=False; unreachable_since=None; final=None
        while final is None:
            try:
                view=self.status(remote_id,offset)
                unreachable_since=None
            except RenderNodeError as e:
                if unreachable_since is None:
                    unreachable_since=time.time()
                    with (jobdir/'run.log').open('a') as log: log.write('[remote] Waiting for the render node: '+str(e)+'\n')
                elif time.time()-unreachable_since>patience:
                    raise RenderNodeError('Lost the render node for '+str(int(patience//60))+' minutes. The job may still finish there.') from e
                time.sleep(min(10,interval*3));continue
            if view.get('log'):
                with (jobdir/'run.log').open('a') as log: log.write(view['log'])
            offset=view.get('log_offset',offset)
            if view.get('progress') is not None:
                tmp=jobdir/'progress.tmp';tmp.write_text(json.dumps(view['progress']));replace_file(tmp,jobdir/'progress.json')
            if view['status'] in {'running','cancelling'} and not running_reported and on_running:
                running_reported=True; on_running(view.get('started'))
            if view['status'] in TERMINAL: final=view
            else: time.sleep(interval)
        final['pending']=[name for name in final.get('files',[]) if is_heavy(name)]
        for name in final.get('files',[]):
            if not is_heavy(name): self.fetch(remote_id,name,jobdir/name)
        return final

    def fetch_pending(self, jobdir, remote):
        """Bring the audio home: WAV before MP3 so the Studio's MP3 freshness check holds. Updates remote.json."""
        jobdir=Path(jobdir)
        for name in sorted(remote.get('pending',[]),key=lambda n:(not n.endswith('.wav'),n)):
            if not (jobdir/name).is_file(): self.fetch(remote['id'],name,jobdir/name)
            remote['pending']=[n for n in remote['pending'] if n!=name]
            tmp=jobdir/'remote.tmp'; tmp.write_text(json.dumps(remote)); replace_file(tmp,jobdir/'remote.json')
        return remote
