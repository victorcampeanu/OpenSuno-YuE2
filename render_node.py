"""OpenSuno Render Node: the hardware API.

Runs on the machine with the GPU. A Studio anywhere sends it a job bundle (request plus input files),
polls for progress, then streams or fetches the finished files. The node keeps no library and holds no OpenAI key;
it only owns the model weights, the resident model processes and the jobs in flight.

    OPENSUNO_NODE_TOKEN   shared secret; requests need "Authorization: Bearer <token>".
                          Unset: no auth, and only loopback clients are accepted.
    OPENSUNO_NODE_HOST    bind address (default 0.0.0.0 with a token, 127.0.0.1 without).
    OPENSUNO_NODE_PORT    default 7863.
    OPENSUNO_NODE_IDLE_MINUTES  release resident model processes after this idle time (default 30, 0 = never).
    OPENSUNO_NODE_KEEP_DAYS     keep finished jobs (WAV, MP3, tokens) this long for Studios to stream or fetch (default 30).

Finished songs stay here: the Studio streams the MP3 the node encodes after each render and fetches the
WAV only when someone asks to download it, which keeps slow links (VPN, internet) usable.
"""
import json, os, re, secrets, shutil, sys, threading, time
from pathlib import Path
from typing import Literal
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field
import audio_export
import hardware
import recordings
import loras
import style_ai
from model_downloads import ModelDownloads
from worker_pool import WorkerPool

ROOT=Path(__file__).resolve().parent
PROTOCOL=2
JOBS=ROOT/'.node-jobs'
UP=ROOT/'uploads'  # worker.py reads recordings from ROOT/uploads
JOBS.mkdir(exist_ok=True); UP.mkdir(exist_ok=True)
TOKEN=os.environ.get('OPENSUNO_NODE_TOKEN','').strip()
IDLE_MINUTES=float(os.environ.get('OPENSUNO_NODE_IDLE_MINUTES','30') or 0)
KEEP_SECONDS=float(os.environ.get('OPENSUNO_NODE_KEEP_DAYS','30') or 30)*86400
ANALYSIS_KINDS={'transcribe','tokenize'}
KINDS={'generate','cover','transcribe','tokenize','plan'}
# Side files a Studio prepares for the worker alongside request.json.
INPUT_FILES={'analysis.json','render-input.json','continue-input.json'}
UPLOAD_NAME=re.compile(r'[a-f0-9]{24}\.[a-z0-9]+')
JOB_ID=re.compile(r'[0-9]{8}-[0-9]{6}-[a-f0-9]{8}')
# Bookkeeping the Studio already mirrors or never needs; everything else in a finished job is downloadable.
PRIVATE_FILES={'state.json','request.json','progress.json','progress.tmp','run.log','.cancel-resident'}
ACTIVE_STATES={'queued','starting','running','encoding','cancelling'}

app=FastAPI(docs_url=None,redoc_url=None,title='OpenSuno Render Node')
app.include_router(style_ai.router,prefix='/v1')  # OpenAI styles and album art: the key lives on the node
downloads=ModelDownloads(ROOT)
gpu=hardware.detect_gpu()
FFMPEG=shutil.which('ffmpeg')
if not FFMPEG: print('[node] ffmpeg is missing: songs will not get an MP3 for streaming. Run the installer again.',file=sys.stderr,flush=True)

def write_state(p,status,**kw):
    data={'status':status,'updated':time.time(),**kw}
    tmp=p/'state.tmp';tmp.write_text(json.dumps(data));tmp.replace(p/'state.json')

def read_state(p):
    return json.loads((p/'state.json').read_text())

class Runner:
    """One job at a time on this GPU, in arrival order; resident model processes stay warm between jobs."""
    def __init__(self):
        self.lock=threading.RLock(); self.pool=WorkerPool(ROOT)
        self.queue=[]; self.active=None; self.process=None
        self.idle_since=time.time(); self.started=time.time(); self.completed=0

    def submit(self,p):
        with self.lock:
            if self.active:
                write_state(p,'queued'); self.queue.append(p.name)
                return {'status':'queued','position':len(self.queue)}
            self.launch(p)
            return {'status':'starting','position':0}

    def launch(self,p):
        write_state(p,'starting'); self.active=p.name
        threading.Thread(target=self.run,args=(p,),daemon=True).start()

    def run(self,p):
        try:
            r=json.loads((p/'request.json').read_text())
            (p/'run.log').write_text('[node] Rendering on '+gpu['name']+'\n')
            started=time.time(); code=0
            with self.lock:
                if read_state(p)['status'] in {'cancelled','cancelling'}:
                    write_state(p,'cancelled');return
                proc,marker=self.pool.submit(p,r['kind'] in ANALYSIS_KINDS)
                self.process=proc
                write_state(p,'running',started=started)
            code=self.pool.wait(proc,marker)
            with self.lock:
                self.process=None
                if code and code!=2: self.pool.discard(proc)
                cancelled=read_state(p)['status'] in {'cancelled','cancelling'}
                if cancelled: write_state(p,'cancelled')
                elif code:
                    error_file=p/'runtime-error.json'
                    if error_file.exists(): error=json.loads(error_file.read_text()).get('error','Generation failed.')
                    elif code<0: error='The model process stopped with signal '+str(-code)+'.'
                    else: error='The model process exited with code '+str(code)+'. See run.log in the job folder for details.'
                    write_state(p,'failed',error=error)
                    with (p/'run.log').open('a') as log: log.write('\n[error] '+error+'\n')
                else:
                    write_state(p,'encoding',started=started)
            if not code and not cancelled:
                self.encode(p)
                with self.lock:
                    if read_state(p)['status']=='encoding': write_state(p,'complete',started=started)
        except Exception as e:
            with self.lock:
                if self.process: self.pool.discard(self.process)
                cancelled=read_state(p)['status'] in {'cancelled','cancelling'}
                write_state(p,'cancelled' if cancelled else 'failed',error=str(e))
        finally:
            with self.lock: self.active=None;self.process=None;self.idle_since=time.time();self.completed+=1
            self.start_next()

    def encode(self,p):
        """An MP3 next to every rendered WAV, so Studios can play the song without moving the WAV."""
        wavs=sorted(f for f in p.rglob('audio.wav') if f.is_file())
        if not wavs: return
        with (p/'run.log').open('a') as log: log.write('[node] Encoding MP3\n')
        for wav in wavs:
            try: audio_export.to_mp3(wav)
            except Exception as e:
                with (p/'run.log').open('a') as log: log.write('[node] MP3 not encoded: '+str(e)+'\n')
                break

    def start_next(self):
        with self.lock:
            if self.active: return
            while self.queue:
                p=JOBS/self.queue.pop(0)
                if not (p/'state.json').is_file() or read_state(p)['status']!='queued': continue
                self.launch(p);return

    def cancel(self,p):
        with self.lock:
            if p.name in self.queue:
                self.queue.remove(p.name); write_state(p,'cancelled'); return True
            if self.active!=p.name: return False
            write_state(p,'cancelling' if self.process else 'cancelled')
            if self.process:
                (p/'.cancel-resident').touch()
                proc=self.process
                def stop_unresponsive():
                    time.sleep(5)
                    with self.lock:
                        if self.active==p.name and self.process is proc: self.pool.discard(proc)
                threading.Thread(target=stop_unresponsive,daemon=True).start()
            return True

    def release_idle(self):
        tick=0
        while True:
            time.sleep(60); tick+=1
            with self.lock:
                if IDLE_MINUTES and not self.active and self.pool.workers and time.time()-self.idle_since>IDLE_MINUTES*60:
                    self.pool.close()
            if tick%60==0: prune_jobs()

def prune_jobs():
    """Finished jobs older than the retention period go; Studios that still want the WAV must have fetched it by then."""
    for p in list(JOBS.iterdir()):
        if not (p/'state.json').is_file(): continue
        try: st=read_state(p)
        except (OSError,ValueError): continue
        if st['status'] not in ACTIVE_STATES and time.time()-st.get('updated',0)>KEEP_SECONDS:
            shutil.rmtree(p,ignore_errors=True)

runner=Runner()
threading.Thread(target=runner.release_idle,daemon=True).start()

for p in JOBS.iterdir():
    if not (p/'state.json').is_file(): continue
    try: st=read_state(p)
    except (OSError,ValueError): continue
    if st['status'] in ACTIVE_STATES:
        write_state(p,'interrupted',error='The render node restarted before this job finished.')
prune_jobs()

@app.middleware('http')
async def authorize(req:Request,call_next):
    if TOKEN:
        supplied=req.headers.get('authorization','')
        if not (supplied.startswith('Bearer ') and secrets.compare_digest(supplied[7:].strip(),TOKEN)):
            return JSONResponse({'detail':'Invalid render node token'},401)
    elif req.headers.get('host','').split(':')[0] not in {'127.0.0.1','localhost','testserver','[::1]'}:
        return JSONResponse({'detail':'This render node has no token and accepts local connections only. Set OPENSUNO_NODE_TOKEN.'},403)
    return await call_next(req)

def folder(j):
    if not JOB_ID.fullmatch(j) or not (JOBS/j).is_dir(): raise HTTPException(404,'Job not found')
    return JOBS/j

def job_view(p,log_offset=0):
    st=read_state(p); st['id']=p.name
    if (p/'progress.json').is_file():
        try: st['progress']=json.loads((p/'progress.json').read_text())
        except (OSError,ValueError): pass
    st['log']='';st['log_offset']=log_offset
    if (p/'run.log').is_file():
        with (p/'run.log').open('rb') as f:
            f.seek(0,2);size=f.tell()
            if log_offset>size: log_offset=0  # the log was rewritten; send it whole
            f.seek(log_offset);st['log']=f.read().decode(errors='replace');st['log_offset']=size
    if st['status'] not in ACTIVE_STATES:
        st['files']=[str(f.relative_to(p)) for f in p.rglob('*') if f.is_file() and f.name not in PRIVATE_FILES and not f.name.startswith('.') and f.suffix not in {'.tmp','.part'}]
        st['kept_until']=st.get('updated',time.time())+KEEP_SECONDS
    return st

@app.get('/v1/health')
def health():
    with runner.lock:
        active=runner.active; queue=list(runner.queue); completed=runner.completed; idle=runner.idle_since; loaded=bool(runner.pool.workers)
    progress=None;title=None
    if active:
        # Live speed and stage of the running job, for monitors such as the menu bar app.
        try:
            progress=json.loads((JOBS/active/'progress.json').read_text()) if (JOBS/active/'progress.json').is_file() else None
            title=json.loads((JOBS/active/'request.json').read_text()).get('title')
        except (OSError,ValueError): pass
    return {'ok':True,'protocol':PROTOCOL,'node':'opensuno-render-node',**hardware.capabilities(ROOT,downloads,gpu),
            'ffmpeg_ready':bool(FFMPEG),'openai_configured':style_ai.openai_configured(ROOT),'busy':active is not None,'active':active,'active_title':title,'progress':progress,'queue':queue,'completed':completed,
            'idle_seconds':0 if active else round(time.time()-idle),'models_loaded':loaded,'uptime_seconds':round(time.time()-runner.started)}

@app.post('/v1/models/download')
def download_models(model:str='bf16'):
    with runner.lock:
        if runner.active: raise HTTPException(409,'Wait for the current generation to finish before downloading models.')
        try: downloads.selected_assets(model)
        except ValueError: raise HTTPException(400,'Unknown model')
        return downloads.start(model)
    with runner.lock:
        if runner.active: raise HTTPException(409,'Wait for the current generation to finish before downloading models.')
        return downloads.start(model)

def require_ready(r):
    caps=hardware.capabilities(ROOT,downloads,gpu)
    if r['kind'] not in ANALYSIS_KINDS:
        model=next((m for m in caps['models'] if m['id']==r.get('model')),None)
        if model is None: raise HTTPException(400,'This render node does not offer the model '+repr(r.get('model'))+'.')
        if not model['ready']: raise HTTPException(503,'The model '+model['label']+' is not installed on the render node. Download it in Models.')
        for lora in (r.get('lora'),r.get('sound_lora')):
            if lora and (not loras.valid_name(lora) or not loras.path_of(ROOT,lora).is_file()):
                raise HTTPException(400,'The LoRA '+repr(lora)+' is not in the loras folder of the render node.')
    elif r['kind']=='transcribe' and not caps['transcriber_ready']: raise HTTPException(503,'Cover analysis is not installed on the render node.')
    elif r['kind']=='tokenize' and not caps['tokens_ready']: raise HTTPException(503,'Audio input is not installed on the render node.')
    name=r.get('audio_id')
    if name and (not UPLOAD_NAME.fullmatch(name) or not (UP/name).is_file()): raise HTTPException(400,'The recording '+repr(name)+' was not uploaded to the render node.')
    if r['kind']=='cover' and not r.get('_analysis_present'): raise HTTPException(400,'A cover needs its analysis.json.')

@app.post('/v1/jobs')
async def create(request:str=Form(...),inputs:list[UploadFile]=File(default=[]),uploads:list[UploadFile]=File(default=[])):
    try: r=json.loads(request)
    except ValueError: raise HTTPException(400,'request must be JSON')
    if not isinstance(r,dict) or r.get('kind') not in KINDS: raise HTTPException(400,'Unknown job kind')
    j=time.strftime('%Y%m%d-%H%M%S')+'-'+secrets.token_hex(4);p=JOBS/j;p.mkdir()
    try:
        (p/'request.json').write_text(json.dumps(r,indent=2))
        for f in inputs:
            if f.filename not in INPUT_FILES: raise HTTPException(400,'Unexpected input file '+repr(f.filename))
            (p/f.filename).write_bytes(await f.read())
        for f in uploads:
            if not UPLOAD_NAME.fullmatch(f.filename or ''): raise HTTPException(400,'Unexpected upload name')
            target=UP/f.filename
            if not target.is_file():
                tmp=target.with_suffix(target.suffix+'.part')
                with tmp.open('wb') as out:
                    while chunk:=await f.read(1024*1024): out.write(chunk)
                tmp.replace(target)
        r['_analysis_present']=(p/'analysis.json').is_file()
        require_ready(r)
    except HTTPException:
        shutil.rmtree(p,ignore_errors=True);raise
    return {'id':j,**runner.submit(p)}

@app.get('/v1/jobs')
def jobs():
    with runner.lock: return {'active':runner.active,'queue':list(runner.queue)}

@app.get('/v1/jobs/{j}')
def getjob(j:str,log_offset:int=0):
    p=folder(j)
    with runner.lock: return job_view(p,max(0,log_offset))

@app.get('/v1/jobs/{j}/files/{name:path}')
def file(j:str,name:str):
    p=folder(j);target=(p/name).resolve()
    if not target.is_relative_to(p.resolve()) or not target.is_file() or target.name in PRIVATE_FILES: raise HTTPException(404)
    return FileResponse(target,filename=target.name)

# Recordings: the Studio sends the browser's file here untouched; decoding, waveforms and cuts happen on this machine.
@app.post('/v1/uploads')
async def upload(file:UploadFile=File(...)):
    return await recordings.save(UP,file)

@app.get('/v1/uploads/{name}')
def upload_file(name:str):
    if not recordings.NAME.fullmatch(name) or not (UP/name).is_file(): raise HTTPException(404,'Recording not found')
    return FileResponse(UP/name,filename=name)

@app.get('/v1/uploads/{name}/tags')
def upload_tags(name:str):
    return recordings.embedded_tags(recordings.editable(UP,name))

@app.get('/v1/uploads/{name}/cover')
def upload_cover(name:str):
    dest=recordings.cover_png(recordings.editable(UP,name))
    if not dest: raise HTTPException(404,'No cover art in this recording')
    return FileResponse(dest,media_type='image/png',filename='artwork.png')

@app.get('/v1/uploads/{name}/waveform')
def upload_waveform(name:str):
    return recordings.waveform(recordings.editable(UP,name))

class Trim(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)
    start:float=Field(ge=0,le=1800)
    end:float=Field(gt=0,le=1800)

@app.post('/v1/uploads/{name}/trim')
def upload_trim(name:str,r:Trim):
    return recordings.trim(recordings.editable(UP,name),r.start,r.end)

@app.post('/v1/uploads/{name}/song')
def upload_song(name:str):
    """Decode a recording into a finished job (WAV + MP3) so a Studio can list it as a song and stream it from here."""
    source=recordings.editable(UP,name)
    j=time.strftime('%Y%m%d-%H%M%S')+'-'+secrets.token_hex(4);p=JOBS/j;p.mkdir()
    try:
        seconds=recordings.probe_seconds(source)
        tags=recordings.embedded_tags(source)
        (p/'request.json').write_text(json.dumps({'kind':'upload','audio_id':name,'lyrics':tags.get('lyrics') or ''}))
        recordings.decode_to_wav(source,p/'audio.wav')
        audio_export.to_mp3(p/'audio.wav')
        if recordings.embedded_cover(source,p/'artwork.png'):
            (p/'artwork.json').write_text(json.dumps({'status':'complete','source':'embedded'}))
        (p/'result.json').write_text(json.dumps({'seconds':round(seconds,3),'uploaded':True}))
        write_state(p,'complete')
    except Exception as e:
        shutil.rmtree(p,ignore_errors=True)
        raise HTTPException(400,'Could not decode the recording. '+(str(e) if isinstance(e,ValueError) else 'Try uploading it again.')) from e
    with runner.lock: view=job_view(p)
    return {**view,'seconds':round(seconds,3),'lyrics':tags.get('lyrics') or ''}

@app.post('/v1/jobs/{j}/cancel')
def cancel(j:str):
    p=folder(j)
    if not runner.cancel(p) and read_state(p)['status'] in ACTIVE_STATES: raise HTTPException(409,'This job is no longer running')
    return {'ok':True}

@app.delete('/v1/jobs/{j}')
def delete(j:str):
    p=folder(j)
    with runner.lock:
        if runner.active==j: raise HTTPException(409,'Stop this job before deleting it')
        if j in runner.queue: runner.queue.remove(j)
    shutil.rmtree(p,ignore_errors=True)
    return {'ok':True}

@app.on_event('shutdown')
def shutdown():
    with runner.lock:
        if runner.active: write_state(JOBS/runner.active,'interrupted',error='The render node was stopped while this job was running.')
        for j in runner.queue: write_state(JOBS/j,'interrupted',error='The render node was stopped while this job was waiting.')
        runner.queue.clear(); runner.pool.close()

if __name__=='__main__':
    import uvicorn
    host=os.environ.get('OPENSUNO_NODE_HOST') or ('0.0.0.0' if TOKEN else '127.0.0.1')
    port=int(os.environ.get('OPENSUNO_NODE_PORT','7863'))
    print(f'OpenSuno Render Node on {gpu["name"]} ({gpu["kind"]}) · http://{host}:{port}/v1/health'+('' if TOKEN else ' · no token: local connections only'),flush=True)
    uvicorn.run(app,host=host,port=port,timeout_graceful_shutdown=2)
