"""YuE2 Studio. Local Studio binds to 127.0.0.1; hosted mode serves the website on Vercel."""
import asyncio, base64, json, os, re, secrets, signal, subprocess, sys, threading, time
from pathlib import Path
from typing import Literal
from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator, ConfigDict
# Vercel imports this file as app/server.py without putting app/ on the path.
if str(Path(__file__).resolve().parent) not in sys.path: sys.path.insert(0,str(Path(__file__).resolve().parent))
from lyrics_provider import router as lyrics_router
import style_ai
from style_ai import ArtworkAsk, generate_artwork, openai_configured
import recordings
from model_downloads import ModelDownloads
from worker_pool import WorkerPool, PageLeases
from model_preloads import ModelPreloads
import song_library
import audio_edit
import audio_export
import audio_speed
import composition_edit
import instrumental
import song_continuation
import loras
import uploaded_songs
from analysis_cache import AnalysisCache, TokenCache
from workspaces import Workspaces, WorkspaceError
from prompts import Prompts, PromptError
from model_registry import available_models, default_model, is_cuda
import hardware
import render_client
from render_client import RenderNode, RenderNodeError

ROOT=Path(__file__).resolve().parents[1]
def _flag(name):
    return os.environ.get(name,'').strip().lower() in {'1','true','yes'}
HOSTED=_flag('OPENSUNO_HOSTED') if 'OPENSUNO_HOSTED' in os.environ else _flag('VERCEL')
DATA_ROOT=Path(os.environ['OPENSUNO_DATA']) if os.environ.get('OPENSUNO_DATA') else ROOT
LIB=DATA_ROOT/'library'; UP=DATA_ROOT/'uploads'
try:
    LIB.mkdir(parents=True,exist_ok=True); UP.mkdir(exist_ok=True)
except OSError:
    DATA_ROOT=Path('/tmp/opensuno'); LIB=DATA_ROOT/'library'; UP=DATA_ROOT/'uploads'
    LIB.mkdir(parents=True,exist_ok=True); UP.mkdir(exist_ok=True)
app=FastAPI(docs_url=None,redoc_url=None)
app.include_router(lyrics_router)
lock=threading.RLock(); active=None; process=None
queue=[]; pending={}  # job ids waiting for the worker, in order, and their parsed requests
pool=WorkerPool(ROOT)
preloads=ModelPreloads(ROOT,pool,lock)
analyses=AnalysisCache(DATA_ROOT/'.analysis-cache')
recording_tokens=TokenCache(DATA_ROOT/'.analysis-cache')
# Jobs that run in the audio-analysis environment instead of a generation model.
ANALYSIS_KINDS={'transcribe','tokenize'}
def save_analysis(r,p):
    """Keep what a finished recording analysis found."""
    (recording_tokens if r.kind=='tokenize' else analyses).save(r,json.loads((p/'result.json').read_text()))
def release_residents():
    # A render node keeps working when the pages close; only this computer's resident models are released.
    if not remote.configured():
        if active:
            state(LIB/active,'cancelling',error='All OpenSuno pages were closed.')
        drop_queue('cancelled',error='All OpenSuno pages were closed.')
    pool.close()
    preloads.clear()
leases=PageLeases(lock,release_residents)
TOKEN=secrets.token_urlsafe(32)
DEFAULTS=json.loads((ROOT/'config/generation-defaults.json').read_text())
downloads=ModelDownloads(ROOT)
GPU=hardware.detect_gpu()
RENDER_SETTINGS=DATA_ROOT/'.render-node.json'

def render_settings():
    """Where songs render: empty url = this computer. The environment overrides the saved setting."""
    if 'OPENSUNO_RENDER_URL' in os.environ:
        return {'url':os.environ['OPENSUNO_RENDER_URL'].strip().rstrip('/'),'token':os.environ.get('OPENSUNO_RENDER_TOKEN','').strip(),'locked':True}
    try: saved=json.loads(RENDER_SETTINGS.read_text())
    except (OSError,ValueError): saved={}
    return {'url':str(saved.get('url') or '').strip().rstrip('/'),'token':str(saved.get('token') or ''),'locked':False}

class RemoteRendering:
    """The connected render node, with a briefly cached health snapshot so polling /api/config stays cheap."""
    def __init__(self):
        self.guard=threading.Lock(); self.node=None; self.health=None; self.error=None; self.checked=0
        self.reload()
    def reload(self):
        s=render_settings()
        with self.guard:
            self.node=RenderNode(s['url'],s['token']) if s['url'] else None
            self.health=None; self.error=None; self.checked=0
    def configured(self):
        return self.node is not None
    def snapshot(self,max_age=3):
        with self.guard:
            node=self.node
            if node is None: return None
            if time.time()-self.checked<max_age: return self.health
        try: health,error=node.health(timeout=3),None
        except RenderNodeError as e: health,error=None,str(e)
        with self.guard:
            if self.node is node: self.health,self.error,self.checked=health,error,time.time()
        return health
    def describe(self):
        s=render_settings(); health=self.snapshot()
        return {'url':s['url'],'token':s['token'],'locked':s['locked'],'connected':health is not None,'error':self.error if s['url'] else None,
                'gpu':(health or {}).get('gpu'),'platform':(health or {}).get('platform'),'busy':(health or {}).get('busy'),'ffmpeg_ready':(health or {}).get('ffmpeg_ready',True),
                'queue':len((health or {}).get('queue',[])),'models':[m['id'] for m in (health or {}).get('models',[]) if m.get('ready')]}
remote=RemoteRendering()
workspaces=Workspaces(DATA_ROOT/'.workspaces.json')
prompts=Prompts(DATA_ROOT/'.prompts.json')
HOSTED_UNAVAILABLE='This hosted preview serves the website only. Music generation is not connected yet.'

class Sampling(BaseModel):
    model_config=ConfigDict(extra='forbid')
    temperature:float=Field(ge=0,le=3)
    top_p:float=Field(gt=0,le=1)
    top_k:int=Field(ge=1,le=184704)
    repetition_penalty:float=Field(gt=0,le=3)
    penalty_window:int=Field(ge=1,le=18000)
    min_tokens:int=Field(ge=0,le=18000)
    max_tokens:int=Field(ge=1,le=18000)
    @model_validator(mode='after')
    def valid(self):
        if self.min_tokens>self.max_tokens: raise ValueError('Minimum tokens cannot exceed maximum tokens')
        return self
RETIRED_REQUEST_KEYS=('align_lyrics','synced_lyrics','source_start','persona_id','keep_sound','latents_id','created','embedded_applied')

class Job(BaseModel):
    model_config=ConfigDict(extra='forbid')
    workspace_id:str=Field(default='default',max_length=24)
    render_source:str=''
    render_candidate:int=Field(default=1,ge=1,le=8)
    # Re-render the saved tokens without the song's LoRA: keeps what a writing+sound adapter wrote, drops what it does to the sound.
    render_without_lora:bool=False
    edit_id:str=''
    # Extend / regenerate from here: continue a saved version from a point in its timeline (None = its end).
    continue_source:str=''
    continue_candidate:int=Field(default=1,ge=1,le=8)
    continue_seconds:float|None=Field(default=None,ge=0,le=1800)
    # Extend an uploaded recording (audio_id, source_seconds) from its music tokens; the kept part is rendered too.
    continue_recording:bool=False
    # Written by the server for songs made from other songs; ignored when supplied by a client.
    origin:dict|None=None
    background_analysis:bool=False
    kind:Literal['generate','cover','transcribe','tokenize','plan']='generate'
    model:Literal['bf16','cuda-bf16','cuda-fp8']=default_model()
    candidates:int=Field(default=1,ge=1,le=8)
    title:str=Field(default='Untitled song',max_length=100)
    style:str=Field(default='',max_length=6000)
    lyrics:str=Field(default='',max_length=30000)
    instrumental:bool=False
    cot:Literal['full','melody','off']='full'
    voice:Literal['any','male','female','duet']='any'
    abc:str=Field(default='',max_length=40000)
    seed:int=Field(default=831001,ge=0,le=4294967295)
    random_seed:bool=False
    cfg_scale:float|None=Field(default=1.2,ge=0,le=10)
    steps:int=Field(default=2,ge=1,le=128)
    abc_sampling:Sampling=Field(default_factory=lambda:Sampling(**DEFAULTS['abc']))
    semantic_sampling:Sampling=Field(default_factory=lambda:Sampling(**DEFAULTS['semantic']))
    audio_id:str=''
    melody_only:bool=True
    source_seconds:float=Field(default=0,ge=0,le=1800)
    # Covers: the model writes chords and instrumental lines around the transcribed melody before generating.
    arrange:bool=True
    # Covers: the transcribed melody is moved by whole octaves into the requested voice's register ('auto'),
    # kept as heard, or moved down one or two / up one on request. The transcriber writes low male voices an octave up.
    vocal_octave:Literal['auto','keep','down','down2','up']='auto'
    # Instrumental songs: the instrument plays the sung melody only in chorus / hook sections; the verses
    # are left to their chords so the model arranges them fully instead of shadowing a lead line throughout.
    hook_melody:bool=False
    # Community LoRAs from the loras folder (file names) folded into the model for this song, and how strongly:
    # the LoRA shapes the writing (or anything), the Sound LoRA is a second slot meant for decoder adapters.
    lora:str=Field(default='',max_length=160)
    lora_strength:float=Field(default=1.0,ge=0,le=2)
    sound_lora:str=Field(default='',max_length=160)
    sound_lora_strength:float=Field(default=1.0,ge=0,le=2)
    api_key:str=Field(default='',max_length=256,exclude=True)
    @model_validator(mode='before')
    @classmethod
    def drop_retired(cls,data):
        # Settings of removed features (lyric matching, personas, keep-sound) and upload bookkeeping still sit in
        # older saved requests that re-render, Extend and a render-node resume post back.
        if isinstance(data,dict):
            for key in RETIRED_REQUEST_KEYS: data.pop(key,None)
        return data
    @model_validator(mode='after')
    def valid(self):
        if self.background_analysis and self.kind not in ANALYSIS_KINDS: raise ValueError('Background analysis must be a recording analysis job')
        if is_cuda(self.model) and self.kind not in ANALYSIS_KINDS and max(self.abc_sampling.penalty_window,self.semantic_sampling.penalty_window)>100:
            raise ValueError('The official CUDA runtime supports Lookback up to 100 tokens. Lower Lookback or select an MLX model.')
        if self.kind in {'generate','cover','plan'}:
            if not self.style.strip() or (not self.instrumental and not self.lyrics.strip()): raise ValueError('Enter a style and lyrics, or enable Instrumental only')
            if self.abc.strip() and self.cot=='off': raise ValueError('A supplied score needs Melody or Melody + chords mode')
            if self.instrumental and self.cot=='off' and self.kind!='cover': raise ValueError('Instrumental songs need a Melody or Melody + chords plan so the vocal voice can be kept silent')
        if self.kind=='plan' and self.cot=='off': raise ValueError('Select Melody only or Melody + chords to plan a score')
        for name in (self.lora,self.sound_lora):
            if not name: continue
            if not loras.valid_name(name): raise ValueError('Not a LoRA file name: '+repr(name))
            if self.kind in {'generate','cover','plan'} and not lora_known(name): raise ValueError('The LoRA '+repr(name)+' is not in the loras folder'+(' of the render node' if remote.configured() else ''))
        if self.lora and self.lora==self.sound_lora: raise ValueError('The LoRA and the Sound LoRA are the same file; choose one of them or two different adapters')
        if self.render_source and not self.edit_id:
            # A re-render reuses the saved tokens of the source version; the Extend / recording flags that arrive
            # with its settings describe how those tokens were made and are rebuilt from the source job.
            self.continue_source='';self.continue_seconds=None;self.continue_recording=False
        if self.continue_source and self.kind!='generate': raise ValueError('Extend applies to song generation')
        if self.continue_source and (self.render_source or self.edit_id): raise ValueError('Extend cannot be combined with re-rendering or saved compositions')
        if self.continue_recording:
            if self.kind!='generate': raise ValueError('Extending a recording applies to song generation')
            if self.continue_source or self.render_source or self.edit_id: raise ValueError('Extending a recording cannot be combined with Extend, re-rendering or saved compositions')
            if not self.audio_id: raise ValueError('Upload a recording first')
            if is_cuda(self.model): raise ValueError('Extending a recording runs on the MLX models for now')
            # Recordings have no plan: the song is written score-free after the recording's tokens, so the
            # vocal voice cannot be silenced by a plan.
            if self.instrumental: raise ValueError('Extending a recording is score-free; turn off Instrumental only')
            self.cot='off';self.abc=''
        if self.kind not in {'generate','cover'} and self.candidates!=1: raise ValueError('Multiple candidates apply only to song generation')
        # With a render node the recording lives only on the node, which checks for it when the job arrives.
        val=self.audio_id
        if val and (not re.fullmatch(r'[a-f0-9]{24}\.[a-z0-9]+',val) or Path(val).suffix not in {'.wav','.mp3','.flac','.m4a','.aiff','.aif','.ogg','.aac'} or not ((UP/val).is_file() or remote.configured())):
            raise ValueError('Upload a valid source file first')
        if self.kind in {'transcribe','tokenize','cover'} and not self.audio_id: raise ValueError('Upload a source recording first')
        if self.kind=='cover':
            self.abc=''
            self.cot='melody' if self.melody_only else 'full'
        return self

def folder(j):
    if not re.fullmatch(r'[0-9]{8}-[0-9]{6}-[a-f0-9]{8}',j) or not (LIB/j).is_dir(): raise HTTPException(404,'Result not found')
    return LIB/j

def read_remote(p):
    """Where this job rendered and which of its files still live only on the node."""
    try: data=json.loads((p/'remote.json').read_text())
    except (OSError,ValueError): return None
    return data if isinstance(data,dict) and data.get('id') else None

def write_remote(p,data):
    tmp=p/'remote.tmp';tmp.write_text(json.dumps(data));tmp.replace(p/'remote.json')

def remote_pending(p,name=None):
    """The node still holding ``name`` (or any audio) for this job, with the saved remote record."""
    saved=read_remote(p)
    if not saved or not saved.get('pending'): return None
    if name is not None and name not in saved['pending']: return None
    return saved

def node_for(saved):
    """The connected node if it is the one that rendered the job; otherwise a client for the saved address."""
    if remote.configured() and remote.node.url==saved['url']: return remote.node
    return RenderNode(saved['url'],render_settings()['token'] if render_settings()['url']==saved['url'] else '')

def fetch_audio(p):
    """Bring a job's audio from the node into the library (download, export, editing). Blocks while it copies."""
    saved=remote_pending(p)
    if not saved: return
    try: node_for(saved).fetch_pending(p,saved)
    except RenderNodeError as e: raise HTTPException(502,'Could not fetch the audio from the render node: '+str(e)) from e

def backfill_upload_job(p, request):
    """Once: copy lyrics and cover art out of an already-listed recording (local file or the render node)."""
    if request.get('embedded_applied'):
        return
    name=request.get('audio_id') or ''
    if not recordings.NAME.fullmatch(name):
        uploaded_songs.mark_embedded(p); return
    source=UP/name
    try:
        if source.is_file():
            uploaded_songs.apply_lyrics(p, recordings.embedded_tags(source).get('lyrics', ''))
            uploaded_songs.attach_cover(p, source)
        elif remote.configured():
            try: tags=via_node('GET','/v1/uploads/'+name+'/tags')
            except HTTPException:
                uploaded_songs.mark_embedded(p); return
            uploaded_songs.apply_lyrics(p, (tags or {}).get('lyrics', ''))
            if not uploaded_songs.has_artwork(p):
                try:
                    body=remote.node.call('GET','/v1/uploads/'+name+'/cover',timeout=30).content
                    if body and body[:8]==recordings.PNG_MAGIC:
                        tmp=p/'artwork.png.part'; tmp.write_bytes(body); tmp.replace(p/'artwork.png')
                        uploaded_songs.write_artwork_meta(p, True)
                    else:
                        uploaded_songs.write_artwork_meta(p, False)
                except RenderNodeError as e:
                    if getattr(e,'status',0)==404: uploaded_songs.write_artwork_meta(p, False)
                    else: return
        else:
            uploaded_songs.mark_embedded(p); return
        uploaded_songs.mark_embedded(p)
    except Exception as error:
        # A recording whose tags cannot be read is listed without them rather than retried on every listing.
        print(f'[library] Could not read the tags of {name}: {error!r}', file=sys.stderr)
        uploaded_songs.mark_embedded(p)

def readjob(p):
    state=json.loads((p/'state.json').read_text()); state['id']=p.name
    request=json.loads((p/'request.json').read_text())
    if request.get('kind')==uploaded_songs.KIND and not request.get('embedded_applied'):
        backfill_upload_job(p, request)
        try: request=json.loads((p/'request.json').read_text())
        except (OSError,ValueError): pass
    state['title']=request['title'];state['kind']=request['kind']
    request.setdefault('model','bf16');request.setdefault('candidates',1)
    state['request']=request
    if (p/'result.json').exists(): state['result']=json.loads((p/'result.json').read_text())
    if (p/'favorite.json').exists(): state['favorite']=json.loads((p/'favorite.json').read_text()).get('candidate')
    state['files']=[str(f.relative_to(p)) for f in p.rglob('*') if f.is_file() and f.suffix in {'.wav','.mp3','.abc','.npy','.mid','.json'} and f.name not in {'state.json','result.json','remote.json'} and not f.name.startswith('.')]
    saved=remote_pending(p)
    if saved:
        # Audio still on the node: the UI plays the MP3 from there and offers to fetch the WAV.
        state['remote']={'pending':saved['pending'],'kept_until':saved.get('kept_until'),'url':saved['url']}
        state['files']+= [n for n in saved['pending'] if n not in state['files']]
    art=None
    if (p/'artwork.json').is_file():
        try: art=json.loads((p/'artwork.json').read_text())
        except (json.JSONDecodeError,OSError): art=None
    if artwork_path(p):
        art=art if isinstance(art,dict) else {}
        art['status']='complete'
        art['url']='/api/jobs/'+p.name+'/artwork'
    if isinstance(art,dict): state['artwork']=art
    return state

ARTWORK_TYPES={'.webp':'image/webp','.png':'image/png'}
def artwork_path(p):
    # New art is saved as WebP; songs from before that still have a PNG.
    return next((f for f in (p/'artwork.webp',p/'artwork.png') if f.is_file()),None)

def state(p,status,**kw):
    data={'status':status,'updated':time.time(),**kw}
    tmp=p/'state.tmp';tmp.write_text(json.dumps(data));tmp.replace(p/'state.json')

def drop_queue(status,**kw):
    for j in queue:
        if (LIB/j).is_dir(): state(LIB/j,status,**kw)
    queue.clear();pending.clear()

@app.middleware('http')
async def local_only(req:Request,call_next):
    host=req.headers.get('host','').split(':')[0]
    if not HOSTED and host not in {'127.0.0.1','localhost','testserver'}: return JSONResponse({'detail':'Local access only'},403)
    if not HOSTED and req.method in {'POST','PUT','PATCH','DELETE'} and req.headers.get('x-studio-token')!=TOKEN:
        return JSONResponse({'detail':'Reload Studio and try again'},403)
    response=await call_next(req)
    if req.url.path=='/' or req.url.path.startswith('/static/'):
        response.headers['Cache-Control']='no-store, max-age=0'
    return response

@app.get('/')
def index(): return FileResponse(ROOT/'web/index.html')
CAPABILITY_KEYS=('models','default_model','downloads','render_environment_ready','transcriber_environment_ready','transcriber_ready','tokens_ready','loras')
def capabilities():
    """What the rendering hardware offers: this computer, or the connected render node (nothing while it is unreachable)."""
    if remote.configured():
        health=remote.snapshot()
        unreachable={'models':[{'id':v,**d,'ready':False} for v,d in available_models().items()],'default_model':default_model(),
                'downloads':{'status':'idle','packages':{},'missing':[],'lora_packages':[]},'render_environment_ready':True,
                'transcriber_environment_ready':True,'transcriber_ready':False,'tokens_ready':False,'loras':[]}
        # A node on an older build may not report every key; a node that answers can render.
        if health: return {k:health.get(k,unreachable[k]) for k in CAPABILITY_KEYS}
        return unreachable
    return {k:v for k,v in hardware.capabilities(ROOT,downloads,GPU).items() if k in CAPABILITY_KEYS}
def model_ready(name):
    return any(m['id']==name and m['ready'] for m in capabilities()['models'])
def lora_known(name):
    """Whether the named LoRA file sits in the loras folder of whatever renders (this computer or the node)."""
    if HOSTED: return False
    if remote.configured(): return any(l['id']==name and not l.get('error') for l in capabilities()['loras'])
    return loras.path_of(ROOT,name).is_file()
@app.get('/api/loras')
def list_loras():
    """LoRA files offered by the renderer, so the picker can be refreshed after a file is dropped into the folder."""
    if HOSTED: return {'loras':[],'folder':'','remote':False}
    return {'loras':capabilities()['loras'],'remote':remote.configured()}
@app.get('/api/config')
def config():
    if HOSTED:
        return {'token':TOKEN,'hosted':True,'defaults':DEFAULTS,'active':None,'queue':[],
          'default_model':'bf16','models':[{'id':'bf16','label':'YuE2','description':'Remote generation','backend':'hosted','ready':True}],
          'downloads':{'status':'idle','packages':{},'missing':[],'lora_packages':[]},'preloads':{},
          'transcriber_environment_ready':True,'render_environment_ready':True,
          'transcriber_ready':False,'tokens_ready':False,'openai_configured':openai_configured(ROOT)}
    caps=capabilities()
    return {'token':TOKEN,'hosted':False,'defaults':DEFAULTS,'active':active,'queue':list(queue),**caps,
      'preloads':{} if remote.configured() else preloads.snapshot(),
      'render_node':remote.describe(),'gpu':GPU,
      'openai_configured':ai_configured()}
def tokens_ready():
    """The Audio input package: the tokenizer head that turns recordings into music tokens (run on the
    Cover analysis MERT) and the real-audio decoder adapter that renders such tokens when a recording is extended."""
    return capabilities()['tokens_ready']

class RenderNodeSettings(BaseModel):
    model_config=ConfigDict(extra='forbid')
    url:str=Field(default='',max_length=300)
    token:str=Field(default='',max_length=300)

@app.get('/api/render-node')
def render_node():
    return remote.describe()

@app.post('/api/render-node')
def set_render_node(s:RenderNodeSettings):
    """Connect the Studio to a render node (or back to this computer with an empty url). Tested before it is saved."""
    with lock:
        if active or queue: raise HTTPException(409,'Wait for the current generation to finish before changing where songs render.')
        if render_settings()['locked']: raise HTTPException(400,'The render node is set by OPENSUNO_RENDER_URL in the environment.')
        url=s.url.strip().rstrip('/');token=s.token.strip()
        if url and not re.match(r'^https?://',url): url='http://'+url
        if url:
            try: health=RenderNode(url,token).health()
            except RenderNodeError as e: raise HTTPException(400,str(e))
            if health.get('protocol')!=render_client.PROTOCOL: raise HTTPException(400,'This render node speaks protocol '+str(health.get('protocol'))+'; update OpenSuno on both machines.')
        RENDER_SETTINGS.write_text(json.dumps({'url':url,'token':token}))
        remote.reload();preloads.clear();pool.close()
        return remote.describe()

@app.post('/api/models/preload')
def preload_model(model:Literal['bf16','cuda-bf16','cuda-fp8','analysis']='bf16'):
    if HOSTED: return {'status':'ready'}
    if remote.configured(): return {'status':'ready','detail':'Loaded by the render node when a song starts'}
    with lock:
        if not leases.pages: return {'status':'waiting'}
        known=preloads.snapshot().get(model)
        if known and known['status'] in {'loading','ready'}: return known
        if active or any(s['status']=='loading' for s in preloads.snapshot().values()):
            return {'status':'waiting'}
        ready=config()['transcriber_ready'] if model=='analysis' else model_ready(model)
        if not ready:return {'status':'missing'}
        request=Job(kind='plan',style='Preload',instrumental=True,model='bf16' if model=='analysis' else model).model_dump()
        if model=='analysis':request['kind']='transcribe'
        preloads.start(model,request)
        return preloads.snapshot().get(model,{'status':'waiting'})

@app.post('/api/models/download')
def download_models(model:str='bf16'):
    if HOSTED: raise HTTPException(404,'Model downloads are only available in the local Studio.')
    with lock:
        if active: raise HTTPException(409,'Wait for the current generation to finish before downloading models.')
        if remote.configured():
            # The node fetches its own weights from Hugging Face; nothing is uploaded from this computer.
            try: return remote.node.download_models(model)
            except RenderNodeError as e: raise HTTPException(502,str(e))
        if not hardware.render_environment_ready(ROOT):
            raise HTTPException(409,'This computer runs the interface only. Connect a render node in Settings → Rendering; models are downloaded there.')
        try: downloads.selected_assets(model)
        except ValueError: raise HTTPException(400,'Unknown model')
        return downloads.start(model)
    if HOSTED: raise HTTPException(404,'Model downloads are only available in the local Studio.')
    with lock:
        if active: raise HTTPException(409,'Wait for the current generation to finish before downloading models.')
        if remote.configured():
            # The node fetches its own weights from Hugging Face; nothing is uploaded from this computer.
            try: return remote.node.download_models(model)
            except RenderNodeError as e: raise HTTPException(502,str(e))
        if not hardware.render_environment_ready(ROOT):
            raise HTTPException(409,'This computer runs the interface only. Connect a render node in Settings → Rendering; models are downloaded there.')
        return downloads.start(model)
@app.get('/api/jobs')
def jobs():
    with lock:
        result=[]
        for p in sorted(LIB.iterdir(),reverse=True):
            if not (p/'state.json').is_file() or not (p/'request.json').is_file():
                continue
            try:
                j=readjob(p)
            except (FileNotFoundError,json.JSONDecodeError,KeyError):
                # An interrupted write or incomplete job must not hide the library.
                continue
            if not j['request'].get('background_analysis'):result.append(j)
        return result

class WorkspaceName(BaseModel):
    model_config=ConfigDict(extra='forbid')
    name:str=Field(min_length=1,max_length=80)

class WorkspaceSong(BaseModel):
    model_config=ConfigDict(extra='forbid')
    job:str=Field(pattern=r'^[0-9]{8}-[0-9]{6}-[a-f0-9]{8}$')
    candidate:int=Field(ge=0,le=8)

class WorkspaceMove(BaseModel):
    model_config=ConfigDict(extra='forbid')
    workspace_id:str=Field(min_length=1,max_length=24)
    songs:list[WorkspaceSong]=Field(min_length=1,max_length=1000)

@app.get('/api/workspaces')
def list_workspaces():
    with lock: return workspaces.read()

@app.post('/api/workspaces')
def create_workspace(value:WorkspaceName):
    with lock:
        try: return workspaces.create(value.name)
        except WorkspaceError as e: raise HTTPException(400,str(e))

@app.patch('/api/workspaces/{workspace_id}')
def rename_workspace(workspace_id:str,value:WorkspaceName):
    with lock:
        try: return workspaces.rename(workspace_id,value.name)
        except WorkspaceError as e: raise HTTPException(400,str(e))

@app.delete('/api/workspaces/{workspace_id}')
def delete_workspace(workspace_id:str):
    with lock:
        try: workspaces.delete(workspace_id)
        except WorkspaceError as e: raise HTTPException(400,str(e))
        return {'ok':True}

class PromptSettings(BaseModel):
    model_config=ConfigDict(extra='ignore')
    instrumental:bool=False
    hook_melody:bool=False
    voice:Literal['any','male','female','duet']='any'
    cot:Literal['full','melody','off']='full'
    candidates:int=Field(default=1,ge=1,le=8)
    cfg_scale:float=Field(default=1.2,ge=0,le=10)
    steps:int=Field(default=2,ge=1,le=128)
    semantic_sampling:Sampling|None=None
    abc_sampling:Sampling|None=None
    lora:str=Field(default='',max_length=160)
    lora_strength:float=Field(default=1.0,ge=0,le=2)
    sound_lora:str=Field(default='',max_length=160)
    sound_lora_strength:float=Field(default=1.0,ge=0,le=2)

class PromptWrite(BaseModel):
    model_config=ConfigDict(extra='forbid')
    title:str=Field(min_length=1,max_length=80)
    style:str=Field(default='',max_length=6000)
    settings:PromptSettings=Field(default_factory=PromptSettings)

class PromptName(BaseModel):
    model_config=ConfigDict(extra='forbid')
    title:str=Field(min_length=1,max_length=80)

@app.get('/api/prompts')
def list_prompts():
    with lock: return {'prompts':prompts.listing()}

@app.post('/api/prompts')
def create_prompt(value:PromptWrite):
    with lock:
        try: return prompts.create(value.title,value.style,value.settings.model_dump())
        except PromptError as e: raise HTTPException(400,str(e))

@app.patch('/api/prompts/{prompt_id}')
def rename_prompt(prompt_id:str,value:PromptName):
    with lock:
        try: return prompts.rename(prompt_id,value.title)
        except PromptError as e: raise HTTPException(400,str(e))

@app.delete('/api/prompts/{prompt_id}')
def delete_prompt(prompt_id:str):
    with lock:
        try: prompts.delete(prompt_id)
        except PromptError as e: raise HTTPException(400,str(e))
        return {'ok':True}

@app.post('/api/workspaces/move')
def move_workspace_songs(value:WorkspaceMove):
    with lock:
        keys=set()
        # Validate the entire selection before writing any membership changes.
        for song in value.songs:
            p=folder(song.job)
            if active==song.job: raise HTTPException(409,'Wait for the selected generation to finish before moving it.')
            if readjob(p)['request'].get('background_analysis'): raise HTTPException(400,'Recording analyses are not library songs.')
            try: song_library.locate(p,song.candidate)
            except song_library.SongNotFound as e: raise HTTPException(404,str(e))
            keys.add(f'{song.job}:{song.candidate}')
        try: workspaces.move(value.workspace_id,keys)
        except WorkspaceError as e: raise HTTPException(400,str(e))
        return {'moved':len(keys)}
@app.get('/api/jobs/{j}')
def getjob(j:str):
    with lock:
        p=folder(j);data=readjob(p)
        if (p/'progress.json').exists(): data['progress']=json.loads((p/'progress.json').read_text())
        if (p/'run.log').exists():
            with (p/'run.log').open('rb') as f:
                f.seek(0,2);n=f.tell();f.seek(max(0,n-12000));data['log']=f.read().decode(errors='replace')
        return data

def write_artwork(p,data):
    tmp=p/'artwork.tmp';tmp.write_text(json.dumps(data));tmp.replace(p/'artwork.json')

def run_artwork(p,data):
    try:
        data.variation=data.variation or p.name
        if remote.configured():
            # The node holds the OpenAI key and talks to OpenAI; the small image comes back to the library.
            reply=via_node('POST','/v1/artwork',json=data.model_dump(),timeout=180)
            image=base64.b64decode(reply.pop('image'));fmt=reply.pop('format','webp')
            tmp=p/('artwork.'+fmt+'.tmp');tmp.write_bytes(image);tmp.replace(p/('artwork.'+fmt));meta=reply
        else: meta=generate_artwork(data,p/'artwork.webp', ROOT);fmt='webp'
        if fmt!='png': (p/'artwork.png').unlink(missing_ok=True)
        write_artwork(p,meta)
    except HTTPException as error:
        write_artwork(p,{'status':'failed','error':str(error.detail)})
    except Exception as error:
        write_artwork(p,{'status':'failed','error':'Could not generate album art: '+(str(error) or type(error).__name__)[:300]})

def start_artwork_job(p,data):
    with lock:
        kind=json.loads((p/'request.json').read_text()).get('kind')
        if kind not in {'generate','cover'}: return None
        current=None
        if (p/'artwork.json').is_file():
            try: current=json.loads((p/'artwork.json').read_text())
            except (json.JSONDecodeError,OSError): current=None
        current=current or {}
        if current.get('status')=='complete' and artwork_path(p): return current
        if current.get('status')=='running' and time.time()-float(current.get('started') or 0)<180: return current
        write_artwork(p,{'status':'running','started':time.time(),'model':'gpt-image-2.5-flare','quality':'low'})
    threading.Thread(target=run_artwork,args=(p,data),daemon=True).start()
    return {'status':'running'}

def inherit_artwork(p,source_name):
    """Re-renders, edits and extensions keep the cover of the song they came from instead of asking OpenAI for a new one."""
    if not source_name: return False
    try: source=folder(source_name)
    except HTTPException: return False
    picture=artwork_path(source)
    if picture is None: return False
    try:
        tmp=p/(picture.name+'.tmp');tmp.write_bytes(picture.read_bytes());tmp.replace(p/picture.name)
    except OSError: return False
    write_artwork(p,{'status':'complete','source':'inherited','from':source.name})
    return True

def queue_artwork(p,r,inherit_from=None):
    if r.kind not in {'generate','cover'}: return
    if inherit_artwork(p,inherit_from): return
    data=ArtworkAsk(api_key=r.api_key,title=r.title,style=r.style,lyrics=(r.lyrics or '')[:4000],instrumental=r.instrumental,variation=p.name)
    if not data.api_key.strip() and not ai_configured(): return
    start_artwork_job(p,data)

def ai_configured():
    """Whether an OpenAI key is available where the AI helpers run: the node when connected, else this computer."""
    if remote.configured(): return bool((remote.snapshot() or {}).get('openai_configured'))
    return openai_configured(ROOT)

def via_node(method,path,timeout=120,**kw):
    """Run a request on the connected node and hand its answer (or its error) back to the browser."""
    try: return remote.node.forward(method,path,timeout=timeout,**kw)
    except RenderNodeError as e: raise HTTPException(e.status,str(e)) from e

# OpenAI helpers run where the key is kept: on the render node when one is connected, else here.
@app.post('/api/style/key')
def style_key(data:style_ai.StyleKey):
    return via_node('POST','/v1/style/key',json=data.model_dump()) if remote.configured() else style_ai.style_key(data)

@app.post('/api/style/models')
def style_models(data:style_ai.StyleKey):
    return via_node('POST','/v1/style/models',json=data.model_dump()) if remote.configured() else style_ai.style_models(data)

@app.post('/api/style/generate')
def style_generate(data:style_ai.StyleAsk):
    return via_node('POST','/v1/style/generate',json=data.model_dump()) if remote.configured() else style_ai.style_generate(data)

@app.get('/api/jobs/{j}/artwork')
def artwork_file(j:str):
    p=folder(j);target=artwork_path(p)
    if not target or not target.resolve().is_relative_to(p.resolve()): raise HTTPException(404,'Album art not found')
    return FileResponse(target,media_type=ARTWORK_TYPES[target.suffix],filename=target.name,content_disposition_type='inline')

@app.post('/api/jobs/{j}/artwork')
def start_artwork(j:str,data:ArtworkAsk):
    p=folder(j)
    if readjob(p).get('kind') not in {'generate','cover'}: raise HTTPException(400,'Album art is only created for songs.')
    return start_artwork_job(p,data) or {'status':'running'}
@app.get('/api/files/{j}/{name:path}')
def file(j:str,name:str,request:Request):
    p=folder(j);target=(p/name).resolve()
    if not target.is_relative_to(p.resolve()) or target.suffix not in {'.wav','.mp3','.abc','.npy','.json','.mid'}: raise HTTPException(404)
    if target.is_file():
        return FileResponse(target,filename=target.name,content_disposition_type='inline' if target.suffix in {'.wav','.mp3'} else 'attachment')
    saved=remote_pending(p,name)
    if not saved: raise HTTPException(404)
    return stream_from_node(node_for(saved),'/v1/jobs/'+saved['id']+'/files/'+name,request.headers.get('range'))

def stream_from_node(node,path,range_header):
    """Pass a file on the render node through to the browser without keeping a copy; Range makes seeking work."""
    try: upstream=node.stream_path(path,range_header)
    except RenderNodeError as e: raise HTTPException(502,'The render node holding this audio is not reachable: '+str(e)) from e
    headers={k:v for k,v in upstream.headers.items() if k.lower() in {'content-length','content-range','accept-ranges','content-type','last-modified','etag'}}
    headers['Content-Disposition']='inline; filename="'+Path(path).name+'"'
    return StreamingResponse(render_client.chunks(upstream),status_code=upstream.status_code,headers=headers)

@app.post('/api/jobs/{j}/fetch')
def fetch_job_audio(j:str):
    """Copy a song's audio from the render node into this library (the Download button; needed before editing)."""
    p=folder(j);fetch_audio(p)
    return readjob(p)

@app.get('/api/uploads/{name}')
def source_file(name:str,request:Request):
    if not recordings.NAME.fullmatch(name): raise HTTPException(404)
    if (UP/name).is_file(): return FileResponse(UP/name)
    if remote.configured(): return stream_from_node(remote.node,'/v1/uploads/'+name,request.headers.get('range'))
    raise HTTPException(404)

@app.get('/api/uploads/{name}/cover')
def upload_cover(name:str,request:Request):
    """Picture stored inside the recording (ID3, MP4, FLAC), for the preparing-audio dialog."""
    if not recordings.NAME.fullmatch(name): raise HTTPException(404)
    if (UP/name).is_file():
        dest=recordings.cover_png(UP/name)
        if not dest: raise HTTPException(404,'No cover art in this recording')
        return FileResponse(dest,media_type='image/png',filename='artwork.png')
    if remote.configured():
        if not recordings.EDITABLE.fullmatch(name): raise HTTPException(404,'Recording not found')
        try: return stream_from_node(remote.node,'/v1/uploads/'+name+'/cover',request.headers.get('range'))
        except HTTPException as e:
            if e.status_code==502:
                raise HTTPException(404,'No cover art in this recording') from e
            raise
    raise HTTPException(404)

def song_download_source(j,index,fetch=True):
    """The WAV of one version. With ``fetch`` it is first copied from the render node if it only exists there."""
    p=folder(j)
    try:_,candidate=song_library.locate(p,index)
    except song_library.SongNotFound as e:raise HTTPException(404,str(e)) from e
    prefix=candidate.get('prefix','') if candidate else ''
    if prefix not in {'',f'candidate-{index:02d}/'}:raise HTTPException(404,'Song audio not found')
    wav=(p/prefix/'audio.wav').resolve()
    if not wav.is_relative_to(p.resolve()):raise HTTPException(404,'Song audio not found')
    if not wav.is_file() and fetch: fetch_audio(p)
    if not wav.is_file():raise HTTPException(404,'This song has no finished audio yet.')
    title=(candidate or {}).get('title') or json.loads((p/'request.json').read_text()).get('title','Untitled song')
    filename=re.sub(r'[\x00-\x1f\x7f\\/:*?"<>|]','_',title).strip().strip('.') or 'Untitled song'
    return wav,filename

@app.post('/api/jobs/{j}/songs/{index}/export')
def prepare_song_download(j:str,index:int,format:Literal['wav','mp3']='mp3'):
    wav,_=song_download_source(j,index)
    if format=='mp3':
        try:audio_export.to_mp3(wav)
        except Exception as e:raise HTTPException(500,'Could not convert this song to MP3. '+(str(e) if isinstance(e,ValueError) else 'Please try again.')) from e
    return {'url':f'/api/jobs/{j}/songs/{index}/download?format={format}'}

@app.get('/api/jobs/{j}/songs/{index}/download')
def download_song(j:str,index:int,format:Literal['wav','mp3']='wav'):
    wav,filename=song_download_source(j,index)
    if format=='mp3' and not audio_export.mp3_ready(wav):raise HTTPException(404,'Prepare the MP3 download first.')
    return FileResponse(wav if format=='wav' else wav.with_suffix('.mp3'),filename=filename+'.'+format,
                        media_type='audio/wav' if format=='wav' else 'audio/mpeg',content_disposition_type='attachment')

@app.get('/api/jobs/{j}/songs/{index}/timeline')
def song_timeline(j:str,index:int):
    """Bars, sections and waveform of a saved version for the Extend picker."""
    p=folder(j)
    try:
        song=song_continuation.song_files(p,index)
    except (song_library.SongNotFound,ValueError) as e:raise HTTPException(400,str(e)) from e
    result=song_continuation.timeline(song['abc'] if song['settings'].get('cot','full')!='off' else '',song['seconds'])
    # The waveform is decoration here; a WAV still on the render node is not worth pulling for it.
    try:result['peaks']=audio_edit.waveform(song_download_source(j,index,fetch=False)[0])['peaks']
    except Exception:result['peaks']=[]
    result.update(title=song['title'],cot=song['settings'].get('cot','full'),style=song['settings'].get('style',''),
                  lyrics=song['settings'].get('lyrics',''),instrumental=bool(song['settings'].get('instrumental')),
                  voice=song['settings'].get('voice','any'),model=song['settings'].get('model'))
    return result

@app.post('/api/upload')
async def upload(file:UploadFile=File(...)):
    """A recording for covers, continuation or the library. With a render node it goes straight there: the node
    decodes, draws and cuts it, and later renders from it, so this computer needs neither ffmpeg nor a copy."""
    if remote.configured():
        try: return await asyncio.to_thread(via_node,'POST','/v1/uploads',timeout=(15,900),files={'file':(file.filename or 'recording',file.file,file.content_type or 'application/octet-stream')})
        finally: await file.close()
    return await recordings.save(UP,file)


class UploadSong(BaseModel):
    model_config=ConfigDict(extra='forbid')
    title:str=Field(default='',max_length=200)
    workspace_id:str=Field(default='default',max_length=24)

@app.post('/api/uploads/{name}/song')
def upload_song(name:str,r:UploadSong):
    """List an uploaded recording in the library (badged Uploaded), keeping the original for covers and continuation."""
    if HOSTED: raise HTTPException(503,HOSTED_UNAVAILABLE)
    with lock:
        try: workspaces.require(workspaces.read(),r.workspace_id)
        except WorkspaceError as e: raise HTTPException(400,str(e))
        if remote.configured():
            # The node decodes the recording into a finished job of its own; the entry here streams from it.
            if not recordings.EDITABLE.fullmatch(name): raise HTTPException(404,'Recording not found')
            view=via_node('POST','/v1/uploads/'+name+'/song',timeout=900)
            job=uploaded_songs.create_remote(LIB,name,r.title,view['seconds'],r.workspace_id,lyrics=view.get('lyrics') or '')
            folder=LIB/job
            write_remote(folder,{'url':remote.node.url,'id':view['id'],'pending':[f for f in view.get('files',[]) if render_client.is_heavy(f)],'kept_until':view.get('kept_until')})
            for filename in view.get('files') or []:
                base=Path(filename).name
                if base in {'artwork.png','artwork.webp','artwork.json'} and not render_client.is_heavy(filename):
                    try: remote.node.fetch(view['id'],filename,folder/base)
                    except RenderNodeError: pass
            if not uploaded_songs.has_artwork(folder):
                uploaded_songs.write_artwork_meta(folder, False)
            return readjob(folder)
        path=recordings.editable(UP,name)
        try: job=uploaded_songs.create(LIB,path,name,r.title,r.workspace_id)
        except Exception as e: raise HTTPException(400,'Could not add the recording to the library. '+(str(e) if isinstance(e,ValueError) else 'Try uploading it again.')) from e
        return readjob(LIB/job)

def recording_waveform_data(name):
    """Peaks and length of an uploaded recording, from the node that holds it or from this computer."""
    if remote.configured():
        if not recordings.EDITABLE.fullmatch(name): raise HTTPException(404,'Recording not found')
        return via_node('GET','/v1/uploads/'+name+'/waveform')
    return recordings.waveform(recordings.editable(UP,name))

@app.get('/api/uploads/{name}/timeline')
def recording_timeline(name:str,source_seconds:float=0):
    """Waveform of an uploaded recording for the Extend picker; recordings have no plan, so no bars or sections.
    A positive ``source_seconds`` (the analysis length) bounds the part whose tokens exist."""
    wave=recording_waveform_data(name)
    total=wave['seconds'];seconds=min(total,source_seconds) if source_seconds>0 else total
    peaks=wave['peaks'][:max(1,round(len(wave['peaks'])*seconds/total))] if seconds<total else wave['peaks']
    return {'seconds':round(seconds,3),'peaks':peaks,'bars':[],'sections':[],'bpm':None,'cot':'off','recording':name}

@app.get('/api/uploads/{name}/waveform')
def recording_waveform(name:str):
    return recording_waveform_data(name)

class TrimRequest(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)
    start:float=Field(ge=0,le=1800)
    end:float=Field(gt=0,le=1800)

@app.post('/api/uploads/{name}/trim')
def trim_recording(name:str,r:TrimRequest):
    if remote.configured():
        if not recordings.EDITABLE.fullmatch(name): raise HTTPException(404,'Recording not found')
        return via_node('POST','/v1/uploads/'+name+'/trim',json=r.model_dump(),timeout=180)
    return recordings.trim(recordings.editable(UP,name),r.start,r.end)

def job_status(p):
    return json.loads((p/'state.json').read_text())['status']

def run_remote(p,r,remote_id=None):
    """Render on the connected node: send the job folder, mirror its progress here, collect the files.

    The node keeps rendering if this Studio loses the connection; ``remote.json`` lets a restarted Studio reattach.
    """
    global active
    node=remote.node
    try:
        if remote_id is None:
            (p/'run.log').write_text('[cache] Using saved recording analysis\n' if r.kind=='cover' else '')
            with (p/'run.log').open('a') as log: log.write('[remote] Sending the job to '+node.url+'\n')
            remote_id=node.submit(p,UP)
            (p/'remote.json').write_text(json.dumps({'url':node.url,'id':remote_id}))
            if (p/'.cancel-remote').exists(): node.cancel(remote_id)
        else:
            with (p/'run.log').open('a') as log: log.write('[remote] Reconnected to '+node.url+'\n')
        def on_running(started):
            with lock:
                if job_status(p) not in {'cancelled','cancelling'}: state(p,'running',started=started or time.time())
        final=node.mirror(p,remote_id,on_running=on_running)
        with lock:
            status=final['status']
            if job_status(p) in {'cancelled','cancelling'} or status=='cancelled': state(p,'cancelled')
            elif status=='complete':
                if r.kind in ANALYSIS_KINDS: save_analysis(r,p)
                # The audio stays on the node: the Studio streams the MP3 from there and fetches the WAV on request.
                write_remote(p,{'url':node.url,'id':remote_id,'pending':final.get('pending',[]),'kept_until':final.get('kept_until')})
                state(p,'complete')
            else: state(p,'failed',error=final.get('error') or 'The render node stopped this job.')
        if status!='complete' or not final.get('pending'): node.delete(remote_id)
    except Exception as e:
        with lock:
            state(p,'cancelled' if job_status(p) in {'cancelled','cancelling'} else 'failed',error=str(e))
            with (p/'run.log').open('a') as log: log.write('\n[error] '+str(e)+'\n')
    finally:
        (p/'.cancel-remote').unlink(missing_ok=True)
        with lock: active=None
        start_next()

def run(p,r):
    global active,process
    if remote.configured(): return run_remote(p,r)
    try:
        (p/'run.log').write_text('[cache] Using saved recording analysis\n' if r.kind=='cover' else '')
        started=time.time()
        stages=[r.kind in ANALYSIS_KINDS]
        code=0
        for analyze in stages:
            with lock:
                if json.loads((p/'state.json').read_text())['status'] in {'cancelled','cancelling'}:
                    state(p,'cancelled');return
                proc,marker=pool.submit(p,analyze)
                process=proc
                state(p,'running',started=started)
            code=pool.wait(proc,marker)
            with lock:
                process=None
                # A tokenize job warms the analysis runtime but not SheetSage2, so it does not count as a preload.
                if code==0 and r.kind!='tokenize': preloads.mark_ready('analysis' if analyze else r.model,proc,p)
                if code and code!=2: pool.discard(proc)
            if code: break
            if analyze: save_analysis(r,p)
        with lock:
            cancelled=json.loads((p/'state.json').read_text())['status'] in {'cancelled','cancelling'}
            if json.loads((p/'state.json').read_text())['status']=='interrupted': pass
            elif cancelled: state(p,'cancelled')
            elif code:
                error_file=p/'runtime-error.json'
                if error_file.exists():error=json.loads(error_file.read_text()).get('error','Analysis failed.')
                elif code<0:error='The model process stopped with signal '+str(-code)+'.'
                else:error='The model process exited with code '+str(code)+'. See run.log in the song folder for details.'
                state(p,'failed',error=error)
                with (p/'run.log').open('a') as log:log.write('\n[error] '+error+'\n')
            else: state(p,'complete')
    except Exception as e:
        with lock:
            if process: pool.discard(process)
            cancelled=json.loads((p/'state.json').read_text())['status'] in {'cancelled','cancelling'}
            state(p,'cancelled' if cancelled else 'failed',error=str(e))
    finally:
        with lock: active=None;process=None
        start_next()

def launch(p,r):
    global active
    state(p,'starting');active=p.name
    threading.Thread(target=run,args=(p,r),daemon=True).start()

def start_next():
    # Runs the next queued job once the worker is free; skipped jobs were cancelled or deleted while waiting.
    with lock:
        if active: return
        while queue:
            j=queue.pop(0);r=pending.pop(j,None);p=LIB/j
            if r is None or not (p/'state.json').is_file() or json.loads((p/'state.json').read_text())['status']!='queued': continue
            if r.kind not in ANALYSIS_KINDS and not model_ready(r.model):
                state(p,'failed',error='The selected model is missing. Open Models to download it.');continue
            launch(p,r);return

@app.post('/api/jobs')
def create(r:Job):
    if HOSTED: raise HTTPException(503,HOSTED_UNAVAILABLE)
    with lock:
        try: workspaces.require(workspaces.read(),r.workspace_id)
        except WorkspaceError as e: raise HTTPException(400,str(e))
        render_data=None
        if r.render_source:
            original=readjob(folder(r.render_source))
            candidate=next((c for c in original.get('result',{}).get('candidates',[]) if c['index']==r.render_candidate),None)
            if candidate is None: raise HTTPException(400,'Select a finished version')
            source_dir=folder(r.render_source)/candidate['prefix']
            if not (source_dir/'tokens.json').is_file(): raise HTTPException(400,'This version has no saved music tokens')
            render_data={'codec':json.loads((source_dir/'tokens.json').read_text()),'abc':candidate.get('abc') or original.get('result',{}).get('abc') or '',
                         # Songs that extend a recording keep its approximate tokens, which render with the real-audio adapter.
                         'real_audio':bool(original['request'].get('continue_recording'))}
            if (source_dir/'render-prefix.json').is_file(): render_data['prefix']=json.loads((source_dir/'render-prefix.json').read_text())
            # The Sound LoRA only touches the decoder, so a re-render may swap it: same tokens, same seed, different sound.
            sound=(r.sound_lora or '');plain=r.render_without_lora and bool(original['request'].get('lora'))
            suffix=' · '+str(r.steps)+' steps'+(' · '+Path(sound).stem if sound else '')+(' · without '+Path(original['request']['lora']).stem if plain else '')
            if sound and not plain and sound==original['request'].get('lora'): raise HTTPException(400,'This song already uses '+repr(sound)+' as its LoRA; pick a different Sound LoRA')
            r=Job(**{**original['request'],'kind':'generate','edit_id':'','continue_source':'','continue_seconds':None,'continue_recording':False,'origin':None,'render_source':r.render_source,'render_candidate':r.render_candidate,'render_without_lora':plain,'steps':r.steps,'seed':candidate['seed'],'random_seed':False,'candidates':1,'abc':render_data['abc'],'title':original['title']+suffix,
                     'sound_lora':sound,'sound_lora_strength':r.sound_lora_strength,**({'lora':'','lora_strength':1.0} if plain else {})})
        r.origin=None
        continuation=None
        if r.continue_source:
            source=folder(r.continue_source)
            try: continuation=song_continuation.continuation_input(source,r.continue_candidate,r.continue_seconds)
            except (ValueError,OSError) as e: raise HTTPException(400,str(e))
            # The plan of the source decides how the continuation is planned; a supplied score cannot apply.
            r.cot=continuation['cot'];r.abc=''
            r.origin={'kind':'continue',**continuation['source'],'cut_seconds':continuation['cut_seconds']}
        if r.continue_recording:
            if not tokens_ready(): raise HTTPException(503,'Audio input is not installed. Download Cover analysis and Audio input in Models first.')
            tokens=recording_tokens.read(token_job(TokenRequest(audio_id=r.audio_id,source_seconds=r.source_seconds)))
            if tokens is None: raise HTTPException(409,'Prepare the recording first.')
            # Uploads keep no name on the server; the song title (minus the client's suffix) names the source.
            try: continuation=song_continuation.recording_continuation(tokens,r.continue_seconds,title=re.sub(r'\s*·\s*extended$','',r.title))
            except ValueError as e: raise HTTPException(400,str(e))
            r.origin={'kind':'continue',**continuation['source'],'cut_seconds':continuation['cut_seconds']}
        artwork_from=r.render_source or r.continue_source or None
        if r.edit_id:
            try: edited=composition_edit.load_draft(ROOT,r.edit_id)
            except ValueError as e:raise HTTPException(400,str(e))
            if r.kind!='generate':raise HTTPException(400,'Saved compositions use song generation.')
            r.abc=edited['abc'];r.cot=edited['cot'];artwork_from=artwork_from or (edited.get('origin') or {}).get('job')
        if r.kind not in ANALYSIS_KINDS and not model_ready(r.model): raise HTTPException(503,'The selected model is missing. Open Models to download it.')
        if r.kind=='transcribe' and not config()['transcriber_ready']:
            installer=(r'scripts\Install OpenSuno.ps1' if os.name=='nt' else 'scripts/Install OpenSuno.command')
            raise HTTPException(503,'Cover analysis is not installed. Download Cover analysis in Models and run '+installer+' to set up its environment.')
        if r.kind=='tokenize' and not tokens_ready():
            raise HTTPException(503,'Audio input is not installed. Download Cover analysis and Audio input in Models first.')
        cached=analyses.read(r) if r.kind=='cover' else None
        if r.kind=='cover' and cached is None: raise HTTPException(409,'The recording must finish analysis before creating a cover.')
        if r.kind in {'generate','cover'} and r.random_seed:
            previous=r.seed
            while r.seed==previous: r.seed=secrets.randbits(32)
        # An instrumental cover follows the recording's own sections; nothing has to be written for it.
        if r.kind=='cover' and r.instrumental: r.lyrics=instrumental.cover_structure(cached.get('abc',''),r.lyrics)
        j=time.strftime('%Y%m%d-%H%M%S')+'-'+secrets.token_hex(4);p=LIB/j;p.mkdir()
        (p/'request.json').write_text(r.model_dump_json(indent=2))
        queue_artwork(p,r,inherit_from=artwork_from)
        if render_data is not None: (p/'render-input.json').write_text(json.dumps(render_data))
        if continuation is not None: (p/'continue-input.json').write_text(json.dumps(continuation))
        if cached is not None:
            (p/'analysis.json').write_text(json.dumps({**cached,'elapsed':0,'cached':True}))
        if active:
            state(p,'queued');queue.append(j);pending[j]=r
            return {'id':j,'seed':r.seed,'status':'queued','position':len(queue)}
        launch(p,r)
        return {'id':j,'seed':r.seed,'status':'starting','position':0}
class AnalysisRequest(BaseModel):
    audio_id:str
    melody_only:bool=True
    source_seconds:float=Field(default=0,ge=0,le=1800)

def analysis_job(r):
    try:return Job(kind='transcribe',background_analysis=True,title='Recording analysis',**r.model_dump())
    except ValueError as e:raise HTTPException(400,'Upload a valid recording and choose valid analysis settings.') from e

def analysis_status(r):
    cached=analyses.read(r)
    if cached is not None:
        return {'status':'ready','sections':instrumental.score_sections(cached.get('abc',''))}
    j=analyses.job(r)
    if j:
        try:
            data=readjob(folder(j))
            return {'status':data['status'] if data['status']!='complete' else 'missing','job':j,'error':data.get('error')}
        except HTTPException:pass
    return {'status':'missing'}

@app.get('/api/analysis')
def get_analysis(audio_id:str,melody_only:bool=True,source_seconds:float=0):
    with lock:
        try:r=AnalysisRequest(audio_id=audio_id,melody_only=melody_only,source_seconds=source_seconds)
        except ValueError as e:raise HTTPException(400,'Invalid analysis settings.') from e
        return analysis_status(analysis_job(r))

@app.post('/api/analysis')
def start_analysis(r:AnalysisRequest):
    with lock:
        job=analysis_job(r);status=analysis_status(job)
        if status['status'] in {'ready','starting','running','cancelling','queued'}:return status
        created=create(job);analyses.remember(job,created['id'])
        if created['status']=='queued':
            # The user is waiting in the Cover dialog, so analysis goes ahead of queued songs.
            queue.remove(created['id']);queue.insert(0,created['id'])
        return {'status':created['status'],'job':created['id']}

class TokenRequest(BaseModel):
    audio_id:str
    source_seconds:float=Field(default=0,ge=0,le=1800)

def token_job(r):
    try:return Job(kind='tokenize',background_analysis=True,title='Recording tokens',**r.model_dump())
    except ValueError as e:raise HTTPException(400,'Upload a valid recording first.') from e

def token_status(r):
    """Like analysis_status, for the music tokens of a recording."""
    cached=recording_tokens.read(r)
    if cached is not None:return {'status':'ready','seconds':cached.get('seconds'),'frames':len(cached['codec'])}
    j=recording_tokens.job(r)
    if j:
        try:
            data=readjob(folder(j))
            return {'status':data['status'] if data['status']!='complete' else 'missing','job':j,'error':data.get('error')}
        except HTTPException:pass
    return {'status':'missing'}

@app.get('/api/tokens')
def get_tokens(audio_id:str,source_seconds:float=0):
    with lock:
        try:r=TokenRequest(audio_id=audio_id,source_seconds=source_seconds)
        except ValueError as e:raise HTTPException(400,'Invalid recording.') from e
        return token_status(token_job(r))

@app.post('/api/tokens')
def start_tokens(r:TokenRequest):
    """Turn an uploaded recording into music tokens in the background; the result is cached per recording."""
    with lock:
        job=token_job(r);status=token_status(job)
        if status['status'] in {'ready','starting','running','cancelling','queued'}:return status
        created=create(job);recording_tokens.remember(job,created['id'])
        if created['status']=='queued':
            queue.remove(created['id']);queue.insert(0,created['id'])
        return {'status':created['status'],'job':created['id']}

class SectionEdit(BaseModel):
    model_config=ConfigDict(extra='forbid')
    id:str=Field(max_length=8)
    fraction:float=Field(default=1,gt=0,le=1)
    solo:bool=False

class CompositionEdit(BaseModel):
    model_config=ConfigDict(extra='forbid')
    style:str=Field(max_length=6000)
    lyrics:str=Field(default='',max_length=30000)
    bpm:int=Field(ge=30,le=300)
    harmony:Literal['preserve','free','sevenths']='preserve'
    sections:list[SectionEdit]|None=Field(default=None,max_length=80)

@app.get('/api/jobs/{j}/songs/{candidate}/composition')
def song_composition(j:str,candidate:int):
    try:return composition_edit.source_song(folder(j),candidate)
    except (ValueError,OSError) as e:raise HTTPException(400,str(e))

class SpeedChange(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)
    factor:float=Field(ge=audio_speed.MIN_FACTOR,le=audio_speed.MAX_FACTOR)
    keep_pitch:bool=True
    workspace_id:str=Field(default='',max_length=24)

@app.post('/api/jobs/{j}/songs/{candidate}/speed')
def change_song_speed(j:str,candidate:int,r:SpeedChange):
    """Save a faster or slower copy of one version as a new song (Suno's Adjust speed). Audio only; nothing is regenerated."""
    if HOSTED: raise HTTPException(503,HOSTED_UNAVAILABLE)
    wav,_=song_download_source(j,candidate)
    p=folder(j)
    with lock:
        workspace_id=r.workspace_id or json.loads((p/'request.json').read_text()).get('workspace_id','default')
        try: workspaces.require(workspaces.read(),workspace_id)
        except WorkspaceError as e: raise HTTPException(400,str(e))
        try: job=audio_speed.create(LIB,UP,p,candidate,wav,r.factor,r.keep_pitch,workspace_id)
        except (ValueError,song_library.SongNotFound) as e: raise HTTPException(400,str(e)) from e
        except (subprocess.SubprocessError,OSError) as e: raise HTTPException(500,'Could not change the speed of this song. '+str(e)) from e
        return readjob(LIB/job)

@app.post('/api/jobs/{j}/songs/{candidate}/edit')
def prepare_song_edit(j:str,candidate:int,r:CompositionEdit):
    try:
        source=composition_edit.source_song(folder(j),candidate)
        edited=composition_edit.apply(source,r.model_dump())
        return composition_edit.save_draft(ROOT,edited)
    except (ValueError,KeyError,TypeError,OSError) as e:raise HTTPException(400,str(e))

def dequeue(j):
    if j not in queue: return False
    queue.remove(j);pending.pop(j,None);state(LIB/j,'cancelled')
    return True

@app.post('/api/jobs/{j}/cancel')
def cancel(j:str):
    with lock:
        p=folder(j)
        if dequeue(j): return {'ok':True}
        if active!=j: raise HTTPException(409,'This job is no longer running')
        if remote.configured() and not process:
            # The node is asked to stop; the mirror thread records the final state when it confirms.
            state(p,'cancelling');(p/'.cancel-remote').touch()
            if (p/'remote.json').is_file():
                remote_id=json.loads((p/'remote.json').read_text())['id'];node=remote.node
                def ask_node():
                    try: node.cancel(remote_id)
                    except RenderNodeError: pass
                threading.Thread(target=ask_node,daemon=True).start()
            return {'ok':True}
        state(p,'cancelling' if process else 'cancelled')
        if process:
            (p/'.cancel-resident').touch()
            proc=process
            def stop_unresponsive():
                time.sleep(5)
                with lock:
                    if active==j and process is proc:
                        pool.discard(proc)
            threading.Thread(target=stop_unresponsive,daemon=True).start()
        return {'ok':True}
class Favorite(BaseModel):
    candidate:int=Field(ge=1,le=8)

@app.post('/api/jobs/{j}/favorite')
def favorite(j:str,choice:Favorite):
    with lock:
        p=folder(j);saved=readjob(p)
        if choice.candidate not in [c['index'] for c in saved.get('result',{}).get('candidates',[])]: raise HTTPException(400,'That candidate has not finished')
        value=None if saved.get('favorite')==choice.candidate else choice.candidate
        (p/'favorite.json').write_text(json.dumps({'candidate':value}))
        return {'ok':True,'favorite':value}

class SongName(BaseModel):
    title:str=Field(min_length=1,max_length=100)

@app.patch('/api/jobs/{j}/songs/{candidate}')
def rename_song(j:str,candidate:int,name:SongName):
    with lock:
        p=folder(j)
        if active==j: raise HTTPException(409,'Wait for this generation to finish before renaming it.')
        title=name.title.strip()
        if not title: raise HTTPException(400,'Enter a song title.')
        try: song_library.rename(p,candidate,title)
        except song_library.SongNotFound as e: raise HTTPException(404,str(e))
        return {'ok':True}

@app.delete('/api/jobs/{j}/songs/{candidate}')
def delete_song(j:str,candidate:int):
    with lock:
        p=folder(j)
        if active==j: raise HTTPException(409,'Stop this generation before deleting it.')
        dequeue(j)
        try: song_library.delete(p,candidate,ROOT/'.trash')
        except song_library.SongNotFound as e: raise HTTPException(404,str(e))
        workspaces.forget(j,candidate,entire_job=not p.exists())
        return {'ok':True}

@app.get('/api/session')
async def page_session(req:Request,token:str=''):
    if not HOSTED and not secrets.compare_digest(token,TOKEN): raise HTTPException(403,'Reload Studio')
    async def connected():
        key=leases.open()
        try:
            while True:
                yield 'data: connected\n\n'
                await asyncio.sleep(5)
                if await req.is_disconnected(): break
        finally:
            leases.close(key)
    return StreamingResponse(connected(),media_type='text/event-stream',
                             headers={'Cache-Control':'no-store','X-Accel-Buffering':'no'})

@app.on_event('shutdown')
def shutdown():
    with lock:
        leases.shutdown()
        # A job on a render node keeps rendering; its state stays so the next Studio start reattaches to it.
        if active and not (LIB/active/'remote.json').is_file():
            state(LIB/active,'interrupted',error='Studio was closed while this job was running.')
        drop_queue('interrupted',error='Studio was closed while this job was waiting in the queue.')
        pool.close()

def reconcile_jobs():
    """Jobs left running by the previous Studio: reattach the one on the render node, interrupt the rest."""
    global active
    for p in sorted(LIB.iterdir(),reverse=True):
        if not (p/'state.json').exists(): continue
        try: status=job_status(p)
        except (ValueError,KeyError): continue
        if status not in {'running','starting','cancelling','queued'}: continue
        if status!='queued' and active is None and remote.configured() and (p/'remote.json').is_file():
            try:
                saved=json.loads((p/'remote.json').read_text());r=Job(**json.loads((p/'request.json').read_text()))
                if saved['url']==remote.node.url:
                    active=p.name;threading.Thread(target=run_remote,args=(p,r,saved['id']),daemon=True).start();continue
            except (ValueError,KeyError,OSError): pass
        state(p,'interrupted',error='Studio restarted before this job finished. You can reuse its settings.')
reconcile_jobs()

app.mount('/static',StaticFiles(directory=ROOT/'web'),name='static')

if __name__=='__main__':
    import uvicorn
    uvicorn.run(app,host='127.0.0.1',port=int(os.environ.get('YUE2_PORT','7862')),timeout_graceful_shutdown=2)
