"""One batch, optionally using models retained by resident_worker."""
import contextlib, json, sys, time, re
from pathlib import Path
from runtime_platform import replace_file
import instrumental
import cover_arrangement
import vocal_register
ROOT=Path(__file__).resolve().parent
MODEL_CACHE=globals().get('MODEL_CACHE',{})
LORA_CACHE=MODEL_CACHE.setdefault('_loras',{})   # parsed LoRA deltas, kept with the resident model
CHECK_CANCEL=globals().get('CHECK_CANCEL',lambda:None)
jobdir=Path(sys.argv[1]).resolve()
r=json.loads((jobdir/'request.json').read_text())
is_analysis=r['kind'] in {'transcribe','tokenize'} or '--analyze' in sys.argv
analysis_elapsed=0
if r['kind']=='cover' and not is_analysis:
    analysis=json.loads((jobdir/'analysis.json').read_text())
    if (analysis['audio_id'],analysis['melody_only'],analysis['source_seconds'])!=(r['audio_id'],r['melody_only'],r['source_seconds']):
        raise RuntimeError('Recording analysis does not match this cover request')
    r['abc']=analysis['abc'];analysis_elapsed=analysis['elapsed']
# Instrumental songs keep their section tags with no words: the plan then leaves the Vocal voice silent.
generation_lyrics=instrumental.structure_only(r['lyrics']) if r.get('instrumental') else r['lyrics']
# The model has no voice input; gender words in the tag block are the only lever, so the choice is appended as its own tag line.
VOICE_TAGS={'male':'Male lead vocal.','female':'Female lead vocal.','duet':'Male and female vocal duet, alternating verses, both voices harmonizing on the chorus.'}
generation_style=instrumental.instrumental_style(r['style']) if r.get('instrumental') else r['style']+('\n'+VOICE_TAGS[r['voice']] if r.get('voice') in VOICE_TAGS else '')
start=time.perf_counter()
progress={'stage':'loading','tokens':0,'tokens_per_second':0,'steps':0,'step_total':r.get('steps',32),'candidate':1,'candidates':r.get('candidates',1)}
def report(**changes):
    CHECK_CANCEL()
    progress.update(changes);progress['elapsed']=round(analysis_elapsed+time.perf_counter()-start,2)
    tmp=jobdir/'progress.tmp';tmp.write_text(json.dumps(progress));replace_file(tmp,jobdir/'progress.json')
def log(s):
    print(s,flush=True)
    if s.startswith('[load]'): report(stage='analysis_loading' if is_analysis else 'loading')
    elif s.startswith('[plan]'): report(stage='planning',tokens=0,tokens_per_second=0,token_limit=r['abc_sampling']['max_tokens'])
    elif s.startswith('[semantic] prefix'): report(stage='music_tokens',tokens=0,tokens_per_second=0,token_limit=r['semantic_sampling']['max_tokens'])
    elif s.startswith('[nar]') and 'frames' in s: report(stage='synthesis',steps=0,tokens=int(re.search(r'\[nar\] (\d+) frames',s)[1]))
    elif s.startswith('[vae]'): report(stage='decoding_audio')
    elif s.startswith('[done]'): report(stage='complete')
if r['kind']=='cover' and not is_analysis and not r.get('instrumental') and r.get('abc'):
    # Before the arrangement, so the chords the model writes harmonise the notes that will actually be sung.
    try:
        setting=r.get('vocal_octave','auto')
        r['abc'],moved,median=vocal_register.fit(r['abc'],r.get('voice','any'),setting)
        if median is not None:
            log(f'[register] Score centred on {vocal_register.note_name(median)}; '
                +(f'moved {"down" if moved<0 else "up"} {"an octave" if abs(moved)==1 else str(abs(moved))+" octaves"} to {vocal_register.note_name(median+12*moved)}'
                  +(f' for the {r["voice"]} voice' if setting=='auto' and r.get('voice') in vocal_register.REGISTERS else '') if moved else 'kept as written'))
    except ValueError as e:
        log(f'[register] The melody could not be moved ({e}); continuing as heard')
report()
loading_tick=None
loading_name=None
def loading(detail):
    global loading_tick,loading_name
    now=time.perf_counter()
    if loading_tick is not None:
        print(f'[startup] {loading_name}: {now-loading_tick:.2f}s',flush=True)
    loading_tick,loading_name=now,detail
    log('[load] '+detail)
    report(loading_detail=detail)
def loaded():
    global loading_tick
    if loading_tick is not None:
        print(f'[startup] {loading_name}: {time.perf_counter()-loading_tick:.2f}s',flush=True)
        loading_tick=None
    report(loading_detail=None)
def save_meta(meta):
    tmp=jobdir/'result.tmp';tmp.write_text(json.dumps(meta,indent=2));replace_file(tmp,jobdir/'result.json')
def instrumental_plan(abc):
    """Silence the Vocal voice of a planned, edited or transcribed score for instrumental songs."""
    if not (r.get('instrumental') and abc): return abc
    try:
        abc,moved,freed=instrumental.instrumental_score(abc,hook_only=bool(r.get('hook_melody')))
        log(f'[mode] Instrumental: {moved} sung bars moved to the instrument voice; the vocal voice is silent'
            +(f'; {freed} verse bars left to the chords for the model to arrange' if freed else ''))
    except ValueError as e:
        log(f'[mode] Instrumental: the score could not be rewritten ({e}); continuing with the planned score')
    return abc
def arrange_cover(abc,make_writer):
    """Covers: the model writes chords and instrumental lines around the transcribed melody, in the new style.

    The transcription is a bare lead line; the arranged score is chord-annotated, so the generation
    switches to the chord-annotated instruction. Failures keep the transcription as it was.
    """
    if not (r['kind']=='cover' and r.get('arrange',True) and abc): return abc
    log('[plan] Arranging the cover: the model adds harmony and instrumental lines around the transcribed melody')
    report(stage='arranging',tokens=0,token_limit=0)
    tick=time.perf_counter()
    try:
        result=cover_arrangement.arrange(abc,make_writer(),on_progress=lambda done,total:report(stage='arranging',tokens=done,token_limit=total))
    except Exception as e:
        # Never lose a cover to its arrangement: cancellation is a BaseException and still propagates.
        log(f'[plan] The arrangement could not be written ({type(e).__name__}: {e}); continuing with the transcribed score')
        return abc
    log(f'[plan] Arranged in {time.perf_counter()-tick:.0f}s: {result.chords} chords added to {result.bars} bars'
        +(f' ({result.corrected_chords} proposals were out of tune with the sung notes and replaced)' if result.corrected_chords else '')
        +(f', {result.kept_chords} transcribed chords kept' if result.kept_chords else '')
        +(f', {result.ins_lines} instrumental lines written' if result.ins_lines else '')
        +(f', {result.ins_rejected} lines refused for not playing their chords' if result.ins_rejected else '')
        +(f', {result.ins_failed} silent sections left as heard' if result.ins_failed else ''))
    r['cot']='full'
    return result.abc

if is_analysis:
    loading('Importing audio analysis runtime')
    import torch
    loading('Importing audio analysis libraries')
    from transformers import AutoModel
    torch.set_num_threads(6)
    device='mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
if is_analysis and r['kind']=='tokenize':
    # Music tokens of a recording (to continue it): stock MERT + the community tokenizer head, no SheetSage2.
    import audio_tokens
    tokenizer=MODEL_CACHE.get('tokens')
    if tokenizer is None:
        loading('Loading MERT and the music tokenizer on '+device)
        tokenizer=MODEL_CACHE['tokens']=audio_tokens.SemanticTokenizer(ROOT,device)
    else:
        log('[cache] Reusing the music tokenizer already in memory')
    loaded()
    log('[tokens] Listening to your recording')
    def token_progress(v):
        report(stage='analysis_'+v['stage'],window=v.get('window',1),windows=v.get('windows',1));log('[tokens] '+json.dumps(v))
    with torch.inference_mode():
        result=tokenizer.tokenize(str(ROOT/'uploads'/r['audio_id']),max_seconds=r['source_seconds'] or None,progress=token_progress)
    (jobdir/'tokens.json').write_text(json.dumps(result['codec']))
    meta={'codec':result['codec'],'frames':len(result['codec']),'seconds':result['seconds'],'tokenizer':result['tokenizer'],
          'audio_id':r['audio_id'],'source_seconds':r['source_seconds'],'elapsed':round(time.perf_counter()-start,2),'device':device}
    log(f"[tokens] {len(result['codec'])} music tokens for {result['seconds']:.1f} seconds of audio")
elif is_analysis:
    model=MODEL_CACHE.get('analysis')
    if model is None:
        loading('Loading SheetSage2 and MERT on '+device)
        model=AutoModel.from_pretrained(str(ROOT/'transcriber'),base_model_path=str(ROOT/'mert'),
            trust_remote_code=True,local_files_only=True).eval().to(device)
        # MERT encodes on the GPU; the tiny BART decoder is ~2x faster on CPU than on MPS.
        if device=='mps': model.offload_decoder('cpu')
        MODEL_CACHE['analysis']=model
    else:
        log('[cache] Reusing audio analysis models already in memory')
    loaded()
    if r.get('_preload'):
        log('[ready] Model loaded in memory')
        raise SystemExit(0)
    log('[transcribe] Reading your recording')
    transcribe_tick=[time.perf_counter()]
    def transcription_progress(v):
        stage=v.get('stage','audio')
        if stage=='encoding': transcribe_tick[0]=time.perf_counter()
        changes={'stage':'analysis_'+stage,'window':v.get('window',1),'windows':v.get('windows',1)}
        if 'tokens' in v:
            changes.update(tokens=v['tokens'],tokens_per_second=round(v['tokens']/max(.001,time.perf_counter()-transcribe_tick[0]),1))
            if device=='mps':
                if torch.mps.driver_allocated_memory()-torch.mps.current_allocated_memory()>4*1024**3:
                    torch.mps.synchronize()
                    torch.mps.empty_cache()
                print('[memory] '+json.dumps({'tokens':changes['tokens'],'gpu_allocated_gb':round(torch.mps.current_allocated_memory()/1e9,2),
                                              'gpu_driver_gb':round(torch.mps.driver_allocated_memory()/1e9,2)}),flush=True)
        report(**changes);log('[transcribe] '+json.dumps(v))
    with torch.inference_mode():
        result=model.transcribe(str(ROOT/'uploads'/r['audio_id']),output_dir=str(jobdir/'score'),
            melody_only=r['melody_only'],dtype='fp32',max_seconds=r['source_seconds'] or None,
            progress=transcription_progress)
    if not result.get('abc'): raise RuntimeError(result.get('abc_error') or 'No melody score was produced. Try a longer recording with a clear melody.')
    abc=result['abc']
    (jobdir/'score.abc').write_text(abc)
    meta={'abc':abc,'elapsed':round(time.perf_counter()-start,2),'device':device,
          'audio_id':r['audio_id'],'melody_only':r['melody_only'],'source_seconds':r['source_seconds']}
elif r.get('model','').startswith('cuda-'):
    loading('Importing official YuE2 CUDA runtime')
    from cuda_engine import run,ScoreWriter
    from yue2.protocol import Sampling as CudaSampling
    def prepare_score(pipe,abc):
        # Arrange first, then silence the voice: instrumental covers keep the model's chords under the moved melody.
        return instrumental_plan(arrange_cover(abc,lambda:ScoreWriter(pipe,generation_style,generation_lyrics,
            CudaSampling(**r['abc_sampling']),r['seed'],{'cancelled':pipe.callbacks(r)['cancelled']})))
    meta=run(ROOT,jobdir,r,MODEL_CACHE,report,CHECK_CANCEL,save_meta,
             generation_style,generation_lyrics,start,analysis_elapsed,abc_transform=instrumental_plan,prepare_score=prepare_score)
else:
    import mlx_continuation
    sys.path.insert(0,str(ROOT/'model'))
    from runtime_platform import prepare_gpu_libraries
    prepare_gpu_libraries()
    loading('Importing MLX GPU runtime')
    import mlx.core as mx
    loading('Importing NumPy')
    import numpy as np
    loading('Importing music generation libraries')
    import generate as engine
    from generate import Yue2Pipeline,Sampling,write_wav,generate_tokens,token_prefix
    variant=r.get('model','bf16')
    pipe=MODEL_CACHE.get(variant)
    if pipe is None:
        loading('Opening YuE2 '+variant+' on '+('Apple GPU' if sys.platform=='darwin' else 'NVIDIA GPU'))
        pipe=Yue2Pipeline(ROOT/'model'/variant,log=log,on_load=loading)
        MODEL_CACHE[variant]=pipe
    else:
        pipe.log=log
        log('[cache] Reusing YuE2 '+variant+' already in memory')
    loaded()
    if r.get('_preload'):
        log('[ready] Model loaded in memory')
        raise SystemExit(0)
    # The LoRA (writing) and the Sound LoRA (decoder) are switched on for this song only; the resident model is
    # cleared of the previous song's adapters first.
    import loras
    wanted=loras.chosen(r)
    def attach(drop_writing_ar=False):
        # The LoRA (writing) and the Sound LoRA (decoder) are switched on for this song only; the resident model is
        # cleared of the previous song's adapters first. Covers drop the Writes adapter from the AR path after
        # arrangement so the transcribed score is not rewritten into a loop.
        import adapters
        if not wanted:
            if getattr(pipe,'model',None) is not None and pipe.model.__dict__.get('_adapters'):
                if pipe.model.__dict__['_adapters'].get('applied') is not None: loading('Restoring the stock model weights')
                adapters.clear(pipe.model)
            return
        loading('Attaching the LoRA '+' and '.join(name for name,_ in wanted)
                +(' to the decoder only' if drop_writing_ar else ''))
        dims=None
        parts=[]
        writing=r.get('lora') or ''
        for name,strength in wanted:
            delta=LORA_CACHE.get(name)
            if delta is None:
                if len(LORA_CACHE)>=3: LORA_CACHE.clear()
                dims=dims or loras.dims_from_config(json.loads((ROOT/'model'/variant/'config.json').read_text()))
                delta=LORA_CACHE[name]=loras.check(loras.load(ROOT,name,dims),dims)
            if drop_writing_ar and name==writing:
                delta=loras.drop_ar(delta)
                if not delta['linears'] and not delta['io']:
                    continue
            parts.append((delta,strength))
        if not parts:
            adapters.clear(pipe.model)
            log('[lora] Cover keeps the transcribed score; the Writes adapter is not applied to music tokens')
            loaded();return
        combined=parts[0][0] if len(parts)==1 else loras.combine(parts)
        strength=parts[0][1] if len(parts)==1 else 1.0
        count=adapters.apply(pipe.model,combined,strength,weights=ROOT/'model'/variant/'model.safetensors',
                             label=tuple(wanted)+(('cover-decoder',) if drop_writing_ar else ()))
        loaded()
        for name,s in wanted: log(f"[lora] {name} at strength {s:g}")
        if drop_writing_ar:
            log('[lora] Cover: decoder half of the Writes adapter only, so the transcribed melody does not loop')
        log(f"[lora] {count} adapted layers")
    attach()
    pipe.abc_sampling=Sampling(**r['abc_sampling'])
    def token_progress(label):
        # Rate is measured from the first token so the prompt prefill does not
        # show up as a slow ramp in the UI; it is the decoder's steady speed.
        tick=None;count=0
        def token(phase,value):
            nonlocal count,tick
            CHECK_CANCEL()
            count+=1
            now=time.perf_counter()
            if tick is None: tick=now
            if count==1 or count%20==0:
                rate=(count-1)/max(.001,now-tick)
                report(stage='planning' if label=='abc' else 'music_tokens',tokens=count,
                    token_limit=r['abc_sampling' if label=='abc' else 'semantic_sampling']['max_tokens'],tokens_per_second=round(rate,1))
                print(f"[{label}] {count} tokens, {rate:.1f} tok/s",flush=True)
        return token
    pipe._progress=token_progress
    if not hasattr(engine,'_resident_synthesize'): engine._resident_synthesize=engine.synthesize
    original_synthesize=engine._resident_synthesize
    def tracked_synthesize(*args,**kwargs):
        previous=kwargs.get('on_progress')
        prefix,codec=args[1:3]
        chunks=len(engine.chunk_ranges(len(codec),len(prefix)))
        completed=0
        def step(i,n):
            nonlocal completed
            completed+=1
            report(stage='synthesis',steps=completed,step_total=n*chunks)
            if previous: previous(i,n)
        kwargs['on_progress']=step
        return original_synthesize(*args,**kwargs)
    engine.synthesize=tracked_synthesize
    def real_audio_decoder(wanted):
        """Synthesis context: songs that keep a recording's approximate tokens render with the community NAR adapter."""
        if not wanted: return contextlib.nullcontext()
        import real_audio
        if pipe.model.__dict__.get('_real_audio') is None:
            loading('Attaching the real-audio decoder');real_audio.attach(pipe.model,ROOT);loaded()
        log('[nar] rendering with the real-audio decoder (community NAR adapter)')
        return real_audio.enabled(pipe.model)
    # Arrange first, then silence the voice: instrumental covers keep the model's chords under the moved melody.
    r['abc']=instrumental_plan(arrange_cover(r.get('abc',''),lambda:engine.ScoreWriter(pipe.model,pipe.tokenizer,
        token_prefix(pipe.tokenizer,generation_style,generation_lyrics,'full'),pipe.abc_sampling,r['seed'])))
    if r['kind']=='cover' and wanted and r.get('lora'):
        attach(drop_writing_ar=True)
    def write_plan(seed):
        log('[plan] Writing the score')
        ids,truncated=generate_tokens(pipe.model,token_prefix(pipe.tokenizer,generation_style,generation_lyrics,r['cot']),
            pipe.abc_sampling,seed,'abc',on_token=pipe._progress('abc'))
        abc=pipe.tokenizer.decode(ids)
        if not abc.strip(): raise RuntimeError('The model did not produce a score. Try a different seed.')
        return instrumental_plan(abc),truncated
    if r['kind']=='plan':
        abc=r['abc']
        truncated=False
        if not abc: abc,truncated=write_plan(r['seed'])
        (jobdir/'score.abc').write_text(abc)
        meta={'abc':abc,'elapsed':round(time.perf_counter()-start,2),'model':variant,'truncated':truncated}
    else:
        count=r.get('candidates',1)
        meta={'candidates':[],'model':variant,'requested_candidates':count}
        for index in range(1,count+1):
            seed=(r['seed']+index-1)%4294967296
            destination=jobdir if count==1 else jobdir/f'candidate-{index:02d}'
            destination.mkdir(exist_ok=True)
            offset='' if count==1 else f'candidate-{index:02d}/'
            report(stage='loading',candidate=index,candidates=count,tokens=0,tokens_per_second=0,steps=0)
            log(f'[candidate] {index}/{count} seed {seed}')
            tick=time.perf_counter()
            if r.get('instrumental'): log('[mode] Instrumental only: section tags without words; the plan keeps the vocal voice silent')
            if r.get('render_source'):
                saved=json.loads((jobdir/'render-input.json').read_text())
                codec=saved['codec'];abc=saved['abc']
                prefix=saved.get('prefix') or token_prefix(pipe.tokenizer,generation_style,generation_lyrics,r['cot'],pipe.tokenizer.encode(abc) if abc else [])
                log(f'[nar] {len(codec)} frames ({len(codec)*0.04:.1f}s), {r["steps"]} midpoint steps; reusing saved music tokens')
                with real_audio_decoder(saved.get('real_audio')):
                    z=engine.synthesize(pipe.model,prefix,codec,seed,r['steps'])
                log('[vae] decoding')
                audio=pipe.decode(z);info={'abc':abc,'codec':codec,'prefix':prefix}
            elif (jobdir/'continue-input.json').is_file():
                # Extend / regenerate from here: continue the cut plan, then the music after the kept tokens.
                saved=json.loads((jobdir/'continue-input.json').read_text())
                abc,abc_ids,_=mlx_continuation.continue_plan(engine,pipe,generation_style,generation_lyrics,r['cot'],
                    saved.get('abc_prefix',''),pipe.abc_sampling,seed,pipe._progress('abc'))
                if abc:
                    rewritten=instrumental_plan(abc)
                    if rewritten!=abc: abc,abc_ids=rewritten,pipe.tokenizer.encode(rewritten)
                log(f'[continue] keeping {len(saved["codec"])} frames ({len(saved["codec"])*0.04:.1f}s) of {saved["source"]["title"]!r}; generating the rest')
                prefix,codec,truncated=mlx_continuation.continue_semantic(engine,pipe,generation_style,generation_lyrics,r['cot'],abc_ids,
                    saved['codec'],Sampling(**r['semantic_sampling']),seed,r['cfg_scale'],pipe._progress('semantic'))
                if truncated: log('[semantic] hit max_tokens')
                log(f'[nar] {len(codec)} frames ({len(codec)*0.04:.1f}s), {r["steps"]} midpoint steps')
                with real_audio_decoder(saved.get('real_audio')):
                    z=engine.synthesize(pipe.model,prefix,codec,seed,r['steps'])
                log('[vae] decoding')
                audio=pipe.decode(z);info={'abc':abc,'codec':codec,'prefix':prefix}
            else:
                abc=r['abc']
                # Instrumental songs plan first so the score can be rewritten before any music tokens are generated.
                if r.get('instrumental') and r['cot']!='off' and not abc: abc,_=write_plan(seed)
                audio,info=pipe(generation_style,generation_lyrics,cot=r['cot'],seed=seed,abc=abc or None,
                    cfg_scale=r['cfg_scale'],semantic_sampling=Sampling(**r['semantic_sampling']),steps=r['steps'])
            write_wav(destination/'audio.wav',audio)
            if info.get('abc'): (destination/'score.abc').write_text(info['abc'])
            (destination/'tokens.json').write_text(json.dumps(info['codec']))
            (destination/'render-prefix.json').write_text(json.dumps(info['prefix']))
            candidate_request={**r,'seed':seed,'candidates':1,'random_seed':False}
            (destination/'settings.json').write_text(json.dumps(candidate_request,indent=2))
            if count>1: (destination/'request.json').write_text(json.dumps(candidate_request,indent=2))
            item={'index':index,'seed':seed,'model':variant,'seconds':round(audio.shape[0]/48000,3),
                'elapsed':round(time.perf_counter()-tick,2),'abc':info.get('abc'),
                'prefix':offset,'peak_memory_gb':round(mx.get_peak_memory()/1024**3,2),'tokens':len(info['codec'])}
            meta['candidates'].append(item)
            if count==1: meta.update({k:v for k,v in item.items() if k not in {'index','prefix'}})
            meta['elapsed']=round(analysis_elapsed+time.perf_counter()-start,2)
            save_meta(meta)
            log(f'[candidate_done] {index}/{count}')
            del audio,info
            mx.clear_cache()
if is_analysis and r['kind']=='cover':
    tmp=jobdir/'analysis.tmp';tmp.write_text(json.dumps(meta,indent=2));replace_file(tmp,jobdir/'analysis.json')
    report(stage='analysis_complete');print('[analysis_done] Recording analyzed; generating cover next',flush=True)
else:
    save_meta(meta)
    log('[done] Finished in '+str(meta['elapsed'])+' seconds')
