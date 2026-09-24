/* Timeline picker for Extend / regenerate from here.
   It works on the saved music tokens of a finished version; the cut snaps to the bars of its plan. */
let timeline=null;
function continuationModel(){
 // The model chosen in the create form; falling back to the one that made the source song.
 const wanted=$('model').value||timeline?.job?.request?.model;
 return config?.models?.find(m=>m.id===wanted)?.id||config?.models?.find(m=>m.ready)?.id||wanted||null;
}
function snapToBar(seconds,bars,total){
 // Snap to the nearest bar start when one is close; otherwise keep the exact time.
 seconds=Math.max(0,Math.min(total,seconds));
 if(!bars?.length)return seconds;
 let best=bars[0];for(const t of bars)if(Math.abs(t-seconds)<Math.abs(best-seconds))best=t;
 return Math.abs(best-seconds)<=0.75?best:seconds;
}
function sectionAt(seconds,sections){return sections.find(s=>seconds>=s.start-1e-6&&seconds<s.end-1e-6)||null}
function sectionLabels(sections){
 const counts={};
 return sections.map(s=>{counts[s.tag]=(counts[s.tag]||0)+1;return s.tag+(sections.filter(x=>x.tag===s.tag).length>1?' '+counts[s.tag]:'')});
}
function setupSongTimeline(){
 const dialog=el('dialog',null,'composition-dialog timeline-dialog');dialog.id='songTimeline';dialog.setAttribute('aria-labelledby','timelineHeading');
 dialog.innerHTML='<div class="editor-heading"><h2 id="timelineHeading">Extend</h2><button type="button" id="closeTimeline" aria-label="Close">×</button></div>'
 +'<div class="timeline-body"><p id="timelineSource" class="muted"></p><p id="timelineNote" class="muted" hidden></p>'
 +'<div id="timelineStrip" class="timeline-strip" tabindex="0" role="slider" aria-label="Song position" aria-valuemin="0"><canvas id="timelineCanvas"></canvas><div id="timelineKeep" class="timeline-keep"></div><div id="timelineSections" class="timeline-sections"></div><div id="timelineCursor" class="timeline-cursor"></div></div>'
 +'<audio id="timelineAudio" preload="auto"></audio>'
 +'<div class="timeline-controls"><button type="button" id="timelinePlay" class="play-main" aria-label="Play from here" title="Play from the cursor · Space"><img src="/static/icons/play.svg?v=20260918" alt=""></button><label>Position (Seconds)<input id="timelineSeconds" type="number" min="0" step="0.1"></label><button type="button" id="timelineEnd" class="secondary">From the End</button></div>'
 +'<p id="timelineReadout" class="timeline-readout"></p>'
 +'<div id="timelineContinueFields"><label>Title<input id="timelineTitle" type="text" maxlength="100"></label><label>Lyrics<textarea id="timelineLyrics" rows="6" maxlength="30000"></textarea></label><p class="muted" id="timelineLyricsHint">Keep the lyrics of the part you keep and add the words for the new part after them.</p><label>Style<textarea id="timelineStyle" rows="3" maxlength="6000"></textarea></label><div class="two-cols"><label>New Part (Seconds)<input id="timelineExtendSeconds" type="number" min="5" max="600" step="1" value="30"></label><label>Versions<input id="timelineCandidates" type="number" min="1" max="8" value="1"></label></div><label class="alignment-check"><input type="checkbox" id="timelineExactLength"> Exactly this long (otherwise the song may end sooner on its own)</label><label>Seed<input id="timelineSeed" type="number" min="0" max="4294967295"></label><label class="alignment-check"><input type="checkbox" id="timelineLockSeed" checked> Lock seed (uncheck for a new random seed)</label></div>'
 +'<p id="timelineStatus" role="status"></p></div>'
 +'<div class="song-dialog-actions"><button type="button" id="timelineCancel">Cancel</button><button type="button" id="timelineSubmit" class="primary">Create</button></div>';
 document.body.append(dialog);
 $('closeTimeline').onclick=$('timelineCancel').onclick=()=>dialog.close();
 dialog.addEventListener('close',()=>{stopTimelinePreview();clearTimeout(recordingTokensTimer);recordingTokensTimer=null});
 $('timelineSubmit').onclick=submitTimeline;
 $('timelinePlay').onclick=toggleTimelinePreview;
 const audio=$('timelineAudio');
 audio.onplay=audio.onpause=()=>{const playing=!audio.paused&&!audio.ended;$('timelinePlay').querySelector('img').src='/static/icons/'+(playing?'pause':'play')+'.svg?v=20260918';$('timelinePlay').setAttribute('aria-label',playing?'Pause':'Play from here');$('timelinePlay').classList.toggle('playing',playing)};
 audio.onended=()=>{audio.onpause();setTimelinePosition(timeline.data.seconds,false)};
 $('timelineEnd').onclick=()=>{$('timelineExtendSeconds').value=30;$('timelineExactLength').checked=true;setTimelinePosition(timeline.data.seconds,false)};
 $('timelineExtendSeconds').onchange=()=>{$('timelineExtendSeconds').value=extendSeconds();renderTimeline()};
 $('timelineExactLength').onchange=renderTimeline;
 $('timelineSeconds').onchange=()=>setTimelinePosition(Number($('timelineSeconds').value),false);
 const strip=$('timelineStrip');
 const pointerTime=e=>{const rect=strip.getBoundingClientRect();return (e.clientX-rect.left)/Math.max(1,rect.width)*timeline.data.seconds};
 let dragging=false;
 strip.onpointerdown=e=>{if(!timeline)return;dragging=true;strip.setPointerCapture(e.pointerId);setTimelinePosition(pointerTime(e),!e.altKey)};
 strip.onpointermove=e=>{if(dragging&&timeline)setTimelinePosition(pointerTime(e),!e.altKey)};
 strip.onpointerup=strip.onpointercancel=()=>{dragging=false};
 strip.onkeydown=e=>{
  if(!timeline)return;const {bars,seconds}=timeline.data;const step=e.shiftKey?10:1;let t=timeline.seconds;
  if(e.key==='ArrowLeft'||e.key==='ArrowRight'){
   const dir=e.key==='ArrowRight'?1:-1;
   if(bars.length&&!e.shiftKey){const next=dir>0?bars.find(b=>b>t+1e-3):[...bars].reverse().find(b=>b<t-1e-3);t=next??(dir>0?seconds:0)}else t+=dir*step;
  }else if(e.key==='Home')t=0;else if(e.key==='End')t=seconds;
  else if(e.key===' '){e.preventDefault();toggleTimelinePreview();return}else return;
  e.preventDefault();setTimelinePosition(t,false);
 };
 window.addEventListener('resize',()=>{if(timeline&&dialog.open)renderTimeline()});
}
/* The recording's music tokens (POST /api/tokens, cached per recording and analysis length). Resolves once
   they are ready; ``status`` shows progress while the dialog stays open, and closing it abandons the wait. */
async function prepareRecordingTokens(options,dialog,status,setTimer){
 let state=await api('/api/tokens',{method:'POST',body:JSON.stringify(options)}),restarted=false;
 if(state.job&&['starting','running','queued'].includes(state.status)){active=selected=state.job;lastStatus='';poll()}
 while(!['ready','failed','cancelled','interrupted'].includes(state.status)){
  status.textContent=state.status==='queued'?'Waiting for the current job to finish…':'Listening to the recording and writing its music tokens…';
  await new Promise(resolve=>setTimer(setTimeout(resolve,1500)));
  if(!dialog.open)return null;
  state=await api('/api/tokens?'+new URLSearchParams(options));
  if(state.status==='missing'){
   if(restarted)throw new Error('The recording could not be turned into music tokens.');
   restarted=true;state=await api('/api/tokens',{method:'POST',body:JSON.stringify(options)});
  }
 }
 if(state.status!=='ready')throw new Error(state.error||'The recording could not be turned into music tokens.');
 return state;
}
/* The recording a picker works on: an uploaded song entry from the library (the whole original recording). */
function uploadRecording(j,c){return {id:j.request.audio_id,name:songTitle(j,c),seconds:Number(c?.seconds)||0,sourceSeconds:0}}
/* Recordings use the same picker as saved songs. Extend keeps the recording's tokens and renders them with the
   real-audio decoder while the model writes what follows, score-free (MLX models). */
let recordingTokensTimer=null;
async function openContinueRecording(rec){
 if(!rec)return;
 try{
  const data=await api('/api/uploads/'+rec.id+'/timeline?'+new URLSearchParams({source_seconds:rec.sourceSeconds}));
  const name=rec.name||'Recording';
  timeline={job:null,recording:rec,index:1,data,seconds:0,labels:[]};
  $('timelineHeading').textContent='Continue This Recording';
  $('timelineSource').textContent=name+' · '+clockTime(data.seconds)+' · recording: cuts are not snapped to bars';
  $('timelineNote').hidden=false;
  $('timelineNote').textContent='The recording is turned into music tokens (once per recording), kept up to the cursor, and the model continues it from your style and lyrics. The kept part is rendered again from its tokens with the real-audio decoder, so expect it close to the original rather than identical. Experimental.';
  $('timelineStatus').textContent='';$('timelineStatus').classList.remove('error');
  stopTimelinePreview();$('timelineAudio').src='/api/uploads/'+rec.id;
  $('timelineTitle').value=(name+' · extended').slice(0,100);
  $('timelineLyrics').value=$('lyrics').value;$('timelineStyle').value=$('style').value;
  $('timelineLyrics').parentElement.hidden=false;$('timelineLyricsHint').hidden=false;
  $('timelineLyricsHint').textContent='Lyrics for the new part only; the kept part sings what the recording sings.';
  $('timelineSeed').value=Number($('seed').value)||831001;$('timelineLockSeed').checked=!!$('lockSeed').checked;$('timelineCandidates').value=1;
  $('timelineExtendSeconds').value=30;$('timelineExactLength').checked=true;
  setTimelinePosition(data.seconds,false);
  $('timelineSubmit').disabled=false;
  $('songTimeline').showModal();renderTimeline();
 }catch(e){notice(e.message,true)}
}
async function submitRecordingTimeline(){
 const {recording:rec,data,seconds}=timeline,status=$('timelineStatus'),dialog=$('songTimeline');
 const options={audio_id:rec.id,source_seconds:rec.sourceSeconds};
 const model=continuationModel();
 if(!model)throw new Error('No model is available. Download one under Models.');
 if(['cuda-bf16','cuda-fp8'].includes(model))throw new Error('Extending a recording runs on the MLX models for now. Select YuE2 bf16.');
 if(!$('timelineStyle').value.trim()||!$('timelineLyrics').value.trim())throw new Error('Enter a style and lyrics for the new part.');
 if(!await prepareRecordingTokens(options,dialog,status,t=>{recordingTokensTimer=t}))return;
 const atEnd=seconds>=data.seconds-0.05;
 const body={kind:'generate',workspace_id:saveWorkspace,model,candidates:Math.max(1,Math.min(8,Number($('timelineCandidates').value)||1)),title:$('timelineTitle').value.trim()||'Untitled song',
  style:$('timelineStyle').value,lyrics:$('timelineLyrics').value,instrumental:false,cot:'off',voice:$('voice').value||'any',abc:'',
  seed:Number($('timelineSeed').value)||0,random_seed:!$('timelineLockSeed').checked,cfg_scale:Number($('cfg').value),steps:Number($('steps').value)||2,
  abc_sampling:config.defaults.abc,semantic_sampling:extendSampling(),
  audio_id:rec.id,source_seconds:options.source_seconds,continue_recording:true,continue_seconds:atEnd?null:Number(seconds.toFixed(3))};
 const result=await api('/api/jobs',{method:'POST',body:JSON.stringify(body)});
 accepted(result);requestArtwork({id:result.id,title:body.title,kind:'generate',request:body});
 dialog.close();await refreshLibrary();updateUI();poll();
}
async function openSongTimeline(job,candidate,seconds=null){
 try{
  const index=candidate?.index||1;
  const data=await api('/api/jobs/'+job.id+'/songs/'+index+'/timeline');
  const sections=data.sections||[];
  timeline={job,index,data,seconds:0,labels:sectionLabels(sections)};
  $('timelineHeading').textContent='Extend';
  $('timelineNote').hidden=true;$('timelineLyricsHint').textContent='Keep the lyrics of the part you keep and add the words for the new part after them.';
  $('timelineSource').textContent=(data.title||songTitle(job,candidate))+' · '+clockTime(data.seconds)+(data.bpm?' · '+data.bpm+' BPM · '+data.bars.length+' bars':' · no plan: cuts are not snapped to bars');
  $('timelineStatus').textContent='';$('timelineStatus').classList.remove('error');
  stopTimelinePreview();$('timelineAudio').src=songURL(job,candidate);
  $('timelineTitle').value=(data.title||songTitle(job,candidate)).replace(/ · extended.*$/,'')+' · extended';
  $('timelineLyrics').value=data.lyrics||'';$('timelineStyle').value=data.style||'';
  // The source song's seed is kept, like when a version's settings are reused; a new random one is opt-in.
  $('timelineSeed').value=candidate?.seed??job.request?.seed??831001;$('timelineLockSeed').checked=true;$('timelineCandidates').value=1;
  $('timelineLyrics').parentElement.hidden=$('timelineLyricsHint').hidden=!!data.instrumental;
  $('timelineSubmit').disabled=false;
  // Default: regenerate the last section, or continue after the end when the song has no plan.
  const start=seconds??(sections.length>1?sections[sections.length-1].start:data.seconds);
  // Regenerating a section: let the model finish the song, up to the removed length. Adding after the end: exactly 30 s.
  const atEnd=start>=data.seconds-0.05;
  $('timelineExtendSeconds').value=atEnd?30:Math.max(5,Math.min(600,Math.round(data.seconds-start+15)));
  $('timelineExactLength').checked=atEnd;
  setTimelinePosition(start,seconds!=null);
  $('songTimeline').showModal();renderTimeline();
 }catch(e){notice(e.message,true)}
}
function extendSeconds(){return Math.max(5,Math.min(600,Math.round(Number($('timelineExtendSeconds').value)||30)))}
function extendSampling(){
 // The token budget applies to the new part only; 25 music frames make one second.
 // "Exactly" forbids the end-of-music token until the length is reached, so the new part fills it.
 const base=sampling(),frames=extendSeconds()*25;
 return {...base,max_tokens:frames,min_tokens:$('timelineExactLength').checked?frames:Math.min(base.min_tokens,frames)};
}
let previewFrame=0;
function toggleTimelinePreview(){
 // Play the song from the cursor; pausing leaves the cursor where the music stopped.
 const audio=$('timelineAudio');if(!timeline||!audio.src)return;
 if(!audio.paused){audio.pause();setTimelinePosition(audio.currentTime,true);return}
 const main=$('playerAudio');if(main&&!main.paused)main.pause();
 audio.currentTime=timeline.seconds>=timeline.data.seconds-0.05?0:timeline.seconds;
 audio.play().catch(e=>notice(e.message,true));
 cancelAnimationFrame(previewFrame);
 const tick=()=>{if(!timeline||audio.paused)return;timeline.seconds=Math.min(timeline.data.seconds,audio.currentTime);renderTimeline();previewFrame=requestAnimationFrame(tick)};
 previewFrame=requestAnimationFrame(tick);
}
function stopTimelinePreview(){
 const audio=$('timelineAudio');cancelAnimationFrame(previewFrame);
 if(!audio)return;audio.pause();audio.removeAttribute('src');audio.load();
}
function setTimelinePosition(seconds,snap=true){
 const {data}=timeline;
 let t=Math.max(0,Math.min(data.seconds,Number(seconds)||0));
 if(snap)t=snapToBar(t,data.bars,data.seconds);
 timeline.seconds=t;renderTimeline();
 const audio=$('timelineAudio');
 if(audio&&!audio.paused&&Math.abs(audio.currentTime-t)>0.25)audio.currentTime=t;
}
function renderTimeline(){
 if(!timeline)return;
 const {data,seconds,labels}=timeline,strip=$('timelineStrip'),canvas=$('timelineCanvas'),total=data.seconds||1;
 const width=strip.clientWidth||600,height=strip.clientHeight||96;
 canvas.width=width*devicePixelRatio;canvas.height=height*devicePixelRatio;canvas.style.width=width+'px';canvas.style.height=height+'px';
 const ctx=canvas.getContext('2d');ctx.scale(devicePixelRatio,devicePixelRatio);ctx.clearRect(0,0,width,height);
 const peaks=data.peaks||[],mid=height*0.42,amp=height*0.34;
 ctx.fillStyle='#6b6b73';
 for(let x=0;x<width;x++){const p=peaks.length?peaks[Math.min(peaks.length-1,Math.floor(x/width*peaks.length))]:0.2;const h=Math.max(1,p*amp);ctx.fillRect(x,mid-h,1,h*2)}
 ctx.fillStyle='#ffffff22';
 for(const b of data.bars){const x=b/total*width;ctx.fillRect(x,0,1,height*0.78)}
 const x=seconds/total*100;
 $('timelineCursor').style.left=x+'%';
 $('timelineKeep').style.width=x+'%';
 const box=$('timelineSections');box.replaceChildren();
 data.sections.forEach((s,i)=>{
  const b=el('button',labels[i],'timeline-section'+(s.sung?' sung':''));b.type='button';b.style.left=s.start/total*100+'%';b.style.width=Math.max(0.5,(s.end-s.start)/total*100)+'%';
  b.title=labels[i]+' · '+clockTime(s.start)+' – '+clockTime(s.end)+(s.sung?' · sung':' · instrumental');
  b.onclick=e=>{e.stopPropagation();setTimelinePosition(s.start,false)};b.onpointerdown=e=>e.stopPropagation();
  box.append(b);
 });
 strip.setAttribute('aria-valuemax',String(total));strip.setAttribute('aria-valuenow',String(seconds));strip.setAttribute('aria-valuetext',clockTime(seconds));
 $('timelineSeconds').value=seconds.toFixed(1);$('timelineSeconds').max=total;
 const section=sectionAt(seconds,data.sections),label=section?labels[data.sections.indexOf(section)]:null;
 const add=extendSeconds(),exact=$('timelineExactLength').checked;
 const part=(exact?'add exactly ':'add up to ')+clockTime(add)+' (song becomes '+(exact?'':'at most ')+clockTime(seconds+add)+').';
 const what=timeline.recording?'recording':'song';
 $('timelineReadout').textContent=seconds>=total-0.05?'Keep the whole '+what+' ('+clockTime(total)+') and '+part
  :'Keep 0:00 – '+clockTime(seconds)+', '+(timeline.recording?'continue from ':'regenerate from ')+(label?label+' at ':'')+clockTime(seconds)+' and '+part;
}
async function submitTimeline(){
 if(!timeline)return;
 const {job,index,data,seconds}=timeline,status=$('timelineStatus');
 status.textContent='';status.classList.remove('error');$('timelineSubmit').disabled=true;
 try{
  askRenderAlerts();
  if(timeline.recording){await submitRecordingTimeline();return}
  const model=continuationModel();
  if(!model)throw new Error('No model is available. Download one under Models.');
  const atEnd=seconds>=data.seconds-0.05;
  const body={...job.request,kind:'generate',model,edit_id:'',render_source:'',render_candidate:1,origin:null,abc:'',audio_id:'',
   title:$('timelineTitle').value.trim()||'Untitled song',style:$('timelineStyle').value,lyrics:data.instrumental?(data.lyrics||''):$('timelineLyrics').value,
   instrumental:!!data.instrumental,cot:data.cot||'full',voice:data.voice||'any',
   seed:Number($('timelineSeed').value)||0,random_seed:!$('timelineLockSeed').checked,candidates:Math.max(1,Math.min(8,Number($('timelineCandidates').value)||1)),
   steps:Number($('steps').value)||job.request.steps||2,cfg_scale:Number($('cfg').value),semantic_sampling:extendSampling(),abc_sampling:config.defaults.abc,
   workspace_id:saveWorkspace,continue_source:job.id,continue_candidate:index,continue_seconds:atEnd?null:Number(seconds.toFixed(3))};
  delete body.api_key;
  const result=await api('/api/jobs',{method:'POST',body:JSON.stringify(body)});
  accepted(result);requestArtwork({id:result.id,title:body.title,kind:'generate',request:body});
  $('songTimeline').close();await refreshLibrary();updateUI();poll();
 }catch(e){status.textContent=e.message;status.classList.add('error')}
 finally{$('timelineSubmit').disabled=false}
}
