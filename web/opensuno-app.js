const $=id=>document.getElementById(id);
let config,mode='create',selected=null,active=null,source=null,timer=null;
let lastStatus='',resultFingerprint='',restoring=false,uploading=false;
let runningJob=null,audioActivity=null,renderNodeRefresh=null,configJSON='';
// Synced (LRC) lyrics kept alongside lyrics chosen from LRCLIB; covers use the timestamps to place lines.
let pageSession=null,pageSessionToken=null;
// Instrumental songs keep their section tags with no words; the planned score then leaves the vocal voice silent.
const DEFAULT_STRUCTURE=['Intro','Verse','Pre-Chorus','Chorus','Verse','Pre-Chorus','Chorus','Bridge','Chorus','Outro'].map(t=>'['+t+']').join('\n\n')+'\n';
function lyricsArePlaceholder(value){
 const t=(value||'').trim();
 return !t||t===DEFAULT_STRUCTURE.trim();
}
function applyRecordingLyrics(lyrics,force=false){
 if(!lyrics||(!force&&!lyricsArePlaceholder($('lyrics').value)))return false;
 $('lyrics').value=lyrics;
 const expanded=$('expandedLyrics');if(expanded)expanded.value=lyrics;
 $('instrumental').checked=false;
 return true;
}
const VOCAL_WORDS=/\b(vocals?|vocalist|singer|singing|sings?|sung|voices?|lead vocal|choir|rap|rapper|rapping|lyrics?|a cappella|acapella|harmonies|male|female|duet|soprano|alto|tenor|baritone|english|spanish|french|german|italian|portuguese|romanian|mandarin|chinese|japanese|korean|hindi|arabic|russian|turkish)\b/gi;
function vocalWords(style){return [...new Set((style||'').toLowerCase().match(VOCAL_WORDS)||[])].sort()}
function connectPageSession(){
 if(!config?.token)return;
 if(pageSession&&pageSessionToken===config.token)return;
 pageSession?.close();pageSessionToken=config.token;
 pageSession=new EventSource('/api/session?token='+encodeURIComponent(config.token));
}
window.addEventListener('pagehide',()=>{pageSession?.close();pageSession=null;});
window.addEventListener('pageshow',connectPageSession);
// Music-token sampling behind the Weirdness and Repetition sliders. Values restored from a saved song that sit off
// the slider curves are kept verbatim (the slider reads "Custom"). Song length is only "Maximum duration".
let semanticState=null;
function durationTokens(){return Math.max(1,Math.min(18000,Math.round(Number($('duration').value)*25)||1))}
let preloadBusy=false;
async function prepareModels(retry=false){
 if(!config||config.hosted||preloadBusy||active)return;
 preloadBusy=true;
 try{
  const targets=config.default_model==='cuda-bf16'?[$('model').value]:(mode==='cover'?['analysis','bf16']:['bf16']);
  if(config.default_model!=='cuda-bf16'&&$('model').value!=='bf16')targets.push($('model').value);
  for(const model of targets){
   const ready=model==='analysis'?config.transcriber_ready:config.models.find(m=>m.id===model)?.ready;
   if(!ready||['ready','loading'].includes(config.preloads?.[model]?.status)||(!retry&&config.preloads?.[model]?.status==='failed'))continue;
   const state=await api('/api/models/preload?model='+model,{method:'POST'});
   config.preloads={...config.preloads,[model]:state};
  }
 }catch{}finally{preloadBusy=false}
}
function showPreload(){
 let label=$('preloadStatus');
 if(!label){label=document.createElement('p');label.id='preloadStatus';label.setAttribute('role','status');$('modelSetup').after(label)}
 const keys=mode==='cover'?['analysis',$('model').value]:[$('model').value];
 const loading=keys.find(k=>config?.preloads?.[k]?.status==='loading');
 const failed=keys.find(k=>config?.preloads?.[k]?.status==='failed');
 const key=loading||failed, state=config?.preloads?.[key];
 label.hidden=!key;
 label.textContent=key?(key==='analysis'?'Cover analysis':generationModelLabel(config.models.find(m=>m.id===key)||key))+': '+state.detail+(loading&&state.elapsed?' · '+state.elapsed+'s':''):'';
}
function notice(text,error=false){$('notice').hidden=!text;$('notice').textContent=text;$('notice').classList.toggle('error',error)}
async function api(url,options={}){
 const res=await fetch(url,{...options,headers:{...(options.body instanceof FormData?{}:{'Content-Type':'application/json'}),'X-Studio-Token':config?.token,...options.headers}});
 let data;try{data=await res.json()}catch{throw Error('Studio returned an unreadable response.')}
 if(!res.ok){let d=data.detail;if(Array.isArray(d))d=d.map(x=>x.loc.slice(1).join(' → ')+': '+x.msg).join('; ');throw Error(d||'Request failed')}return data;
}
const samplingDescriptions={
 steps:'How many passes polish the audio. More sounds cleaner and takes longer; the song itself stays the same, so you can re-render a song you like with more steps later.',
 composition:'Familiar to unexpected. How adventurous the model is while writing the score: chord choices, melodic turns and how busy the instrumental line gets. 50% is the normal result. Only applies while a plan is written (not with No Plan, or covers that keep the recording\u2019s score as is).',
 weirdness:'Safe to chaos. 50% is the normal result. Lower for a safer, more conventional song; higher for surprises, which can get messy.',
 style_influence:'Loose to strong. How closely the song follows your style text and lyrics. Raise it if the genre, instruments or mood get ignored; too strong can sound forced. Does not change the melody.',
 repetition:'How much the music may repeat what it just played. Allow more for steady grooves and long held notes; allow less if a song gets stuck in loops.',
 lora_strength:'How much of this adapter is mixed into the model for this song. 100% is the usual recipe; lower is subtler, above 100% can overpower the base model.',
 sound_lora_strength:'How much of the sound adapter is mixed into the decoder. 100% is the usual recipe; it only changes the rendered audio, so the song\u2019s notes and words stay the same.'
};

let activeSettingTip=null;
function closeSettingTip(){if(activeSettingTip){activeSettingTip.tip.hidden=true;activeSettingTip=null}}
document.addEventListener('pointerdown',event=>{if(activeSettingTip&&!activeSettingTip.button.contains(event.target)&&!activeSettingTip.tip.contains(event.target))closeSettingTip()});
document.addEventListener('keydown',event=>{if(event.key==='Escape')closeSettingTip()});
window.addEventListener('resize',closeSettingTip);
document.addEventListener('scroll',event=>{if(activeSettingTip&&(event.target===document||event.target.contains?.(activeSettingTip.button)))closeSettingTip()},true);
function infoTip(host,name,text,id){
 const button=document.createElement('button');button.type='button';button.className='setting-info';button.textContent='i';button.setAttribute('aria-label','About '+name);
 const tip=document.createElement('div');tip.className='setting-tooltip';tip.id=id;tip.setAttribute('role','tooltip');tip.hidden=true;
 tip.textContent=text;button.setAttribute('aria-describedby',id);document.body.append(tip);
 let leaveTimer;
 const show=()=>{clearTimeout(leaveTimer);closeSettingTip();tip.hidden=false;activeSettingTip={button,tip};const r=button.getBoundingClientRect(),t=tip.getBoundingClientRect();tip.style.left=Math.max(8,Math.min(innerWidth-t.width-8,r.left+r.width/2-t.width/2))+'px';tip.style.top=(r.top-t.height-9>=8?r.top-t.height-9:Math.min(innerHeight-t.height-8,r.bottom+9))+'px'};
 const leave=()=>{clearTimeout(leaveTimer);leaveTimer=setTimeout(()=>{if(activeSettingTip?.button===button)closeSettingTip()},150)};
 button.addEventListener('pointerenter',event=>{if(event.pointerType!=='touch')show()});button.addEventListener('pointerleave',leave);
 button.addEventListener('focus',show);button.addEventListener('blur',leave);button.addEventListener('click',event=>{event.preventDefault();event.stopPropagation();show()});
 tip.addEventListener('pointerenter',()=>clearTimeout(leaveTimer));tip.addEventListener('pointerleave',leave);
 host.append(button);
}
function settingInfo(label,key,name,defaultValue){
 const wrap=document.createElement('span');wrap.className='setting-label';
 label.removeAttribute('title');label.before(wrap);wrap.append(label);
 infoTip(wrap,name,samplingDescriptions[key]+' Default: '+defaultValue+'.','help_'+label.htmlFor);
}

// Ticks brighten near the thumb; CSS reads --p (0..1) to position the highlight.
function paintSliderTicks(){
 for(const r of document.querySelectorAll('.sampling-slider-row input[type=range]')){
  const min=Number(r.min),max=Number(r.max),p=(Number(r.value)-min)/(max-min);
  r.style.setProperty('--p',Number.isFinite(p)?Math.max(0,Math.min(1,p)).toFixed(4):'.5');
 }
}
function sampling(){
 const cap=durationTokens(),d=config.defaults.semantic,s=semanticState||d;
 return {temperature:s.temperature,top_p:s.top_p,top_k:s.top_k,repetition_penalty:s.repetition_penalty,penalty_window:s.penalty_window,
  min_tokens:Math.min(s.min_tokens??d.min_tokens,cap),max_tokens:cap};
}
function setSampling(v){
 const d=config.defaults.semantic;
 semanticState={...d};
 for(const key of ['temperature','top_p','top_k','repetition_penalty','penalty_window','min_tokens'])if(Number.isFinite(v?.[key]))semanticState[key]=v[key];
 $('duration').value=(Number.isFinite(v?.max_tokens)?v.max_tokens:d.max_tokens)/25;
 syncSimpleSliders();
}
function request(kind=mode==='cover'?'cover':'generate'){
 const body={kind,workspace_id:saveWorkspace,edit_id:typeof compositionDraft!=='undefined'?compositionDraft?.id||'':'',model:$('model').value,candidates:['generate','cover'].includes(kind)?Number($('candidates').value):1,
 instrumental:$('instrumental').checked,hook_melody:$('instrumental').checked&&$('hookMelody').checked,title:$('title').value.trim()||'Untitled song',style:$('style').value,lyrics:$('instrumental').checked?($('lyrics').value.trim()||DEFAULT_STRUCTURE):$('lyrics').value,
 cot:mode==='cover'?$('preserve').value:$('cot').value,voice:$('voice').value,abc:'',
 seed:Number($('seed').value),random_seed:!$('lockSeed').checked,cfg_scale:Number($('cfg').value),steps:Number($('steps').value),
 abc_sampling:abcSampling(),semantic_sampling:sampling(),audio_id:mode==='cover'?source?.id||'':'',
 melody_only:$('preserve').value==='melody',source_seconds:Number($('sourceSeconds').value),
 arrange:$('arrangeCover').checked,vocal_octave:$('vocalOctave').value,
 lora:$('lora').value,lora_strength:Number($('loraStrengthSlider').value),
 sound_lora:$('soundLora').value,sound_lora_strength:Number($('soundLoraStrengthSlider').value)};
 if(['generate','cover'].includes(kind)){
  const settings=typeof StyleAISettings==='undefined'?{}:StyleAISettings.get();
  if(settings.apiKey)body.api_key=settings.apiKey;
 }
 return body;
}
function setupRenderSliders(){
 settingInfo(document.querySelector('label[for=stepsSlider]'),'steps','Audio Steps',2);
 $('stepsSlider').oninput=()=>{$('steps').value=$('stepsSlider').value;updateUI();saveDraft()};
 $('steps').addEventListener('input',()=>{if($('steps').value!=='')$('stepsSlider').value=$('steps').value;paintSliderTicks()});
 $('resetSteps').onclick=()=>{$('steps').value=2;updateUI();saveDraft()};
}

/* Friendly controls. Each one is a curve through the checkpoint defaults at the midpoint; a restored song whose
   values no longer sit on the curve reads back "Custom" and keeps them until the slider is moved. */
const lerp=(a,b,t)=>a+(b-a)*t;
function curve(v,[lo,mid,hi]){return v<=50?lerp(lo,mid,v/50):lerp(mid,hi,(v-50)/50)}
function uncurve(x,[lo,mid,hi]){return x<=mid?50*(x-lo)/(mid-lo):50+50*(x-mid)/(hi-mid)}
// Left allows more repetition (steady grooves), right pushes away from it.
const REPETITION_LEVELS=[['More allowed',1.05],['A bit more',1.12],['Normal',1.2],['Less',1.3],['Much less',1.45]];
let simpleCurves=null;
// Planner (score-writing) sampling: the Composition slider bends these; values restored off the curve stay as they were.
let abcCustom=null;
function simpleSetup(){
 const d=config.defaults.semantic,a=config.defaults.abc;
 simpleCurves={temperature:[.6,d.temperature,1.4],top_p:[.85,d.top_p,1],top_k:[40,d.top_k,250],cfg:[1,1.2,2],
  abc_temperature:[.5,a.temperature,1],abc_top_p:[.8,a.top_p,.97],abc_top_k:[12,a.top_k,64]};
}
function abcSampling(){
 if(abcCustom)return {...config.defaults.abc,...abcCustom};
 const c=simpleCurves,v=Number($('compositionSlider').value);
 if(!c)return config.defaults.abc;
 return {...config.defaults.abc,temperature:Number(curve(v,c.abc_temperature).toFixed(2)),top_p:Number(curve(v,c.abc_top_p).toFixed(2)),top_k:Math.round(curve(v,c.abc_top_k))};
}
function setComposition(s){
 // A saved request's planner sampling: back onto the slider when it sits on the curve, otherwise kept verbatim.
 abcCustom=null;
 const c=simpleCurves,a=config.defaults.abc;
 if(!s||!c){$('compositionSlider').value=50;syncSimpleSliders();return}
 const v=Math.max(0,Math.min(100,uncurve(Number(s.temperature),c.abc_temperature)));
 const fits=s.temperature>=c.abc_temperature[0]&&s.temperature<=c.abc_temperature[2]&&Math.abs(s.top_p-curve(v,c.abc_top_p))<.011&&Math.abs(s.top_k-curve(v,c.abc_top_k))<1.5;
 if(fits)$('compositionSlider').value=Math.round(v);
 else abcCustom={temperature:s.temperature??a.temperature,top_p:s.top_p??a.top_p,top_k:s.top_k??a.top_k,repetition_penalty:s.repetition_penalty??a.repetition_penalty,penalty_window:s.penalty_window??a.penalty_window};
 syncSimpleSliders();
}
function applyWeirdness(w){
 const c=simpleCurves;semanticState??={...config.defaults.semantic};
 semanticState.temperature=Number(curve(w,c.temperature).toFixed(2));
 semanticState.top_p=Number(curve(w,c.top_p).toFixed(2));
 semanticState.top_k=Math.round(curve(w,c.top_k));
}
function applyStyleInfluence(s){$('cfg').value=curve(s,simpleCurves.cfg).toFixed(2)}
function applyRepetition(i){semanticState??={...config.defaults.semantic};semanticState.repetition_penalty=REPETITION_LEVELS[i][1]}
function settle(slider,value){if(Math.abs(Number(slider.value)-value)>=1)slider.value=Math.round(value)}
function syncSimpleSliders(){
 if(!simpleCurves)return;
 const c=simpleCurves,st=semanticState||config.defaults.semantic,t=Number(st.temperature);
 const w=Math.max(0,Math.min(100,uncurve(t,c.temperature)));
 settle($('weirdnessSlider'),w);$('weirdnessValue').value=$('weirdnessSlider').value+'%';
 const g=Number($('cfg').value);let s=Math.max(0,Math.min(100,uncurve(g,c.cfg)));
 settle($('styleInfluenceSlider'),s);$('styleInfluenceValue').value=g<c.cfg[0]-1e-9||g>c.cfg[2]+1e-9?'Custom':$('styleInfluenceSlider').value+'%';
 const r=Number(st.repetition_penalty),level=REPETITION_LEVELS.findIndex(([,v])=>Math.abs(v-r)<.005);
 if(level>=0)$('repetitionSlider').value=level;$('repetitionValue').value=level>=0?REPETITION_LEVELS[level][0]:'Custom';
 $('compositionValue').value=abcCustom?'Custom':$('compositionSlider').value+'%';
 paintSliderTicks();
}
function syncComposition(){
 // The planner only runs when a score is written: not with No Plan, nor for covers that keep the recording's score.
 const planned=mode==='cover'?$('arrangeCover').checked:$('cot').value!=='off';
 const row=$('compositionSliderRow');row.classList.toggle('inactive',!planned);
 $('compositionSlider').disabled=!planned;$('resetComposition').disabled=!planned;
 row.title=planned?'':mode==='cover'?'Not used: the cover keeps the recording\u2019s score as it was heard. Turn on Arrange with the model to let the planner write around it.':'Not used with No Plan: no score is written.';
}
// Suno-style bottom grab bar instead of the native corner grip; heights persist per editor.
function setupResizeHandles(){
 for(const handle of document.querySelectorAll('.resize-handle')){
  const area=$(handle.dataset.for);if(!area)continue;
  const key='opensuno-height-'+area.id,min=parseFloat(getComputedStyle(area).minHeight)||80,max=()=>Math.max(min,window.innerHeight*.8);
  const apply=h=>{h=Math.round(Math.min(max(),Math.max(min,h)));area.style.height=h+'px';return h};
  const save=h=>{try{localStorage.setItem(key,String(h))}catch{}};
  try{const saved=Number(localStorage.getItem(key));if(saved>=min)apply(saved)}catch{}
  let startY=0,startH=0;
  handle.onpointerdown=e=>{
   if(e.button!==0)return;
   e.preventDefault();startY=e.clientY;startH=area.getBoundingClientRect().height;
   handle.classList.add('dragging');handle.setPointerCapture(e.pointerId);
  };
  handle.onpointermove=e=>{if(handle.classList.contains('dragging'))apply(startH+e.clientY-startY)};
  const stop=e=>{if(!handle.classList.contains('dragging'))return;handle.classList.remove('dragging');try{handle.releasePointerCapture(e.pointerId)}catch{}save(area.getBoundingClientRect().height)};
  handle.onpointerup=handle.onpointercancel=stop;
  handle.ondblclick=()=>{area.style.height='';try{localStorage.removeItem(key)}catch{}};
  handle.onkeydown=e=>{
   const step=e.shiftKey?80:24;
   if(e.key==='ArrowUp'||e.key==='ArrowDown'){e.preventDefault();save(apply(area.getBoundingClientRect().height+(e.key==='ArrowDown'?step:-step)))}
  };
 }
}
function setupSimpleSliders(){
 simpleSetup();
 settingInfo(document.querySelector('label[for=compositionSlider]'),'composition','Composition','50%');
 settingInfo(document.querySelector('label[for=weirdnessSlider]'),'weirdness','Weirdness','50%');
 settingInfo(document.querySelector('label[for=styleInfluenceSlider]'),'style_influence','Style Influence','50%');
 settingInfo(document.querySelector('label[for=repetitionSlider]'),'repetition','Repetition','Normal');
 settingInfo(document.querySelector('label[for=loraStrengthSlider]'),'lora_strength','LoRA Strength','100%');
 settingInfo(document.querySelector('label[for=soundLoraStrengthSlider]'),'sound_lora_strength','Sound LoRA Strength','100%');
 const commit=()=>{updateUI();saveDraft()};
 $('compositionSlider').oninput=()=>{abcCustom=null;commit()};
 $('resetComposition').onclick=()=>{abcCustom=null;$('compositionSlider').value=50;commit()};
 $('weirdnessSlider').oninput=()=>{applyWeirdness(Number($('weirdnessSlider').value));commit()};
 $('styleInfluenceSlider').oninput=()=>{applyStyleInfluence(Number($('styleInfluenceSlider').value));commit()};
 $('repetitionSlider').oninput=()=>{applyRepetition(Number($('repetitionSlider').value));commit()};
 $('resetWeirdness').onclick=()=>{applyWeirdness(50);commit()};
 $('resetStyleInfluence').onclick=()=>{applyStyleInfluence(50);commit()};
 $('resetRepetition').onclick=()=>{applyRepetition(2);commit()};
 syncSimpleSliders();
}
function updateUI(){
 if(typeof showCompositionDraft==='function')showCompositionDraft();
 $('stepsSlider').value=$('steps').value;
 syncSimpleSliders();
 const available=config?.models?.find(x=>x.id===$('model').value)?.ready;
 const instrumental=$('instrumental').checked;
 // Empty [Intro]/[Verse] tags still go to the model; they do not need to sit in the lyrics box.
 const hideLyrics=instrumental;
 $('findLyrics').hidden=instrumental;$('addLyricTag').hidden=hideLyrics;$('expandLyrics').hidden=hideLyrics;$('lyricsGroup').hidden=hideLyrics;$('hookMelodyBox').hidden=!instrumental;
 $('lyricsLabel').textContent='Lyrics';
 $('lyrics').placeholder='[Verse]\nWrite your lyrics here…\n\n[Chorus]\n';
 const pulls=instrumental?vocalWords($('style').value):[];
 $('vocalWarning').hidden=!pulls.length;$('vocalWarning').textContent=pulls.length?'Style mentions '+pulls.join(', ')+'. Words about voices or languages pull the model towards singing; describe the instruments instead.':'';
 $('clearLyrics').disabled=!$('lyrics').value.trim();window.refreshLyricsStructure?.();
 $('clearStyle').disabled=!$('style').value.trim();
 $('coverPanel').hidden=mode!=='cover';
 // Covers pick their plan with "Preserve"; a saved composition always carries a score, so "No plan" cannot apply to it.
 $('compositionRow').hidden=mode==='cover';
 $('voice').disabled=instrumental;
 const hasDraft=typeof compositionDraft!=='undefined'&&!!compositionDraft;
 // Instrumental songs need a score so its vocal voice can be kept silent.
 $('cot').querySelector('option[value=off]').disabled=hasDraft||instrumental;if((hasDraft||instrumental)&&$('cot').value==='off')$('cot').value='full';

 $('generate').replaceChildren(icon('musical-note'),document.createTextNode('Create'));
 const analysisReady=!!source&&!source.awaitingTrim&&sourceAnalysis.key===analysisKey()&&sourceAnalysis.status==='ready';
 $('generate').disabled=uploading||!available||(mode==='cover'&&!analysisReady);
 $('arrangeBox').hidden=mode!=='cover'||!source;
 // Instrumental covers move the melody to the instrument; there is no singer to fit it to.
 $('vocalOctaveRow').hidden=instrumental;
 syncComposition();syncLora();
 showModelSetup();showPreload();
 // The panel follows the running job; a song of yours that is waiting is described underneath with its own button.
 const stopping=!!active&&runningJob?.id===active&&runningJob.status==='cancelling',queued=!!selected&&queuePosition(selected)>0;
 $('cancel').hidden=!(active&&selected&&(selected===active||queued));
 $('cancel').disabled=stopping;
 $('cancel').replaceChildren(icon('x-mark'),document.createTextNode(stopping?'Stopping…':'Stop generation'));
 const waiting=(config.queue||[]).length;
 $('generationQueue').hidden=!(queued||(active&&waiting));
 $('dequeue').hidden=!queued;
 $('generationQueueText').textContent=queued?(waiting>1?`Your song is ${ordinal(queuePosition(selected))} of ${waiting} waiting; it starts when the current generation finishes.`:'Your song is next; it starts when the current generation finishes.'):waiting===1?'One more song is waiting in the queue.':`${waiting} more songs are waiting in the queue.`;
 for(const b of document.querySelectorAll('[data-mode]'))b.setAttribute('aria-selected',b.dataset.mode===mode);
 let help=mode==='cover'?(source?.awaitingTrim?'Choose the part of your recording to analyze.':source&&sourceAnalysis.key===analysisKey()&&sourceAnalysis.status==='ready'?'Ready. Covers reuse the saved recording analysis.':'Upload a recording, choose a section, then analyze it for covers.'):'';
 if(uploading)help='Uploading your recording…';
 else if(!available)help=config.hosted?'Music generation is not connected yet.':config.render_node?.url&&!config.render_node.connected?'The render node is not reachable. Check Settings.':config.render_environment_ready===false?'Connect a render node in Settings → Rendering before creating.':'Download the selected model in Models before creating.';
 $('generate').title=$('generate').disabled?help:active?'Added to the queue and starts when the current generation finishes.':'';
}
function queuePosition(id){const i=(config?.queue||[]).indexOf(id);return i<0?0:i+1}
function ordinal(n){return n+(n%10===1&&n%100!==11?'st':n%10===2&&n%100!==12?'nd':n%10===3&&n%100!==13?'rd':'th')}
function generationModelLabel(model){
 const id=model?.id||model;
 return {bf16:'YuE2 BF16','8bit':'YuE2 8-bit','cuda-bf16':'YuE2 CUDA BF16','cuda-fp8':'YuE2 CUDA FP8'}[id]||model?.label||id||'YuE2';
}
function applyHostedMode(){
 if(!config?.hosted)return;
 document.body.classList.add('hosted');
 $('navModels').hidden=true;$('navModels').setAttribute('aria-hidden','true');
 $('modelSetup').hidden=true;
 const picker=$('model').closest('.model-picker');if(picker)picker.hidden=true;
}
function applyModelChoices(){
 // The model list belongs to whatever renders: this computer, or the connected render node.
 const current=$('model').value;
 $('model').replaceChildren(...config.models.map(m=>new Option(generationModelLabel(m),m.id)));
 $('model').value=config.models.some(m=>m.id===current)?current:config.default_model;
 if(!$('model').value&&config.models.length)$('model').value=config.models[0].id;
}
/* LoRA picker: community adapters found in the loras folder of whatever renders. The list is refreshed each time
   the picker is opened so a file dropped into the folder shows up without reloading the page. Documented downloads
   carry trigger words and recommended settings; choosing one fills Controls. */
function loraName(id){const l=(config?.loras||[]).find(x=>x.id===id);return l?l.name:(id||'').replace(/\.safetensors$/,'').replace(/_/g,' ')}
function loraPlanLabel(cot){return {full:'Melody and Chords',melody:'Melody Only',off:'No Plan'}[cot]||''}
function loraBranchLabel(l){
 if(!l?.branches?.length)return '';
 return l.branches.length===2?'writing + sound':l.branches[0]==='nar'?'sound':'writing';
}
function loraDetail(l,opts){
 if(!l||l.error)return l?.error||'';
 const s=l.settings||{},parts=[],branch=loraBranchLabel(l);
 if(opts?.branch!==false&&branch)parts.push(branch);
 if(opts?.trigger!==false&&l.trigger)parts.push('Trigger '+l.trigger);
 if(s.instrumental)parts.push('Instrumental');
 if(Number.isFinite(s.lora_strength))parts.push('strength '+Number(s.lora_strength).toFixed(2).replace(/0+$/,'').replace(/\.$/,''));
 const plan=loraPlanLabel(s.cot);if(plan)parts.push(plan);
 if(Number.isFinite(s.cfg_scale))parts.push('cfg '+Number(s.cfg_scale).toFixed(1));
 if(opts?.steps!==false&&Number.isFinite(l.steps))parts.push(Number(l.steps).toLocaleString('en-US')+' steps');
 if(l.rank&&!branch)parts.push('rank '+l.rank);
 return parts.join(' · ');
}
function loraTooltip(l){
 if(!l)return '';
 if(l.error)return l.error;
 return [l.description,loraDetail(l),l.notes,l.prompt&&('Example style: '+l.prompt)].filter(Boolean).join('\n\n');
}
function loraKnownTriggers(){
 return [...new Set((config?.loras||[]).map(l=>l.trigger).filter(Boolean).concat(
  (config?.downloads?.lora_packages||[]).map(l=>l.trigger).filter(Boolean)))].sort((a,b)=>b.length-a.length);
}
function stripLoraTriggers(style){
 let next=(style||'').trim();
 for(const trigger of loraKnownTriggers()){
  const escaped=trigger.replace(/[.*+?^${}()|[\]\\]/g,'\\$&');
  next=next.replace(new RegExp('^'+escaped+'\\s*,\\s*','i'),'');
  if(next.toLowerCase()===trigger.toLowerCase())next='';
 }
 return next.trim();
}
function applyLoraStyle(l){
 if(!l)return;
 const rest=stripLoraTriggers($('style').value);
 // A style that is only a trigger word or a card's example prompt (with or without its trigger) counts as empty,
 // so switching cards swaps the whole example instead of keeping the previous card's text behind the new trigger.
 const starters=new Set(loraKnownTriggers().map(t=>t.toLowerCase()));
 (config?.loras||[]).concat(config?.downloads?.lora_packages||[]).forEach(item=>{if(item.prompt){starters.add(item.prompt.toLowerCase());starters.add(stripLoraTriggers(item.prompt).toLowerCase())}});
 const empty=!rest||starters.has(rest.toLowerCase());
 if(l.trigger&&empty&&l.prompt)$('style').value=l.prompt;
 else if(l.trigger)$('style').value=rest?l.trigger+', '+rest:l.trigger;
 else if(empty&&l.prompt)$('style').value=l.prompt;
 else $('style').value=rest;
}
function applyLoraCard(l){
 if(!l||l.error)return;
 applyLoraStyle(l);
 const s=l.settings||{};
 if(typeof s.instrumental==='boolean'){
  $('instrumental').checked=s.instrumental;
  if(s.instrumental&&!$('lyrics').value.trim()){
   $('lyrics').value=DEFAULT_STRUCTURE;
   const expanded=$('expandedLyrics');if(expanded)expanded.value=DEFAULT_STRUCTURE;
  }
 }
 if(Number.isFinite(s.lora_strength))$('loraStrengthSlider').value=s.lora_strength;
 if(mode!=='cover'&&['full','melody','off'].includes(s.cot)){
  // Instrumental songs need a score; No Plan is forced off after updateUI.
  $('cot').value=(s.cot==='off'&&s.instrumental)?'full':s.cot;
 }
 if(Number.isFinite(s.cfg_scale))$('cfg').value=s.cfg_scale;
 if(simpleCurves){
  if(Number.isFinite(s.abc_temperature)){abcCustom=null;$('compositionSlider').value=Math.max(0,Math.min(100,Math.round(uncurve(s.abc_temperature,simpleCurves.abc_temperature))))}
  if(Number.isFinite(s.temperature)){const w=Math.max(0,Math.min(100,Math.round(uncurve(s.temperature,simpleCurves.temperature))));$('weirdnessSlider').value=w;applyWeirdness(w)}
 }
 if(Number.isFinite(s.duration))$('duration').value=s.duration;
}
/* Two slots, folded into the model together. Style LoRA: every file that touches the writing (AR) branch — genre,
   artist and planner adapters, including ones that also carry a sound half; its card fills Controls. Sound LoRA:
   decoder-only (NAR) files, so the two lists never overlap. */
function loraIsSoundOnly(l){const b=l.branches||[];return b.includes('nar')&&!b.includes('ar')}
const LORA_SLOTS={
 lora:{select:'lora',slider:'loraStrengthSlider',row:'loraRow',strengthRow:'loraStrengthRow',value:'loraStrengthValue',hint:'loraHint',reset:'resetLoraStrength',other:'sound',
  fits:l=>!!l.error||!loraIsSoundOnly(l),card:true,
  idle:'A community adapter from the loras folder, folded into the model for this song only. It changes how the model writes; it does not clone a singer.'},
 sound:{select:'soundLora',slider:'soundLoraStrengthSlider',row:'soundLoraRow',strengthRow:'soundLoraStrengthRow',value:'soundLoraStrengthValue',hint:'soundLoraHint',reset:'resetSoundLoraStrength',other:'lora',
  fits:l=>!l.error&&loraIsSoundOnly(l),card:false,
  idle:'A decoder adapter from the loras folder, folded into the model for this song only. It changes how the finished audio sounds, not what is written.'}
};
function loraSlotValue(slot){return $(LORA_SLOTS[slot].select)?.value||''}
function applyLoraChoices(){
 for(const slot of Object.keys(LORA_SLOTS))applySlotChoices(slot);
}
function applySlotChoices(slot){
 const cfg=LORA_SLOTS[slot],select=$(cfg.select);if(!select)return;
 const current=select.value,list=(config.loras||[]).filter(cfg.fits),taken=loraSlotValue(cfg.other);
 const options=[new Option('None','')];
 for(const l of list){
  const o=new Option(l.error?l.name+' (not a YuE2 LoRA)':l.name,l.id);
  o.disabled=!!l.error||(!!taken&&l.id===taken);
  if(l.trigger)o.dataset.trigger=l.trigger;
  const detail=loraDetail(l,{branch:false,trigger:false,steps:false});if(detail)o.dataset.detail=detail;
  o.title=loraTooltip(l)||'';
  options.push(o);
 }
 if(current&&!list.some(l=>l.id===current))options.push(Object.assign(new Option(loraName(current)+loraSlotMismatch(current,slot),current),{className:'missing'}));
 select.replaceChildren(...options);select.value=current;if(select.value!==current)select.value='';
 syncLora(slot);
}
function loraSlotMismatch(name,slot){
 // A file that exists but belongs in the other slot (older requests, or a picker change) is named as such.
 const l=(config?.loras||[]).find(x=>x.id===name);
 if(!l||l.error)return ' (not in the loras folder)';
 return slot==='sound'?' (a style LoRA — pick it in Style LoRA)':' (a sound LoRA — pick it in Sound LoRA)';
}
function setLora(name,strength,slot='lora'){
 const cfg=LORA_SLOTS[slot],select=$(cfg.select);if(!select)return;
 if(name&&!Array.from(select.options).some(o=>o.value===name))select.append(Object.assign(new Option(loraName(name)+loraSlotMismatch(name,slot),name),{className:'missing'}));
 select.value=name||'';
 const s=Number.isFinite(strength)?Math.max(0,Math.min(2,strength)):1;
 $(cfg.slider).value=s;
 syncLora(slot);
}
function setSoundLora(name,strength){setLora(name,strength,'sound')}
function syncLora(slot){
 if(!slot){for(const s of Object.keys(LORA_SLOTS))syncLora(s);return}
 const cfg=LORA_SLOTS[slot],select=$(cfg.select);if(!select)return;
 const chosen=select.value,s=Number($(cfg.slider).value);
 $(cfg.row).hidden=!!config?.hosted;
 $(cfg.strengthRow).hidden=!chosen;
 $(cfg.value).value=Math.round(s*100)+'%';
 paintSliderTicks();
 const l=(config?.loras||[]).find(x=>x.id===chosen);
 const missing=chosen&&!(l&&!l.error);
 select.classList.toggle('missing',!!missing);
 select.title=missing?'This file is not in the loras folder'+(config?.render_node?.connected?' of the render node.':'.')+' Drop it there, or choose another.':(l?loraTooltip(l):cfg.idle);
 const hint=$(cfg.hint);
 if(hint){
  hint.hidden=!chosen||missing||!l;
  hint.textContent=chosen&&!missing&&l?loraDetail(l,{branch:false,trigger:false,steps:false}):'';
 }
}
let lorasRefreshing=false;
async function refreshLoras(){
 if(lorasRefreshing||config?.hosted)return;
 lorasRefreshing=true;
 try{const r=await api('/api/loras');config.loras=r.loras||[];applyLoraChoices()}catch{}finally{lorasRefreshing=false}
}
function setupLoras(){
 applyLoraChoices();
 for(const [slot,cfg] of Object.entries(LORA_SLOTS)){
  const select=$(cfg.select);if(!select)continue;
  select.addEventListener('pointerdown',()=>refreshLoras());
  select.addEventListener('focus',()=>refreshLoras());
  select.onchange=()=>{
   const chosen=select.value,l=(config?.loras||[]).find(x=>x.id===chosen);
   if(l&&cfg.card)applyLoraCard(l);
   else if(l&&Number.isFinite(l.settings?.lora_strength))$(cfg.slider).value=l.settings.lora_strength;
   // The same file cannot sit in both slots: the other picker greys it out.
   applySlotChoices(cfg.other);
   updateUI();saveDraft();
  };
  $(cfg.slider).oninput=()=>{syncLora(slot);saveDraft()};
  $(cfg.reset).onclick=()=>{$(cfg.slider).value=1;syncLora(slot);saveDraft()};
 }
}
function renderNodeName(node){
 node=node||config?.render_node;
 if(!node?.connected)return '';
 return (node.gpu?.name||node.platform?.hostname||'render node')+(node.platform?.hostname&&node.gpu?.name?' · '+node.platform.hostname:'');
}
function setupRenderNode(){
 const url=$('settingsRenderUrl'),token=$('settingsRenderToken'),connect=$('settingsRenderConnect'),disconnect=$('settingsRenderDisconnect'),status=$('settingsRenderStatus');
 if(!url||!connect)return;
 let busy=false,edited=false;
 function message(text,kind=''){status.textContent=text;status.classList.toggle('settings-error',kind==='error');status.classList.toggle('settings-ok',kind==='ok')}
 function show(node){
  node=node||config?.render_node;if(!node)return;
  if(!edited){url.value=node.url||'';token.value=node.token||''}
  url.disabled=token.disabled=!!node.locked||busy;connect.disabled=busy||!!node.locked;
  disconnect.hidden=!node.url||node.locked;disconnect.disabled=busy;
  const interfaceOnly=config?.render_environment_ready===false;
  disconnect.textContent=interfaceOnly?'Disconnect':'Render on this computer';
  if(interfaceOnly)$('settingsRenderIntro').textContent='This computer runs the interface; songs render on a render node. The node is a Mac or PC with the OpenSuno Render Node app: copy its address and token from its menu bar icon.';
  if(node.locked)message('Set by OPENSUNO_RENDER_URL in the environment of this Studio.'+(node.connected?' Connected to '+renderNodeName(node)+'.':node.error?' '+node.error:''),node.connected?'ok':'error');
  else if(!node.url)message(config?.render_environment_ready===false?'This computer runs the interface only. Enter a render node address and token to create songs.':'Rendering on this computer'+(config?.gpu?.name?' · '+config.gpu.name:'')+'.');
  else if(node.connected)message('Connected to '+renderNodeName(node)+(node.models?.length?' · models ready: '+node.models.map(id=>generationModelLabel(id)).join(', '):' · no models installed there yet; download them in Models')+(node.busy?' · rendering now':'')+(node.ffmpeg_ready===false?' · ffmpeg is missing there, so songs cannot be streamed as MP3; run the installer on that machine':''),'ok');
  else message(node.error||'The render node is not reachable.','error');
 }
 async function save(nextUrl,nextToken){
  busy=true;show();
  try{
   const node=await api('/api/render-node',{method:'POST',body:JSON.stringify({url:nextUrl,token:nextToken})});
   edited=false;config=await api('/api/config');config.render_node=node;applyModelChoices();applyLoraChoices();updateUI();show(node);
   notice(node.url?'Songs now render on '+renderNodeName(node)+'.':'Songs now render on this computer.');
  }catch(e){message(e.message,'error')}
  finally{busy=false;url.disabled=token.disabled=connect.disabled=false;disconnect.disabled=false}
 }
 url.oninput=token.oninput=()=>{edited=true};
 connect.onclick=()=>save(url.value.trim(),token.value.trim());
 disconnect.onclick=()=>save('','');
 show();
 renderNodeRefresh=()=>{if(!busy&&!edited)show()};
}
function showModelSetup(){
 if(config?.hosted){$('modelSetup').hidden=true;return}
 $('modelSetup').hidden=!modelPanelOpen;
 const d=config?.downloads;if(!d)return;
 const busy=d.status==='downloading',envMissing=!config.transcriber_environment_ready;
 const nodeName=renderNodeName();
 $('modelSetupTitle').textContent=nodeName?'Models on '+nodeName:'Models';
 // An interface-only install (the Mac disk image) has no renderer of its own: nothing to download here.
 const interfaceOnly=!nodeName&&config.render_environment_ready===false;
 $('modelInterfaceOnly').hidden=!interfaceOnly;
 $('modelPackages').hidden=interfaceOnly;
 const loraSection=$('loraSetup');
 if(interfaceOnly){$('modelTransfer').hidden=true;$('modelEnvironmentHelp').hidden=true;$('modelDownloadAll').hidden=true;const loraAll=$('loraDownloadAll');if(loraAll)loraAll.hidden=true;if(loraSection)loraSection.hidden=true;return}
 const packages=[...config.models.map(m=>[m.id,generationModelLabel(m),m.description]),['analysis','Cover analysis','Transcribes uploaded recordings into a melody and chord score for covers (SheetSage2 and MERT)'],['tokens','Audio input','Turns an uploaded recording into music tokens so the model can continue it, plus the decoder adapter that renders such tokens (needs Cover analysis)']];
 const container=$('modelPackages');
 async function startDownload(id){try{config.downloads=await api('/api/models/download?model='+id,{method:'POST'});updateUI()}catch(e){notice(e.message,true)}}
 function addPackageRow(host,id,title,description){
  const row=document.createElement('div');row.className='model-package';
  const info=document.createElement('div');info.className='model-package-info';
  const heading=document.createElement('div');heading.className='setting-label';
  const name=document.createElement('strong');name.textContent=title;heading.append(name);
  infoTip(heading,title,description,'help_package_'+id);
  const size=document.createElement('small');size.id='packageSize-'+id;
  info.append(heading,size);
  const button=document.createElement('button');button.type='button';button.className='secondary';button.id='packageDownload-'+id;button.textContent='Download';
  button.onclick=()=>startDownload(id);
  row.append(info,button);host.append(row);
 }
 function loraHelp(l){
  return [l.description,loraDetail(l),l.notes,l.prompt&&('Example style: '+l.prompt)].filter(Boolean).join('\n\n');
 }
 function loraSourceHref(repo){
  const r=(repo||'').trim();
  if(!r)return '';
  if(/^https?:\/\//i.test(r))return r;
  if(/^(github|gitlab)\.com\//i.test(r))return 'https://'+r;
  return 'https://huggingface.co/'+r;
 }
 function loraSourceLabel(href,many){
  let host='',path='';
  try{const u=new URL(href);host=u.hostname.replace(/^www\./,'');path=u.pathname.replace(/^\/+|\/$/g,'')}catch{return 'Open'}
  const org=path.split('/')[0]||path;
  if(host==='huggingface.co'||host.endsWith('.huggingface.co'))return many?org||'Hugging Face':'Hugging Face';
  if(host==='github.com')return many?org||'GitHub':'GitHub';
  if(host==='gitlab.com')return many?org||'GitLab':'GitLab';
  return many?org||host:'Open';
 }
 function addLoraRow(host,l){
  document.getElementById('help_package_'+l.id)?.remove();
  const row=document.createElement('article');row.className='model-package lora-package';
  const info=document.createElement('div');info.className='model-package-info';
  const heading=document.createElement('div');heading.className='setting-label';
  const name=document.createElement('strong');name.textContent=l.name;heading.append(name);
  infoTip(heading,l.name,loraHelp(l),'help_package_'+l.id);
  const about=document.createElement('p');about.className='lora-about';about.textContent=l.description||'';
  info.append(heading,about);
  // Recommended recipe from the card (trigger word, strength, plan, cfg, training steps), so the full page shows it without the tip.
  const meta=loraDetail(l);if(meta){const line=document.createElement('p');line.className='lora-meta';line.textContent=meta;info.append(line)}
  const button=document.createElement('button');button.type='button';button.className='secondary';button.id='packageDownload-'+l.id;button.textContent='Download';
  button.onclick=()=>startDownload(l.id);
  row.append(info,button);host.append(row);
 }
 const LORA_SLOT_GROUPS=[
  ['sound','Sound LoRAs','Change how the decoder renders the audio, not the notes or words. Picked as the Sound LoRA and stacked with a style adapter.'],
  ['style','Style LoRAs','Change what the model writes: genre, artist voice and instruments, or the planner. Picked as the Style LoRA under Controls.']
 ];
 if(!container.children.length){
  for(const [id,title,description] of packages)addPackageRow(container,id,title,description);
 }
 const loraBox=$('loraPackages'),loraList=d.lora_packages||[];
 if(loraSection){
  loraSection.hidden=!loraList.length;
  const ids=loraList.map(l=>l.id).join()+'|card|src|notag';
  if(loraBox&&loraBox.dataset.ids!==ids){
   loraBox.replaceChildren();
   for(const [slot,title,about] of LORA_SLOT_GROUPS){
    const members=loraList.filter(l=>(l.slot||'style')===slot);
    if(!members.length)continue;
    const group=document.createElement('section');group.className='lora-slot-group lora-slot-group-'+slot;
    const head=document.createElement('header');head.className='lora-slot-group-head';
    const name=document.createElement('h3');name.textContent=title;
    const text=document.createElement('p');text.className='muted';text.textContent=about;
    head.append(name,text);group.append(head);
    const groups=new Map();
    for(const l of members){const family=l.family||'Community';if(!groups.has(family))groups.set(family,[]);groups.get(family).push(l)}
    for(const [family,items] of groups){
     const block=document.createElement('section');block.className='lora-family';
     const headRow=document.createElement('div');headRow.className='lora-family-head';
     const heading=document.createElement('h4');heading.textContent=family+' · '+items.length;headRow.append(heading);
     const hrefs=[...new Set(items.map(l=>loraSourceHref(l.repo)).filter(Boolean))];
     for(const href of hrefs){
      const link=document.createElement('a');link.className='secondary lora-source';link.href=href;link.target='_blank';link.rel='noopener noreferrer';link.title=href;
      const icon=document.createElement('img');icon.src='/static/icons/arrow-top-right-on-square.svg';icon.alt='';
      link.append(icon,document.createTextNode(loraSourceLabel(href,hrefs.length>1)));
      headRow.append(link);
     }
     block.append(headRow);
     for(const l of items)addLoraRow(block,l);
     group.append(block);
    }
    loraBox.append(group);
   }
   loraBox.dataset.ids=ids;
  }
 }
 const all=$('modelDownloadAll'),allPkg=d.packages?.missing,allCurrent=allPkg?.status==='downloading';
 const remaining=packages.some(([id])=>!d.packages?.[id]?.ready);
 all.hidden=!remaining&&!allCurrent;
 all.onclick=()=>startDownload('missing');
 const allLabel=allCurrent?'Downloading…':allPkg?.status==='failed'?'Retry all missing':'Install all missing';
 const allAmount=allPkg?.missing_bytes||(!allCurrent&&remaining?packages.reduce((n,[id])=>n+(d.packages?.[id]?.ready?0:d.packages?.[id]?.missing_bytes||0),0):0);
 all.textContent=allAmount&&!allCurrent?allLabel+' '+(allAmount/1e9).toFixed(2)+' GB':allLabel;
 all.disabled=!remaining||allCurrent||!!active;
 all.title='Download every package that is not installed yet.';
 const loraAll=$('loraDownloadAll');
 if(loraAll){
  const pending=loraList.filter(l=>!d.packages?.[l.id]?.ready);
  const loraBusy=pending.filter(l=>d.packages?.[l.id]?.status==='downloading');
  const loraFailed=pending.some(l=>d.packages?.[l.id]?.status==='failed');
  const loraAllBusy=pending.length&&loraBusy.length===pending.length;
  loraAll.hidden=!pending.length&&!loraBusy.length;
  loraAll.onclick=()=>startDownload('loras');
  const loraLabel=loraAllBusy?'Downloading…':loraFailed?'Retry all':'Download all';
  const loraBytes=d.packages?.loras?.missing_bytes||pending.reduce((n,l)=>n+(d.packages?.[l.id]?.missing_bytes||d.packages?.[l.id]?.bytes||0),0);
  loraAll.textContent=loraBytes&&!loraAllBusy?loraLabel+' '+(loraBytes/1e9).toFixed(2)+' GB':loraLabel;
  loraAll.disabled=!pending.length||loraAllBusy||!!active;
  loraAll.title='Download every LoRA that is not installed yet.';
 }
 for(const id of [...packages.map(([id])=>id),...loraList.map(l=>l.id)]){
  const pkg=d.packages?.[id],current=pkg?.status==='downloading',failed=pkg?.status==='failed',button=$('packageDownload-'+id);
  if(!button)continue;
  const label=pkg?.ready?'Installed':current?'Downloading'+(pkg.file_total?' '+Math.min(100,Math.round(100*(pkg.file_bytes||0)/pkg.file_total))+'%':'…'):failed?'Retry':'Download';
  const amount=pkg?.ready?pkg.bytes:(pkg?.missing_bytes??pkg?.bytes);
  button.textContent=current||!amount?label:label+' '+(amount/1e9).toFixed(2)+' GB';
  button.disabled=!!pkg?.ready||current||!!active;
  const size=$('packageSize-'+id);
  if(size){size.hidden=!!pkg?.ready;size.textContent=pkg?.ready?'':'Not Installed'}
 }
 const envHelp=$('modelEnvironmentHelp');
 if(envHelp.dataset.remote!==String(!!nodeName)){
  envHelp.dataset.remote=String(!!nodeName);
  if(nodeName)envHelp.textContent='The render node needs its cover-analysis environment: run Install OpenSuno on that machine.';
  else envHelp.replaceChildren('Run ',Object.assign(document.createElement('b'),{textContent:'Install OpenSuno.command'}),' to install the cover-analysis environment.');
 }
 $('modelTransfer').hidden=!busy&&d.status!=='failed';
 const names={'model/bf16/model.safetensors':'BF16 weights','model/8bit/vae.safetensors':'Shared audio decoder','model/8bit/qwen.tiktoken':'Shared tokenizer','tokens/tokenizer_head_joint_v9.safetensors':'Music tokenizer'};
 $('modelTransferName').textContent=d.status==='failed'?'Download interrupted':d.model==='multiple'?'Downloading '+(d.file_count||0)+' packages':(d.message?.startsWith('Verifying')?'Verifying ':'Downloading ')+(names[d.file]||(d.file?.startsWith('loras/')?d.file.slice(6).replace(/\.safetensors$/i,''):'components'));
 $('modelTransferPercent').textContent=busy&&d.file_total?Math.min(100,100*d.file_bytes/d.file_total).toFixed(0)+'%':'';
 $('modelSetupStatus').textContent=d.status==='failed'?d.message:busy&&d.file_total?(d.file_bytes/1e9).toFixed(2)+' / '+(d.file_total/1e9).toFixed(2)+' GB'+(d.bytes_per_second?' · '+(d.bytes_per_second/1e6).toFixed(1)+' MB/s':'')+(d.model==='multiple'?' · '+d.file_count+' packages':d.file_count>1?' · file '+d.file_index+' of '+d.file_count:''):'Preparing download…';
 $('modelEnvironmentHelp').hidden=!envMissing;
 $('modelDownloadProgress').value=d.file_total?d.file_bytes/d.file_total:0;
}
function setMode(next){if(typeof compositionDraft!=='undefined')compositionDraft=null;mode=next==='cover'?'cover':'create';updateUI();saveDraft();prepareModels(true)}
function saveDraft(){if(!config||restoring)return;try{localStorage.setItem('yue2-studio-draft',JSON.stringify({mode,request:request(),source,compositionDraft:typeof compositionDraft!=='undefined'?compositionDraft:null}))}catch{}}
function restore(r){
 if(typeof compositionDraft!=='undefined')compositionDraft=r.edit_id?{id:r.edit_id}:null;
 restoring=true;$('lockSeed').checked=r.random_seed===false;$('instrumental').checked=r.instrumental??(/^\s*\[Instrumental\]\s*$/i.test(r.lyrics||''));$('hookMelody').checked=!!r.hook_melody;$('model').value=config.models.some(m=>m.id===r.model)?r.model:(config.default_model||'bf16');$('candidates').value=r.candidates||1;
 for(const key of ['title','style','lyrics','seed','steps'])$(key).value=r[key]??'';
 setSampling(r.semantic_sampling||config.defaults.semantic);
 $('cfg').value=r.cfg_scale??1.2;
 $('cot').value=r.kind!=='cover'&&['full','melody','off'].includes(r.cot)?r.cot:'full';
 $('voice').value=['male','female','duet'].includes(r.voice)?r.voice:'any';
 $('preserve').value=r.melody_only===false?'full':'melody';$('sourceSeconds').value=r.source_seconds??0;
 $('arrangeCover').checked=r.arrange!==false;
 $('vocalOctave').value=['auto','keep','down','down2','up'].includes(r.vocal_octave)?r.vocal_octave:'auto';
 setComposition(r.abc_sampling);
 setLora(r.lora||'',r.lora_strength);
 setSoundLora(r.sound_lora||'',r.sound_lora_strength);
 restoring=false;updateUI();
}
function restoreDefaults(){$('lockSeed').checked=false;$('model').value=config.default_model||'bf16';$('candidates').value='1';$('cot').value='full';$('voice').value='any';setSampling(config.defaults.semantic);setComposition(null);setLora('',1);setSoundLora('',1);$('steps').value=config.defaults.ode_steps;$('cfg').value=1.2;updateUI();saveDraft()}
let sourceAnalysis={key:'',status:'missing'},analysisBusy=false,analysisRefreshTimer=null;
let analysisLiveJob=null,pendingAudioUpload=null,analysisRetryRequested=false;
function openAnalysisDialog(){
 if(source?.awaitingTrim){openTrimEditor();return}
 if(!source&&!pendingAudioUpload)return;
 showAnalysis();
 if(!$('analysisDialog').open){$('analysisDialog').showModal();$('analysisHeading').focus()}
}
function renderAnalysisDialog(s){
 if(s==='queued')s='waiting';
 if(!source&&!pendingAudioUpload){$('analysisDialog').close();return}
 const importing=!!pendingAudioUpload;
 const analysed=!importing&&s==='ready',failed=!importing&&['failed','cancelled','interrupted'].includes(s),live=analysisLiveJob?.id===sourceAnalysis.job?analysisLiveJob:null,p=live?.progress||{};
 const ready=analysed;
 $('analysisFilename').textContent=(pendingAudioUpload||source).name;$('analysisFilename').title=(pendingAudioUpload||source).name;
 showAnalysisArt();
 $('analysisSummary').textContent=importing?'Uploading recording…':ready?'Ready for covers':failed?'Analysis '+(s==='failed'?'failed':s==='cancelled'?'stopped':'interrupted'):s==='waiting'?'Analysis queued':s==='cancelling'?'Stopping analysis…':'Analyzing recording…';
 $('analysisHeading').textContent=ready?'Your audio is ready':failed?'Analysis needs attention':'Preparing your audio';
 const stages={analysis_audio:'Reading your recording…',analysis_encoding:'Finding the melody and harmony…',analysis_decoding:'Transcribing the performance…',analysis_notation:'Finishing recording analysis…'};
 let detail=importing?'Saving and checking your recording…':ready?'':failed?(sourceAnalysis.error||live?.error||'Your recording is still available. Retry analysis to prepare it for covers.'):s==='waiting'?'Waiting for the current job to finish. You can go back while analysis is queued.':s==='cancelling'?'Stopping the current analysis…':p.loading_detail?p.loading_detail+'…':stages[p.stage]||'Starting the audio analysis runtime…';
 if(!analysed&&!failed&&p.windows>1)detail+=' Segment '+p.window+' of '+p.windows+'.';
 $('analysisDetail').textContent=detail;$('analysisDetail').hidden=!detail;
 const bar=$('analysisProgress');if(ready)bar.value=1;else if(failed)bar.value=0;else bar.removeAttribute('value');
 bar.setAttribute('aria-valuetext',$('analysisSummary').textContent);
 $('analysisTokens').textContent=p.tokens?Number(p.tokens).toLocaleString():'—';$('analysisSpeed').textContent=p.tokens_per_second?Number(p.tokens_per_second).toFixed(1)+' tok/s':'—';
 const elapsed=analysed||failed?(live?.result?.elapsed??p.elapsed??(live?.started?live.updated-live.started:null)):(live?.started?Date.now()/1000-live.started:p.elapsed);
 $('analysisElapsed').textContent=Number.isFinite(elapsed)?Math.max(0,Math.floor(elapsed))+'s':'—';
 $('analysisCancel').hidden=analysed;$('analysisCancel').textContent=failed?'Retry analysis':'Cancel';$('analysisCancel').disabled=importing||s==='cancelling'||(analysisBusy&&!sourceAnalysis.job);
 $('analysisContinue').disabled=!ready;
}
async function cancelRecordingAnalysis(){
 const s=sourceAnalysis.status;
 if(['failed','cancelled','interrupted'].includes(s)){checkSourceAnalysis(true);return}
 if(analysisBusy&&!sourceAnalysis.job)return;
 const key=analysisKey(),job=sourceAnalysis.job;
 if(!job){sourceAnalysis={key,status:'cancelled',requestFailed:true};showAnalysis();updateUI();return}
 try{await api('/api/jobs/'+job+'/cancel',{method:'POST'});if(key===analysisKey()){sourceAnalysis={...sourceAnalysis,status:'cancelling'};showAnalysis();poll()}}
 catch(e){notice(e.message,true);checkSourceAnalysis()}
}

function analysisOptions(){return {audio_id:source?.id||'',melody_only:$('preserve').value==='melody',source_seconds:Number($('sourceSeconds').value)}}
function analysisKey(){return JSON.stringify(analysisOptions())}
const ANALYSIS_NOTE='/static/icons/musical-note.svg';
function showAnalysisArt(){
 const box=$('analysisArt'),img=$('analysisArtImage');
 if(!box||!img)return;
 const rec=pendingAudioUpload?null:source,url=rec?.art||(rec?.id&&rec.cover!==false?'/api/uploads/'+rec.id+'/cover':'');
 const showNote=()=>{box.classList.remove('has-cover');img.onload=img.onerror=null;delete img.dataset.cover;if(!img.src.endsWith('musical-note.svg'))img.src=ANALYSIS_NOTE};
 if(!url){showNote();return}
 if(img.dataset.cover===url)return;
 img.dataset.cover=url;
 img.onload=()=>{if(img.dataset.cover===url)box.classList.add('has-cover')};
 img.onerror=()=>{if(img.dataset.cover===url)showNote()};
 img.src=url;
}
function showAnalysis(){
 const s=sourceAnalysis.key===analysisKey()?sourceAnalysis.status:'missing';
 const labels={ready:'Ready for covers · saved analysis will be reused',starting:'Starting recording analysis…',running:'Analyzing recording…',waiting:'Analysis queued until the current job finishes…',queued:'Analysis queued until the current job finishes…',failed:'Analysis failed. Retry to prepare this recording.',cancelled:'Analysis stopped.',cancelling:'Stopping analysis…',interrupted:'Analysis was interrupted.',missing:'Preparing recording analysis…'};
 $('sourceAnalysisStatus').textContent=source?.awaitingTrim?'Choose your selection before starting analysis.':source?labels[s]||labels.missing:'';
 $('sourceAnalysisStatus').hidden=!source||s==='ready';
 $('retryAnalysis').hidden=!source||source.awaitingTrim||!['failed','cancelled','interrupted'].includes(s);
 renderAnalysisDialog(s);
}
async function checkSourceAnalysis(startNow=false){
 if(!source||!config||uploading||source.awaitingTrim||$('trimDialog').open)return;
 if(analysisBusy){if(startNow)analysisRetryRequested=true;return}
 const key=analysisKey(),options=analysisOptions();
 if(!$('sourceSeconds').checkValidity()||!Number.isFinite(options.source_seconds))return;
 if(!startNow&&sourceAnalysis.key===key&&sourceAnalysis.requestFailed)return;
 if(startNow){notice('');sourceAnalysis={key,status:'starting'};analysisLiveJob=null;showAnalysis()}
 analysisBusy=true;
 try{
  const state=await api('/api/analysis?'+new URLSearchParams(options));
  if(key!==analysisKey()||source.awaitingTrim||$('trimDialog').open)return;
  const waiting=sourceAnalysis.key===key&&sourceAnalysis.status==='waiting';
  sourceAnalysis={key,...state,job:state.job||(sourceAnalysis.key===key?sourceAnalysis.job:undefined)};
  if((startNow&&state.status!=='ready')||(state.status==='missing')||(waiting&&!active)){
   const launched=await api('/api/analysis',{method:'POST',body:JSON.stringify(options)});
   if(key!==analysisKey())return;
   sourceAnalysis={key,...launched};
  }
  if(['starting','running','cancelling'].includes(sourceAnalysis.status)&&sourceAnalysis.job){
   if(active!==sourceAnalysis.job||selected!==sourceAnalysis.job){active=selected=sourceAnalysis.job;lastStatus='';poll()}
  }
 }catch(e){if(key===analysisKey()){sourceAnalysis={key,status:'failed',requestFailed:true,error:e.message};notice('Could not analyze recording: '+e.message,true)}}
 finally{analysisBusy=false;showAnalysis();updateUI();if(analysisRetryRequested){analysisRetryRequested=false;checkSourceAnalysis(true)}}
}
function changedAnalysisSettings(){
 sourceAnalysis={key:analysisKey(),status:'missing'};showAnalysis();updateUI();saveDraft();
 clearTimeout(analysisRefreshTimer);analysisRefreshTimer=setTimeout(()=>checkSourceAnalysis(true),350);
}

function recordingTitle(item){
 const name=item?.name||'';
 if(!name||name==='Saved recording')return '';
 return name.replace(/\.[^.]+$/,'').trim();
}
function showSource(){
 $('sourceUpload').hidden=!!source;
 $('removeSource').hidden=!source;
 const title=recordingTitle(source);
 $('sourceHeading').textContent=title||'Audio';
 $('sourceHeading').title=source?.name&&source.name!=='Saved recording'?source.name:'';
 showAnalysis();
}
// Like Suno, an uploaded recording is kept in the library too, badged “Uploaded” instead of a model name.
async function listUploadedSong(id,name){
 if(config?.hosted)return;
 try{await api('/api/uploads/'+id+'/song',{method:'POST',body:JSON.stringify({title:name,workspace_id:saveWorkspace})});await refreshLibrary()}
 catch(e){notice(e.message,true)}
}
function isUpload(j){return j?.kind==='upload'}
function modelBadge(j,c){return isUpload(j)?'UPLOADED':(c?.model||j.request.model).toUpperCase()}
function setupRecordingDrop(){
 const input=$('sourceAudio'),zone=input.closest('.upload');let depth=0;
 const reset=()=>{depth=0;zone.classList.remove('drag-over')};
 const hasFiles=e=>Array.from(e.dataTransfer?.types||[]).includes('Files');
 zone.addEventListener('dragenter',e=>{if(!hasFiles(e))return;e.preventDefault();depth++;if(!uploading)zone.classList.add('drag-over')});
 zone.addEventListener('dragover',e=>{if(!hasFiles(e))return;e.preventDefault();e.dataTransfer.dropEffect=uploading?'none':'copy'});
 zone.addEventListener('dragleave',()=>{if(--depth<=0)reset()});
 zone.addEventListener('drop',e=>{
  e.preventDefault();reset();if(uploading)return;
  const files=Array.from(e.dataTransfer?.files||[]);
  if(files.length!==1){notice('Drop one recording at a time.',true);return}
  if(!/\.(wav|mp3|flac|m4a|aiff|aif|ogg|aac)$/i.test(files[0].name)){notice('Choose a WAV, MP3, FLAC, M4A, AIFF, OGG or AAC recording.',true);return}
  uploadFile(input,files[0]);
 });
 window.addEventListener('dragend',reset);
 window.addEventListener('blur',reset);
}
async function uploadFile(input,droppedFile){
 const file=droppedFile||input.files[0];if(!file||uploading)return;input.value='';notice('');uploading=true;
 showTrimUpload(file.name);const revision=trimRevision;let openEditor=false;updateUI();
 try{const data=new FormData;data.append('file',file);const result=await api('/api/upload',{method:'POST',body:data});
  if(revision!==trimRevision)return;
  source={...result,awaitingTrim:true};sourceAnalysis={key:analysisKey(),status:'missing'};analysisLiveJob=null;analysisRetryRequested=false;
  $('title').value=(result.title||file.name.replace(/\.[^.]+$/,'').trim()||'Untitled song').slice(0,100);
  // Lyrics stored inside the file fill the box on first load, even if leftover section tags are sitting there.
  if(applyRecordingLyrics(result.lyrics,true))notice('Title'+(result.artist?', artist':'')+' and lyrics came from the file\u2019s own tags.');
  $('sourceSeconds').value=0;openEditor=true;
  listUploadedSong(result.id,result.title?result.title+(result.artist?' \u2014 '+result.artist:''):file.name);
 showSource();saveDraft();
 }catch(e){notice(e.message,true);if(revision===trimRevision)trimError(e.message)}
 finally{uploading=false;updateUI();if(openEditor)openTrimEditor();else showSource()}
}
function validate(kind){
 if(['generate','cover'].includes(kind)){
  if(!$('instrumental').checked&&!$('lyrics').value.trim())throw Error('Add your lyrics first.');if(!$('style').value.trim())throw Error('Describe the style you want.');
  if(kind==='cover'&&!source)throw Error('Upload a recording first.');
  if(kind==='cover'&&source?.awaitingTrim)throw Error('Choose a section and analyze it before creating a cover.');
 }
 if(['generate','cover'].includes(kind))for(const el of document.querySelectorAll('#composer input:not([type=file]),#controls input,#controls select'))if(!el.checkValidity()){el.reportValidity();throw Error('Check '+(el.closest('label')?.childNodes[0]?.textContent||'the options')+'.')}
}
const artworkStarted=new Set();
function requestArtwork(j){
 if(!j?.id||!['generate','cover'].includes(j.kind||j.request?.kind))return;
 if(j.artwork?.url||j.artwork?.status==='complete'||j.artwork?.status==='running')return;
 if(artworkStarted.has(j.id))return;
 const settings=typeof StyleAISettings==='undefined'?{}:StyleAISettings.get();
 if(!settings.apiKey&&!config?.openai_configured)return;
 artworkStarted.add(j.id);
 const lyrics=(j.request?.lyrics||'').slice(0,4000);
 api('/api/jobs/'+j.id+'/artwork',{method:'POST',body:JSON.stringify({api_key:settings.apiKey||'',title:j.title||j.request?.title||'',style:j.request?.style||'',lyrics,instrumental:!!j.request?.instrumental})})
  .then(()=>{refreshLibrary();poll()})
  .catch(error=>{artworkStarted.delete(j.id);notice('Album art failed: '+error.message,true)});
}
// A new job either starts now or waits behind the running one; the progress panel follows whichever the user just submitted.
function accepted(data){
 selected=data.id;
 if(data.status==='queued'){config.queue=[...(config.queue||[]),data.id];notice('Added to the queue · position '+data.position+'. It starts when the current generation finishes.')}
 else active=data.id;
 askRenderAlerts();
}
let renderAudioCtx=null;
const announcedRenders=new Set();
function askRenderAlerts(){
 try{
  const Ctx=window.AudioContext||window.webkitAudioContext;
  if(Ctx){renderAudioCtx=renderAudioCtx||new Ctx();if(renderAudioCtx.state==='suspended')renderAudioCtx.resume()}
  if(typeof Notification!=='undefined'&&Notification.permission==='default')Notification.requestPermission();
 }catch{}
}
function renderChime(ok){
 try{
  const Ctx=window.AudioContext||window.webkitAudioContext;if(!Ctx)return;
  renderAudioCtx=renderAudioCtx||new Ctx();
  if(renderAudioCtx.state==='suspended')renderAudioCtx.resume();
  const now=renderAudioCtx.currentTime,ctx=renderAudioCtx;
  const ding=(freq,when,dur,gain)=>{
   const o=ctx.createOscillator(),g=ctx.createGain();
   o.type='sine';o.frequency.setValueAtTime(freq,when);
   g.gain.setValueAtTime(0,when);
   g.gain.linearRampToValueAtTime(gain,when+.018);
   g.gain.exponentialRampToValueAtTime(.0001,when+dur);
   o.connect(g).connect(ctx.destination);o.start(when);o.stop(when+dur+.02);
  };
  if(ok){ding(880,now,.2,.14);ding(1320,now+.11,.32,.12)}
  else {ding(420,now,.22,.12);ding(320,now+.14,.34,.1)}
 }catch{}
}
function announceRender(j){
 if(!j||announcedRenders.has(j.id))return;
 if(!['generate','cover'].includes(j.kind))return;
 if(!['complete','failed'].includes(j.status))return;
 announcedRenders.add(j.id);
 const ok=j.status==='complete',name=(j.title||'Untitled Song').trim()||'Untitled Song';
 const away=document.hidden||(typeof document.hasFocus==='function'&&!document.hasFocus());
 let posted=false;
 try{
  if(away&&typeof Notification!=='undefined'&&Notification.permission==='granted'){
   const note=new Notification(ok?name+' is ready':name+' failed',{body:ok?'OpenSuno finished rendering.':'Rendering did not finish.',tag:'opensuno-'+j.id,silent:false});
   note.onclick=()=>{window.focus();try{navigate('library');const song=jobsCache.find(x=>x.id===j.id);if(song)inspectSong(song,versions(song)[0])}catch{}};
   posted=true;
  }
 }catch{}
 if(!posted)renderChime(ok);
}
async function start(kind){try{
 notice('');validate(kind);askRenderAlerts();const r=request(kind);
 const data=await api('/api/jobs',{method:'POST',body:JSON.stringify(r)});accepted(data);if(['generate','cover'].includes(kind)){$('seed').value=data.seed;saveDraft();requestArtwork({id:data.id,title:r.title,kind,request:r})}lastStatus='';resultFingerprint='';
 updateUI();await refreshLibrary();poll();
}catch(e){notice(e.message,true)}}
/* Generation progress weighted by how long each stage actually takes, so the bar moves at a steady pace.
   Costs are in "music-token equivalents": one autoregressive token is the unit; one NAR step over the whole
   song costs about 2.7% of the music-token stage (measured: 0.85 s per step for 3,189 frames against ~105 tokens/s),
   and the VAE decode about 6%. With the default 2 audio steps the music tokens are ~90% of the work; at 32 steps
   they are about half, which is why the old fixed 55–91% band for audio steps sat wrong either way. */
const STAGE_ORDER=['loading','planning','arranging','music_tokens','synthesis','decoding_audio','complete'];
const NAR_STEP_COST=0.00027,VAE_COST=0.0006;
function stagedProgress(j){
 const p=j.progress,r=j.request||{};if(!p||!STAGE_ORDER.includes(p.stage))return null;
 const rate=p.tokens_per_second>0&&['planning','arranging','music_tokens'].includes(p.stage)?p.tokens_per_second:(stagedProgress.rate||100);
 if(p.stage==='music_tokens'&&p.tokens_per_second>0)stagedProgress.rate=p.tokens_per_second;
 const S=Math.max(1,r.semantic_sampling?.max_tokens||9000),abcMax=r.abc_sampling?.max_tokens||4096;
 const P=Math.min(abcMax,Math.max(200,Math.round(S*.3)));
 const frames=Math.max(1,['synthesis','decoding_audio','complete'].includes(p.stage)&&p.tokens>0?p.tokens:S);
 const steps=Math.max(1,p.step_total||r.steps||2);
 const stages=['loading'];
 if(!r.render_source){
  if(r.kind==='generate'&&r.cot!=='off'&&!r.abc&&!r.edit_id)stages.push('planning');
  if(r.kind==='cover'&&r.arrange!==false)stages.push('arranging');
  stages.push('music_tokens');
 }
 stages.push('synthesis','decoding_audio');
 // A stage the plan did not predict (e.g. an instrumental cover planning first) is slotted in where it belongs.
 if(p.stage!=='complete'&&!stages.includes(p.stage)){stages.push(p.stage);stages.sort((a,b)=>STAGE_ORDER.indexOf(a)-STAGE_ORDER.indexOf(b))}
 const weight={loading:150,planning:P,arranging:P,music_tokens:S,synthesis:NAR_STEP_COST*rate*steps*frames,decoding_audio:VAE_COST*rate*frames};
 const total=stages.reduce((sum,s)=>sum+weight[s],0);
 if(p.stage==='complete')return 100;
 const index=stages.indexOf(p.stage);let done=0;
 for(let i=0;i<index;i++)done+=weight[stages[i]];
 const within=p.stage==='planning'?Math.min((p.tokens||0)/P,.97):p.stage==='arranging'?(p.token_limit>0?Math.min((p.tokens||0)/p.token_limit,1):0):p.stage==='music_tokens'?Math.min((p.tokens||0)/S,1):p.stage==='synthesis'?Math.min((p.steps||0)/steps,1):p.stage==='decoding_audio'?.5:0;
 return Math.max(1,Math.min(99,(done+within*weight[p.stage])/total*100));
}
function progressFor(j){const text=(j.log||'').split('[candidate]').at(-1);
 if(j.status==='queued'){const n=queuePosition(j.id);return[0,'Waiting in queue'+(n?' · position '+n:'')]}
 if(j.status==='complete')return[100,'Finished'];if(j.status==='failed')return[0,'Generation failed'];if(j.status==='cancelled')return[0,'Stopped'];if(j.status==='interrupted')return[0,'Interrupted'];if(j.status==='cancelling')return[0,'Stopping…'];
 const staged=stagedProgress(j);
 if(staged!==null)return[staged,{loading:'Getting ready…',planning:'Composing the music…',arranging:'Arranging the cover…',music_tokens:'Writing the music…',synthesis:'Making the audio…',decoding_audio:'Finishing the audio…',complete:'Saving your song…'}[j.progress.stage]];
 if(text.includes('[vae]'))return[94,'Finishing the audio…'];const steps=[...text.matchAll(/\[nar\] step (\d+)\/(\d+)/g)];if(steps.length){let m=steps.at(-1);return[55+Number(m[1])/Number(m[2])*36,'Making the audio…']}
 if(text.includes('[nar]'))return[55,'Making the audio…'];const toks=[...text.matchAll(/\[semantic\] (\d+) tokens/g)];if(toks.length)return[20+Math.min(Number(toks.at(-1)[1])/j.request.semantic_sampling.max_tokens,1)*32,'Writing the music…'];
 if(text.includes('[semantic]'))return[20,'Writing the music…'];if(text.includes('[transcribe]'))return[30,'Analyzing your recording…'];if(text.includes('[plan]'))return[10,'Composing the music…'];return[3,'Getting ready…'];
}
function audioStepActivity(j,now=Date.now()/1000){
 const p=j.progress;
 if(j.status!=='running'||p?.stage!=='synthesis'){audioActivity=null;return null}
 const key=j.id+':'+p.candidate+':'+p.step_total;
 if(!audioActivity||audioActivity.key!==key||p.steps<audioActivity.steps)audioActivity={key,steps:p.steps,elapsed:p.elapsed,changed:now,rate:null};
 else if(p.steps>audioActivity.steps){
  const seconds=(p.elapsed-audioActivity.elapsed)/(p.steps-audioActivity.steps);
  if(Number.isFinite(seconds)&&seconds>0)audioActivity.rate=audioActivity.rate==null?seconds:audioActivity.rate*.5+seconds*.5;
  Object.assign(audioActivity,{steps:p.steps,elapsed:p.elapsed,changed:now});
 }
 return {rate:audioActivity.rate,age:Math.max(0,now-audioActivity.changed)};
}
/* One friendly sentence under the progress bar: which version, how fast tokens are coming, and whether a step has stalled.
   No time estimate: a song can stop well before its token cap, so any figure would swing too much to trust. */
function generationRateText(j){
 const p=j?.progress;
 if(!p||!['starting','running'].includes(j.status))return '';
 if(!(p.tokens_per_second>0)||['loading','analysis_loading','synthesis','decoding_audio','complete'].includes(p.stage))return '';
 return Math.round(p.tokens_per_second)+' tokens/sec';
}
function generationHint(j,activity){
 const p=j?.progress;
 if(!p||!['starting','running'].includes(j.status))return '';
 const parts=[];
 if((p.candidates||1)>1)parts.push('Version '+(p.candidate||1)+' of '+p.candidates);
 const closing=activity&&activity.age>Math.max(120,(activity.rate||0)*4)?'this step is slower than usual — you can stop, and finished versions stay saved':'';
 if(closing)parts.push(closing);
 if(!parts.length)return '';
 const text=parts.join(' · ');
 return text[0].toUpperCase()+text.slice(1)+(closing?'.':'');
}
function showLive(j){
 const detail=$('generationActivity');
 const rateEl=$('generationRate');
 if(typeof syncGenerationChrome==='function')syncGenerationChrome(j||{});
 const rate=generationRateText(j);
 if(rateEl){rateEl.textContent=rate;rateEl.hidden=!rate;}
 if(!detail)return;
 const hint=generationHint(j,audioStepActivity(j));
 detail.textContent=hint;
 detail.hidden=!hint;
}
// When a job ends the server immediately starts the next queued one; pick it up without waiting for the 2 s config refresh.
async function followQueue(){
 try{const fresh=await api('/api/config');config=fresh;if(fresh.active&&fresh.active!==active){active=fresh.active;if(!selected||!queuePosition(selected)){selected=active;lastStatus='';resultFingerprint=''}}}catch{}
}
async function poll(){clearTimeout(timer);try{
 if(!selected)return;
 if(active&&selected!==active){const running=await api('/api/jobs/'+active);runningJob=running;if(['complete','failed','cancelled','interrupted'].includes(running.status)){announceRender(running);active=null;runningJob=null;await refreshLibrary();await followQueue();updateUI()}}
 const chosen=await api('/api/jobs/'+selected);if(chosen.id===active)runningJob=chosen;if(chosen.kind==='transcribe'&&chosen.request.audio_id===source?.id){analysisLiveJob=chosen;showAnalysis()}
 // While your song waits in the queue, the panel shows the generation that is actually running.
 const j=chosen.status==='queued'&&runningJob&&runningJob.id===active&&['starting','running','cancelling'].includes(runningJob.status)?runningJob:chosen;
 showLive(j);
 let[pct,label]=progressFor(j);if(j.progress&&['running','starting'].includes(j.status)){const p=j.progress;const names={loading:'Getting ready…',analysis_loading:'Getting ready to listen…',planning:'Composing the music…',arranging:'Arranging the cover…',music_tokens:'Writing the music…',synthesis:'Making the audio…',decoding_audio:'Finishing the audio…',analysis_audio:'Reading the recording…',analysis_encoding:'Analyzing melody and harmony…',analysis_decoding:'Transcribing the performance…',analysis_notation:'Finishing audio analysis…',analysis_voice_audio:'Reading the isolated voice…',analysis_voice_encoding:'Listening to the voice alone…',analysis_voice_decoding:'Transcribing the voice alone…',analysis_voice_notation:'Finishing audio analysis…',analysis_complete:'Finishing audio analysis…',analysis_tokens_audio:'Reading the recording…',analysis_tokens_encoding:'Listening to the recording…',analysis_tokens_predicting:'Writing music tokens…',complete:'Saving your song…'};label=names[p.stage]||label;}
 if((j.request.candidates||1)>1&&!['complete','failed','cancelled'].includes(j.status)){const m=[...(j.log||'').matchAll(/\[candidate\] (\d+)\/(\d+)/g)].at(-1);if(m){pct=((Number(m[1])-1)*100+pct)/Number(m[2])}}
 if(j.kind==='transcribe'){if(j.status==='complete')label='Recording ready for covers';else if(j.status==='failed')label='Recording analysis failed';else if(j.progress?.windows>1)label+=' · segment '+j.progress.window+'/'+j.progress.windows;}
 if(j.kind==='tokenize'){if(j.status==='complete')label='Recording ready to continue';else if(j.status==='failed')label='Recording could not be tokenized';else if(j.progress?.windows>1)label+=' · segment '+j.progress.window+'/'+j.progress.windows;}
 const loading=['starting','running'].includes(j.status)&&(!j.progress||['loading','analysis_loading'].includes(j.progress.stage));
 if(loading&&j.progress?.loading_detail)label=j.progress.loading_detail+'…';
 $('progress').classList.toggle('loading',loading);
 $('progress').style.width=(loading?35:pct)+'%';$('status').textContent=label;$('progress').classList.toggle('busy',['running','starting'].includes(j.status));$('elapsed').textContent=j.started?formatGenerationTime(Date.now()/1000-j.started):'';
 const fingerprint=j.id+':'+j.status+':'+(j.result?.candidates?.length||0)+':'+(j.artwork?.status||'')+':'+(j.artwork?.url||'')+':'+(runningJob?.artwork?.status||'')+':'+(runningJob?.artwork?.url||'');if(fingerprint!==resultFingerprint){resultFingerprint=fingerprint;await refreshLibrary()}
 if(j.status!==lastStatus){const wasLive=['queued','starting','running','cancelling'].includes(lastStatus);lastStatus=j.status;
  if(['complete','failed','cancelled','interrupted'].includes(j.status)){
   if(active===j.id){active=null;runningJob=null}$('elapsed').textContent=j.result?.elapsed?formatGenerationTime(j.result.elapsed):'';await refreshLibrary();
   if(wasLive)announceRender(j);
   if(j.status==='failed'){notice(j.kind==='transcribe'?'Recording analysis failed. Use Retry analysis beside the recording.':j.kind==='tokenize'?'The recording could not be turned into music tokens. See the log for details.':'This job failed. Finished candidates are retained.',true)}
   await followQueue();
  }updateUI();
 }
 if(active||['queued','starting','running','cancelling'].includes(j.status)||j.artwork?.status==='running')timer=setTimeout(poll,1200);
}catch(e){notice('Connection lost: '+e.message,true);timer=setTimeout(poll,4000)}}
let jobsCache=[],detailJob=null,detailCandidate=null,playing=null,playQueue=[];
const icon=name=>{const img=document.createElement('img');img.src='/static/icons/'+name+'.svg?v=20260918';img.alt='';return img};
const clockTime=s=>Number.isFinite(s)?Math.floor(s/60)+':'+String(Math.floor(s%60)).padStart(2,'0'):'0:00';
function formatGenerationTime(seconds){
 const total=Math.max(0,Math.round(seconds)),hours=Math.floor(total/3600),minutes=Math.floor(total%3600/60),remaining=total%60;
 return (hours?hours+'h ':'')+(hours||minutes?minutes+'m ':'')+remaining+'s';
}
function generationSeconds(j,c){
 const r=j.result||{},multiple=(r.requested_candidates??j.request?.candidates??r.candidates?.length??1)>1;
 const seconds=multiple?c?.elapsed:(r.elapsed??c?.elapsed);
 return Number.isFinite(seconds)&&seconds>=0?seconds:null;
}
const VOICE_LABELS={male:'Male vocals',female:'Female vocals',duet:'Duet vocals'};
function voiceLabel(r){return r?.instrumental?'Instrumental':VOICE_LABELS[r?.voice]||'Vocals'}
/* Settings that shaped a song: tags next to its title so versions can be compared at a glance. */
function coverTags(j){
 const r=j.request||{},cover=j.kind==='cover';
 return [...(cover&&r.arrange===true?['Arranged']:[]),...(r.instrumental&&r.hook_melody?['Chorus melody']:[])];
}
function detailsShows(j,c){
 return document.body.classList.contains('details-open')&&detailJob?.id===j.id&&(detailCandidate?.index||0)===(c?.index||0);
}
function songKindTags(j){
 const tags=[];
 if(j.kind==='cover')tags.push(['cover','COVER']);
 for(const t of coverTags(j))tags.push([t.toLowerCase().replace(/\s+/g,'-'),t.toUpperCase()]);
 if(j.request.origin?.kind==='continue')tags.push(['extended','EXTENDED']);
 else if(j.request.origin?.kind==='speed')tags.push(['speed',Number(j.request.origin.factor).toFixed(2)+'X SPEED']);
 return tags;
}
function loraBadge(name,strength){return loraName(name)+(Number.isFinite(strength)&&Math.abs(strength-1)>.001?' ×'+Number(strength).toFixed(2):'')}
/* Job ids start with the local time the song was requested (YYYYMMDD-HHMMSS). */
function createdAt(j){
 const m=/^(\d{4})(\d\d)(\d\d)-(\d\d)(\d\d)(\d\d)/.exec(j?.id||'');if(!m)return null;
 const d=new Date(+m[1],m[2]-1,+m[3],+m[4],+m[5],+m[6]);if(Number.isNaN(d.getTime()))return null;
 const now=new Date(),sameDay=d.toDateString()===now.toDateString(),time=d.toLocaleTimeString(undefined,{hour:'2-digit',minute:'2-digit'});
 if(sameDay)return 'Today '+time;
 return d.toLocaleDateString(undefined,{day:'numeric',month:'short',...(d.getFullYear()!==now.getFullYear()?{year:'numeric'}:{})})+' '+time;
}
function timingBadges(j,c,{duration=true}={}){
 const badges=document.createElement('span');badges.className='song-timings';
 if(!c)return badges;
 const raw=j.request?.cfg_scale,guidance=typeof raw==='number'&&Number.isFinite(raw)?String(Number(raw.toFixed(2))):null;
 const upload=isUpload(j);
 const created=createdAt(j);
 for(const [label,value]of [['Duration',duration?clockTime(c.seconds):null],['Took',upload||generationSeconds(j,c)===null?null:formatGenerationTime(generationSeconds(j,c))],['Steps',!upload&&Number.isFinite(j.request?.steps)?j.request.steps:null],['Style LoRA',!upload&&j.request?.lora?loraBadge(j.request.lora,j.request.lora_strength):null],['Sound LoRA',!upload&&j.request?.sound_lora?loraBadge(j.request.sound_lora,j.request.sound_lora_strength):null],['Style influence',upload?null:guidance],['Created',created]]){
  if(value===null)continue;
  const badge=document.createElement('span');badge.className='timing-badge'+(label==='Created'?' timing-created':'');
  badge.title=label+' '+value;
  if(label==='Created'){
   badge.append(icon('clock'));
   const time=document.createElement('b');time.textContent=value;badge.append(time);
  }else{
   const caption=document.createElement('span');caption.textContent=label+' ';const time=document.createElement('b');time.textContent=value;
   badge.append(caption,time);
  }
  badges.append(badge);
 }
 return badges;
}
function songTitle(j,c){return c?.title||j.title}
function originLabel(r){
 const o=r?.origin;if(!o)return '';
 if(o.kind==='continue')return ' · Extended from '+(o.title||'a saved song')+(o.cut_seconds!=null&&o.cut_seconds<(o.seconds??Infinity)-0.05?' at '+clockTime(o.cut_seconds):' at its end');
 if(o.kind==='speed')return ' · '+Number(o.factor).toFixed(2)+'x speed of '+(o.title||'a saved song')+(o.keep_pitch?', pitch kept':', pitch shifted');
 return '';
}
function audioOnNode(j,c){return !!j.remote?.pending?.includes((c?.prefix||'')+'audio.wav')}
function nodeKeepLabel(j){const t=j.remote?.kept_until;return t?'Audio on the render node until '+new Date(t*1000).toLocaleDateString(undefined,{day:'numeric',month:'short'})+' · download to keep it':'Audio on the render node · download to keep it'}
// A song rendered on a node plays its MP3 from there; the WAV only comes over when it is downloaded or edited.
function songURL(j,c){const prefix=c?.prefix||'';return '/api/files/'+j.id+'/'+prefix+(audioOnNode(j,c)&&j.files.includes(prefix+'audio.mp3')?'audio.mp3':'audio.wav')}
function songDownloadURL(j,c,format='wav'){return '/api/jobs/'+j.id+'/songs/'+(c?.index||1)+'/download?format='+format}
const pendingDownloads=new Set();let downloadNoticeTimer;
function showDownloadNotice(message,error=false){
 clearTimeout(downloadNoticeTimer);const el=$('downloadNotice');el.textContent=message;el.hidden=!message;el.classList.toggle('error',error);
 if(message&&!error&&!pendingDownloads.size)downloadNoticeTimer=setTimeout(()=>{el.hidden=true},5000);
}
async function downloadSong(j,c,format){
 if(!c)return;
 const key=j.id+':'+c.index+':'+format;if(pendingDownloads.has(key))return;
 pendingDownloads.add(key);
 try{
  let url=songDownloadURL(j,c,format);
  if(audioOnNode(j,c)){
   showDownloadNotice('Fetching “'+songTitle(j,c)+'” from the render node…');
   await api('/api/jobs/'+j.id+'/fetch',{method:'POST'});
  }
  if(format==='mp3'){
   showDownloadNotice('Preparing MP3 for “'+songTitle(j,c)+'”…');
   const result=await api('/api/jobs/'+j.id+'/songs/'+c.index+'/export?format=mp3',{method:'POST'});url=result.url;
  }
  const link=document.createElement('a');link.href=url;link.download=songTitle(j,c)+'.'+format;document.body.append(link);link.click();link.remove();
  pendingDownloads.delete(key);showDownloadNotice(format.toUpperCase()+' download started.');
  if(j.remote){await refreshLibrary();const fresh=jobsCache.find(x=>x.id===j.id);if(fresh&&detailJob?.id===j.id)inspectSong(fresh,c.index?versions(fresh).find(v=>v.index===c.index)||c:c)}
 }catch(e){pendingDownloads.delete(key);showDownloadNotice(e.message,true)}
}
function versions(j){
 if(j.result?.candidates?.length)return j.result.candidates;
 if(j.files.includes('audio.wav'))return [{index:1,prefix:'',seconds:j.result?.seconds||0,model:j.request.model,seed:j.request.seed}];
 return [];
}
async function copySongText(field){
 const text=field==='style'?detailJob?.request.style:detailJob?.request.instrumental?'':detailJob?.request.lyrics;
 if(!text?.trim())return;
 try{await navigator.clipboard.writeText(text);showDownloadNotice(field==='style'?'Description copied.':'Lyrics copied.')}
 catch{showDownloadNotice('Could not copy. Select the text and copy it manually.',true)}
}
function setupSidebar(){
 let collapsed=window.matchMedia('(max-width: 950px)').matches;
 try{const saved=localStorage.getItem('opensuno-sidebar-collapsed');if(saved!==null)collapsed=saved==='true'}catch{}
 const apply=()=>{
  for(const node of [document.documentElement,document.body]){node.classList.toggle('sidebar-collapsed',collapsed);node.classList.toggle('sidebar-expanded',!collapsed)}
  const button=$('toggleSidebar'),label=collapsed?'Expand Sidebar':'Collapse Sidebar';
  button.setAttribute('aria-expanded',String(!collapsed));button.setAttribute('aria-label',label);button.title=label;
  for(const item of $('sidebarNav').querySelectorAll('.nav-item')){
   const name=item.querySelector('.nav-label')?.textContent?.trim()||item.getAttribute('aria-label')||'';
   if(collapsed)item.title=name;else item.removeAttribute('title');
  }
 };
 $('toggleSidebar').onclick=()=>{collapsed=!collapsed;apply();try{localStorage.setItem('opensuno-sidebar-collapsed',String(collapsed))}catch{}};
 apply();
}
function showDetailCover(j,c){
 const cover=$('detailCover'),panel=$('songDetails');if(!cover||!panel)return;
 requestArtwork(j);
 if(j?.artwork?.url){
  cover.hidden=false;cover.src=j.artwork.url;cover.alt=songTitle(j,c);
  panel.classList.add('has-art-bg');panel.style.setProperty('--detail-art','url('+JSON.stringify(j.artwork.url)+')');
 }else{
  cover.hidden=true;cover.removeAttribute('src');cover.alt='';
  panel.classList.remove('has-art-bg');panel.style.removeProperty('--detail-art');
 }
}
function hideSongDetails(){
 const panel=$('songDetails');if(!panel)return;
 clearTimeout(hideSongDetails.timer);
 if(inspectSong.frame){cancelAnimationFrame(inspectSong.frame);inspectSong.frame=0}
 document.body.classList.remove('details-open');
 detailJob=null;
 const finish=()=>{
  panel.hidden=true;panel.classList.remove('has-art-bg');panel.style.removeProperty('--detail-art');
 };
 if(panel.hidden||matchMedia('(prefers-reduced-motion:reduce)').matches){finish();return}
 hideSongDetails.timer=setTimeout(finish,400);
}
function inspectSong(j,c){
 clearTimeout(hideSongDetails.timer);
 if(inspectSong.frame){cancelAnimationFrame(inspectSong.frame);inspectSong.frame=0}
 const panel=$('songDetails'),reopen=panel.hidden||!document.body.classList.contains('details-open');
 panel.hidden=false;
 detailJob=j;detailCandidate=c;
 if(reopen){
  document.body.classList.remove('details-open');
  panel.getBoundingClientRect();
  inspectSong.frame=requestAnimationFrame(()=>{
   inspectSong.frame=requestAnimationFrame(()=>{
    inspectSong.frame=0;
    if(detailJob)document.body.classList.add('details-open');
   });
  });
 }else document.body.classList.add('details-open');
 showDetailCover(j,c);$('detailTitle').textContent=songTitle(j,c);
 $('detailMeta').textContent=isUpload(j)?'Uploaded recording · '+(Number.isFinite(c?.seconds)?clockTime(c.seconds):'')+(j.request.origin?.name?' · '+j.request.origin.name:'')+originLabel(j.request):(j.kind==='cover'?'Cover · ':'')+modelBadge(j,c)+' · '+voiceLabel(j.request)+' · '+j.status+originLabel(j.request)+coverTags(j).map(t=>' · '+t).join('')+(audioOnNode(j,c)?' · '+nodeKeepLabel(j):'');
 const seedEl=$('detailSeed'),seed=c?.seed??j.request.seed;
 $('detailTimings').after(seedEl);
 $('detailTimings').replaceChildren(timingBadges(j,c));
 seedEl.hidden=!Number.isFinite(seed);
 if(Number.isFinite(seed)){
  const caption=document.createElement('span');caption.textContent='Seed ';
  const value=document.createElement('b');value.textContent=String(seed);
  seedEl.replaceChildren(caption,value);
  $('detailTimings').querySelector('.song-timings').append(seedEl);
 }else seedEl.replaceChildren();
 $('detailStyle').textContent=j.request.style||(isUpload(j)?'Your own recording. Use it as a cover source or let the model continue it.':'No style saved.');
 $('detailLyrics').textContent=isUpload(j)?(j.request.lyrics||''):j.request.instrumental?'Instrumental'+(j.request.lyrics?.trim()?'\n\n'+j.request.lyrics.trim():''):j.request.lyrics||'No lyrics saved.';
 $('copyStyle').disabled=!j.request.style?.trim();$('copyLyrics').disabled=!j.request.lyrics?.trim()||(!isUpload(j)&&!!j.request.instrumental);
 if(!active){selected=j.id;lastStatus='';resultFingerprint='';poll()}
 renderLibrary();
}
function reuseSong(j,c){
 if(isUpload(j)){
  // The whole original recording becomes the Cover source; tagged lyrics fill the box the first time.
  mode='cover';source={id:j.request.audio_id,name:songTitle(j,c),seconds:c?.seconds,cover:!!j.artwork?.url,art:j.artwork?.url||''};sourceAnalysis={key:analysisKey(),status:'missing'};analysisLiveJob=null;analysisRetryRequested=false;
  $('title').value=songTitle(j,c).slice(0,100);$('sourceSeconds').value=0;
  const lyrics=applyRecordingLyrics(j.request.lyrics);
  showSource();saveDraft();navigate('create');updateUI();
  notice(lyrics?'Recording loaded as the cover source. Lyrics came from the file\u2019s own tags.':'Recording loaded as the cover source.');return;
 }
 mode=j.kind==='cover'||j.request.audio_id?'cover':'create';
 source=j.request.audio_id?{id:j.request.audio_id,name:'Saved recording'}:null;
 restore({...j.request,edit_id:'',title:songTitle(j,c),...(c?{seed:c.seed,candidates:1,random_seed:false}:{})});
 showSource();saveDraft();navigate('create');notice('Settings loaded. Edit them and create a new version.');
}
let songMenu=null,songMenuAnchor=null,songMenuRefreshPending=false;
function closeSongMenu(focus=false){
 const anchor=songMenuAnchor;
 songMenu?.remove();songMenu=null;songMenuAnchor=null;
 anchor?.setAttribute('aria-expanded','false');if(focus&&anchor?.isConnected)anchor.focus();
 if(songMenuRefreshPending){songMenuRefreshPending=false;queueMicrotask(()=>renderLibrary())}
}
function rerenderSong(j,c){
 const steps=Math.min(128,Math.max(1,Number(j.request.steps)||2));
 const sounds=(config.loras||[]).filter(LORA_SLOTS.sound.fits),hadSound=j.request.sound_lora||'',hadStrength=Number(j.request.sound_lora_strength??1)||1;
 const dialog=document.createElement('dialog');dialog.className='song-dialog';
 dialog.innerHTML='<form><h2>Re-render Audio</h2><p>Reuse saved music tokens and the same seed.</p>'
  +'<div class="sampling-slider-row"><label for="rerenderStepsSlider">Audio Steps</label>'
  +'<button id="resetRerenderSteps" class="sampling-reset" type="button" aria-label="Reset Audio Steps" title="Reset to '+steps+'"><img src="/static/icons/arrow-path.svg" alt=""></button>'
  +'<input id="rerenderStepsSlider" type="range" min="1" max="128" step="1" value="'+steps+'">'
  +'<input type="number" id="rerenderSteps" name="steps" min="1" max="128" value="'+steps+'" required aria-label="Audio Steps value"></div>'
  +(sounds.length?'<div class="rerender-sound"><label for="rerenderSound">Sound LoRA</label><select id="rerenderSound" name="sound_lora"></select>'
   +'<small>Same notes, different sound.</small></div>'
   +'<div class="sampling-slider-row" id="rerenderSoundStrengthRow" hidden><label for="rerenderSoundStrengthSlider">Sound Strength</label>'
   +'<button id="resetRerenderSoundStrength" class="sampling-reset" type="button" aria-label="Reset Sound Strength" title="Reset to 1"><img src="/static/icons/arrow-path.svg" alt=""></button>'
   +'<input id="rerenderSoundStrengthSlider" type="range" min="0.1" max="2" step="0.05" value="'+hadStrength+'">'
   +'<input type="number" id="rerenderSoundStrength" name="sound_lora_strength" min="0.1" max="2" step="0.05" value="'+hadStrength+'" aria-label="Sound Strength value"></div>':'')
  +'<div class="song-dialog-actions"><button type="button" id="rerenderCancel">Cancel</button><button type="submit" class="secondary">Re-render</button></div></form>';
 const form=dialog.querySelector('form'),input=dialog.querySelector('#rerenderSteps'),slider=dialog.querySelector('#rerenderStepsSlider');
 /* A Style LoRA that also adapts the decoder can be left out of the render: the tokens it wrote stay, its sound goes. */
 const styleLora=(config.loras||[]).find(l=>l.id===(j.request.lora||'')),styleTouchesSound=!!styleLora&&(styleLora.branches||[]).includes('nar');
 let withoutLora=null;
 if(styleTouchesSound){
  const label=document.createElement('label');label.className='rerender-plain';
  withoutLora=document.createElement('input');withoutLora.type='checkbox';withoutLora.id='rerenderWithoutLora';
  const text=document.createElement('span'),b=document.createElement('b'),small=document.createElement('small');
  b.textContent=loraName(j.request.lora);text.append('Render without ',b);
  small.textContent='This Style LoRA also changes the decoder. Keep what it wrote, render the sound with the stock model.';text.append(small);
  label.append(withoutLora,' ',text);form.querySelector('.song-dialog-actions').before(label);
 }
 const sync=()=>{slider.value=input.value;paintSliderTicks()};
 slider.oninput=()=>{input.value=slider.value;paintSliderTicks()};
 input.addEventListener('input',()=>{if(input.value!=='')sync()});
 dialog.querySelector('#resetRerenderSteps').onclick=()=>{input.value=steps;sync()};
 const sound=dialog.querySelector('#rerenderSound'),strengthRow=dialog.querySelector('#rerenderSoundStrengthRow'),strength=dialog.querySelector('#rerenderSoundStrength'),strengthSlider=dialog.querySelector('#rerenderSoundStrengthSlider');
 if(sound){
  sound.append(new Option('None',''));
  for(const l of sounds){const o=new Option(l.name,l.id);o.disabled=l.id===(j.request.lora||'');sound.append(o)}
  sound.value=sounds.some(l=>l.id===hadSound)?hadSound:'';
  const show=()=>{strengthRow.hidden=!sound.value;paintSliderTicks()};
  sound.onchange=show;show();
  strengthSlider.oninput=()=>{strength.value=strengthSlider.value;paintSliderTicks()};
  strength.addEventListener('input',()=>{if(strength.value!=='')strengthSlider.value=strength.value;paintSliderTicks()});
  dialog.querySelector('#resetRerenderSoundStrength').onclick=()=>{strength.value=strengthSlider.value=1;paintSliderTicks()};
 }
 dialog.querySelector('#rerenderCancel').onclick=()=>dialog.close();
 form.onsubmit=async e=>{e.preventDefault();try{askRenderAlerts();
  const body={...j.request,kind:'generate',edit_id:'',candidates:1,random_seed:false,render_source:j.id,render_candidate:c.index,steps:Number(input.value)};
  if(sound){body.sound_lora=sound.value;body.sound_lora_strength=sound.value?Number(strength.value)||1:1}
  if(withoutLora?.checked)body.render_without_lora=true;
  const result=await api('/api/jobs',{method:'POST',body:JSON.stringify(body)});accepted(result);dialog.close();await refreshLibrary();updateUI();poll()}catch(error){notice(error.message,true)}};
 dialog.onclose=()=>dialog.remove();document.body.append(dialog);dialog.showModal();paintSliderTicks();slider.focus();
}
/* Like Suno's Adjust speed: the finished audio is stretched into a new library entry; nothing is regenerated. */
function adjustSpeedSong(j,c){
 const seconds=Number(c?.seconds)||0,dialog=document.createElement('dialog');dialog.className='song-dialog speed-dialog';
 dialog.innerHTML='<form><h2>Adjust Speed</h2><p>Saves a faster or slower copy of this song. Your original stays saved.</p>'
  +'<output id="speedReadout" class="speed-readout" for="speedSlider" aria-live="polite">1.00x</output>'
  +'<input id="speedSlider" class="speed-slider" type="range" min="-200" max="200" step="1" value="0" aria-label="Speed" aria-valuetext="1.00x">'
  +'<div class="speed-range-labels"><span>0.25x</span><span>4.00x</span></div>'
  +'<div class="chips speed-presets" role="group" aria-label="Speed presets"></div>'
  +'<label class="instrumental-toggle speed-toggle"><span>Keep Pitch</span><input id="speedKeepPitch" type="checkbox" role="switch" checked><span class="switch" aria-hidden="true"></span></label>'
  +'<p id="speedHint" class="muted"></p><p id="speedError" class="speed-error" role="alert" hidden></p>'
  +'<div class="song-dialog-actions"><button type="button" id="speedCancel">Cancel</button><button type="submit" class="primary">Save Copy</button></div></form>';
 const form=dialog.querySelector('form'),slider=dialog.querySelector('#speedSlider'),readout=dialog.querySelector('#speedReadout'),keepPitch=dialog.querySelector('#speedKeepPitch'),hint=dialog.querySelector('#speedHint'),error=dialog.querySelector('#speedError'),submit=dialog.querySelector('button[type=submit]');
 // Logarithmic track so 1.00x sits in the middle, halving and doubling take the same distance.
 const factorOf=v=>Math.round(Math.pow(2,v/100)*100)/100,positionOf=f=>Math.round(Math.log2(f)*100);
 const paint=()=>{
  const f=factorOf(Number(slider.value)),same=Math.abs(f-1)<.005;
  readout.value=f.toFixed(2)+'x';slider.setAttribute('aria-valuetext',f.toFixed(2)+'x');slider.style.setProperty('--p',(Number(slider.value)+200)/400);
  hint.textContent=same?'Move the slider to speed the song up or slow it down.':(seconds?'New length about '+clockTime(seconds/f)+' (now '+clockTime(seconds)+'). ':'')+(keepPitch.checked?'The pitch stays the same; the song is time-stretched.':'The pitch moves with the speed, like a tape.');
  submit.disabled=same;
  for(const b of dialog.querySelectorAll('.speed-presets button'))b.classList.toggle('active',Number(b.dataset.factor)===f);
 };
 for(const f of [.5,.75,1,1.25,1.5,2]){const b=document.createElement('button');b.type='button';b.dataset.factor=f;b.textContent=f.toFixed(2)+'x';b.onclick=()=>{slider.value=positionOf(f);paint()};dialog.querySelector('.speed-presets').append(b)}
 slider.oninput=paint;keepPitch.onchange=paint;
 dialog.querySelector('#speedCancel').onclick=()=>dialog.close();
 form.onsubmit=async e=>{
  e.preventDefault();const factor=factorOf(Number(slider.value));submit.disabled=true;submit.textContent='Saving…';error.hidden=true;
  try{const made=await api('/api/jobs/'+j.id+'/songs/'+c.index+'/speed',{method:'POST',body:JSON.stringify({factor,keep_pitch:keepPitch.checked})});dialog.close();await refreshLibrary();updateUI();notice('Saved “'+made.title+'” to the library.')}
  catch(failure){submit.disabled=false;submit.textContent='Save Copy';error.textContent=failure.message;error.hidden=false}
 };
 dialog.onclose=()=>dialog.remove();document.body.append(dialog);paint();dialog.showModal();slider.focus();
}
async function toggleFavorite(j,c){
 try{await api('/api/jobs/'+j.id+'/favorite',{method:'POST',body:JSON.stringify({candidate:c.index})});await refreshLibrary()}catch(e){notice(e.message,true)}
}
function openSongMenu(anchor,j,c,downloadsOnly=false){
 if(songMenuAnchor===anchor){closeSongMenu(true);return}closeSongMenu();
 const menu=document.createElement('div');menu.className='song-menu';menu.setAttribute('role','menu');menu.setAttribute('aria-label','Song options');
 songMenu=menu;songMenuAnchor=anchor;anchor.setAttribute('aria-expanded','true');
 const hasTokens=!!c&&j.files.includes((c.prefix||'')+'tokens.json'),upload=isUpload(j);
 const mlx=!['cuda-bf16','cuda-fp8'].includes($('model').value);
 const favoriteLabel=j.favorite===c?.index?'Remove from Favorites':'Add to Favorites';
 const groups=upload?[
  [['cover','Use as Cover Source'],['continueRecording','Continue This Recording']],
  [['speed','Adjust Speed'],['favorite',favoriteLabel]],
  [['rename','Rename'],['workspace','Move to Project']],
  [['wav','Download WAV'],['mp3','Download MP3']],
  [['delete','Delete']]
 ]:[
  [['rerender','Re-render Audio'],['speed','Adjust Speed'],['extend','Extend'],['arrange','Edit']],
  [['favorite',favoriteLabel]],
  [['rename','Rename'],['workspace','Move to Project']],
  [['wav','Download WAV'],['mp3','Download MP3']],
  [['delete','Delete']]
 ];
 const addItem=([action,label])=>{
  if(downloadsOnly&&!['wav','mp3'].includes(action))return null;
  const button=document.createElement('button');button.type='button';button.setAttribute('role','menuitem');button.textContent=label;
  if(action==='delete')button.className='danger';
  const download=['wav','mp3'].includes(action),recording=action==='continueRecording';
  button.disabled=['rerender','extend'].includes(action)?!hasTokens:recording?(!config?.tokens_ready||config?.hosted||(action==='continueRecording'&&!mlx)):action==='favorite'?!c:action==='arrange'?(!c||j.request.origin?.kind==='speed'):action==='speed'?(!c||j.status!=='complete'||config?.hosted):download?(!c||pendingDownloads.has(j.id+':'+c.index+':'+action)):j.id===active;
  if(button.disabled)button.title=download?'Finished audio is required, or this download is already being prepared.':['extend'].includes(action)?'This version has no saved music tokens.':action==='speed'?'Finished audio is required.':action==='arrange'&&j.request.origin?.kind==='speed'?'A speed copy has no saved composition. Edit the original song instead.':recording?(!config?.tokens_ready?'Download Cover analysis and Audio input under Models first.':'Select an MLX model (YuE2 bf16) to extend recordings.'):'Wait for this generation to finish or stop it first.';
  button.onclick=()=>{closeSongMenu();if(action==='rerender')rerenderSong(j,c);else if(action==='speed')adjustSpeedSong(j,c);else if(action==='extend')openSongTimeline(j,c);else if(action==='cover')reuseSong(j,c);else if(action==='continueRecording')openContinueRecording(uploadRecording(j,c));else if(action==='favorite')toggleFavorite(j,c);else if(action==='arrange')openSongEditor(j,c);else if(action==='workspace')openMoveWorkspace([{job:j.id,candidate:c?.index||0}]);else if(download)downloadSong(j,c,action);else editSong(action,j,c,anchor)};
  return button;
 };
 let wrote=false;
 for(const group of groups){
  const buttons=group.map(addItem).filter(Boolean);
  if(!buttons.length)continue;
  if(wrote){const line=document.createElement('hr');line.className='song-menu-sep';menu.append(line)}
  for(const button of buttons)menu.append(button);
  wrote=true;
 }
 document.body.append(menu);const rect=anchor.getBoundingClientRect();
 menu.style.left=Math.max(8,Math.min(rect.right-menu.offsetWidth,innerWidth-menu.offsetWidth-8))+'px';
 menu.style.top=(rect.bottom+menu.offsetHeight+8>innerHeight?Math.max(8,rect.top-menu.offsetHeight-4):rect.bottom+4)+'px';
 menu.querySelector('button:not(:disabled)')?.focus();
 menu.onkeydown=e=>{if(['ArrowDown','ArrowUp','Home','End'].includes(e.key)){e.preventDefault();const items=[...menu.querySelectorAll('button:not(:disabled)')];if(!items.length)return;let i=items.indexOf(document.activeElement);i=e.key==='Home'?0:e.key==='End'?items.length-1:(i+(e.key==='ArrowDown'?1:-1)+items.length)%items.length;items[i].focus()}else if(e.key==='Tab')closeSongMenu()};
}
document.addEventListener('pointerdown',e=>{if(songMenu&&!songMenu.contains(e.target)&&!songMenuAnchor?.contains(e.target))closeSongMenu()});
document.addEventListener('keydown',e=>{if(e.key==='Escape'&&songMenu){e.preventDefault();closeSongMenu(true)}});
window.addEventListener('resize',()=>closeSongMenu());
document.addEventListener('scroll',e=>{if(songMenu&&!songMenu.contains(e.target)&&(e.target===document||e.target.contains?.(songMenuAnchor)))closeSongMenu()},true);
function songDialog(name){return new Promise(resolve=>{
 const dialog=document.createElement('dialog');dialog.className='song-dialog';dialog.setAttribute('aria-labelledby','songDialogTitle');
 const form=document.createElement('form');form.method='dialog';
 const h=document.createElement('h2');h.id='songDialogTitle';h.textContent='Rename Song';form.append(h);
 const label=document.createElement('label');label.textContent='Song Title';const input=document.createElement('input');input.type='text';input.value=name;input.maxLength=100;input.required=true;label.append(input);form.append(label);
 const buttons=document.createElement('div');buttons.className='song-dialog-actions';
 const cancel=document.createElement('button');cancel.type='button';cancel.textContent='Cancel';cancel.onclick=()=>dialog.close();
 const submit=document.createElement('button');submit.type='submit';submit.className='primary';submit.textContent='Save';buttons.append(cancel,submit);form.append(buttons);dialog.append(form);document.body.append(dialog);
 let value=null;
 form.onsubmit=e=>{e.preventDefault();if(!input.value.trim()){input.setCustomValidity('Enter a song title.');input.reportValidity();return}value=input.value.trim();dialog.close()};
 input.oninput=()=>input.setCustomValidity('');
 dialog.addEventListener('close',()=>{dialog.remove();resolve(value)},{once:true});dialog.showModal();
 input.focus();input.select();
})}
async function editSong(action,j,c,anchor){
 const value=action==='rename'?await songDialog(songTitle(j,c)):true;if(value===null){if(anchor.isConnected)anchor.focus();return}
 try{
  await api('/api/jobs/'+j.id+'/songs/'+(c?.index||0),{method:action==='rename'?'PATCH':'DELETE',...(action==='rename'?{body:JSON.stringify({title:value})}:{})});
  if(action==='delete'){
   if(playing?.key===j.id+':'+(c?.index||0)){const audio=$('playerAudio');playing=null;audio.pause();audio.removeAttribute('src');audio.load();syncPlayer()}
   if(selected===j.id){clearTimeout(timer);selected=active;lastStatus='';$('status').textContent='Ready to create';$('elapsed').textContent='';$('progress').style.width='0';showLive({});if(active)poll()}
   if(detailJob?.id===j.id)hideSongDetails()
  }
  await refreshLibrary();
  if(action==='rename'){
   const fresh=jobsCache.find(x=>x.id===j.id),version=versions(fresh).find(x=>x.index===c?.index);
   if(playing?.key===j.id+':'+(c?.index||0)){playing.j=fresh;playing.c=version;$('playerTitle').textContent=songTitle(fresh,version)}
   if(detailJob?.id===j.id)inspectSong(fresh,version);
  }
  if(action==='rename')showDownloadNotice('Song renamed.');
  if(action==='delete')showDownloadNotice('Song moved to trash.');
 }catch(e){showDownloadNotice(e.message,true)}
}

function renderLibrary(){
 if(songMenu){songMenuRefreshPending=true;return}
 let jobs=[...jobsCache];const query=$('searchLibrary').value.trim().toLowerCase(),filter=$('libraryFilter').value;
 if(query)jobs=jobs.filter(j=>(j.title+' '+j.request.style+' '+versions(j).map(c=>songTitle(j,c)).join(' ')).toLowerCase().includes(query));
 if(filter==='covers')jobs=jobs.filter(j=>j.kind==='cover');
 if(filter==='uploads')jobs=jobs.filter(isUpload);
 if($('librarySort').value==='old')jobs.reverse();
 $('library').replaceChildren();playQueue=[];visibleWorkspaceSongs=[];
 for(const j of jobs){
  const candidates=versions(j).filter(c=>filter!=='favorites'||c.index===j.favorite);
  if(filter==='favorites'&&!candidates.length)continue;
  const rows=candidates.length?candidates:[null];
  for(const c of rows){
   if(!workspaceMatches(j,c))continue;
   const name=songTitle(j,c);
   const key=j.id+':'+(c?.index||0),row=document.createElement('article');row.className='song-row'+(detailsShows(j,c)?' selected':'');
   const play=document.createElement('button');play.className='song-play';play.type='button';play.disabled=!c;play.setAttribute('aria-label',c?'Play '+name+' version '+c.index:'Audio unavailable: '+name);
   if(j.artwork?.url){play.classList.add('has-art');const art=document.createElement('img');art.className='song-art';art.alt='';art.src=j.artwork.url;play.append(art)}
   else if(j.artwork?.status==='running')play.classList.add('art-loading');
   const glyph=icon('play');glyph.className='song-play-glyph';
   const eq=document.createElement('span');eq.className='song-eq';eq.setAttribute('aria-hidden','true');for(let i=0;i<4;i++)eq.append(document.createElement('i'));
   // Total length, shown statically on the cover; the player bar carries the live position.
   const time=document.createElement('span');time.className='song-play-time';time.hidden=!Number.isFinite(c?.seconds);if(!time.hidden)time.textContent=clockTime(c.seconds);
   play.append(glyph,eq,time);
   if(c)playQueue.push({j,c,key});
   play.onclick=e=>{e.stopPropagation();if(c)playSong(j,c)};
   const openDetails=()=>{if(detailsShows(j,c)){hideSongDetails();renderLibrary();return}inspectSong(j,c)};
   const info=document.createElement('button');info.className='song-info';info.onclick=e=>{e.stopPropagation();openDetails()};
   row.onclick=e=>{if(e.target.closest('button, input, a'))return;openDetails()};
   const heading=document.createElement('span');heading.className='song-heading';const title=document.createElement('strong');title.textContent=name;
   const tags=document.createElement('span');tags.className='song-tags';
   const model=modelBadge(j,c);if(model){const tag=document.createElement('span');tag.className='song-tag '+(isUpload(j)?'uploaded':'model');tag.textContent=model;tags.append(tag)}
   for(const [kind,label]of songKindTags(j)){const tag=document.createElement('span');tag.className='song-tag '+kind;tag.textContent=label;tags.append(tag)}
   if(j.request.origin)tags.title=originLabel(j.request).replace(/^ · /,'');
   heading.append(title,tags);
   const style=document.createElement('span');style.className='song-style';style.textContent=j.request.style||(isUpload(j)?'Uploaded recording':'Audio session');
   const status=document.createElement('span');status.className='song-status';
   if(c){
    const extra=document.createElement('span');extra.className='song-version';extra.textContent=isUpload(j)?'Your recording':'Version '+c.index+' · '+voiceLabel(j.request);
    status.append(timingBadges(j,c,{duration:false}));if(versions(j).length>1)status.append(extra);
   }else{
    status.textContent=(j.status==='queued'?'Queued'+(queuePosition(j.id)?' · position '+queuePosition(j.id):''):j.status)+(j.kind==='transcribe'?' · Recording analysis':j.kind==='tokenize'?' · Recording tokens':'');
   }
   if(currentWorkspace==='all'){const tag=document.createElement('span');tag.className='song-workspace-tag';tag.textContent=workspaceName(songWorkspace(j,c));status.append(tag)}
   info.append(heading,style,status);
   const actions=document.createElement('div');actions.className='row-actions';
   if(c){
    const favorite=document.createElement('button');favorite.className='icon-button'+(j.favorite===c.index?' chosen':'');favorite.setAttribute('aria-label',j.favorite===c.index?'Remove from Favorites':'Add to Favorites');favorite.setAttribute('aria-pressed',String(j.favorite===c.index));favorite.append(icon(j.favorite===c.index?'heart-solid':'heart'));favorite.onclick=e=>{e.stopPropagation();toggleFavorite(j,c)};actions.append(favorite);
   }
   const more=document.createElement('button');more.className='icon-button';more.setAttribute('aria-label','More Options: '+name+(c?' version '+c.index:''));more.setAttribute('aria-haspopup','menu');more.setAttribute('aria-expanded','false');more.append(icon('ellipsis-horizontal'));more.onclick=e=>{e.stopPropagation();openSongMenu(more,j,c)};actions.append(more);
   row.append(play,info,actions);decorateWorkspaceRow(row,j,c);row.dataset.key=key;$('library').append(row);
  }
 }
 if(!$('library').children.length){const workspaceEmpty=currentWorkspace!=='all'&&!workspaceCounts()[currentWorkspace];const empty=document.createElement('div');empty.className='empty-library';empty.append(icon('musical-note'));const h=document.createElement('h3');h.textContent=workspaceEmpty?'This project is empty':jobsCache.length?'No matching songs':'Your songs will appear here';const p=document.createElement('p');p.textContent=workspaceEmpty?'Move songs here or create a new song.':jobsCache.length?'Try another search or filter.':'Enter lyrics and a style, then create.';empty.append(h,p);$('library').append(empty)}
 finishWorkspaceRows();syncPlayer();
}
async function refreshLibrary(){
 const [jobs,data]=await Promise.all([api('/api/jobs'),api('/api/workspaces')]);jobsCache=jobs;acceptWorkspaces(data);$('count').textContent=jobsCache.reduce((sum,j)=>sum+Math.max(versions(j).length,1),0);
 for(const j of jobsCache){
  if(['queued','starting','running','cancelling'].includes(j.status)){if(j.status!=='queued')active=j.id;requestArtwork(j)}
 }
 const waiting=jobsCache.find(j=>j.id===selected)||detailJob;
 if(waiting)requestArtwork(waiting);
 if(detailJob){const fresh=jobsCache.find(x=>x.id===detailJob.id);if(fresh){detailJob=fresh;showDetailCover(fresh,detailCandidate||versions(fresh)[0])}}
 renderLibrary();updateUI();
}
function navigate(view,opts){
 if(config?.hosted&&view==='models')view='create';
 view=STUDIO_VIEWS.includes(view)?view:'create';
 if(typeof openedFromProjects!=='undefined'){
  if(view==='library'&&!opts?.fromProject)openedFromProjects=false;
  if(view!=='library'&&view!=='projects')openedFromProjects=false;
 }
 for(const name of ['library','projects','prompts','settings','models']){
  const on=view===name;
  document.body.classList.toggle(name+'-view',on);
  document.documentElement.classList.toggle(name+'-view',on);
 }
 const tab=view==='library'&&typeof openedFromProjects!=='undefined'&&openedFromProjects?'projects':view;
 for(const [id,v]of [['navCreate','create'],['navLibrary','library'],['navWorkspaces','projects'],['navPrompts','prompts'],['navModels','models'],['navSettings','settings']])$(id)?.classList.toggle('active',tab===v);
 modelPanelOpen=view==='models';
 $('studioSettings').hidden=view!=='settings';
 if(view==='settings')setupStudioSettings.sync?.();
 if(view==='projects'){hideSongDetails();if(typeof renderProjectList==='function')renderProjectList()}
 if(view==='prompts'){hideSongDetails();if(typeof refreshPrompts==='function')refreshPrompts()}
 else if(typeof closePromptMenu==='function')closePromptMenu();
 if(typeof syncLibraryBack==='function')syncLibraryBack();
 showModelSetup();
 const want=view==='create'?'':view;
 const have=decodeURIComponent((location.hash||'').replace(/^#/,'')).split(/[/?#]/)[0];
 if(have!==want){
  const url=want?'#'+want:(location.pathname+location.search||'/');
  history[opts?.fromHash?'replaceState':'pushState'](null,'',url);
 }
}
const STUDIO_VIEWS=['create','library','projects','prompts','models','settings'];
function studioHashView(){
 const raw=decodeURIComponent((location.hash||'').replace(/^#/,'')).split(/[/?#]/)[0].toLowerCase();
 return STUDIO_VIEWS.includes(raw)?raw:'create';
}
window.addEventListener('popstate',()=>navigate(studioHashView(),{fromHash:true}));
let modelPanelOpen=false;
async function playSong(j,c){
 const key=j.id+':'+c.index,audio=$('playerAudio');
 if(playing?.key===key){if(audio.paused)await audio.play().catch(e=>notice('Could not play audio: '+e.message,true));else audio.pause();return}
 playing={j,c,key};audio.src=songURL(j,c);$('playerTitle').textContent=songTitle(j,c);$('playerMeta').textContent=isUpload(j)?'Uploaded recording':c.model.toUpperCase();
 await audio.play().catch(e=>notice('Could not play audio: '+e.message,true));syncPlayer();
}
function updatePlayerProgress(){
 const audio=$('playerAudio'),seek=$('playerSeek');
 const duration=Number.isFinite(audio.duration)&&audio.duration>0?audio.duration:0;
 const current=duration?Math.min(duration,Math.max(0,audio.currentTime||0)):0;
 seek.max=duration||100;seek.value=current;seek.style.setProperty('--played',duration?(current/duration*100)+'%':'0%');
 seek.setAttribute('aria-valuetext',duration?clockTime(current)+' of '+clockTime(duration):'No audio loaded');
 $('playerTime').textContent=clockTime(current);$('playerDuration').textContent=duration?clockTime(duration):'—:—';
}
function showPlayerCover(j){
 const cover=$('playerCover');if(!cover)return;
 if(j?.artwork?.url){cover.hidden=false;cover.src=j.artwork.url;cover.alt=''}
 else{cover.hidden=true;cover.removeAttribute('src');cover.alt=''}
}
function syncPlayer(){
 $('playerEnd').hidden=!playing;
 const audio=$('playerAudio'),idx=playQueue.findIndex(x=>x.key===playing?.key);
 if(!playing){$('playerTitle').textContent='';$('playerMeta').textContent='';showPlayerCover()}
 else showPlayerCover(playing.j);
 updatePlayerProgress();
 $('playerPlay').disabled=!playing;$('playerSeek').disabled=!playing||!Number.isFinite(audio.duration);
 $('playerPrevious').disabled=idx<=0;$('playerNext').disabled=idx<0||idx>=playQueue.length-1;
 $('playerPlay').setAttribute('aria-label',audio.paused?'Play':'Pause');$('playerPlay').replaceChildren(icon(audio.paused?'play':'pause'));
 for(const row of $('library').children){const current=row.dataset.key===playing?.key;row.classList.toggle('playing',current&&!audio.paused);const button=row.querySelector('.song-play');if(button&&row.dataset.key?.split(':')[1]!=='0'){const img=button.querySelector('.song-play-glyph');if(img)img.src='/static/icons/'+(current&&!audio.paused?'pause':'play')+'.svg?v=20260918';button.setAttribute('aria-label',(current&&!audio.paused?'Pause ':'Play ')+row.querySelector('strong').textContent+' version '+row.dataset.key.split(':')[1]);}}
}
const LYRIC_TAGS=['Intro','Verse','Pre-Chorus','Chorus','Post-Chorus','Bridge','Interlude','Instrumental','Rap','Outro'];
function insertLyricTag(field,tag,start,end){
 if(!field||field.disabled)return;
 const text=field.value,a=Math.max(0,Math.min(start,text.length)),b=Math.max(a,Math.min(end,text.length));
 const before=text.slice(0,a),selected=text.slice(a,b),after=text.slice(b);
 const prefix=!before?'':before.endsWith('\n\n')?'':before.endsWith('\n')?'\n':'\n\n';
 const insert=selected?prefix+'['+tag+']\n'+selected:prefix+'['+tag+']';
 field.value=before+insert+after;
 const pos=before.length+insert.length;
 field.setSelectionRange(pos,pos);field.focus();
 field.dispatchEvent(new Event('input',{bubbles:true}));
}
function setupLyricTags(){
 const lyrics=$('lyrics'),expanded=$('expandedLyrics'),dialog=$('lyricsEditorDialog');
 let caret={field:lyrics,start:0,end:0},menu=null,anchor=null;
 function remember(field){if(!field||field.disabled)return;caret={field,start:field.selectionStart,end:field.selectionEnd}}
 function closeMenu(focus=false){
  if(!menu)return;
  const button=anchor;anchor?.setAttribute('aria-expanded','false');
  if(menu.matches?.(':popover-open'))menu.hidePopover();
  menu.remove();menu=null;anchor=null;if(focus)button?.focus();
 }
 function activeField(){return dialog?.open&&!$('addExpandedLyricTag')?.hidden?expanded:lyrics}
 function insert(tag){
  const field=activeField();
  const fromCaret=caret.field===field||caret.field===lyrics||caret.field===expanded;
  insertLyricTag(field,tag,fromCaret?caret.start:field.selectionStart,fromCaret?caret.end:field.selectionEnd);
  remember(field);closeMenu();
 }
 function place(button){
  const r=button.getBoundingClientRect(),gap=6,edge=8,width=Math.min(220,innerWidth-2*edge);
  menu.style.width=width+'px';
  const below=innerHeight-r.bottom-gap-edge,above=r.top-gap-edge,up=below<240&&above>below;
  menu.style.left=Math.max(edge,Math.min(r.left,innerWidth-width-edge))+'px';
  menu.style.top=(up?Math.max(edge,r.top-gap-menu.offsetHeight):r.bottom+gap)+'px';
 }
 function openMenu(button){
  if(anchor===button){closeMenu(true);return}
  closeMenu();
  anchor=button;button.setAttribute('aria-expanded','true');
  menu=document.createElement('div');menu.className='lyric-tag-menu';menu.setAttribute('role','menu');menu.setAttribute('aria-label','Section tags');
  if(HTMLElement.prototype.showPopover)menu.setAttribute('popover','manual');
  for(const tag of LYRIC_TAGS){
   const item=document.createElement('button');item.type='button';item.setAttribute('role','menuitem');item.textContent=tag;
   item.addEventListener('mousedown',e=>e.preventDefault());
   item.onclick=()=>insert(tag);
   menu.append(item);
  }
  (button.closest('dialog')||document.body).append(menu);
  if(menu.showPopover)menu.showPopover();
  place(button);
  menu.querySelector('button')?.focus();
  menu.onkeydown=e=>{
   const items=[...menu.querySelectorAll('button')];
   if(['ArrowDown','ArrowUp','Home','End'].includes(e.key)){
    e.preventDefault();let i=items.indexOf(document.activeElement);
    i=e.key==='Home'?0:e.key==='End'?items.length-1:(i+(e.key==='ArrowDown'?1:-1)+items.length)%items.length;items[i].focus();
   }else if(e.key==='Tab')closeMenu();
  };
 }
 for(const field of [lyrics,expanded]){
  if(!field)continue;
  for(const ev of ['keyup','pointerup','select','focus','click'])field.addEventListener(ev,()=>remember(field));
 }
 for(const id of ['addLyricTag','addExpandedLyricTag']){
  const button=$(id);if(!button)continue;
  button.addEventListener('pointerdown',()=>{const active=document.activeElement;if(active===lyrics||active===expanded)remember(active)});
  button.onclick=e=>{e.preventDefault();e.stopPropagation();openMenu(button)};
 }
 document.addEventListener('pointerdown',e=>{if(menu&&!anchor?.contains(e.target)&&!menu.contains(e.target))closeMenu()},true);
 document.addEventListener('keydown',e=>{if(e.key==='Escape'&&menu){e.preventDefault();e.stopPropagation();closeMenu(true)}},true);
 dialog?.addEventListener('close',()=>closeMenu());
}
function setupEditorToggles(){
 for(const block of document.querySelectorAll('details.editor-block')){
  const summary=block.querySelector(':scope>summary');
  if(!summary)continue;
  summary.addEventListener('click',e=>{
   if(e.target.closest('button,label,input,a,select,textarea'))e.preventDefault();
  });
 }
}
function setupLyricsEditor(){
 const dialog=$('lyricsEditorDialog'),editor=$('expandedLyrics');
 const styleButton=$('expandLyrics').cloneNode(true);styleButton.id='expandStyle';styleButton.setAttribute('aria-label','Expand Styles Editor');styleButton.title='Expand Styles Editor';
 $('style').closest('details').querySelector('summary').append(styleButton);
 let target=$('lyrics'),opener=$('expandLyrics');
 for(const [button,field,title] of [[$('expandLyrics'),$('lyrics'),'Lyrics'],[styleButton,$('style'),'Styles']]){
  button.onclick=e=>{e.preventDefault();e.stopPropagation();target=field;opener=button;editor.value=field.value;editor.disabled=field.disabled;$('lyricsEditorHeading').textContent=title;editor.setAttribute('aria-label',title);editor.placeholder=field.placeholder;const tagButton=$('addExpandedLyricTag');if(tagButton)tagButton.hidden=field!==$('lyrics');dialog.showModal();editor.focus()};
 }
 editor.oninput=()=>{target.value=editor.value;target.dispatchEvent(new Event('input',{bubbles:true}))};
 $('clearLyrics').onclick=()=>{$('lyrics').value='';if(target===$('lyrics'))editor.value='';$('lyrics').dispatchEvent(new Event('input',{bubbles:true}));$('lyrics').focus()};
 $('clearStyle').onclick=()=>{$('style').value='';if(target===$('style'))editor.value='';$('style').dispatchEvent(new Event('input',{bubbles:true}));$('style').focus()};
 $('closeLyricsEditor').setAttribute('aria-label','Close editor');
 $('closeLyricsEditor').onclick=$('doneLyricsEditor').onclick=()=>dialog.close();
 dialog.addEventListener('close',()=>opener.focus());
 setupLyricTags();
}
function setupPlayer(){
 const audio=$('playerAudio');
 $('playerPlay').onclick=()=>{if(!playing)return;if(audio.paused)audio.play().catch(e=>notice(e.message,true));else audio.pause()};
 for(const [id,step]of [['playerPrevious',-1],['playerNext',1]])$(id).onclick=()=>{const index=playQueue.findIndex(x=>x.key===playing?.key),next=playQueue[index+step];if(next)playSong(next.j,next.c)};
 $('playerRepeat').onclick=()=>{audio.loop=!audio.loop;$('playerRepeat').setAttribute('aria-pressed',String(audio.loop))};
 $('playerVolume').oninput=()=>{audio.volume=Number($('playerVolume').value);$('playerVolume').style.setProperty('--played',(audio.volume*100)+'%')};
 $('playerVolume').style.setProperty('--played',(audio.volume*100)+'%');
 $('playerSeek').oninput=()=>{if(Number.isFinite(audio.duration)){audio.currentTime=Number($('playerSeek').value);updatePlayerProgress()}};
 for(const event of ['loadedmetadata','durationchange'])audio.addEventListener(event,syncPlayer);
 audio.addEventListener('timeupdate',updatePlayerProgress);
 for(const event of ['play','pause','ended','emptied'])audio.addEventListener(event,syncPlayer);
 audio.addEventListener('error',()=>{if(playing)notice('This audio could not be loaded. Try selecting the song again.',true)});
 syncPlayer();
}

async function init(){try{
 setupSidebar();setupUIKit();$('copyStyle').onclick=()=>copySongText('style');$('copyLyrics').onclick=()=>copySongText('lyrics');
 config=await api('/api/config');applyHostedMode();applyModelChoices();setupLoras();setupSongEditing();if(typeof setupSongTimeline==='function')setupSongTimeline();setupWorkspaces();if(typeof setupPrompts==='function')setupPrompts();setupStyleBuilder();setupPlayer();setupEditorToggles();setupLyricsEditor();if(typeof setupLyricsStructure==='function')setupLyricsStructure();setupStyleAI();setupStudioSettings();setupRenderNode();setupRenderSliders();setupSimpleSliders();setupResizeHandles();setupTrimEditor();
 infoTip($('arrangeBox').querySelector('.setting-label'),'Arrange with the Model','Keeps the melody and instrumental lines heard in the recording; the model adds chords on every bar and writes its own lines where nothing was heard, in the new style.','help_arrangeCover');
 infoTip($('vocalOctaveRow').querySelector('.setting-label'),'Vocal Range','The transcriber often writes a low male singer an octave up, and the model then sings those high notes with a female voice whatever the Voice setting says. Match the Voice moves the melody by whole octaves into the register of the voice you chose under Controls; As Heard keeps the transcription; Down, Down Two and Up force the shift. Two octaves down gives a bass-baritone reading; lower than that is below any singer. Chords, timing and the instrument lines do not change.','help_vocalOctave');
 infoTip($('hookMelodyBox').querySelector('.setting-label'),'Melody Only in the Chorus','The instrument plays the sung melody only in the choruses. The verses keep just their chords, so the model arranges them as a full band instead of following a lead line for the whole song.','help_hookMelody');
 for(const [id,view]of [['navCreate','create'],['navLibrary','library'],['navWorkspaces','projects'],['navPrompts','prompts'],['navModels','models'],['navSettings','settings']]){const el=$(id);if(el)el.onclick=()=>navigate(view)}
 $('searchLibrary').oninput=renderLibrary;$('libraryFilter').onchange=renderLibrary;$('librarySort').onchange=renderLibrary;
 $('closeDetails').onclick=()=>{hideSongDetails();renderLibrary()};
 $('removeSource').onclick=()=>{source=null;$('sourceAudio').value='';showSource();updateUI();saveDraft()};
 try{const d=JSON.parse(localStorage.getItem('yue2-studio-draft'));if(d){mode=d.mode==='cover'?'cover':'create';source=d.source;restore(d.request);compositionDraft=d.compositionDraft||compositionDraft}}catch{}
 connectPageSession();showSource();active=config.active;selected=active;updateUI();await refreshLibrary();navigate(studioHashView(),{fromHash:true});if(active){selected=active;await poll();if(analysisLiveJob?.id===active)openAnalysisDialog()}
 checkSourceAnalysis();
 for(const b of document.querySelectorAll('[data-mode]'))b.onclick=()=>setMode(b.dataset.mode);
 $('seed').oninput=()=>{$('lockSeed').checked=true};
 $('randomSeed').onclick=()=>{$('seed').value=crypto.getRandomValues(new Uint32Array(1))[0];$('lockSeed').checked=true;updateUI();saveDraft()};
 $('lockSeed').onchange=()=>{updateUI();saveDraft()};$('model').onchange=()=>{updateUI();saveDraft()};$('cot').onchange=()=>{updateUI();saveDraft()};$('voice').onchange=()=>{updateUI();saveDraft()};
 $('instrumental').closest('label').addEventListener('click',e=>e.stopPropagation());
 $('instrumental').onchange=()=>{if($('instrumental').checked&&!$('lyrics').value.trim()){$('lyrics').value=DEFAULT_STRUCTURE;$('expandedLyrics').value=DEFAULT_STRUCTURE}updateUI();saveDraft()};
 $('hookMelody').onchange=()=>{updateUI();saveDraft()};
 $('vocalOctave').onchange=()=>{updateUI();saveDraft()};
 $('duration').onchange=()=>{if($('duration').value==='')return;$('duration').value=durationTokens()/25;saveDraft()};
 $('duration').oninput=()=>{if($('duration').value!=='')saveDraft()};
 for(const id of ['preserve','sourceSeconds'])$(id).onchange=changedAnalysisSettings;
 $('retryAnalysis').onclick=()=>{openAnalysisDialog();checkSourceAnalysis(true)};
 $('analysisBack').onclick=$('analysisContinue').onclick=()=>$('analysisDialog').close();
 $('analysisCancel').onclick=cancelRecordingAnalysis;
 $('sourceAudio').onchange=()=>uploadFile($('sourceAudio'));setupRecordingDrop();$('resetDefaults').onclick=restoreDefaults;
 $('generate').onclick=()=>start(mode==='cover'?'cover':'generate');
 $('cancel').onclick=async()=>{if(!active)return;try{await api('/api/jobs/'+active+'/cancel',{method:'POST'});poll()}catch(e){notice(e.message,true)}};
 $('dequeue').onclick=async()=>{const target=selected;if(!target||!queuePosition(target))return;try{await api('/api/jobs/'+target+'/cancel',{method:'POST'});config.queue=(config.queue||[]).filter(id=>id!==target);notice('Removed from the queue.');poll()}catch(e){notice(e.message,true)}};
 $('workspaceNew').onclick=()=>{clearTimeout(timer);selected=active||null;mode='create';source=null;restore({title:'',style:'',lyrics:'',seed:831001,steps:2,cfg_scale:1.2});showSource();$('sourceAudio').value='';hideSongDetails();$('status').textContent='Ready';$('elapsed').textContent='';$('progress').style.width='0';showLive({});notice('');saveDraft();refreshLibrary();navigate('create');if(active)poll()};
 $('composer').onsubmit=e=>e.preventDefault();for(const panel of [$('composer'),$('controls'),$('title')]){panel.addEventListener('input',()=>{updateUI();saveDraft()});panel.addEventListener('change',saveDraft)}
}catch(e){notice('Could not start Studio: '+e.message,true)}}
document.addEventListener('play',e=>{if(e.target.tagName==='AUDIO')document.querySelectorAll('audio').forEach(a=>{if(a!==e.target)a.pause()})},true);
// The page redraws only when the server's picture changed: models, downloads, queue, render node, LoRAs.
init().then(()=>{prepareModels(true)});setInterval(async()=>{if(!config)return;try{const fresh=await api('/api/config');const text=JSON.stringify(fresh);const changed=text!==configJSON;configJSON=text;const modelsChanged=fresh.models.map(m=>m.id).join()!==config.models.map(m=>m.id).join();const lorasChanged=(fresh.loras||[]).map(l=>l.id).join()!==(config.loras||[]).map(l=>l.id).join();config=fresh;if(modelsChanged)applyModelChoices();if(lorasChanged)applyLoraChoices();connectPageSession();if(fresh.active&&!active){active=fresh.active;selected=active;lastStatus='';poll()}if(changed){updateUI();renderNodeRefresh?.()}checkSourceAnalysis();prepareModels()}catch{}},2000);
