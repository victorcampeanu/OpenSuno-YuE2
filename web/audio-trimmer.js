let trimState=null,trimRevision=0,trimSaving=false,trimFrame=null;
function trimTime(seconds){
 const tenths=Math.round(Math.max(0,seconds||0)*10);
 return Math.floor(tenths/600)+':'+String(Math.floor(tenths/10)%60).padStart(2,'0')+'.'+tenths%10;
}
function trimDisplayValue(seconds){return Number(seconds.toFixed(3))}
function trimError(message){$('trimStatus').textContent=message;$('trimStatus').classList.toggle('error',!!message)}
function showTrimUpload(name){
 trimRevision++;trimState=null;trimSaving=false;$('trimAudio').pause();
 $('trimFilename').textContent=name;trimError('');$('trimStatus').textContent='Saving your recording…';
 renderTrim();$('analysisDialog').close();
 if(!$('trimDialog').open)$('trimDialog').showModal();
 $('trimHeading').focus();
}
async function openTrimEditor(){
 if(!source||uploading||trimSaving)return;
 const revision=++trimRevision,original=source.original||source;
 const saved=source.pendingTrim||source.trim;
 trimState={source:{id:original.id,name:original.name,seconds:original.seconds,bytes:original.bytes},
  seconds:Number(original.seconds)||0,start:saved?.start||0,end:saved?.end??(Number($('sourceSeconds').value)||Number(original.seconds)||0),peaks:[]};
 $('trimFilename').textContent=original.name;trimError('');$('trimStatus').textContent='Loading waveform…';
 $('trimAudio').src='/api/uploads/'+original.id;$('analysisDialog').close();
 if(!$('trimDialog').open)$('trimDialog').showModal();
 renderTrim();$('trimHeading').focus();
 try{
  const data=await api('/api/uploads/'+original.id+'/waveform');
  if(revision!==trimRevision)return;
  trimState.seconds=data.seconds;trimState.source.seconds=data.seconds;trimState.peaks=data.peaks;
  trimState.end=Math.min(trimState.end||data.seconds,data.seconds);
  trimState.start=Math.min(trimState.start,Math.max(0,trimState.end-Math.min(1,data.seconds)));
  $('trimStatus').textContent='Only the selected audio will be analyzed. Your original recording is kept.';
  renderTrim();
 }catch(e){if(revision===trimRevision){trimError(e.message+' You can still trim using playback.');renderTrim()}}
}
function renderTrim(){
 const t=trimState,ready=!!t?.seconds&&!trimSaving;
 for(const id of ['trimStartSlider','trimEndSlider','trimStartValue','trimEndValue','trimPlay','trimReset','trimAnalyze'])$(id).disabled=!ready;
 $('trimClose').disabled=$('trimBack').disabled=trimSaving;
 $('trimAnalyze').textContent=trimSaving?'Preparing selection…':'Analyze selection';
 $('trimTotal').textContent=t?trimTime(t.seconds):'—';
 $('trimDuration').textContent=t?trimTime(t.end-t.start):'—';
 $('trimBounds').textContent=t?trimTime(t.start)+' – '+trimTime(t.end):'';
 for(const [key,label]of [['start','Start'],['end','End']]){
  const value=t?.[key]||0;
  $( 'trim'+label+'Slider').max=t?.seconds||1;$( 'trim'+label+'Slider').value=value;
  $( 'trim'+label+'Slider').setAttribute('aria-valuetext',trimTime(value));
  $( 'trim'+label+'Value').max=trimDisplayValue(t?.seconds||0);$( 'trim'+label+'Value').value=trimDisplayValue(value);
 }
 $('trimSelection').hidden=!t?.seconds;
 if(t?.seconds){$('trimSelection').style.left=(100*t.start/t.seconds)+'%';$('trimSelection').style.right=(100-100*t.end/t.seconds)+'%'}
 drawTrimWaveform();syncTrimPlayback();
}
function drawTrimWaveform(){
 const canvas=$('trimWaveform'),ctx=canvas.getContext('2d'),t=trimState;
 ctx.clearRect(0,0,canvas.width,canvas.height);
 if(!t?.peaks.length)return;
 const width=canvas.width/t.peaks.length,peak=Math.max(.01,...t.peaks);
 for(let i=0;i<t.peaks.length;i++){
  const time=i/t.peaks.length*t.seconds,height=Math.max(2,t.peaks[i]/peak*190);
  ctx.fillStyle=time>=t.start&&time<=t.end?'#f56eae':'#55545d';
  ctx.fillRect(i*width,(canvas.height-height)/2,Math.max(1,width*.7),height);
 }
}
function changeTrim(key,value,editingId=null){
 if(!trimState||trimSaving||!Number.isFinite(value))return;
 const t=trimState,gap=Math.min(1,t.seconds);
 if(key==='start')t.start=Math.min(Math.max(0,value),t.end-gap);
 else t.end=Math.max(t.start+gap,Math.min(value,t.seconds));
 $('trimAudio').pause();$('trimAudio').currentTime=key==='start'?t.start:Math.max(t.start,t.end-3);
 if(source?.awaitingTrim){source.pendingTrim={start:t.start,end:t.end};saveDraft()}
 const editing=editingId?$(editingId).value:null;
 renderTrim();
 if(editingId)$(editingId).value=editing;
}
function syncTrimPlayback(){
 const audio=$('trimAudio'),playing=!audio.paused;
 $('trimPlay').replaceChildren(icon(playing?'pause':'play'),document.createTextNode(playing?'Pause':'Play selection'));
 $('trimPlay').setAttribute('aria-label',playing?'Pause selection':'Play selection');
 $('trimPosition').textContent=trimTime(audio.currentTime);
 $('trimPlayhead').hidden=!trimState?.seconds;
 if(trimState?.seconds)$('trimPlayhead').style.left=(100*Math.min(audio.currentTime,trimState.seconds)/trimState.seconds)+'%';
}
function trimPlaybackTick(){
 const audio=$('trimAudio');
 if(trimState&&audio.currentTime>=trimState.end){audio.pause();audio.currentTime=trimState.end}
 syncTrimPlayback();
 if(!audio.paused)trimFrame=requestAnimationFrame(trimPlaybackTick);
}
function closeTrimEditor(){
 if(trimSaving)return;
 trimRevision++;$('trimAudio').pause();$('trimDialog').close();
}
async function analyzeTrim(){
 if(!trimState?.seconds||trimSaving)return;
 let start=Number($('trimStartValue').value),end=Number($('trimEndValue').value);
 // Keep the precise bounds when the fields still show their rounded values.
 // Otherwise a whole-recording end rounded up can exceed the source duration.
 if(start===trimDisplayValue(trimState.start))start=trimState.start;
 if(end===trimDisplayValue(trimState.end))end=trimState.end;
 if($('trimStartValue').value===''||$('trimEndValue').value===''||!Number.isFinite(start)||!Number.isFinite(end)||start<0||end>trimState.seconds||end-start<Math.min(1,trimState.seconds)-.001){
  trimError('Choose valid start and end times with at least one second selected.');return;
 }
 trimState.start=start;trimState.end=end;
 const t=trimState,revision=trimRevision;
 trimSaving=true;$('trimAudio').pause();trimError('');$('trimStatus').textContent='Preparing your selected audio…';renderTrim();
 try{
  const clip=await api('/api/uploads/'+t.source.id+'/trim',{method:'POST',body:JSON.stringify({start:t.start,end:t.end})});
  if(revision!==trimRevision)return;
  source={...clip,name:t.source.name,original:t.source,trim:{start:clip.start,end:clip.end},awaitingTrim:false};
  $('sourceSeconds').value=0;sourceAnalysis={key:analysisKey(),status:'missing'};analysisLiveJob=null;
  showSource();saveDraft();trimSaving=false;closeTrimEditor();updateUI();openAnalysisDialog();await checkSourceAnalysis(true);
 }catch(e){trimError(e.message)}finally{trimSaving=false;renderTrim()}
}
function setupTrimEditor(){
 $('trimAnalyze').onclick=analyzeTrim;
 $('trimBack').onclick=$('trimClose').onclick=closeTrimEditor;
 $('trimDialog').addEventListener('cancel',e=>{e.preventDefault();closeTrimEditor()});
 $('trimDialog').addEventListener('close',()=>{$('trimAudio').pause();cancelAnimationFrame(trimFrame)});
 for(const [key,label]of [['start','Start'],['end','End']]){
  $('trim'+label+'Slider').oninput=e=>changeTrim(key,Number(e.target.value));
  $('trim'+label+'Value').oninput=e=>{
   if(!trimState||e.target.value==='')return;
   const value=Number(e.target.value),gap=Math.min(1,trimState.seconds);
   // Let an incomplete typed value remain in the field until it is valid or blurred.
   if(Number.isFinite(value)&&(key==='start'?value>=0&&value<=trimState.end-gap:value>=trimState.start+gap&&value<=trimState.seconds))changeTrim(key,value,e.target.id);
  };
  $('trim'+label+'Value').onchange=e=>{if(e.target.value===''){renderTrim();return}changeTrim(key,Number(e.target.value))};
 }
 $('trimReset').onclick=()=>{if(!trimState)return;trimState.start=0;changeTrim('end',trimState.seconds)};
 $('trimPlay').onclick=async()=>{
  const audio=$('trimAudio');if(!trimState)return;
  if(!audio.paused){audio.pause();return}
  if(audio.currentTime<trimState.start||audio.currentTime>=trimState.end-.05)audio.currentTime=trimState.start;
  try{await audio.play()}catch{trimError('This recording cannot be previewed by your browser. You can still analyze the selection.')}
 };
 $('trimWaveform').onclick=e=>{
  if(!trimState||trimSaving)return;
  const box=e.target.getBoundingClientRect();
  $('trimAudio').currentTime=Math.max(trimState.start,Math.min(trimState.end,(e.clientX-box.left)/box.width*trimState.seconds));syncTrimPlayback();
 };
 const audio=$('trimAudio');
 audio.addEventListener('play',()=>{cancelAnimationFrame(trimFrame);trimPlaybackTick()});
 for(const event of ['pause','seeked','ended','loadedmetadata'])audio.addEventListener(event,syncTrimPlayback);
 audio.addEventListener('timeupdate',()=>{if(trimState&&audio.currentTime>=trimState.end)audio.pause();syncTrimPlayback()});
}
