const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const elements=new Map();
function element(id){
 if(!elements.has(id))elements.set(id,{id,value:'',textContent:'',style:{},paused:true,currentTime:0,
  classList:{toggle(){}},setAttribute(){},replaceChildren(){},addEventListener(){},
  getContext:()=>({clearRect(){},fillRect(){}}),pause(){this.paused=true},close(){this.open=false}});
 return elements.get(id);
}
let saved,requests=0,analyses=0,lastRequest;
const context=vm.createContext({$:element,source:{awaitingTrim:true},icon:()=>({}),document:{createTextNode:()=>({})},
 saveDraft:()=>{if(context.source.pendingTrim)saved=JSON.parse(JSON.stringify(context.source.pendingTrim))},
 api:async(url,options)=>{requests++;lastRequest=JSON.parse(options.body);return {id:'clip.wav',...lastRequest,seconds:lastRequest.end-lastRequest.start}},
 analysisKey:()=>'',showSource(){},updateUI(){},openAnalysisDialog(){},checkSourceAnalysis:async()=>{analyses++},
 requestAnimationFrame:()=>0,cancelAnimationFrame(){}});
vm.runInContext(fs.readFileSync('web/audio-trimmer.js','utf8'),context);
vm.runInContext("trimState={seconds:14,start:0,end:14,peaks:[]};setupTrimEditor();renderTrim()",context);
function type(id,value){element(id).value=value;element(id).oninput({target:element(id)})}
(async()=>{
 type('trimStartValue','2');type('trimEndValue','10');
 assert.equal(element('trimStartSlider').value,2);
 assert.equal(element('trimEndSlider').value,10);
 assert.deepEqual(saved,{start:2,end:10},'Typed bounds must update and persist before confirmation');
 type('trimEndValue','1');
 await context.analyzeTrim();
 assert.equal(requests,0,'Invalid typed bounds must not submit an old selection');
 assert.match(element('trimStatus').textContent,/valid start and end/);
 element('trimAudio').currentTime=10.2;element('trimAudio').paused=false;
 context.trimPlaybackTick();
 assert.equal(element('trimAudio').paused,true);
 assert.equal(element('trimAudio').currentTime,10,'Preview must stop at the selected out point');
 context.changeTrim('start',20);
 assert.equal(element('trimStartSlider').value,9,'The handles cannot cross');

 function selection(seconds,start=0,end=seconds){
  vm.runInContext(`trimState={source:{id:'original.mp3',name:'Original'},seconds:${seconds},start:${start},end:${end},peaks:[]};renderTrim()`,context);
 }
 // Untouched whole-recording ends must work whether display rounding goes up
 // or down, including recordings shorter than the usual one-second minimum.
 for(const seconds of [242.946939,242.947111,.123456]){
  selection(seconds);
  assert.ok(Number(element('trimEndValue').value)<=Number(element('trimEndValue').max));
  const before=requests;
  await context.analyzeTrim();
  assert.equal(requests,before+1,'Untouched default bounds must submit');
  assert.deepEqual(lastRequest,{start:0,end:seconds},'Keep the exact source endpoint');
  assert.equal(analyses,requests,'Confirmation must proceed to analysis');
 }
 selection(242.946939,2,200);
 element('trimReset').onclick();
 await context.analyzeTrim();
 assert.deepEqual(lastRequest,{start:0,end:242.946939},'Use whole recording must restore the precise endpoint');
 selection(242.946939,1.234567,20.765432);
 await context.analyzeTrim();
 assert.deepEqual(lastRequest,{start:1.234567,end:20.765432},'Rounded fields must not alter unchanged selected bounds');
 for(const [id,value] of [['trimEndValue','242.948'],['trimEndValue',''],['trimEndValue','NaN'],['trimStartValue','-1']]){
  selection(242.946939);
  type(id,value);
  const before=requests;
  await context.analyzeTrim();
  assert.equal(requests,before,'Invalid typed bounds must still be rejected');
  assert.match(element('trimStatus').textContent,/valid start and end/);
 }
 console.log('Trim entry, persistence, validation, preview and precise default bounds: passed');
})().catch(e=>{console.error(e);process.exitCode=1});
