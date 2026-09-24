const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const code=fs.readFileSync('web/opensuno-app.js','utf8');
const body=code.slice(code.indexOf('async function checkSourceAnalysis('),code.indexOf('function changedAnalysisSettings('));
(async()=>{
 let resolveFirst,posts=0,gets=0;
 const context=vm.createContext({URLSearchParams,source:{id:'song.wav'},config:{},analysisBusy:false,uploading:false,
   analysisRetryRequested:false,sourceAnalysis:{key:'recording',status:'failed'},analysisLiveJob:null,active:null,selected:null,lastStatus:'',
   analysisKey:()=> 'recording',analysisOptions:()=>({audio_id:'song.wav',source_seconds:0}),
   $:()=>({checkValidity:()=>true}),notice:()=>{},showAnalysis:()=>{},updateUI:()=>{},poll:()=>{},
   api:async(url,options)=>{if(options?.method==='POST'){posts++;return {status:'starting',job:'retry-job'}}
     if(++gets===1)return new Promise(resolve=>resolveFirst=resolve);
     return {status:'failed',job:'old-job'};}
 });
 vm.runInContext(body,context);
 const first=context.checkSourceAnalysis();
 await context.checkSourceAnalysis(true);
 assert.equal(context.analysisRetryRequested,true);
 resolveFirst({status:'failed',job:'old-job'});await first;
 await new Promise(resolve=>setImmediate(resolve));
 assert.equal(posts,1,'Explicit Retry must survive an in-flight status check');
 assert.equal(context.sourceAnalysis.job,'retry-job');
 assert.equal(context.analysisRetryRequested,false);
 const previousGets=gets;
 context.source.awaitingTrim=true;
 await context.checkSourceAnalysis(true);
 assert.equal(posts,1,'An uploaded recording must wait for trim confirmation');
 assert.equal(gets,previousGets);
 context.source.awaitingTrim=false;
 context.$=id=>({open:id==='trimDialog',checkValidity:()=>true});
 await context.checkSourceAnalysis(true);
 assert.equal(posts,1,'Analysis must not start while the trim editor is open');
 assert.equal(gets,previousGets);
 console.log('Analysis retry race: passed');
})().catch(e=>{console.error(e);process.exitCode=1});
