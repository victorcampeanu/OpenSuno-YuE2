const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const code=fs.readFileSync('web/opensuno-app.js','utf8');
// The readiness check is split over two statements: the analysisReady constant and the button state.
const analysis=code.match(/const analysisReady=[^;]+;/)[0];
const expression=code.match(/\$\('generate'\)\.disabled=([^;]+);/)[1];
const ready={active:null,uploading:false,available:true,mode:'cover',source:{awaitingTrim:false},
 sourceAnalysis:{key:'ready',status:'ready'},analysisKey:()=> 'ready',config:{downloads:{status:'downloading',model:'8bit'}}};
const disabled=overrides=>vm.runInNewContext(analysis+expression,{...ready,...overrides});
assert.equal(disabled({}),false,'Ready BF16 covers must work while another model downloads');
assert.equal(disabled({mode:'create'}),false,'Ready text-to-music generation must also work');
assert.equal(disabled({available:false}),true,'A missing selected model must still block generation');
assert.equal(disabled({active:'another-job'}),false,'A running job queues the next one instead of blocking it');
assert.equal(disabled({uploading:true}),true,'Nothing starts while a recording is still uploading');
assert.equal(disabled({source:null}),true,'A cover needs a recording');
assert.equal(disabled({source:{awaitingTrim:true}}),true,'Uploaded recordings must still be confirmed');
assert.equal(disabled({sourceAnalysis:{key:'ready',status:'running'}}),true,'Cover analysis must still finish');
console.log('Model availability and generation prerequisites: passed');
