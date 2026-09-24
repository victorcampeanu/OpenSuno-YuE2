const StyleAISettings=(()=>{
 const KEY='opensuno-openai-key';
 const MODEL='opensuno-openai-model';
 const fallback=[{id:'gpt-4o-mini',label:'gpt-4o-mini'},{id:'gpt-4o',label:'gpt-4o'},{id:'gpt-4.1-mini',label:'gpt-4.1-mini'},{id:'gpt-4.1',label:'gpt-4.1'}];
 function get(){
  let model='';
  try{model=localStorage.getItem(MODEL)||''}catch{}
  if(!/^[A-Za-z0-9._:-]{1,80}$/.test(model))model='gpt-4o-mini';
  let apiKey='';
  try{apiKey=localStorage.getItem(KEY)||''}catch{}
  return {apiKey,model};
 }
 function setKey(value){
  value=(value||'').trim();
  try{if(value)localStorage.setItem(KEY,value);else localStorage.removeItem(KEY)}catch{}
 }
 function setModel(value){
  const model=/^[A-Za-z0-9._:-]{1,80}$/.test(value||'')?value:'gpt-4o-mini';
  try{localStorage.setItem(MODEL,model)}catch{}
  return model;
 }
 return {fallback,get,setKey,setModel};
})();

function setupStudioSettings(){
 const key=$('settingsOpenAIKey'),model=$('settingsOpenAIModel'),test=$('settingsTestKey'),keyStatus=$('settingsOpenAIKeyStatus');
 if(!key||!model)return;
 let refreshTimer=0,request=0,loaded=StyleAISettings.fallback.slice();
 function fill(models){
  loaded=models||[];
  const settings=StyleAISettings.get();
  const ordered=(loaded||[]).slice();
  model.replaceChildren();
  for(const item of ordered){
   const option=new Option(item.label||item.id,item.id);
   model.append(option);
  }
  if(settings.model&&![...model.options].some(option=>option.value===settings.model)){
   model.append(new Option(settings.model,settings.model));
  }
  if(settings.model&&[...model.options].some(option=>option.value===settings.model))model.value=settings.model;
  else if(model.options.length&&!settings.model){model.value=model.options[0].value;StyleAISettings.setModel(model.value)}
 }
 function keyMessage(text,kind=''){
  if(!keyStatus)return;
  keyStatus.textContent=text;
  keyStatus.classList.toggle('settings-error',kind==='error');
  keyStatus.classList.toggle('settings-ok',kind==='ok');
 }
 async function refreshModels(){
  const settings=StyleAISettings.get();
  if(!settings.apiKey&&!config?.openai_configured){
   fill(StyleAISettings.fallback);
   return;
  }
  const id=++request;
  try{
   const data=await api('/api/style/models',{method:'POST',body:JSON.stringify({api_key:settings.apiKey})});
   if(id!==request)return;
   fill(data.models);
  }catch(error){
   if(id!==request)return;
   fill(StyleAISettings.fallback);
   keyMessage(error.message,'error');
  }
 }
 function sync(loadModels=true){
  const settings=StyleAISettings.get();
  if(document.activeElement!==key)key.value=settings.apiKey;
  if(!model.options.length)fill(loaded);
  if([...model.options].some(option=>option.value===settings.model))model.value=settings.model;
  if(loadModels)refreshModels();
 }
 key.addEventListener('input',()=>{
  StyleAISettings.setKey(key.value);
  keyMessage('');
  clearTimeout(refreshTimer);
  refreshTimer=setTimeout(refreshModels,700);
 });
 model.addEventListener('change',()=>StyleAISettings.setModel(model.value));
 test?.addEventListener('click',async()=>{
  StyleAISettings.setKey(key.value);
  const settings=StyleAISettings.get();
  if(!settings.apiKey&&!config?.openai_configured){keyMessage('Paste an OpenAI API key first.','error');return}
  test.disabled=true;keyMessage('Checking this key with OpenAI…');
  try{
   await api('/api/style/key',{method:'POST',body:JSON.stringify({api_key:settings.apiKey})});
   keyMessage('This OpenAI key is valid.','ok');
   refreshModels();
  }catch(error){keyMessage(error.message,'error')}
  finally{test.disabled=false}
 });
 setupStudioSettings.sync=sync;
 sync(false);
}

function setupStyleAI(){
 const dialog=document.createElement('dialog');
 dialog.id='styleAI';
 dialog.className='style-builder style-ai';
 dialog.setAttribute('aria-labelledby','styleAIHeading');
 dialog.innerHTML=`
  <header class="builder-header"><h2 id="styleAIHeading" tabindex="-1">Ask AI for a Style</h2><button id="styleAIClose" class="icon-button" type="button" aria-label="Close style AI"><img src="/static/icons/x-mark.svg" alt=""></button></header>
  <div class="builder-body">
   <section class="builder-section ai-ask">
    <textarea id="styleAIAsk" rows="3" maxlength="2000" aria-label="Describe the sound you want" placeholder="late-90s dance pop, bright synths, tight drums, confident belted vocal…"></textarea>
    <div class="ai-ask-row">
     <p class="builder-hint">Mood, era, instruments, vocal character.</p>
     <button id="styleAIGenerate" type="button" class="primary"><img src="/static/icons/sparkles.svg" alt=""><span>Generate</span></button>
    </div>
    <p id="styleAIError" class="builder-error" role="alert" hidden></p>
    <textarea id="styleAIPreview" rows="5" maxlength="6000" aria-label="Suggested style" spellcheck="false" hidden></textarea>
   </section>
  </div>
  <footer id="styleAIFooter" class="builder-footer ai-footer" hidden>
   <div class="builder-apply-mode" role="group" aria-label="How to apply">
    <label><input type="radio" name="styleAIApplyMode" value="replace" checked>Replace Current Prompt</label>
    <label><input type="radio" name="styleAIApplyMode" value="append">Add to Current Prompt</label>
   </div>
   <button id="styleAIApply" type="button" class="primary">Use This Style</button>
  </footer>`;
 document.body.append(dialog);
 const button=document.createElement('button');
 button.id='openStyleAI';
 button.type='button';
 button.className='icon-button expand-lyrics';
 button.setAttribute('aria-label','Ask AI for a Style');
 button.title='Ask AI for a Style';
 button.innerHTML='<img src="/static/icons/sparkles.svg" alt="">';
 const summary=$('style').closest('details').querySelector('summary');
 summary.insertBefore(button,$('expandStyle')||null);
 let generated='',returnToStyle=false;
 const generate=$('styleAIGenerate'),generateLabel=generate.querySelector('span');
 function applyMode(){return dialog.querySelector('input[name="styleAIApplyMode"]:checked').value}
 function showError(text){$('styleAIError').hidden=!text;$('styleAIError').textContent=text||''}
 function current(){return $('styleAIPreview').value.trim()}
 function preview(){
  const text=current(),has=!!generated;
  $('styleAIPreview').hidden=!has;$('styleAIFooter').hidden=!has;
  generateLabel.textContent=has?'Generate again':'Generate';
  if(!has)return;
  const result=StyleBuilder.merge($('style').value,text,applyMode()),over=result.length>6000;
  showError(over?'Together with the current prompt this is over 6,000 characters. Shorten it or choose Replace.':'');
  $('styleAIApply').disabled=!text||over;
 }
 button.onclick=e=>{
  e.preventDefault();e.stopPropagation();returnToStyle=false;
  dialog.querySelector('input[value="replace"]').checked=true;
  showError('');preview();dialog.showModal();$('styleAIAsk').focus();
 };
 $('styleAIClose').onclick=()=>dialog.close();
 dialog.addEventListener('click',event=>{if(event.target===dialog){const r=dialog.getBoundingClientRect();if(event.clientX<r.left||event.clientX>r.right||event.clientY<r.top||event.clientY>r.bottom)dialog.close()}});
 dialog.addEventListener('close',()=>{(returnToStyle?$('style'):button).focus()});
 for(const input of dialog.querySelectorAll('[name="styleAIApplyMode"]'))input.onchange=preview;
 $('styleAIPreview').oninput=preview;
 $('styleAIAsk').addEventListener('keydown',event=>{if(event.key==='Enter'&&(event.metaKey||event.ctrlKey)){event.preventDefault();generate.click()}});
 $('styleAIPreview').addEventListener('keydown',event=>{if(event.key==='Enter'&&(event.metaKey||event.ctrlKey)){event.preventDefault();$('styleAIApply').click()}});
  generate.onclick=async()=>{
  const prompt=$('styleAIAsk').value.trim();
  if(prompt.length<2){showError('Describe the style you want in a few words.');$('styleAIAsk').focus();return}
  const settings=StyleAISettings.get();
  if(!settings.apiKey&&!config?.openai_configured){showError('Add an OpenAI API key in Settings first.');return}
  generate.disabled=true;generate.classList.add('ai-busy');generateLabel.textContent='Writing…';$('styleAIApply').disabled=true;showError('');
  try{
   const data=await api('/api/style/generate',{method:'POST',body:JSON.stringify({prompt,api_key:settings.apiKey,model:settings.model,instrumental:$('instrumental').checked,current_style:$('style').value})});
   generated=(data.style||'').trim();
   if(!generated)throw Error('OpenAI returned an empty style.');
   $('styleAIPreview').hidden=false;
   $('styleAIPreview').value=generated||'';
   preview();
   $('styleAIPreview').focus({preventScroll:true});$('styleAIPreview').setSelectionRange(0,0);
   $('styleAIPreview').scrollIntoView({block:'nearest',behavior:'smooth'});
  }catch(error){showError(error.message);preview()}
  finally{generate.disabled=false;generate.classList.remove('ai-busy');generateLabel.textContent=generated?'Generate again':'Generate'}
 };
 $('styleAIApply').onclick=()=>{
  preview();if($('styleAIApply').disabled)return;
  $('style').value=StyleBuilder.merge($('style').value,current(),applyMode());
  updateUI();saveDraft();returnToStyle=true;dialog.close();
 };
}
