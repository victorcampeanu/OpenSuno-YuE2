let promptCache=[];

function promptPayload(){
 const r=request('generate');
 return {
  style:$('style').value,
  settings:{
   instrumental:!!$('instrumental').checked,
   hook_melody:!!$('instrumental').checked&&!!$('hookMelody').checked,
   voice:['male','female','duet'].includes($('voice').value)?$('voice').value:'any',
   cot:['full','melody','off'].includes($('cot').value)?$('cot').value:'full',
   candidates:Number($('candidates').value)||1,
   cfg_scale:Number($('cfg').value),
   steps:Number($('steps').value),
   semantic_sampling:r.semantic_sampling,
   abc_sampling:r.abc_sampling,
   lora:r.lora||'',lora_strength:r.lora_strength??1,
   sound_lora:r.sound_lora||'',sound_lora_strength:r.sound_lora_strength??1
  }
 };
}

function promptTitleGuess(){
 const style=$('style').value.trim().replace(/\s+/g,' ');
 return style?style.slice(0,80):'';
}

function promptMeta(p){
 const s=p.settings||{};
 const voice={any:'Any voice',male:'Male',female:'Female',duet:'Duet'}[s.voice]||'Any voice';
 const plan={full:'Melody and chords',melody:'Melody only',off:'No plan'}[s.cot]||'Melody and chords';
 const parts=[voice,plan];
 if(s.instrumental)parts.push(s.hook_melody?'Instrumental · chorus melody':'Instrumental');
 return parts.join(' · ');
}

function applySavedPrompt(p){
 const current=request();
 mode='create';
 if(typeof compositionDraft!=='undefined')compositionDraft=null;
 restore({
  ...current,
  kind:'generate',
  style:p.style||'',
  instrumental:!!p.settings?.instrumental,
  hook_melody:!!p.settings?.hook_melody,
  voice:p.settings?.voice||'any',
  cot:p.settings?.cot||'full',
  candidates:p.settings?.candidates||1,
  cfg_scale:p.settings?.cfg_scale??1.2,
  steps:p.settings?.steps??2,
  semantic_sampling:p.settings?.semantic_sampling||current.semantic_sampling,
  abc_sampling:p.settings?.abc_sampling||current.abc_sampling,
  lora:p.settings?.lora||'',lora_strength:p.settings?.lora_strength??1,
  sound_lora:p.settings?.sound_lora||'',sound_lora_strength:p.settings?.sound_lora_strength??1,
  edit_id:''
 });
 saveDraft();navigate('create');notice('Applied “'+p.title+'”. Edit and create when you are ready.');
}

async function copySavedPrompt(p){
 const text=(p.style||'').trim();
 if(!text){notice('This prompt has no style text to copy.',true);return}
 try{
  if(navigator.clipboard?.writeText)await navigator.clipboard.writeText(text);
  else throw new Error('clipboard');
  notice('Copied “'+p.title+'”.');
 }catch{
  const ta=document.createElement('textarea');ta.value=text;document.body.append(ta);ta.select();
  const ok=document.execCommand('copy');ta.remove();
  notice(ok?'Copied “'+p.title+'”.':'Could not copy. Select the style text and copy it manually.',!ok);
 }
}

function promptDialog(title){
 if(typeof workspaceDialog==='function')return workspaceDialog(title);
 const dialog=document.createElement('dialog');dialog.className='song-dialog workspace-dialog';dialog.setAttribute('aria-label',title);
 const heading=document.createElement('div');heading.className='section-head';const h=document.createElement('h2');h.textContent=title;
 const close=document.createElement('button');close.className='icon-button';close.type='button';close.setAttribute('aria-label','Close');close.append(icon('x-mark'));close.onclick=()=>dialog.close();heading.append(h,close);dialog.append(heading);
 const error=document.createElement('p');error.className='workspace-error';error.setAttribute('role','alert');error.hidden=true;dialog.append(error);
 dialog.showError=e=>{error.textContent=e.message||e;error.hidden=false};
 dialog.addEventListener('close',()=>dialog.remove(),{once:true});document.body.append(dialog);return dialog;
}

function askPromptTitle(prompt=null){return new Promise(resolve=>{
 const dialog=promptDialog(prompt?'Rename Prompt':'Save Prompt'),form=document.createElement('form');
 const label=document.createElement('label');label.textContent='Title';
 const input=document.createElement('input');input.maxLength=80;input.required=true;input.value=prompt?.title||promptTitleGuess();
 const help=document.createElement('p');help.className='muted';help.textContent=prompt?'Only the name changes. Style and settings stay as they were.':'Saves the current style and Controls (plan, voice, LoRA, versions, sliders).';
 label.append(input);form.append(label,help);
 const actions=document.createElement('div');actions.className='song-dialog-actions';
 const cancel=document.createElement('button');cancel.type='button';cancel.textContent='Cancel';cancel.onclick=()=>dialog.close();
 const submit=document.createElement('button');submit.type='submit';submit.className='primary';submit.textContent=prompt?'Save':'Save Prompt';
 actions.append(cancel,submit);form.append(actions);dialog.append(form);
 let result=null;dialog.addEventListener('close',()=>resolve(result),{once:true});
 form.onsubmit=async e=>{
  e.preventDefault();const title=input.value.trim();
  if(!title){dialog.showError('Enter a title.');return}
  submit.disabled=true;
  try{
   if(prompt)result=await api('/api/prompts/'+prompt.id,{method:'PATCH',body:JSON.stringify({title})});
   else result=await api('/api/prompts',{method:'POST',body:JSON.stringify({title,...promptPayload()})});
   await refreshPrompts();dialog.close();
   if(!prompt)notice('Saved “'+result.title+'”.');
  }catch(error){dialog.showError(error)}finally{submit.disabled=false}
 };
 dialog.showModal();input.focus();input.select();
})}

function confirmDeletePrompt(prompt){return new Promise(resolve=>{
 const dialog=promptDialog('Delete Prompt?'),p=document.createElement('p');
 p.textContent='Delete “'+prompt.title+'”? This does not change the Create form.';dialog.append(p);
 const actions=document.createElement('div');actions.className='song-dialog-actions';
 const cancel=document.createElement('button');cancel.textContent='Cancel';cancel.onclick=()=>dialog.close();
 const submit=document.createElement('button');submit.className='danger-action';submit.textContent='Delete Prompt';
 actions.append(cancel,submit);dialog.append(actions);
 let deleted=false;dialog.addEventListener('close',()=>resolve(deleted),{once:true});
 submit.onclick=async()=>{submit.disabled=true;try{await api('/api/prompts/'+prompt.id,{method:'DELETE'});deleted=true;await refreshPrompts();dialog.close()}catch(e){dialog.showError(e)}finally{submit.disabled=false}};
 dialog.showModal();cancel.focus();
})}

let promptMenu=null,promptMenuAnchor=null;
function closePromptMenu(focus=false){
 const anchor=promptMenuAnchor;
 promptMenu?.remove();promptMenu=null;promptMenuAnchor=null;
 anchor?.setAttribute('aria-expanded','false');if(focus&&anchor?.isConnected)anchor.focus();
}
function openPromptMenu(anchor,p){
 if(promptMenuAnchor===anchor){closePromptMenu(true);return}
 closePromptMenu();
 if(typeof closeSongMenu==='function')closeSongMenu();
 const menu=document.createElement('div');menu.className='song-menu';menu.setAttribute('role','menu');menu.setAttribute('aria-label','Prompt options');
 promptMenu=menu;promptMenuAnchor=anchor;anchor.setAttribute('aria-expanded','true');
 const item=(label,action,danger)=>{
  const button=document.createElement('button');button.type='button';button.setAttribute('role','menuitem');button.textContent=label;
  if(danger)button.className='danger';
  button.onclick=()=>{closePromptMenu();action()};
  return button;
 };
 menu.append(item('Rename',()=>askPromptTitle(p)));
 const line=document.createElement('hr');line.className='song-menu-sep';menu.append(line,item('Delete',()=>confirmDeletePrompt(p),true));
 document.body.append(menu);
 const rect=anchor.getBoundingClientRect();
 menu.style.left=Math.max(8,Math.min(rect.right-menu.offsetWidth,innerWidth-menu.offsetWidth-8))+'px';
 menu.style.top=(rect.bottom+menu.offsetHeight+8>innerHeight?Math.max(8,rect.top-menu.offsetHeight-4):rect.bottom+4)+'px';
 menu.querySelector('button')?.focus();
 menu.onkeydown=e=>{
  if(['ArrowDown','ArrowUp','Home','End'].includes(e.key)){
   e.preventDefault();const items=[...menu.querySelectorAll('button')];if(!items.length)return;
   let i=items.indexOf(document.activeElement);
   i=e.key==='Home'?0:e.key==='End'?items.length-1:(i+(e.key==='ArrowDown'?1:-1)+items.length)%items.length;
   items[i].focus();
  }else if(e.key==='Tab')closePromptMenu();
 };
}

function renderPrompts(){
 closePromptMenu();
 const list=$('promptList');if(!list)return;
 const query=($('searchPrompts')?.value||'').trim().toLowerCase();
 const items=promptCache.filter(p=>!query||(p.title||'').toLowerCase().includes(query)||(p.style||'').toLowerCase().includes(query));
 if($('promptCount'))$('promptCount').textContent=String(promptCache.length);
 if(!items.length){
  const empty=document.createElement('div');empty.className='empty-library';
  empty.append(icon('document-text'));
  const h=document.createElement('h3');h.textContent=promptCache.length?'No matching prompts':'No saved prompts yet';
  const p=document.createElement('p');p.textContent=promptCache.length?'Try a different search.':'Write a style, then save it with a title. Copy or apply it later from here.';
  empty.append(h,p);list.replaceChildren(empty);return;
 }
 list.replaceChildren(...items.map(p=>{
  const row=document.createElement('article');row.className='prompt-row';row.dataset.id=p.id;
  const copy=document.createElement('div');copy.className='prompt-copy';
  const title=document.createElement('h3');title.textContent=p.title;
  const preview=document.createElement('p');preview.className='prompt-preview';preview.textContent=(p.style||'').trim()||'No style text — settings only.';
  const meta=document.createElement('small');meta.className='prompt-meta';meta.textContent=promptMeta(p);
  copy.append(title,preview,meta);
  const actions=document.createElement('div');actions.className='prompt-actions';
  const copyBtn=document.createElement('button');copyBtn.type='button';copyBtn.className='secondary';copyBtn.append(icon('copy'),document.createTextNode('Copy'));copyBtn.setAttribute('aria-label','Copy '+p.title);copyBtn.onclick=()=>copySavedPrompt(p);
  const applyBtn=document.createElement('button');applyBtn.type='button';applyBtn.className='secondary prompt-apply';applyBtn.textContent='Apply';applyBtn.setAttribute('aria-label','Apply '+p.title);applyBtn.onclick=()=>applySavedPrompt(p);
  const more=document.createElement('button');more.type='button';more.className='icon-button';more.setAttribute('aria-label','More options for '+p.title);more.setAttribute('aria-haspopup','menu');more.setAttribute('aria-expanded','false');more.append(icon('ellipsis-horizontal'));more.onclick=e=>{e.stopPropagation();openPromptMenu(more,p)};
  actions.append(copyBtn,applyBtn,more);
  row.append(copy,actions);
  return row;
 }));
}

async function refreshPrompts(){
 try{
  const data=await api('/api/prompts');
  promptCache=Array.isArray(data.prompts)?data.prompts:[];
 }catch{promptCache=[]}
 renderPrompts();
}

function setupPrompts(){
 const save=$('savePrompt');
 if(save)save.onclick=e=>{e.preventDefault();e.stopPropagation();askPromptTitle()};
 $('searchPrompts')?.addEventListener('input',renderPrompts);
 document.addEventListener('pointerdown',e=>{if(promptMenu&&!promptMenu.contains(e.target)&&!promptMenuAnchor?.contains(e.target))closePromptMenu()});
 document.addEventListener('keydown',e=>{if(e.key==='Escape'&&promptMenu){e.preventDefault();closePromptMenu(true)}});
 window.addEventListener('resize',()=>closePromptMenu());
 document.addEventListener('scroll',e=>{if(promptMenu&&!promptMenu.contains(e.target)&&(e.target===document||e.target.contains?.(promptMenuAnchor)))closePromptMenu()},true);
 refreshPrompts();
}
