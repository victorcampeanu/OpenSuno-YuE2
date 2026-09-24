function osEl(tag,cls,text){
 const node=document.createElement(tag);
 if(cls)node.className=cls;
 if(text!=null)node.textContent=text;
 return node;
}
function osMark(state,index){
 const mark=osEl('span','os-mark',state==='done'?'✓':state==='failed'?'×':state==='running'?'•':String(index??''));
 mark.setAttribute('aria-hidden','true');
 return mark;
}
function osRenderTasks(root,items,{capsules=false,plain=false}={}){
 if(!root)return;
 root.classList.toggle('capsules',!!capsules);
 root.classList.toggle('plain',!!plain);
 root.replaceChildren(...items.map((item,i)=>{
  const row=osEl('div','os-task '+(item.state||'waiting'));
  row.setAttribute('role','listitem');
  row.append(osMark(item.state,plain?null:i+1),osEl('b',null,item.label));
  if(item.meta)row.append(osEl('small',null,item.meta));
  if(item.badge)row.append(osEl('span','os-badge',item.badge));
  return row;
 }));
}
function generationStageOrder(stage){
 const order=['loading','analysis_loading','planning','arranging','music_tokens','synthesis','decoding_audio','complete'];
 const i=order.indexOf(stage);
 return i<0?0:i;
}
/* Four plain-language steps; each one covers the technical stages the worker reports. */
function generationTaskItems(j){
 const p=j?.progress||{},status=j?.status||'';
 const failed=['failed','cancelled','interrupted'].includes(status);
 const complete=status==='complete';
 const running=['starting','running','cancelling'].includes(status);
 const stage=p.stage||(running?'loading':'');
 const idx=generationStageOrder(stage);
 const rows=[
  {label:'Getting ready',from:0,to:1},
  {label:stage==='arranging'?'Arranging the cover':'Composing the music',from:2,to:3},
  {label:'Writing the music',from:4,to:4},
  {label:'Making the audio',from:5,to:7}
 ];
 return rows.map(row=>{
  let state='waiting';
  if(complete||idx>row.to)state='done';
  else if(idx>=row.from)state=failed?'failed':running?'running':'waiting';
  return {label:row.label,state};
 });
}
function syncGenerationChrome(j){
 // A finished song speaks for itself in the library, so the whole panel goes away; it stays for a job that
 // is queued, running, or stopped early, because those are the only states with something left to say.
 const panel=$('generationStatus');
 const running=['starting','running','cancelling'].includes(j?.status);
 if(panel){
  panel.hidden=!j?.id||j.status==='complete';
  panel.classList.toggle('is-live',!!j?.id&&running);
  panel.classList.toggle('is-queued',j?.status==='queued');
 }
 const tasks=$('generationTasks');
 if(tasks){
  osRenderTasks(tasks,generationTaskItems(j),{plain:true});
  tasks.hidden=!j?.id||j.status==='queued';
 }
}

function setupSidebarGlide(){
 const nav=$('sidebarNav');if(!nav||nav.querySelector('.nav-glide'))return;
 const glide=osEl('div','nav-glide');nav.prepend(glide);
 const place=el=>{
  if(!el||el.hidden)return;
  glide.style.setProperty('--glide-y',(el.offsetTop)+'px');
  glide.style.height=el.offsetHeight+'px';
 };
 nav.addEventListener('pointermove',e=>{
  const item=e.target.closest('.nav-item');if(item)place(item);
 });
 nav.addEventListener('pointerleave',()=>place(nav.querySelector('.nav-item.active')));
 nav.addEventListener('focusin',e=>{
  const item=e.target.closest('.nav-item');if(item)place(item);
 });
 nav.addEventListener('focusout',e=>{
  if(!nav.contains(e.relatedTarget))place(nav.querySelector('.nav-item.active'));
 });
 place(nav.querySelector('.nav-item.active'));
}

function setupCommandSearch(){
 const dialog=$('commandSearch'),input=$('commandSearchInput'),list=$('commandSearchResults');
 if(!dialog||!input)return;
 function items(q){
  q=(q||'').trim().toLowerCase();
  const rows=[
   {label:'Create',meta:'New Song',go:()=>navigate('create')},
   {label:'Library',meta:'Your Songs',go:()=>navigate('library')},
   {label:'Projects',meta:'Your projects',go:()=>navigate('projects')},
   {label:'Prompts',meta:'Saved styles',go:()=>navigate('prompts')},
   {label:'Models',meta:'Downloads',go:()=>navigate('models')},
   {label:'Settings',meta:'OpenAI key',go:()=>navigate('settings')}
  ];
  for(const p of (typeof promptCache!=='undefined'?promptCache:[])){
   rows.push({label:p.title,meta:'Saved prompt',go:()=>{if(typeof applySavedPrompt==='function')applySavedPrompt(p)}});
  }
  for(const j of (typeof jobsCache!=='undefined'?jobsCache:[])){
   const name=typeof songTitle==='function'?songTitle(j,versions(j)[0]):j.title||j.id;
   rows.push({label:name,meta:(j.kind==='cover'?'Cover':'Song')+' · '+(j.status||''),go:()=>{navigate('library');inspectSong(j,versions(j)[0])}});
  }
  return rows.filter(row=>!q||row.label.toLowerCase().includes(q)||row.meta.toLowerCase().includes(q)).slice(0,12);
 }
 function render(){
  const rows=items(input.value);
  list.replaceChildren(...rows.map((row,i)=>{
   const b=osEl('button',null,row.label);b.type='button';b.append(osEl('small',null,row.meta));
   if(!i)b.setAttribute('aria-selected','true');
   b.onclick=()=>{dialog.close();row.go()};
   return b;
  }));
  if(!rows.length)list.append(osEl('p','muted','No matching songs or pages.'));
 }
 input.oninput=render;
 document.addEventListener('keydown',e=>{
  if((e.metaKey||e.ctrlKey)&&e.key.toLowerCase()==='k'){e.preventDefault();input.value='';render();dialog.showModal();input.focus()}
 });
}

function setupUIKit(){
 setupSidebarGlide();
 setupCommandSearch();
}
