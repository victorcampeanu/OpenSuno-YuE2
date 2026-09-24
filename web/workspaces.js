let workspaceData={workspaces:[{id:'default',name:'My Project'}],memberships:{}},currentWorkspace='default',saveWorkspace='default';
let selectingSongs=false,selectedSongs=new Set(),visibleWorkspaceSongs=[],lastWorkspaceSelection=null;
function workspaceKey(j,c){return j.id+':'+(c?.index||0)}
function workspaceName(id){return workspaceData.workspaces.find(w=>w.id===id)?.name||workspaceData.workspaces[0]?.name||'My Project'}
function songWorkspace(j,c){
 const id=workspaceData.memberships[workspaceKey(j,c)]||j.request.workspace_id||'default';
 return workspaceData.workspaces.some(w=>w.id===id)?id:'default';
}
function workspaceMatches(j,c){return currentWorkspace==='all'||songWorkspace(j,c)===currentWorkspace}
function saveWorkspacePreferences(){try{localStorage.setItem('opensuno-workspaces',JSON.stringify({current:currentWorkspace,saveTo:saveWorkspace}))}catch{}}
function acceptWorkspaces(data){
 workspaceData=data;
 if(currentWorkspace!=='all'&&!data.workspaces.some(w=>w.id===currentWorkspace))currentWorkspace='default';
 if(!data.workspaces.some(w=>w.id===saveWorkspace))saveWorkspace='default';
 saveWorkspacePreferences();
}
function workspaceOptions(select,items,value){
 const signature=JSON.stringify(items);
 if(select.dataset.options!==signature){
  select.replaceChildren();for(const item of items){const option=document.createElement('option');option.value=item.id;option.textContent=item.name;select.append(option)}
  select.dataset.options=signature;
 }
 select.value=value;
}
function workspaceCounts(){
 const counts=Object.fromEntries(workspaceData.workspaces.map(w=>[w.id,0]));
 for(const j of jobsCache){const candidates=versions(j);for(const c of candidates.length?candidates:[null])counts[songWorkspace(j,c)]++}
 return counts;
}
function fitWorkspaceSelect(){
 const select=$('workspaceSelect'),option=select?.selectedOptions[0];
 if(!select||!option)return;
 const probe=fitWorkspaceSelect.probe||=document.createElement('span');
 const cs=getComputedStyle(select);
 probe.style.cssText='position:absolute;visibility:hidden;white-space:nowrap;pointer-events:none;top:0;left:0';
 probe.style.font=cs.font;
 probe.style.letterSpacing=cs.letterSpacing;
 probe.textContent=option.text;
 if(!probe.isConnected)document.body.append(probe);
 select.style.width=Math.ceil(probe.offsetWidth+20)+'px';
}
function renderWorkspaceControls(){
 const counts=workspaceCounts(),total=Object.values(counts).reduce((a,b)=>a+b,0);
 workspaceOptions($('workspaceSelect'),[{id:'all',name:'All Projects · '+total},...workspaceData.workspaces.map(w=>({id:w.id,name:w.name+' · '+counts[w.id]}))],currentWorkspace);
 workspaceOptions($('saveWorkspace'),workspaceData.workspaces,saveWorkspace);
 fitWorkspaceSelect();
 $('selectSongs').textContent=selectingSongs?'Done':'Select Songs';$('selectSongs').setAttribute('aria-pressed',String(selectingSongs));
 $('workspaceSelection').hidden=!selectingSongs;
 renderProjectList();
 const movable=visibleWorkspaceSongs.filter(s=>s.job!==active);
 $('selectAllSongs').disabled=!movable.length;
 $('selectAllSongs').checked=!!movable.length&&movable.every(s=>selectedSongs.has(s.key));
 $('selectAllSongs').indeterminate=selectedSongs.size>0&&!$('selectAllSongs').checked;
 $('selectionCount').textContent=selectedSongs.size+' selected';
 $('moveSelectedSongs').disabled=!selectedSongs.size||selectedSongs.size>1000;
 $('moveSelectedSongs').title=selectedSongs.size>1000?'Move up to 1,000 songs at a time.':'';
 $('deleteSelectedSongs').disabled=!selectedSongs.size||selectedSongs.size>1000;
 $('deleteSelectedSongs').title=selectedSongs.size>1000?'Delete up to 1,000 songs at a time.':'';
}
let openedFromProjects=false;
function syncLibraryBack(){
 const back=$('libraryBack');if(!back)return;
 back.hidden=!openedFromProjects;
 back.closest('.workspace-heading')?.classList.toggle('with-back',openedFromProjects);
}
function chooseWorkspace(id){
 currentWorkspace=id;if(id!=='all')saveWorkspace=id;
 selectedSongs.clear();lastWorkspaceSelection=null;hideSongDetails();
 saveWorkspacePreferences();renderLibrary();
}
function openProject(id){
 openedFromProjects=true;chooseWorkspace(id);navigate('library',{fromProject:true});
}
function decorateWorkspaceRow(row,j,c){
 const key=workspaceKey(j,c);visibleWorkspaceSongs.push({key,job:j.id,candidate:c?.index||0});
 if(!selectingSongs)return;
 const box=document.createElement('input');box.type='checkbox';box.className='song-selection';box.checked=selectedSongs.has(key);box.disabled=j.id===active;
 box.setAttribute('aria-label','Select '+songTitle(j,c)+(c?' version '+c.index:''));row.classList.toggle('bulk-selected',box.checked);
 box.onclick=e=>{
  e.stopPropagation();const checked=box.checked,keys=visibleWorkspaceSongs.filter(s=>s.job!==active).map(s=>s.key);
  let targets=[key];if(e.shiftKey&&keys.includes(lastWorkspaceSelection)){const a=keys.indexOf(lastWorkspaceSelection),b=keys.indexOf(key);targets=keys.slice(Math.min(a,b),Math.max(a,b)+1)}
  for(const k of targets)checked?selectedSongs.add(k):selectedSongs.delete(k);
  lastWorkspaceSelection=key;finishWorkspaceRows();
 };
 row.prepend(box);
}
function finishWorkspaceRows(){
 const visible=new Set(visibleWorkspaceSongs.filter(s=>s.job!==active).map(s=>s.key));
 selectedSongs=new Set([...selectedSongs].filter(k=>visible.has(k)));
 // Selection cannot silently include songs hidden by a new search or filter.
 for(const row of $('library').querySelectorAll('[data-key]')){const box=row.querySelector('.song-selection');if(box){box.checked=selectedSongs.has(row.dataset.key);row.classList.toggle('bulk-selected',box.checked)}}
 renderWorkspaceControls();
}
function workspaceDialog(title){
 const dialog=document.createElement('dialog');dialog.className='song-dialog workspace-dialog';dialog.setAttribute('aria-label',title);
 const heading=document.createElement('div');heading.className='section-head';const h=document.createElement('h2');h.textContent=title;
 const close=document.createElement('button');close.className='icon-button';close.type='button';close.setAttribute('aria-label','Close '+title.toLowerCase());close.append(icon('x-mark'));close.onclick=()=>dialog.close();heading.append(h,close);dialog.append(heading);
 const error=document.createElement('p');error.className='workspace-error';error.setAttribute('role','alert');error.hidden=true;dialog.append(error);
 dialog.showError=e=>{error.textContent=e.message||e;error.hidden=false};
 dialog.addEventListener('close',()=>dialog.remove(),{once:true});document.body.append(dialog);return dialog;
}
function askWorkspaceName(workspace=null){return new Promise(resolve=>{
 const dialog=workspaceDialog(workspace?'Rename Project':'New Project'),form=document.createElement('form');
 const input=document.createElement('input');input.maxLength=80;input.required=true;input.value=workspace?.name||'';input.setAttribute('aria-label','Project name');form.append(input);
 const actions=document.createElement('div');actions.className='song-dialog-actions';const cancel=document.createElement('button');cancel.type='button';cancel.textContent='Cancel';cancel.onclick=()=>dialog.close();const submit=document.createElement('button');submit.type='submit';submit.className='primary';submit.textContent=workspace?'Save':'Create project';actions.append(cancel,submit);form.append(actions);dialog.append(form);
 let result=null;dialog.addEventListener('close',()=>resolve(result),{once:true});
 form.onsubmit=async e=>{e.preventDefault();if(!input.value.trim()){dialog.showError('Enter a project name.');return}submit.disabled=true;
  try{result=await api('/api/workspaces'+(workspace?'/'+workspace.id:''),{method:workspace?'PATCH':'POST',body:JSON.stringify({name:input.value.trim()})});await refreshLibrary();dialog.close()}catch(error){dialog.showError(error)}finally{submit.disabled=false}
 };
 dialog.showModal();input.focus();input.select();
})}
function confirmDeleteWorkspace(workspace){return new Promise(resolve=>{
 const dialog=workspaceDialog('Delete Project?'),p=document.createElement('p');p.textContent='Delete “'+workspace.name+'”? Its songs will move to “'+workspaceName('default')+'”. All audio and generation details are kept.';dialog.append(p);
 const actions=document.createElement('div');actions.className='song-dialog-actions';const cancel=document.createElement('button');cancel.textContent='Cancel';cancel.onclick=()=>dialog.close();const submit=document.createElement('button');submit.className='danger-action';submit.textContent='Delete Project';actions.append(cancel,submit);dialog.append(actions);
 let deleted=false;dialog.addEventListener('close',()=>resolve(deleted),{once:true});submit.onclick=async()=>{submit.disabled=true;try{await api('/api/workspaces/'+workspace.id,{method:'DELETE'});deleted=true;await refreshLibrary();dialog.close()}catch(e){dialog.showError(e)}finally{submit.disabled=false}};
 dialog.showModal();cancel.focus();
})}
function renderProjectList(){
 const list=$('projectList');if(!list)return;
 const q=($('searchProjects')?.value||'').trim().toLowerCase(),counts=workspaceCounts();
 const items=workspaceData.workspaces.filter(w=>!q||w.name.toLowerCase().includes(q));
 list.replaceChildren();
 for(const workspace of items){
  const row=document.createElement('div');row.className='project-row';
  const open=document.createElement('button');open.className='project-open';open.type='button';open.append(icon('folder'));
  const copy=document.createElement('span'),name=document.createElement('b'),meta=document.createElement('small');
  const n=counts[workspace.id]||0;
  name.textContent=workspace.name;meta.textContent=n+' '+(n===1?'song':'songs')+(workspace.id==='default'?' · Default':'');
  copy.append(name,meta);open.append(copy);open.onclick=()=>openProject(workspace.id);
  const actions=document.createElement('div');actions.className='project-actions';
  const rename=document.createElement('button');rename.className='text-button';rename.type='button';rename.textContent='Rename';rename.setAttribute('aria-label','Rename '+workspace.name);
  rename.onclick=async e=>{e.stopPropagation();await askWorkspaceName(workspace);renderProjectList()};
  actions.append(rename);
  if(workspace.id!=='default'){
   const remove=document.createElement('button');remove.className='icon-button workspace-delete';remove.type='button';remove.setAttribute('aria-label','Delete '+workspace.name);remove.append(icon('trash'));
   remove.onclick=async e=>{e.stopPropagation();await confirmDeleteWorkspace(workspace);renderProjectList()};
   actions.append(remove);
  }
  row.append(open,actions);list.append(row);
 }
 if(!list.children.length){
  const empty=document.createElement('div');empty.className='empty-library';empty.append(icon('rectangle-stack'));
  const h=document.createElement('h3');h.textContent=workspaceData.workspaces.length?'No matching projects':'No projects yet';
  const p=document.createElement('p');p.textContent=workspaceData.workspaces.length?'Try another search.':'Create a project to keep songs together.';
  empty.append(h,p);list.append(empty);
 }
}
function openWorkspaces(){navigate('projects')}
function openMoveWorkspace(songs){
 if(!songs.length)return;
 const dialog=workspaceDialog('Move to Project'),p=document.createElement('p');p.textContent='Move '+songs.length+' '+(songs.length===1?'song':'songs')+'. Audio files and song settings stay intact.';dialog.append(p);
 const label=document.createElement('label');label.textContent='Destination Project';const select=document.createElement('select');select.setAttribute('aria-label','Destination Project');label.append(select);dialog.append(label);
 // Prefer another project when moving from a single project.
 workspaceOptions(select,workspaceData.workspaces,workspaceData.workspaces.find(w=>w.id!==currentWorkspace)?.id||'default');
 const create=document.createElement('button');create.className='text-button workspace-new-inline';create.textContent='+ New Project';create.onclick=async()=>{const workspace=await askWorkspaceName();if(workspace)workspaceOptions(select,workspaceData.workspaces,workspace.id)};dialog.append(create);
 const actions=document.createElement('div');actions.className='song-dialog-actions';const cancel=document.createElement('button');cancel.textContent='Cancel';cancel.onclick=()=>dialog.close();const move=document.createElement('button');move.className='primary';move.textContent='Move '+songs.length+' '+(songs.length===1?'song':'songs');actions.append(cancel,move);dialog.append(actions);
 move.onclick=async()=>{move.disabled=true;try{
  const target=select.value,result=await api('/api/workspaces/move',{method:'POST',body:JSON.stringify({workspace_id:target,songs:songs.map(({job,candidate})=>({job,candidate}))})});
  selectedSongs.clear();selectingSongs=false;await refreshLibrary();if(detailJob)hideSongDetails()
  dialog.close();showDownloadNotice('Moved '+result.moved+' '+(result.moved===1?'song':'songs')+' to “'+workspaceName(target)+'”.');
 }catch(e){dialog.showError(e)}finally{move.disabled=false}};
 dialog.showModal();select.focus();
}
async function deleteSelectedSongs(){
 const songs=visibleWorkspaceSongs.filter(s=>selectedSongs.has(s.key));
 if(!songs.length)return;
 const keys=new Set(songs.map(s=>s.job+':'+s.candidate));
 const btn=$('deleteSelectedSongs');btn.disabled=true;
 try{
  for(const s of songs)await api('/api/jobs/'+s.job+'/songs/'+s.candidate,{method:'DELETE'});
  selectedSongs.clear();selectingSongs=false;
  if(playing&&keys.has(playing.key)){const audio=$('playerAudio');playing=null;audio.pause();audio.removeAttribute('src');audio.load();syncPlayer()}
  await refreshLibrary();
  if(selected&&!jobsCache.some(j=>j.id===selected)){clearTimeout(timer);selected=active;lastStatus='';$('status').textContent='Ready to create';$('elapsed').textContent='';$('progress').style.width='0';showLive({});if(active)poll()}
  if(detailJob){const dkey=detailJob.id+':'+(detailCandidate?.index||0);if(keys.has(dkey)||!jobsCache.some(j=>j.id===detailJob.id))hideSongDetails()}
  showDownloadNotice(songs.length===1?'Song moved to trash.':songs.length+' songs moved to trash.');
 }catch(e){notice(e.message,true);await refreshLibrary()}
}
function setupWorkspaces(){
 try{const saved=JSON.parse(localStorage.getItem('opensuno-workspaces'));currentWorkspace=saved?.current||'default';saveWorkspace=saved?.saveTo||'default'}catch{}
 $('workspaceSelect').onchange=()=>{openedFromProjects=false;chooseWorkspace($('workspaceSelect').value);syncLibraryBack()};
 // Projects change from other tabs or Studios rarely; pick those up when this tab comes back instead of polling.
 document.addEventListener('visibilitychange',()=>{if(!document.hidden&&config)refreshLibrary().catch(()=>{})});
 $('libraryBack').onclick=()=>navigate('projects');
 $('searchProjects').oninput=renderProjectList;
 $('projectNew').onclick=async()=>{const workspace=await askWorkspaceName();if(workspace)openProject(workspace.id)};
 $('saveWorkspace').onchange=()=>{saveWorkspace=$('saveWorkspace').value;saveWorkspacePreferences()};
 $('selectSongs').onclick=()=>{selectingSongs=!selectingSongs;if(!selectingSongs)selectedSongs.clear();lastWorkspaceSelection=null;renderLibrary()};
 $('selectAllSongs').onchange=()=>{selectedSongs=$('selectAllSongs').checked?new Set(visibleWorkspaceSongs.filter(s=>s.job!==active).map(s=>s.key)):new Set();finishWorkspaceRows()};
 $('clearSongSelection').onclick=()=>{selectedSongs.clear();selectingSongs=false;renderLibrary()};
 $('moveSelectedSongs').onclick=()=>openMoveWorkspace(visibleWorkspaceSongs.filter(s=>selectedSongs.has(s.key)));
 $('deleteSelectedSongs').onclick=deleteSelectedSongs;
}
