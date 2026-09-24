/* Keep native selects as the form state; show one shared, themed popup. */
(() => {
 let opened=null,serial=0;
 const enhanced=new WeakSet();
 const enabled=option=>!option.disabled&&!option.hidden&&!option.closest('optgroup')?.disabled;
 function close(focus=false){
  if(!opened)return;
  const {select,popup}=opened;opened=null;
  select.setAttribute('aria-expanded','false');select.removeAttribute('aria-controls');select.removeAttribute('aria-activedescendant');
  if(popup.matches(':popover-open'))popup.hidePopover();popup.remove();
  if(focus&&select.isConnected&&!select.disabled)select.focus({preventScroll:true});
 }
 function highlight(index,scroll=false){
  if(!opened)return;
  opened.active=index;
  for(const row of opened.list.querySelectorAll('[role=option]'))row.classList.toggle('is-active',Number(row.dataset.index)===index);
  const row=opened.list.querySelector('[data-index="'+index+'"]');
  if(row){opened.select.setAttribute('aria-activedescendant',row.id);opened.search?.setAttribute('aria-activedescendant',row.id);if(scroll)row.scrollIntoView({block:'nearest'})}
  else {opened.select.removeAttribute('aria-activedescendant');opened.search?.removeAttribute('aria-activedescendant')}
 }
 function choose(index){
  if(!opened)return;
  const select=opened.select,option=select.options[index];if(!option||!enabled(option))return;
  const changed=select.selectedIndex!==index;select.selectedIndex=index;close(true);
  if(changed){select.dispatchEvent(new Event('input',{bubbles:true}));select.dispatchEvent(new Event('change',{bubbles:true}))}
 }
 function position(){
  if(!opened)return;
  const {select,popup}=opened,r=select.getBoundingClientRect(),gap=6,edge=8;
  const minWidth=Number(select.dataset.popupMinWidth)||230;
  const maxHeight=Number(select.dataset.popupMaxHeight)||350;
  const width=Math.min(Math.max(r.width,minWidth),Math.max(0,innerWidth-2*edge));
  const below=innerHeight-r.bottom-gap-edge,above=r.top-gap-edge,up=below<200&&above>below;
  popup.style.inset='auto';popup.style.margin='0';
  popup.style.width=width+'px';
  popup.style.maxHeight=Math.max(90,Math.min(maxHeight,up?above:below))+'px';
  popup.style.left=Math.max(edge,Math.min(r.left,innerWidth-width-edge))+'px';
  popup.style.top=(up?Math.max(edge,r.top-gap-popup.getBoundingClientRect().height):Math.max(edge,r.bottom+gap))+'px';
 }
 function render(){
  if(!opened)return;
  const s=opened,query=(s.search?.value||'').trim().toLocaleLowerCase();s.list.replaceChildren();s.indices=[];
  let group=null;
  Array.from(s.select.options).forEach((option,index)=>{
   const parent=option.closest('optgroup');
   const haystack=[option.label||option.textContent,option.title,option.dataset.trigger,option.dataset.branch,option.dataset.detail,parent?.label].filter(Boolean).join(' ').toLocaleLowerCase();
   if(option.hidden||(query&&!haystack.includes(query)))return;
   if(parent&&parent!==group){const label=document.createElement('div');label.className='dropdown-group';label.textContent=parent.label;s.list.append(label)}group=parent;
   const row=document.createElement('div');row.id=s.list.id+'-'+index;row.className='dropdown-option';row.dataset.index=index;
   if(option.dataset.detail)row.classList.add('has-detail');
   row.setAttribute('role','option');row.setAttribute('aria-selected',String(index===s.select.selectedIndex));row.setAttribute('aria-disabled',String(!enabled(option)));
   const text=document.createElement('span');text.className='dropdown-copy';
   const name=document.createElement('span');name.className='dropdown-label';name.textContent=option.label||option.textContent;text.append(name);
   if(option.dataset.detail){const detail=document.createElement('span');detail.className='dropdown-detail';detail.textContent=option.dataset.detail;text.append(detail)}
   row.append(text);
   const right=document.createElement('span');right.className='dropdown-meta';
   if(option.dataset.branch){const chip=document.createElement('span');chip.className='dropdown-chip';chip.textContent=option.dataset.branch;right.append(chip)}
   if(option.dataset.trigger){const chip=document.createElement('span');chip.className='dropdown-chip';chip.textContent=option.dataset.trigger;right.append(chip)}
   if(index===s.select.selectedIndex){const check=document.createElement('span');check.className='dropdown-check';check.textContent='✓';check.setAttribute('aria-hidden','true');right.append(check)}
   if(right.childNodes.length)row.append(right)
   if(enabled(option)){s.indices.push(index);row.addEventListener('pointermove',()=>highlight(index));row.addEventListener('click',()=>choose(index))}
   s.list.append(row);
  });
  if(!s.list.children.length){const empty=document.createElement('p');empty.className='dropdown-empty';empty.textContent='No matches';s.list.append(empty)}
  highlight(s.indices.includes(s.active)?s.active:s.indices.includes(s.select.selectedIndex)?s.select.selectedIndex:s.indices[0]??-1);
  position();
 }
 function open(select){
  if(select.disabled)return;
  if(opened?.select===select){close(true);return}
  close();
  const popup=document.createElement('div');popup.className='dropdown-popup';popup.setAttribute('popover','manual');
  const list=document.createElement('div');list.className='dropdown-list';list.id='dropdown-list-'+(++serial);list.setAttribute('role','listbox');
  const name=select.getAttribute('aria-label')||Array.from(select.labels||[]).map(l=>Array.from(l.childNodes).filter(n=>n.nodeType===3).map(n=>n.textContent).join('').trim()).filter(Boolean).join(' ')||'Options';
  list.setAttribute('aria-label',name);
  let search=null;
  if(select.options.length>8||select.dataset.search){
   search=document.createElement('input');search.type='search';search.className='dropdown-search';search.placeholder=select.dataset.searchPlaceholder||'Search options…';search.setAttribute('aria-label','Search '+name);search.setAttribute('role','combobox');search.setAttribute('aria-expanded','true');search.setAttribute('aria-controls',list.id);search.autocomplete='off';
   search.addEventListener('input',render);popup.append(search);
  }
  popup.append(list);(select.closest('dialog')||document.body).append(popup);
  opened={select,popup,list,search,active:select.selectedIndex,indices:[],typeahead:'',typedAt:0};
  select.setAttribute('aria-expanded','true');select.setAttribute('aria-controls',list.id);
  popup.showPopover();render();highlight(opened.active,true);
  if(search)search.focus({preventScroll:true});else select.focus({preventScroll:true});
 }
 function keydown(event){
  const select=event.target.closest?.('select');
  if(!opened){
   if(select&&enhanced.has(select)&&!select.disabled&&['ArrowDown','ArrowUp','Enter',' '].includes(event.key)){event.preventDefault();open(select)}
   return;
  }
  if(event.target!==opened.select&&!opened.popup.contains(event.target))return;
  const s=opened;
  if(event.key==='Escape'){event.preventDefault();event.stopPropagation();close(true);return}
  if(event.key==='Tab'){const fromSearch=event.target===s.search;close(fromSearch);return}
  if(['ArrowDown','ArrowUp','Home','End'].includes(event.key)){
   if(event.target===s.search&&['Home','End'].includes(event.key))return;
   event.preventDefault();const at=s.indices.indexOf(s.active);
   const next=event.key==='Home'?0:event.key==='End'?s.indices.length-1:event.key==='ArrowDown'?Math.min(s.indices.length-1,at+1):Math.max(0,at-1);
   highlight(s.indices[next]??-1,true);return;
  }
  if(event.key==='Enter'||(event.key===' '&&event.target!==s.search)){event.preventDefault();choose(s.active);return}
  if(event.target!==s.search&&event.key.length===1&&!event.ctrlKey&&!event.metaKey&&!event.altKey){
   event.preventDefault();const now=Date.now();s.typeahead=now-s.typedAt>700?event.key:s.typeahead+event.key;s.typedAt=now;
   const match=s.indices.find(i=>s.select.options[i].textContent.toLocaleLowerCase().startsWith(s.typeahead.toLocaleLowerCase()));if(match!=null)highlight(match,true);
  }
 }
 function enhance(select){
  if(enhanced.has(select)||select.multiple||select.size>1)return;
  enhanced.add(select);select.classList.add('themed-select');select.setAttribute('aria-haspopup','listbox');select.setAttribute('aria-expanded','false');
  select.addEventListener('pointerdown',e=>{if(e.button===0&&!select.disabled){e.preventDefault();select.focus({preventScroll:true});open(select)}});
  select.addEventListener('mousedown',e=>{if(e.button===0)e.preventDefault()});
  select.addEventListener('click',e=>{e.preventDefault();if(e.detail===0&&!opened)open(select)});
  select.addEventListener('change',()=>{if(opened?.select===select)close()});
 }
 function scan(node){if(node.nodeType!==1)return;if(node.matches('select'))enhance(node);node.querySelectorAll('select').forEach(enhance)}
 if(!HTMLElement.prototype.showPopover)return; // Older browsers retain functional native menus.
 scan(document.documentElement);
 new MutationObserver(records=>{
  let refresh=false;
  for(const r of records){r.addedNodes.forEach(scan);if(opened&&(r.target===opened.select||opened.select.contains(r.target)))refresh=true}
  if(opened&&(!opened.select.isConnected||opened.select.disabled||!opened.select.getClientRects().length)){close();return}
  if(refresh)render();
 }).observe(document.body,{childList:true,subtree:true,characterData:true,attributes:true,attributeFilter:['disabled','hidden','selected','label']});
 document.addEventListener('keydown',keydown,true);
 document.addEventListener('pointerdown',e=>{if(opened&&e.target!==opened.select&&!opened.popup.contains(e.target))close()},true);
 document.addEventListener('scroll',e=>{if(opened&&!opened.popup.contains(e.target)&&(e.target===document||e.target.contains?.(opened.select)))close()},true);
 document.addEventListener('close',e=>{if(opened&&e.target.contains(opened.select))close()},true);
 window.addEventListener('resize',()=>close());
})();
