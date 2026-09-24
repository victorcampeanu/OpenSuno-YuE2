function lyricsQueryFromName(name){
 return (name||'').normalize('NFC').replace(/\.(mp3|wav|flac|m4a|aac|ogg|aiff?|opus)$/i,'')
  .replace(/_/g,' ').replace(/^\s*\d{1,3}[.\- ]+\s*/,'')
  .replace(/\s*[([](?:19|20)\d{2}[)\]]/g,'')
  .replace(/\s*[([](?:official(?: music)? (?:video|audio)|lyrics?(?: video)?|hd|hq|4k)[)\]]/gi,'')
  .replace(/\s+/g,' ').trim().slice(0,200);
}
(() => {
 const dialog=document.createElement('dialog');dialog.className='lyrics-search-dialog';dialog.setAttribute('aria-labelledby','lyricsSearchHeading');
 dialog.innerHTML='<div class="section-head"><h2 id="lyricsSearchHeading">Find Lyrics</h2><button type="button" id="lyricsSearchClose" aria-label="Close Lyrics Search">Close</button></div><div class="lyrics-search-body"><form id="lyricsSearchForm"><div class="lyrics-search-input"><input id="lyricsQuery" required minlength="2" maxlength="200" placeholder="Song title, artist or album" aria-label="Song title, artist or album"><button type="submit" class="secondary">Search</button></div></form><p class="muted">Search online with LRCLIB. Choose a version, preview its lyrics, then use it.</p><p id="lyricsSearchStatus" role="status"></p><div id="lyricsMatches"></div><section id="lyricsPreviewPanel" hidden><h3 id="lyricsPreviewHeading"></h3><pre id="lyricsPreview"></pre></section></div><div id="lyricsSearchFooter" class="lyrics-search-footer" hidden><button id="lyricsUse" type="button" class="secondary">Use Lyrics</button></div>';
 document.body.append(dialog);
 const el=id=>document.getElementById(id);let selected=null,revision=0;
 let lastRecording=null;
 el('findLyrics').onclick=()=>{
  const name=source?.name&&source.name!=='Saved recording'?source.name:el('title').value;
  const recording=JSON.stringify([source?.id||'',name]);
  if(recording!==lastRecording){
   revision++;selected=null;el('lyricsPreviewPanel').hidden=true;el('lyricsSearchFooter').hidden=true;el('lyricsMatches').replaceChildren();el('lyricsSearchStatus').textContent='';
   el('lyricsQuery').value=/^(?:Untitled song|Saved recording)$/i.test(name||'')?'':lyricsQueryFromName(name);
   lastRecording=recording;
  }
  dialog.showModal();el('lyricsQuery').focus();
 };
 el('lyricsSearchClose').onclick=()=>dialog.close();
 dialog.addEventListener('close',()=>{revision++});
 el('lyricsSearchForm').onsubmit=async event=>{
  event.preventDefault();const query=el('lyricsQuery').value.trim();if(query.length<2)return;
  const request=++revision;selected=null;el('lyricsPreviewPanel').hidden=true;el('lyricsSearchFooter').hidden=true;el('lyricsMatches').replaceChildren();el('lyricsSearchStatus').textContent='Searching…';
  try{
   const data=await api('/api/lyrics/search?q='+encodeURIComponent(query));if(request!==revision)return;
   el('lyricsSearchStatus').textContent=data.results.length?data.results.length+' versions found. Choose one to preview.':'No matches. Try a different title or artist.';
   for(const result of data.results){
    const button=document.createElement('button');button.type='button';button.className='lyrics-match';
    const title=document.createElement('strong');title.textContent=result.title+' — '+result.artist;
    const meta=document.createElement('span');const seconds=Math.round(result.duration||0);meta.textContent=[result.album,Math.floor(seconds/60)+':'+String(seconds%60).padStart(2,'0'),result.instrumental?'Instrumental':!result.lyrics?'Lyrics unavailable':''].filter(Boolean).join(' · ');
    button.append(title,meta);button.setAttribute('aria-pressed','false');
    button.onclick=()=>{selected=result;for(const item of el('lyricsMatches').children)item.setAttribute('aria-pressed',String(item===button));el('lyricsPreviewPanel').hidden=false;el('lyricsSearchFooter').hidden=false;el('lyricsPreviewHeading').textContent=result.title+' — '+result.artist;el('lyricsPreview').textContent=result.lyrics||'No lyrics available for this version.';el('lyricsUse').disabled=!result.lyrics||result.lyrics.length>30000;if(result.lyrics.length>30000)el('lyricsSearchStatus').textContent='This version exceeds the 30,000 character limit.'};
    el('lyricsMatches').append(button);
   }
  }catch(error){if(request===revision)el('lyricsSearchStatus').textContent=error.message}
 };
 el('lyricsUse').onclick=()=>{if(!selected?.lyrics)return;el('lyrics').value=selected.lyrics;el('expandedLyrics').value=selected.lyrics;el('instrumental').checked=false;updateUI();saveDraft();dialog.close()};
})();
