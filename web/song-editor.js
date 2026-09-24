/* Composition editing stays internal; users review style, tempo, harmony and words. */
let compositionDraft=null,songEditor=null;
function el(tag,text,cls){const n=document.createElement(tag);if(text!=null)n.textContent=text;if(cls)n.className=cls;return n}
function editorMessage(text,error=false){$('editStatus').textContent=text;$('editStatus').classList.toggle('error',error)}
function setupSongEditing(){
 const banner=el('div',null,'composition-banner');banner.id='compositionBanner';banner.hidden=true;
 banner.append(el('strong','Using Saved Melody'),el('p','A new recording will follow the saved composition. The original stays in your library.'));
 const clear=el('button','Clear Saved Melody','text-button');clear.type='button';clear.onclick=()=>{compositionDraft=null;saveDraft();updateUI()};banner.append(clear);$('composer').before(banner);
 const dialog=el('dialog',null,'composition-dialog');dialog.id='compositionEditor';dialog.setAttribute('aria-labelledby','editHeading');
 dialog.innerHTML='<div class="editor-heading"><h2 id="editHeading">Edit Song</h2><button type="button" id="closeSongEditor" aria-label="Close Song Editor">×</button></div><p id="editSource" class="muted"></p><div class="chips" id="editPresets"></div><label>Arrangement / Style<textarea id="editStyle" rows="4" maxlength="6000"></textarea></label><div class="two-cols"><label>Tempo (BPM)<input id="editTempo" type="number" min="30" max="300"></label><label>Harmony<select id="editHarmony"><option value="preserve">Keep Chords</option><option value="free">New Chords, Same Melody</option><option value="sevenths">Jazz Seventh Chords</option></select></label></div><details><summary>Lyrics</summary><label>Lyrics<textarea id="editLyrics" rows="5" maxlength="30000"></textarea></label></details><p id="editStatus" role="status"></p><div class="song-dialog-actions"><button type="button" id="applySongEdit" class="primary">Use Edited Composition</button></div>';
 document.body.append(dialog);$('closeSongEditor').onclick=()=>dialog.close();$('applySongEdit').onclick=applySongEdit;
 for(const [name,style]of Object.entries({Jazz:'Rich live jazz ensemble, expressive lead, Rhodes piano, upright bass, drums, saxophone, trumpet and warm strings. Preserve the vocal melody and phrasing.',Orchestral:'Full symphonic orchestra, sweeping strings, bold brass, woodwinds, harp, timpani and expressive lead vocal. Preserve the vocal melody and phrasing.',Electronic:'Electronic arrangement, layered synthesizers, deep bass, driving drums and spacious textures. Preserve the vocal melody and phrasing.'})){
  const b=el('button',name);b.type='button';b.onclick=()=>{$('editStyle').value=style;$('editHarmony').value='free'};$('editPresets').append(b)
 }
 try{for(const key of ['opensuno-auto-lyrics','opensuno-lyrics-language','opensuno-lyrics-task','opensuno-editor-task'])localStorage.removeItem(key)}catch{}
}
function showCompositionDraft(){if($('compositionBanner'))$('compositionBanner').hidden=!compositionDraft}
async function openSongEditor(job,candidate){
 try{
  const index=candidate?.index||1,data=await api('/api/jobs/'+job.id+'/songs/'+index+'/composition');songEditor={job,index,data};
  $('editSource').textContent=data.title+' · Saved melody';$('editStyle').value=data.style;$('editTempo').value=data.bpm;$('editLyrics').value=data.lyrics;$('editHarmony').value='preserve';$('editHarmony').querySelector('[value=sevenths]').disabled=!data.has_chords;
  editorMessage('');$('compositionEditor').showModal();
 }catch(e){showDownloadNotice(e.message,true)}
}
/* No sections are sent: the saved structure stays as it is and only style, tempo, harmony and words change. */
function editOptions(){return {style:$('editStyle').value,lyrics:$('editLyrics').value,bpm:Number($('editTempo').value),harmony:$('editHarmony').value}}
async function applySongEdit(){
 if(!songEditor)return;$('applySongEdit').disabled=true;
 try{
  const draft=await api('/api/jobs/'+songEditor.job.id+'/songs/'+songEditor.index+'/edit',{method:'POST',body:JSON.stringify(editOptions())});
  const r={...songEditor.data.request,kind:'generate',edit_id:draft.id,title:(songEditor.data.title+' · Edit').slice(0,100),style:draft.style,lyrics:draft.lyrics,cot:draft.cot,random_seed:true,seed:songEditor.data.request.seed,candidates:1,audio_id:'',abc:'',semantic_sampling:{...songEditor.data.request.semantic_sampling}};
  r.semantic_sampling.max_tokens=Math.min(18000,Math.ceil(draft.seconds*25)+25);r.semantic_sampling.min_tokens=Math.min(r.semantic_sampling.min_tokens,r.semantic_sampling.max_tokens);
  mode='create';restore(r);compositionDraft=draft;updateUI();saveDraft();navigate('create');$('compositionEditor').close();notice(draft.summary+' Click Create to render a new version.');
 }catch(e){editorMessage(e.message,true)}finally{$('applySongEdit').disabled=false}
}
