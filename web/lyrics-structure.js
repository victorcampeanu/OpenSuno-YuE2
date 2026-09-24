/* Section tags are the only structural instruction YuE2 reads from the lyrics: "(Chorus)", "{hook}" or a bare
   "Verse 2:" line is sung as words instead of shaping the song. This rewrites them into [Tag] form and estimates
   whether the sung syllables fit under the duration cap. Adapted from SongScribe (TheLocalLab, MIT). */
const LyricsStructure = (() => {
 const ALIASES={intro:'Intro',introduction:'Intro',verse:'Verse',vs:'Verse',prechorus:'Pre-Chorus','pre-chorus':'Pre-Chorus','pre chorus':'Pre-Chorus',build:'Pre-Chorus',chorus:'Chorus',hook:'Chorus',refrain:'Chorus',postchorus:'Post-Chorus','post-chorus':'Post-Chorus','post chorus':'Post-Chorus',bridge:'Bridge',middle8:'Bridge','middle 8':'Bridge',interlude:'Interlude',break:'Instrumental',breakdown:'Instrumental',instrumental:'Instrumental',solo:'Instrumental',rap:'Rap',outro:'Outro',ending:'Outro',coda:'Outro',end:'Outro'};
 const BRACKETED=/^\s*[\[({]\s*([^\])}]+?)\s*[\])}][\s:]*$/;
 const BARE=/^\s*([A-Za-z][A-Za-z\- ]{1,14}?)\s*(\d+)?\s*:\s*$/;
 const SYLLABLES_PER_SECOND=3.2;
 function canonical(raw){
  const cleaned=raw.trim().toLowerCase().replace(/\s+/g,' ');
  if(!cleaned)return null;
  if(ALIASES[cleaned])return {tag:ALIASES[cleaned],number:''};
  const m=cleaned.match(/^(.*?)[\s_]*(\d+)$/);
  if(m&&ALIASES[m[1].trim()])return {tag:ALIASES[m[1].trim()],number:m[2]};
  return null;
 }
 function syllables(text){
  let total=0;
  for(let word of text.match(/[A-Za-z\u00C0-\u024F']+/g)||[]){
   word=word.toLowerCase();let groups=(word.match(/[aeiouy\u00C0-\u024F]+/g)||[]).length;
   if(word.endsWith('e')&&groups>1)groups--;
   total+=Math.max(1,groups);
  }
  return total;
 }
 /* Rewrite recognised tags into [Tag] (keeping a number the writer used). Returns the text plus what changed. */
 function normalize(text){
  const out=[],sections=[],changes=[],unknown=[];let lyricLines=0,count=0;
  (text||'').split('\n').forEach((line,index)=>{
   const bracket=line.match(BRACKETED),bare=bracket?null:line.match(BARE);
   let raw=null;
   if(bracket)raw=bracket[1];
   else if(bare&&canonical(bare[1]))raw=bare[1]+(bare[2]?' '+bare[2]:'');
   if(raw!==null){
    const found=canonical(raw);
    if(found){
     const replacement='['+found.tag+(found.number?' '+found.number:'')+']';
     // Case is the writer's choice; only a different shape is a change worth pointing out.
     if(line.trim().toLowerCase()===replacement.toLowerCase()){out.push(line);sections.push(found.tag);return}
     changes.push({line:index+1,from:line.trim(),to:replacement});
     out.push(replacement);sections.push(found.tag);return;
    }
    if(bracket&&line.trim().startsWith('['))unknown.push({line:index+1,text:line.trim()});
   }
   out.push(line);
   if(line.trim()){lyricLines++;count+=syllables(line)}
  });
  return {lyrics:out.join('\n'),sections,changes,unknown,lyricLines,syllables:count};
 }
 /* Sung duration is reported as a range: a ballad and a rap verse differ by more than 2× at the same word count. */
 function estimate(syllableCount,sectionCount){
  if(syllableCount<=0)return {low:0,mid:0,high:0};
  const mid=syllableCount/SYLLABLES_PER_SECOND+sectionCount*4;
  return {low:mid*.65,mid,high:mid*1.6};
 }
 const clock=s=>Math.floor(s/60)+':'+String(Math.round(s%60)).padStart(2,'0');
 function check(text,capSeconds){
  const result=normalize(text),warnings=[];
  if(result.changes.length)warnings.push({kind:'tags',text:result.changes.length+(result.changes.length===1?' section tag is':' section tags are')+' not in the [Tag] form the model reads, e.g. '+result.changes[0].from+' → '+result.changes[0].to+'.'});
  const fit=estimate(result.syllables,result.sections.length);result.estimate=fit;
  if(capSeconds&&fit.low>0&&capSeconds<fit.low)warnings.push({kind:'fit',text:'These lyrics need roughly '+clock(fit.low)+'–'+clock(fit.high)+' of singing; the '+clock(capSeconds)+' cap will cut or rush them.'});
  return {...result,warnings};
 }
 return {ALIASES,canonical,syllables,normalize,estimate,check};
})();
if(typeof module!=='undefined')module.exports=LyricsStructure;

/* The note under the lyrics box: what would go wrong, with a one-click fix for the tags. */
function setupLyricsStructure(){
 const lyrics=$('lyrics');if(!lyrics)return;
 const note=document.createElement('p');note.id='lyricsStructure';note.className='muted warning-text lyrics-structure';note.setAttribute('role','status');note.hidden=true;
 const text=document.createElement('span'),fix=document.createElement('button');fix.type='button';fix.className='text-button';fix.textContent='Fix Tags';
 note.append(text,fix);$('lyricsGroup').append(note);
 function refresh(){
  const instrumental=$('instrumental').checked;
  const cap=instrumental?0:Number($('duration').value)||0;
  const report=LyricsStructure.check(lyrics.value,cap);
  const shown=report.warnings.filter(w=>!(instrumental&&w.kind==='fit'));
  note.hidden=!shown.length||$('lyricsGroup').hidden;
  text.textContent=shown.map(w=>w.text).join(' ');
  fix.hidden=!shown.some(w=>w.kind==='tags');
 }
 fix.onclick=()=>{
  const fixed=LyricsStructure.normalize(lyrics.value).lyrics;
  if(fixed!==lyrics.value){lyrics.value=fixed;lyrics.dispatchEvent(new Event('input',{bubbles:true}))}
  refresh();lyrics.focus();
 };
 lyrics.addEventListener('input',refresh);
 $('duration').addEventListener('change',refresh);
 $('instrumental').addEventListener('change',refresh);
 window.refreshLyricsStructure=refresh;
 refresh();
}
