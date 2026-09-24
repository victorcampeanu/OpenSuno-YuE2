/* Optional prompt composition. Only Apply writes to the music form. The vocal type is not written into the
   style text: Apply sets the song's Voice option (or Instrumental) and the worker adds the one voice line. */
const StyleBuilder = (() => {
 const genres = ['Dance pop','Modern house','Deep house','Tech house','Disco / Nu-disco','Latin pop','Jazz','Jazz-funk','Soul / R&B','Symphonic orchestral','Cinematic','Rock','Hip-hop','Arabic pop','Mahraganat','Romanian manele','Folk','Acoustic pop','Synthwave'];
 const instruments = {
  'Rhythm & Bass':['Acoustic drums','Electronic drums','Claps','Shakers','Congas','Darbuka','Electric bass','Upright bass','Sub-bass'],
  'Keys & Strings':['Piano','Rhodes piano','Organ','Acoustic guitar','Electric guitar','Nylon-string guitar','Strings ensemble','Solo violin','Cello','Harp'],
  'Horns & Winds':['Saxophone','Trumpet','Trombone','French horns','Flute','Clarinet'],
  'Synths & Regional':['Synth lead','Atmospheric pads','Arpeggiated synth','Accordion','Oud','Qanun','Ney','Cimbalom']
 };
 const moods = ['Uplifting','Melancholic','Dreamy','Dark','Romantic','Celebratory','Dramatic','Laid-back'];
 const voices = {'female':'Female lead vocal','male':'Male lead vocal','duet':'Male and female vocal duet','choir':'Choir vocals'};
 const LEAD_VOICES=['male','female','duet'];   // carried by the Voice option, never by the text
 const deliveries = ['Powerful, expressive singing','Warm, soulful singing','Soft, intimate singing','Airy, breathy singing','Operatic singing','Rhythmic rap delivery','Ornamented, melismatic singing'];
 const productions = ['Full, layered arrangement with a clear mix','Polished club production with punchy drums and deep bass','Warm, organic live-band production','Rich orchestral arrangement with wide dynamics','Raw, intimate acoustic production'];
 const intros = ['Short instrumental intro','Atmospheric intro building into the groove','Start immediately with the lead vocal','Start immediately with the main instrumental hook'];
 function empty(){return {preset:'',blendWith:'',blendAmount:35,detail:'full',era:'',texture:'',moodShift:'',genre:'',voice:'',delivery:'',register:'',instruments:[],moods:[],tempoEnabled:false,bpm:120,energy:'',production:'',intro:'',language:'',extra:''}}
 function instrumental(state,current=false){return state.voice==='instrumental'||(!state.voice&&current)}
 function presetPrompt(state,currentInstrumental=false){
  if(!state.preset||typeof StylePresets==='undefined')return '';
  let preset=StylePresets.get(state.preset);if(!preset)return '';
  const other=state.blendWith&&StylePresets.get(state.blendWith);
  if(other&&other.id!==preset.id)preset=StylePresets.blend(preset,other,(Number(state.blendAmount)||0)/100);
  preset=StylePresets.modify(preset,{era:state.era,texture:state.texture,moodShift:state.moodShift});
  const noVocals=instrumental(state,currentInstrumental);
  // Lead voice gender comes from the Voice option; the preset only contributes its vocal texture.
  return StylePresets.prompt(preset,{vocal:noVocals?'instrumental':state.voice==='choir'?'choir':'auto',language:noVocals?'':state.language,detail:state.detail||'full',bpm:!state.tempoEnabled});
 }
 function build(state,currentInstrumental=false){
  const parts=[];const add=value=>{if(value?.trim())parts.push(value.trim())};
  const fromPreset=presetPrompt(state,currentInstrumental);
  add(state.genre);
  if(state.tempoEnabled&&Number.isFinite(Number(state.bpm)))add(Math.max(30,Math.min(240,Math.round(Number(state.bpm))))+' BPM');
  const noVocals=instrumental(state,currentInstrumental);
  // With a preset, language is already inside its line. "Instrumental." and the lead voice are added by the worker.
  if(!noVocals){if(!fromPreset&&state.voice==='choir')add(voices.choir);add(state.delivery);add(state.register?state.register+' vocal register':'');if(state.language?.trim()&&!fromPreset)add(state.language.trim()+' vocals')}
  add((state.instruments||[]).join(', '));add((state.moods||[]).join(', '));
  add(state.energy);add(state.production);
  if(!(noVocals&&state.intro==='Start immediately with the lead vocal'))add(state.intro);
  add(state.extra);
  const sentences=parts.map(part=>/[.!?]$/.test(part)?part:part+'.').join(' ');
  return fromPreset?(sentences?fromPreset+'\n'+sentences:fromPreset):sentences;
 }
 function merge(current,generated,mode){
  if(mode==='replace'||!current.trim())return generated;
  if(current.trim()===generated||current.trimEnd().endsWith('\n\n'+generated))return current;
  return current+'\n\n'+generated;
 }
 return {genres,instruments,moods,voices,LEAD_VOICES,deliveries,productions,intros,empty,instrumental,presetPrompt,build,merge};
})();
if(typeof module!=='undefined')module.exports=StyleBuilder;

function setupStyleBuilder(){
 const dialog=document.createElement('dialog');dialog.id='styleBuilder';dialog.className='style-builder';dialog.setAttribute('aria-labelledby','styleBuilderHeading');
 dialog.innerHTML=`
  <header class="builder-header"><div><h2 id="styleBuilderHeading" tabindex="-1">Build a Style</h2><p>Choose what you want. Leave anything else open.</p></div><button id="builderClose" class="icon-button" type="button" aria-label="Close Style Builder"><img src="/static/icons/x-mark.svg" alt=""></button></header>
  <div class="builder-body">
   <section class="builder-section builder-presets"><div class="builder-section-heading"><h3>Start from a Preset</h3><span id="builderPresetInfo"></span></div>
    <div class="builder-grid"><label>Preset<select id="builderPreset"></select></label><label>Blend With<select id="builderBlendWith"></select></label></div>
    <div id="builderPresetOptions" class="builder-grid builder-preset-options" hidden>
     <label>Detail<select id="builderDetail"><option value="tags">Tags — genre, mood and voice</option><option value="full" selected>Full — adds instruments and production</option><option value="rich">Rich — adds more colour and a scene</option></select></label>
     <label id="builderBlendAmountLabel">Blend Amount <output id="builderBlendAmountValue">35%</output><input type="range" id="builderBlendAmount" min="10" max="90" step="5" value="35" aria-label="Blend amount"></label>
     <label>Era<select id="builderEra"></select></label><label>Texture<select id="builderTexture"></select></label><label>Mood Shift<select id="builderMoodShift"></select></label>
    </div>
    <p class="builder-help">Presets set genre, tempo, instruments and production as one YuE2 line. Your choices below add to it. The vocal type sets the song's Voice option rather than the text.</p>
   </section>
   <div class="builder-grid">
    <section class="builder-section"><h3>Style & Rhythm</h3><label>Genre<select id="builderGenre"></select></label><div class="builder-tempo-heading"><label><input type="checkbox" id="builderTempoEnabled">Suggest a Tempo</label><output id="builderTempoLabel" for="builderTempo">120 BPM</output></div><input type="range" id="builderTempo" aria-label="Suggested tempo" min="30" max="240" step="1" value="120" disabled><p class="builder-hint">A style cue. Covers may follow the recording’s tempo.</p><label>Energy<select id="builderEnergy"></select></label></section>
    <section class="builder-section"><h3>Voice</h3><label>Vocal Type<select id="builderVoice"></select></label><div id="builderVocalFields"><div class="builder-voice-pair"><label>Delivery<select id="builderDelivery"></select></label><label>Register<select id="builderRegister"></select></label></div><label>Vocal Language<input id="builderLanguage" type="text" maxlength="80" placeholder="Optional — Turkish, Romanian…"></label></div><p id="builderVoiceHelp" class="builder-hint">Voice choices guide the model; they do not clone a singer.</p></section>
   </div>
   <section class="builder-section"><div class="builder-section-heading"><h3>Instruments</h3><span id="builderInstrumentCount">None Selected</span></div><div id="builderInstruments"></div></section>
   <section class="builder-section"><h3>Mood & Production</h3><div id="builderMoods" class="builder-chips" role="group" aria-label="Mood"></div><div class="builder-grid"><label>Arrangement & Production<select id="builderProduction"></select></label><label>Intro<select id="builderIntro"></select></label></div><label>Other Details<textarea id="builderExtra" rows="2" maxlength="2000" placeholder="Optional — syncopated groove, brass responses, a saxophone solo…"></textarea></label></section>
   <section class="builder-preview"><h3>Prompt Preview</h3><p id="builderPreview" role="status" aria-live="polite"></p><div class="builder-apply-mode" role="group" aria-label="Apply prompt"><label><input type="radio" name="builderApplyMode" value="append" checked>Add to Current Prompt</label><label><input type="radio" name="builderApplyMode" value="replace">Replace Current Prompt</label></div><p id="builderApplyHelp" class="builder-hint">Your current prompt stays as written. You can edit the result in Styles.</p><p id="builderError" class="builder-error" role="alert" hidden></p></section>
  </div>
  <footer class="builder-footer"><button id="builderCancel" type="button" class="secondary">Cancel</button><div class="builder-footer-actions"><button id="builderReset" type="button" class="secondary">Clear</button><button id="builderApply" type="button" class="primary" disabled>Add to Prompt</button></div></footer>`;
 document.body.append(dialog);
 let state=StyleBuilder.empty(),returnToStyle=false;
 const button=document.createElement('button');button.id='openStyleBuilder';button.type='button';button.className='secondary add-lyric-tag';button.setAttribute('aria-label','Add Style');button.title='Add Style';button.textContent='Add Style';
 $('style').closest('details').querySelector('summary').append(button);
 function options(id,values,placeholder){
  const select=$(id);select.setAttribute('aria-label',select.parentElement.firstChild.textContent.trim());select.append(new Option(placeholder,''));
  for(const item of values){const [value,label]=Array.isArray(item)?item:[item,item];select.append(new Option(label,value))}
 }
 const presetsReady=typeof StylePresets!=='undefined'&&StylePresets.data.length;
 if(presetsReady){
  for(const id of ['builderPreset','builderBlendWith']){
   const select=$(id);select.append(new Option(id==='builderPreset'?'No preset — build from scratch':'None',''));
   for(const group of StylePresets.grouped()){const g=document.createElement('optgroup');g.label=group.label;for(const p of group.presets)g.append(new Option(p.name,p.id));select.append(g)}
  }
  $('builderPresetInfo').textContent=StylePresets.data.length+' styles';
  const modifierLabels={era:'Any era',texture:'Any texture',moodShift:'Keep the preset mood'};
  for(const [axis,placeholder]of Object.entries(modifierLabels)){
   const select=$('builder'+axis[0].toUpperCase()+axis.slice(1));select.append(new Option(placeholder,''));
   for(const choice of Object.keys(StylePresets.MODIFIERS[axis]))select.append(new Option(choice,choice));
  }
 }else $('builderPresetInfo').closest('section').hidden=true;
 options('builderGenre',StyleBuilder.genres,'Any genre');
 options('builderVoice',[...Object.entries(StyleBuilder.voices),['instrumental','Instrumental only']],'Keep current vocal mode');
 options('builderDelivery',StyleBuilder.deliveries,'Any delivery');
 options('builderRegister',['Soprano','Mezzo-soprano','Alto','Tenor','Baritone','Bass'],'Any register');
 options('builderEnergy',['Gentle, restrained energy','Steady, moderate energy','High energy','Explosive, anthemic energy'],'Any energy');
 options('builderProduction',StyleBuilder.productions,'Leave open');
 options('builderIntro',StyleBuilder.intros,'Leave open');
 function chip(parent,value,key){
  const b=document.createElement('button');b.type='button';b.textContent=value;b.setAttribute('aria-pressed','false');b.dataset.choice=key;
  b.onclick=()=>{state[key]=state[key].includes(value)?state[key].filter(x=>x!==value):[...state[key],value];render()};parent.append(b);
 }
 for(const [group,values]of Object.entries(StyleBuilder.instruments)){
  const row=document.createElement('div');row.className='builder-instrument-group';const heading=document.createElement('h4');heading.textContent=group;
  const chips=document.createElement('div');chips.className='builder-chips';chips.setAttribute('role','group');chips.setAttribute('aria-label',group);
  values.forEach(value=>chip(chips,value,'instruments'));row.append(heading,chips);$('builderInstruments').append(row);
 }
 StyleBuilder.moods.forEach(value=>chip($('builderMoods'),value,'moods'));
 const fields={Preset:'preset',BlendWith:'blendWith',Detail:'detail',Era:'era',Texture:'texture',MoodShift:'moodShift',Genre:'genre',Voice:'voice',Delivery:'delivery',Register:'register',Energy:'energy',Production:'production',Intro:'intro',Language:'language',Extra:'extra'};
 for(const [suffix,key]of Object.entries(fields))$('builder'+suffix).addEventListener('input',event=>{state[key]=event.target.value;render()});
 $('builderBlendAmount').oninput=event=>{state.blendAmount=Number(event.target.value);render()};
 $('builderTempo').oninput=event=>{state.bpm=Number(event.target.value);render()};
 $('builderTempoEnabled').onchange=event=>{state.tempoEnabled=event.target.checked;render()};
 function applyMode(){return dialog.querySelector('input[name="builderApplyMode"]:checked').value}
 function render(){
  const noVocals=StyleBuilder.instrumental(state,$('instrumental').checked);
  for(const suffix of ['Delivery','Register','Language'])$('builder'+suffix).disabled=noVocals;
  $('builderVocalFields').classList.toggle('builder-disabled',noVocals);
  $('builderVoiceHelp').textContent=noVocals?'Instrumental only is active. Voice and language choices are omitted.':'Voice choices guide the model; they do not clone a singer.';
  $('builderTempo').disabled=!state.tempoEnabled;$('builderTempoLabel').textContent=state.bpm+' BPM';$('builderTempoLabel').classList.toggle('builder-disabled',!state.tempoEnabled);
  $('builderPresetOptions').hidden=!state.preset;$('builderBlendWith').disabled=!state.preset;
  $('builderBlendAmountLabel').hidden=!(state.preset&&state.blendWith&&state.blendWith!==state.preset);$('builderBlendAmountValue').textContent=state.blendAmount+'%';
  for(const b of dialog.querySelectorAll('[data-choice]'))b.setAttribute('aria-pressed',String(state[b.dataset.choice].includes(b.textContent)));
  $('builderInstrumentCount').textContent=state.instruments.length?state.instruments.length+' Selected':'None Selected';
  const prompt=StyleBuilder.build(state,$('instrumental').checked),replacement=applyMode()==='replace';
  $('builderPreview').textContent=prompt||'Choose a style, voice or instrument to preview your prompt.';
  $('builderPreview').classList.toggle('builder-preview-empty',!prompt);
  const result=StyleBuilder.merge($('style').value,prompt,applyMode());
  $('builderError').hidden=result.length<=6000;$('builderError').textContent='The combined prompt is over 6,000 characters. Shorten the details or replace the current prompt.';
  $('builderApply').disabled=!prompt||result.length>6000;$('builderApply').textContent=replacement?'Use This Prompt':'Add to Prompt';
  $('builderApplyHelp').textContent=replacement?'Replaces the text in Styles. You can edit the result before creating.':'Your current prompt stays as written. You can edit the result in Styles.';
  $('builderReset').disabled=JSON.stringify(state)===JSON.stringify(StyleBuilder.empty())&&applyMode()==='append';
 }
 function syncFields(){
  for(const [suffix,key]of Object.entries(fields))$('builder'+suffix).value=state[key];
  $('builderTempo').value=state.bpm;$('builderTempoEnabled').checked=state.tempoEnabled;$('builderBlendAmount').value=state.blendAmount;render();
 }
 button.onclick=e=>{e.preventDefault();e.stopPropagation();returnToStyle=false;dialog.querySelector('input[value="append"]').checked=true;syncFields();dialog.showModal();$('styleBuilderHeading').focus()};
 for(const id of ['builderClose','builderCancel'])$(id).onclick=()=>dialog.close();
 dialog.addEventListener('click',event=>{if(event.target===dialog){const r=dialog.getBoundingClientRect();if(event.clientX<r.left||event.clientX>r.right||event.clientY<r.top||event.clientY>r.bottom)dialog.close()}});
 dialog.addEventListener('close',()=>{(returnToStyle?$('style'):button).focus()});
 for(const input of dialog.querySelectorAll('[name="builderApplyMode"]'))input.onchange=render;
 $('builderReset').onclick=()=>{state=StyleBuilder.empty();dialog.querySelector('input[value="append"]').checked=true;syncFields()};
 $('builderApply').onclick=()=>{
  render();if($('builderApply').disabled)return;
  $('style').value=StyleBuilder.merge($('style').value,StyleBuilder.build(state,$('instrumental').checked),applyMode());
  if(state.voice)$('instrumental').checked=state.voice==='instrumental';
  if(StyleBuilder.LEAD_VOICES.includes(state.voice))$('voice').value=state.voice;
  updateUI();saveDraft();returnToStyle=true;dialog.close();
 };
}
