/* Style presets as YuE2 prompts. Presets are structured (genre, mood, instruments, ...), so two can be blended
   as lists and every preset renders through one grammar: genre → tempo/mood → instruments → vocal → production
   → scene → language, the order that works best in YuE2's published prompts.
   Ported from SongScribe (TheLocalLab, MIT). */
const StylePresets = (() => {
 const data=typeof STYLE_PRESET_DATA!=='undefined'?STYLE_PRESET_DATA:(typeof require!=='undefined'?require('./style-presets-data.js'):[]);
 const CATEGORIES={acoustic:'Acoustic',cinematic:'Cinematic',country:'Country',electronic:'Electronic',gospel:'Gospel',hiphop:'Hip-hop',jazz:'Jazz',metal:'Metal',pop:'Pop',rock:'Rock',soul:'Soul / R&B',swing:'Swing',vocal:'Vocal',world:'World'};
 const LIST_FIELDS=['genre','mood','scene','production','instruments','vocal_timbre','vocal_delivery'];
 // Optional flavour layered on top of a preset; these add, never replace.
 const MODIFIERS={
  era:{'1960s':{production:['raw live-room recording','narrow mono-leaning image']},'1970s':{production:['warm analog tape saturation']},'1980s':{production:['wide spacious stereo image','long cavernous hall reverb']},'1990s':{production:['crisp studio multitrack production']},'2000s':{production:['loud heavily compressed master']},modern:{production:['clean modern polished mix']}},
  texture:{'lo-fi':{production:['muddy lo-fi bedroom mix','heavy vinyl crackle and surface noise']},tape:{production:['tape hiss and wow-flutter pitch wobble']},'hi-fi':{production:['bright airy hi-fi mix']},intimate:{production:["dry close-mic'd intimate sound"]},cavernous:{production:['heavy reverb drenching everything']}},
  moodShift:{darker:{mood:['dark and brooding']},brighter:{mood:['bright and joyful']},sadder:{mood:['melancholy and wistful']},dreamier:{mood:['hazy and drifting']},harder:{mood:['energetic and driving']}}
 };
 const VOCALS={female:'female vocals',male:'male vocals',duet:'duet, male and female vocals',choir:'full choir',group:'group vocals, layered harmonies',rap:'rap vocals',instrumental:'instrumental, no vocals'};
 // Vocabulary phrases are prose fragments; YuE2 tags never start with an article and a few read wrong as tags.
 const ALIASES={'808 drum machine':'heavy 808s','punchy electronic drum machine':'punchy electronic drums','vinyl crackle texture':'vinyl crackle','warm Rhodes electric piano':'warm Rhodes piano','field recording ambience':'field recordings','reversed ambient swells':'reversed swells','tambourine and shaker':'tambourine, shaker','congas and bongos':'congas, bongos','orchestral timpani':'timpani','acoustic drum kit':'live drum kit'};
 const LIMITS={tags:{mood:1,instruments:0,production:0,scene:0},full:{mood:1,instruments:5,production:2,scene:0},rich:{mood:3,instruments:6,production:3,scene:2}};
 const tag=value=>{const bare=String(value).replace(/^(a|an|the)\s+/i,'').trim();return ALIASES[bare]||bare};
 const plain=(values,limit)=>{const out=(values||[]).filter(Boolean).map(tag);return limit===undefined?out:out.slice(0,limit)};
 const byId=Object.fromEntries(data.map(p=>[p.id,p]));
 function get(id){return byId[id]||null}
 function grouped(){
  const groups={};
  for(const p of data)(groups[p.category]??=[]).push(p);
  return Object.keys(groups).sort().map(key=>({key,label:CATEGORIES[key]||key,presets:groups[key].map(p=>({id:p.id,name:p.name.replace(/^[^/]+\/\s*/,'')}))}));
 }
 /* Merge two presets; weight is how much of the second to admit (0–1). Lists interleave proportionally; scalars
    (bpm, key, vocal presence) cross over at the halfway point rather than averaging to a tempo nobody asked for. */
 function blend(primary,secondary,weight){
  weight=Math.max(0,Math.min(1,Number(weight)||0));
  const result={...primary,name:primary.name+' × '+secondary.name,id:primary.id+'+'+secondary.id};
  for(const field of LIST_FIELDS){
   const first=primary[field]||[],second=secondary[field]||[];
   if(!second.length){result[field]=first;continue}
   if(!first.length){result[field]=second;continue}
   const total=Math.max(first.length,second.length),takeSecond=Math.round(total*weight);
   result[field]=[...new Set([...first.slice(0,total-takeSecond),...second.slice(0,takeSecond)])];
  }
  if(weight>=.5)for(const scalar of ['bpm','key','vocal_presence'])if(secondary[scalar]!=null)result[scalar]=secondary[scalar];
  return result;
 }
 function modify(preset,choices){
  const result={...preset};
  for(const [axis,choice]of Object.entries(choices||{})){
   const additions=MODIFIERS[axis]?.[choice];if(!additions)continue;
   for(const [field,values]of Object.entries(additions))result[field]=[...values,...(result[field]||[]).filter(x=>!values.includes(x))];
  }
  return result;
 }
 /* detail: tags = genre, mood and vocal only; full = the standard order; rich = adds production and scene. */
 /* A genre word that says "instrumental" belongs only in an instrumental song; sung songs keep the genre without it. */
 const INSTRUMENTAL_WORD=/^instrumental$/i,INSTRUMENTAL_PREFIX=/^instrumental\s+/i;
 const sungGenre=values=>plain(values).filter(g=>!INSTRUMENTAL_WORD.test(g)).map(g=>g.replace(INSTRUMENTAL_PREFIX,''));
 /* Whether a preset leans instrumental (jazz trio, techno, ...) only shows when the song itself is instrumental;
    a song with lyrics is sung whatever the genre usually does, so 'auto' never writes "no vocals". */
 function prompt(preset,{vocal='auto',language='',detail='full',bpm=true}={}){
  const limits=LIMITS[detail]||LIMITS.full,parts=[],noVocals=vocal==='instrumental';
  parts.push(...(noVocals?plain(preset.genre):sungGenre(preset.genre)));
  if(bpm&&preset.bpm)parts.push(Math.round(preset.bpm)+' BPM');
  parts.push(...plain(preset.mood,limits.mood));
  if(limits.instruments)parts.push(...plain(preset.instruments,limits.instruments));
  // An instrumental song says so once, in the worker's own line; here it only shapes the genre words.
  let resolved=noVocals?null:VOCALS[vocal]||null;
  if(!resolved&&vocal==='auto')resolved=[...plain(preset.vocal_timbre,1),...plain(preset.vocal_delivery,1)].join(', ')||null;
  if(resolved){
   parts.push(tag(resolved));
   if(vocal!=='auto'&&!noVocals&&detail!=='tags')parts.push(...plain(preset.vocal_delivery,1));
  }
  if(limits.production)parts.push(...plain(preset.production,limits.production));
  if(limits.scene)parts.push(...plain(preset.scene,limits.scene));
  if(language&&language.trim()&&!noVocals)parts.push(language.trim());
  const seen=new Set();
  return parts.filter(p=>{const k=p.toLowerCase();return seen.has(k)?false:seen.add(k)}).join(', ');
 }
 return {data,CATEGORIES,MODIFIERS,VOCALS,get,grouped,blend,modify,prompt};
})();
if(typeof module!=='undefined')module.exports=StylePresets;
