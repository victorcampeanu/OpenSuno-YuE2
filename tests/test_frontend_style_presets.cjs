const assert=require('node:assert/strict');
const StylePresets=require('../web/style-presets.js');
global.StylePresets=StylePresets;
const StyleBuilder=require('../web/style-builder.js');
const LyricsStructure=require('../web/lyrics-structure.js');

// Presets render as one YuE2 line in the published order: genre, tempo, mood, instruments, voice, production.
assert.equal(StylePresets.data.length,73);
const reggae=StylePresets.get('world-reggae');
const full=StylePresets.prompt(reggae);
assert.match(full,/^reggae, roots reggae, dub, rocksteady, jamaican, 76 BPM, laid-back and dreamy, /);
assert.ok(full.indexOf('offbeat guitar skank')<full.indexOf('warm mid-range voice'),'instruments come before the voice');
assert.equal(StylePresets.prompt(reggae,{detail:'tags',vocal:'female',language:'Spanish'}),'reggae, roots reggae, dub, rocksteady, jamaican, 76 BPM, laid-back and dreamy, female vocals, Spanish');
assert.ok(!StylePresets.prompt(reggae,{vocal:'instrumental',language:'Spanish'}).includes('Spanish'),'no language on an instrumental');
assert.ok(StylePresets.prompt(reggae,{detail:'rich'}).includes('beach party'),'rich adds the scene');
// A preset that usually has no singer never silences a song that has lyrics; only the Instrumental choice does.
for(const p of StylePresets.data){
 const sung=StylePresets.prompt(p),withVoice=StylePresets.prompt(p,{vocal:'male'});
 // "extended instrumental solo section" is fine in a sung song; the genre tag "instrumental" and "no vocals" are not.
 for(const text of [sung,withVoice])assert.ok(!/no vocals|(^|, )instrumental(,|$| rock)/i.test(text),p.id+' must not read as instrumental when sung: '+text);
 const silent=StylePresets.prompt(p,{vocal:'instrumental'});
 assert.ok(!/no vocals|\b(fe)?male\b/i.test(silent),p.id+' instrumental choice leaves the one "Instrumental." line to the worker: '+silent);
}
assert.match(StylePresets.prompt(StylePresets.get('electronic-dnb'),{vocal:'male'}),/^drum and bass, electronic, jungle, breakbeat, edm, 174 BPM, .*male vocals/);
assert.ok(StylePresets.prompt(StylePresets.get('rock-post_rock')).includes('rock, ambient rock'),'"instrumental rock" becomes "rock" for a sung song');
assert.ok(StylePresets.prompt(StylePresets.get('rock-post_rock'),{vocal:'instrumental'}).includes('instrumental rock'),'and stays for an instrumental');
assert.ok(!StyleBuilder.build({...StyleBuilder.empty(),preset:'electronic-dnb'},false).toLowerCase().includes('no vocals'),'the builder with the Instrumental box unticked keeps the singer');
// Article stripping and aliases keep the tags in YuE2 shape.
assert.equal(StylePresets.prompt({genre:['x'],instruments:['the 808 drum machine','a grand piano']}),'x, heavy 808s, grand piano');

// Blending interleaves lists and switches the scalars only past the halfway point.
const techno=StylePresets.get('electronic-techno');
const mild=StylePresets.blend(reggae,techno,.25),strong=StylePresets.blend(reggae,techno,.75);
assert.equal(mild.bpm,reggae.bpm);assert.equal(strong.bpm,techno.bpm);
assert.ok(mild.genre.includes('reggae')&&mild.genre.includes('techno'));
assert.equal(new Set(strong.instruments).size,strong.instruments.length,'no duplicates');
// Modifiers add phrases in front and never replace.
const shifted=StylePresets.modify(reggae,{era:'1970s',moodShift:'darker',texture:''});
assert.equal(shifted.production[0],'warm analog tape saturation');assert.equal(shifted.mood[0],'dark and brooding');
assert.ok(shifted.mood.includes(reggae.mood[0]));
assert.equal(StylePresets.grouped().reduce((n,g)=>n+g.presets.length,0),73);

// The Style Builder puts the preset line first, then the extra sentences; language folds into the preset.
// The lead voice is never written: Apply sets the Voice option and the worker adds the single voice line.
const state={...StyleBuilder.empty(),preset:'acoustic-bluegrass',voice:'female',language:'Spanish',instruments:['Harp']};
const built=StyleBuilder.build(state);
const [line,rest]=built.split('\n');
assert.match(line,/^bluegrass, country, folk, americana, appalachian, 140 BPM, .*high lonesome tenor above the melody/);
assert.ok(!/female|male/i.test(line),'no gender words in the text: '+line);
assert.ok(line.endsWith('Spanish'));assert.equal(rest,'Harp.');
assert.ok(!StyleBuilder.build({...StyleBuilder.empty(),genre:'Jazz',voice:'male'}).includes('vocal'),'no preset, no voice words either');
assert.equal(StyleBuilder.build({...StyleBuilder.empty(),genre:'Jazz',voice:'choir'}),'Jazz. Choir vocals.','a choir is an arrangement word, not a lead voice');
const silentBuild=StyleBuilder.build({...StyleBuilder.empty(),preset:'acoustic-bluegrass'},true);
assert.ok(!/no vocals|\b(fe)?male\b/i.test(silentBuild),'current instrumental mode adds no vocal words: '+silentBuild);
assert.ok(!StyleBuilder.build({...StyleBuilder.empty(),genre:'Jazz',voice:'instrumental'}).includes('Instrumental'),'the worker writes "Instrumental."');
assert.equal(StyleBuilder.build({...StyleBuilder.empty(),preset:'acoustic-bluegrass',tempoEnabled:true,bpm:100}).split('\n')[1],'100 BPM.');
assert.ok(!StyleBuilder.build({...StyleBuilder.empty(),preset:'acoustic-bluegrass',tempoEnabled:true,bpm:100}).includes('140 BPM'),'a chosen tempo replaces the preset tempo');
assert.equal(StyleBuilder.build({...StyleBuilder.empty(),genre:'Jazz'}),'Jazz.','no preset keeps the old sentences');

// Lyrics structure: section tags become [Tag], numbers kept, case left alone, unknown tags untouched.
const report=LyricsStructure.check('(Intro)\nVerse 1:\nSome words to sing tonight\n{HOOK}\nOh oh\n[verse 2]\nMore words\n[Weird]\nla la',300);
assert.equal(report.lyrics,'[Intro]\n[Verse 1]\nSome words to sing tonight\n[Chorus]\nOh oh\n[verse 2]\nMore words\n[Weird]\nla la');
assert.deepEqual(report.sections,['Intro','Verse','Chorus','Verse']);
assert.equal(report.changes.length,3);assert.deepEqual(report.unknown.map(u=>u.text),['[Weird]']);
assert.equal(report.warnings.length,1);assert.match(report.warnings[0].text,/3 section tags are not in the \[Tag\] form/);
assert.equal(LyricsStructure.check('[Verse]\nhello there\n[Chorus]\nla la la',360).warnings.length,0,'clean tags and a generous cap warn about nothing');
const long=LyricsStructure.check('[Verse]\n'+'Twinkle twinkle little star how I wonder what you are\n'.repeat(40),30);
assert.equal(long.warnings.length,1);assert.equal(long.warnings[0].kind,'fit');assert.match(long.warnings[0].text,/0:30 cap will cut/);
assert.equal(LyricsStructure.normalize('Chorus:\nwords\nMy heart says:\nmore').lyrics,'[Chorus]\nwords\nMy heart says:\nmore','a lyric line ending in a colon is not a tag');
console.log('style presets, builder and lyrics structure OK');
