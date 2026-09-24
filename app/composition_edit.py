"""Validated, non-destructive edits of saved YuE2 compositions."""
import copy, hashlib, json, re, secrets
from pathlib import Path
from vendor import yue2_abc as abc_tools
import instrumental
import song_library

def digest(text):return hashlib.sha256(text.encode()).hexdigest()
def canonical(text):return '\n'.join(x.rstrip() for x in text.strip().splitlines())+'\n'
def split_score(text):
    text=canonical(text);score=abc_tools.parse(text)
    lines=text.splitlines();header=lines[:8];parts=[];current=None
    meter,key=header[2],header[7]
    for line in lines[8:]:
        if line.startswith('% '):
            if current and current['lines']:parts.append(current)
            current={'name':line[2:].strip() or 'Section','lines':[],'meter':meter,'key':key}
        else:
            if current is None:current={'name':'Song','lines':[],'meter':meter,'key':key}
            current['lines'].append(line)
            if line.startswith('M:'):meter=line
            if line.startswith('K:'):key=line
    if current and current['lines']:parts.append(current)
    counts={}
    for i,p in enumerate(parts):
        n=p['name'];counts[n]=counts.get(n,0)+1;p.update(id=str(i),label=n.title()+' '+str(counts[n]),lyrics='')
    return header,parts,score

def lyric_sections(lyrics,parts):
    """Assign each [Tag] block to the next score section of that name; False (and no assignments) when the tags do not fit."""
    for p in parts:p['lyrics']=''
    tagged=re.split(r'(?m)^\s*\[([^\]\n]+)\]\s*\n?',lyrics.strip())
    if len(tagged)<3 or tagged[0].strip():return False
    cursor=0;assigned={}
    for name,text in zip(tagged[1::2],tagged[2::2]):
        norm=re.sub(r'[^a-z]','',name.lower())
        found=next((i for i in range(cursor,len(parts)) if re.sub(r'[^a-z]','',parts[i]['name'].lower()) in (norm,re.sub(r'\d','',norm))),None)
        if found is None:return False
        assigned[found]=text.strip();cursor=found+1
    for i,text in assigned.items():parts[i]['lyrics']=text
    return True

def tagged_lyrics(sections,instrumental):
    blocks=[]
    for p in sections:
        words=p['lyrics'].strip()
        if words or instrumental:blocks.append('['+p['name'].title()+']'+('\n'+words if words else ''))
    return '\n\n'.join(blocks)

def source_song(job,index):
    result,c=song_library.locate(job,index);c=c or result
    prefix=c.get('prefix','')
    if prefix not in ('',f'candidate-{index:02d}/'):raise ValueError('Invalid song version')
    request=json.loads((job/'request.json').read_text());path=job/prefix/'score.abc'
    text=path.read_text() if path.is_file() else c.get('abc')
    if not text:raise ValueError('This version has no saved composition. Upload its WAV in Cover / Remix to analyze it first.')
    header,parts,score=split_score(text)
    mapped=lyric_sections(request.get('lyrics',''),parts)
    return dict(title=c.get('title') or request['title'],style=request['style'],lyrics=request['lyrics'],instrumental=request.get('instrumental',False),bpm=score.bpm,has_chords=bool(score.voices['Vocal'].chords),sections=parts,lyrics_mapped=mapped,abc=canonical(text),header=header,request=request,origin={'job':job.name,'candidate':index,'score_sha256':digest(canonical(text))})

def render_section(p):
    lines=list(p['lines'])
    # Restore context at section boundaries so reorder/repeat cannot inherit another key or meter.
    for voice in ('Ins','Vocal'):
        i=lines.index('V: '+voice)+1
        fields=[]
        while i<len(lines) and lines[i].startswith(('M:','K:')):fields.append(lines.pop(i))
        values={x[:2]:x for x in [p['meter'],p['key']]+fields}
        lines[i:i]=[values['M:'],values['K:']]
    return '% '+p['name']+'\n'+'\n'.join(lines)+'\n'

def shorten(p,fraction):
    if not 0<fraction<1:raise ValueError('Shortening must use a fraction between 0 and 1')
    lines=p['lines'];groups=[];i=0
    while i<len(lines):
        if lines[i]!='V: Vocal':raise ValueError('Unsupported section structure')
        j=lines.index('V: Ins',i+1);end=next((k for k in range(j+1,len(lines)) if lines[k]=='V: Vocal'),len(lines))
        voices=[]
        for body in (lines[i+1:j],lines[j+1:end]):
            fields=body[:-1];music=re.sub(r'Z([1-9][0-9]*)',lambda m:'|'.join(['Z']*int(m.group(1))),body[-1]);bars=music.rstrip('|').split('|');voices.append((fields,bars))
        groups.append(voices);i=end
    total=sum(len(g[0][1]) for g in groups);keep=max(1,round(total*fraction))
    if keep==total:raise ValueError('This section is too short to shorten at a safe bar boundary.')
    out=[];remaining=keep
    for group in groups:
        n=min(remaining,len(group[0][1]))
        for name,(fields,bars) in zip(('Vocal','Ins'),group):
            selected=bars[:n]
            if n==remaining:selected[-1]=re.sub(r'-\s*$','',selected[-1])
            out+=['V: '+name]+fields+['|'.join(selected)+'|']
        remaining-=n
        if not remaining:break
    p['lines']=out
    words=p['lyrics'].splitlines();p['lyrics']='\n'.join(words[:max(1,round(len(words)*keep/total))])

def close_phrase_ties(p):
    # A moved/cut phrase cannot sustain through a newly inserted section or rest.
    for voice in ('Vocal','Ins'):
        positions=[i for i,line in enumerate(p['lines']) if line=='V: '+voice]
        if not positions:continue
        start=positions[-1]+1
        while start<len(p['lines']) and p['lines'][start].startswith(('M:','K:')):start+=1
        p['lines'][start]=re.sub(r'-(?=\s*\|\s*$)','',p['lines'][start])

def make_solo(p):
    # Reuse the source vocal motif as an instrumental break, retaining bar durations.
    lines=p['lines'];out=[];i=0
    while i<len(lines):
        if lines[i]!='V: Vocal':raise ValueError('Unsupported solo source')
        j=lines.index('V: Ins',i+1);end=next((k for k in range(j+1,len(lines)) if lines[k]=='V: Vocal'),len(lines))
        vocal=lines[i+1:j];fields=[x for x in vocal if x.startswith(('M:','K:'))]
        rests,motif,_=instrumental.move_vocal(vocal[-1],lines[end-1])
        out+=['V: Vocal']+fields+[rests,'V: Ins']+fields+[motif];i=end
    p['lines']=out;p['name']='interlude';p['lyrics']='';close_phrase_ties(p)

def apply(source,options):
    header=list(source['header']);sections=copy.deepcopy(source['sections']);style=options.get('style',source['style']).strip();lyrics=options.get('lyrics',source['lyrics'])
    bpm=source['bpm'] if options.get('bpm') is None else options['bpm'];bpm=int(bpm)
    if not 30<=bpm<=300:raise ValueError('Choose a tempo between 30 and 300 BPM')
    if not style or len(style)>6000:raise ValueError('Enter a style of at most 6,000 characters')
    header[4]='Q:1/4='+str(bpm)
    supplied=options.get('sections')
    changed=False;solo_only=False;resectioned=None
    if supplied is not None:
        if not supplied or len(supplied)>80:raise ValueError('Keep between 1 and 80 sections')
        # The tagged lyrics are the single source of truth: words follow the structure only when their tags fit the score.
        mapped=lyric_sections(lyrics,sections)
        by_id={p['id']:p for p in sections};chosen=[]
        changed=[str(x['id']) for x in supplied]!=list(by_id) or any(x.get('fraction',1)!=1 or x.get('solo',False) for x in supplied)
        for row in supplied:
            if not isinstance(row,dict):raise ValueError('Invalid section')
            if str(row['id']) not in by_id:raise ValueError('Unknown section')
            p=copy.deepcopy(by_id[str(row['id'])])
            if row.get('fraction',1)!=1:shorten(p,float(row['fraction']))
            if row.get('solo',False):
                make_solo(p)
            chosen.append(p)
        solo_only=changed and [p['id'] for p,row in zip(chosen,supplied) if not row.get('solo')]==list(by_id) and all(row.get('fraction',1)==1 for row in supplied)
        for i,p in enumerate(chosen):
            next_row=supplied[i+1] if i+1<len(supplied) else None
            original_next=str(int(p['id'])+1)
            if changed and (supplied[i].get('solo') or next_row is None or next_row.get('solo') or str(next_row['id'])!=original_next):close_phrase_ties(p)
        sections=chosen
        if changed and not solo_only:
            resectioned=mapped
            if mapped:lyrics=tagged_lyrics(sections,source['instrumental'])
    text='\n'.join(header)+'\n'+''.join(render_section(p) for p in sections)
    harmony=options.get('harmony','preserve')
    if harmony not in ('preserve','free','sevenths'):raise ValueError('Unknown harmony option')
    if harmony=='sevenths' and not abc_tools.parse(text).voices['Vocal'].chords:
        raise ValueError('This composition has melody only. Choose New chords, same melody and describe the desired harmony in Style.')
    if harmony!='preserve':
        lines=text.splitlines()
        for i in range(8,len(lines)):
            if lines[i].startswith(('V:','%','M:','K:')):continue
            def chord(m):
                if harmony=='free':return ''
                value=m.group(1)
                if re.fullmatch(r'[A-G][b#]?m?',value):value+= '7' if value.endswith('m') else 'maj7'
                return '"'+value+'"'
            lines[i]=re.sub(r'"([^"\n]*)"',chord,lines[i])
        text='\n'.join(lines)+'\n'
    if len(text)>40000 or len(lyrics)>30000:raise ValueError('The edit exceeds the composition or lyrics size limit')
    after=abc_tools.parse(text)
    if float(after.voices['Vocal'].time)*60/bpm>720:raise ValueError('The edited song exceeds the 12-minute generation limit.')
    if not changed:
        check=abc_tools.compare(abc_tools.parse(source['abc']),after,allow_tempo_change=True)
        if not check['match']:raise ValueError('The edit would change melody unexpectedly: '+', '.join(check['differences']))
    if not source['instrumental'] and not lyrics.strip():raise ValueError('Keep lyrics for this vocal version, or choose an instrumental source')
    return dict(abc=text,style=style,lyrics=lyrics,bpm=bpm,sections=sections,instrumental=source['instrumental'],origin=source['origin'],cot='melody' if harmony=='free' else 'full',seconds=round(float(after.voices['Vocal'].time)*60/bpm,2),summary=('Instrumental break added; original lyric order retained.' if solo_only else 'Structure updated; the lyrics were re-sectioned to match.' if resectioned else 'Structure updated; the lyrics stay exactly as written because their section tags do not match the score.' if changed else 'Melody notes and timing verified.')+(' Tempo updated.' if bpm!=source['bpm'] else '')+(' Harmony updated.' if harmony!='preserve' else ''))

def save_draft(root,edit):
    path=Path(root)/'.song-edits';path.mkdir(exist_ok=True);key=secrets.token_hex(12)
    (path/(key+'.json')).write_text(json.dumps(edit,indent=2));return {'id':key,**{k:edit[k] for k in ('style','lyrics','bpm','seconds','summary','origin','cot')}}

def load_draft(root,key):
    if not re.fullmatch('[a-f0-9]{24}',key):raise ValueError('Invalid edit')
    path=Path(root)/'.song-edits'/(key+'.json')
    if not path.is_file():raise ValueError('Saved edit not found. Prepare the edit again.')
    return json.loads(path.read_text())
