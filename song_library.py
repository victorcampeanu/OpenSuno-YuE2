"""Rename individual versions and move deleted songs into local trash."""
import json
import shutil
import uuid

class SongNotFound(ValueError): pass

def write_json(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(path)

def locate(job,index):
    path=job/'result.json'
    result=json.loads(path.read_text()) if path.exists() else {}
    candidates=result.get('candidates',[])
    if candidates:
        candidate=next((c for c in candidates if c['index']==index),None)
        if candidate is None: raise SongNotFound('Song version not found')
        return result,candidate
    if index!=(1 if has_audio(job) else 0): raise SongNotFound('Song version not found')
    return result,None

def has_audio(job):
    """A single-version song has audio locally, or still on the render node that made it."""
    if (job/'audio.wav').exists(): return True
    try: return 'audio.wav' in json.loads((job/'remote.json').read_text()).get('pending',[])
    except (OSError,ValueError,AttributeError): return False

def rename(job,index,title):
    result,candidate=locate(job,index)
    if candidate is not None:
        candidate['title']=title
        write_json(job/'result.json',result)
    if candidate is None or len(result.get('candidates',[]))==1:
        request=json.loads((job/'request.json').read_text());request['title']=title
        write_json(job/'request.json',request)

def delete(job,index,trash):
    result,candidate=locate(job,index)
    trash.mkdir(exist_ok=True)
    target=trash/(job.name+'-'+uuid.uuid4().hex)
    if candidate is None or len(result['candidates'])==1:
        shutil.move(str(job),str(target))
        return
    prefix=candidate.get('prefix','')
    expected=f'candidate-{index:02d}/'
    if prefix!=expected: raise ValueError('Unexpected song version directory')
    # Move only the selected version, retaining its siblings and the source upload.
    source=job/prefix
    if source.exists(): shutil.move(str(source),str(target))
    result['candidates']=[c for c in result['candidates'] if c['index']!=index]
    write_json(job/'result.json',result)
    favorite=job/'favorite.json'
    if favorite.exists() and json.loads(favorite.read_text()).get('candidate')==index:
        favorite.unlink()
