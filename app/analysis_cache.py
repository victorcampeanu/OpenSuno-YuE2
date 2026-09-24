"""Persist recording analysis independently of generated songs and model memory."""
import hashlib
import json
from pathlib import Path

class AnalysisCache:
    VERSION=1
    def __init__(self,root):self.root=Path(root)
    def key(self,r):
        value=[self.VERSION,r.audio_id,r.melody_only,float(r.source_seconds)]
        return hashlib.sha256(json.dumps(value).encode()).hexdigest()
    def path(self,r,suffix):return self.root/(self.key(r)+suffix)
    def write(self,path,value):
        self.root.mkdir(exist_ok=True)
        temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value));temp.replace(path)
    def read(self,r):
        try:
            data=json.loads(self.path(r,'.json').read_text())
            if not data.get('abc') or (data['audio_id'],data['melody_only'],data['source_seconds'])!=(r.audio_id,r.melody_only,r.source_seconds):return None
            return data
        except (OSError,ValueError,KeyError,TypeError,AttributeError):return None
    def save(self,r,data):self.write(self.path(r,'.json'),data)
    def remember(self,r,job):self.write(self.path(r,'.job'),{'id':job})
    def job(self,r):
        try:return json.loads(self.path(r,'.job').read_text())['id']
        except (OSError,ValueError,KeyError,TypeError):return None

class TokenCache(AnalysisCache):
    """Music tokens of an uploaded recording (audio_tokens), shared by every continuation made from it.
    The tokenizer's name is part of the key: a new tokenizer must not reuse tokens the old one wrote."""
    KIND='tokens'
    def key(self,r):
        import audio_tokens
        value=[self.VERSION,self.KIND,audio_tokens.NAME,r.audio_id,float(r.source_seconds)]
        return hashlib.sha256(json.dumps(value).encode()).hexdigest()
    def read(self,r):
        try:
            data=json.loads(self.path(r,'.json').read_text())
            if not isinstance(data.get('codec'),list) or not data['codec'] or (data['audio_id'],data['source_seconds'])!=(r.audio_id,r.source_seconds):return None
            return data
        except (OSError,ValueError,KeyError,TypeError,AttributeError):return None
