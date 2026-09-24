import json
import sys
import tempfile
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import song_library

class SongLibraryTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.job=self.root/'song';self.job.mkdir();self.trash=self.root/'.trash'
        (self.job/'request.json').write_text(json.dumps({'title':'Original'}))
        (self.root/'source.wav').write_bytes(b'source')
        self.result={'candidates':[{'index':i,'prefix':f'candidate-{i:02d}/'} for i in (1,2)]}
        (self.job/'result.json').write_text(json.dumps(self.result))
        for i in (1,2):
            p=self.job/f'candidate-{i:02d}';p.mkdir();(p/'audio.wav').write_bytes(bytes([i]))
    def test_rename_only_selected_version(self):
        song_library.rename(self.job,2,'New name')
        result=json.loads((self.job/'result.json').read_text())
        self.assertNotIn('title',result['candidates'][0]);self.assertEqual(result['candidates'][1]['title'],'New name')
        self.assertEqual(json.loads((self.job/'request.json').read_text())['title'],'Original')
    def test_delete_one_version_retains_sibling_source_and_trashes_audio(self):
        (self.job/'favorite.json').write_text('{"candidate":1}')
        song_library.delete(self.job,1,self.trash)
        self.assertTrue((self.job/'candidate-02/audio.wav').exists())
        self.assertFalse((self.job/'candidate-01').exists());self.assertFalse((self.job/'favorite.json').exists())
        self.assertEqual([c['index'] for c in json.loads((self.job/'result.json').read_text())['candidates']],[2])
        self.assertTrue(list(self.trash.glob('*/audio.wav')));self.assertTrue((self.root/'source.wav').exists())
        song_library.delete(self.job,2,self.trash);self.assertFalse(self.job.exists())
    def test_legacy_song_and_unfinished_session(self):
        (self.job/'result.json').unlink();(self.job/'audio.wav').write_bytes(b'legacy')
        song_library.rename(self.job,1,'Legacy name')
        self.assertEqual(json.loads((self.job/'request.json').read_text())['title'],'Legacy name')
        with self.assertRaises(song_library.SongNotFound):song_library.delete(self.job,0,self.trash)
        (self.job/'audio.wav').unlink();song_library.delete(self.job,0,self.trash)
        self.assertFalse(self.job.exists())
    def test_invalid_version_does_not_change_files(self):
        with self.assertRaises(song_library.SongNotFound):song_library.delete(self.job,99,self.trash)
        self.assertTrue((self.job/'candidate-01/audio.wav').exists())
        self.assertFalse(self.trash.exists())

if __name__=='__main__':unittest.main()
