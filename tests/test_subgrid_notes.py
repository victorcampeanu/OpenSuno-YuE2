import unittest
import numpy as np
import pretty_midi
from transcriber.notation_sheetsage2 import _notes_to_arr, MelodyVoiceError


class SubgridNotesTest(unittest.TestCase):
    def test_short_note_is_reported_without_changing_midi_or_other_notes(self):
        tiny=pretty_midi.Note(90,76,.201,.221)
        regular=pretty_midi.Note(90,60,.5,.75)
        diagnostics=[]
        result=_notes_to_arr([tiny,regular],np.arange(0,1.01,.125),'Ins',diagnostics)
        expected=_notes_to_arr([regular],np.arange(0,1.01,.125),'Ins')
        np.testing.assert_array_equal(result,expected)
        self.assertEqual((tiny.start,tiny.end),(.201,.221))
        self.assertEqual(len(diagnostics),1)

    def test_outside_grid_and_overlap_still_fail(self):
        grid=np.arange(0,1.01,.125)
        for notes in ([pretty_midi.Note(90,60,2,3)],
                      [pretty_midi.Note(90,60,0,.5),pretty_midi.Note(90,62,.25,.75)]):
            with self.assertRaises(MelodyVoiceError):
                _notes_to_arr(notes,grid,'Ins',[])
