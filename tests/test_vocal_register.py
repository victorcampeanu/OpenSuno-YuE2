"""Covers: the transcribed melody is moved by whole octaves into the requested voice's register."""
import sys, unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import vocal_register as vr
from vendor import yue2_abc as abc

HEADER = 'X:1\nT:\nM:4/4\nL:1/16\nQ:1/4=100\nV: Vocal clef=treble name="Vocal Melody" snm="Vocal"\nV: Ins clef=treble name="Ins Melody" snm="Inst."\nK:G\n'
def group(label, vocal, ins): return f'% {label}\nV: Vocal\n{vocal}\nV: Ins\n{ins}\n'
# A melody written where SheetSage2 puts a low male singer: mostly e'-g' (E5-G5), with an F natural
# against the key signature, an inline key change, a tie across a barline and a rest.
HIGH = HEADER + group('intro', '"G"z16|Z|', 'G4B4d4g4|g8d8|') \
    + group('verse', '"G"e4e4f4=f4|"D"g8g4d4-|', 'z16|Z|') \
    + group('chorus', '"Em"d4g12|[K:D]"D"a4^g4a8|', 'D4F4A4d4|[K:D]z16|')
LOW = HEADER + group('verse', '"G"G,4B,4D,4G,4|"D"D,8z8|', 'Z|Z|')


class RegisterTest(unittest.TestCase):
    def test_median_is_duration_weighted(self):
        self.assertEqual(vr.note_name(vr.median_pitch(HIGH)), 'G5')   # the tied g' lasts longest
        self.assertIsNone(vr.median_pitch(HEADER + group('intro', '"G"z16|Z|', 'G4B4d4g4|Z|')))

    def test_high_melody_moves_down_for_a_male_voice_only(self):
        self.assertEqual(vr.choose(vr.median_pitch(HIGH), 'male'), -1)
        for voice in ('female', 'duet', 'any'):
            self.assertEqual(vr.choose(vr.median_pitch(HIGH), voice), 0, voice)

    def test_low_melody_moves_up_for_a_female_voice_only(self):
        self.assertEqual(vr.choose(vr.median_pitch(LOW), 'female'), 1)
        self.assertEqual(vr.choose(vr.median_pitch(LOW), 'male'), 0)

    def test_explicit_choices_ignore_the_voice(self):
        self.assertEqual(vr.choose(76, 'male', 'keep'), 0)
        self.assertEqual(vr.choose(60, 'any', 'down'), -1)
        self.assertEqual(vr.choose(60, 'any', 'down2'), -2)
        self.assertEqual(vr.choose(60, 'any', 'up'), 1)
        self.assertEqual(vr.choose(None, 'male'), 0)
        with self.assertRaises(ValueError): vr.choose(60, 'male', 'octave')

    def test_shift_moves_only_the_sung_notes(self):
        before = abc.parse(HIGH)
        moved, octaves, median = vr.fit(HIGH, 'male')
        after = abc.parse(moved)
        self.assertEqual(octaves, -1)
        self.assertEqual([[t, p - 12, d] for t, p, d in before.voices['Vocal'].notes], [list(n) for n in after.voices['Vocal'].notes])
        self.assertEqual(after.voices['Ins'].notes, before.voices['Ins'].notes)
        for name in ('Vocal', 'Ins'):
            self.assertEqual(after.voices[name].chords, before.voices[name].chords)
            self.assertEqual(after.voices[name].bars, before.voices[name].bars)
            self.assertEqual(after.voices[name].keys, before.voices[name].keys)
        self.assertEqual(vr.note_name(vr.median_pitch(moved)), 'G4')
        self.assertIn('"G"E4E4F4=F4|"D"G8G4D4-|', moved, 'lowercase notes become uppercase; accidentals and ties stay')
        self.assertIn('"Em"D4G12|[K:D]"D"A4^G4A8|', moved, 'chords and inline key changes stay in place')
        self.assertEqual(vr.fit(moved, 'male')[1], 0, 'a melody already in register is left alone')

    def test_shift_up_and_round_trip(self):
        up = vr.shift_octaves(LOW, 1)
        self.assertIn('"G"G4B4D4G4|"D"D8z8|', up)
        self.assertEqual(vr.shift_octaves(up, -1), LOW.strip() + '\n')

    def test_two_octaves_down_keeps_a_valid_score(self):
        moved, octaves, median = vr.fit(HIGH, 'any', 'down2')
        self.assertEqual(octaves, -2)
        self.assertEqual(vr.note_name(vr.median_pitch(moved)), 'G3')
        self.assertIn('"G"E,4E,4F,4=F,4|"D"G,8G,4D,4-|', moved)
        self.assertEqual(vr.shift_octaves(moved, 2), HIGH.strip() + '\n')

    def test_keep_returns_the_score_untouched(self):
        self.assertIs(vr.fit(HIGH, 'male', 'keep')[0], HIGH)



if __name__ == '__main__':
    unittest.main()
