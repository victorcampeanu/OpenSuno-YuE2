"""The cover arrangement pass: forced transcription, model-written chords and instrumental lines."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'app'))
sys.path.insert(0, str(ROOT / 'model'))

import cover_arrangement as ca
from vendor import yue2_abc as abc_tools

HEADER = ('X:1\nT:\nM:4/4\nL:1/16\nQ:1/4=100\n'
          'V: Vocal clef=treble name="Vocal Melody" snm="Vocal"\nV: Ins clef=treble name="Ins Melody" snm="Inst."\nK:C\n')
# A SheetSage2 melody-only transcription: a silent intro, a sung verse, a heard riff and a meter change.
TRANSCRIPTION = HEADER + (
    '% intro\n'
    'V: Vocal\nZ2|\nV: Ins\nZ2|\n'
    '% verse\n'
    'V: Vocal\nc4d4e4f4|g8z8|\nV: Ins\nZ2|\n'
    '% interlude\n'
    'V: Vocal\nZ|\nV: Ins\nc2d2e2f2g2a2b2c\'2|\n'
    'V: Vocal\nM:3/4\nz12|\nV: Ins\nM:3/4\nc4d4e4|\n'
)


class ScriptedWriter:
    """Replays proposals in order and records the text each proposal continued from."""

    def __init__(self, proposals):
        self.proposals = list(proposals)
        self.text = ''
        self.contexts = []

    def commit(self, text):
        self.text += text

    def propose(self, max_tokens, stop, first=None):
        self.contexts.append((self.text, max_tokens, first))
        reply = self.proposals.pop(0) if self.proposals else ''
        # A real writer stops early; the caller must cope with either form.
        for n in range(1, len(reply) + 1):
            if stop(reply[:n]):
                return reply[:n]
        return reply


class RestBarTest(unittest.TestCase):
    def test_full_bar_rests_use_supported_durations(self):
        self.assertEqual(ca.rest_bar((4, 4), 16), 'z16')
        self.assertEqual(ca.rest_bar((3, 4), 16), 'z12')
        self.assertEqual(ca.rest_bar((6, 8), 16), 'z12')
        self.assertEqual(ca.rest_bar((4, 4), 32), 'z32')
        self.assertEqual(ca.rest_bar((5, 4), 16), 'z16z4')
        self.assertEqual(ca.rest_bar((1, 4), 16), 'z4')

    def test_bars_of_expands_multi_measure_rests(self):
        self.assertEqual(ca.bars_of('Z3|'), ['Z', 'Z', 'Z'])
        self.assertEqual(ca.bars_of('c4|Z2|d4|'), ['c4', 'Z', 'Z', 'd4'])
        with self.assertRaises(ca.ArrangementError):
            ca.bars_of('c4')


class ChordProposalTest(unittest.TestCase):
    def test_reads_the_chord_name_up_to_the_closing_quote(self):
        self.assertEqual(ca.read_chord('Gm"b'), 'Gm')
        self.assertEqual(ca.read_chord('F/A"'), 'F/A')
        self.assertEqual(ca.read_chord('Bbmaj7"z16|'), 'Bbmaj7')

    def test_unusable_names_are_rejected(self):
        self.assertIsNone(ca.read_chord('N.C."z'), 'outside the supported chord vocabulary')
        self.assertIsNone(ca.read_chord('Gm'), 'never closed')
        self.assertIsNone(ca.read_chord('"'), 'empty')
        self.assertIsNone(ca.read_chord(''))

    def test_after_a_barline_the_quote_is_committed_and_withdrawn_when_no_name_fits(self):
        writer = ScriptedWriter(['N.C."z', 'x"'])
        writer.commit('c4')
        self.assertEqual(ca.propose_chord(writer, '|'), (None, False))
        self.assertEqual(writer.text, 'c4|', 'the bar goes on without a chord')
        self.assertEqual([text for text, _, _ in writer.contexts], ['c4|"', 'c4|"'], 'both attempts saw the opened quote')
        writer = ScriptedWriter(['Am"c'])
        writer.commit('c4')
        self.assertEqual(ca.propose_chord(writer, '|'), ('Am', False))
        self.assertEqual(writer.text, 'c4|"Am"')

    def test_at_a_line_start_the_model_opens_the_chord_itself(self):
        writer = ScriptedWriter(['"Dm"c4'])
        self.assertEqual(ca.propose_chord(writer, ''), ('Dm', False))
        self.assertEqual(writer.text, '"Dm"')
        self.assertEqual(writer.contexts[0][2], ca.OPENING_QUOTE)
        writer = ScriptedWriter(['z4"Dm"', '"N.C."'])
        self.assertEqual(ca.propose_chord(writer, ''), (None, False), 'rests first (an unconstrained writer), then an unusable name')
        self.assertEqual(writer.text, '')
        self.assertEqual(len(writer.contexts), 2)

    def test_an_out_of_tune_proposal_is_corrected_before_it_is_committed(self):
        # E-flat minor melody (Eb Db Gb F): the model's "E" (a semitone off the key) becomes E-flat minor.
        melody = {3: 2.0, 1: 1.0, 6: 1.0, 5: 1.0}
        writer = ScriptedWriter(['E"e'])
        writer.commit('e4')
        self.assertEqual(ca.propose_chord(writer, '|', melody, 'Eb'), ('Ebm', True))
        self.assertEqual(writer.text, 'e4|"Ebm"', 'the score never contains the wrong chord for the model to copy')
        writer = ScriptedWriter(['"E"e'])
        self.assertEqual(ca.propose_chord(writer, '', melody, 'Eb'), ('Ebm', True))
        self.assertEqual(writer.text, '"Ebm"')


class HarmonyTest(unittest.TestCase):
    def test_chord_tones(self):
        self.assertEqual(ca.chord_tones('C'), (0, 4, {0, 4, 7}))
        self.assertEqual(ca.chord_tones('Ebm'), (3, 3, {3, 6, 10}))
        self.assertEqual(ca.chord_tones('Bbmaj7'), (10, 4, {10, 2, 5, 9}))
        self.assertEqual(ca.chord_tones('F/A'), (5, 4, {5, 9, 0}))
        self.assertEqual(ca.chord_tones('Gsus4')[1], None)
        self.assertEqual(ca.chord_tones('F#m7b5'), (6, 3, {6, 9, 0, 4}))

    def test_roots_outside_both_parallel_modes_are_out_of_the_question(self):
        self.assertIsNone(ca.chord_fit('E', {}, 'Eb'), 'a semitone above the tonic')
        self.assertIsNone(ca.chord_fit('A', {}, 'Eb'), 'the sharp fourth')
        self.assertEqual(ca.chord_fit('Ebm', {}, 'Eb'), 1.0, 'the parallel minor is allowed: the key label is trusted for its tonic only')
        self.assertEqual(ca.chord_fit('Cb', {}, 'Eb'), 1.0, 'flat sixth of the parallel minor')
        self.assertEqual(ca.chord_fit('G', {}, 'Cm'), 1.0, 'the major dominant of a minor key')
        self.assertIsNone(ca.chord_fit('D#', {}, 'E'), 'a major triad on the leading tone reaches the sharp fourth')
        self.assertIsNone(ca.chord_fit('D', {}, 'C'), 'a secondary dominant reaches the sharp fourth')
        self.assertEqual(ca.chord_fit('D', {}, 'E'), 1.0, 'the flat seventh of the parallel minor')
        self.assertIsNone(ca.chord_fit('C#', {}, 'E'), 'its third is the flat second')

    def test_the_sung_third_decides_major_or_minor(self):
        minor_melody = {3: 2.0, 6: 1.0, 10: 1.0}       # Eb Gb Bb
        major_melody = {3: 2.0, 7: 1.0, 10: 1.0}       # Eb G Bb
        self.assertIsNone(ca.chord_fit('Eb', minor_melody, 'Eb'), 'a major chord under a sung minor third')
        self.assertEqual(ca.chord_fit('Ebm', minor_melody, 'Eb'), 1.0)
        self.assertIsNone(ca.chord_fit('Ebm', major_melody, 'Eb'))
        self.assertEqual(ca.chord_fit('Eb', major_melody, 'Eb'), 1.0)
        both = {3: 2.0, 6: 1.0, 7: 1.0}
        self.assertEqual(ca.chord_fit('Eb', both, 'Eb'), 0.75, 'a passing minor third under a mostly major melody is fine')
        self.assertEqual(ca.chord_fit('Ebsus4', minor_melody, 'Eb'), 0.75, 'sus chords have no third to clash')

    def test_fit_is_the_share_of_sung_time_on_chord_tones(self):
        melody = {0: 1.0, 2: 1.0, 4: 1.0, 5: 1.0}      # c d e f
        self.assertEqual(ca.chord_fit('C', melody, 'C'), 0.5)
        self.assertEqual(ca.chord_fit('Dm', melody, 'C'), 0.5)
        self.assertEqual(ca.chord_fit('G7', melody, 'C'), 0.5)
        self.assertTrue(ca.acceptable('C', melody, 'C'))
        self.assertFalse(ca.acceptable('Am', {2: 1.0, 5: 1.0, 7: 1.0}, 'C'), 'none of the sung notes is a chord tone')

    def test_best_chord_keeps_the_previous_chord_while_it_fits(self):
        melody = {7: 1.0, 11: 1.0}                      # g b
        self.assertEqual(ca.best_chord(melody, 'C', previous='G'), 'G')
        self.assertEqual(ca.best_chord(melody, 'C', previous='Em'), 'Em')
        self.assertIn(ca.best_chord(melody, 'C', previous='F'), ('G', 'Em'), 'F fits nothing sung; a diatonic triad holding both notes wins')
        self.assertEqual(ca.best_chord({3: 2.0, 6: 1.0, 10: 1.0}, 'Eb', previous='Eb'), 'Ebm')
        self.assertEqual(ca.best_chord({}, 'Eb'), 'Eb', 'nothing sung: the tonic')
        self.assertEqual(ca.best_chord({}, 'Cm'), 'Cm')
        self.assertEqual(ca.best_chord({}, 'Cm', previous='Ab'), 'Ab', 'nothing sung: the previous chord stays')

    def test_best_chord_spells_roots_for_the_key(self):
        self.assertEqual(ca.best_chord({8: 1.0, 0: 1.0, 3: 1.0}, 'Eb'), 'Ab')
        self.assertEqual(ca.best_chord({8: 1.0, 0: 1.0, 3: 1.0}, 'E'), 'G#', 'G# B# D# is the G-sharp major triad, spelled with sharps')
        self.assertEqual(ca.best_chord({8: 1.0, 11: 1.0, 3: 1.0}, 'E'), 'G#m')
        self.assertEqual(ca.root_variants('E', 'Eb'), ['Eb', 'F'])
        self.assertEqual(ca.root_variants('Em7/G', 'D'), ['D#m7/G', 'Fm7/G'])

    def test_tune_prefers_the_model_then_its_respelt_root_then_the_best_chord(self):
        melody = {3: 2.0, 7: 1.0, 10: 1.0}              # Eb G Bb: E-flat major
        self.assertEqual(ca.tune('Eb', melody, 'Eb', None), ('Eb', False))
        self.assertEqual(ca.tune('E', melody, 'Eb', None), ('Eb', True), 'the E/Eb tokenizer trap')
        self.assertEqual(ca.tune('E', {3: 2.0, 6: 1.0, 10: 1.0}, 'Eb', None), ('Ebm', True))
        self.assertEqual(ca.tune('A', {0: 1.0, 4: 1.0}, 'C', 'F'), ('F', True), 'F is kept when it still fits the bar')
        self.assertEqual(ca.tune('A', {0: 1.0, 4: 1.0}, 'C', None), ('C', True))
        self.assertEqual(ca.tune('A', {0: 1.0, 4: 1.0}, 'C', None), ('C', True))
        self.assertEqual(ca.tune('E', {}, 'Eb', 'Cm'), ('Eb', True), 'nothing sung: the respelt root, not the previous chord')
        self.assertEqual(ca.tune('E', None, 'Eb', None), ('E', False), 'no melody: checks off')
        self.assertEqual(ca.tune('Abm', {}, 'E', None), ('G#m', False), 'accepted chords are respelled for the key')
        self.assertEqual(ca.tune('Abm7/Eb', {}, 'E', None), ('G#m7/D#', False))
        self.assertEqual(ca.respell('Cb', 'Eb'), 'B')

    def test_borrowed_chords_are_kept_only_when_the_melody_asks_for_them(self):
        self.assertTrue(ca.diatonic('G', 'Cm'), 'the major dominant belongs to a minor key')
        self.assertFalse(ca.diatonic('G#', 'E'))
        self.assertTrue(ca.diatonic('G#m', 'E'))
        no_third = {3: 12, 8: 2, 1: 2}                    # D# G# C#: G# major and G# minor fit alike
        self.assertEqual(ca.tune('G#', no_third, 'E', None), ('G#m', True), 'the diatonic chord wins a tie')
        self.assertEqual(ca.tune('C', {4: 6, 1: 2}, 'E', None), ('A', True), 'a diatonic chord fits E–C# better than the borrowed C')
        self.assertEqual(ca.tune('C', {0: 2, 4: 1, 7: 1}, 'E', None), ('C', False), 'C–E–G: nothing diatonic fits as well')
        self.assertEqual(ca.tune('B#', {}, 'E', 'B'), ('B', True), 'nothing sung: the diatonic previous chord stays')
        self.assertEqual(ca.tune('G#', {3: 2, 8: 0.5, 1: 0.5}, 'E', 'B'), ('G#m', True), 'the best diatonic chord, not merely the previous one, is the yardstick')
        self.assertEqual(ca.tune('B#', {}, 'E', None), ('E', True), 'nothing sung, nothing before: the tonic')
        self.assertEqual(ca.tune('Ab', {}, 'Cm', 'Ab'), ('Ab', False), 'the flat sixth belongs to the minor key')
        self.assertEqual(ca.best_chord({8: 4}, 'Eb', 'Eb', strict=True), 'Ab')

    def test_melody_by_bar_weights_pitch_classes_by_sung_time_and_splits_ties(self):
        abc = HEADER + 'V: Vocal\nc8e8-|e4z12|\nV: Ins\nZ2|\n'
        bars = ca.melody_by_bar(abc_tools.parse(abc))
        self.assertEqual(len(bars), 2)
        self.assertEqual(bars[0], {0: 2, 4: 2})
        self.assertEqual(bars[1], {4: 1}, 'the tied E counts only where it sounds in the second bar')
        self.assertEqual(ca.melody_by_bar(abc_tools.parse(HEADER + 'V: Vocal\nZ|\nV: Ins\nZ|\n')), [{}])

    def test_where_nothing_is_sung_the_heard_instrumental_line_guides_the_chord(self):
        abc = HEADER + 'V: Vocal\nZ|c16|\nV: Ins\nf4a4c\'4f\'4|g16|\n'
        self.assertEqual(ca.melody_by_bar(abc_tools.parse(abc)), [{5: 2, 9: 1, 0: 1}, {0: 4}],
                         'the intro riff counts in its silent bar; the sung bar ignores the accompaniment')
        # The transcribed intro of an E-flat song: the model holds E-flat under an A-flat bar; the heard notes correct it.
        abc = HEADER.replace('K:C', 'K:Eb') + '% intro\nV: Vocal\nZ2|\nV: Ins\nF4E2E4E4E2|A4A2A4A6|\n'
        writer = ScriptedWriter(['"Eb"z', 'Eb"z'])
        result = ca.arrange(abc, writer)
        self.assertEqual([c for _, c in abc_tools.parse(result.abc).voices['Vocal'].chords], ['Eb', 'Ab'])
        self.assertEqual(result.corrected_chords, 1)

    def test_instrumental_lines_must_play_their_chords_and_stay_in_key(self):
        header = HEADER.strip().splitlines()
        meter = (4, 4)
        voice = ca.parse_ins_line('c4e4g4c\'4|f4a4c\'4f\'4|', 2, header, meter, 'C')
        self.assertEqual(ca.ins_fit(voice, ['C', 'F'], 'C'), (1.0, 0.0))
        self.assertTrue(ca.in_tune_ins_line(voice, ['C', 'F'], 'C'))
        self.assertFalse(ca.in_tune_ins_line(voice, ['G', 'G'], 'C'), 'a good line under the wrong chords')
        self.assertTrue(ca.in_tune_ins_line(voice, ['G', 'Dm'], 'C'), 'half the notes on chord tones is the floor')
        wander = ca.parse_ins_line('F4E2E4E4E2|^F4F2F4F4F2|A4A2A4A6|G4G2G4G4G2|', 4, header, meter, 'Eb')
        on_chord, out_of_key = ca.ins_fit(wander, ['Eb'] * 4, 'Eb')
        self.assertLess(on_chord, ca.MIN_INS_FIT, 'the chromatic wander plays almost none of its chord')
        self.assertFalse(ca.in_tune_ins_line(wander, ['Eb'] * 4, 'Eb'))
        chromatic = ca.parse_ins_line('c4^c4d4^d4|', 1, header, meter, 'C')
        self.assertGreater(ca.ins_fit(chromatic, ['C'], 'C')[1], ca.MAX_INS_OUT_OF_KEY)
        self.assertEqual(ca.ins_fit(ca.parse_ins_line('z16|', 1, header, meter, 'C'), [None], 'C'), (1.0, 0.0), 'nothing played fits')
        self.assertEqual(ca.ins_fit(voice, [None, None], 'C')[0], 1.0, 'bars without a chord are not held against the line')


class ArrangeTest(unittest.TestCase):
    def test_chords_are_written_where_the_transcription_has_none_and_the_melody_is_kept(self):
        # intro: 2 bars, verse: 2 bars, interlude: 1 bar, 3/4 bar: 1 bar = 6 chord proposals plus one Ins line.
        writer = ScriptedWriter(['"C"z', 'F"z', 'c2e2g2c\'2c2e2g2c\'2|e4g4c\'4e\'4|\nV: Vocal',
                                 '"C"c', 'G"g', '"Am"z', '"F"z'])
        result = ca.arrange(TRANSCRIPTION, writer)
        before, after = abc_tools.parse(TRANSCRIPTION), abc_tools.parse(result.abc)
        self.assertEqual(after.voices['Vocal'].notes, before.voices['Vocal'].notes)
        self.assertEqual(after.voices['Vocal'].bars, before.voices['Vocal'].bars)
        self.assertEqual(result.abc, writer.text, 'the arrangement is exactly what was committed')
        self.assertEqual((result.bars, result.chords, result.kept_chords, result.ins_lines, result.ins_failed),
                         (6, 6, 0, 1, 0))
        lines = result.abc.splitlines()
        self.assertIn('"C"z16|"F"z16|', lines, 'full-measure Vocal rests become explicit rests carrying a chord')
        self.assertIn('c2e2g2c\'2c2e2g2c\'2|e4g4c\'4e\'4|', lines, 'the model wrote the silent intro line')
        self.assertIn('"C"c4d4e4f4|"G"g8z8|', lines)
        self.assertIn('"F"z12|', lines, 'the 3/4 rest bar follows the meter change')
        self.assertIn('M:3/4', lines)
        self.assertIn('c2d2e2f2g2a2b2c\'2|', lines, 'the heard riff is kept')
        self.assertEqual([c for _, c in after.voices['Vocal'].chords], ['C', 'F', 'C', 'G', 'Am', 'F'])

    def test_the_model_continues_from_the_previous_bar_without_a_barline(self):
        writer = ScriptedWriter(['"C"z', 'F"z', 'Z2|\n', '"C"c', 'G"g', '"Am"z', '"F"z'])
        ca.arrange(TRANSCRIPTION, writer)
        contexts = [text for text, _, _ in writer.contexts]
        self.assertTrue(contexts[0].endswith('% intro\nV: Vocal\n'), 'first bar: at the line start')
        self.assertEqual(writer.contexts[0][2], ca.OPENING_QUOTE, 'the first token must open a chord')
        self.assertTrue(contexts[1].endswith('V: Vocal\n"C"z16|"'), 'second bar: the merged barline-and-quote is committed')
        self.assertIsNone(writer.contexts[1][2])
        self.assertTrue(contexts[2].endswith('V: Ins\n'), 'the silent Ins line is proposed after its voice header')
        self.assertEqual(writer.contexts[2][1], ca.INS_TOKENS_PER_BAR * 2 + ca.CHORD_TOKENS)
        self.assertTrue(contexts[4].endswith('"C"c4d4e4f4|"'))

    def test_transcribed_chords_are_kept_and_not_proposed_again(self):
        abc = TRANSCRIPTION.replace('c4d4e4f4|g8z8|', '"Dm"c4d4e4f4|g8z8|')
        writer = ScriptedWriter(['"C"z', 'F"z', 'Z2|\n', 'G"g', '"Am"z', '"F"z'])
        result = ca.arrange(abc, writer)
        self.assertEqual((result.chords, result.kept_chords), (5, 1))
        self.assertIn('"Dm"c4d4e4f4|"G"g8z8|', result.abc.splitlines())

    def test_a_model_that_never_offers_a_chord_leaves_a_valid_score(self):
        writer = ScriptedWriter(['z' * 8] * 10)
        result = ca.arrange(TRANSCRIPTION, writer)
        self.assertEqual((result.chords, result.ins_lines, result.ins_failed), (0, 0, 1))
        after = abc_tools.parse(result.abc)
        self.assertEqual(after.voices['Vocal'].chords, [])
        self.assertIn('z16|z16|', result.abc.splitlines())

    def test_a_bad_instrumental_line_is_retried_then_left_as_heard(self):
        # Wrong bar count, then a chord in the Ins voice, so both attempts are rejected.
        writer = ScriptedWriter(['"C"z', 'F"z', 'c4d4e4f4|\n', '"G"c4d4e4f4|c4d4e4f4|\n'])
        result = ca.arrange(TRANSCRIPTION, writer)
        self.assertEqual((result.ins_lines, result.ins_failed), (0, 1))
        self.assertIn('% intro\nV: Vocal\n"C"z16|"F"z16|\nV: Ins\nZ2|\n', result.abc)
        self.assertEqual(len(writer.contexts), 2 + ca.INS_ATTEMPTS + 4 * 2, 'every Ins attempt, then two failed tries on each remaining bar')

    def test_an_out_of_tune_instrumental_line_is_refused_and_a_fitting_retry_taken(self):
        # Intro chords C and F; the first line is well formed but mostly off both chords, the second plays them.
        writer = ScriptedWriter(['"C"z', 'F"z', 'g4b4d\'4b\'4|d4e4a4b4|\n', 'c4e4g4c\'4|f4a4c\'4f\'4|\n',
                                 '"C"c', 'G"g', '"Am"z', '"F"z'])
        result = ca.arrange(TRANSCRIPTION, writer)
        self.assertEqual((result.ins_lines, result.ins_rejected, result.ins_failed, result.corrected_chords), (1, 1, 0, 0))
        self.assertIn('c4e4g4c\'4|f4a4c\'4f\'4|', result.abc.splitlines())
        self.assertNotIn('g4b4d\'4b\'4|d4e4a4b4|', result.abc)

    def test_out_of_tune_chords_are_corrected_and_the_wrong_ones_never_reach_the_score(self):
        # An E-flat song whose verse sings the minor third: the model offers E three times; the sung bars get E-flat minor.
        abc = HEADER.replace('K:C', 'K:Eb') + '% verse\nV: Vocal\ne4_d4^f4f4|^f8e8|z16|\nV: Ins\nZ3|\n'
        writer = ScriptedWriter(['"E"e', 'E"^f', 'E"z', 'Z3|\n'])
        result = ca.arrange(abc, writer)
        self.assertEqual((result.chords, result.corrected_chords), (3, 3))
        chords = [c for _, c in abc_tools.parse(result.abc).voices['Vocal'].chords]
        self.assertEqual(chords[:2], ['Ebm', 'Ebm'])
        self.assertNotIn('"E"', result.abc)
        self.assertEqual(chords[2], 'Eb', 'nothing sung in the last bar: the respelt root is taken as the model meant it')
        self.assertEqual(abc_tools.parse(result.abc).voices['Vocal'].notes, abc_tools.parse(abc).voices['Vocal'].notes)

    def test_checks_can_be_turned_off_to_take_proposals_as_written(self):
        abc = HEADER.replace('K:C', 'K:Eb') + '% verse\nV: Vocal\ne4_d4^f4f4|^f8e8|\nV: Ins\nZ2|\n'
        writer = ScriptedWriter(['"E"e', 'E"^f'])
        result = ca.arrange(abc, writer, in_tune=False)
        self.assertEqual((result.chords, result.corrected_chords), (2, 0))
        self.assertEqual([c for _, c in abc_tools.parse(result.abc).voices['Vocal'].chords], ['E', 'E'])

    def test_instrumental_fill_can_be_turned_off(self):
        writer = ScriptedWriter(['"C"z', 'C"z', '"C"c', 'C"g', '"C"z', '"C"z'])
        result = ca.arrange(TRANSCRIPTION, writer, fill_instrumental=False)
        self.assertEqual((result.ins_lines, result.chords), (0, 6))
        self.assertEqual(len(writer.contexts), 6)

    def test_progress_counts_bars(self):
        seen = []
        ca.arrange(TRANSCRIPTION, ScriptedWriter([]), on_progress=lambda done, total: seen.append((done, total)))
        self.assertEqual(seen, [(i, 6) for i in range(1, 7)])

    def test_scores_outside_the_dialect_are_refused(self):
        with self.assertRaises(ca.ArrangementError):
            ca.arrange('X:1\nT:\nM:4/4\n', ScriptedWriter([]))
        with self.assertRaises(ca.ArrangementError):
            ca.arrange(TRANSCRIPTION.replace('% verse\nV: Vocal', '% verse\nV: Lead'), ScriptedWriter([]))


try:
    import mlx.core as mx
    import generate as engine
except ImportError:  # pragma: no cover - CUDA machines
    mx = None


@unittest.skipIf(mx is None, 'MLX is not installed')
class ScoreWriterTest(unittest.TestCase):
    """The MLX writer keeps its KV cache in step with the re-encoded committed text."""

    class Tokenizer:
        def encode(self, text):
            # Digit pairs merge into one token so a boundary can re-tokenise, like the BPE's |" merge.
            ids, i = [], 0
            while i < len(text):
                if text[i].isdigit() and i + 1 < len(text) and text[i + 1].isdigit():
                    ids.append(1000 + int(text[i:i + 2])); i += 2
                else:
                    ids.append(ord(text[i])); i += 1
            return ids

        def decode(self, ids):
            return ''.join(chr(i) if i < 1000 else f'{i - 1000:02d}' for i in ids)

    class Model:
        """Emits the digits of a fixed reply one token per step; records every cache write."""

        def __init__(self, reply):
            from types import SimpleNamespace
            self.model = SimpleNamespace(layers=[object()])
            self.reply, self.steps = reply, []

        def ar_step(self, tokens, caches, window=None):
            batch, n = tokens.shape
            for cache in caches:
                self.steps.append((cache.offset, n))
                cache.update(mx.zeros((batch, 1, n, 4), mx.float32), mx.zeros((batch, 1, n, 4), mx.float32))
            position = caches[0].offset
            logits = mx.full((batch, engine.VOCAB), -1e4, dtype=mx.float32)
            logits[:, ord(self.reply[position % len(self.reply)])] = 0.0
            return logits

    def writer(self, reply='|"G"z', min_tokens=1):
        model = self.Model(reply)
        sampling = engine.Sampling(temperature=0, min_tokens=min_tokens, max_tokens=100)
        return model, engine.ScoreWriter(model, self.Tokenizer(), [engine.EOD, engine.ABC_START], sampling, 1)

    def test_proposals_continue_the_committed_text_and_the_cache_follows_commits(self):
        model, w = self.writer()
        w.commit('c4|')
        text = w.propose(8, lambda t: t.count('"') >= 2)
        self.assertEqual(text, '|"G"')
        self.assertEqual(w.fed, [engine.EOD, engine.ABC_START] + w.tok.encode('c4|') + w.tok.encode('|"G"'))
        self.assertEqual(model.steps[0], (0, 5), 'the request and the committed text were prefilled together')
        w.commit('"F"d4')
        w._sync()
        self.assertEqual(w.fed, [engine.EOD, engine.ABC_START] + w.tok.encode('c4|"F"d4'))
        self.assertEqual({c.offset for c in w.caches}, {len(w.fed)})
        self.assertEqual(model.steps[-1], (5, 5), 'trimmed back to the barline, then fed only the disagreeing tail')

    def test_retokenised_boundaries_are_refed_not_duplicated(self):
        model, w = self.writer(reply='1')
        w.commit('z1')
        w.propose(1, lambda t: True)          # sampled '1' after the single-digit token '1'
        w.commit('2')                         # the committed text now merges into the pair token 12
        w._sync()
        self.assertEqual(w.fed, [engine.EOD, engine.ABC_START, ord('z'), 1012])
        self.assertEqual({c.offset for c in w.caches}, {4})

    def test_the_first_token_can_be_constrained_to_open_a_chord(self):
        model, w = self.writer(reply='z')          # the model would rather write a rest
        w.commit('V: Vocal\n')
        self.assertEqual(w.propose(1, lambda t: True), 'z')
        w = engine.ScoreWriter(model, self.Tokenizer(), [engine.EOD, engine.ABC_START], engine.Sampling(temperature=0, min_tokens=1), 1)
        w.commit('V: Vocal\n')
        # Every char token is a candidate; only the quote matches, so it wins despite the model's preference.
        self.assertEqual(w.propose(1, lambda t: True, first='"'), '"')
        self.assertEqual(int(w.first_mask('"').argmax()), ord('"'))
        with self.assertRaises(ValueError):
            w.first_mask('no such token text')

    def test_the_end_token_stops_a_proposal(self):
        model, w = self.writer(min_tokens=0)

        def ar_step(tokens, caches, window=None):
            for cache in caches:
                cache.update(mx.zeros((1, 1, tokens.shape[1], 4)), mx.zeros((1, 1, tokens.shape[1], 4)))
            return mx.where(mx.arange(engine.VOCAB) == engine.ABC_END, 0.0, -1e4)[None]
        model.ar_step = ar_step
        w.commit('c4')
        self.assertEqual(w.propose(4, lambda t: False), '')
        self.assertEqual(w.fed, [engine.EOD, engine.ABC_START, ord('c'), ord('4')], 'nothing was fed after the end token')


if __name__ == '__main__':
    unittest.main()
