import unittest

from transcriber.notation_sheetsage2 import BeatEvent, repair_isolated_missing_beats


class MissingBeatTest(unittest.TestCase):
    def beats(self, times=(0, .72, 2.16, 2.88), ids=(4, 1, 3, 4)):
        return [BeatEvent(t, n, 4, 4, i + 1) for i, (t, n) in enumerate(zip(times, ids))]

    def test_repairs_isolated_gap_without_changing_original_events(self):
        original = self.beats()
        repaired, notes = repair_isolated_missing_beats(original)
        self.assertEqual([b.beat_id for b in repaired], [4, 1, 2, 3, 4])
        self.assertAlmostEqual(repaired[2].time, 1.44)
        self.assertEqual(repaired[:2] + repaired[3:], original)
        self.assertEqual(len(notes), 1)

    def test_does_not_repair_inconsistent_timing(self):
        original = self.beats(times=(0, .72, 1.44, 2.16))
        self.assertEqual(repair_isolated_missing_beats(original), (original, []))

    def test_does_not_repair_multiple_missing_beats(self):
        original = self.beats(ids=(4, 1, 4, 1))
        self.assertEqual(repair_isolated_missing_beats(original), (original, []))

    def test_leaves_valid_grid_unchanged(self):
        original = self.beats(times=(0, .72, 1.44, 2.16), ids=(1, 2, 3, 4))
        self.assertEqual(repair_isolated_missing_beats(original), (original, []))
