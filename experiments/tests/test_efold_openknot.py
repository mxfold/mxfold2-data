import numpy as np
import unittest

from scripts.efold_hf.openknot_analysis import score, valid_values


class OpenKnotScoreTests(unittest.TestCase):
    def test_official_structure_score_and_sparse_mask(self):
        # Two stacked stems cross, matching the official (([[..))..]] example shape.
        pairs = [(0, 7), (1, 6), (2, 11), (3, 10)]
        values = np.array([.1, .6, .2, .24, .5, .125, .3, .3, .2, .1, .3, .2])
        seq = 'A' * len(values)
        result = score(seq, pairs, values, np.ones(len(values), dtype=bool))
        self.assertAlmostEqual(result['classic_official_span'], 75.)
        self.assertAlmostEqual(result['cpq_official_span'], 50.)
        self.assertAlmostEqual(result['openknot_official_span'], 62.5)
        valid = np.ones(len(values), dtype=bool)
        valid[1] = False
        result = score(seq, pairs, values, valid)
        self.assertAlmostEqual(result['classic_official_span'], 75.)
        self.assertAlmostEqual(result['classic_observed_only'], 900 / 11)
        self.assertAlmostEqual(result['cpq_official_span'], 50.)
        self.assertAlmostEqual(result['cpq_observed_only'], 400 / 7)


    def test_flanking_crossing_is_half_weighted(self):
        pairs = [(0, 7), (1, 6), (2, 11), (3, 10)]
        values = np.array([.1, .6, .2, .24, .5, .125, .3, .3, .2, .1, .3, .2])
        valid = np.array([False, False] + [True] * 10)
        result = score('A' * 12, pairs, values, valid)
        self.assertAlmostEqual(result['cpq_official_span'], 25.)


    def test_no_pseudoknot_and_singletons(self):
        values = np.array([.125, .126, .5, np.nan])
        valid = np.isfinite(values)
        result = score('ACGU', [(0, 3)], values, valid)
        self.assertAlmostEqual(result['classic_observed_only'], 200 / 3)
        self.assertEqual(result['cpq_observed_only'], 0)
        self.assertAlmostEqual(result['openknot_observed_only'], 100 / 3)


    def test_dms_measurements_only_admissible_for_a_and_c(self):
        values = np.array([.1, -.2, .1, np.nan])
        self.assertEqual(valid_values('ACGU', 'DMS', values).tolist(), [True, True, False, False])


    def test_bad_pairing_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'multiple pairing'):
            score('AAAA', [(0, 2), (0, 3)], np.array([.1] * 4), np.ones(4, dtype=bool))
