import unittest

import numpy as np

from scripts.chemical_consistency.metrics import auc_binary_truth, correlation, score_profile, thresholds


class ChemicalConsistencyTests(unittest.TestCase):
    def test_auc_orientation_and_ties(self):
        self.assertEqual(auc_binary_truth([0, 0, 1, 1], [.1, .2, .8, .9]), 1.)
        self.assertEqual(auc_binary_truth([0, 0, 1, 1], [.5] * 4), .5)
        self.assertEqual(auc_binary_truth([0, 0, 1, 1], [.8, .9, .1, .2]), 0.)
        self.assertIsNone(auc_binary_truth([1, 1], [.1, .9]))

    def test_median_labels_and_missing(self):
        p = {'dataset': 'x', 'modifier': 'SHAPE', 'sequence': 'ACGU',
             'values': [.1, -.2, .8, -999.]}
        self.assertAlmostEqual(thresholds([p])[('x', 'SHAPE')], .1)
        r = score_profile(p, np.array([.4, .1, .9, .5]), .1)
        self.assertEqual((r['n_observed'], r['n_positive'], r['n_negative']), (3, 1, 2))
        self.assertEqual(r['auroc'], 1.)
        self.assertEqual(r['spearman_all'], 1.)

    def test_dms_restricts_ac_and_constant_prediction_is_undefined(self):
        p = {'dataset': 'x', 'modifier': 'DMS', 'sequence': 'ACGU',
             'values': [.1, .8, .6, .7]}
        r = score_profile(p, np.array([.5] * 4), .4)
        self.assertEqual(r['n_observed'], 2)
        self.assertEqual(r['auroc'], .5)
        self.assertIsNone(r['spearman_all'])
        self.assertIsNone(r['spearman_probe_balanced'])
        self.assertIsNone(correlation([1, 2], [.4, .4]))

    def test_invalid_unpaired_probability_is_rejected(self):
        p = {'dataset': 'x', 'modifier': 'SHAPE', 'sequence': 'AC', 'values': [.1, .8]}
        with self.assertRaises(ValueError):
            score_profile(p, np.array([.2, 1.2]), .4)
