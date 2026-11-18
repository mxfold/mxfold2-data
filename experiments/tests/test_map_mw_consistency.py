import itertools
import unittest

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from scripts.chemical_consistency.map_mw_metrics import binary_metrics, from_partners, score_profile, summarize
from scripts.chemical_consistency.map_mw_report import common_frames, merge_shard_summaries
import pandas as pd
from mxfold2.loss.nonparametric_rank import mann_whitney_rank_loss
import torch


class SingleStructureConsistencyTest(unittest.TestCase):
    def test_all_binary_splits_match_frozen_training_objective_and_independent_metrics(self):
        values = np.array([0., 0., 2., 1., 3.])
        for states in itertools.product((0, 1), repeat=5):
            u = np.asarray(states)
            result = binary_metrics(values, u)
            loss = float(mann_whitney_rank_loss(torch.tensor(values), torch.tensor(1-u, dtype=torch.float64)))
            self.assertAlmostEqual(result['auroc_mw_neutral'], 1-loss, places=12)
            if sum(states) in (0, 5):
                self.assertIsNone(result['auroc'])
                self.assertIsNone(result['spearman'])
            else:
                self.assertAlmostEqual(result['auroc'], roc_auc_score(u, values), places=12)
                self.assertAlmostEqual(result['spearman'], spearmanr(u, values).statistic, places=12)

    def test_stratification_mask_and_no_clipping(self):
        profile = {'sequence': 'AAACCCGU', 'modifier': 'DMS',
                   'values': [-999., 1., 2., 3., 4., float('nan'), 100., -20.]}
        rows = score_profile(profile, [1, 0, 1, 0, 1, 0, 1, 0])
        self.assertEqual([r['stratum'] for r in rows], ['A', 'C'])
        self.assertEqual([r['n_observed'] for r in rows], [2, 2])
        self.assertEqual([r['auroc'] for r in rows], [1., 1.])
        self.assertEqual([r['spearman'] for r in rows], [1., 1.])

    def test_probability_inputs_and_invalid_partners_rejected(self):
        with self.assertRaises(ValueError):
            binary_metrics([0., 1.], [0.2, 0.8])
        with self.assertRaises(ValueError):
            from_partners('ACGU', [0, 4, 0, 0, 0])
        np.testing.assert_array_equal(from_partners('ACGU', [0, 4, 3, 2, 1]), [0, 0, 0, 0])

    def test_constant_reactivity_and_insufficient_observations(self):
        result = binary_metrics([0.2, 0.2], [0, 1])
        self.assertEqual(result['auroc'], 0.5)
        self.assertIsNone(result['spearman'])
        self.assertIsNone(binary_metrics([0.2], [1])['auroc_mw_neutral'])

    def test_replicate_average_precedes_sequence_average(self):
        rows = []
        for profile_id, unit, auc in [(0, 'x', 1.), (1, 'x', 0.), (2, 'y', 1.)]:
            rows.append({'dataset': 'test', 'modifier': 'SHAPE', 'stratum': 'all',
                         'profile_id': profile_id, 'unit_id': unit, 'cluster': unit,
                         'auroc': auc, 'spearman': 2*auc-1, 'auroc_mw_neutral': auc,
                         'n_observed': 2, 'prediction_constant': False})
        result = summarize(pd.DataFrame(rows))[0]
        self.assertEqual(result['auroc_sequence_macro'], 0.75)
        self.assertEqual(result['spearman_sequence_macro'], 0.5)
        self.assertEqual(result['units'], 2)

    def test_model_comparison_uses_identical_defined_strata(self):
        a = pd.DataFrame({'profile_id': [0, 0, 1], 'stratum': ['A', 'C', 'A'],
                          'auroc': [0.8, 0.7, 0.9], 'spearman': [0.3, 0.2, 0.4]})
        b = a.copy()
        b.loc[0, ['auroc', 'spearman']] = np.nan
        frames = common_frames([a, b])
        for frame in frames:
            self.assertEqual(list(zip(frame.profile_id, frame.stratum)), [(0, 'C'), (1, 'A')])

    def test_shard_aggregation_weights_rnas_and_defined_coverage(self):
        def row(unit, auc, rho):
            return {'dataset': 'full', 'modifier': 'SHAPE', 'profile_id': unit,
                    'unit_id': unit, 'cluster': unit, 'auroc': auc, 'spearman': rho,
                    'auroc_mw_neutral': 0.5 if auc is None else auc,
                    'n_observed': 4, 'prediction_constant': auc is None}
        first = pd.DataFrame([row('a', 1., 0.8)])
        second = pd.DataFrame([row('b', 0.6, 0.2), row('c', None, None)])
        merged = merge_shard_summaries([summarize(first), summarize(second)])[0]
        combined = summarize(pd.concat([first, second], ignore_index=True))[0]
        for metric in ('auroc', 'spearman', 'auroc_mw_neutral'):
            self.assertAlmostEqual(merged[metric + '_sequence_macro'], combined[metric + '_sequence_macro'])
            self.assertEqual(merged['n_units_' + metric + '_defined'], combined['n_units_' + metric + '_defined'])


if __name__ == '__main__':
    unittest.main()
