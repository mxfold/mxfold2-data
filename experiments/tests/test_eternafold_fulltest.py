import unittest

import numpy as np
import pandas as pd

from scripts.chemical_consistency.map_mw_metrics import summarize
from scripts.eternafold.fulltest_new_metrics import (
    attach_benchmark_ids, common_frames, merge_summaries, probe_scores, task_index,
)


class FullTestEvaluationTest(unittest.TestCase):
    def test_sge_non_array_job_environment(self):
        self.assertEqual(task_index('undefined'), 1)
        self.assertEqual(task_index(None), 1)
        self.assertEqual(task_index('16'), 16)

    def test_original_replicates_retained_and_highest_snr_matches_frozen_rows(self):
        raw = pd.DataFrame({'raw_row_id': [8, 12, 20], 'sequence': ['AAACCC'] * 3,
                            'modifier': ['DMS'] * 3, 'SNR': [1., 2., 2.],
                            'reactivity': ['[0]', '[1]', '[2]']}).set_index('raw_row_id', drop=False)
        frozen = pd.DataFrame({'sequence': ['AAACCC'], 'modifier': ['DMS'],
                               'SNR': [2.], 'reactivity': ['[1]']})
        out = attach_benchmark_ids(raw, frozen, 3)
        self.assertEqual(len(out), 3)
        self.assertEqual(list(out.benchmark_profile_id), ['', '3:0', ''])
        frozen.loc[0, 'reactivity'] = '[2]'
        with self.assertRaises(ValueError):
            attach_benchmark_ids(raw, frozen, 3)

    def test_bzcn_uses_raw_all_base_shape_scores(self):
        expected = probe_scores('ACGU', '1M7', [-1., 0., 2., 3.], [0, 0, 1, 1])
        actual = probe_scores('ACGU', 'BzCN', [-1., 0., 2., 3.], [0, 0, 1, 1])
        self.assertEqual(actual, expected)
        self.assertEqual(actual[0]['n_observed'], 4)
        self.assertEqual(actual[0]['auroc'], 1.)

    def test_merge_weights_defined_rnas_and_keeps_cohorts_separate(self):
        def row(dataset, unit, auc, rho):
            return {'dataset': dataset, 'modifier': '2A3', 'profile_id': unit, 'unit_id': unit,
                    'cluster': unit, 'stratum': 'all', 'auroc': auc, 'spearman': rho,
                    'auroc_mw_neutral': .5 if auc is None else auc,
                    'n_observed': 4, 'prediction_constant': auc is None}
        a = pd.DataFrame([row('raw', 'a', .9, .5), row('benchmark', 'a', .9, .5)])
        b = pd.DataFrame([row('raw', 'b', .6, .2), row('raw', 'c', None, None)])
        merged = merge_summaries([summarize(a), summarize(b)])
        combined = summarize(pd.concat([a, b], ignore_index=True))
        for left, right in zip(merged, combined):
            self.assertEqual(left['dataset'], right['dataset'])
            self.assertEqual(left['units'], right['units'])
            self.assertAlmostEqual(left['auroc_sequence_macro'], right['auroc_sequence_macro'])
            self.assertAlmostEqual(left['spearman_sequence_macro'], right['spearman_sequence_macro'])

    def test_matched_comparison_excludes_constant_reactivity_from_both_metrics(self):
        a = pd.DataFrame({'profile_id': ['1:0', '1:0', '1:1'], 'stratum': ['A', 'C', 'all'],
                          'auroc': [.7, .8, .5], 'spearman': [.3, .4, np.nan]})
        b = a.copy()
        b.loc[0, ['auroc', 'spearman']] = np.nan
        left, right = common_frames([a, b])
        self.assertEqual(list(zip(left.profile_id, left.stratum)), [('1:0', 'C')])
        self.assertEqual(list(zip(right.profile_id, right.stratum)), [('1:0', 'C')])


if __name__ == '__main__':
    unittest.main()
