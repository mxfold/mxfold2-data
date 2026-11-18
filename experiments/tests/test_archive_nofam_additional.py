import tempfile
from pathlib import Path
import unittest

from scripts.efold_hf import archive_nofam_additional as experiment


class AdditionalNoFamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scoring=experiment.load('additional_nofam_test_scoring',experiment.BASE/'all_conditions_external_eval_20261104/score_adapter.py')

    def test_equal_group_weight_is_distinct_from_RNA_weight(self):
        items=[{'id':name,**self.scoring.pair_metrics([[0,3]],pairs)}
               for name,pairs in [('a',[[0,3]]),('b',[[0,3]]),('c',[])]]
        scores=self.scoring.aggregate(items,{'a':'large','b':'large','c':'small'})
        self.assertEqual(scores['group_macro']['f1'],.5)
        self.assertAlmostEqual(scores['rna_macro']['f1'],2/3)

    def test_prediction_rejects_duplicate_or_out_of_bounds_contact(self):
        item={'id':'a','sequence':'ACGU','sequence_sha256':'digest','pairs':[[0,3]]}
        for pairs in ([[0,3],[0,3]], [[0,4]], [[2,2]], [[3,0]]):
            with self.assertRaises(ValueError):
                experiment.checked_prediction(item,pairs,self.scoring)
        value=experiment.checked_prediction(item,[[0,3]],self.scoring)
        self.assertEqual(value['f1'],1.)
        self.assertEqual(value['sequence_sha256'],'digest')

    def test_native_control_mismatch_blocks_evaluation(self):
        controls=[{'id':'a','expected_pairs':[[0,3]]}]
        experiment.check_controls(controls,lambda _: [[0,3]])
        with self.assertRaisesRegex(RuntimeError,'differs from saved'):
            experiment.check_controls(controls,lambda _: [])

    def test_spot_tfrecord_names_are_unique_between_experiments(self):
        first=experiment.spot_basename('/tmp/archive_nofam_one')
        second=experiment.spot_basename('/tmp/archive_nofam_two')
        self.assertNotEqual(first,second)
        self.assertTrue(first.startswith('archive_nofam_extra_'))
        self.assertNotIn('bprnanew_full',first)
        self.assertEqual(first,experiment.spot_basename('/tmp/archive_nofam_one'))

    def test_actual_cohorts_keep_SPOT_paper_exclusion_separate(self):
        cohorts=experiment.read(experiment.PRIOR/'cohorts.json')['cohorts']
        sizes={key:(len(value['ids']),len(value['groups'])) for key,value in cohorts.items()}
        self.assertEqual(sizes['paper_bprna_filtered'],(79,40))
        self.assertEqual(sizes['literal_caption_union'],(44,22))
        self.assertEqual(sizes['full89'],(89,47))
        self.assertTrue(set(cohorts['literal_caption_union']['ids']) < set(cohorts['paper_bprna_filtered']['ids']))


if __name__=='__main__':
    unittest.main()
