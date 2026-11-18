import unittest
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/efold_hf'))
from rnacentral_overlap import source_targets,nearest_in_cohort,local_rows


class RNAcentralOverlapTest(unittest.TestCase):
    def test_final_membership_requires_same_sequence_and_records_renamed_ids(self):
        source={'same':{'sequence':'atgc'},'renamed':{'sequence':'AAAA'},
                'absent':{'sequence':'GGGG'},'boundary':{'sequence':'A'*1000},
                'too_long':{'sequence':'A'*1001}}
        targets=source_targets(source,{'same':'AUGC','different_id':'AAAA'})
        by_id={r['id']:r for r in targets}
        self.assertEqual(set(by_id),{'same','renamed','absent','boundary'})
        self.assertTrue(by_id['same']['same_id_and_sequence_in_final'])
        self.assertTrue(by_id['renamed']['in_final_training'])
        self.assertFalse(by_id['renamed']['same_id_and_sequence_in_final'])
        self.assertEqual(by_id['renamed']['final_training_references'],['different_id'])
        self.assertFalse(by_id['absent']['in_final_training'])

    def test_excluded_source_hit_cannot_be_nearest_final_training_hit(self):
        import numpy as np
        from rapidfuzz import process
        from rapidfuzz.distance import Levenshtein
        targets=[{'id':'excluded','sequence':'ACGU'}, {'id':'retained','sequence':'ACGA'}]
        query={'sequence':'ACGU'}
        scores=process.cdist([query['sequence']],[r['sequence'] for r in targets],
                             scorer=Levenshtein.normalized_similarity,dtype=np.float64)[0]
        raw=nearest_in_cohort(query,targets,np.array([0,1]),scores)
        final=nearest_in_cohort(query,targets,np.array([1]),scores)
        self.assertEqual(raw['best_target_id'],'excluded')
        self.assertEqual(raw['global_edit_similarity'],1.)
        self.assertEqual(final['best_target_id'],'retained')
        self.assertEqual(final['global_edit_similarity'],.75)

    def test_local_coverage_source_membership_and_missing_hits_remain_distinct(self):
        targets=[{'id':'excluded','sequence':'ACGU','in_final_training':False},
                 {'id':'retained','sequence':'ACGUAA','in_final_training':True}]
        queries=[{'id':'hit','source':'PDB','sequence':'ACGU','in_original_conservative_panel':False},
                 {'id':'missing','source':'PDB','sequence':'CCCC','in_original_conservative_panel':False}]
        hits=['Q000000 T000000 100 4 0 0 1 4 1 4 1e-5 20 1 1',
              'Q000000 T000001 100 4 0 0 1 4 1 4 1e-4 18 1 .667']
        result={(r['id'],r['cohort']):r for r in local_rows(targets,queries,hits)}
        self.assertEqual(result['hit','source_all']['both_cov80_best_target_id'],'excluded')
        final=result['hit','efold_train_retained']
        self.assertEqual(final['pident'],100)
        self.assertIsNone(final['both_cov80_pident'])
        self.assertTrue(final['full_query_identical_substring'])
        self.assertIsNone(result['missing','source_all']['pident'])


if __name__=='__main__':unittest.main()
