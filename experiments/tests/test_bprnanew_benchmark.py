import unittest
import tempfile
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts/efold_hf'),str(ROOT)]
sys.path.append(str(ROOT/'.cache/bprna-benchmark-deps'))
from bprnanew_benchmark import bpseq,pair_metrics
from bprnanew_external import parse_ct
from bprnanew_spot import input_basename
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein


def independent_edit_distance(a,b):
    row=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        out=[i]
        for j,y in enumerate(b,1):out.append(min(out[-1]+1,row[j]+1,row[j-1]+(x!=y)))
        row=out
    return row[-1]


class BpRNAnewTest(unittest.TestCase):
    def test_bpseq_preserves_unknown_bases_and_crossing_reference(self):
        result=bpseq('1 A 4\n2 G 5\n3 N 0\n4 U 1\n5 C 2\n','test')
        self.assertEqual(result['sequence'],'AGNUC')
        self.assertEqual(result['pairs'],[[0,3],[1,4]])
        with self.assertRaises(ValueError):bpseq('1 A 2\n2 U 0\n','bad')

    def test_exact_pair_metrics_and_empty_predictions(self):
        result=pair_metrics([[0,3],[1,4]],[[0,3],[1,5]])
        self.assertEqual((result['precision'],result['recall'],result['f1']),(0.5,0.5,0.5))
        self.assertIsNone(pair_metrics([[0,3]],[])['precision'])
        self.assertEqual(pair_metrics([[0,3]],[])['f1'],0)
        self.assertEqual(pair_metrics([],[])['f1'],1)

    def test_exhaustive_normalized_edit_similarity_matches_independent_dp(self):
        queries=['AUGC','AU','ACGUAC','']
        targets=['AUGC','AUAC','UUUU','AUGCA','']
        matrix=process.cdist(queries,targets,scorer=Levenshtein.normalized_similarity)
        for i,a in enumerate(queries):
            for j,b in enumerate(targets):
                expected=1-independent_edit_distance(a,b)/max(len(a),len(b)) if a or b else 1
                self.assertAlmostEqual(float(matrix[i,j]),expected,places=6)

    def test_ct_parser_uses_first_structure_and_one_based_coordinates(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'output.ct'
            p.write_text('4 ENERGY = -1\n1 A 0 2 4 1\n2 C 1 3 0 2\n3 G 2 4 0 3\n4 U 3 0 1 4\n4 second structure\n')
            self.assertEqual(parse_ct(p,4),[[0,3]])

    def test_spot_shared_tfrecord_names_are_unique_for_all_shards(self):
        names={input_basename(s,smoke) for s in range(4) for smoke in (True,False)}
        self.assertEqual(len(names),8)


if __name__=='__main__':unittest.main()
