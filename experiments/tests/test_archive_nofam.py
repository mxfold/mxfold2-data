import unittest

from scripts.efold_hf.archive_nofam import ct_record
from scripts.efold_hf.archive_nofam_report import aggregate, checked_pairs


CT = '4 example\n1 G 0 2 4 1\n2 A 1 3 0 2\n3 A 2 4 0 3\n4 C 3 0 1 4\n'


class ArchiveNoFamTest(unittest.TestCase):
    def test_ct_reference_strict_and_first_predicted_structure_explicit(self):
        result = ct_record(CT, 'example')
        self.assertEqual(result['sequence'], 'GAAC')
        self.assertEqual(result['pairs'], [[0, 3]])
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            ct_record(CT + CT, 'example')
        self.assertEqual(ct_record(CT + CT, 'example', first_structure=True), result)
        with self.assertRaisesRegex(ValueError, 'Inconsistent CT pair'):
            ct_record(CT.replace('4 C 3 0 1 4', '4 C 3 0 0 4'), 'example')


    def test_group_macro_averages_per_rna_f1_before_groups(self):
        rows = [dict(id='a1', tp=2, fp=0, fn=0, precision=1., recall=1., f1=1.),
                dict(id='a2', tp=0, fp=0, fn=6, precision=None, recall=0., f1=0.),
                dict(id='b1', tp=1, fp=3, fn=3, precision=.25, recall=.25, f1=.25)]
        value = aggregate(rows, {'a1': 'A', 'a2': 'A', 'b1': 'B'})
        self.assertEqual(value['group_macro']['f1'], .375)
        self.assertAlmostEqual(value['rna_macro']['f1'], 1.25/3)
        self.assertEqual(value['by_group']['A']['f1'], .5)
        self.assertEqual(value['by_group']['A']['precision'], .5)
        self.assertAlmostEqual(value['micro_f1'], 6/18)
        self.assertEqual(value['undefined_precision_rnas'], 1)


    def test_discrete_contacts_preserve_multiplets_and_reject_invalid_edges(self):
        self.assertEqual(checked_pairs([[0, 3], [0, 4]], 5), [(0, 3), (0, 4)])
        for pairs in [[[0, 0]], [[0, 5]], [[3, 0]], [[0, 3], [0, 3]]]:
            with self.assertRaises(ValueError):
                checked_pairs(pairs, 5)


if __name__ == '__main__':
    unittest.main()
