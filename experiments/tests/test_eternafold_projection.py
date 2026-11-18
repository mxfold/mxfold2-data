import itertools
import random
import unittest

from scripts.eternafold.project_structure import maximum_subset, prune_record, supported


class StructuralProjectionTest(unittest.TestCase):
    def test_large_single_loop_requires_removal_but_long_hairpin_is_valid(self):
        self.assertTrue(supported({(1, 100)}))
        self.assertFalse(supported({(1, 100), (40, 60)}))
        kept = maximum_subset({(1, 100), (5, 96), (40, 60)})
        self.assertEqual(len(kept), 2)
        self.assertTrue(supported(kept))

    def test_every_small_projection_has_bruteforce_optimal_pair_count(self):
        rng = random.Random(7)
        for _ in range(100):
            points = rng.sample(range(1, 85), 14)
            pairs = {tuple(sorted(points[i:i+2])) for i in range(0, 14, 2)}
            pairs = sorted(p for p in pairs if p[1] - p[0] >= 4)
            possible = [set(p for p, bit in zip(pairs, bits) if bit)
                        for bits in itertools.product((0, 1), repeat=len(pairs))]
            optimum = max(map(len, (p for p in possible if supported(p))))
            kept = maximum_subset(pairs)
            self.assertEqual(len(kept), optimum)
            self.assertTrue(kept.issubset(pairs))
            self.assertTrue(supported(kept))

    def test_short_noncanonical_and_ambiguous_pairs_removed_without_sequence_change(self):
        sequence = 'GANAAGAC'
        # 1-8 is GC; 2-7 is AA; 3-6 has N; 4-5 is too close.
        partners = [8, 7, 6, 5, 4, 3, 2, 1]
        result, reasons = prune_record(sequence, partners)
        self.assertEqual(result, [8, 0, 0, 0, 0, 0, 0, 1])
        self.assertEqual(set(reasons), {(2, 7), (3, 6), (4, 5)})
        self.assertEqual(sequence, 'GANAAGAC')

    def test_original_representable_annotation_remains_identical(self):
        original = [8, 7, 0, 0, 0, 0, 2, 1]
        result, reasons = prune_record('GCAAAAGC', original)
        self.assertEqual(result, original)
        self.assertEqual(reasons, {})


if __name__ == '__main__':
    unittest.main()
