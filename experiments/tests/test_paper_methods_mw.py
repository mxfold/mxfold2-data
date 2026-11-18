from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/efold_hf'))
from paper_methods_mw import paired_union


class DiscreteContactStateTest(unittest.TestCase):
    def test_multiplet_orientation_and_duplicate_contacts_do_not_change_pairing_state(self):
        states,audit=paired_union('ACGUA',[[0,4],[4,0],[0,3],[0,4]])
        self.assertEqual(states.tolist(),[0,1,1,0,0])
        self.assertEqual(audit['unique_nonself_pairs'],2)
        self.assertEqual(audit['multiplet_bases'],1)

    def test_self_contacts_and_invalid_positions_are_distinguished(self):
        states,audit=paired_union('ACGU',[[1,1]])
        self.assertEqual(states.tolist(),[1,1,1,1])
        self.assertEqual(audit['self_contacts_ignored'],1)
        with self.assertRaises(ValueError):paired_union('ACGU',[[0,4]])


if __name__=='__main__':unittest.main()
