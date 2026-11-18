import sys
from pathlib import Path
import unittest
from collections import Counter

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/efold_hf'))
from paper_full_evaluation import build_panel,COUNTS


class FullPaperPanelTest(unittest.TestCase):
    def test_all_published_records_and_previously_missing_lengths_are_retained(self):
        panel=build_panel()
        self.assertEqual(dict(Counter(r['source'] for r in panel)),COUNTS)
        long=Counter(r['source'] for r in panel if len(r['sequence'])>1000)
        self.assertEqual(dict(long),{'archiveII':11,'lncRNA':27})
        self.assertEqual(max(len(r['sequence']) for r in panel),1995)

    def test_same_norad_sequence_at_different_temperatures_remains_separate(self):
        panel=build_panel()
        samples=[r for r in panel if r['source']=='lncRNA' and r['reference'].startswith('NORAD3_')]
        self.assertEqual(len(samples),3)
        self.assertEqual(len({r['id'] for r in samples}),3)
        self.assertEqual(len({r['sequence_sha256'] for r in samples}),1)
        self.assertGreater(len({tuple(r['pairs']) for r in samples}),1)


if __name__=='__main__':unittest.main()
