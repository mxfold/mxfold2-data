from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from scripts.efold_hf import pretrain_variants as module


class PretrainVariantsTest(unittest.TestCase):
    def test_curated_label_priority_preserves_duplicate_provenance(self):
        rows = [dict(id='bp:1', source='bpRNA-1m', sequence='ACGU', pairs=[]),
                dict(id='B:1', source='TrainSetB', sequence='ACGU', pairs=[[0, 3]]),
                dict(id='A:1', source='TrainSetA', sequence='ACGU', pairs=[[1, 2]]),
                dict(id='bp:2', source='bpRNA-1m', sequence='AAAA', pairs=[])]
        unique, mapping = module.exact_deduplicate(rows)
        self.assertEqual({r['id'] for r in unique}, {'A:1', 'bp:2'})
        self.assertEqual(next(r['pairs'] for r in unique if r['id'] == 'A:1'), [[1, 2]])
        self.assertEqual(mapping['bp:1'], 'A:1')
        self.assertEqual(mapping['B:1'], 'A:1')

    def test_cdhit_cluster_membership_and_singletons(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'clusters'
            path.write_text('>Cluster 0\n0 100nt, >S0000001... *\n1 90nt, >S0000002... at 1:90:1:90/+/90.00%\n>Cluster 1\n0 4nt, >S0000003... *\n')
            self.assertEqual(module.parse_clusters(path), {'S0000001': 'S0000001',
                'S0000002': 'S0000001', 'S0000003': 'S0000003'})
            path.write_text('>Cluster 0\n0 100nt, >S0000001... *\n1 90nt, >S0000002... *\n')
            with self.assertRaises(ValueError): module.parse_clusters(path)

    def test_all_job_walltimes_below24h_and_resume_dependency_retained(self):
        captured = []
        def fake_run(command, **kwargs):
            captured.append(shlex.split(command[3]))
            return subprocess.CompletedProcess(command, 0, stdout='12345\n', stderr='')
        with tempfile.TemporaryDirectory() as temporary, patch.object(module, 'ROOT', Path(temporary)), \
                patch.object(module.subprocess, 'run', side_effect=fake_run):
            for action in ['prepare', 'preflight', 'work', 'evaluate']:
                module.qsub_job('bprna20', action, epoch=10, hold='777')
        self.assertEqual(len(captured), 4)
        for args in captured:
            self.assertTrue(Path(args[0]).is_absolute())
            value = next(x for x in args if x.startswith('h_rt=')).split('=')[1]
            h, m, s = map(int, value.split(':'))
            self.assertLess(3600*h+60*m+s, 24*3600)
            self.assertEqual(args[args.index('-hold_jid')+1], '777')
        self.assertLess(module.SOFT_HOURS*3600, module.HARD_SECONDS)


if __name__ == '__main__': unittest.main()
