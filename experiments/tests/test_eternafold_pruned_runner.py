import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts.eternafold import pruned_traina_workflow as runner
from scripts.eternafold import fulltest_new_metrics as fulltest


class PrunedModelEvaluationTest(unittest.TestCase):
    def test_shared_full_test_uses_the_new_model_and_rebinds_input_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            template, experiment = tmp / 'template', tmp / 'experiment'
            for root in (template, experiment):
                (root / 'bin').mkdir(parents=True)
                (root / 'data').mkdir()
                for name in ('contrafold', 'contrafold-mpi'):
                    (root / 'bin' / name).write_text('same validated executable')
            script = template / 'source/scripts/eternafold/fulltest_new_metrics.py'
            script.parent.mkdir(parents=True)
            script.write_text('# Native execution is mocked in this model/data identity test.\n')
            (template / 'model.params').write_text('old model')
            shared = template / 'data/shared_input'
            shared.write_text('identical original test measurements')
            base = {'model_sha256': fulltest.sha(template / 'model.params'),
                    'source_sha256': {'scripts/eternafold/fulltest_new_metrics.py': fulltest.sha(script)},
                    'binary_sha256': {name: fulltest.sha(template / 'bin' / name) for name in ('contrafold', 'contrafold-mpi')}}
            fulltest.write_json(template / 'plan.json', base)
            fulltest.write_json(template / 'data/complete.json', {
                'plan_sha256': fulltest.sha(template / 'plan.json'),
                'shards': {'1': {'files_sha256': {'data/shared_input': fulltest.sha(shared)}}}})
            model = experiment / 'runs/pretrain/model.params'
            model.parent.mkdir(parents=True)
            model.write_text('new 3166-record model')
            fulltest.write_json(model.with_name('complete.json'), {'complete': True, 'model_sha256': fulltest.sha(model)})
            sequence = 'GCAAAAGC'
            partners = [8, 7, 0, 0, 0, 0, 2, 1]
            fulltest.write_json(experiment / 'data/evaluation_entries.json', [
                {'kind': 'chemical', 'name': 'known', 'sequence': sequence}])
            bpseq = experiment / 'evaluation/pretrain/bpseq/known'
            bpseq.parent.mkdir(parents=True)
            bpseq.write_text(''.join(f'{i} {b} {j}\n' for i, (b, j) in enumerate(zip(sequence, partners), 1)))
            plan = {'source_and_input_sha256': {}, 'full_test_template': str(template),
                    'template_plan_sha256': fulltest.sha(template / 'plan.json')}
            fulltest.write_json(experiment / 'plan.json', plan)
            fulltest.write_json(experiment / 'prepared.json', {'plan_sha256': fulltest.sha(experiment / 'plan.json')})
            with patch.object(runner.subprocess, 'run') as native:
                runner.full_setup(experiment, 'pretrain')
                native.assert_called_once()
            destination = experiment / 'full_test/pretrain'
            copied = fulltest.verify(destination)
            meta = fulltest.prepared(destination, copied)
            self.assertEqual(copied['model_sha256'], fulltest.sha(model))
            self.assertNotEqual(copied['model_sha256'], base['model_sha256'])
            self.assertEqual((destination / 'model.params').read_text(), 'new 3166-record model')
            self.assertEqual(meta['plan_sha256'], fulltest.sha(destination / 'plan.json'))
            self.assertNotEqual(meta['plan_sha256'], fulltest.sha(template / 'plan.json'))
            self.assertTrue((destination / 'data/shared_input').is_symlink())
            self.assertEqual(fulltest.sha(destination / 'data/shared_input'), fulltest.sha(shared))
            with np.load(copied['saved_panel']) as predicted:
                np.testing.assert_array_equal(predicted['values'], [0, 0, 1, 1, 1, 1, 0, 0])


if __name__ == '__main__':
    unittest.main()
