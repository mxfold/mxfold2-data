import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest
from unittest.mock import patch


PROJECT = Path(__file__).resolve().parents[1]


def condition1_module(name, filename):
    with patch.dict(os.environ, {'FT_CONDITION': '1'}, clear=False):
        with patch.dict(os.environ):
            os.environ.pop('CURRICULUM_ROOT', None)
            spec = importlib.util.spec_from_file_location(name, PROJECT/'scripts/efold_hf'/filename)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
    return module


class Condition1FTTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ribo = condition1_module('condition1_ribo_test', 'condition3_ribonanza_ft.py')
        cls.post = condition1_module('condition1_post_test', 'condition3_pri_mrna_ft.py')

    def test_matching_condition_checkpoint_replay_and_optimizer_steps(self):
        first, second = self.ribo, self.post
        self.assertEqual(first.CONDITION, 1)
        self.assertEqual(first.PRETRAIN.name, 'bprna20')
        self.assertEqual(first.SOURCE_STEP, 910140)
        self.assertEqual(first.REPLAY_POOL_COUNT, 45507)
        self.assertEqual(first.sha(first.PRETRAIN/'runs/pretrain/epoch-020.pt'), first.CHECKPOINT_SHA)
        self.assertEqual(second.STAGE1, first.ROOT)
        self.assertEqual(second.PARENT_NETWORK_STEP, first.SOURCE_STEP+42873)
        self.assertEqual(second.PARENT_NETWORK_STEP, 953013)
        self.assertEqual(second.REPLAY_POOL_COUNT, first.REPLAY_POOL_COUNT)
        self.assertEqual(second.REPLAY_START, 3*11432)

    def test_replay_covers_bpRNA_pool_and_keeps_twenty_percent(self):
        from scripts.efold_hf.matched_losses import make_replay_stream
        for epoch in (1, 2, 3):
            order = self.post.stage_schedule(2471, 45507, epoch)
            replay = [i for kind, i in order if kind == 'replay']
            self.assertEqual(len(order), 3089)
            self.assertEqual(len(replay), 618)
            expected = make_replay_stream(45507, 34296+(epoch-1)*618, 618, 0)
            self.assertEqual(sorted(replay), sorted(expected))
        stream = make_replay_stream(45507, 0, 45507, 0)
        self.assertEqual(len(set(stream)), 45507)

    def test_new_condition_is_preserved_by_all_child_jobs(self):
        for module in (self.ribo, self.post):
            captured = []
            def fake_run(command, **kwargs):
                captured.append(shlex.split(command[3]))
                return subprocess.CompletedProcess(command, 0, stdout=f'{100+len(captured)}\n', stderr='')
            with tempfile.TemporaryDirectory() as temp, patch.object(module, 'ROOT', Path(temp)), \
                    patch.object(module.subprocess, 'run', side_effect=fake_run):
                for action in ('prepare', 'preflight', 'work', 'evaluate'):
                    module.submit_once(action, action, 'nll_restored', hold='99')
                self.assertEqual(len(captured), 4)
                for args in captured:
                    self.assertIn('FT_CONDITION=1', args[args.index('-v')+1])
                    self.assertTrue(args[args.index('-N')+1].startswith('c1'))
                    self.assertEqual(args[-1], str(Path(temp)/'job.sh'))
                    self.assertEqual(args[args.index('-hold_jid')+1], '99')
                    duration = next(s.split('=')[1] for s in args if s.startswith('h_rt='))
                    h, m, s = map(int, duration.split(':'))
                    self.assertLess(h*3600+m*60+s, 86400)

    def test_second_stage_preparation_waits_for_first_stage_preparation(self):
        module = self.post
        with tempfile.TemporaryDirectory() as temp:
            first, second = Path(temp)/'first', Path(temp)/'second'
            module.write_json(first/'dispatch/prepare.json', {'job_id': '77'})
            with patch.object(module, 'ROOT', second), patch.object(module, 'STAGE1', first), \
                    patch.object(module, 'submit_once', side_effect=['100', '101', '102', '103']) as submit, \
                    patch.object(module, 'latest_parent_work', side_effect=['201', '202', '203']), \
                    patch.object(module, 'parent_completed', return_value=False):
                module.submit()
            self.assertEqual(submit.call_args_list[0].kwargs['hold'], '77')
            for call, upstream in zip(submit.call_args_list[1:], ['201', '202', '203'], strict=True):
                self.assertEqual(call.kwargs['hold'], '100,'+upstream)
                self.assertEqual(call.kwargs['waited_job'], upstream)
            self.assertTrue((second/'controller.py').is_file())
            self.assertTrue((second/'job.sh').is_file())


if __name__ == '__main__':
    unittest.main()
