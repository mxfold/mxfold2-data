import copy
import json
from pathlib import Path
import shlex
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from pytorch_optimizer import AdaBelief

from scripts.efold_hf import condition3_pri_mrna_ft as module
from scripts.efold_hf import workflow


class Condition3Stage2Tests(unittest.TestCase):
    def test_schedule_covers_primary_once_and_continues_replay(self):
        from scripts.efold_hf.matched_losses import make_replay_stream
        for epoch in (1, 2, 3):
            order = module.stage_schedule(2471, 14831, epoch)
            self.assertEqual(len(order), 3089)
            self.assertEqual(sorted(i for k, i in order if k == 'primary'), list(range(2471)))
            replay = [i for k, i in order if k == 'replay']
            expected = make_replay_stream(14831, 34296+(epoch-1)*618, 618, 0)
            self.assertEqual(sorted(replay), sorted(expected))
            self.assertEqual(len(set(replay)), 618)
            self.assertEqual(order, module.stage_schedule(2471, 14831, epoch))

    def test_initial_payload_changes_only_stage_counters(self):
        payload = {'model_state_dict': {'w': torch.tensor([1.])},
                   'optimizer_state_dict': {'param_groups': [{'step': 339493}]},
                   'shape_model_state_dict': [{'w': torch.tensor([.25])}],
                   'rng': {'torch': torch.get_rng_state()},
                   'state': {'epoch': 3, 'cursor': 57161, 'updates': 42873, 'loss_sum': 4.}}
        snapshot = copy.deepcopy(payload)
        initial = module.initial_payload(payload)
        module.assert_same(payload, snapshot)
        for key in payload.keys()-{'state'}:
            module.assert_same(initial[key], payload[key])
        self.assertEqual(initial['state'], {'epoch': 1, 'cursor': 0, 'updates': 0, 'loss_sum': 0.})

    def test_cpu_handoff_restores_real_optimizer_and_Wu_without_updates(self):
        with tempfile.TemporaryDirectory() as temp:
            parent, target = Path(temp)/'stage1', Path(temp)/'stage2'
            arm = 'nll_restored'
            run = parent/'runs'/arm
            model, shape = torch.nn.Linear(2, 1), torch.nn.Linear(1, 1)
            optimizer = AdaBelief([{'params': model.parameters()}, {'params': shape.parameters()}], lr=1e-4)
            for _ in range(3):
                workflow.backward_step(model(torch.ones(2)).square().mean()+
                                       shape(torch.ones(1)).square().mean(), model, optimizer)
            optimizer.param_groups[0]['step'] = module.PARENT_NETWORK_STEP
            state = {'epoch': 3, 'cursor': 57161, 'updates': 42873, 'loss_sum': 3.}
            path = run/'epoch-003.pt'
            workflow.save_checkpoint(path, model, optimizer, [shape], state)
            module.write_json(run/'epoch-003-complete.json', {'success': True, 'checkpoint_sha256': module.sha(path)})
            module.write_json(run/'complete.json', {'success': True, 'epochs': 3, 'updates': 42873})
            restored_model, restored_shape = torch.nn.Linear(2, 1), torch.nn.Linear(1, 1)
            restored_optimizer = AdaBelief([{'params': restored_model.parameters()},
                                           {'params': restored_shape.parameters()}], lr=1e-4)
            fake = SimpleNamespace(seed_all=workflow.seed_all, restore=workflow.restore,
                save_checkpoint=workflow.save_checkpoint,
                build=lambda *_: (restored_model, restored_optimizer, None, None, [restored_shape], 'cpu'))
            with patch.object(module, 'STAGE1', parent), patch.object(module, 'ROOT', target), \
                    patch.object(module, 'shared', return_value=SimpleNamespace(workflow=lambda: fake)):
                audit = module.handoff(arm)
                initial = torch.load(target/'runs'/arm/'initial.pt', weights_only=False)
                source = torch.load(path, weights_only=False)
                for key in source.keys()-{'state'}:
                    module.assert_same(initial[key], source[key])
                self.assertTrue(audit['model_optimizer_Wu_exact'])
                self.assertEqual(audit['network_optimizer_step'], 339493)
                self.assertEqual(audit['state']['updates'], 0)
                # Repeating the bridge must never overwrite later learning.
                progressed = {**initial, 'state': {**initial['state'], 'updates': 1}}
                module.save_payload(target/'runs'/arm/'latest.pt', progressed)
                module.handoff(arm)
                self.assertEqual(torch.load(target/'runs'/arm/'latest.pt', weights_only=False)['state']['updates'], 1)

    def test_handoff_rejects_unfinished_parent(self):
        with patch.object(module, 'parent_completed', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'has not completed'):
                module.handoff('mw_rank_restored')

    def test_bridge_follows_registered_upstream_continuation(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(module, 'ROOT', Path(temp)), \
                patch.object(module, 'verify'), patch.object(module, 'parent_completed', return_value=False), \
                patch.object(module, 'latest_parent_work', return_value='200'), \
                patch.object(module, 'submit_once', return_value='300') as submit, \
                patch.object(module, 'handoff') as handoff:
            module.bridge('mw_rank_restored', '100')
            submit.assert_called_once_with('bridge-mw_rank_restored-after-200', 'bridge',
                                          'mw_rank_restored', hold='200', waited_job='200')
            handoff.assert_not_called()

    def test_bridge_rejects_failed_parent_instead_of_spinning(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(module, 'ROOT', Path(temp)), \
                patch.object(module, 'verify'), patch.object(module, 'parent_completed', return_value=False), \
                patch.object(module, 'latest_parent_work', return_value='100'), \
                patch.object(module, 'submit_once') as submit:
            with self.assertRaisesRegex(RuntimeError, 'Upstream incomplete'):
                module.bridge('nll_restored', '100')
            submit.assert_not_called()

    def test_completed_bridge_submits_GPU_preflight_then_training(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(module, 'ROOT', Path(temp)), \
                patch.object(module, 'verify'), patch.object(module, 'parent_completed', return_value=True), \
                patch.object(module, 'handoff', return_value={'source_sha256': 'abc'}), \
                patch.object(module, 'submit_once', side_effect=['200', '300']) as submit, \
                patch.dict(module.os.environ, {'JOB_ID': '100'}):
            module.bridge('shape_fy_restored', '99')
            self.assertEqual(submit.call_args_list[0].kwargs['hold'], '100')
            self.assertEqual(submit.call_args_list[1].kwargs['hold'], '200')

    def test_qsub_freezes_script_preserves_dependencies_and_24h_limit(self):
        captured = []
        def fake_run(command, **kwargs):
            args = shlex.split(command[3])
            captured.append(args)
            return subprocess.CompletedProcess(command, 0, stdout=str(100+len(captured))+'\n', stderr='')
        with tempfile.TemporaryDirectory() as temp, patch.object(module, 'ROOT', Path(temp)), \
                patch.object(module.subprocess, 'run', side_effect=fake_run):
            for action in ('prepare', 'bridge', 'preflight', 'work', 'evaluate'):
                job = module.submit_once(action, action, 'nll_restored', hold='55,56', waited_job='56')
                self.assertEqual(job, module.submit_once(action, action, 'nll_restored', hold='55,56'))
            self.assertEqual(len(captured), 5)
            for args in captured:
                self.assertEqual(args[0], module.QSUB)
                self.assertEqual(args[args.index('-hold_jid')+1], '55,56')
                self.assertEqual(args[-1], str(Path(temp)/'job.sh'))
                self.assertIn('WAITED_STAGE1_JOB=56', args[args.index('-v')+1])
                duration = next(a.split('=')[1] for a in args if a.startswith('h_rt='))
                h, m, s = map(int, duration.split(':'))
                self.assertLess(3600*h+60*m+s, 24*3600)
            self.assertLess(module.SOFT_HOURS*3600, module.HARD_SECONDS)


if __name__ == '__main__':
    unittest.main()
