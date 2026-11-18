from datetime import datetime,timedelta
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from scripts.efold_hf import finetune_milestones as gate


def load_template(filename,condition,epoch):
    with patch.dict(os.environ,{'FT_CONDITION':str(condition),'FT_PRETRAIN_EPOCH':str(epoch)}):
        with patch.dict(os.environ):
            os.environ.pop('CURRICULUM_ROOT',None)
            os.environ.pop('FT_STAGE1_ROOT',None)
            path=gate.PROJECT/'scripts/efold_hf'/filename
            spec=importlib.util.spec_from_file_location(f'test_c{condition}p{epoch}_{path.stem}',path)
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            return module


class MilestoneFTTests(unittest.TestCase):
    def fixture(self,root):
        parent=root/'pretrain';parent.mkdir()
        gate.write(parent/'plan.json',{'pretrain_epochs':10})
        return {'id':'c2p5','condition':2,'pretrain_epoch':5,'pretrain_root':str(parent),
            'pretrain_plan_sha256':gate.sha(parent/'plan.json'),'expected_optimizer_step':1278835,
            'stage1':str(root/'first'),'stage2':str(root/'second')}

    def checkpoint(self,case,epoch=5,updates=1278835):
        path=Path(case['pretrain_root'])/f'runs/pretrain/epoch-{epoch:03d}.pt'
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'committed model state')
        gate.write(path.with_name(f'epoch-{epoch:03d}-complete.json'),{'success':True,'epoch':epoch,
            'updates':updates,'checkpoint_sha256':gate.sha(path)})
        return path

    def test_gate_requires_requested_epoch_and_does_not_use_latest(self):
        with tempfile.TemporaryDirectory() as temp:
            case=self.fixture(Path(temp))
            self.checkpoint(case,epoch=6,updates=1534602)
            (Path(case['pretrain_root'])/'runs/pretrain/latest.pt').write_bytes(b'later model')
            self.assertIsNone(gate.checkpoint_binding(case))
            expected=self.checkpoint(case)
            value=gate.checkpoint_binding(case)
            self.assertEqual(value['checkpoint'],str(expected))
            self.assertEqual(value['expected_optimizer_step'],1278835)
            gate.bind(case,value)
            self.assertEqual(gate.bind(case,value)['checkpoint_sha256'],gate.sha(expected))
            expected.write_bytes(b'changed')
            with self.assertRaisesRegex(RuntimeError,'mismatch'):gate.checkpoint_binding(case)

    def test_checkpoint_marker_alone_is_not_readiness(self):
        with tempfile.TemporaryDirectory() as temp:
            case=self.fixture(Path(temp));path=self.checkpoint(case);path.unlink()
            self.assertIsNone(gate.checkpoint_binding(case))
            gate.write(Path(case['pretrain_root'])/'runs/pretrain/complete.json',{'success':True})
            with self.assertRaisesRegex(RuntimeError,'missing'):gate.checkpoint_binding(case)

    def test_checkpoint_binding_cannot_switch_to_another_model(self):
        with tempfile.TemporaryDirectory() as temp:
            case=self.fixture(Path(temp));self.checkpoint(case)
            binding=gate.checkpoint_binding(case);gate.bind(case,binding)
            with self.assertRaisesRegex(RuntimeError,'binding changed'):
                gate.bind(case,{**binding,'checkpoint_sha256':'other model'})

    def test_tick_launches_ready_case_and_schedules_one_future_check(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);ready=self.fixture(root);self.checkpoint(ready)
            waiting={**ready,'id':'c2p10','pretrain_epoch':10,'expected_optimizer_step':2557670}
            now=datetime.now(gate.JST)
            with patch.object(gate,'ROOT',root),patch.object(gate,'verify'), \
                    patch.object(gate,'cases',return_value=[ready,waiting]), \
                    patch.object(gate,'submit_once',side_effect=['100','101']) as submit, \
                    patch.dict(os.environ,{'JOB_ID':'99'}):
                gate.tick()
            self.assertEqual(submit.call_count,2)
            self.assertEqual(submit.call_args_list[0].args,('launch-c2p5','launch','c2p5'))
            self.assertEqual(submit.call_args_list[0].kwargs['hold'],'99')
            start=datetime.strptime(submit.call_args_list[1].kwargs['start_at'],'%Y%m%d%H%M.%S').replace(tzinfo=gate.JST)
            self.assertGreater(start,now+timedelta(minutes=14))
            self.assertEqual(gate.read(root/'state/c2p5.json')['state'],'launch_queued')
            self.assertEqual(gate.read(root/'state/c2p10.json')['state'],'waiting_for_checkpoint')
            self.assertEqual(gate.read(root/'status.json')['next_gate_job'],'101')

    def test_finished_registry_does_not_relaunch_or_poll(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);case=self.fixture(root)
            gate.write(root/'state/c2p5.json',{'case':'c2p5','state':'launched'})
            with patch.object(gate,'ROOT',root),patch.object(gate,'verify'), \
                    patch.object(gate,'cases',return_value=[case]),patch.object(gate,'submit_once') as submit:
                gate.tick()
            submit.assert_not_called()

    def test_all_new_conditions_preserve_checkpoint_identity_and_epoch(self):
        for condition,epoch,expected in [(1,10,455070),(2,5,1278835),(2,10,2557670),(3,10,148310),(4,10,600650),(4,20,1201300)]:
            first=load_template('condition3_ribonanza_ft.py',condition,epoch)
            second=load_template('condition3_pri_mrna_ft.py',condition,epoch)
            self.assertEqual(first.SOURCE_STEP,expected)
            self.assertEqual(second.PARENT_NETWORK_STEP,expected+42873)
            self.assertEqual(second.STAGE1,first.ROOT)
            self.assertIn(f'_pt{epoch:03d}_',str(first.ROOT))
            captured=[]
            def fake_run(command,**kwargs):
                captured.append(shlex.split(command[3]))
                return subprocess.CompletedProcess(command,0,stdout='123\n',stderr='')
            with tempfile.TemporaryDirectory() as temp,patch.object(first,'ROOT',Path(temp)), \
                    patch.object(first.subprocess,'run',side_effect=fake_run):
                first.submit_once('work','work','nll_restored',hold='99')
            environment=captured[0][captured[0].index('-v')+1]
            self.assertIn(f'FT_CONDITION={condition},',environment)
            self.assertIn(f'FT_PRETRAIN_EPOCH={epoch},',environment)

    def test_final_report_requires_all_three_completed_epoch6_evaluations(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);case=self.fixture(root)
            case['methods']=['mw_rank_restored','nll_restored','shape_fy_restored']
            first,second=Path(case['stage1']),Path(case['stage2'])
            gate.write(first/'evaluation/pretrain/epoch-005/summary.json',{'success':True})
            for arm in case['methods']:
                gate.write(second/'runs'/arm/'complete.json',{'success':True})
            for arm in case['methods'][:-1]:
                gate.write(second/'evaluation'/arm/'epoch-003/summary.json',{'success':True,'global_ft_epoch':6})
            self.assertFalse(gate.report_ready(case))
            final=second/'evaluation/shape_fy_restored/epoch-003/summary.json'
            gate.write(final,{'success':True,'global_ft_epoch':3})
            self.assertFalse(gate.report_ready(case))
            gate.write(final,{'success':True,'global_ft_epoch':6})
            self.assertTrue(gate.report_ready(case))

    def test_report_wait_does_not_stop_when_training_jobs_are_launched(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);case=self.fixture(root)
            gate.write(root/'cases.json',{'cases':[case],'final_reports':True})
            gate.write(root/'state/c2p5.json',{'case':'c2p5','state':'launched'})
            with patch.object(gate,'ROOT',root),patch.object(gate,'verify'), \
                    patch.object(gate,'monitor_report',return_value=False), \
                    patch.object(gate,'submit_once',return_value='123') as submit, \
                    patch.dict(os.environ,{'JOB_ID':'99'}):
                gate.tick()
            self.assertEqual(gate.read(root/'status.json')['pending_reports'],['c2p5'])
            submit.assert_called_once()
            self.assertEqual(submit.call_args.args,('tick-after-99','tick'))
            self.assertIn('start_at',submit.call_args.kwargs)

    def test_ready_report_is_submitted_once_and_completion_stops_polling(self):
        calls=[]
        def fake_run(command,**kwargs):
            calls.append(command)
            return subprocess.CompletedProcess(command,0,stdout='123\n',stderr='')
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);case=self.fixture(root)
            with patch.object(gate,'ROOT',root),patch.object(gate,'report_ready',return_value=True), \
                    patch.object(gate.subprocess,'run',side_effect=fake_run):
                self.assertFalse(gate.monitor_report(case))
                self.assertFalse(gate.monitor_report(case))
                self.assertEqual(len(calls),1)
                gate.write(root/'reports/c2p5/status.json',{'state':'complete'})
                self.assertTrue(gate.monitor_report(case))
                self.assertEqual(len(calls),1)

    def test_gate_jobs_are_delayed_by_scheduler_and_below24h(self):
        captured=[]
        def fake_run(command,**kwargs):
            captured.append(shlex.split(command[3]))
            self.assertEqual(kwargs['env']['TZ'],'Asia/Tokyo')
            return subprocess.CompletedProcess(command,0,stdout=f'{100+len(captured)}\n',stderr='')
        with tempfile.TemporaryDirectory() as temp,patch.object(gate,'ROOT',Path(temp)), \
                patch.object(gate.subprocess,'run',side_effect=fake_run):
            gate.submit_once('future','tick',hold='99',start_at='202611021200.00')
            gate.submit_once('start','launch','c1p10')
            gate.submit_once('audit','audit-data')
            gate.submit_once('future','tick')
        self.assertEqual(len(captured),3)
        self.assertEqual(captured[0][captured[0].index('-a')+1],'202611021200.00')
        self.assertEqual(captured[0][captured[0].index('-hold_jid')+1],'99')
        for args in captured:
            duration=next(x.split('=')[1] for x in args if x.startswith('h_rt='))
            h,m,s=map(int,duration.split(':'));self.assertLess(h*3600+m*60+s,86400)

    def test_baseline_monitor_cache_falls_back_to_full_inference(self):
        module=load_template('condition3_ribonanza_ft.py',2,5)
        with tempfile.TemporaryDirectory() as temp:
            root,parent=Path(temp)/'FT',Path(temp)/'pretrain'
            cache=parent/'evaluation/epoch-005';cache.mkdir(parents=True)
            (root/'data').mkdir(parents=True)
            (root/'metrics.py').write_text('def from_pairs(seq,pairs): return [1]*len(seq)\n')
            tests=[{'id':x,'source':'PDB','sequence':'ACGU','pairs':[]} for x in ('a','b')]
            module.write_rows(root/'data/test.jsonl',tests)
            module.write_json(root/'test_exact_overlap.json',{'ids':[]})
            predictions=cache/'per_sequence.jsonl'
            module.write_rows(predictions,[{'id':'a','predicted_pairs':[]}])
            module.write_json(cache/'summary.json',{'checkpoint_sha256':'expected','predictions_sha256':module.sha(predictions)})
            with patch.object(module,'ROOT',root),patch.object(module,'PRETRAIN',parent),patch.object(module,'CHECKPOINT_SHA','expected'):
                self.assertFalse(module.score_saved_baseline())
                module.write_rows(predictions,[{'id':x,'predicted_pairs':[]} for x in ('a','b')])
                module.write_json(cache/'summary.json',{'checkpoint_sha256':'expected','predictions_sha256':module.sha(predictions)})
                self.assertTrue(module.score_saved_baseline())
            self.assertEqual(gate.read(root/'evaluation/pretrain/epoch-005/summary.json')['epoch'],5)


if __name__=='__main__':unittest.main()
