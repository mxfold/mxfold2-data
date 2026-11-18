import contextlib
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.efold_hf import all_conditions_external_eval as evaluation


class ExternalEvaluationTests(unittest.TestCase):
    def test_group_mean_preserves_equal_structural_group_weights(self):
        scoring = evaluation.load('test_external_score_adapter',evaluation.REFERENCE/'score_adapter.py')
        values = [{'id':name,**scoring.pair_metrics([[0,3]],pairs)} for name,pairs in
                  [('a',[[0,3]]),('b',[[0,3]]),('c',[])]]
        result = scoring.aggregate(values,{'a':'large','b':'large','c':'small'})
        self.assertEqual(result['group_macro']['f1'],.5)
        self.assertAlmostEqual(result['rna_macro']['f1'],2/3)

    def test_cache_rejects_duplicate_ID_changed_sequence_and_wrong_F1(self):
        scoring = evaluation.load('test_external_cache_adapter',evaluation.REFERENCE/'score_adapter.py')
        item = {'id':'a','sequence':'ACGU','sequence_sha256':'sequence_hash','pairs':[[0,3]]}
        row = evaluation.checked_prediction(item,[[0,3]],scoring)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'predictions.jsonl'
            path.write_text(json.dumps(row)+'\n')
            self.assertEqual(evaluation.cached_predictions(path,[item],scoring),[row])
            for corrupt in [[row,row],[{**row,'f1':0.}],[{**row,'sequence_sha256':'other'}]]:
                path.write_text(''.join(json.dumps(r)+'\n' for r in corrupt))
                with self.assertRaises(RuntimeError):
                    evaluation.cached_predictions(path,[item],scoring)

    def fixture(self, temporary):
        root = Path(temporary);out=root/'evaluation';first=root/'first';second=root/'second';parent=root/'PT'
        for folder in [first,second]:
            evaluation.write(folder/'plan.json',{'source_sha256':{}})
            (folder/'baseline.conf').write_text('frozen config')
        evaluation.write(parent/'plan.json',{'n':123,'pretrain_epochs':20})
        case={'id':'c5p10','condition':5,'pretrain_epoch':10,'pretrain_root':str(parent),
              'stage1':str(first),'stage2':str(second)}
        for folder,arm,epoch in [(first,'pretrain',10),(second,'mw_rank_restored',3)]:
            checkpoint=folder/f'runs/{arm}/epoch-{epoch:03d}.pt'
            checkpoint.parent.mkdir(parents=True);checkpoint.write_bytes(b'frozen model')
            evaluation.write(checkpoint.with_name(f'epoch-{epoch:03d}-complete.json'),
                             {'success':True,'checkpoint_sha256':evaluation.sha(checkpoint)})
            panel=[{'id':str(i),'source':'viral_fragments','sequence':'A'*(i+4),'pairs':[]} for i in range(3)]
            (folder/'data').mkdir();(folder/'data/test.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in panel))
            prediction=folder/f'evaluation/{arm}/epoch-{epoch:03d}/per_sequence.jsonl'
            prediction.parent.mkdir(parents=True);prediction.write_text(''.join(json.dumps(
                {'id':r['id'],'predicted_pairs':[]})+'\n' for r in panel))
            evaluation.write(prediction.with_name('summary.json'),{'success':True,'global_ft_epoch':6,
                'checkpoint_sha256':evaluation.sha(checkpoint),'predictions_sha256':evaluation.sha(prediction)})
            evaluation.write(checkpoint.parent/'complete.json',{'success':True})
        return out,case

    def test_selected_checkpoint_binds_model_controls_and_expected_optimizer_step(self):
        with tempfile.TemporaryDirectory() as temporary:
            out,case=self.fixture(temporary)
            with patch.object(evaluation,'OUT',out):
                first=evaluation.bind_model(case,'pretrain')
                final=evaluation.bind_model(case,'mw_rank_restored')
                self.assertEqual(first['expected_optimizer_step'],1230)
                self.assertEqual(final['expected_optimizer_step'],1230+45192)
                self.assertEqual(len(first['controls']),3)
                self.assertEqual(evaluation.checked_spec(first['model']),first)
                Path(first['checkpoint']).write_bytes(b'changed')
                with self.assertRaisesRegex(RuntimeError,'model/configuration changed'):
                    evaluation.bind_model(case,'pretrain')

    def test_waits_for_exact_epoch_and_final_FT_six(self):
        with tempfile.TemporaryDirectory() as temporary:
            out,case=self.fixture(temporary)
            with patch.object(evaluation,'OUT',out):
                self.assertIsNone(evaluation.bind_model({**case,'pretrain_epoch':20,'id':'c5p20'},'pretrain'))
                summary=Path(case['stage2'])/'evaluation/mw_rank_restored/epoch-003/summary.json'
                value=evaluation.read(summary);value['global_ft_epoch']=5;evaluation.write(summary,value)
                with self.assertRaisesRegex(RuntimeError,'cumulative FT epoch6'):
                    evaluation.bind_model(case,'mw_rank_restored')

    def test_prediction_checksum_mismatch_blocks_binding(self):
        with tempfile.TemporaryDirectory() as temporary:
            out,case=self.fixture(temporary)
            prediction=Path(case['stage1'])/'evaluation/pretrain/epoch-010/per_sequence.jsonl'
            prediction.write_text(prediction.read_text()+'\n')
            with patch.object(evaluation,'OUT',out):
                with self.assertRaisesRegex(RuntimeError,'Control prediction checksum'):
                    evaluation.bind_model(case,'pretrain')

    def test_bound_model_ids_include_all_four_methods(self):
        cases=[{'id':'c1p10'},{'id':'c5p20'}]
        identifiers=[r[2] for r in evaluation.model_ids({'cases':cases})]
        self.assertEqual(len(identifiers),8)
        self.assertIn('c5p20_shape_fy_restored',identifiers)
        self.assertIn('c1p10_pretrain',identifiers)

    def test_segmented_worker_preserves_predictions_and_submits_held_continuation(self):
        scoring=evaluation.load('test_external_resume_adapter',evaluation.REFERENCE/'score_adapter.py')
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);items=[{'id':str(i),'sequence':'ACGU','sequence_sha256':str(i),'pairs':[]} for i in range(3)]
            path=root/'panel.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in items))
            plan={'panels':{'Archive-NoFam':{'path':str(path),'n':3},'bpRNAnew':{'path':str(path),'n':3}}}
            spec={'source':'frozen','config':'frozen','checkpoint':'frozen','checkpoint_sha256':'digest',
                  'expected_optimizer_step':1230,'controls':[{**items[0],'expected_pairs':[]}]}
            torch=SimpleNamespace(set_num_threads=lambda _:None,backends=SimpleNamespace(cudnn=SimpleNamespace()),
                                  inference_mode=contextlib.nullcontext)
            adapter=SimpleNamespace(create_model=lambda *_:lambda seq,**kw:(None,None,[[0]*5]),
                restore=lambda *_:{'optimizer_state_dict':{'param_groups':[{'step':1230}]}})
            loader=lambda name,path:adapter if str(path).endswith('mx_adapter.py') else scoring
            with patch.object(evaluation,'OUT',root),patch.object(evaluation,'verify',return_value=plan), \
                    patch.object(evaluation,'checked_spec',return_value=spec),patch.object(evaluation,'load',side_effect=loader), \
                    patch.dict('sys.modules',{'torch':torch}),patch.dict('os.environ',{'JOB_ID':'101'}), \
                    patch.object(evaluation,'submit',return_value='102') as submit:
                with patch.object(evaluation,'SOFT_SECONDS',0):
                    evaluation.work('example')
                submit.assert_called_once_with('work-example-after-101','work','example',hold='101')
                output=root/'predictions/Archive-NoFam/example/predictions.jsonl'
                self.assertEqual(len(evaluation.rows(output)),1)
                evaluation.work('example')
                self.assertTrue(evaluation.model_complete(plan,'example'))
                self.assertEqual(len(evaluation.rows(output)),3)
                self.assertEqual(len(evaluation.rows(root/'predictions/bpRNAnew/example/predictions.jsonl')),3)


if __name__=='__main__':
    unittest.main()
