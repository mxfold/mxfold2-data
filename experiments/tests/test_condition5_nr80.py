import importlib.util
import json
import os
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from scripts.efold_hf import condition5_nr80_data as data
from scripts.efold_hf import pretrain_condition5_nr80 as pretrain
from scripts.efold_hf import finetune_milestones as gate
from tests.test_finetune_milestones import load_template


class Condition5Tests(unittest.TestCase):
    def test_native_priority_boundary_all_original_bpRNA_and_labels(self):
        # Regression: the forward native 2D pass misses this 80/100nt hit.
        experimental='CTCGACGATTTGCAGATTCGACGCGCCGCGCAACAAGTAGAATTTGAGCGAGGTCGGGGACACTGTAGCAAGGGGAAGGT'
        pseudo='ACGTACGTACCTGGACGCTTTGGAGATACGACTCGCCTCGCACCAAGAAGAAATTGATCGAGTTCGGTGACAGTGTATCAAGTGGAATGTTGCATGCATG'
        rng=random.Random(20261103)
        a=''.join(rng.choices('ACGT',k=100))
        mutate=lambda seq,start,stop:seq[:start]+seq[start:stop].translate(str.maketrans('ACGT','CGTA'))+seq[stop:]
        b=mutate(a,0,20)
        c=mutate(b,40,60)
        independent=''.join(rng.choices('ACGT',k=120))
        sequences=[('bp_a',a),('bp_b',b),('bp_boundary',experimental),('bp_short','AC'),
                   ('rc_boundary',pseudo),('rc_transitive',c),('rc_independent',independent)]
        original=[{'id':identifier,'source':'bpRNA-1m' if identifier.startswith('bp_') else 'rnacentral_synthetic',
            'sequence':seq.replace('T','U'),'pairs':[],'train_pairs':[],'provenance':identifier}
            for identifier,seq in sequences]
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source=root/'input.jsonl'
            source.write_text(''.join(json.dumps(r)+'\n' for r in original))
            summary=data.build(root/'output',source,data.sha(source),{'bpRNA-1m':4,'rnacentral_synthetic':3})
            retained=list(data.rows(root/'output/train.jsonl'))
            identifiers={r['id'] for r in retained}
            self.assertIn('bp_a',identifiers)
            self.assertNotIn('bp_b',identifiers)
            self.assertIn('bp_boundary',identifiers)
            self.assertIn('bp_short',identifiers)
            self.assertIn('rc_independent',identifiers)
            self.assertNotIn('rc_boundary',identifiers)
            self.assertNotIn('rc_transitive',identifiers)
            self.assertTrue(summary['cross_filter']['final_full_reverse_scan_zero_matches'])
            self.assertGreater(sum(r['removed_n'] for r in summary['cross_filter']['reverse']),0)
            self.assertEqual(summary['total_n'],4)
            originals={r['id']:r for r in original}
            self.assertTrue(all(r==originals[r['id']] for r in retained))
            again=data.build(root/'output',source,data.sha(source),{'bpRNA-1m':4,'rnacentral_synthetic':3})
            self.assertEqual(summary['train_sha256'],again['train_sha256'])
            (root/'output/train.jsonl').write_text('changed')
            with self.assertRaisesRegex(RuntimeError,'Completed dataset changed'):
                data.build(root/'output',source,data.sha(source),{'bpRNA-1m':4,'rnacentral_synthetic':3})

    def test_both_milestones_restore_dynamic_pretraining_optimizer_step(self):
        actual_read=Path.read_text
        def read(path,*args,**kwargs):
            if str(path).endswith('pretrain_condition5_nr80_20_20261103/bprna_rnacentral_nr80_20/plan.json'):
                return json.dumps({'n':54321,'pretrain_epochs':20})
            return actual_read(path,*args,**kwargs)
        with patch.object(Path,'read_text',autospec=True,side_effect=read):
            for epoch in (10,20):
                first=load_template('condition3_ribonanza_ft.py',5,epoch)
                second=load_template('condition3_pri_mrna_ft.py',5,epoch)
                self.assertEqual(first.REPLAY_POOL_COUNT,54321)
                self.assertEqual(first.SOURCE_STEP,54321*epoch)
                self.assertEqual(second.REPLAY_POOL_COUNT,54321)
                self.assertEqual(second.PARENT_NETWORK_STEP,54321*epoch+42873)
                self.assertEqual(second.STAGE1,first.ROOT)
                self.assertIn(f'condition5_pt{epoch:03d}',str(first.ROOT))

    def test_registry_freezes_both_requested_points_with_actual_NR80_count(self):
        templates=gate.PROJECT/'scripts/efold_hf'
        metric=Path(__file__).with_name('fixtures')/'condition1_metrics.py'
        with tempfile.TemporaryDirectory() as temporary:
            project=Path(temporary);root=project/'registry'
            parent=project/'qsub_logs/efold_hf'/gate.PRETRAINS[5][0]
            gate.write(parent/'plan.json',{'n':54321,'pretrain_epochs':20,
                'data_sha256':{'train.jsonl':'frozen_NR80_hash'}})
            target=project/'qsub_logs/efold_hf/condition1_ribonanza3_replay20_20261101/metrics.py'
            target.parent.mkdir(parents=True);target.write_bytes(metric.read_bytes())
            with patch.object(gate,'ROOT',root),patch.object(gate,'PROJECT',project), \
                    patch.object(gate,'TEMPLATES',templates):
                gate.setup([(5,10),(5,20)],final_reports=True)
                cases=gate.cases()
                gate.verify()
            self.assertEqual([c['id'] for c in cases],['c5p10','c5p20'])
            self.assertEqual([c['expected_optimizer_step'] for c in cases],[543210,1086420])
            self.assertTrue(all(c['replay_pool']==54321 and c['FT_epochs']==[3,3] for c in cases))
            self.assertTrue(all(c['pretrain_data_sha256']=='frozen_NR80_hash' for c in cases))
            self.assertTrue(all('20261103' in c['stage1'] for c in cases))

    def test_pretraining_jobs_wait_for_verified_data_and_gpu_preflight(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);data_root=root/'dataset'
            gate.write(data_root/'submitted_job.json',{'job_id':'101'})
            calls=[]
            def submit(arm,key,action,**kwargs):
                calls.append((arm,key,action,kwargs))
                return str(102+len(calls)-1)
            engine=SimpleNamespace(verify_code=lambda:None,submit_once=submit,write_json=gate.write,
                branch=lambda arm:root/'PT'/arm)
            with patch.object(pretrain,'ROOT',root/'PT'),patch.object(pretrain,'DATA',data_root):
                pretrain.submit(engine)
            self.assertEqual([c[3]['hold'] for c in calls],['101','102','103'])
            self.assertEqual([c[2] for c in calls],['prepare','preflight','work'])
            result=gate.read(root/'PT/pipeline.json')
            self.assertEqual(result['FT_start_epochs'],[10,20])

    def test_failed_combined_verification_blocks_pretraining_and_FT(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);dataset=root/'dataset';branch=root/'PT'
            for name in ['complete.json','summary.json','label_preservation_verification.json']:
                gate.write(dataset/name,{'success':True})
            gate.write(dataset/'combined_verification/result.json',{'success':False,'merged_n':1})
            engine=SimpleNamespace(verify_code=lambda:None,legacy=lambda:None,branch=lambda arm:branch)
            with patch.object(pretrain,'DATA',dataset),patch.object(pretrain,'register_ft') as register:
                with self.assertRaisesRegex(RuntimeError,'verification is incomplete'):
                    pretrain.prepare(engine)
            self.assertFalse((branch/'data').exists())
            register.assert_not_called()


if __name__=='__main__':
    unittest.main()
