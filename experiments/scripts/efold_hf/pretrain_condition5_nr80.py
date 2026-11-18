#!/usr/bin/env python3
"""Condition5: independent NR80 bpRNA/RNAcentral PT20, followed by PT10/PT20 FT.

Reuse the frozen conditions1..4 training engine and sources byte-for-byte.
Dataset preparation and frozen milestone registration are the only additions.
"""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess

PROJECT = Path('/gs/bs/tga-satolab-gtex/satoken/repos/mxfold2-ryntaro')
REFERENCE = PROJECT/'qsub_logs/efold_hf/pretrain_variants_20261031'
ROOT = Path(os.environ.get('VARIANTS_ROOT',str(PROJECT/'qsub_logs/efold_hf/pretrain_condition5_nr80_20_20261103')))
DATA = PROJECT/'qsub_logs/efold_hf/condition5_nr80_dataset_20261103'
FT_ROOT = PROJECT/'qsub_logs/efold_hf/ft_condition5_pt010_pt020_20261103'
ARM = 'bprna_rnacentral_nr80_20'


def read(path):
    return json.loads(Path(path).read_text())


def engine():
    spec=importlib.util.spec_from_file_location('condition5_frozen_engine',ROOT/'engine.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.ROOT=ROOT;module.ARMS={ARM:20}
    return module


def init():
    if (ROOT/'setup-complete.json').exists():
        engine().verify_code()
        return
    if ROOT.exists():
        raise RuntimeError('Inspect incomplete condition5 setup before reuse')
    # The reference manifest and source hashes must both still be valid.
    import hashlib
    def sha(path):
        h=hashlib.sha256()
        with Path(path).open('rb') as handle:
            for block in iter(lambda:handle.read(1<<20),b''):h.update(block)
        return h.hexdigest()
    for path,digest in read(REFERENCE/'implementation.json').items():
        if sha(path)!=digest:
            raise RuntimeError('Reference implementation changed: '+path)
    ROOT.mkdir(parents=True)
    shutil.copytree(REFERENCE/'source',ROOT/'source',symlinks=True,
        ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ['source_hashes.json','baseline.conf','test_full.jsonl','test_monitor.jsonl','common_bp_monitor.jsonl','job.sh']:
        shutil.copyfile(REFERENCE/name,ROOT/name)
    shutil.copyfile(REFERENCE/'controller.py',ROOT/'engine.py')
    shutil.copyfile(__file__,ROOT/'controller.py')
    templates=ROOT/'ft_templates';templates.mkdir()
    names=['finetune_milestones.py','finetune_milestones_job.sh','completed_ft_comparison.py',
        'condition3_ribonanza_ft.py','condition3_ribonanza_ft_job.sh',
        'condition3_pri_mrna_ft.py','condition3_pri_mrna_ft_job.sh']
    for name in names:
        shutil.copyfile(PROJECT/'scripts/efold_hf'/name,templates/name)
    module=engine();folder=module.branch(ARM)
    (folder/'scheduler').mkdir(parents=True)
    (folder/'source').symlink_to(ROOT/'source',target_is_directory=True)
    os.link(ROOT/'baseline.conf',folder/'baseline.conf')
    os.link(ROOT/'job.sh',folder/'job.sh')
    for relative,digest in read(ROOT/'source_hashes.json').items():
        if sha(ROOT/'source'/relative)!=digest:
            raise RuntimeError('Frozen model/worker source differs: '+relative)
    files=[ROOT/name for name in ['controller.py','engine.py','job.sh','baseline.conf',
        'source_hashes.json','test_full.jsonl','test_monitor.jsonl','common_bp_monitor.jsonl']]
    files += [templates/name for name in names]
    module.write_json(ROOT/'implementation.json',{str(p):sha(p) for p in files})
    module.write_json(ROOT/'reference_equivalence.json',{'success':True,
        'reference_root':str(REFERENCE),'reference_manifest':read(REFERENCE/'implementation.json'),
        'controller_engine_identical':sha(ROOT/'engine.py')==sha(REFERENCE/'controller.py'),
        'all_model_and_worker_sources_identical':True,
        'baseline_identical':sha(ROOT/'baseline.conf')==sha(REFERENCE/'baseline.conf'),
        'test_panels_identical':all(sha(ROOT/n)==sha(REFERENCE/n) for n in
            ['test_full.jsonl','test_monitor.jsonl','common_bp_monitor.jsonl'])})
    module.write_json(ROOT/'setup-complete.json',{'success':True,'condition':5,'arms':{ARM:20},
        'training_max_length':1000,'hard_wall_seconds':module.HARD_SECONDS,'worker_soft_hours':module.SOFT_HOURS,
        'initialization':'Fresh model and optimizer with seed1234; independent condition5',
        'FT_checkpoints':[10,20],'FT_methods':['MW Rank','IMLE-NLL','SHAPE-fy'],
        'FT_curriculum':'Ribonanza3 epochs then pri-miRNA/human mRNA3 epochs, replay20%',
        'thermodynamic_regularization':.125})
    module.verify_code()
    print('Frozen condition5 pretraining and FT pipeline',ROOT,flush=True)


def register_ft(module):
    module.verify_code()
    environment={**os.environ,'MILESTONE_ROOT':str(FT_ROOT),'MILESTONE_TEMPLATES':str(ROOT/'ft_templates'),
        'TZ':'Asia/Tokyo'}
    subprocess.run([str(PROJECT/'.venv/bin/python'),str(ROOT/'ft_templates/finetune_milestones.py'),
        'submit','--cases','5:10,5:20','--final-report'],cwd=PROJECT,env=environment,check=True)
    module.write_json(ROOT/'FT_registration.json',{'success':True,'root':str(FT_ROOT),
        'cases':read(FT_ROOT/'cases.json')['cases'],'initial_jobs':read(FT_ROOT/'initial_job.json')})


def prepare(module):
    module.verify_code();folder=module.branch(ARM);wf=module.legacy()
    if (folder/'prepare-complete.json').exists():
        wf.verify(folder);register_ft(module)
        return
    complete,summary,labels,native=[read(DATA/name) for name in
        ['complete.json','summary.json','label_preservation_verification.json','combined_verification/result.json']]
    if not all(d['success'] for d in [complete,summary,labels,native]):
        raise RuntimeError('Condition5 dataset verification is incomplete')
    checksum=module.sha(DATA/'train.jsonl')
    if any(d['train_sha256']!=checksum for d in [complete,summary,labels]) or native['merged_n']!=0:
        raise RuntimeError('Condition5 verified dataset changed')
    values=module.read_jsonl(DATA/'train.jsonl')
    count=len(values);counts=dict(Counter(r['source'] for r in values))
    if count!=summary['total_n'] or counts!=summary['sources_retained'] or set(counts)!={'bpRNA-1m','rnacentral_synthetic'}:
        raise RuntimeError('Wrong condition5 source counts')
    if len({r['id'] for r in values})!=count or len({r['sequence'] for r in values})!=count:
        raise RuntimeError('Duplicate condition5 IDs/sequences')
    if any(not 1<=len(r['sequence'])<=1000 or 'train_pairs' not in r for r in values):
        raise RuntimeError('Invalid condition5 training sequence or labels')
    data=folder/'data';data.mkdir(exist_ok=True)
    shutil.copyfile(DATA/'train.jsonl',data/'train.jsonl')
    os.link(data/'train.jsonl',data/'bprna.jsonl')
    module.write_jsonl(data/'train_monitor.jsonl',module.stable_sample(values,128,namespace='actual-training-monitor'))
    train_sequences={r['sequence'] for r in values}
    tests=module.read_jsonl(ROOT/'test_full.jsonl')
    overlaps=[r for r in tests if r['sequence'] in train_sequences]
    module.write_json(folder/'test_exact_overlap.json',{'ids':[r['id'] for r in overlaps],
        'n':len(overlaps),'by_source':dict(Counter(r['source'] for r in overlaps)),
        'note':'Same full-test and condition-specific exact-overlap exclusion metrics as conditions1..4'})
    module.write_json(folder/'common_bp_membership.json',{
        r['id']:r['sequence'] in train_sequences for r in module.read_jsonl(ROOT/'common_bp_monitor.jsonl')})
    module.write_json(folder/'dataset.json',{'n':count,'sources':counts,'max_length':1000,
        'original_dataset':str(DATA/'train.jsonl'),'dataset_sha256':checksum,
        'length_range':[min(len(r['sequence']) for r in values),max(len(r['sequence']) for r in values)],
        'original_pairs':sum(len(r['pairs']) for r in values),'train_pairs':sum(len(r['train_pairs']) for r in values),
        'labels_unchanged':True,'priority':summary['priority'],
        'RNAcentral_labels':'Supplied RNAstructure Fold sequence-only pseudo-labels'})
    reference_plan=read(REFERENCE/'bprna_trainab_nr80_20/plan.json')
    plan={**reference_plan,'condition':5,'arm':ARM,'n':count,'pretrain_epochs':20,'finetune_epochs':0,
        'arms':[],'seed':1234,'source_sha256':read(ROOT/'source_hashes.json'),
        'data_sha256':{n:module.sha(data/n) for n in ['train.jsonl','bprna.jsonl','train_monitor.jsonl']},
        'config_sha256':module.sha(folder/'baseline.conf'),'job_sha256':module.sha(folder/'job.sh'),
        'initialization':'Fresh initialization with seed1234','initial_checkpoint_sha256':None,
        'original_epoch10_sha256':None,'dataset_provenance':str(DATA/'summary.json')}
    fields=['optimizer','lr','weight_decay','score_loss_weight','loss_func','perturb','dropout','accumulation',
        'amp','max_length','seed','config_sha256','job_sha256','source_sha256','checkpoint_every_examples',
        'hard_wall_seconds','soft_wall_seconds','worker_hours']
    differences={k:(reference_plan[k],plan[k]) for k in fields if reference_plan[k]!=plan[k]}
    if differences or plan['score_loss_weight']!=.125:
        raise RuntimeError('Baseline training settings differ: '+str(differences))
    run=folder/'runs/pretrain';run.mkdir(parents=True,exist_ok=True)
    if any(run.glob('*.pt')):
        raise RuntimeError('Independent condition5 must start without checkpoints')
    module.write_json(folder/'plan.json',plan)
    wf.verify(folder)
    module.write_json(ROOT/'baseline_equivalence.json',{'success':True,'compared_fields':fields,
        'differences':differences,'pretrain_epochs':20,'fresh_model_and_optimizer':True,
        'dataset_sha256':checksum,'dataset_n':count})
    module.write_json(folder/'prepare-complete.json',{'success':True,'n':count,'train_sha256':checksum})
    register_ft(module)
    print('Condition5 prepared',count,counts,'PT10/PT20 FT registered',flush=True)


def submit(module):
    module.verify_code()
    data_job=read(DATA/'submitted_job.json')['job_id']
    prep=module.submit_once(ARM,'prepare','prepare',hold=data_job)
    preflight=module.submit_once(ARM,'preflight','preflight',hold=prep)
    work=module.submit_once(ARM,'work-initial','work',hold=preflight)
    module.write_json(ROOT/'initial_jobs.json',{ARM:{'data':data_job,'prepare':prep,
        'preflight':preflight,'work':work}})
    module.write_json(ROOT/'pipeline.json',{'condition':5,'data_root':str(DATA),
        'pretraining_root':str(module.branch(ARM)),'pretraining_epochs':20,
        'FT_registry':str(FT_ROOT),'FT_start_epochs':[10,20],
        'automatic_FT_registration':'CPU prepare after dataset verification',
        'initial_jobs':{'data':data_job,'prepare':prep,'preflight':preflight,'work':work}})


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['init','submit','prepare','preflight','work','evaluate','summarize'])
    parser.add_argument('--arm',choices=[ARM],default=ARM)
    parser.add_argument('--epoch',type=int,default=int(os.environ.get('EVALUATION_EPOCH','0')))
    args=parser.parse_args()
    if args.action=='init':
        init();return
    module=engine()
    if args.action=='prepare':prepare(module)
    elif args.action=='submit':submit(module)
    elif args.action=='preflight':module.preflight(ARM)
    elif args.action=='work':module.work(ARM)
    elif args.action=='evaluate':module.evaluate(ARM,args.epoch)
    else:module.summarize_all()


if __name__=='__main__':
    main()
