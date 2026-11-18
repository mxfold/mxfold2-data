#!/usr/bin/env python3
"""Ribonanza FT3 from a selected pretraining condition and epoch.

Running experiments use frozen copies; selecting condition1 never mutates the
existing condition3 experiment or its queued continuations.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time

PROJECT = Path('/gs/bs/tga-satolab-gtex/satoken/repos/mxfold2-ryntaro')
CONDITION = int(os.environ.get('FT_CONDITION', '3'))
CASE_CONFIG = {
    1: ('pretrain_variants_20261031/bprna20', 45507, 20),
    2: ('pretrain_variants_20261031/bprna_rnacentral10', 255767, 10),
    3: ('pretrain_variants_20261031/bprna_trainab_nr80_20', 14831, 20),
    4: ('pretrain_combined_nr80_20_20261031/combined_nr80_20', 60065, 20),
    5: ('pretrain_condition5_nr80_20_20261103/bprna_rnacentral_nr80_20', None, 20),
}
if CONDITION not in CASE_CONFIG:
    raise ValueError('Unsupported pretraining condition')
PRETRAIN_NAME, REPLAY_POOL_COUNT, MAX_PRETRAIN_EPOCH = CASE_CONFIG[CONDITION]
PRETRAIN = PROJECT/'qsub_logs/efold_hf'/PRETRAIN_NAME
if REPLAY_POOL_COUNT is None:
    _parent_plan = json.loads((PRETRAIN/'plan.json').read_text())
    REPLAY_POOL_COUNT = int(_parent_plan['n'])
    if REPLAY_POOL_COUNT <= 0 or _parent_plan['pretrain_epochs'] != MAX_PRETRAIN_EPOCH:
        raise RuntimeError('Invalid frozen condition5 pretraining plan')
PRETRAIN_EPOCH = int(os.environ.get('FT_PRETRAIN_EPOCH', str(MAX_PRETRAIN_EPOCH)))
if not 1 <= PRETRAIN_EPOCH <= MAX_PRETRAIN_EPOCH:
    raise ValueError('Requested pretraining epoch exceeds the registered training plan')
SOURCE_STEP = REPLAY_POOL_COUNT*PRETRAIN_EPOCH
RUN_TAG = f'condition{CONDITION}' + (f'_pt{PRETRAIN_EPOCH:03d}' if 'FT_PRETRAIN_EPOCH' in os.environ else '')
RUN_DATE = '20261103' if CONDITION==5 else '20261101'
ROOT = Path(os.environ.get('CURRICULUM_ROOT', str(PROJECT /
    f'qsub_logs/efold_hf/{RUN_TAG}_ribonanza3_replay20_{RUN_DATE}')))
_binding = ROOT/'checkpoint_binding.json'
_plan = ROOT/'plan.json'
_parent_marker = PRETRAIN/f'runs/pretrain/epoch-{PRETRAIN_EPOCH:03d}-complete.json'
CHECKPOINT_SHA = (json.loads(_binding.read_text())['checkpoint_sha256'] if _binding.exists()
    else json.loads(_plan.read_text()).get('pretrain_checkpoint_sha256') if _plan.exists()
    else json.loads(_parent_marker.read_text())['checkpoint_sha256'] if _parent_marker.exists() else None)
REFERENCE = PROJECT / 'qsub_logs/efold_hf/bprna10_restored_ft5_replay20_20261026'
PANEL = PROJECT / 'qsub_logs/efold_hf/paper_full_20261030/panel.jsonl'
ARMS = ('mw_rank_restored', 'nll_restored', 'shape_fy_restored')
QSUB = '/apps/t4/rhel9/uge/latest/bin/lx-amd64/qsub'
SOFT_HOURS = 22


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def rows(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_rows(path, values):
    with Path(path).open('w') as handle:
        for row in values:
            handle.write(json.dumps(row, separators=(',', ':'), allow_nan=False)+'\n')


@contextmanager
def lock(path, nonblocking=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0))
        try: yield
        finally: fcntl.flock(handle, fcntl.LOCK_UN)


def module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def workflow():
    sys.path[:0] = [str(ROOT/'source'), str(ROOT/'source/scripts/efold_hf')]
    sys.path.append(str(PROJECT/'.cache/bprna-benchmark-deps'))
    import workflow as wf
    if Path(wf.__file__).resolve() != (ROOT/'source/scripts/efold_hf/workflow.py').resolve():
        raise RuntimeError('Wrong frozen workflow')
    return wf


def verify():
    for path, checksum in json.loads((ROOT/'implementation.json').read_text()).items():
        if sha(path) != checksum: raise RuntimeError('Frozen implementation changed: '+path)
    plan = workflow().verify(ROOT)
    if plan.get('condition', CONDITION) != CONDITION:
        raise RuntimeError('Wrong curriculum condition')
    if plan['pretrain_epochs'] != PRETRAIN_EPOCH:
        raise RuntimeError('Wrong starting pretraining epoch')
    checkpoint = ROOT/f'runs/pretrain/epoch-{PRETRAIN_EPOCH:03d}.pt'
    if not plan.get('global_ft_epoch_offset') and sha(checkpoint) != plan['pretrain_checkpoint_sha256']:
        raise RuntimeError('Frozen starting checkpoint changed')
    marker = ROOT/'prepare-complete.json'
    if marker.exists():
        expected = json.loads(marker.read_text()).get('plan_sha256')
        if expected and sha(ROOT/'plan.json') != expected:
            raise RuntimeError('Frozen curriculum plan changed')
    return plan


def prepare():
    if (ROOT/'prepare-complete.json').exists(): verify(); return
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT/'source').exists(): raise RuntimeError('Inspect incomplete preparation before reuse')
    bootstrap = ROOT/'bootstrap.json'
    if bootstrap.exists():
        for path, digest in json.loads(bootstrap.read_text())['files'].items():
            if sha(path) != digest: raise RuntimeError('Frozen bootstrap changed: '+path)
    if not CHECKPOINT_SHA: raise RuntimeError('Requested checkpoint is not ready or bound')
    reference_plan = json.loads((REFERENCE/'plan.json').read_text())
    for relative, checksum in reference_plan['source_sha256'].items():
        if sha(REFERENCE/'source'/relative) != checksum:
            raise RuntimeError('Successful FT source changed: '+relative)
    checkpoint = PRETRAIN/f'runs/pretrain/epoch-{PRETRAIN_EPOCH:03d}.pt'
    if sha(checkpoint) != CHECKPOINT_SHA: raise RuntimeError('Pretraining checkpoint changed')
    pretrain_plan = json.loads((PRETRAIN/'plan.json').read_text())
    if sha(PRETRAIN/'data/train.jsonl') != pretrain_plan['data_sha256']['train.jsonl']:
        raise RuntimeError('Pretraining replay data changed')
    for relative, checksum in reference_plan['source_sha256'].items():
        if relative.startswith('mxfold2/') and pretrain_plan['source_sha256'][relative] != checksum:
            raise RuntimeError('Pretraining/FT model source differs: '+relative)
    shutil.copytree(REFERENCE/'source', ROOT/'source', symlinks=True,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copyfile(REFERENCE/'baseline.conf', ROOT/'baseline.conf')
    if Path(__file__).resolve() != (ROOT/'controller.py').resolve():
        shutil.copyfile(Path(__file__), ROOT/'controller.py')
    if not (ROOT/'job.sh').exists():
        shutil.copyfile(PROJECT/'scripts/efold_hf/condition3_ribonanza_ft_job.sh', ROOT/'job.sh')
    if not (ROOT/'metrics.py').exists():
        shutil.copyfile(PROJECT/'scripts/chemical_consistency/map_mw_metrics.py', ROOT/'metrics.py')
    wf = workflow(); data = ROOT/'data'; data.mkdir()
    source = REFERENCE/'data/chemical.jsonl'
    if sha(source) != reference_plan['data_sha256']['chemical.jsonl']:
        raise RuntimeError('Frozen chemical source changed')
    primary = [r for r in rows(source) if r['source']=='ribo500-blast']
    replay = rows(PRETRAIN/'data/train.jsonl')
    if len(primary)!=45729 or len(replay)!=REPLAY_POOL_COUNT:
        raise RuntimeError('Unexpected first-stage data counts')
    replay_sequences = {r['sequence'] for r in replay}
    shared_sequences = {r['sequence'] for r in primary} & replay_sequences
    if shared_sequences and CONDITION in (1,3):
        raise RuntimeError('Primary/replay sequence overlap')
    write_json(ROOT/'primary_replay_overlap.json', {'n':len(shared_sequences),
        'primary_ids':[r['id'] for r in primary if r['sequence'] in shared_sequences],
        'policy':'Use the complete pretraining set for replay; any within-training overlap is recorded'})
    if any(not 1<=len(r['sequence'])<=1000 for r in primary+replay):
        raise RuntimeError('Training length exceeds1000')
    write_rows(data/'chemical.jsonl', primary)
    shutil.copyfile(PRETRAIN/'data/train.jsonl', data/'bprna.jsonl')
    write_rows(data/'assisted.jsonl', [])
    tests = rows(PANEL)
    viral = json.loads((PROJECT/'data/efold_hf/raw/viral_fragments.json').read_text())
    for row in tests:
        row['observations'] = (wf.clean_observations(viral[row['reference']])
                               if row['source']=='viral_fragments' else [])
    if dict(Counter(r['source'] for r in tests)) != {
            'PDB':355,'archiveII':3370,'viral_fragments':40,'lncRNA':30}:
        raise RuntimeError('Wrong full published test panel')
    if {r['sequence'] for r in primary} & {r['sequence'] for r in tests}:
        raise RuntimeError('Ribonanza contains a test sequence')
    write_rows(data/'test.jsonl', tests)
    overlap = {r['sequence'] for r in replay+primary}
    excluded = [r['id'] for r in tests if r['sequence'] in overlap]
    write_json(ROOT/'test_exact_overlap.json', {'ids':excluded,'n':len(excluded),
        'by_source':dict(Counter(r['source'] for r in tests if r['id'] in set(excluded)))})
    pre = ROOT/'runs/pretrain'; pre.mkdir(parents=True)
    shutil.copyfile(checkpoint, pre/checkpoint.name)
    shutil.copyfile(_parent_marker, pre/_parent_marker.name)
    for arm in ARMS:
        (ROOT/'runs'/arm).mkdir(parents=True)
    for name in ('scheduler','dispatch','validation'): (ROOT/name).mkdir(exist_ok=True)
    n_replay = int(math.floor(len(primary)/4+.5)); slots=len(primary)+n_replay
    plan = {**reference_plan, 'root':str(ROOT), 'pretrain_epochs':PRETRAIN_EPOCH,
        'condition':CONDITION, 'parent_experiment':str(PRETRAIN),
        'matched_finetune_reference':str(REFERENCE),
        'pretrain_checkpoint_sha256':CHECKPOINT_SHA, 'finetune_epochs':3,
        'arms':list(ARMS), 'per_arm_preflight':list(ARMS),
        'worker_hours':SOFT_HOURS, 'checkpoint_every_examples':100,
        'counts':{'primary':len(primary),'replay_pool':len(replay),
                  'primary_per_epoch':len(primary),'replay_per_epoch':n_replay,
                  'slots_per_epoch':slots,'optimizer_updates_per_epoch':math.ceil(slots/4)},
        'epoch_definition':f'Each of {len(primary)} Ribonanza sequences once plus {n_replay} cyclic replay draws; {slots} slots and {math.ceil(slots/4)} optimizer updates per epoch',
        'initialization':f'Restore condition{CONDITION} epoch{PRETRAIN_EPOCH} model and AdaBelief moments/step{SOURCE_STEP}; FT seed0; new Wu moments0',
        'optimizer_initialization':f'Restore named condition{CONDITION} epoch{PRETRAIN_EPOCH} AdaBelief moments and step{SOURCE_STEP}; new Wu moments0',
        'data_sha256':{p.name:sha(p) for p in data.glob('*.jsonl')},
        'source_sha256':reference_plan['source_sha256'],
        'config_sha256':sha(ROOT/'baseline.conf'),'job_sha256':sha(ROOT/'job.sh'),
        'replay_fraction':.2,'replay_source':f'Condition{CONDITION} pretraining{REPLAY_POOL_COUNT} sequences',
        'filtering':'Frozen eFold Ribonanza primary; complete condition-specific pretraining replay, lengths<=1000; within-training overlaps audited',
        'excluded_source':'RNAcentral is absent from chemical primary; replay follows the registered pretraining condition',
        'RNAcentral_policy':'Synthetic structural replay retained' if CONDITION in (2,4,5) else 'No RNAcentral data',
        'evaluation':'Every FT epoch on full published test3795; predicted structure only, no BPP; exact-pair macro/micro F1 and MW-oriented AUROC/Spearman on measured viral profiles',
        'next_stage':'Matching arm continues for3 epochs on pri_miRNA and human_mRNA; viral and lncRNA remain test data',
        'test_limitations':'Full published PDB355/ArchiveII3370/viral40/lncRNA30; pretraining exact overlaps are reported and excluded in parallel; no new homology exclusion',
        'scheduler_hard_seconds':85800}
    write_json(ROOT/'plan.json', plan)
    files=[ROOT/n for n in ('controller.py','job.sh','metrics.py','baseline.conf')]
    files += [p for p in (ROOT/'checkpoint_binding.json',ROOT/'bootstrap.json') if p.exists()]
    write_json(ROOT/'implementation.json',{str(p):sha(p) for p in files})
    verify()
    audit_cpu_initialization()
    baseline_cached = score_saved_baseline()
    write_json(ROOT/'prepare-complete.json',{'success':True,'primary_n':len(primary),
        'replay_pool_n':len(replay),'slots_per_epoch':slots,'starting_checkpoint_sha256':CHECKPOINT_SHA,
        'plan_sha256':sha(ROOT/'plan.json'),'condition':CONDITION,'source_optimizer_step':SOURCE_STEP,
        'CPU_initialization_validated':list(ARMS),'baseline_cached':baseline_cached})
    print(json.dumps(plan['counts']), flush=True)


def audit_cpu_initialization():
    import torch
    wf = workflow()
    source = torch.load(ROOT/f'runs/pretrain/epoch-{PRETRAIN_EPOCH:03d}.pt', map_location='cpu', weights_only=False)
    for arm in ARMS:
        wf.seed_all(0)
        model, optimizer, _, _, shapes, _ = wf.build(ROOT, arm, -1)
        path = ROOT/'validation'/f'cpu-initialization-{arm}.json'
        wf.initialize_finetuning(ROOT, arm, model, optimizer, shapes, audit_path=path)
        for name, parameter in model.state_dict().items():
            torch.testing.assert_close(parameter, source['model_state_dict'][name], rtol=0, atol=0)
        audit = json.loads(path.read_text())
        if audit['source_step'] != SOURCE_STEP or audit['target_step'] != SOURCE_STEP:
            raise RuntimeError('Wrong CPU initialization optimizer step')
        for group in optimizer.param_groups[1:]:
            if group['step'] != 0:
                raise RuntimeError('New Wu group did not start at step0')
            for parameter in group['params']:
                for key in ('exp_avg', 'exp_avg_var'):
                    if torch.count_nonzero(optimizer.state[parameter][key]):
                        raise RuntimeError('New Wu moments are not zero')
        write_json(path, {**audit,'success':True,'condition':CONDITION,
                         'all_model_parameters_equal_checkpoint':True})
        print('CPU initialization verified',CONDITION,arm,SOURCE_STEP,flush=True)


def submit_once(key, action, arm=ARMS[0], epoch=0, hold=None):
    if arm not in ARMS and not (arm=='pretrain' and action=='evaluate'): raise ValueError(arm)
    marker=ROOT/'dispatch'/(key+'.json')
    with lock(ROOT/'submission.lock'):
        if marker.exists(): return json.loads(marker.read_text())['job_id']
        duration='06:00:00' if action=='prepare' else '00:45:00' if action=='preflight' else '23:50:00'
        resource='cpu_8=1' if action=='prepare' else 'gpu_h=1'
        env=f'CURRICULUM_ROOT={ROOT},FT_CONDITION={CONDITION},FT_PRETRAIN_EPOCH={PRETRAIN_EPOCH},FT_ACTION={action},FT_ARM={arm},FT_EPOCH={epoch}'
        args=[QSUB,'-terse','-cwd','-j','y','-A','tga-satolab-gtex','-r','y',
              '-N',f'c{CONDITION}p{PRETRAIN_EPOCH}r_'+arm[:13]+'_'+action[:3]+str(epoch),'-l',resource,
              '-l','h_rt='+duration,'-l','hostname=!r22n6','-o',str(ROOT/'scheduler'),'-v',env]
        if hold: args+=['-hold_jid',str(hold)]
        args.append(str(ROOT/'job.sh'))
        command=['sg','tga-satolab-gtex','-c',shlex.join(args)]
        for attempt in range(3):
            result=subprocess.run(command,cwd=PROJECT,capture_output=True,text=True)
            if result.returncode==0: break
            if attempt==2: raise RuntimeError(result.stderr)
            time.sleep(2**attempt)
        if not re.fullmatch(r'\d+(?:\..+)?',result.stdout.strip()):
            raise RuntimeError('Unexpected qsub output: '+result.stdout)
        job=result.stdout.strip().split('.')[0]
        record={'job_id':job,'action':action,'arm':arm,'epoch':epoch,
                'hold_jid':hold,'time':time.time(),'command':args}
        write_json(marker,record)
        with (ROOT/'submissions.jsonl').open('a') as handle: handle.write(json.dumps(record)+'\n')
        print('submitted',arm,action,epoch,job,flush=True)
        return job


def dispatch(arm):
    previous=None
    for epoch in range(1,4):
        path=ROOT/'runs'/arm/f'epoch-{epoch:03d}-complete.json'
        if path.exists():
            previous=submit_once(f'evaluate-{arm}-{epoch}','evaluate',arm,epoch,hold=previous)


def train(arm):
    verify(); wf=workflow()
    marker=ROOT/'validation'/f'preflight-{arm}.json'
    if not json.loads(marker.read_text())['success']: raise RuntimeError('GPU resume validation missing')
    with lock(ROOT/'runs'/arm/'worker.lock',nonblocking=True):
        wf.dispatch_completed=lambda *_: dispatch(arm)
        def resume(*args,**kwargs):
            job=os.environ.get('JOB_ID')
            return submit_once(f'resume-{arm}-after-{job}','work',arm,hold=job)
        wf.submit_job=resume
        dispatch(arm)
        wf.train_worker(ROOT,arm,0,hours=SOFT_HOURS)


def preflight(arm):
    import numpy as np
    import torch
    verify()
    if not json.loads((ROOT/'prepare-complete.json').read_text())['success']:
        raise RuntimeError('Dataset preparation failed')
    wf=workflow(); wf.preflight(ROOT,0,arms=(arm,))
    audit=json.loads((ROOT/f'preflight-{arm}-initialization.json').read_text())
    if audit['source_step']!=SOURCE_STEP or audit['target_step']!=SOURCE_STEP:
        raise RuntimeError('Wrong pretrained optimizer step')
    primary=sorted(rows(ROOT/'data/chemical.jsonl'),key=lambda r:(len(r['sequence']),r['id']))
    replay=sorted(rows(ROOT/'data/bprna.jsonl'),key=lambda r:(len(r['sequence']),r['id']))
    primary=[primary[i] for i in (0,len(primary)//6,2*len(primary)//6,len(primary)//2,
                                  4*len(primary)//6,5*len(primary)//6,len(primary)-1)]
    replay=[replay[i] for i in (0,len(replay)//2,len(replay)-1)]
    plan=json.loads((ROOT/'plan.json').read_text())
    def mini(name):
        root=ROOT/'preflight-tests'/arm/name
        if root.exists(): shutil.rmtree(root)
        (root/'data').mkdir(parents=True); (root/'runs'/arm).mkdir(parents=True)
        (root/'runs/pretrain').mkdir()
        (root/'source').symlink_to(ROOT/'source',target_is_directory=True)
        os.link(ROOT/'baseline.conf',root/'baseline.conf'); os.link(ROOT/'job.sh',root/'job.sh')
        checkpoint_name=f'epoch-{PRETRAIN_EPOCH:03d}.pt'
        os.link(ROOT/'runs/pretrain'/checkpoint_name,root/'runs/pretrain'/checkpoint_name)
        write_rows(root/'data/chemical.jsonl',primary); write_rows(root/'data/bprna.jsonl',replay)
        write_rows(root/'data/assisted.jsonl',[]); write_rows(root/'data/test.jsonl',[])
        local={**plan,'finetune_epochs':2,'data_sha256':{p.name:sha(p) for p in (root/'data').glob('*.jsonl')}}
        write_json(root/'plan.json',local); write_json(root/f'preflight-{arm}.json',{'success':True})
        return root
    unbroken,segmented=mini('unbroken'),mini('segmented')
    calls=[]; wf.submit_job=lambda *a,**k:calls.append((str(a),k)); wf.dispatch_completed=lambda *_:None
    wf.train_worker(unbroken,arm,0,hours=1)
    for _ in range(12):
        wf.train_worker(segmented,arm,0,hours=0)
        if (segmented/'runs'/arm/'complete.json').exists(): break
    if not (segmented/'runs'/arm/'complete.json').exists(): raise RuntimeError('Segmented smoke did not finish')
    a=torch.load(unbroken/'runs'/arm/'latest.pt',map_location='cpu',weights_only=False)
    b=torch.load(segmented/'runs'/arm/'latest.pt',map_location='cpu',weights_only=False)
    def compare(x,y):
        if torch.is_tensor(x): torch.testing.assert_close(x,y,rtol=1e-6,atol=1e-7)
        elif isinstance(x,np.ndarray):
            if not np.array_equal(x,y): raise RuntimeError('NumPy state differs')
        elif isinstance(x,dict):
            if x.keys()!=y.keys(): raise RuntimeError('State keys differ')
            for key in x: compare(x[key],y[key])
        elif isinstance(x,(list,tuple)):
            if len(x)!=len(y): raise RuntimeError('State size differs')
            for xx,yy in zip(x,y,strict=True): compare(xx,yy)
        elif x!=y: raise RuntimeError('State value differs')
    for key in ('model_state_dict','optimizer_state_dict','shape_model_state_dict','rng','state'):
        compare(a[key],b[key])
    if a['state']['updates']!=6 or not calls: raise RuntimeError('Partial-batch resume not exercised')
    delta=max(float((x[torch.isfinite(x)]-b['model_state_dict'][k][torch.isfinite(x)]).abs().max())
              if torch.isfinite(x).any() else 0. for k,x in a['model_state_dict'].items())
    write_json(ROOT/'validation'/f'preflight-{arm}.json',{'success':True,
        'source_optimizer_step':SOURCE_STEP,'condition':CONDITION,'continuous_vs_segmented_max_parameter_delta':delta,
        'model_optimizer_Wu_RNG_cursor_restored':True,'partial_accumulation_batch_tested':True,
        'forced_resumptions':len(calls),'smoke_optimizer_updates':6,'epochs':2,
        'accumulation_batch_sizes_per_epoch':[4,4,1]})
    print('Full GPU resume validation passed',arm,flush=True)


def pair_metrics(reference,predicted):
    ref,pred=set(map(tuple,reference)),set(map(tuple,predicted)); tp=len(ref&pred)
    both=not ref and not pred
    return {'tp':tp,'fp':len(pred)-tp,'fn':len(ref)-tp,
        'precision':tp/len(pred) if pred else 1. if both else None,
        'recall':tp/len(ref) if ref else 1. if both else None,
        'f1':2*tp/(len(ref)+len(pred)) if ref or pred else 1.}


def aggregate(values):
    output={}
    for source in sorted({r['source'] for r in values}):
        group=[r for r in values if r['source']==source]; result={'n':len(group)}
        for key in ('precision','recall','f1'):
            vals=[r[key] for r in group if r[key] is not None]
            result[key]=sum(vals)/len(vals) if vals else None
        tp,fp,fn=(sum(r[k] for r in group) for k in ('tp','fp','fn'))
        result['micro_f1']=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.
        output[source]=result
    return output


def chemical_scores(item,pairs,metrics):
    binary=metrics.from_pairs(item['sequence'],pairs); answer=[]
    for obs in item.get('observations',[]):
        modifier='DMS' if obs['probe']=='dms' else 'SHAPE'
        strata=metrics.score_profile({'sequence':item['sequence'],'modifier':modifier,'values':obs['values']},binary)
        answer.append({'probe':modifier,'strata':strata})
    return answer


def score_saved_baseline():
    """Reuse this condition's published-test structures with the same MW metrics."""
    metrics=module_at('frozen_baseline_structure_metrics',ROOT/'metrics.py')
    original=PRETRAIN/f'evaluation/epoch-{PRETRAIN_EPOCH:03d}'
    if not (original/'summary.json').exists(): return False
    metadata=json.loads((original/'summary.json').read_text())
    path=original/'per_sequence.jsonl'
    if (metadata['checkpoint_sha256']!=CHECKPOINT_SHA or
            metadata['predictions_sha256']!=sha(path)):
        raise RuntimeError('Cached pretrained evaluation changed')
    tests={r['id']:r for r in rows(ROOT/'data/test.jsonl')}
    result=[]
    for saved in rows(path):
        if saved['id'] not in tests: continue
        item=tests[saved['id']]; pairs=saved['predicted_pairs']
        result.append({'id':item['id'],'source':item['source'],'length':len(item['sequence']),
            'predicted_pairs':pairs,**pair_metrics(item['pairs'],pairs),
            'chemical':chemical_scores(item,pairs,metrics)})
    if {r['id'] for r in result}!=set(tests): return False
    if len(result)!=len(tests): raise RuntimeError('Duplicate cached baseline test IDs')
    destination=ROOT/f'evaluation/pretrain/epoch-{PRETRAIN_EPOCH:03d}'; destination.mkdir(parents=True,exist_ok=True)
    output=destination/'per_sequence.jsonl'; write_rows(output,result)
    excluded=set(json.loads((ROOT/'test_exact_overlap.json').read_text())['ids'])
    write_json(destination/'summary.json',{'success':True,'arm':'pretrain','epoch':PRETRAIN_EPOCH,
        'checkpoint_sha256':CHECKPOINT_SHA,'test':aggregate(result),
        'without_exact_training_overlap':aggregate([r for r in result if r['id'] not in excluded]),
        'chemical_strata_in_predictions':True,'no_BPP':True,
        'cached_prediction_source':str(path),'cached_source_sha256':sha(path),
        'predictions_sha256':sha(output)})
    return True


def evaluate(arm,epoch):
    import torch
    verify(); wf=workflow(); metrics=module_at('frozen_structure_metrics',ROOT/'metrics.py')
    checkpoint=ROOT/'runs'/arm/f'epoch-{epoch:03d}.pt'
    marker=json.loads(checkpoint.with_name(f'epoch-{epoch:03d}-complete.json').read_text())
    if sha(checkpoint)!=marker['checkpoint_sha256']: raise RuntimeError('Checkpoint changed')
    destination=ROOT/'evaluation'/arm/f'epoch-{epoch:03d}'; destination.mkdir(parents=True,exist_ok=True)
    with lock(destination/'worker.lock',nonblocking=True):
        if (destination/'summary.json').exists(): return
        wf.seed_all(0); model,_,_,_,_,_=wf.build(ROOT,'pretrain',0); wf.restore(checkpoint,model); model.eval()
        path=destination/'per_sequence.jsonl'; done={r['id']:r for r in rows(path)} if path.exists() else {}
        tests=rows(ROOT/'data/test.jsonl'); deadline=time.monotonic()+SOFT_HOURS*3600
        with torch.inference_mode(),path.open('a') as handle:
            for item in sorted(tests,key=lambda r:(-len(r['sequence']),r['id'])):
                if item['id'] in done: continue
                result=model([item['sequence']],return_partfunc=False)
                partners=result[2][0]
                pairs=[[i-1,int(j)-1] for i,j in enumerate(partners) if i>0 and j>i]
                row={'id':item['id'],'source':item['source'],'length':len(item['sequence']),
                     'predicted_pairs':pairs,**pair_metrics(item['pairs'],pairs),
                     'chemical':chemical_scores(item,pairs,metrics)}
                handle.write(json.dumps(row,allow_nan=False)+'\n'); handle.flush(); done[row['id']]=row
                if len(done)%100==0: print(arm,epoch,'evaluation',len(done),flush=True)
                if time.monotonic()>=deadline:
                    job=os.environ.get('JOB_ID')
                    submit_once(f'eval-{arm}-{epoch}-resume-{job}','evaluate',arm,epoch,hold=job); return
        if set(done)!={r['id'] for r in tests}: raise RuntimeError('Evaluation IDs differ')
        excluded=set(json.loads((ROOT/'test_exact_overlap.json').read_text())['ids'])
        write_json(destination/'summary.json',{'success':True,'arm':arm,'epoch':epoch,
            'checkpoint_sha256':sha(checkpoint),'test':aggregate(list(done.values())),
            'without_exact_training_overlap':aggregate([r for r in done.values() if r['id'] not in excluded]),
            'chemical_metric':'MW-oriented AUROC and Spearman from one predicted structure; DMS A/C stratified; no BPP or reactivity threshold; undefined values remain null',
            'chemical_strata_in_predictions':True,'cross_model_comparison_requires_common_valid_strata':True,
            'predictions_sha256':sha(path)})


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('action',choices=['prepare','submit','preflight','work','evaluate'])
    parser.add_argument('--arm',choices=ARMS+('pretrain',),default=os.environ.get('FT_ARM',ARMS[0]))
    parser.add_argument('--epoch',type=int,default=int(os.environ.get('FT_EPOCH','0')))
    args=parser.parse_args()
    if args.action=='prepare': prepare()
    elif args.action=='submit':
        ROOT.mkdir(parents=True,exist_ok=True); (ROOT/'scheduler').mkdir(exist_ok=True)
        frozen=Path(__file__).resolve()==(ROOT/'controller.py').resolve()
        job_source=ROOT/'job.sh' if frozen else PROJECT/'scripts/efold_hf/condition3_ribonanza_ft_job.sh'
        for source,name in [(Path(__file__),'controller.py'),(job_source,'job.sh')]:
            target=ROOT/name
            if not target.exists(): shutil.copyfile(source,target)
            elif sha(source)!=sha(target): raise RuntimeError('Frozen bootstrap changed: '+name)
        prep=submit_once('prepare','prepare'); jobs={}
        baseline=submit_once('evaluate-pretrain','evaluate','pretrain',PRETRAIN_EPOCH,hold=prep)
        for arm in ARMS:
            pf=submit_once('preflight-'+arm,'preflight',arm,hold=prep)
            work=submit_once('work-initial-'+arm,'work',arm,hold=pf)
            jobs[arm]={'preflight':pf,'work':work}
        write_json(ROOT/'initial_jobs.json',{'prepare':prep,'baseline_evaluation':baseline,'methods':jobs})
    elif args.action=='preflight': preflight(args.arm)
    elif args.action=='work': train(args.arm)
    else: evaluate(args.arm,args.epoch)


if __name__=='__main__': main()
