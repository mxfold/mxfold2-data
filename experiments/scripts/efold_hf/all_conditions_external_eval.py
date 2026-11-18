#!/usr/bin/env python3
"""Evaluate all registered PT/FT checkpoints on the frozen external panels."""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timedelta
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

PROJECT = Path('/gs/bs/tga-satolab-gtex/satoken/repos/mxfold2-ryntaro')
sys.path[:0] = [str(PROJECT), str(PROJECT/'.cache/bprna-benchmark-deps')]
BASE = PROJECT/'qsub_logs/efold_hf'
OUT = Path(os.environ.get('ALL_EVAL_ROOT', str(BASE/'all_conditions_external_eval_20261104')))
REFERENCE = BASE/'condition2_pt005_external_eval_20261104_cpu'
ARMS = ('pretrain', 'mw_rank_restored', 'nll_restored', 'shape_fy_restored')
LABELS = ('事前学習後', 'MW Rank', 'IMLE-NLL', 'SHAPE-fy')
QSUB = '/apps/t4/rhel9/uge/latest/bin/lx-amd64/qsub'
JST = ZoneInfo('Asia/Tokyo')
HARD_SECONDS, SOFT_SECONDS = 85800, 79200
PRETRAINS = {
    1: 'pretrain_variants_20261031/bprna20',
    2: 'pretrain_variants_20261031/bprna_rnacentral10',
    3: 'pretrain_variants_20261031/bprna_trainab_nr80_20',
    4: 'pretrain_combined_nr80_20_20261031/combined_nr80_20',
    5: 'pretrain_condition5_nr80_20_20261103/bprna_rnacentral_nr80_20',
}


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def now():
    return datetime.now(JST).isoformat(timespec='seconds')


@contextmanager
def lock(path, nonblocking=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0))
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def discover_cases():
    found = {}
    for path in sorted(BASE.glob('ft*/cases.json')):
        for case in read(path).get('cases', []):
            key = case['id']
            if key in found:
                for field in ('condition', 'pretrain_epoch', 'stage1', 'stage2'):
                    if found[key][field] != case[field]:
                        raise RuntimeError('Conflicting case registration: '+key)
            found[key] = case
    for condition in (1, 3):
        key = f'c{condition}p20'
        found.setdefault(key, {'id':key, 'condition':condition, 'pretrain_epoch':20,
            'pretrain_root':str(BASE/PRETRAINS[condition]),
            'stage1':str(BASE/f'condition{condition}_ribonanza3_replay20_20261101'),
            'stage2':str(BASE/f'condition{condition}_pri_mrna3_after_ribonanza3_replay20_20261101')})
    selected = sorted(found.values(), key=lambda c:(c['condition'], c['pretrain_epoch']))
    expected = {(1,10),(1,20),(2,5),(2,10),(3,10),(3,20),(4,10),(4,20),(5,10),(5,20)}
    if {(c['condition'],c['pretrain_epoch']) for c in selected} != expected:
        raise RuntimeError('Unexpected registered condition/origin set')
    return selected


def setup():
    if (OUT/'implementation.json').exists():
        return verify()
    if OUT.exists():
        raise RuntimeError('Inspect incomplete external evaluation setup before reuse')
    OUT.mkdir()
    for name in ('scheduler','dispatch','models','predictions','validation','panels','reports'):
        (OUT/name).mkdir()
    # Reuse the already verified CPU adapter without changing model mathematics.
    for path, digest in read(REFERENCE/'implementation.json').items():
        if sha(path) != digest:
            raise RuntimeError('Validated CPU evaluation source changed: '+path)
    shutil.copyfile(__file__, OUT/'controller.py')
    for name in ('mx_adapter.py','score_adapter.py','cohorts.json'):
        shutil.copyfile(REFERENCE/name, OUT/name)
    panels = {}
    for dataset, parent, expected in [('Archive-NoFam',BASE/'archive_nofam_20261031',89),
                                      ('bpRNAnew',BASE/'bprnanew_20261029',5401)]:
        prior = read(parent/'plan.json')
        path = parent/'panel.jsonl'
        if sha(path) != prior['panel_sha256']:
            raise RuntimeError('External panel changed: '+dataset)
        values = rows(path)
        if len(values) != expected or len({r['id'] for r in values}) != expected:
            raise RuntimeError('Wrong external panel IDs/count')
        if max(len(r['sequence']) for r in values) > 1000:
            raise RuntimeError('Unexpected external benchmark length')
        target = OUT/'panels'/(dataset+'.jsonl')
        shutil.copyfile(path,target)
        panels[dataset] = {'path':str(target),'sha256':sha(target),'n':expected,'prior_root':str(parent)}
    prior = BASE/'archive_nofam_20261031'
    if not read(prior/'published_validation.json')['success']:
        raise RuntimeError('Published NoFam aggregation has not been validated')
    if read(prior/'cohorts-complete.json')['cohorts_sha256'] != sha(OUT/'cohorts.json'):
        raise RuntimeError('Published NoFam cohorts changed')
    cases = discover_cases()
    training = []
    for condition, relative in PRETRAINS.items():
        parent = BASE/relative
        plan = read(parent/'plan.json')
        training.append({'name':f'condition{condition}','path':str(parent/'data/train.jsonl'),
                         'sha256':plan['data_sha256']['train.jsonl']})
    chemical = BASE/'bprna10_restored_ft5_replay20_20261026'
    training.append({'name':'chemical_FT','path':str(chemical/'data/chemical.jsonl'),
                     'sha256':read(chemical/'plan.json')['data_sha256']['chemical.jsonl']})
    plan = {'created_at_jst':now(),'cases':cases,'panels':panels,'training_files':training,
        'methods':list(ARMS),'FT_epoch':6,'checkpoint_policy':'Exact requested PT epoch; final stage2 epoch3',
        'reference_policy':'Original contacts, exact base-pair matching, no pruning or tolerance',
        'Archive-NoFam_primary':'Fixed published bpRNA exclusions:79 RNAs/40 groups; equal group mean',
        'bpRNAnew_primary':'All5401 RNAs; equal RNA mean; also ACGU-only5388',
        'execution_device':'cpu','cpu_threads':4,'no_BPP':True,
        'hard_wall_seconds':HARD_SECONDS,'soft_wall_seconds':SOFT_SECONDS,'poll_minutes':15,
        'validated_adapter_reference':str(REFERENCE)}
    write(OUT/'plan.json',plan)
    job = f'''#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 TZ=Asia/Tokyo
cd {shlex.quote(str(PROJECT))}
exec .venv/bin/python "$ALL_EVAL_ROOT/controller.py" "$ALL_EVAL_ACTION" --model "${{ALL_EVAL_MODEL:-}}"
'''
    (OUT/'job.sh').write_text(job)
    immutable = [OUT/name for name in ('controller.py','job.sh','plan.json','mx_adapter.py','score_adapter.py','cohorts.json')]
    immutable += list((OUT/'panels').iterdir())
    write(OUT/'implementation.json',{str(p):sha(p) for p in immutable})
    return verify()


def verify():
    for path, digest in read(OUT/'implementation.json').items():
        if sha(path) != digest:
            raise RuntimeError('Frozen evaluation input changed: '+path)
    return read(OUT/'plan.json')


def model_ids(plan):
    return [(case,arm,case['id']+'_'+arm) for case in plan['cases'] for arm in ARMS]


def completed(path):
    return path.exists() and read(path).get('success',False)


def bind_model(case, arm):
    identifier = case['id']+'_'+arm
    folder = OUT/'models'/identifier
    if (folder/'binding.json').exists():
        return checked_spec(identifier)
    root = Path(case['stage1'] if arm=='pretrain' else case['stage2'])
    epoch = case['pretrain_epoch'] if arm=='pretrain' else 3
    checkpoint = root/f'runs/{arm}/epoch-{epoch:03d}.pt'
    marker = checkpoint.with_name(f'epoch-{epoch:03d}-complete.json')
    destination = root/f'evaluation/{arm}/epoch-{epoch:03d}'
    if not completed(marker) or not completed(destination/'summary.json'):
        return None
    if arm!='pretrain' and not completed(root/f'runs/{arm}/complete.json'):
        return None
    summary, metadata = read(destination/'summary.json'), read(marker)
    digest = sha(checkpoint)
    if digest != metadata['checkpoint_sha256'] or digest != summary['checkpoint_sha256']:
        raise RuntimeError('Checkpoint binding checksum mismatch: '+identifier)
    if arm!='pretrain' and summary.get('global_ft_epoch') != 6:
        raise RuntimeError('Expected the final cumulative FT epoch6')
    predictions = destination/'per_sequence.jsonl'
    if sha(predictions) != summary['predictions_sha256']:
        raise RuntimeError('Control prediction checksum mismatch')
    training_plan = read(root/'plan.json')
    parent_plan = read(Path(case['pretrain_root'])/'plan.json')
    source_hashes = {str(root/'source'/relative):value for relative,value in training_plan['source_sha256'].items()
                     if relative.startswith('mxfold2/')}
    for path, expected in source_hashes.items():
        if sha(path) != expected:
            raise RuntimeError('Frozen model source changed: '+path)
    panel = {r['id']:r for r in rows(root/'data/test.jsonl') if r['source']=='viral_fragments'}
    control_ids = sorted(panel,key=lambda k:(len(panel[k]['sequence']),k))
    control_ids = [control_ids[0],control_ids[len(control_ids)//2],control_ids[-1]]
    saved = {r['id']:r for r in rows(predictions) if r['id'] in control_ids}
    folder.mkdir(parents=True,exist_ok=True)
    target, config = folder/'checkpoint.pt',folder/'baseline.conf'
    shutil.copyfile(checkpoint,target)
    shutil.copyfile(root/'baseline.conf',config)
    expected_step = parent_plan['n']*case['pretrain_epoch'] + (0 if arm=='pretrain' else 45192)
    spec = {'model':identifier,'case':case['id'],'condition':case['condition'],
        'pretrain_epoch':case['pretrain_epoch'],'arm':arm,'label':LABELS[ARMS.index(arm)],
        'checkpoint':str(target),'checkpoint_sha256':digest,'original_checkpoint':str(checkpoint),
        'config':str(config),'config_sha256':sha(config),'source':str(root/'source'),
        'source_hashes':source_hashes,'expected_optimizer_step':expected_step,
        'controls':[{**panel[k],'expected_pairs':saved[k]['predicted_pairs']} for k in control_ids],
        'bound_at_jst':now()}
    write(folder/'spec.json',spec)
    write(folder/'binding.json',{'spec_sha256':sha(folder/'spec.json'),'checkpoint_sha256':digest,
                               'config_sha256':spec['config_sha256']})
    return spec


def checked_spec(identifier):
    folder = OUT/'models'/identifier
    spec, binding = read(folder/'spec.json'),read(folder/'binding.json')
    if sha(folder/'spec.json') != binding['spec_sha256']:
        raise RuntimeError('Bound model specification changed')
    if sha(spec['checkpoint']) != binding['checkpoint_sha256'] or sha(spec['config']) != binding['config_sha256']:
        raise RuntimeError('Bound model/configuration changed')
    for path, expected in spec['source_hashes'].items():
        if sha(path) != expected:
            raise RuntimeError('Frozen inference source changed: '+path)
    return spec


def submit(key, action, model='', hold=None, start_at=None):
    receipt = OUT/'dispatch'/(key+'.json')
    with lock(OUT/'submission.lock'):
        if receipt.exists():
            return read(receipt)['job_id']
        duration = {'tick':'00:10:00','work':'23:50:00','overlap':'02:00:00','report':'00:30:00'}[action]
        args = [QSUB,'-terse','-cwd','-j','y','-A','tga-satolab-gtex','-r','y',
            '-N','allbench_'+action,'-l','cpu_4=1','-l','h_rt='+duration,
            '-l','hostname=!r22n6','-o',str(OUT/'scheduler'),
            '-v',f'ALL_EVAL_ROOT={OUT},ALL_EVAL_ACTION={action},ALL_EVAL_MODEL={model}']
        if hold:
            args += ['-hold_jid',str(hold)]
        if start_at:
            args += ['-a',start_at]
        args.append(str(OUT/'job.sh'))
        result = subprocess.run(['sg','tga-satolab-gtex','-c',shlex.join(args)],cwd=PROJECT,
            env={**os.environ,'TZ':'Asia/Tokyo'},capture_output=True,text=True,check=True)
        job = result.stdout.strip().split('.')[0]
        if not job.isdigit():
            raise RuntimeError('Unexpected qsub output: '+result.stdout)
        write(receipt,{'job_id':job,'action':action,'model':model,'hold_jid':hold,
                       'start_at_jst':start_at,'submitted_at_jst':now(),'command':args})
        print('submitted',action,model,job,flush=True)
        return job


def checked_prediction(item, pairs, scoring):
    pairs = [list(p) for p in scoring.checked_pairs(pairs,len(item['sequence']))]
    return {'id':item['id'],'sequence_sha256':item['sequence_sha256'],'length':len(item['sequence']),
            'predicted_pairs':pairs,**scoring.pair_metrics(item['pairs'],pairs)}


def cached_predictions(path, items, scoring):
    reference = {r['id']:r for r in items}
    saved = rows(path) if path.exists() else []
    if len({r['id'] for r in saved}) != len(saved) or not {r['id'] for r in saved} <= set(reference):
        raise RuntimeError('Invalid cached target coverage')
    for row in saved:
        if row != checked_prediction(reference[row['id']],row['predicted_pairs'],scoring):
            raise RuntimeError('Cached sequence/prediction metric mismatch')
    return saved


def model_complete(plan, identifier):
    return all(completed(OUT/'predictions'/dataset/identifier/'complete.json') for dataset in plan['panels'])


def reuse_condition2(spec, plan):
    if spec['case'] != 'c2p5':
        return False
    previous = read(REFERENCE/'plan.json')
    model = next(m for m in previous['models'] if m['model']==spec['arm'])
    if model['checkpoint_sha256'] != spec['checkpoint_sha256'] or model['config_sha256'] != spec['config_sha256']:
        raise RuntimeError('Existing condition2 evaluation checkpoint differs')
    scoring = load('reuse_external_scoring',OUT/'score_adapter.py')
    for dataset, panel in plan['panels'].items():
        if previous['panels'][dataset]['sha256'] != panel['sha256']:
            raise RuntimeError('Existing condition2 evaluation panel differs')
        source = REFERENCE/'predictions'/dataset/spec['arm']
        metadata = read(source/'complete.json')
        if not metadata['success'] or metadata['checkpoint_sha256'] != spec['checkpoint_sha256']:
            raise RuntimeError('Invalid existing condition2 completion')
        if sha(source/'predictions.jsonl') != metadata['predictions_sha256']:
            raise RuntimeError('Existing condition2 predictions changed')
        cached = cached_predictions(source/'predictions.jsonl',rows(panel['path']),scoring)
        if len(cached) != panel['n']:
            raise RuntimeError('Existing condition2 evaluation incomplete')
        target = OUT/'predictions'/dataset/spec['model']
        target.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source/'predictions.jsonl',target/'predictions.jsonl')
        write(target/'complete.json',{**metadata,'model':spec['model'],'reused_from':str(source)})
    write(OUT/'validation'/(spec['model']+'.json'),{'success':True,'reused_from':str(REFERENCE),
        'prior_validation':read(REFERENCE/'validation'/(spec['arm']+'.json'))})
    return True


def work(identifier):
    import torch
    plan, spec = verify(),checked_spec(identifier)
    with lock(OUT/'models'/identifier/'worker.lock',nonblocking=True):
        if model_complete(plan,identifier):
            return
        torch.set_num_threads(4)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        adapter = load('frozen_all_external_mx_adapter',OUT/'mx_adapter.py')
        scoring = load('frozen_all_external_scoring',OUT/'score_adapter.py')
        model = adapter.create_model(spec['source'],spec['config'])
        payload = adapter.restore(model,spec['checkpoint'])
        if payload['optimizer_state_dict']['param_groups'][0]['step'] != spec['expected_optimizer_step']:
            raise RuntimeError('Restored optimizer step does not match selected PT/FT checkpoint')
        def predict(item):
            partners = model([item['sequence']],return_partfunc=False)[2][0]
            return [[i-1,int(j)-1] for i,j in enumerate(partners) if i>0 and j>i]
        started = time.monotonic()
        with torch.inference_mode():
            for item in spec['controls']:
                if sorted(map(tuple,predict(item))) != sorted(map(tuple,item['expected_pairs'])):
                    raise RuntimeError('CPU prediction differs from saved GPU control: '+item['id'])
            ambiguous = [r for r in rows(plan['panels']['bpRNAnew']['path']) if set(r['sequence'])-set('ACGU')]
            for item in ambiguous:
                checked_prediction(item,predict(item),scoring)
            write(OUT/'validation'/(identifier+'.json'),{'success':True,
                'saved_GPU_controls_reproduced':[r['id'] for r in spec['controls']],
                'ambiguous_inputs_tested':len(ambiguous),'checkpoint_sha256':spec['checkpoint_sha256'],
                'device':'cpu','no_BPP':True})
            for dataset, panel in plan['panels'].items():
                folder = OUT/'predictions'/dataset/identifier
                if completed(folder/'complete.json'):
                    continue
                folder.mkdir(parents=True,exist_ok=True)
                path = folder/'predictions.jsonl'
                items = sorted(rows(panel['path']),key=lambda r:(-len(r['sequence']),r['id']))
                saved = cached_predictions(path,items,scoring)
                done = {r['id'] for r in saved}
                with path.open('a') as handle:
                    for item in items:
                        if item['id'] in done:
                            continue
                        row = checked_prediction(item,predict(item),scoring)
                        handle.write(json.dumps(row,allow_nan=False)+'\n');handle.flush();done.add(item['id'])
                        if len(done)%250==0:
                            print(identifier,dataset,len(done),'/',len(items),flush=True)
                            write(OUT/'models'/identifier/'progress.json',{'dataset':dataset,'n':len(done),
                                'total':len(items),'updated_at_jst':now(),'job_id':os.environ.get('JOB_ID')})
                        if time.monotonic()-started >= SOFT_SECONDS:
                            job = os.environ.get('JOB_ID','local')
                            continuation = submit('work-'+identifier+'-after-'+job,'work',identifier,
                                                  hold=job if job.isdigit() else None)
                            write(OUT/'models'/identifier/'continuation.json',{'job_id':continuation,'after':job})
                            return
                if len(done) != len(items):
                    raise RuntimeError('Incomplete benchmark target coverage')
                write(folder/'complete.json',{'success':True,'model':identifier,'n':len(done),
                    'checkpoint_sha256':spec['checkpoint_sha256'],'predictions_sha256':sha(path),
                    'finished_at_jst':now()})
        write(OUT/'models'/identifier/'complete.json',{'success':True,'seconds':time.monotonic()-started,
                                                       'finished_at_jst':now()})


def overlap():
    plan = verify()
    indices = {}
    for dataset,panel in plan['panels'].items():
        index = defaultdict(list)
        for item in rows(panel['path']):
            index[item['sequence'].upper().replace('T','U')].append(item['id'])
        indices[dataset] = index
    union = {dataset:set() for dataset in indices}
    individual = {}
    for source in plan['training_files']:
        if sha(source['path']) != source['sha256']:
            raise RuntimeError('Frozen training data changed')
        found = {dataset:set() for dataset in indices}
        with Path(source['path']).open() as handle:
            for line in handle:
                sequence = json.loads(line)['sequence'].upper().replace('T','U')
                for dataset,index in indices.items():
                    found[dataset].update(index.get(sequence,[]))
        individual[source['name']] = {dataset:sorted(ids) for dataset,ids in found.items()}
        for dataset,ids in found.items():
            union[dataset].update(ids)
    write(OUT/'training_overlap.json',{'success':True,'by_training_source':individual,
        'union':{dataset:sorted(ids) for dataset,ids in union.items()},
        'policy':'Common exact-sequence exclusion across all five conditions and chemical FT; no additional homology filter'})


def rna_aggregate(values):
    tp,fp,fn = (sum(row[key] for row in values) for key in ('tp','fp','fn'))
    return {'n_rnas':len(values),'rna_macro':{
        key:statistics.mean(r[key] for r in values if r[key] is not None) if any(r[key] is not None for r in values) else None
        for key in ('precision','recall','f1')},
        'micro_f1':2*tp/(2*tp+fp+fn) if tp+fp+fn else 1.}


def report(tag):
    plan = verify()
    scoring = load('frozen_all_external_report_scoring',OUT/'score_adapter.py')
    cohorts,overlaps = read(OUT/'cohorts.json'),read(OUT/'training_overlap.json')
    specs = [checked_spec(identifier) for _,_,identifier in model_ids(plan) if model_complete(plan,identifier)]
    metrics = [];count = 0
    for dataset,panel in plan['panels'].items():
        reference = {r['id']:r for r in rows(panel['path'])}
        excluded = set(overlaps['union'][dataset])
        for spec in specs:
            folder = OUT/'predictions'/dataset/spec['model']
            metadata = read(folder/'complete.json');path = folder/'predictions.jsonl'
            if metadata['checkpoint_sha256'] != spec['checkpoint_sha256'] or sha(path) != metadata['predictions_sha256']:
                raise RuntimeError('Completed benchmark binding changed')
            saved = cached_predictions(path,list(reference.values()),scoring)
            if len(saved) != panel['n']:
                raise RuntimeError('Incomplete benchmark panel')
            count += len(saved)
            if dataset=='Archive-NoFam':
                selections = {key:set(value['ids']) for key,value in cohorts['cohorts'].items()}
                selections['common_no_exact_training_overlap'] = selections['paper_bprna_filtered']-excluded
            else:
                selections = {'all_5401':set(reference),'ACGU_only_5388':{
                    k for k,r in reference.items() if not set(r['sequence'])-set('ACGU')},
                    'common_no_exact_training_overlap':set(reference)-excluded}
            for cohort, identifiers in selections.items():
                selected = [r for r in saved if r['id'] in identifiers]
                value = scoring.aggregate(selected,cohorts['group_map']) if dataset=='Archive-NoFam' else rna_aggregate(selected)
                metrics.append({'dataset':dataset,'model':spec['model'],'case':spec['case'],'arm':spec['arm'],
                    'condition':spec['condition'],'pretrain_epoch':spec['pretrain_epoch'],'cohort':cohort,**value})
    destination = OUT/'reports'/tag
    destination.mkdir(parents=True,exist_ok=True)
    result = {'success':True,'generated_at_jst':now(),'models':len(specs),'prediction_rows_verified':count,
              'results':metrics,'no_BPP':True,'training_overlap':overlaps}
    write(destination/'results.json',result)
    text = ['# 条件1〜5：Archive-NoFam / bpRNAnew','',
        '事前学習の指定エポックと、Ribonanza3ep→pri-miRNA＋ヒトmRNA3epの最終FT6epを評価。',
        'Archive-NoFamは論文再現済み79配列・40群の等重み群平均。bpRNAnewは公式5401配列のRNA平均。',
        '両データには実測reactivityがないため、参照塩基対に対するF1を評価。元の参照構造を保持し、位置の許容誤差なし。','',
        '|条件・起点|データセット|事前学習後|MW Rank|IMLE-NLL|SHAPE-fy|','|---|---|---:|---:|---:|---:|']
    for case in plan['cases']:
        for dataset,cohort in [('Archive-NoFam','paper_bprna_filtered'),('bpRNAnew','all_5401')]:
            cells = []
            for arm in ARMS:
                found = [r for r in metrics if (r['case'],r['arm'],r['dataset'],r['cohort'])==(case['id'],arm,dataset,cohort)]
                value = found[0]['group_macro' if dataset=='Archive-NoFam' else 'rna_macro']['f1'] if found else None
                cells.append('—' if value is None else f'{value:.4f}')
            text.append(f"|条件{case['condition']}・PT{case['pretrain_epoch']}|{dataset}|"+'|'.join(cells)+'|')
    text += ['', '—はモデルの学習・既存テスト評価・本評価のいずれかが未完了。全89配列・47群、ACGUのみ5388配列、全条件共通の訓練完全一致除外後もresults.jsonに保存。',
        'Archive-NoFamは従来と同じ著者プレプリント補足版。群・除外対象は987個の補足表値と21個の平均値で検証した定義を継承。',
        'CPU推論は各チェックポイントの保存済みGPUテスト3構造との完全一致と、曖昧塩基を含む13配列を検証する。',
        f"検証済みモデル{len(specs)}、予測行{count}。共通完全一致除外数: "+str({k:len(v) for k,v in overlaps['union'].items()})+'.','']
    (destination/'README_ja.md').write_text('\n'.join(text))
    write(destination/'complete.json',{'success':True,'models':len(specs),'finished_at_jst':now(),
                                     'results_sha256':sha(destination/'results.json')})
    with lock(OUT/'report-pointer.lock'):
        latest = OUT/'latest_report.json'
        if not latest.exists() or read(latest)['models'] <= len(specs):
            write(latest,{'tag':tag,'root':str(destination),'models':len(specs),
                         'finished_at_jst':now()})
            shutil.copyfile(destination/'README_ja.md',OUT/'REPORT_ja.md')
    print('External comparison complete',len(specs),count,destination,flush=True)


def tick():
    plan = verify()
    with lock(OUT/'tick.lock'):
        states = [];finished = [];pending = []
        for case,arm,identifier in model_ids(plan):
            try:
                spec = bind_model(case,arm)
                if spec is None:
                    states.append({'model':identifier,'state':'waiting_for_checkpoint_or_test_evaluation'})
                    pending.append(identifier);continue
                if not model_complete(plan,identifier):
                    reuse_condition2(spec,plan)
                if model_complete(plan,identifier):
                    finished.append(identifier);states.append({'model':identifier,'state':'complete'});continue
                error = OUT/'models'/identifier/'error.json'
                if error.exists():
                    states.append({'model':identifier,'state':'error','error':read(error)});continue
                job = submit('work-'+identifier+'-initial','work',identifier,hold=os.environ.get('JOB_ID'))
                states.append({'model':identifier,'state':'submitted','job_id':job})
                pending.append(identifier)
            except Exception as exc:
                states.append({'model':identifier,'state':'error','error':str(exc)})
        report_job = None
        reporting = False
        if finished and completed(OUT/'training_overlap.json'):
            tag = 'models-'+hashlib.sha256(json.dumps(sorted(finished)).encode()).hexdigest()[:12]
            if not completed(OUT/'reports'/tag/'complete.json') and not (OUT/'reports'/tag/'error.json').exists():
                report_job = submit('report-'+tag,'report',tag,hold=os.environ.get('JOB_ID'))
                reporting = True
        continuation = None
        if pending or reporting:
            current = os.environ.get('JOB_ID','initial')
            start = (datetime.now(JST)+timedelta(minutes=plan['poll_minutes'])).strftime('%Y%m%d%H%M.%S')
            continuation = submit('tick-after-'+current,'tick',hold=current if current.isdigit() else None,start_at=start)
        write(OUT/'status.json',{'checked_at_jst':now(),'models':states,'complete_models':len(finished),
            'registered_models':len(model_ids(plan)),'pending_models':len(pending),
            'report_job':report_job,'next_gate_job':continuation})
        print('External evaluation registered',len(finished),'complete',len(pending),'pending',flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action',choices=['setup','submit','tick','work','overlap','report'])
    parser.add_argument('--model',default='')
    args = parser.parse_args()
    if args.action in ('setup','submit'):
        setup()
        if args.action=='submit':
            audit = submit('overlap-initial','overlap')
            gate = submit('tick-initial','tick')
            write(OUT/'initial_jobs.json',{'overlap':audit,'gate':gate})
    elif args.action=='tick':
        tick()
    elif args.action=='overlap':
        overlap()
    elif args.action=='report':
        try:
            report(args.model)
        except Exception as exc:
            write(OUT/'reports'/args.model/'error.json',{'error':str(exc),'failed_at_jst':now()})
            raise
    else:
        try:
            work(args.model)
        except Exception as exc:
            write(OUT/'models'/args.model/'error.json',{'error':str(exc),'failed_at_jst':now(),
                                                     'job_id':os.environ.get('JOB_ID')})
            raise


if __name__=='__main__':
    main()
