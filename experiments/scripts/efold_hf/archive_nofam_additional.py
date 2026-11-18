#!/usr/bin/env python3
"""Add the three already validated public bpRNAnew baselines to Archive-NoFam."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

PROJECT = Path('/gs/bs/tga-satolab-gtex/satoken/repos/mxfold2-ryntaro')
BASE = PROJECT/'qsub_logs/efold_hf'
PRIOR = BASE/'archive_nofam_20261031'
BPRNA = BASE/'bprnanew_20261029'
ROOT = Path(os.environ.get('NOFAM_EXTRA_ROOT', str(BASE/'archive_nofam_additional_20261105')))
MODELS = ('RNAformer_bprna', 'CNNFold_600', 'SPOT_RNA_public')
QSUB = '/apps/t4/rhel9/uge/latest/bin/lx-amd64/qsub'


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).open() if line.strip()]


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


def write_rows(path, values):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('w') as handle:
        for value in values:
            handle.write(json.dumps(value, allow_nan=False)+'\n')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextmanager
def lock(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def spot_basename(root):
    return 'archive_nofam_extra_'+hashlib.sha256(str(root).encode()).hexdigest()[:16]+'.fasta'


def setup():
    if (ROOT/'implementation.json').exists():
        return verify()
    if ROOT.exists():
        raise RuntimeError('Inspect incomplete additional-model setup before reuse')
    prior, bpplan = read(PRIOR/'plan.json'), read(BPRNA/'plan.json')
    if sha(PRIOR/'panel.jsonl') != prior['panel_sha256']:
        raise RuntimeError('Archive-NoFam reference panel changed')
    if not read(PRIOR/'published_validation.json')['success']:
        raise RuntimeError('Published NoFam cohort definitions are not validated')
    if sha(PRIOR/'cohorts.json') != read(PRIOR/'cohorts-complete.json')['cohorts_sha256']:
        raise RuntimeError('Archive-NoFam cohort definitions changed')
    if sha(BPRNA/'panel.jsonl') != bpplan['panel_sha256']:
        raise RuntimeError('bpRNAnew control source changed')
    ROOT.mkdir()
    for name in ('adapters', 'controls', 'scheduler', 'dispatch', 'validation', 'predictions', 'native'):
        (ROOT/name).mkdir()
    shutil.copyfile(__file__, ROOT/'controller.py')
    shutil.copyfile(PROJECT/'scripts/efold_hf/archive_nofam_additional_job.sh', ROOT/'job.sh')
    for name in ('panel.jsonl', 'cohorts.json'):
        shutil.copyfile(PRIOR/name, ROOT/name)
    adapter = PROJECT/'scripts/efold_hf/bprnanew_external.py'
    if sha(adapter) != prior['adapters_sha256'][str(adapter)]:
        raise RuntimeError('Previously validated neural adapter changed')
    shutil.copyfile(adapter, ROOT/'adapters/neural.py')
    score_root = BASE/'all_conditions_external_eval_20261104'
    expected = read(score_root/'implementation.json')[str(score_root/'score_adapter.py')]
    if sha(score_root/'score_adapter.py') != expected:
        raise RuntimeError('Previously validated scoring adapter changed')
    shutil.copyfile(score_root/'score_adapter.py', ROOT/'adapters/score.py')
    bpitems = {r['id']:r for r in rows(BPRNA/'panel.jsonl')}
    registry = {m['model']:m for m in read(BPRNA/'external_models.json')}
    specifications = []
    for name in MODELS:
        spec = registry[name]
        for key in ('checkpoint', 'config'):
            if key in spec and sha(spec[key]) != spec[key+'_sha256']:
                raise RuntimeError('Public model changed: '+name+'/'+key)
        smoke = BPRNA/'smoke'/name
        marker = read(smoke/'shard-0-complete.json')
        if not marker['success'] or sha(smoke/'shard-0.jsonl') != marker['predictions_sha256']:
            raise RuntimeError('Previously validated model controls changed: '+name)
        controls = [{**bpitems[r['id']], 'expected_pairs':r['predicted_pairs']} for r in rows(smoke/'shard-0.jsonl')
                    if not set(bpitems[r['id']]['sequence'])-set('ACGU')]
        if len(controls) < 2:
            raise RuntimeError('Insufficient saved controls: '+name)
        write_rows(ROOT/'controls'/(name+'.jsonl'), controls)
        source = Path(spec['source'])
        source_files = [p for p in source.rglob('*.py') if '.git' not in p.parts]
        if spec['backend'] == 'spotrna':
            source_files += [p for p in (source/'SPOT-RNA-models').iterdir() if p.is_file()]
        specifications.append({**spec, 'source_sha256':{str(p):sha(p) for p in source_files},
                               'controls':str(ROOT/'controls'/(name+'.jsonl'))})
    panel, cohorts = rows(ROOT/'panel.jsonl'), read(ROOT/'cohorts.json')
    if len(panel) != 89 or len({r['id'] for r in panel}) != 89:
        raise RuntimeError('Wrong Archive-NoFam coverage')
    if {k:(len(v['ids']),len(v['groups'])) for k,v in cohorts['cohorts'].items()} != {
            'full89':(89,47), 'paper_bprna_filtered':(79,40), 'literal_caption_union':(44,22)}:
        raise RuntimeError('Wrong published cohorts')
    plan = {'dataset':'Archive-NoFam', 'n':89, 'models':specifications,
            'original_seven_model_root':str(PRIOR), 'reference_policy':'Original CT contacts, exact pairs, no pruning or tolerance',
            'cohorts':'Common79/40 groups; SPOT-paper exclusions44/22 groups; full89/47 groups',
            'aggregation':'Equal RNA mean inside structural group, then equal group mean',
            'protocol':'Reuse the public models and native postprocessing already used on bpRNAnew',
            'hard_wall_seconds':85800, 'SPOT_subprocess_timeout_seconds':79200,
            'RNAformer':'bpRNA checkpoint; six recycles; bf16; sigmoid>0.5; eval mode',
            'CNNFold':'CNNFold-600; native create_matrix/binarize; repeat2; retain serialized BN mode',
            'SPOT_RNA':'Five-model ensemble; official postprocessing; TensorFlow compat.v1 CPU4',
            'provided_SPOT_predictions':'Author-supplied reference predictions will be compared separately'}
    write(ROOT/'plan.json', plan)
    immutable = [ROOT/name for name in ('controller.py','job.sh','plan.json','panel.jsonl','cohorts.json')]
    immutable += list((ROOT/'adapters').iterdir())+list((ROOT/'controls').iterdir())
    # The frozen neural adapter imports this parser module from its original location.
    immutable.append(PROJECT/'scripts/efold_hf/bprnanew_benchmark.py')
    write(ROOT/'implementation.json', {str(p):sha(p) for p in immutable})
    return verify()


def verify():
    for path, digest in read(ROOT/'implementation.json').items():
        if sha(path) != digest:
            raise RuntimeError('Frozen additional-evaluation input changed: '+path)
    plan = read(ROOT/'plan.json')
    for spec in plan['models']:
        for key in ('checkpoint','config'):
            if key in spec and sha(spec[key]) != spec[key+'_sha256']:
                raise RuntimeError('Model/configuration changed: '+spec['model'])
        for path,digest in spec['source_sha256'].items():
            if sha(path) != digest:
                raise RuntimeError('Public model source/weights changed: '+path)
    return plan


def submit(action, model='', hold=None):
    receipt = ROOT/'dispatch'/(action+'-'+(model or 'all')+'.json')
    if receipt.exists():
        return read(receipt)['job_id']
    resource = 'gpu_h=1' if action=='work' and model!='SPOT_RNA_public' else 'cpu_4=1'
    duration = '23:50:00' if action=='work' else '01:00:00'
    args = [QSUB,'-terse','-cwd','-j','y','-A','tga-satolab-gtex','-r','y',
            '-N','an_extra_'+('spot' if model=='SPOT_RNA_public' else 'rnaformer' if model=='RNAformer_bprna' else 'cnnfold' if model else action),
            '-l',resource,'-l','h_rt='+duration,'-l','hostname=!r22n6','-o',str(ROOT/'scheduler'),
            '-v',f'NOFAM_EXTRA_ROOT={ROOT},NOFAM_EXTRA_ACTION={action},NOFAM_EXTRA_MODEL={model}']
    if hold:
        args += ['-hold_jid',str(hold)]
    args.append(str(ROOT/'job.sh'))
    result = subprocess.run(['sg','tga-satolab-gtex','-c',shlex.join(args)],cwd=PROJECT,
                            capture_output=True,text=True,check=True)
    job = result.stdout.strip().split('.')[0]
    if not job.isdigit():
        raise RuntimeError('Unexpected qsub result: '+result.stdout)
    write(receipt, {'job_id':job,'action':action,'model':model,'hold':hold,'command':args})
    print('submitted',action,model,job,flush=True)
    return job


def checked_prediction(item, pairs, scoring):
    pairs = [list(p) for p in scoring.checked_pairs(pairs,len(item['sequence']))]
    return {'id':item['id'],'sequence_sha256':item['sequence_sha256'],'length':len(item['sequence']),
            'predicted_pairs':pairs, **scoring.pair_metrics(item['pairs'],pairs)}


def check_controls(controls, predict):
    for item in controls:
        if sorted(map(tuple,predict(item))) != sorted(map(tuple,item['expected_pairs'])):
            raise RuntimeError('Prediction differs from saved bpRNAnew control: '+item['id'])


def spot_predict(spec, items):
    sys.path.insert(0,str(PROJECT/'scripts/efold_hf'))
    from bprnanew_benchmark import bpseq
    folder = ROOT/'native'/spec['model']
    folder.mkdir(parents=True,exist_ok=True)
    fasta = folder/spot_basename(ROOT)
    fasta.write_text(''.join('>'+r['id']+'\n'+r['sequence']+'\n' for r in items))
    outputs = folder/'outputs';outputs.mkdir(exist_ok=True)
    source = Path(spec['source'])
    code = ('import tensorflow as tf; tf.compat.v1.disable_eager_execution(); import runpy; '
            'runpy.run_path('+repr(str(source/'SPOT-RNA.py'))+",run_name='__main__')")
    environment = {**os.environ,'PYTHONPATH':str(PROJECT/'.cache/bprna-spot-tensorflow')+os.pathsep+
                   str(PROJECT/'.cache/bprna-benchmark-deps'),'TF_CPP_MIN_LOG_LEVEL':'2','CUDA_VISIBLE_DEVICES':'-1'}
    command = [str(PROJECT/'.venv/bin/python'),'-c',code,'--inputs',str(fasta),
               '--outputs',str(outputs)+'/', '--cpu','4','--gpu','-1']
    with (folder/'predict.log').open('w') as log:
        subprocess.run(command,cwd=source,env=environment,stdout=log,stderr=subprocess.STDOUT,
                       check=True,timeout=79200)
    predictions = {}
    for item in items:
        value = bpseq((outputs/(item['id']+'.bpseq')).read_text(), item['id'])
        if value['sequence'] != item['sequence']:
            raise RuntimeError('SPOT-RNA output sequence changed: '+item['id'])
        predictions[item['id']] = value['pairs']
    return predictions


def work(name):
    plan = verify()
    spec = next(m for m in plan['models'] if m['model']==name)
    folder = ROOT/'predictions'/name
    with lock(folder/'worker.lock'):
        if (folder/'complete.json').exists():
            return
        scoring = load('additional_nofam_scoring', ROOT/'adapters/score.py')
        items = sorted(rows(ROOT/'panel.jsonl'), key=lambda r:(-len(r['sequence']),r['id']))
        controls = rows(spec['controls'])
        started = time.monotonic()
        if spec['backend']=='spotrna':
            if {r['id'] for r in controls} & {r['id'] for r in items}:
                raise RuntimeError('Control/test identifier collision')
            predictions = spot_predict(spec,controls+items)
            predict = lambda item:predictions[item['id']]
        else:
            sys.path[:0] = [str(PROJECT/'scripts/efold_hf'),str(PROJECT/'.cache/bprna-benchmark-deps')]
            import torch
            torch.manual_seed(0)
            adapter = load('additional_nofam_native_neural_adapter', ROOT/'adapters/neural.py')
            adapter.ROOT = PROJECT
            native = ROOT/'native'/name;native.mkdir(parents=True,exist_ok=True)
            predict = adapter.predictor(spec,native)
        check_controls(controls,predict)
        write(ROOT/'validation'/(name+'.json'), {'success':True,'checkpoint_sha256':spec['checkpoint_sha256'],
              'saved_bpRNAnew_controls_reproduced':[r['id'] for r in controls]})
        # Full coverage is small (89 targets, maximum339nt); save every completed structure.
        path = folder/'predictions.jsonl'
        cached = rows(path) if path.exists() else []
        if len({r['id'] for r in cached}) != len(cached):
            raise RuntimeError('Duplicate cached prediction')
        reference = {r['id']:r for r in items}
        for row in cached:
            if row['id'] not in reference or row != checked_prediction(reference[row['id']],row['predicted_pairs'],scoring):
                raise RuntimeError('Cached sequence or F1 changed')
        done = {r['id'] for r in cached}
        with path.open('a') as handle:
            for item in items:
                if item['id'] in done:
                    continue
                value = checked_prediction(item,predict(item),scoring)
                handle.write(json.dumps(value,allow_nan=False)+'\n');handle.flush();done.add(item['id'])
                if len(done)%10==0:
                    print(name,len(done),'/',len(items),flush=True)
        if done != set(reference):
            raise RuntimeError('Incomplete target coverage')
        write(folder/'complete.json', {'success':True,'n':len(done),'checkpoint_sha256':spec['checkpoint_sha256'],
              'predictions_sha256':sha(path),'seconds':time.monotonic()-started})
        print('Completed',name,len(done),flush=True)


def report():
    plan = verify()
    scoring = load('additional_nofam_report_scoring', ROOT/'adapters/score.py')
    panel = {r['id']:r for r in rows(ROOT/'panel.jsonl')}
    cohorts = read(ROOT/'cohorts.json')
    prior_plan = read(PRIOR/'plan.json')
    legacy = prior_plan['models']+prior_plan['external_models']
    specifications = legacy+plan['models']
    results, all_rows, independent = [], [], 0
    for spec in specifications:
        name = spec['model']
        if name in MODELS:
            folder = ROOT/'predictions'/name
            marker, path = read(folder/'complete.json'),folder/'predictions.jsonl'
            if not read(ROOT/'validation'/(name+'.json'))['success']:
                raise RuntimeError('Model control validation missing')
        else:
            folder = PRIOR/'predictions'/name
            marker, path = read(folder/'shard-0-complete.json'),folder/'shard-0.jsonl'
        if not marker['success'] or marker['n']!=89 or sha(path)!=marker['predictions_sha256']:
            raise RuntimeError('Changed/incomplete result: '+name)
        saved = rows(path)
        if len(saved)!=89 or {r['id'] for r in saved} != set(panel):
            raise RuntimeError('Changed target identifiers')
        verified = []
        for row in saved:
            actual = checked_prediction(panel[row['id']],row['predicted_pairs'],scoring)
            if any(actual[k]!=row[k] for k in actual if k!='predicted_pairs'):
                raise RuntimeError('Independent F1/sequence mismatch: '+name+'/'+row['id'])
            verified.append(actual);independent+=1
            all_rows.append({'model':name,'id':row['id'],'group':cohorts['group_map'][row['id']],
                             **{k:actual[k] for k in ('tp','fp','fn','precision','recall','f1')}})
        for key, subset in cohorts['cohorts'].items():
            selected = [r for r in verified if r['id'] in set(subset['ids'])]
            results.append({'model':name,'label':spec['label'],'cohort':key,
                            **scoring.aggregate(selected,cohorts['group_map'])})
    supplied = rows(PRIOR/'provided_predictions/spot-rna.jsonl')
    supplied_rows = [{'id':r['id'],**scoring.pair_metrics(panel[r['id']]['pairs'],r['pairs'])} for r in supplied]
    author = {key:scoring.aggregate([r for r in supplied_rows if r['id'] in set(subset['ids'])],cohorts['group_map'])
              for key,subset in cohorts['cohorts'].items()}
    native = {r['id']:r['predicted_pairs'] for r in rows(ROOT/'predictions/SPOT_RNA_public/predictions.jsonl')}
    agreement = sum(set(map(tuple,r['pairs']))==set(map(tuple,native[r['id']])) for r in supplied)
    output = {'success':True,'results':results,'independent_per_sequence_F1_checks':independent,
              'SPOT_author_supplied':author,'SPOT_native_author_exact_structure_matches':agreement,
              'reference':'Original frozen Archive-NoFam CT contacts, exact pairs',
              'cohorts_sha256':sha(ROOT/'cohorts.json'),'panel_sha256':sha(ROOT/'panel.jsonl')}
    write(ROOT/'results.json',output)
    with (ROOT/'per_rna.tsv').open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(all_rows[0]),delimiter='\t');writer.writeheader();writer.writerows(all_rows)
    flat=[{'model':r['model'],'cohort':r['cohort'],'n_rnas':r['n_rnas'],'n_groups':r['n_groups'],
           'group_F1':r['group_macro']['f1'],'RNA_F1':r['rna_macro']['f1'],'precision':r['group_macro']['precision'],
           'recall':r['group_macro']['recall']} for r in results]
    with (ROOT/'results.tsv').open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(flat[0]),delimiter='\t');writer.writeheader();writer.writerows(flat)
    text=['# Archive-NoFam：RNAformer・CNNFold・SPOT-RNAを追加','',
          '従来の7モデルに、bpRNAnewと同じ公開重み・公式後処理の3モデルを追加して実行した。',
          '全89配列・47構造群を予測。群内のRNA平均を取ってから群間を等重み平均する。参照構造の剪定や位置の許容誤差は加えない。','',
          '|手法|共通79配列・40群 F1|SPOT論文除外44配列・22群 F1|全89配列・47群 F1|',
          '|---|---:|---:|---:|']
    for spec in specifications:
        values={r['cohort']:r for r in results if r['model']==spec['model']}
        text.append('|'+spec['label']+'|'+'|'.join(f"{values[c]['group_macro']['f1']:.4f}" for c in
            ['paper_bprna_filtered','literal_caption_union','full89'])+'|')
    text+=['','SPOT-RNAの原論文比較用の値は44配列・22群。79配列版は他手法と同一対象の比較。追加3モデルは今回再推論し、既存7モデルは検証済みの保存予測を再集計した。',
           f"参考：著者提供SPOT-RNA予測の44配列F1は{author['literal_caption_union']['group_macro']['f1']:.4f}。今回の予測構造と著者提供構造の完全一致は{agreement}/89。",'',
           'CNNFoldは公式スクリプトと同じ公開pickleのモードを維持。RNAformerはbpRNAモデル・6 recycling・bf16・閾値0.5。SPOT-RNAは公開5モデルensembleと公式後処理。',
           '旧MXfold2 MW Rankは従来のFT5epモデルであり、条件1〜5のFT6epモデルは別の外部評価表を参照。',
           f'全10モデルの配列別F1 {independent}件を参照塩基対と予測構造から再計算して一致。保存済みbpRNAnew対照構造の再現も各モデルで検証。','']
    (ROOT/'README_ja.md').write_text('\n'.join(text))
    write(ROOT/'complete.json', {'success':True,'models':len(specifications),'new_models':list(MODELS),
                               'results_sha256':sha(ROOT/'results.json'),'independent_F1_checks':independent})
    print('\n'.join(text),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['setup','submit','work','report'])
    parser.add_argument('--model',choices=MODELS,default=os.environ.get('NOFAM_EXTRA_MODEL') or MODELS[0])
    args=parser.parse_args()
    if args.action=='setup':
        setup()
    elif args.action=='submit':
        setup()
        jobs={name:submit('work',name) for name in MODELS}
        jobs['report']=submit('report',hold=','.join(jobs.values()))
        write(ROOT/'jobs.json',jobs)
    elif args.action=='work':
        work(args.model)
    else:
        report()


if __name__=='__main__':
    main()
