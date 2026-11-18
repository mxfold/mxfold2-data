#!/usr/bin/env python3
"""Evaluate fixed condition2 PT5/curriculum checkpoints on CPU compute nodes."""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
from datetime import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
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
OUT = BASE/'condition2_pt005_external_eval_20261104_cpu'
FIRST = BASE/'condition2_pt005_ribonanza3_replay20_20261101'
SECOND = BASE/'condition2_pt005_pri_mrna3_after_ribonanza3_replay20_20261101'
QSUB = '/apps/t4/rhel9/uge/latest/bin/lx-amd64/qsub'
ARMS = ('pretrain', 'mw_rank_restored', 'nll_restored', 'shape_fy_restored')
LABELS = ('事前学習5ep', 'MW Rank FT6ep', 'IMLE-NLL FT6ep', 'SHAPE-fy FT6ep')


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def functions(source, names, header):
    tree = ast.parse(source.read_text())
    found = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in found} != set(names):
        raise RuntimeError('Missing frozen adapter function')
    output = header+'\n\n'+'\n\n'.join(ast.unparse(node) for node in found)+'\n'
    copies = {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(output).body
              if isinstance(node, ast.FunctionDef)}
    assert all(copies[node.name] == ast.dump(node, include_attributes=False) for node in found)
    return output


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify():
    for path, expected in read(OUT/'implementation.json').items():
        if sha(path) != expected:
            raise RuntimeError('Frozen evaluation input changed: '+path)
    return read(OUT/'plan.json')


def prepare():
    if (OUT/'implementation.json').exists():
        verify()
        return
    if OUT.exists():
        raise RuntimeError('Inspect incomplete evaluation setup before reuse')
    OUT.mkdir()
    for name in ('scheduler', 'dispatch', 'models', 'panels', 'validation'):
        (OUT/name).mkdir()
    shutil.copyfile(__file__, OUT/'controller.py')
    model_adapter = PROJECT/'scripts/chemical_consistency/predict_mx.py'
    benchmark = PROJECT/'scripts/efold_hf/bprnanew_benchmark.py'
    nofam = PROJECT/'scripts/efold_hf/archive_nofam_report.py'
    adapter = functions(model_adapter, ['frozen_mxfold','create_model','restore'],
        'import argparse\nfrom importlib.machinery import PathFinder\nfrom pathlib import Path\nimport sys')
    assert adapter.count("model.to('cuda:0').eval()") == 1
    (OUT/'mx_adapter.py').write_text(adapter.replace("model.to('cuda:0').eval()", "model.to('cpu').eval()"))
    (OUT/'score_adapter.py').write_text(functions(benchmark, ['pair_metrics'], '')+'\n'+functions(
        nofam, ['aggregate','checked_pairs'], 'from collections import defaultdict\nimport statistics'))
    panels = {}
    for dataset, parent, expected in [('bpRNAnew',BASE/'bprnanew_20261029',5401),
                                      ('Archive-NoFam',BASE/'archive_nofam_20261031',89)]:
        prior = read(parent/'plan.json')
        assert sha(parent/'panel.jsonl') == prior['panel_sha256']
        selected = rows(parent/'panel.jsonl')
        assert len(selected) == expected and len({r['id'] for r in selected}) == expected
        assert max(len(r['sequence']) for r in selected) <= 1000
        path = OUT/'panels'/(dataset+'.jsonl')
        shutil.copyfile(parent/'panel.jsonl', path)
        panels[dataset] = {'path':str(path), 'sha256':sha(path), 'n':expected, 'prior_root':str(parent)}
    prior = BASE/'archive_nofam_20261031'
    assert read(prior/'cohorts-complete.json')['cohorts_sha256'] == sha(prior/'cohorts.json')
    assert read(prior/'published_validation.json')['success']
    shutil.copyfile(prior/'cohorts.json', OUT/'cohorts.json')
    panel = {r['id']:r for r in rows(FIRST/'data/test.jsonl')}
    controls = sorted((r for r in panel.values() if r['source']=='viral_fragments'),
                      key=lambda r:(len(r['sequence']),r['id']))
    controls = [controls[0],controls[len(controls)//2],controls[-1]]
    models = []
    for arm, label in zip(ARMS, LABELS, strict=True):
        root, epoch = (FIRST,5) if arm=='pretrain' else (SECOND,3)
        checkpoint = root/f'runs/{arm}/epoch-{epoch:03d}.pt'
        marker = read(checkpoint.with_name(f'epoch-{epoch:03d}-complete.json'))
        summary = read(root/f'evaluation/{arm}/epoch-{epoch:03d}/summary.json')
        checksum = sha(checkpoint)
        assert marker['success'] and checksum == marker['checkpoint_sha256'] == summary['checkpoint_sha256']
        if arm != 'pretrain':
            assert read(root/f'runs/{arm}/complete.json')['success'] and summary['global_ft_epoch'] == 6
        saved = root/f'evaluation/{arm}/epoch-{epoch:03d}/per_sequence.jsonl'
        assert sha(saved) == summary['predictions_sha256']
        predictions = {r['id']:r for r in rows(saved)}
        local = OUT/'models'/(arm+'.pt');shutil.copyfile(checkpoint, local)
        config = OUT/'models'/(arm+'.conf');shutil.copyfile(root/'baseline.conf',config)
        plan = read(root/'plan.json')
        source_hashes = {str(root/'source'/relative):digest for relative,digest in plan['source_sha256'].items()
                         if relative.startswith('mxfold2/')}
        models.append({'model':arm,'label':label,'checkpoint':str(local),'checkpoint_sha256':checksum,
            'original_checkpoint':str(checkpoint),'config':str(config),'config_sha256':sha(config),
            'source':str(root/'source'),'source_hashes':source_hashes,
            'controls':[{**item,'expected_pairs':predictions[item['id']]['predicted_pairs']} for item in controls]})
    training_files = []
    for root, name in [(FIRST,'bprna.jsonl'),(FIRST,'chemical.jsonl'),(SECOND,'chemical.jsonl')]:
        path = root/'data'/name;plan = read(root/'plan.json')
        expected = plan['data_sha256'][name]
        training_files.append({'path':str(path),'sha256':expected})
    plan = {'condition':2,'pretrain_epoch':5,'FT_epoch':6,'FT_curriculum':'Ribonanza3 then pri-miRNA/human mRNA3',
        'panels':panels,'models':models,'training_files':training_files,
        'reference_policy':'Original reference contacts; exact base-pair matches; no pruning or tolerance',
        'Archive-NoFam_primary':'paper_bprna_filtered:79 RNAs/40 groups; equal group mean',
        'bpRNAnew_primary':'all5401; equal RNA mean; also ACGU-only5388',
        'no_BPP':True,'execution_device':'cpu','hard_wall_seconds':3600,'soft_wall_seconds':3300,
        'adapter_source_sha256':{str(p):sha(p) for p in [model_adapter,benchmark,nofam]}}
    write(OUT/'plan.json', plan)
    job = f"#!/bin/bash\nset -euo pipefail\ncd '{PROJECT}'\nexport OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TZ=Asia/Tokyo\nexec '{PROJECT}/.venv/bin/python' -u '{OUT}/controller.py' \"$EVAL_ACTION\" --index \"${{EVAL_INDEX:-1}}\"\n"
    (OUT/'job.sh').write_text(job)
    immutable = [OUT/name for name in ['controller.py','mx_adapter.py','score_adapter.py','cohorts.json','plan.json','job.sh']]
    immutable += list((OUT/'panels').iterdir())+list((OUT/'models').iterdir())
    write(OUT/'implementation.json',{str(p):sha(p) for p in immutable})
    verify()
    print('Prepared four fixed models and both original panels', flush=True)


def submit(key, action, index=1, hold=None):
    receipt = OUT/'dispatch'/(key+'.json')
    if receipt.exists():
        return read(receipt)['job_id']
    resource = 'cpu_4=1'
    args = [QSUB,'-terse','-cwd','-j','y','-A','tga-satolab-gtex','-r','y',
        '-N',f'c2p5cpu_{action}{index}','-l',resource,'-l','h_rt=01:00:00',
        '-l','hostname=!r22n6','-o',str(OUT/'scheduler'),'-v',f'EVAL_ACTION={action},EVAL_INDEX={index}']
    if hold:
        args += ['-hold_jid',','.join(map(str,hold))]
    args.append(str(OUT/'job.sh'))
    result = subprocess.run(['sg','tga-satolab-gtex','-c',shlex.join(args)],cwd=PROJECT,
        env={**os.environ,'TZ':'Asia/Tokyo'},capture_output=True,text=True,check=True)
    job = result.stdout.strip().split('.')[0]
    if not job.isdigit():
        raise RuntimeError('Unexpected qsub result: '+result.stdout)
    write(receipt,{'job_id':job,'action':action,'index':index,'hold':hold,'command':args})
    print('submitted',action,index,job,flush=True)
    return job


def checked_prediction(item, pairs, scoring):
    pairs = [list(p) for p in scoring.checked_pairs(pairs,len(item['sequence']))]
    return {'id':item['id'],'sequence_sha256':item['sequence_sha256'],'length':len(item['sequence']),
            'predicted_pairs':pairs,**scoring.pair_metrics(item['pairs'],pairs)}


def work(index):
    import torch
    plan = verify();model_spec = plan['models'][index-1]
    for path, digest in model_spec['source_hashes'].items():
        assert sha(path) == digest,path
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    adapter = load('frozen_external_mx_adapter',OUT/'mx_adapter.py')
    scoring = load('frozen_external_scoring',OUT/'score_adapter.py')
    model = adapter.create_model(model_spec['source'],model_spec['config'])
    payload = adapter.restore(model,model_spec['checkpoint'])
    assert payload['optimizer_state_dict']['param_groups'][0]['step'] == (1278835 if index==1 else 1324027)
    def predict(item):
        native = model([item['sequence']],return_partfunc=False)[2][0]
        return [[i-1,int(j)-1] for i,j in enumerate(native) if i>0 and j>i]
    started = time.monotonic();controls=[]
    with torch.inference_mode():
        for item in model_spec['controls']:
            pairs = predict(item)
            assert sorted(map(tuple,pairs)) == sorted(map(tuple,item['expected_pairs'])),item['id']
            controls.append(item['id'])
        ambiguous = [r for r in rows(plan['panels']['bpRNAnew']['path']) if set(r['sequence'])-set('ACGU')]
        for item in ambiguous:
            checked_prediction(item,predict(item),scoring)
        write(OUT/'validation'/(model_spec['model']+'.json'),{'success':True,
            'saved_eFold_structures_reproduced':controls,'ambiguous_bpRNAnew_inputs_tested':len(ambiguous),
            'checkpoint_sha256':model_spec['checkpoint_sha256'],'no_BPP':True})
        for dataset, spec in plan['panels'].items():
            items = sorted(rows(spec['path']),key=lambda r:(-len(r['sequence']),r['id']))
            reference = {r['id']:r for r in items}
            folder = OUT/'predictions'/dataset/model_spec['model'];folder.mkdir(parents=True,exist_ok=True)
            path = folder/'predictions.jsonl'
            previous = rows(path) if path.exists() else []
            done = {r['id'] for r in previous}
            assert len(done) == len(previous) and done <= set(reference)
            for cached in previous:
                assert cached == checked_prediction(reference[cached['id']],cached['predicted_pairs'],scoring)
            with path.open('a') as handle:
                for item in items:
                    if item['id'] in done:
                        continue
                    row = checked_prediction(item,predict(item),scoring)
                    handle.write(json.dumps(row,allow_nan=False)+'\n');handle.flush();done.add(item['id'])
                    if len(done)%250 == 0:
                        print(model_spec['model'],dataset,len(done),'/',len(items),flush=True)
                    if time.monotonic()-started >= plan['soft_wall_seconds']:
                        raise TimeoutError('Predictions saved; rerun this model to resume')
            assert done == set(reference)
            write(folder/'complete.json',{'success':True,'n':len(done),
                'checkpoint_sha256':model_spec['checkpoint_sha256'],'predictions_sha256':sha(path)})
    write(OUT/'validation'/(model_spec['model']+'-complete.json'),{'success':True,'seconds':time.monotonic()-started})
    print('Complete',model_spec['model'],flush=True)


def report():
    plan = verify();scoring = load('frozen_external_report_scoring',OUT/'score_adapter.py')
    cohorts = read(OUT/'cohorts.json');metrics=[];predictions={};per_rna=[]
    panel_by_dataset = {name:{r['id']:r for r in rows(spec['path'])} for name,spec in plan['panels'].items()}
    for dataset, reference in panel_by_dataset.items():
        for model in plan['models']:
            folder = OUT/'predictions'/dataset/model['model'];marker=read(folder/'complete.json')
            path=folder/'predictions.jsonl'
            assert marker['success'] and marker['predictions_sha256']==sha(path)
            assert marker['checkpoint_sha256']==model['checkpoint_sha256']
            saved=rows(path)
            assert len(saved)==len(reference) and {r['id'] for r in saved}==set(reference)
            for item in saved:
                assert item==checked_prediction(reference[item['id']],item['predicted_pairs'],scoring)
            predictions[(dataset,model['model'])]=saved
            per_rna.extend({'dataset':dataset,'model':model['model'],**r} for r in saved)
            if dataset=='Archive-NoFam':
                for name, selection in cohorts['cohorts'].items():
                    selected=[r for r in saved if r['id'] in set(selection['ids'])]
                    value=scoring.aggregate(selected,cohorts['group_map'])
                    metrics.append({'dataset':dataset,'model':model['model'],'label':model['label'],'cohort':name,**value})
            else:
                for name in ['all_5401','ACGU_only_5388']:
                    selected=saved if name=='all_5401' else [r for r in saved if not set(reference[r['id']]['sequence'])-set('ACGU')]
                    tp,fp,fn=(sum(r[k] for r in selected) for k in ['tp','fp','fn'])
                    value={'n_rnas':len(selected),'rna_macro':{k:statistics.mean(r[k] for r in selected if r[k] is not None) for k in ['precision','recall','f1']},
                        'micro_f1':2*tp/(2*tp+fp+fn) if tp+fp+fn else 1.}
                    metrics.append({'dataset':dataset,'model':model['model'],'label':model['label'],'cohort':name,**value})
    overlaps={};union={name:set() for name in panel_by_dataset}
    for training in plan['training_files']:
        assert sha(training['path'])==training['sha256']
        matches={name:set() for name in panel_by_dataset}
        indices={name:defaultdict(list) for name in panel_by_dataset}
        for name, reference in panel_by_dataset.items():
            for identifier,item in reference.items():indices[name][item['sequence']].append(identifier)
        with Path(training['path']).open() as handle:
            for line in handle:
                sequence=json.loads(line)['sequence'].upper().replace('T','U')
                for name,index in indices.items():matches[name].update(index.get(sequence,[]))
        overlaps[training['path']]={name:sorted(ids) for name,ids in matches.items()}
        for name in union:union[name].update(matches[name])
    for dataset, excluded in union.items():
        for model in plan['models']:
            selected=[r for r in predictions[(dataset,model['model'])] if r['id'] not in excluded]
            if dataset=='Archive-NoFam':
                primary=set(cohorts['cohorts']['paper_bprna_filtered']['ids'])
                selected=[r for r in selected if r['id'] in primary]
                value=scoring.aggregate(selected,cohorts['group_map'])
            else:
                tp,fp,fn=(sum(r[k] for r in selected) for k in ['tp','fp','fn'])
                value={'n_rnas':len(selected),'rna_macro':{k:statistics.mean(r[k] for r in selected if r[k] is not None) for k in ['precision','recall','f1']},
                    'micro_f1':2*tp/(2*tp+fp+fn) if tp+fp+fn else 1.}
            metrics.append({'dataset':dataset,'model':model['model'],'label':model['label'],'cohort':'common_no_exact_training_overlap',**value})
    write(OUT/'training_exact_overlap.json',{'by_training_file':overlaps,'union':{k:sorted(v) for k,v in union.items()}})
    write(OUT/'results.json',{'success':True,'results':metrics,'prediction_rows_verified':len(per_rna),'no_BPP':True})
    text=['# 条件2・事前学習5ep起点：Archive-NoFamとbpRNAnew','',
        '事前学習5epと、その状態からRibonanza3ep→pri-miRNA＋ヒトmRNA3epのFTを終えた最終モデルを比較。replay20%、熱力学的正則化0.125。既存の凍結パネル・参照塩基対・復元関数・指標集計を使用。','',
        '|モデル|Archive-NoFam F1（79配列・40群）|Archive-NoFam F1（全89配列・47群）|bpRNAnew F1（5401配列）|','|---|---:|---:|---:|']
    def score(dataset,model,cohort):
        row=next(r for r in metrics if (r['dataset'],r['model'],r['cohort'])==(dataset,model,cohort))
        return row['group_macro']['f1'] if dataset=='Archive-NoFam' else row['rna_macro']['f1']
    for model in plan['models']:
        values=[score('Archive-NoFam',model['model'],c) for c in ['paper_bprna_filtered','full89']]
        values.append(score('bpRNAnew',model['model'],'all_5401'))
        text.append('| '+model['label']+' | '+' | '.join(f'{v:.6f}' for v in values)+' |')
    text+=['','Archive-NoFamはRNAごとの値を構造群内で平均した後に群間で等重み平均。主対象は従来の論文再現済み79配列・40群で固定し、全89配列および44配列の補足対象も保存。bpRNAnewは5401配列のRNA平均で、ACGUのみ5388配列も保存。参照構造の剪定や塩基対位置の許容誤差は加えない。','',
        '各モデルは保存済みeFoldテスト3構造との予測完全一致と、bpRNAnewの曖昧塩基を含む13配列の推論を検証してから全件推論。全予測のID・配列ハッシュ・塩基対・指標を再検証。','',
        '訓練完全一致の和集合: '+', '.join(f'{k}={len(v)}配列' for k,v in union.items())+'。主表は従来と同じ全対象。全4モデル共通の完全一致除外後の集計はresults.jsonのcommon_no_exact_training_overlapに保存。より広い相同性の追加除外はしていない。','',
        '両パネルには実測reactivityがないため構造指標を評価。Archive-NoFamの入力は従来と同じ著者プレプリント補足版で、群・除外対象・論文集計の検証は既存評価の記録を継承。入力パネルとモデル、参照した既存評価の場所はplan.jsonに記録。','']
    (OUT/'REPORT_ja.md').write_text('\n'.join(text))
    write(OUT/'complete.json',{'success':True,'models':4,'n_bpRNAnew':5401,'n_Archive_NoFam':89,
        'prediction_rows_verified':len(per_rna),'finished_at_jst':datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(),
        'results_sha256':sha(OUT/'results.json')})
    print((OUT/'REPORT_ja.md').read_text(),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['prepare','submit','work','report'])
    parser.add_argument('--index',type=int,default=1)
    args=parser.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='submit':
        verify();jobs=[submit(f'work-{i}-initial','work',i) for i in range(1,5)]
        collector=submit('report-initial','report',hold=jobs)
        write(OUT/'jobs.json',{'work':jobs,'report':collector})
    elif args.action=='work':work(args.index)
    else:report()


if __name__=='__main__':
    main()
