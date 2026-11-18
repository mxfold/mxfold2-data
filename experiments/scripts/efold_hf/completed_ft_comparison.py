#!/usr/bin/env python3
"""Audit and compare completed two-stage FT runs using their saved structures."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path

PROJECT = Path('/gs/bs/tga-satolab-gtex/satoken/repos/mxfold2-ryntaro')
BASE = PROJECT/'qsub_logs/efold_hf'
OUT = BASE/'completed_ft_comparison_20261102'
ARMS = ('pretrain','mw_rank_restored','nll_restored','shape_fy_restored')
LABELS = {'pretrain':'事前学習後','mw_rank_restored':'MW Rank',
          'nll_restored':'IMLE-NLL','shape_fy_restored':'SHAPE-fy'}
CASES = [
    ('c1p10','条件1・PT10','condition1_pt010_ribonanza3_replay20_20261101','condition1_pt010_pri_mrna3_after_ribonanza3_replay20_20261101',10),
    ('c1p20','条件1・PT20','condition1_ribonanza3_replay20_20261101','condition1_pri_mrna3_after_ribonanza3_replay20_20261101',20),
    ('c3p20','条件3・PT20','condition3_ribonanza3_replay20_20261101','condition3_pri_mrna3_after_ribonanza3_replay20_20261101',20),
]
SOURCES = ('PDB','archiveII','viral_fragments','lncRNA')
EXPECTED = {'PDB':355,'archiveII':3370,'viral_fragments':40,'lncRNA':30}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def rows(path):return [json.loads(line) for line in Path(path).open() if line.strip()]


def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def equal(a,b):
    if a is None or b is None:
        if a is not b:raise AssertionError(f'Undefined mismatch: {a}, {b}')
    elif not math.isclose(float(a),float(b),rel_tol=1e-11,abs_tol=1e-12):
        raise AssertionError(f'Metric mismatch: {a}, {b}')


def f1_counts(reference,predicted):
    ref,pred=set(map(tuple,reference)),set(map(tuple,predicted))
    tp=len(ref&pred);fp=len(pred)-tp;fn=len(ref)-tp
    return {'tp':tp,'fp':fp,'fn':fn,'f1':2*tp/(2*tp+fp+fn) if ref or pred else 1.}


def aggregate_f1(values):
    groups={}
    for source in SOURCES:
        group=[r for r in values if r['source']==source]
        if not group:continue
        tp,fp,fn=(sum(r[k] for r in group) for k in ('tp','fp','fn'))
        groups[source]={'n':len(group),'f1':sum(r['f1'] for r in group)/len(group),
                        'micro_f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.}
    return groups


def common_strata(frames):
    keys=None
    for frame in frames:
        valid={k for k,v in frame.items() if v['auroc'] is not None and v['spearman'] is not None}
        keys=valid if keys is None else keys&valid
    return keys or set()


def summarize_chemical(frame,keys):
    # Strata -> replicate/profile -> probe -> RNA, with equal RNA weights.
    result={}
    for probe in ('ALL','DMS','SHAPE'):
        profiles={}
        for key in sorted(keys):
            identifier,modifier,profile,stratum=key
            if probe!='ALL' and modifier!=probe:continue
            profiles.setdefault((identifier,modifier,profile),[]).append(frame[key])
        probes={}
        for (identifier,modifier,profile),strata in profiles.items():
            probes.setdefault((identifier,modifier),[]).append({k:sum(s[k] for s in strata)/len(strata)
                                                               for k in ('auroc','spearman')})
        units={}
        for (identifier,modifier),replicates in probes.items():
            units.setdefault(identifier,[]).append({k:sum(s[k] for s in replicates)/len(replicates)
                                                    for k in ('auroc','spearman')})
        values=[{k:sum(s[k] for s in unit)/len(unit) for k in ('auroc','spearman')} for unit in units.values()]
        result[probe]={'n_rnas':len(values),'n_profiles':len(profiles),
            'n_strata':sum(len(p) for p in profiles.values()),
            **{k:sum(v[k] for v in values)/len(values) if values else None for k in ('auroc','spearman')}}
    return result


def save_tsv(path,records):
    with Path(path).open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(records[0]),delimiter='\t')
        writer.writeheader();writer.writerows(records)


def render(report):
    models=len(CASES)*len(ARMS)
    fmt=lambda value:'—' if value is None else f'{value:.4f}'
    lines=['# 完了FTの比較：'+'・'.join(c[1] for c in CASES),'',
        '最終FTはRibonanza3epoch＋pri-miRNA/ヒトmRNA3epochの通算6epoch。checkpoint選択は最終epochで固定。F1は配列ごとの厳密な塩基対F1のmacro平均。公開テスト全3,795配列を用いる。','',
        '| 条件・起点 | テスト集合（n） | 事前学習後 | MW Rank | IMLE-NLL | SHAPE-fy |',
        '|---|---|---:|---:|---:|---:|']
    for case,label,*_ in CASES:
        for source in SOURCES:
            count=report['models'][case]['pretrain']['F1_full'][source]['n']
            cells=[f"{report['models'][case][arm]['F1_full'][source]['f1']:.4f}" for arm in ARMS]
            lines.append('| '+label+' | '+source+f'（{count}） | '+' | '.join(cells)+' |')
    lines+=['','## 単一予測構造による新AUROC / 新Spearman','',
        f'AUROCは単一予測構造の未対合0/1をラベル、raw reactivityをスコアとして計算する。Spearmanは同じ未対合0/1とraw reactivityの相関。塩基対確率・reactivity閾値・値のclippingは使わない。有効観測はfiniteかつreactivity>−10。DMSはA/C別に計算して配列内で等重み平均し、配列macro平均する。全{models}モデルに共通して両指標を定義できる測定・塩基種層に固定する。未定義を0.5で補完する補助指標は主表に用いない。','',
        '| 条件・起点 | viralの対象 | n | 事前学習後 | MW Rank | IMLE-NLL | SHAPE-fy |',
        '|---|---|---:|---:|---:|---:|---:|']
    for case,label,*_ in CASES:
        for probe in ('ALL','DMS','SHAPE'):
            n=report['models'][case]['pretrain']['chemical_common'][probe]['n_rnas']
            cells=[]
            for arm in ARMS:
                scores=report['models'][case][arm]['chemical_common'][probe]
                cells.append(fmt(scores['auroc'])+' / '+fmt(scores['spearman']))
            lines.append('| '+label+' | '+probe+f' | {n} | '+' | '.join(cells)+' |')
    lines+=['',f"全{models}モデルの共通対象は{report['chemical_common_coverage']['RNAs']} RNA、{report['chemical_common_coverage']['profiles']}測定、{report['chemical_common_coverage']['strata']}塩基種層。PDB・ArchiveII・lncRNAには、この凍結パネルではreactivity観測を付与していない。",'',
        '## 全系列共通の訓練完全一致除外','',
        f"訓練集合との完全一致の和集合{len(report['common_exclusion_ids'])}配列を、全{models}モデルから共通に除外する。完全一致以外の相同性除外は追加していない。",'',
        '| 条件・起点 | テスト集合（共通除外後n） | 事前学習後 | MW Rank | IMLE-NLL | SHAPE-fy |',
        '|---|---|---:|---:|---:|---:|']
    for case,label,*_ in CASES:
        for source in ('PDB','archiveII'):
            n=report['models'][case]['pretrain']['F1_common_no_exact_overlap'][source]['n']
            cells=[f"{report['models'][case][arm]['F1_common_no_exact_overlap'][source]['f1']:.4f}" for arm in ARMS]
            lines.append('| '+label+' | '+source+f'（{n}） | '+' | '.join(cells)+' |')
    lines+=['','## 監査','',
        f"保存済み予測ファイル・checkpointのSHA-256と完了marker、全3,795のID集合を照合した。F1をreference pairsと予測pairsから再計算し、保存済みの配列別値・集計値と一致することを確認した。chemical指標は保存済み予測pairsから未対合0/1を再構成し、凍結raw observationsから再計算した。さらにscikit-learn ROC AUCとSciPy spearmanrで全{report['verification_counts']['chemical_strata']}塩基種層を独立照合した。モデル推論は追加実行していない。",'',
        '- results.json：全精度、coverage、完全一致除外、差分、入力provenance。',
        '- f1_full.tsv・f1_common_no_exact_overlap.tsv：F1表。',
        '- chemical_common.tsv・chemical_deltas.tsv：新AUROC/Spearmanと事前学習との差分。',
        '- chemical_strata.tsv：再計算した測定・塩基種ごとの指標。',
        '- verification.json：チェックサム・独立指標照合の検証記録。','']
    return '\n'.join(lines)


def main():
    import numpy as np
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score
    OUT.mkdir(parents=True,exist_ok=True)
    metric_path=BASE/CASES[0][2]/'metrics.py'
    spec=importlib.util.spec_from_file_location('frozen_completed_structure_metrics',metric_path)
    metrics=importlib.util.module_from_spec(spec);spec.loader.exec_module(metrics)
    panel_path=BASE/CASES[0][2]/'data/test.jsonl'
    panel={r['id']:r for r in rows(panel_path)}
    if len(panel)!=3795:raise RuntimeError('Wrong panel size')
    if {s:sum(r['source']==s for r in panel.values()) for s in SOURCES}!=EXPECTED:
        raise RuntimeError('Wrong published dataset counts')
    all_excluded=set()
    for _,_,first,second,_ in CASES:
        for root in (BASE/first,BASE/second):
            if sha(root/'data/test.jsonl')!=sha(panel_path):raise RuntimeError('Different test panel')
            if sha(root/'metrics.py')!=sha(metric_path):raise RuntimeError('Different frozen metric implementation')
            all_excluded.update(read(root/'test_exact_overlap.json')['ids'])
    report={'definition':{'F1':'per-RNA exact-base-pair macro F1; empty reference and prediction=1',
        'AUROC':'predicted unpaired binary label, measured raw reactivity score; no BPP',
        'Spearman':'same predicted unpaired binary state vs raw measured reactivity',
        'aggregation':'DMS A/C equal mean within profile; replicate mean; probe mean per RNA; equal RNA macro',
        'comparison_mask':f'both metrics defined in every one of all{len(CASES)*len(ARMS)} models',
        'final_checkpoint':'stage2 epoch3, cumulative FT epoch6'},
        'models':{},'common_exclusion_ids':sorted(all_excluded),'provenance':[]}
    frames={};independent_checks=0;f1_checks=0;export_strata=[]
    for case,label,first,second,epoch in CASES:
        report['models'][case]={}
        for arm in ARMS:
            root=BASE/(first if arm=='pretrain' else second)
            checkpoint_epoch=epoch if arm=='pretrain' else 3
            destination=root/'evaluation'/arm/f'epoch-{checkpoint_epoch:03d}'
            summary=read(destination/'summary.json');path=destination/'per_sequence.jsonl'
            checkpoint=root/'runs'/arm/f'epoch-{checkpoint_epoch:03d}.pt'
            marker=read(checkpoint.with_name(f'epoch-{checkpoint_epoch:03d}-complete.json'))
            if not summary['success'] or sha(path)!=summary['predictions_sha256']:
                raise RuntimeError('Prediction checksum mismatch')
            if sha(checkpoint)!=summary['checkpoint_sha256'] or sha(checkpoint)!=marker['checkpoint_sha256']:
                raise RuntimeError('Checkpoint checksum mismatch')
            if arm!='pretrain' and (not read(root/'runs'/arm/'complete.json')['success'] or summary.get('global_ft_epoch')!=6):
                raise RuntimeError('Final FT checkpoint is not complete epoch6')
            saved=rows(path)
            if len(saved)!=len(panel) or {r['id'] for r in saved}!=set(panel):
                raise RuntimeError('Prediction panel IDs differ')
            structural=[];frame={}
            for prediction in saved:
                item=panel[prediction['id']]
                if prediction['length']!=len(item['sequence']) or prediction['source']!=item['source']:
                    raise RuntimeError('Sequence metadata mismatch')
                binary=metrics.from_pairs(item['sequence'],prediction['predicted_pairs'])
                recomputed=f1_counts(item['pairs'],prediction['predicted_pairs'])
                for key in recomputed:equal(recomputed[key],prediction[key])
                structural.append({'id':item['id'],'source':item['source'],**recomputed});f1_checks+=1
                observations=item.get('observations',[])
                if len(observations)!=len(prediction['chemical']):raise RuntimeError('Chemical observation mismatch')
                for index,(obs,cached) in enumerate(zip(observations,prediction['chemical'],strict=True)):
                    probe='DMS' if obs['probe']=='dms' else 'SHAPE'
                    if probe!=cached['probe']:raise RuntimeError('Chemical probe mismatch')
                    strata=metrics.score_profile({'sequence':item['sequence'],'modifier':probe,'values':obs['values']},binary)
                    if len(strata)!=len(cached['strata']):raise RuntimeError('Wrong saved strata')
                    for score,prior in zip(strata,cached['strata'],strict=True):
                        if score.keys()!=prior.keys():raise RuntimeError('Saved chemical schema differs')
                        for key in score:
                            if isinstance(score[key],str):
                                if score[key]!=prior[key]:raise RuntimeError('Stratum differs')
                            else:equal(score[key],prior[key])
                        values=np.asarray(obs['values'],dtype=np.float64)
                        valid=np.isfinite(values)&(values>-10)
                        if probe=='DMS':valid &= np.asarray(list(item['sequence']))==score['stratum']
                        if score['auroc'] is not None:
                            equal(score['auroc'],float(roc_auc_score(binary[valid],values[valid])))
                        if score['spearman'] is not None:
                            equal(score['spearman'],float(spearmanr(binary[valid],values[valid]).statistic))
                        independent_checks+=1
                        key=(item['id'],probe,index,score['stratum'])
                        if key in frame:raise RuntimeError('Duplicate profile stratum')
                        frame[key]=score
                        export_strata.append({'case':case,'model':arm,'id':item['id'],'probe':probe,
                            'profile_index':index,**score})
            full=aggregate_f1(structural)
            for source in SOURCES:
                for key in ('n','f1','micro_f1'):equal(full[source][key],summary['test'][source][key])
            local_excluded=set(read(root/'test_exact_overlap.json')['ids'])
            local=aggregate_f1([r for r in structural if r['id'] not in local_excluded])
            for source in SOURCES:
                for key in ('n','f1','micro_f1'):equal(local[source][key],summary['without_exact_training_overlap'][source][key])
            report['models'][case][arm]={'F1_full':full,'F1_common_no_exact_overlap':aggregate_f1([r for r in structural if r['id'] not in all_excluded]),
                'F1_condition_specific_no_exact_overlap':local}
            frames[(case,arm)]=frame
            report['provenance'].append({'case':case,'model':arm,'checkpoint':str(checkpoint),
                'checkpoint_sha256':sha(checkpoint),'predictions':str(path),'predictions_sha256':sha(path)})
    common=common_strata(list(frames.values()))
    coverage={'RNAs':len({k[0] for k in common}),'profiles':len({k[:3] for k in common}),
        'strata':len(common),'by_probe':{p:{'RNAs':len({k[0] for k in common if k[1]==p}),
            'strata':sum(k[1]==p for k in common)} for p in ('DMS','SHAPE')}}
    report['chemical_common_coverage']=coverage
    deltas=[]
    for case,_,_,_,_ in CASES:
        for arm in ARMS:
            frame=frames[(case,arm)]
            report['models'][case][arm]['chemical_common']=summarize_chemical(frame,common)
            available={k for k,v in frame.items() if v['auroc'] is not None and v['spearman'] is not None}
            report['models'][case][arm]['chemical_available']=summarize_chemical(frame,available)
            if arm!='pretrain':
                for probe in ('ALL','DMS','SHAPE'):
                    value=report['models'][case][arm]['chemical_common'][probe]
                    initial=report['models'][case]['pretrain']['chemical_common'][probe]
                    deltas.append({'case':case,'model':arm,'probe':probe,
                        'delta_auroc':value['auroc']-initial['auroc'],'delta_spearman':value['spearman']-initial['spearman']})
    for scope,file in [('F1_full','f1_full.tsv'),('F1_common_no_exact_overlap','f1_common_no_exact_overlap.tsv')]:
        exported=[]
        for case,label,*_ in CASES:
            for source in SOURCES:
                result={'case':case,'label':label,'dataset':source,
                        'n':report['models'][case]['pretrain'][scope][source]['n']}
                result.update({arm:report['models'][case][arm][scope][source]['f1'] for arm in ARMS})
                exported.append(result)
        save_tsv(OUT/file,exported)
    chemicals=[]
    for case,label,*_ in CASES:
        for arm in ARMS:
            for probe,value in report['models'][case][arm]['chemical_common'].items():
                chemicals.append({'case':case,'label':label,'model':arm,'dataset':'viral_fragments','probe':probe,**value})
    save_tsv(OUT/'chemical_common.tsv',chemicals)
    save_tsv(OUT/'chemical_deltas.tsv',deltas)
    save_tsv(OUT/'chemical_strata.tsv',export_strata)
    verification={'success':True,'models':len(frames),'F1_per_sequence_checks':f1_checks,
        'chemical_strata_independently_checked':independent_checks,
        'common_coverage':coverage,'test_panel_sha256':sha(panel_path),'metric_source_sha256':sha(metric_path),
        'analysis_script_sha256':sha(__file__),'no_new_model_inference':True,'no_BPP_used':True}
    report['verification_counts']={'chemical_strata':independent_checks,'F1_per_sequence':f1_checks}
    write(OUT/'verification.json',verification);write(OUT/'results.json',report)
    (OUT/'README_ja.md').write_text(render(report))
    print(json.dumps(verification,ensure_ascii=False))
    print((OUT/'README_ja.md').read_text())


if __name__=='__main__':main()
