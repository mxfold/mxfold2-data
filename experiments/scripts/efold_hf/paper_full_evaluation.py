#!/usr/bin/env python3
"""Complete eFold's published test panels, including sequences above 1000 nt."""
from collections import Counter, defaultdict
from datetime import datetime
import json
import math
import os
from pathlib import Path
import statistics
import sys
from zoneinfo import ZoneInfo

sys.path.insert(0,str(Path(__file__).parent))
import bprnanew_benchmark as base
import paper_evaluation as previous

ROOT=base.ROOT
OUT=ROOT/'qsub_logs/efold_hf/paper_full_20261030'
PRIOR=previous.OUTPUT
DATA=ROOT/'data/efold_hf/paper_2026'
COUNTS={'PDB':355,'archiveII':3370,'viral_fragments':40,'lncRNA':30}
NAMES={'PDB':'PDB','archiveII':'ArchiveII','viral_fragments':'Viral mRNA','lncRNA':'Long ncRNA'}
MODELS={'author_eFold':None,'mx_bprna10':'pretrain','mx_mw_rank_restored':'mw_rank_restored',
        'mx_nll_restored':'nll_restored','mx_shape_fy_restored':'shape_fy_restored'}


def build_panel():
    rows=[]
    for source,n in COUNTS.items():
        raw=json.loads((base.RAW/(source+'.json')).read_text())
        if len(raw)!=n:raise ValueError('Published panel count differs: '+source)
        for reference,item in raw.items():
            sequence=base.normalize(item['sequence'])
            if not sequence or len(sequence)>2000 or set(sequence)-set('ACGU'):
                raise ValueError('Unexpected complete test sequence: '+reference)
            pairs=sorted({tuple(sorted(map(int,p))) for p in item['structure']})
            if any(not 0<=a<b<len(sequence) for a,b in pairs):raise ValueError('Invalid reference: '+reference)
            rows.append({'id':source+':'+reference,'source':source,'reference':reference,
                         'sequence':sequence,'sequence_sha256':base.hashlib.sha256(sequence.encode()).hexdigest(),
                         'pairs':pairs})
    if len({r['id'] for r in rows})!=sum(COUNTS.values()):raise ValueError('Duplicate reference ID')
    return rows


def aggregate(rows,panel):
    result={}
    for source,n in COUNTS.items():
        selected=[r for r in rows if panel[r['id']]['source']==source]
        result[source]={'n':len(selected)}
        for metric in ('precision','recall','f1'):
            values=[r[metric] for r in selected if r[metric] is not None]
            result[source][metric]=statistics.fmean(values) if values else None
            result[source]['n_'+metric]=len(values)
    return result


def prepare():
    import pyarrow as pa
    import pyarrow.feather as feather
    pa.set_cpu_count(1);pa.set_io_thread_count(1)
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'prepare-complete.json').exists():raise FileExistsError('Already prepared')
    rows=build_panel();panel={r['id']:r for r in rows}
    before={r['id']:r for r in base.read_jsonl(PRIOR/'panel.jsonl')}
    for key,r in before.items():
        now=panel[key]
        if r['sequence']!=now['sequence'] or {tuple(p) for p in r['pairs']}!={tuple(p) for p in now['pairs']}:
            raise ValueError('Previously scored input changed: '+key)
    extra=[r for r in rows if r['id'] not in before]
    if len(before)!=3757 or len(extra)!=38 or any(len(r['sequence'])<=1000 for r in extra):
        raise ValueError('Expected precisely 38 previously excluded long test entries')
    base.write_jsonl(OUT/'panel.jsonl',rows)
    base.write_jsonl(OUT/'additional_long_sequences.jsonl',extra)
    registry=json.loads((base.OUT/'plan.json').read_text())['models'][:5]
    if {m['model'] for m in registry}!=set(MODELS):raise ValueError('Unexpected frozen model registry')
    ordered=sorted(rows,key=lambda r:(-len(r['sequence']),r['id']))
    reusable={}
    for spec in registry:
        model=spec['model']
        if base.sha(spec['checkpoint'])!=spec['checkpoint_sha256']:raise ValueError('Checkpoint changed')
        if model=='author_eFold':
            folder=PRIOR/'author_efold_paper_gpu'
            if json.loads((folder/'plan.json').read_text())['weights_sha256']!=spec['checkpoint_sha256']:
                raise ValueError('eFold reuse checkpoint differs')
            records=[]
            for shard in range(4):
                path=folder/f'shard-{shard}.jsonl'
                marker=json.loads((folder/f'shard-{shard}-complete.json').read_text())
                if not marker['success'] or marker['predictions_sha256']!=base.sha(path):raise ValueError('eFold reuse changed')
                records+=base.read_jsonl(path)
        else:
            arm=MODELS[model]
            checkpoint=previous.previous_paths(arm)[1]
            if base.sha(checkpoint)!=spec['checkpoint_sha256']:raise ValueError('MXfold2 reuse checkpoint differs')
            records=list(previous.load_old(arm,panel).values())
            path=PRIOR/'predictions'/(arm+'.jsonl')
            marker=json.loads((PRIOR/'predictions'/(arm+'-complete.json')).read_text())
            if not marker['success'] or marker['checkpoint_sha256']!=spec['checkpoint_sha256'] or marker['predictions_sha256']!=base.sha(path):
                raise ValueError('Previous complete model marker differs')
            records+=base.read_jsonl(path)
        if len(records)!=len(before) or {r['id'] for r in records}!=set(before):raise ValueError('Incomplete cached evaluation')
        cached={}
        for record in records:
            item=panel[record['id']];metrics=previous.pair_metrics(item['pairs'],record['predicted_pairs'])
            if any(record[k]!=v for k,v in metrics.items()):raise ValueError('Cached structural score differs')
            cached[item['id']]={'id':item['id'],'length':len(item['sequence']),
                    'sequence_sha256':item['sequence_sha256'],'predicted_pairs':record['predicted_pairs'],**metrics}
        for shard in range(4):
            selected=[cached[r['id']] for i,r in enumerate(ordered) if i%4==shard and r['id'] in cached]
            base.write_jsonl(OUT/'predictions'/model/f'shard-{shard}.jsonl',selected)
        reusable[model]=len(cached)
    # Public author predictions give exact reference IDs and their original scores.
    table=feather.read_table(DATA/'results_benchmark_algos.feather',
            columns=['reference','dataset','model','length','F1','structure'],use_threads=False)
    comparisons=[];mismatches=[]
    for row in table.to_pylist():
        if row['model'] not in ('EternaFold','MXFold2'):
            continue
        rid=row['dataset']+':'+row['reference']
        if rid not in panel:raise ValueError('Author reference absent: '+rid)
        if row['length']!=len(panel[rid]['sequence']):raise ValueError('Author input length differs')
        if row['structure'] is None:continue
        metrics=previous.pair_metrics(panel[rid]['pairs'],row['structure'])
        if row['F1'] is not None and abs(metrics['f1']-row['F1'])>1e-6:
            mismatches.append({'model':row['model'],'id':rid,'recomputed':metrics['f1'],'saved':row['F1']})
        comparisons.append({'model':'paper_'+row['model'],'id':rid,'source':row['dataset'],**metrics})
    author_counts={}
    for model in sorted({r['model'] for r in comparisons}):
        part=[r for r in comparisons if r['model']==model]
        if len({r['id'] for r in part})!=len(part):raise ValueError('Duplicate author prediction')
        author_counts[model]=dict(Counter(r['source'] for r in part))
    for model in ('paper_EternaFold','paper_MXFold2'):
        if author_counts.get(model)!=COUNTS:raise ValueError('Author comparison coverage differs: '+model)
    if any(r['model'] in ('EternaFold','MXFold2') for r in mismatches):raise ValueError('Required author score audit failed')
    # PT files explicitly name every sequence, independently verifying the inputs.
    for name in ('test_results_PT.feather','test_results_PT+FT.feather'):
        data=feather.read_table(DATA/name,columns=['reference','sequence'],use_threads=False).to_pylist()
        seqs={r['reference']:r['sequence'] for r in data}
        if any(seqs.get(r['reference'])!=r['sequence'] for r in rows):raise ValueError('Author eFold input differs')
    base.write_jsonl(OUT/'author_comparators.jsonl',comparisons)
    base.write_json(OUT/'author_metric_audit.json',{'counts':author_counts,'f1_mismatches':mismatches})
    notebook=json.loads((DATA/'Figure5_structure_performance.ipynb').read_text())
    traces=notebook['cells'][6]['outputs'][0]['data']['application/vnd.plotly.v1+json']['data']
    figure={}
    for name in ('eFold','eFold (pre-trained)'):
        trace=next(t for t in traces if t['name']==name)
        figure[name]={}
        for source,label in NAMES.items():
            values=[float(y) for x,y in zip(trace['x'],trace['y']) if x==label]
            if len(values)!=COUNTS[source]:raise ValueError('Published figure count differs')
            figure[name][source]={'n':len(values),'f1':statistics.fmean(values)}
    base.write_json(OUT/'published_figure_scores.json',figure)
    plan={'dataset':'Complete author-published eFold evaluation panels','panel_counts':COUNTS,
          'n':len(rows),'panel_sha256':base.sha(OUT/'panel.jsonl'),'max_length':2000,
          'training_max_length':1000,'evaluation_length_override':'Include all published test entries, as requested 2026-10-30; do not change training.',
          'models':registry,'shards_per_model':4,'reused_per_model':reusable,
          'additional_by_source':dict(Counter(r['source'] for r in extra)),
          'additional_n':len(extra),'checkpoint_selection':'Fixed pretrain10 / FT5; no retraining or checkpoint selection',
          'author_counts':author_counts,'author_comparators_sha256':base.sha(OUT/'author_comparators.jsonl'),
          'input_sha256':{str(p):base.sha(p) for p in [*(base.RAW/(s+'.json') for s in COUNTS),DATA/'results_benchmark_algos.feather',DATA/'test_results_PT+FT.feather',DATA/'Figure5_structure_performance.ipynb']}}
    base.write_json(OUT/'plan.json',plan)
    base.write_json(OUT/'prepare-complete.json',{'success':True,'panel_sha256':plan['panel_sha256']})
    base.write_json(OUT/'status.json',{'complete':False,'reused_per_model':reusable,'additional_per_model':38})
    print(json.dumps(plan,indent=2,ensure_ascii=False),flush=True)


def predict(index,smoke=False):
    if not smoke:
        plan=json.loads((OUT/'plan.json').read_text())
        model=plan['models'][(index-1)//4]['model']
        validation=OUT/'smoke'/model/'reuse-validation.json'
        if not validation.exists() or not json.loads(validation.read_text())['success']:
            raise ValueError('Short-input cache validation has not succeeded: '+model)
    base.OUT=OUT
    base.work((index-1)*4+1 if smoke else index,smoke=smoke)
    if smoke:
        plan=json.loads((OUT/'plan.json').read_text());model=plan['models'][index-1]['model']
        cached={r['id']:r for r in base.read_jsonl(OUT/'predictions'/model/'shard-0.jsonl')}
        checked=0
        for row in base.read_jsonl(OUT/'smoke'/model/'shard-0.jsonl'):
            if row['id'] in cached:
                if row['predicted_pairs']!=cached[row['id']]['predicted_pairs']:raise ValueError('Restored model differs from prior input')
                checked+=1
        if checked<1:raise ValueError('No earlier short prediction validated')
        base.write_json(OUT/'smoke'/model/'reuse-validation.json',{'success':True,'matched_cached_predictions':checked})


def report():
    import csv
    plan=json.loads((OUT/'plan.json').read_text())
    if base.sha(OUT/'panel.jsonl')!=plan['panel_sha256']:raise ValueError('Full panel changed')
    panel={r['id']:r for r in base.read_jsonl(OUT/'panel.jsonl')}
    models={};counts={}
    for spec in plan['models']:
        model=spec['model'];records=[]
        for shard in range(4):
            path=OUT/'predictions'/model/f'shard-{shard}.jsonl'
            marker=json.loads((path.parent/f'shard-{shard}-complete.json').read_text())
            if not marker['success'] or marker['predictions_sha256']!=base.sha(path) or marker['checkpoint_sha256']!=spec['checkpoint_sha256']:
                raise ValueError('Incomplete or modified prediction shard')
            records+=base.read_jsonl(path)
        if len(records)!=len(panel) or {r['id'] for r in records}!=set(panel):raise ValueError('Not the full common test panel')
        for row in records:
            item=panel[row['id']]
            if row['sequence_sha256']!=item['sequence_sha256']:raise ValueError('Prediction sequence changed')
            if any(row[k]!=v for k,v in previous.pair_metrics(item['pairs'],row['predicted_pairs']).items()):raise ValueError('Metric verification failed')
        models[model]=aggregate(records,panel);counts[model]=len(records)
    author=base.read_jsonl(OUT/'author_comparators.jsonl')
    if base.sha(OUT/'author_comparators.jsonl')!=plan['author_comparators_sha256']:raise ValueError('Author scores changed')
    for model in sorted({r['model'] for r in author}):
        models[model]=aggregate([r for r in author if r['model']==model],panel)
    figure=json.loads((OUT/'published_figure_scores.json').read_text())
    result={'panel_counts':COUNTS,'models':models,'published_figure':figure,'models_inferred_complete':counts}
    base.write_json(OUT/'results.json',result)
    rows=[{'dataset':source,'model':model,**values} for model,data in models.items() for source,values in data.items()]
    with (OUT/'comparison.tsv').open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]),delimiter='\t');writer.writeheader();writer.writerows(rows)
    labels=[('author_eFold','著者公開eFold・論文手順'),('paper_EternaFold','論文著者EternaFold'),
            ('mx_bprna10','今回の事前学習10ep'),('mx_mw_rank_restored','MW Rank FT5ep'),
            ('mx_nll_restored','IMLE-NLL FT5ep'),('mx_shape_fy_restored','SHAPE-fy FT5ep')]
    lines=['# eFold論文と同じ全件での評価（1000nt超も含む）','',
           '集計日時：'+datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(),'',
           '旧評価は1000nt以下に限定していたため、ArchiveII 11件・Long ncRNA 27件が不足していた。'
           '今回の論文比較では評価のみ上限を2000ntに拡張し、学習データ・学習済み重みは固定した。'
           '全3,795件の参照ID・配列を著者公開ファイルと照合し、既存3,757件を再利用して長鎖38件を追加推論した。','',
           '## 全件での塩基対完全一致F1（配列ごとの平均）','',
           '| データ | 件数 | 論文eFold・保存図 | '+' | '.join(label for _,label in labels)+' |',
           '|---|---:|---:|'+'---:|'*len(labels)]
    for source,name in NAMES.items():
        vals=[f"{models[m][source]['f1']:.4f}" for m,_ in labels]
        lines.append('| '+name+' | '+str(COUNTS[source])+' | '+f"{figure['eFold'][source]['f1']:.4f}"+' | '+' | '.join(vals)+' |')
    lines+=['','eFoldの保存図は図に保存されたF1分布の平均で、配列IDの推定を使わない。'
            '公開eFoldの直接推論は同じ重み・GPU後処理を使うが、保存図との小さな差は明示して区別する。'
            'EternaFold列は論文著者の保存予測を同じ全件・同じ参照で再採点した値。','',
            '## 論文著者が保存した比較手法の予測（全件を再採点）','',
            '| 手法 | PDB | ArchiveII | Viral mRNA | Long ncRNA |','|---|---:|---:|---:|---:|']
    for model in sorted(m for m in models if m.startswith('paper_')):
        vals=[f"{models[model][s]['f1']:.4f} (n={models[model][s]['n']})" if models[model][s]['f1'] is not None else '—' for s in COUNTS]
        lines.append('| '+model.removeprefix('paper_')+' | '+' | '.join(vals)+' |')
    lines+=['','eFold・EternaFold・今回のMXfold2・論文著者のMXFold2は、すべて同じ全件で評価する。','',
            'precision・recall・F1とそれぞれの有効件数はcomparison.tsv。'
            '参照の塩基対は削除せず、F1には空予測も含む全件を使う。分母0のprecision/recallは未定義として平均からのみ除く。','',
            'bpRNAnewおよびRNAgym/TrainSetBの評価集合は今回変更していない。化学反応性との新AUROC/Spearmanの定義も変更しない。','',
            '件数の基準は著者のFigure 1/5の実評価データと公式配布である。Table 2にはArchiveII_filtered 355・lncRNA 10という別の件数があるが、'
            '本比較には実際の図データのArchiveII 3,370・Long ncRNA 30を使う。','',
            '出典：[公式Dryad配布・説明](https://datadryad.org/dataset/doi:10.5061/dryad.79cnp5j95)、'
            '[eFold論文](https://pmc.ncbi.nlm.nih.gov/articles/PMC12935039/)。','']
    (OUT/'REPORT_ja.md').write_text('\n'.join(lines))
    base.write_json(OUT/'status.json',{'generated_at':datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(),
           'complete':True,'panel_counts':COUNTS,'models_inferred_complete':counts,'new_predictions_per_model':38})
    print(json.dumps(result,indent=2,ensure_ascii=False),flush=True)


if __name__=='__main__':
    action=os.environ.get('PAPER_FULL_ACTION') or sys.argv[1]
    raw=os.environ.get('SGE_TASK_ID','1');index=int(raw) if raw.isdigit() else 1
    if action=='prepare':prepare()
    elif action in ('predict','smoke'):predict(index,action=='smoke')
    elif action=='report':report()
    else:raise ValueError(action)
