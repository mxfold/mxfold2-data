#!/usr/bin/env python3
"""Apply the agreed structure-only MW AUROC/Spearman to eFold comparators."""
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.chemical_consistency import map_mw_metrics as metrics
from scripts.chemical_consistency.map_mw_report import common_frames,table,atomic_tsv

OUT=ROOT/'qsub_logs/efold_hf/paper_methods_mw_20261031'
FULL=ROOT/'qsub_logs/efold_hf/paper_full_20261030'
AUTHOR=ROOT/'data/efold_hf/paper_2026/results_benchmark_algos.feather'


def paired_union(sequence,pairs):
    """Use every non-diagonal predicted contact, preserving multiplet outputs."""
    import numpy as np
    unpaired=np.ones(len(sequence),dtype=np.uint8)
    edges=set();self_edges=0
    for pair in pairs:
        a,b=map(int,pair)
        if not 0<=a<len(sequence) or not 0<=b<len(sequence):raise ValueError('Out-of-range predicted contact')
        if a==b:
            self_edges+=1
            continue
        edges.add(tuple(sorted((a,b))))
    degree=np.zeros(len(sequence),dtype=int)
    for a,b in edges:unpaired[a]=unpaired[b]=0;degree[a]+=1;degree[b]+=1
    return unpaired,{'unique_nonself_pairs':len(edges),'self_contacts_ignored':self_edges,
                     'multiplet_bases':int((degree>1).sum())}


def main():
    import pandas as pd
    import pyarrow as pa
    import pyarrow.feather as feather
    pa.set_cpu_count(1);pa.set_io_thread_count(1)
    OUT.mkdir(parents=True,exist_ok=True)
    profiles=[p for p in metrics.load_profiles() if p['dataset']=='eFold_viral']
    if len(profiles)!=40:raise ValueError('Expected the forty measured viral test profiles')
    sequences={p['sample_id']:p['sequence'] for p in profiles}
    if len(sequences)!=40:raise ValueError('Unexpected duplicate viral reference')
    labels={'author_eFold':'eFold（公開重み・論文手順）'}
    edges_by_model=defaultdict(dict)
    provenance={}
    for model in ('author_eFold','mx_bprna10','mx_mw_rank_restored','mx_nll_restored','mx_shape_fy_restored'):
        paths=sorted((FULL/'predictions'/model).glob('shard-*.jsonl'))
        for path in paths:
            marker=json.loads((path.parent/(path.stem+'-complete.json')).read_text())
            if not marker['success'] or marker['predictions_sha256']!=metrics.sha(path):raise ValueError('Saved prediction changed')
            for line in path.open():
                row=json.loads(line)
                if row['id'] in sequences:edges_by_model[model][row['id']]=row['predicted_pairs']
        provenance[model]={'source':'fixed public eFold or project checkpoint; sequence-only inference',
                          'files':{str(p):metrics.sha(p) for p in paths}}
    labels.update({'mx_bprna10':'今回のMXfold2 bpRNA事前学習10ep',
        'mx_mw_rank_restored':'今回のMXfold2 MW Rank FT5ep','mx_nll_restored':'今回のMXfold2 IMLE-NLL FT5ep',
        'mx_shape_fy_restored':'今回のMXfold2 SHAPE-fy FT5ep'})
    raw=feather.read_table(AUTHOR,columns=['reference','dataset','model','structure'],use_threads=False).to_pylist()
    for row in raw:
        if row['dataset']!='viral_fragments':continue
        model='paper_'+row['model'];reference='viral_fragments:'+row['reference']
        if reference not in sequences or reference in edges_by_model[model]:raise ValueError('Invalid/duplicate author test reference')
        edges_by_model[model][reference]=row['structure']
        labels[model]=row['model']+'（eFold論文著者の保存予測）'
        provenance[model]={'path':str(AUTHOR),'sha256':metrics.sha(AUTHOR),'source':'eFold paper author saved discrete structures'}
    frames={};individual=[];audit={}
    for model,predictions in edges_by_model.items():
        if set(predictions)!=set(sequences):raise ValueError('Incomplete viral test coverage: '+model)
        states={};checks=[]
        for reference,sequence in sequences.items():
            state,counts=paired_union(sequence,predictions[reference])
            states[metrics.sequence_hash(sequence)]=state
            checks.append({'id':reference,**counts})
        frame=metrics.profile_rows(profiles,states);frames[model]=frame
        destination=OUT/'metrics'/model;destination.mkdir(parents=True,exist_ok=True)
        frame.to_parquet(destination/'profiles.parquet',index=False)
        summaries=metrics.summarize(frame)
        individual.extend({'model':model,'label':labels[model],**r} for r in summaries)
        metrics.write_json(destination/'summary.json',{'model':model,'label':labels[model],
            'summary':summaries,'provenance':provenance[model],
            'profiles_sha256':metrics.sha(destination/'profiles.parquet')})
        audit[model]=checks
    # Check the three previously available public comparator calculations.
    prior_checks={}
    for model,old_name in [('author_eFold','author_eFold'),('paper_EternaFold','author_EternaFold'),('paper_MXFold2','author_MXFold2')]:
        old=json.loads((metrics.OUT/'metrics'/old_name/'epoch-000/summary.json').read_text())['summary']
        new=metrics.summarize(frames[model]);old={r['modifier']:r for r in old}
        for row in new:
            for metric in ('auroc_sequence_macro','spearman_sequence_macro'):
                a,b=row[metric],old[row['modifier']][metric]
                if a is None or b is None:
                    if a!=b:raise ValueError('Previously undefined metric changed')
                elif abs(a-b)>1e-12:raise ValueError('Previous new-metric calculation differs')
        prior_checks[model]=True
    groups={
        'paper_comparators':['author_eFold']+sorted(m for m in frames if m.startswith('paper_')),
        'requested_seven':['author_eFold','paper_UFold','paper_MXFold2','mx_bprna10',
                           'mx_mw_rank_restored','paper_EternaFold','paper_RNAstructure'],
        'project_finetuning':['mx_bprna10','mx_mw_rank_restored','mx_nll_restored','mx_shape_fy_restored']}
    titles={'paper_comparators':'eFold論文の比較手法（保存予測を再採点）',
            'requested_seven':'指定7モデル（公開比較予測と今回のMXfold2）','project_finetuning':'今回の事前学習後と各FT手法'}
    lines=['# eFoldデータの新AUROC・新Spearman：比較手法','',
        '対象は実測reactivityが公開されているviral RNA40配列（SHAPE15・DMS25）。'
        'PDB・ArchiveII・Long ncRNAの公開評価ファイルにはreactivityがないため、これらの整合性指標は計算できない。','',
        'AUROCは予測未対合=1・対合=0をラベル、実測reactivityを連続スコアとする。'
        'Spearmanも同じ0/1状態と実測reactivityから計算する。塩基対確率・中央値二値化・clipは使わない。'
        'DMSはA/C別、有限値かつreactivity>−10を使い、従来の新指標と同じ集計方法を使う。','',
        '全対合・全未対合などは未定義として件数を記録する。各主比較では全モデルで両指標が定義できる共通対象を使う。'
        '同一塩基に複数の予測パートナーを持つ手法も、いずれかの非対角予測塩基対に参加すれば対合とする。'
        '自己接触は塩基対に数えず、複数パートナー・自己接触の件数はprediction_audit.jsonに保存する。','']
    comparisons=[]
    for group,models in groups.items():
        matched=common_frames([frames[m] for m in models])
        summaries=[metrics.summarize(f) if not f.empty else [] for f in matched]
        entries=[(labels[m],m,10 if m=='mx_bprna10' else 5 if m.startswith('mx_') else 0) for m in models]
        text,exported=table(titles[group],entries,summaries,('SHAPE','DMS'),group,'eFold_viral')
        text=text.replace('| 0 |','| — |')
        lines.extend([text,'']);comparisons.extend(exported)
    atomic_tsv(OUT/'individual_results.tsv',pd.DataFrame(individual))
    atomic_tsv(OUT/'comparisons.tsv',pd.DataFrame(comparisons))
    metrics.write_json(OUT/'prediction_audit.json',audit)
    metrics.write_json(OUT/'previous_result_validation.json',prior_checks)
    definition=json.loads((metrics.OUT/'definition.json').read_text())
    metrics.write_json(OUT/'definition.json',definition)
    (OUT/'REPORT_ja.md').write_text('\n'.join(lines))
    metrics.write_json(OUT/'status.json',{'generated_at':datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(),
        'complete':True,'models_scored':len(frames),'viral_profiles':40,
        'comparisons':groups,'previous_results_match':prior_checks})
    print(json.dumps({'complete':True,'models':list(frames)},indent=2),flush=True)


if __name__=='__main__':main()
