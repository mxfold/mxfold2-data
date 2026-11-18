#!/usr/bin/env python3
"""Audit the released RNAcentral source and its membership in eFold training."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import gc
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).parent))
from bprnanew_benchmark import ROOT, RAW, normalize, read_jsonl, sha, write_json, write_jsonl

OUT = ROOT/'qsub_logs/efold_hf/rnacentral_overlap_20261030'
PREVIOUS = ROOT/'qsub_logs/efold_hf/bprnanew_20261029'
SHARDS = 8
COHORTS = ('source_all', 'efold_train_retained')
DATASETS = ('PDB', 'archiveII', 'archiveII_original_313')


def write_tsv(path, rows):
    import csv
    if not rows:
        return
    with path.open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)


def source_targets(source, final_references):
    """Identify source membership by exact normalized sequence, recording IDs too."""
    final_sequences = defaultdict(list)
    for reference, sequence in final_references.items():
        final_sequences[sequence].append(reference)
    targets = []
    for reference, item in source.items():
        sequence = normalize(item['sequence'])
        if not sequence or set(sequence)-set('ACGU'):
            raise ValueError('Unexpected RNAcentral alphabet: '+reference)
        if len(sequence) > 1000:
            continue
        references = final_sequences.get(sequence, [])
        targets.append({'id': reference, 'sequence': sequence,
                        'in_final_training': bool(references),
                        'same_id_and_sequence_in_final': final_references.get(reference)==sequence,
                        'final_training_references': references})
    return targets


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    source_file = RAW/'rnacentral_synthetic.json'
    train_file = RAW/'efold_train.json'
    release_files = json.loads((RAW/'rnacentral_synthetic.files.json').read_text())
    source_release = next(item for item in release_files if item['path']=='data.json')
    source_sha256 = sha(source_file)
    if source_sha256!=source_release['lfs']['oid'] or source_file.stat().st_size!=source_release['size']:
        raise ValueError('RNAcentral source differs from its official LFS SHA256 or size')
    full = json.loads(train_file.read_text())
    final_references = {key: normalize(value['sequence']) for key, value in full.items()}
    del full
    gc.collect()
    source = json.loads(source_file.read_text())
    if len(source) != 226729 or len(final_references) != 306557:
        raise ValueError('Public source or eFold training snapshot differs')
    raw_n = len(source)
    targets = source_targets(source, final_references)
    del source
    gc.collect()
    queries = [{key: value for key, value in q.items()
                if key in ('id', 'source', 'sequence', 'in_original_conservative_panel')}
               for q in read_jsonl(PREVIOUS/'similarity_queries.jsonl')]
    if len(queries) != 3714 or len({q['id'] for q in queries}) != len(queries):
        raise ValueError('Prior PDB/ArchiveII evaluation cohort changed')
    write_jsonl(OUT/'targets.jsonl', targets)
    write_jsonl(OUT/'queries.jsonl', queries)
    by_sequence = defaultdict(list)
    for target in targets:
        by_sequence[target['sequence']].append(target)
    exact = []
    for query in queries:
        for cohort in COHORTS:
            hits = [r for r in by_sequence.get(query['sequence'], [])
                    if cohort=='source_all' or r['in_final_training']]
            exact.append({'id': query['id'], 'source': query['source'], 'cohort': cohort,
                          'length': len(query['sequence']), 'exact_match': bool(hits),
                          'matching_target_ids': [r['id'] for r in hits]})
    write_jsonl(OUT/'exact_matches.jsonl', exact)
    metadata = json.loads((RAW/'rnacentral_synthetic.metadata.json').read_text())
    train_metadata = json.loads((RAW/'efold_train.metadata.json').read_text())
    selected = [r for r in targets if r['in_final_training']]
    plan = {
        'generated_at': datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(),
        'source_url': 'https://huggingface.co/datasets/rouskinlab/rnacentral_synthetic',
        'source_revision': metadata['sha'], 'efold_train_revision': train_metadata['sha'],
        'source_verified_official_lfs_sha256': source_sha256,
        'source_raw_n': raw_n, 'source_le1000_n': len(targets),
        'source_le1000_unique_sequences': len(by_sequence),
        'excluded_source_over1000': raw_n-len(targets),
        'retained_in_efold_train_n': len(selected),
        'retained_unique_sequences': len({r['sequence'] for r in selected}),
        'retained_same_reference_and_sequence': sum(r['same_id_and_sequence_in_final'] for r in selected),
        'retained_sequence_with_different_reference': sum(not r['same_id_and_sequence_in_final'] for r in selected),
        'source_length_range': [min(len(r['sequence']) for r in targets), max(len(r['sequence']) for r in targets)],
        'membership': 'Exact normalized sequence membership in the frozen public efold_train; original reference ID agreement is recorded separately.',
        'cohorts': list(COHORTS), 'max_length': 1000, 'global_shards': SHARDS,
        'queries': {name: sum(q['source']==name for q in queries) for name in ('PDB','archiveII')},
        'archiveII_note': 'Full public ArchiveII <=1000nt (3359) is the main cohort. The paper figure evaluates 3370; the Table2 archiveII_filtered count355 is a different count and is not the actual figure panel. The prior conservative313 panel is supplemental.',
        'global_protocol': 'Exhaustive forward-strand nearest normalized Levenshtein similarity, 1-distance/max(lengths); not alignment identity.',
        'local_protocol': 'MMseqs2 forward nucleotide local search, query coverage>=80%; additionally require both coverages>=80%. Heuristic sensitivity7.5; no hit is undefined.',
        'input_sha256': {str(p): sha(p) for p in (source_file, train_file, PREVIOUS/'similarity_queries.jsonl')},
        'targets_sha256': sha(OUT/'targets.jsonl'), 'queries_sha256': sha(OUT/'queries.jsonl'),
        'job': os.environ.get('JOB_ID'), 'host': socket.gethostname(),
    }
    write_json(OUT/'plan.json', plan)
    write_json(OUT/'prepare-complete.json', {'success': True, 'targets_sha256': plan['targets_sha256']})
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)


def checked_inputs():
    plan = json.loads((OUT/'plan.json').read_text())
    marker = json.loads((OUT/'prepare-complete.json').read_text())
    if not marker['success'] or sha(OUT/'targets.jsonl')!=plan['targets_sha256'] or sha(OUT/'queries.jsonl')!=plan['queries_sha256']:
        raise ValueError('Prepared sequence inputs changed')
    return plan, read_jsonl(OUT/'targets.jsonl'), read_jsonl(OUT/'queries.jsonl')


def nearest_in_cohort(query, targets, indices, similarities):
    import numpy as np
    from rapidfuzz.distance import Levenshtein
    values = similarities[indices]
    index = int(indices[int(np.argmax(values))])
    target = targets[index]
    score = float(np.max(values))
    distance = Levenshtein.distance(query['sequence'], target['sequence'])
    if abs(score-(1-distance/max(len(query['sequence']),len(target['sequence']))))>1e-12:
        raise ValueError('Edit similarity normalization differs')
    return {'best_target_id': target['id'], 'best_target_length': len(target['sequence']),
            'global_edit_similarity': score, 'edit_distance': distance,
            'ties_at_maximum': int((values==score).sum())}


def global_search(index):
    import numpy as np
    from rapidfuzz import process
    from rapidfuzz.distance import Levenshtein
    plan, targets, queries = checked_inputs()
    selected = [q for i,q in enumerate(queries) if i%SHARDS==index-1]
    sequences = [r['sequence'] for r in targets]
    indices = {'source_all': np.arange(len(targets)),
               'efold_train_retained': np.flatnonzero([r['in_final_training'] for r in targets])}
    rows = []
    start = time.monotonic()
    for offset in range(0,len(selected),16):
        batch = selected[offset:offset+16]
        matrix = process.cdist([q['sequence'] for q in batch], sequences,
                               scorer=Levenshtein.normalized_similarity, dtype=np.float64, workers=4)
        for query, scores in zip(batch,matrix):
            for cohort in COHORTS:
                rows.append({'id': query['id'], 'source': query['source'], 'cohort': cohort,
                             'length': len(query['sequence']),
                             'in_original_conservative_panel': query['in_original_conservative_panel'],
                             **nearest_in_cohort(query,targets,indices[cohort],scores)})
        print('global',index,min(offset+16,len(selected)),'/',len(selected),f'{time.monotonic()-start:.1f}s',flush=True)
    path = OUT/'global'/f'shard-{index}.jsonl'
    write_jsonl(path,rows)
    write_json(path.with_name(f'shard-{index}-complete.json'),
               {'success': True, 'n_queries': len(selected), 'n_rows': len(rows),
                'predictions_sha256': sha(path), 'targets_sha256': plan['targets_sha256'],
                'seconds': time.monotonic()-start, 'job': os.environ.get('JOB_ID'), 'host': socket.gethostname()})


def local_search():
    plan, targets, queries = checked_inputs()
    folder = OUT/'local'
    folder.mkdir(exist_ok=True)
    for name, rows, prefix in [('queries',queries,'Q'),('targets',targets,'T')]:
        with (folder/f'{name}.fasta').open('w') as handle:
            for i,row in enumerate(rows):
                handle.write(f'>{prefix}{i:06d}\n'+row['sequence'].replace('U','T')+'\n')
    binary = ROOT/'.cache/rnagym-tools/mmseqs/bin/mmseqs'
    fields = 'query,target,pident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits,qcov,tcov'
    command = [str(binary),'easy-search',str(folder/'queries.fasta'),str(folder/'targets.fasta'),
               str(folder/'hits.tsv'),str(folder/'tmp'),'--search-type','3','--min-seq-id','0.6',
               '-c','0.8','--cov-mode','2','--alignment-mode','3','--max-seqs','50000',
               '--threads',os.environ.get('NSLOTS','4'),'--strand','1','--mask','0','-e','1000','-s','7.5',
               '--split-memory-limit','12G','--format-output',fields]
    start = time.monotonic()
    subprocess.run(command,check=True)
    write_json(folder/'complete.json',{'success': True, 'command': command, 'format': fields,
               'binary_sha256': sha(binary), 'hits_sha256': sha(folder/'hits.tsv'),
               'targets_sha256': plan['targets_sha256'], 'seconds': time.monotonic()-start,
               'job': os.environ.get('JOB_ID'), 'host': socket.gethostname()})


def selected_dataset(row, dataset):
    return row['source']==dataset or (dataset=='archiveII_original_313' and
           row['source']=='archiveII' and row['in_original_conservative_panel'])


def local_rows(targets, queries, hits):
    best, both, containment, paper_cut = {}, {}, set(), set()
    for line in hits:
        values = line.split()
        qi,ti = int(values[0][1:]),int(values[1][1:])
        query,target = queries[qi],targets[ti]
        identity,qcov,tcov = float(values[2]),float(values[12]),float(values[13])
        if qcov<.8:
            continue
        row = {'pident': identity, 'qcov': qcov, 'tcov': tcov,
               'bits': float(values[11]), 'evalue': float(values[10]),
               'best_target_id': target['id'], 'best_target_length': len(target['sequence'])}
        rank = (identity,qcov,row['bits'])
        for cohort in COHORTS:
            if cohort=='efold_train_retained' and not target['in_final_training']:
                continue
            key = (qi,cohort)
            if key not in best or rank>best[key][0]:best[key]=(rank,row)
            if tcov>=.8 and (key not in both or rank>both[key][0]):both[key]=(rank,row)
            if identity>=80 and int(values[3])>112:paper_cut.add(key)
            if identity==100 and qcov==1 and int(values[4])==0 and int(values[5])==0:
                if query['sequence'] not in target['sequence']:
                    raise ValueError('Putative exact containment is not an actual substring')
                containment.add(key)
    rows = []
    for qi,query in enumerate(queries):
        for cohort in COHORTS:
            key = (qi,cohort)
            item = {'id': query['id'], 'source': query['source'], 'cohort': cohort,
                    'length': len(query['sequence']),
                    'in_original_conservative_panel': query['in_original_conservative_panel'],
                    'full_query_identical_substring': key in containment,
                    'alignment_gt112_identity_ge80': key in paper_cut}
            for prefix,table in [('',best),('both_cov80_',both)]:
                for name in ('pident','qcov','tcov','bits','evalue','best_target_id','best_target_length'):
                    item[prefix+name] = table[key][1][name] if key in table else None
            rows.append(item)
    return rows


def report():
    import numpy as np
    plan,targets,queries = checked_inputs()
    global_rows = []
    for shard in range(1,SHARDS+1):
        path = OUT/'global'/f'shard-{shard}.jsonl'
        marker = json.loads(path.with_name(f'shard-{shard}-complete.json').read_text())
        if not marker['success'] or sha(path)!=marker['predictions_sha256'] or marker['targets_sha256']!=plan['targets_sha256']:
            raise ValueError('Invalid global result shard')
        global_rows.extend(read_jsonl(path))
    expected = {(q['id'],c) for q in queries for c in COHORTS}
    if len(global_rows)!=len(expected) or {(r['id'],r['cohort']) for r in global_rows}!=expected:
        raise ValueError('Global comparisons are incomplete')
    local_marker = json.loads((OUT/'local/complete.json').read_text())
    if not local_marker['success'] or sha(OUT/'local/hits.tsv')!=local_marker['hits_sha256']:
        raise ValueError('Local hit table changed')
    with (OUT/'local/hits.tsv').open() as handle:
        local = local_rows(targets,queries,handle)
    exact = {(r['id'],r['cohort']):r for r in read_jsonl(OUT/'exact_matches.jsonl')}
    summary = []
    for cohort in COHORTS:
        for dataset in DATASETS:
            g = [r for r in global_rows if r['cohort']==cohort and selected_dataset(r,dataset)]
            l = [r for r in local if r['cohort']==cohort and selected_dataset(r,dataset)]
            values = np.asarray([r['global_edit_similarity'] for r in g])
            record = {'cohort': cohort, 'dataset': dataset, 'n': len(g),
                      'global_mean': float(values.mean()), 'global_median': float(np.median(values)),
                      'global_min': float(values.min()), 'global_max': float(values.max()),
                      'full_sequence_exact': sum(exact[(r['id'],cohort)]['exact_match'] for r in g),
                      **{f'global_ge_{cut}':int((values>=cut/100).sum()) for cut in (80,90,95,99,100)},
                      'local_queries_with_hit': sum(r['pident'] is not None for r in l),
                      'full_query_identical_substring': sum(r['full_query_identical_substring'] for r in l),
                      'alignment_gt112_identity_ge80': sum(r['alignment_gt112_identity_ge80'] for r in l)}
            for cut in (80,90,95,99,100):
                for prefix in ('','both_cov80_'):
                    record[prefix+f'local_identity_ge_{cut}'] = sum(r[prefix+'pident'] is not None and r[prefix+'pident']>=cut for r in l)
            if record['global_ge_100']!=record['full_sequence_exact']:
                raise ValueError('Exhaustive global exact matches differ from direct string matching')
            summary.append(record)
    write_tsv(OUT/'train_test_nearest_global.tsv',global_rows)
    write_tsv(OUT/'train_test_nearest_local.tsv',local)
    write_tsv(OUT/'summary.tsv',summary)
    write_json(OUT/'summary.json',{'plan': plan, 'summary': summary})
    lines=['# eFold RNAcentral訓練配列とPDB・ArchiveIIの配列類似度','',
           '集計日時：'+datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(),' ',
           'eFoldはRNAcentral由来配列にRNAstructure Foldで構造を付与したデータを事前学習に使用する。'
           '公開元rnacentral_syntheticは226,729配列。今回のMXfold2の学習には使用していない。', '',
           f"1000nt以下の公開元は{plan['source_le1000_n']:,}配列。そのうち、公開efold_trainに同じ配列が存在するのは"
           f"{plan['retained_in_efold_train_n']:,}配列（同じ参照IDかつ同じ配列：{plan['retained_same_reference_and_sequence']:,}）。",'',
           '主な比較対象は前回と同じPDB 355配列・ArchiveII 3,359配列（1000nt以下）。'
           '以前の保守的除去後ArchiveII 313配列を別行で示す。論文の実際の図の対象は3,370配列であり、Table 2のarchiveII_filtered 355配列とは区別する。','',
           '## 最終eFold訓練集合に含まれるRNAcentral配列','',
           '| テスト集合 | 配列数 | 全長編集類似度の中央値 | 全長完全一致 | 局所一致率≥80% | ≥90% | ≥95% |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        if r['cohort']=='efold_train_retained':
            lines.append(f"| {r['dataset']} | {r['n']} | {r['global_median']:.4f} | {r['full_sequence_exact']} | {r['both_cov80_local_identity_ge_80']} | {r['both_cov80_local_identity_ge_90']} | {r['both_cov80_local_identity_ge_95']} |")
    lines+=['','局所一致率はMMseqs2のアラインメントにおけるpercent identityで、両配列のcoverage≥80%を条件とする。'
            '全長編集類似度は1−Levenshtein編集距離/max(両配列長)であり、全長の全組合せを総当たりした最大値。'
            'アラインメント一致率とは異なる。いずれも公開RNA配列の順方向を使う。','',
            '## 公開元全体と最終訓練集合の比較','',
            '| 配列集合 | テスト集合 | 全長完全一致 | query coverage≥80%・一致率≥90% | クエリ全長の完全包含 |',
            '|---|---|---:|---:|---:|']
    for r in summary:
        lines.append(f"| {r['cohort']} | {r['dataset']} | {r['full_sequence_exact']} | {r['local_identity_ge_90']} | {r['full_query_identical_substring']} |")
    lines+=['','全長完全一致は両配列の全体が一致する例。完全包含はテスト配列全長が訓練配列の一部と一致する例で、全長完全一致も含む。'
            '局所検索はヒューリスティックで、ヒットなしを類似度0とは扱わない。','',
            '論文はBLASTによる類似配列除去を記載しているが、今回のMMseqs2検索はそのBLAST実行の再現ではない。'
            '現在公開されているテスト集合の大きさも論文記載の集合と異なるため、この結果だけで論文のテストへの混入を断定しない。'
            'alignment長>112ntかつ一致率≥80%のヒット件数もsummary.tsvに別途記録した。','',
            'RNAごとの最近傍ID・一致率・coverageはtrain_test_nearest_local.tsv、全長編集距離はtrain_test_nearest_global.tsv。'
            '全長完全一致のID対応はexact_matches.jsonl。入力データの固定版・SHA256・コマンドはplan.jsonとlocal/complete.json。','',
            '出典：[eFold論文](https://pmc.ncbi.nlm.nih.gov/articles/PMC12935039/)、'
            '[RNAcentral合成データ](https://huggingface.co/datasets/rouskinlab/rnacentral_synthetic)、'
            '[最終eFold訓練集合](https://huggingface.co/datasets/rouskinlab/efold_train)。','']
    temporary = OUT/'.REPORT_ja.md.tmp'
    temporary.write_text('\n'.join(lines))
    temporary.replace(OUT/'REPORT_ja.md')
    write_json(OUT/'status.json',{'generated_at': datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(),
               'complete': True, 'global_queries': len(queries), 'global_cohort_rows': len(global_rows),
               'local_complete': True, 'source_all_n': len(targets),
               'efold_train_retained_n': plan['retained_in_efold_train_n']})
    print(json.dumps(summary,indent=2,ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare','global','local','report'])
    parser.add_argument('--index',type=int)
    args = parser.parse_args()
    raw = os.environ.get('SGE_TASK_ID','1')
    index = args.index or (int(raw) if raw.isdigit() else 1)
    if args.action=='prepare':prepare()
    elif args.action=='global':global_search(index)
    elif args.action=='local':local_search()
    else:report()
