#!/usr/bin/env python3
"""Reconstruct the published Archive-NoFam cohorts and score frozen predictions."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime
import csv
import json
from pathlib import Path
import re
import statistics
import sys
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).parent))
import archive_nofam as experiment

base = experiment.base
OUT, DATA = experiment.OUT, experiment.DATA
PROVIDED = ['rnastructure', 'viennarna', 'eternafold', 'mxfold2', 'spot-rna',
            'rna-fm-ResNet_bpRNA', 'rinalmo-bprna']
ORDER = ['author_eFold', 'UFold_public', 'MXfold2_public', 'mx_bprna10',
         'mx_mw_rank_restored', 'EternaFold_public', 'RNAstructure_Fold']


def published_rows(text, heading):
    block = text.split(heading + '\n', 1)[1].split('Descriptive statistics', 1)[0]
    result = {}
    for line in block.splitlines():
        match = re.match(r'^(.+?)\s+([01]\.\d{2}[†‡]?(?:\s+[01]\.\d{2}[†‡]?){6})\s*$', line.strip())
        if not match:
            continue
        name = match[1].replace('∗', '')
        result[name] = {'values': [float(x.strip('†‡')) for x in match[2].split()],
                        'bpRNA_overlap': '∗' in match[1],
                        'SPOT_overlap': bool(set('†‡') & set(match[2]))}
    if len(result) != 47:
        raise ValueError('Expected all 47 published structural groups')
    return result


def aggregate(rows, group_map):
    """Compute per-RNA metrics before averaging within and then across groups."""
    groups = defaultdict(list)
    for row in rows:
        groups[group_map[row['id']]].append(row)
    group_scores = {}
    for group, values in sorted(groups.items()):
        group_scores[group] = {'n_rnas': len(values), **{
            key: statistics.mean(0. if r[key] is None else r[key] for r in values)
            for key in ('precision', 'recall', 'f1')}}
    tp, fp, fn = (sum(r[key] for r in rows) for key in ('tp', 'fp', 'fn'))
    return {'n_rnas': len(rows), 'n_groups': len(groups),
            'group_macro': {key: statistics.mean(r[key] for r in group_scores.values())
                            for key in ('precision', 'recall', 'f1')},
            'rna_macro': {key: statistics.mean(0. if r[key] is None else r[key] for r in rows)
                          for key in ('precision', 'recall', 'f1')},
            'micro_f1': 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.,
            'undefined_precision_rnas': sum(r['precision'] is None for r in rows),
            'by_group': group_scores}


def checked_pairs(pairs, n):
    edges = set()
    for a, b in pairs:
        a, b = int(a), int(b)
        if not 0 <= a < b < n:
            raise ValueError('Invalid unordered/diagonal/out-of-bounds prediction')
        if (a, b) in edges:
            raise ValueError('Duplicate predicted contact')
        edges.add((a, b))
    return sorted(edges)


def provided_metrics(reference):
    result = {}
    for model in PROVIDED:
        predictions = base.read_jsonl(OUT / 'provided_predictions' / f'{model}.jsonl')
        if {p['id'] for p in predictions} != set(reference):
            raise ValueError('Incomplete published predictions: ' + model)
        result[model] = [{'id': p['id'], **base.pair_metrics(reference[p['id']]['pairs'], p['pairs'])}
                         for p in predictions]
    return result


def cohorts():
    base.checked_plan()
    reference = {x['id']: x for x in base.read_jsonl(OUT / 'panel.jsonl')}
    toc = (DATA / 'media-4.txt').read_text().split('=== PAGE 6 ===', 1)[0]
    names, target_codes = {}, {}
    for line in toc.splitlines():
        match = re.match(r'^\s*(\d+\.\d+)(?!\.)\s+(.+?)\s+\.\s', line)
        if match:
            names[match[1]] = match[2]
        match = re.match(r'^\s*(\d+\.\d+)\.\d+\s+([a-z0-9]{4})\s+([A-Za-z0-9]+)\s+\.', line)
        if match:
            target_codes[match[2] + '_' + match[3]] = match[1]
    if set(target_codes) != set(reference):
        raise ValueError('Supplemental File S1 did not map all targets exactly')
    original_codes = dict(target_codes)
    # The description file has 50 headings. The performance tables use 47:
    # Cas9 includes the additional BlCas9 8x5v_B; type-V/OMEGA/Fanzor guides
    # fall under Cas12. Validate this reconciliation against every published
    # per-group PPV/SEN/F1 value for all seven provided methods below.
    for identifier, code in list(target_codes.items()):
        if code == '2.2':
            target_codes[identifier] = '2.1' if identifier == '8x5v_B' else '2.3'
        elif code in ('2.5', '2.6'):
            target_codes[identifier] = '2.3'
    names['7.1'] = 'ToxI'
    group_map = {identifier: names[code] for identifier, code in target_codes.items()}
    text = (DATA / 'media-1.txt').read_text()
    tables = {key: published_rows(text, heading) for key, heading in [
        ('precision', '1.1  Positive Predictive Value'), ('recall', '1.2  Sensitivity'),
        ('f1', '1.3  F-score')]}
    if set(group_map.values()) != set(tables['f1']):
        raise ValueError('Reconciled target groups do not match the result tables')
    bp_groups = {k for k, v in tables['f1'].items() if v['bpRNA_overlap']}
    spot_groups = {k for k, v in tables['f1'].items() if v['SPOT_overlap']}
    cohort_groups = {'full89': set(group_map.values()),
                     'paper_bprna_filtered': set(group_map.values()) - bp_groups,
                     'literal_caption_union': set(group_map.values()) - bp_groups - spot_groups}
    definitions = {name: {'groups': sorted(groups), 'ids': sorted(k for k, g in group_map.items() if g in groups)}
                   for name, groups in cohort_groups.items()}
    if [(len(definitions[k]['ids']), len(definitions[k]['groups'])) for k in definitions] != [(89, 47), (79, 40), (44, 22)]:
        raise ValueError('Unexpected published-filter cohort sizes')
    supplied = provided_metrics(reference)
    comparisons = []
    max_group_error = 0.
    for col, model in enumerate(PROVIDED):
        full = aggregate(supplied[model], group_map)
        for group, values in full['by_group'].items():
            for metric in tables:
                expected = tables[metric][group]['values'][col]
                difference = abs(values[metric] - expected)
                max_group_error = max(max_group_error, difference)
                if difference > .005001:
                    raise ValueError(f'Published per-group result mismatch: {model}/{group}/{metric}: {values[metric]} vs {expected}')
        # Table S4-S6 actually use bpRNA exclusions for six methods and the
        # union with SPOT exclusions only for SPOT-RNA. A literal reading of
        # the caption would use the union for every method and fails to
        # reproduce the published MXfold2/EternaFold/RNAstructure means.
        subset = 'literal_caption_union' if model == 'spot-rna' else 'paper_bprna_filtered'
        selection = set(definitions[subset]['ids'])
        scored = aggregate([r for r in supplied[model] if r['id'] in selection], group_map)
        mean_expected = {'precision': [.741, .731, .742, .754, .701, .732, .665],
                         'recall': [.758, .757, .792, .746, .726, .708, .488],
                         'f1': [.740, .735, .758, .742, .702, .710, .544]}
        for metric, values in mean_expected.items():
            expected = values[col]
            observed = scored['group_macro'][metric]
            if abs(observed - expected) > .000501:
                raise ValueError(f'Published filtered mean mismatch: {model}/{metric}: {observed} vs {expected}')
            comparisons.append({'model': model, 'metric': metric, 'cohort': subset,
                                'observed': observed, 'published_rounded': expected})
    metadata = {'group_map': group_map, 'description_section': original_codes,
                'bpRNA_overlap_groups': sorted(bp_groups), 'SPOT_overlap_groups': sorted(spot_groups),
                'cohorts': definitions,
                'primary': 'paper_bprna_filtered',
                'primary_rationale': 'Reproduces the published Table S6 MXfold2/EternaFold/RNAstructure means. SPOT-RNA alone has further exclusions in the actual published calculations; none of the seven requested models is SPOT-RNA.',
                'group_reconciliation': '50 description headings reconciled to 47 table groups using guide identities; all 987 published rounded group-metric values validated.',
                'source_sha256': {str(DATA / name): base.sha(DATA / name)
                                 for name in ['media-1.pdf', 'media-1.txt', 'media-4.pdf', 'media-4.txt']}}
    base.write_json(OUT / 'cohorts.json', metadata)
    base.write_json(OUT / 'published_validation.json', {'success': True,
        'per_group_metrics_checked': 47*7*3, 'max_absolute_group_difference_vs_rounded': max_group_error,
        'filtered_means': comparisons, 'note': metadata['primary_rationale']})
    base.write_json(OUT / 'cohorts-complete.json', {'success': True, 'cohorts_sha256': base.sha(OUT / 'cohorts.json')})
    print(json.dumps({'success': True, 'cohorts': {k: [len(v['ids']), len(v['groups'])] for k, v in definitions.items()},
                      'published_filtered_means': comparisons}, indent=2), flush=True)


def write_tsv(path, rows):
    with path.open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)


def report():
    experiment.check_adapters()
    plan = base.checked_plan()
    marker = json.loads((OUT / 'cohorts-complete.json').read_text())
    if not marker['success'] or marker['cohorts_sha256'] != base.sha(OUT / 'cohorts.json'):
        raise ValueError('Published cohort reconstruction was not validated')
    cohort = json.loads((OUT / 'cohorts.json').read_text())
    reference = {x['id']: x for x in base.read_jsonl(OUT / 'panel.jsonl')}
    specs = {m['model']: m for m in plan['models'] + plan['external_models']}
    all_scores, individual, group_export, audit, native_predictions = [], [], [], {}, {}
    for name in ORDER:
        folder = OUT / 'predictions' / name
        path = folder / 'shard-0.jsonl'
        done = json.loads((folder / 'shard-0-complete.json').read_text())
        if not done['success'] or done['n'] != 89 or done['predictions_sha256'] != base.sha(path):
            raise ValueError('Incomplete or changed prediction: ' + name)
        predictions = base.read_jsonl(path)
        if len(predictions) != 89 or {r['id'] for r in predictions} != set(reference):
            raise ValueError('Incomplete target coverage: ' + name)
        rows, degree_audit = [], []
        native_predictions[name] = {}
        for pred in predictions:
            item = reference[pred['id']]
            if pred['sequence_sha256'] != item['sequence_sha256'] or pred['length'] != len(item['sequence']):
                raise ValueError('Prediction sequence changed')
            pairs = checked_pairs(pred['predicted_pairs'], len(item['sequence']))
            native_predictions[name][item['id']] = pairs
            value = base.pair_metrics(item['pairs'], pairs)
            if abs(pred['f1'] - value['f1']) > 1e-12:
                raise ValueError('Worker/report F1 mismatch')
            row = {'model': name, 'id': item['id'], 'group': cohort['group_map'][item['id']],
                   'length': len(item['sequence']), **value}
            rows.append(row)
            degree = defaultdict(int)
            for a, b in pairs:
                degree[a] += 1; degree[b] += 1
            degree_audit.append({'id': item['id'], 'multiplet_bases': sum(v > 1 for v in degree.values())})
        individual.extend(rows)
        audit[name] = degree_audit
        for subset, selection in cohort['cohorts'].items():
            ids = set(selection['ids'])
            scores = aggregate([r for r in rows if r['id'] in ids], cohort['group_map'])
            all_scores.append({'model': name, 'label': specs[name]['label'], 'cohort': subset,
                'n_rnas': scores['n_rnas'], 'n_groups': scores['n_groups'],
                'group_precision': scores['group_macro']['precision'],
                'group_recall': scores['group_macro']['recall'], 'group_f1': scores['group_macro']['f1'],
                'rna_precision': scores['rna_macro']['precision'], 'rna_recall': scores['rna_macro']['recall'],
                'rna_f1': scores['rna_macro']['f1'], 'micro_f1': scores['micro_f1'],
                'undefined_precision_rnas_as_zero': scores['undefined_precision_rnas']})
            group_export.extend({'model': name, 'cohort': subset, 'group': group, **metrics}
                                for group, metrics in scores['by_group'].items())
    supplied_agreement = {}
    for name, supplied_name in [('MXfold2_public', 'mxfold2'), ('EternaFold_public', 'eternafold'),
                                 ('RNAstructure_Fold', 'rnastructure')]:
        supplied = base.read_jsonl(OUT / 'provided_predictions' / f'{supplied_name}.jsonl')
        different = [p['id'] for p in supplied if sorted(map(tuple, p['pairs'])) != native_predictions[name][p['id']]]
        supplied_agreement[name] = {'different_structures': len(different), 'ids': different,
            'note': 'Fresh local public model inference compared to author-supplied NoFam structures; version/settings differences are retained and disclosed.'}
    validation = json.loads((OUT / 'published_validation.json').read_text())
    write_tsv(OUT / 'results.tsv', all_scores)
    write_tsv(OUT / 'per_rna.tsv', individual)
    write_tsv(OUT / 'per_group.tsv', group_export)
    base.write_json(OUT / 'prediction_audit.json', audit)
    base.write_json(OUT / 'author_prediction_agreement.json', supplied_agreement)
    lines = ['# Archive-NoFam：指定7モデルの評価', '',
        'MXfold2原著モデルは、指定どおり2021年公開TrainSetABを使用した。今回のモデルはbpRNA事前学習10epと、そこからモデル・optimizer状態を復元したMW Rank FT5ep（replay20%、熱力学的正則化0.125）。全7モデルを同じ89配列に対して新規推論した。', '',
        '入力は著者提供Archive-NoFamのCT89構造（21〜339nt）。正解塩基対を追加剪定せず、完全一致のPPV・SEN・F1を計算した。各RNAのスコアをグループ内で平均し、各グループを等重みで平均した。予測塩基対が空の場合のPPVは0として論文の数値に合わせた。RNA単位macro・microもTSVに保存した。', '',
        '補足資料はNIH PMCに公開された著者プレプリントv1（PMC12637522.1）から取得した。検証した補足表S1〜S6もこのプレプリント添付版である。最終掲載版（DOI:10.1261/rna.080846.125）は同じ89構造・47グループと記載され、本文のEternaFold・MXfold2・SPOT-RNA・RNA-FM・RiNALMoのF1も再現値と一致する。最終掲載版の補足ファイルとのバイト単位の同一性は確認していない。資料の版とハッシュはplan.json・downloads.jsonに記録している。', '',
        '本文・図注の「全重複を除外」という表現には注意が必要である。実際の補足表S6は、SPOT-RNA以外ではbpRNA重複7グループを除いた79配列・40グループ、SPOT-RNAでは追加重複を除いた44配列・22グループで再現できた。指定7モデルにはSPOT-RNAが含まれないため、主比較は全モデル共通の79配列・40グループとした。図注を文字どおり全モデルに適用した44配列版もresults.tsvに別集計した。', '']
    titles = {'paper_bprna_filtered': '論文の数値を再現する共通対象：79配列・40グループ',
              'full89': '全89配列・47グループ（重複を含む参考値）',
              'literal_caption_union': '図注の全重複除外を一律適用：44配列・22グループ（補足）'}
    for subset in ['paper_bprna_filtered', 'full89', 'literal_caption_union']:
        lines.extend(['## ' + titles[subset], '', '| モデル | PPV（群平均） | SEN（群平均） | F1（群平均） | F1（RNA平均） |',
                      '|---|---:|---:|---:|---:|'])
        for row in (r for r in all_scores if r['cohort'] == subset):
            lines.append(f'| {row["label"]} | {row["group_precision"]:.4f} | {row["group_recall"]:.4f} | {row["group_f1"]:.4f} | {row["rna_f1"]:.4f} |')
        lines.append('')
    lines.extend(['## 著者提供予測による集計方法の検証', '',
                  '47グループ×7手法×3指標の987値が補足表の丸め値と一致した。除外後の21個の平均値も補足表S4〜S6の小数3桁まで一致した。記述資料の50見出しは、ガイドRNAの種類を結果表のCas9・Cas12・Cas13の3群へ整理し、47群として再現した。', '',
                  '| 手法 | 提供予測を再集計したF1 | 論文F1（丸め） |', '|---|---:|---:|'])
    for row in validation['filtered_means']:
        if row['metric'] == 'f1':
            lines.append(f'| {row["model"]} | {row["observed"]:.6f} | {row["published_rounded"]:.3f} |')
    lines.extend(['', '新規推論と著者提供予測が異なる構造数：' + '、'.join(f'{k} {v["different_structures"]}/89' for k, v in supplied_agreement.items()) + '。この比較表の主結果はすべて新規推論を採用した。', '',
        'bpRNA実訓練データとeFold公開訓練データの完全一致は同じ5配列で、いずれも主比較の除外対象に含まれる。化学プロービングFT訓練データとの完全一致は0配列。配列完全一致がないことは、より広い構造・配列相同性がないことを意味しない。詳細はtraining_exact_overlap.jsonを参照。', '',
        'Archive-NoFamの89 CTには実測reactivityがないため、ここでは構造F1を評価した。eFoldの化学整合性（新AUROC・新Spearman）は別レポート paper_methods_mw_20261031/REPORT_ja.md に掲載している。', '',
        '論文：[RNA 32(4):428](https://rnajournal.cshlp.org/content/32/4/428)。入力資料：NIH NLM NCBI PMC Article Datasets（ダウンロード元はdata/efold_hf/archive_nofam_20261031/downloads.json）。'])
    (OUT / 'REPORT_ja.md').write_text('\n'.join(lines) + '\n')
    base.write_json(OUT / 'implementation.json', {str(Path(__file__)): base.sha(Path(__file__))})
    base.write_json(OUT / 'status.json', {'complete': True, 'models': ORDER, 'n_per_model': 89,
        'primary_cohort': 'paper_bprna_filtered', 'primary_n': 79, 'primary_groups': 40,
        'published_validation': True, 'generated_at': datetime.now(ZoneInfo('Asia/Tokyo')).isoformat()})
    print(json.dumps({'complete': True, 'models': 7, 'primary_n': 79, 'primary_groups': 40}), flush=True)


if __name__ == '__main__':
    if sys.argv[1] == 'cohorts':
        cohorts()
    elif sys.argv[1] == 'report':
        report()
    else:
        raise ValueError(sys.argv[1])
