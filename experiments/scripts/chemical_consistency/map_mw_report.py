#!/usr/bin/env python3
"""Dataset-specific matched-epoch tables, with common-observation coverage."""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.chemical_consistency.map_mw_metrics import OUT, METRICS, summarize, write_json


def fmt(value):
    return '—' if value is None or pd.isna(value) else f'{float(value):.4f}'


def merge_shard_summaries(pieces):
    """Exact sequence macro: each unique sequence belongs to only one shard."""
    answer = []
    modifiers = sorted({s['modifier'] for p in pieces for s in p})
    for modifier in modifiers:
        rows = [s for p in pieces for s in p if s['modifier'] == modifier]
        row = {'dataset': rows[0]['dataset'], 'modifier': modifier}
        for key in ('profiles', 'units', 'strata', 'n_bases', 'n_collapsed_strata', 'n_insufficient_strata'):
            row[key] = sum(s[key] for s in rows)
        for metric in METRICS:
            count = 'n_units_' + metric + '_defined'
            n = sum(s[count] for s in rows)
            row[count] = n
            row['n_strata_' + metric + '_defined'] = sum(s['n_strata_' + metric + '_defined'] for s in rows)
            row[metric + '_sequence_macro'] = (sum(s[metric + '_sequence_macro'] * s[count]
                                                 for s in rows if s[count]) / n if n else None)
        answer.append(row)
    return answer


def read_index():
    index, rows = {}, []
    for path in sorted((OUT / 'metrics').glob('*/epoch-*/summary.json')):
        info = json.loads(path.read_text())
        aliases = info['provenance'].get('aliases', [info['model']])
        for alias in aliases:
            key = alias, info['epoch']
            if key in index and index[key] != path:
                raise ValueError('Duplicate evaluation alias: ' + str(key))
            index[key] = path
            rows.extend({'model': alias, 'epoch': info['epoch'], **item} for item in info['summary'])
    return index, rows


def common_frames(frames):
    keys = ['profile_id', 'stratum']
    indexed = [f.set_index(keys, verify_integrity=True) for f in frames]
    common = None
    for frame in indexed:
        valid = frame.auroc.notna() & frame.spearman.notna()
        good = frame.index[valid]
        common = good if common is None else common.intersection(good)
    return [f.loc[common].reset_index() for f in indexed]


def comparison(index, title, dataset, entries, modifiers, comparison_id):
    missing = [(model, epoch) for _, model, epoch in entries if (model, epoch) not in index]
    if missing:
        return f'{title}：再評価中（未完了 {len(missing)} 条件）。', []
    frames = [pd.read_parquet(index[(model, epoch)].parent / 'profiles.parquet') for _, model, epoch in entries]
    frames = [f[f.dataset == dataset] for f in frames]
    if any(f.empty for f in frames):
        return f'{title}：対象データセットの再評価待ち。', []
    matched = common_frames(frames)
    summaries = [summarize(f) if not f.empty else [] for f in matched]
    return table(title, entries, summaries, modifiers, comparison_id, dataset)


def table(title, entries, summaries, modifiers, comparison_id, dataset):
    lines = [title, '', '| モデル | Epoch | ' + ' | '.join(p + ' AUROC / Spearman' for p in modifiers) + ' |',
             '|---|---:|' + '---:|' * len(modifiers)]
    exported = []
    baseline = {s['modifier']: s for s in summaries[0]} if summaries else {}
    counts = {}
    for (label, model, epoch), rows in zip(entries, summaries):
        by_probe = {s['modifier']: s for s in rows}
        cells = []
        for probe in modifiers:
            row = by_probe.get(probe)
            cells.append('— / —' if row is None else
                         fmt(row['auroc_sequence_macro']) + ' / ' + fmt(row['spearman_sequence_macro']))
            if row:
                base = baseline[probe]
                exported.append({'comparison': comparison_id, 'label': label, 'model': model,
                                 'epoch': epoch, **row,
                                 'delta_auroc': row['auroc_sequence_macro'] - base['auroc_sequence_macro'],
                                 'delta_spearman': row['spearman_sequence_macro'] - base['spearman_sequence_macro']})
                counts[probe] = (row['units'], row['profiles'], row['strata'])
        epoch_label = '—' if model.startswith(('EternaFold_', 'author_')) else str(epoch)
        lines.append('| ' + label + ' | ' + epoch_label + ' | ' + ' | '.join(cells) + ' |')
    lines.extend(['', '共通評価対象：' + '、'.join(f'{p} {u} RNA/試料・{n}測定・{s}塩基種層'
                                                    for p, (u, n, s) in counts.items()) + '。'])
    if not counts:
        lines[-1] = '全モデルに共通してAUROCとSpearmanを定義できる観測群はありません。個別の定義可能件数は全結果TSVに記録しています。'
    return '\n'.join(lines), exported


def full_comparison(title, labels, comparison_id):
    paths = {label: [OUT / 'full' / label / f'shard-{s:02d}' for s in range(1, 17)] for _, label, _ in labels}
    if any(not (p / 'summary.json').exists() for items in paths.values() for p in items):
        n = sum((p / 'summary.json').exists() for items in paths.values() for p in items)
        return f'{title}：再評価中（{n}/{len(labels)*16} シャード）。', []
    pieces = [[] for _ in labels]
    for shard in range(16):
        frames = [pd.read_parquet(paths[label][shard] / 'profiles.parquet') for _, label, _ in labels]
        matched = common_frames(frames)
        for output, frame in zip(pieces, matched):
            output.append(summarize(frame) if not frame.empty else [])
    summary = [merge_shard_summaries(p) for p in pieces]
    return table(title, labels, summary, ('2A3', 'DMS'), comparison_id, 'RNAgym_full_test')


def atomic_tsv(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    frame.to_csv(temp, sep='\t', index=False, float_format='%.8f')
    temp.replace(path)


def main():
    index, rows = read_index()
    panel_plan = json.loads((OUT / 'panel_tasks.json').read_text())
    full_plan = json.loads((OUT / 'full_tasks.json').read_text())
    for label in sorted({t['model'] for t in full_plan['tasks']}):
        paths = [OUT / 'full' / label / f'shard-{s:02d}/summary.json' for s in range(1, 17)]
        if all(p.exists() for p in paths):
            info = [json.loads(p.read_text()) for p in paths]
            merged = merge_shard_summaries([p['summary'] for p in info])
            rows.extend({'model': label, 'epoch': info[0]['epoch'], **s} for s in merged)
    all_rows = pd.DataFrame(rows)
    atomic_tsv(OUT / 'all_results.tsv', all_rows)
    for dataset, frame in all_rows.groupby('dataset'):
        selected = ['model', 'epoch', 'modifier', 'auroc_sequence_macro', 'spearman_sequence_macro',
                    'units', 'n_units_auroc_defined', 'n_units_spearman_defined',
                    'n_collapsed_strata', 'auroc_mw_neutral_sequence_macro']
        atomic_tsv(OUT / 'tables' / (dataset + '.tsv'), frame[selected].sort_values(['model', 'epoch', 'modifier']))
        wide = frame.set_index(['model', 'epoch', 'modifier'])[
            ['auroc_sequence_macro', 'spearman_sequence_macro', 'units',
             'n_units_auroc_defined', 'n_units_spearman_defined']].unstack('modifier')
        wide.columns = [probe + '_' + metric for metric, probe in wide.columns]
        atomic_tsv(OUT / 'tables' / (dataset + '_comparison.tsv'), wide.reset_index())
    expected = [(alias, t['epoch']) for t in panel_plan['tasks'] for alias in t.get('aliases', [t['model']])]
    panel_done = sum(x in index for x in expected)
    full_done = sum((OUT / 'full' / t['model'] / f"shard-{t['shard']:02d}/summary.json").exists()
                    for t in full_plan['tasks'])
    saved_names = [key for key in index if not any(key == e for e in expected)]
    status = {'generated_at': datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(),
              'panel_evaluations_completed': panel_done, 'panel_evaluations_expected': len(expected),
              'panel_unique_checkpoints_expected': len(panel_plan['tasks']),
              'saved_structure_evaluations_completed': len(saved_names),
              'saved_structure_evaluations_expected': 63,
              'full_shards_completed': full_done, 'full_shards_expected': len(full_plan['tasks']),
              'unavailable_checkpoints': len(panel_plan['missing']),
              'all_available_evaluations_complete': (panel_done == len(expected) and
                  full_done == len(full_plan['tasks']) and len(saved_names) == 63)}
    write_json(OUT / 'status.json', status)
    write_json(OUT / 'unavailable.json', panel_plan['missing'])
    lines = ['# 単一予測構造とreactivityの整合性：MW対応AUROC / Spearman', '',
             'AUROCは予測構造の未対合=1・対合=0をラベル、実測reactivityを連続スコアとする。'
             '同値の比較は0.5。塩基対確率、reactivityの中央値二値化、値のclipは使わない。'
             'Spearmanも同じ未対合0/1と実測reactivityから計算する。', '',
             'DMSはA/C別、CMCTはG/U別に計算して測定内で平均する。有限値かつreactivity > −10を観測とし、'
             '配列長1000 nt以下のみを使う。測定内の塩基種平均→反復測定平均→RNA平均の順で集計する。'
             'TrainSetBでは同じ配列でも参照構造が異なる試料を別単位とする。', '',
             '主比較表は、比較する全モデルでAUROCとSpearmanの両方を定義できる共通の測定・塩基種群に揃える。'
             '全対合・全未対合は両指標が未定義、reactivity一定はSpearman未定義。'
             '独立集計の平均・除外件数は all_results.tsv と tables/*.tsv に記録する。'
             '比較表ごとに共通集合が変わるため、同じ事前学習モデルでも表間で平均値が異なる場合がある。', '',
             '補助列 auroc_mw_neutral は2観測以上の退化群を0.5とし、学習時のMW項の1−lossと一致させた値。'
             '通常のAUROCと区別する。主表のAUROCは退化群を0.5として水増ししない。', '',
             'RNAgym公式のBPP評価とは定義が異なるため、論文のAUROC値との直接比較には使わない。'
             'EternaFoldは従来実験で保存したgamma=6の構造を使い、MXfold2はMAP構造を使う。', '',
             f"進捗：共通パネル {panel_done}/{len(expected)} 評価、RNAgym全テスト {full_done}/{len(full_plan['tasks'])} シャード。"
             f"チェックポイント欠損 {len(panel_plan['missing'])} 評価は unavailable.json に列挙。", '',
             '集計日時：' + status['generated_at'], '',
             'eFoldはウイルスRNA40配列（SHAPE 15、DMS 25）。RNAgym共通パネルは1024配列、'
             '全テストは433539配列・702609測定。古い100評価の再評価は共通パネル上の新しい評価であり、'
             '元の全テスト／1-of-100抽出結果の再現とは区別する。', '']
    comparisons = []
    e_entries = [('事前学習 bpRNA', 'pretrain', 10), ('MW Rank', 'mw_rank_restored', 5),
                 ('IMLE-NLL', 'nll_restored', 5), ('SHAPE-fy', 'shape_fy_restored', 5)]
    blocks = [('eFold / FT 5 epochs・replay 20%', 'eFold_viral', e_entries, ('SHAPE', 'DMS'), 'efold_ft5')]
    blocks.append(('eFold / assisted folding label', 'eFold_viral',
                   [('事前学習 bpRNA', 'pretrain', 10), ('assisted label（初回設定）', 'original_assisted', 5)],
                   ('SHAPE', 'DMS'), 'efold_assisted'))
    blocks.append(('eFold / 著者公開モデル比較', 'eFold_viral',
                   [('事前学習 bpRNA', 'pretrain', 10), ('著者eFold', 'author_eFold', 0),
                    ('著者EternaFold', 'author_EternaFold', 0), ('著者MXFold2', 'author_MXFold2', 0)],
                   ('SHAPE', 'DMS'), 'efold_authors'))
    rn_prefix = 'replay20-full-test-20261023__training__'
    rn_entries = [('事前学習 TrainSetA', rn_prefix + 'rank', 0),
                  ('MW Rank', rn_prefix + 'rank', 3), ('IMLE-NLL', rn_prefix + 'nll', 3),
                  ('SHAPE-fy', rn_prefix + 'shape_fy', 3)]
    blocks.append(('RNAgym / 共通パネル・FT 3 epochs・replay 20%', 'RNAgym_test', rn_entries,
                   ('2A3', 'DMS', '1M7', 'NMIA'), 'rnagym_panel_ft3'))
    for probe in ('2A3', 'DMS'):
        synth_entries = [('事前学習 TrainSetA', 'trainsetb-synthetic-20261020__conditions__imle-rank-2A3-replay05', 0),
                         ('MW Rank', f'trainsetb-synthetic-20261020__conditions__imle-rank-{probe}-replay20', 100),
                         ('IMLE-NLL', f'trainsetb-synthetic-nll-20261021__conditions__imle-nll-{probe}-replay20', 100),
                         ('SHAPE-fy', f'trainsetb-synthetic-20261020__conditions__shape-fy-{probe}-replay20', 100)]
        blocks.append((f'TrainSetB / {probe}合成反応性でFT 100 epochs・replay 20%（訓練適合度）',
                       'TrainSetB_simulated_training', synth_entries, ('2A3', 'DMS'), 'trainsetb_' + probe))
        blocks.append((f'RNAgym共通パネル / TrainSetB {probe}でFT 100 epochs・replay 20%',
                       'RNAgym_test', synth_entries, ('2A3', 'DMS'), 'rnagym_synthetic_' + probe))
    synthetic_all = [('事前学習 TrainSetA', 'trainsetb-synthetic-20261020__conditions__imle-rank-2A3-replay05', 0)]
    for probe in ('2A3', 'DMS'):
        synthetic_all.extend([
            ('MW Rank / ' + probe, f'trainsetb-synthetic-20261020__conditions__imle-rank-{probe}-replay20', 100),
            ('IMLE-NLL / ' + probe, f'trainsetb-synthetic-nll-20261021__conditions__imle-nll-{probe}-replay20', 100),
            ('SHAPE-fy / ' + probe, f'trainsetb-synthetic-20261020__conditions__shape-fy-{probe}-replay20', 100)])
    blocks.append(('TrainSetB / 両訓練試薬・全6条件の共通比較・FT 100 epochs・replay 20%（訓練適合度）',
                   'TrainSetB_simulated_training', synthetic_all, ('2A3', 'DMS'), 'trainsetb_all_six'))
    eterna = [('EternaFold事前学習', 'EternaFold_pretrain', 0),
              ('EternaFold 分岐2A3', 'EternaFold_branch_2A3', 0),
              ('EternaFold 分岐DMS', 'EternaFold_branch_DMS', 0),
              ('EternaFold 同時2A3', 'EternaFold_joint_2A3', 0),
              ('EternaFold 同時DMS', 'EternaFold_joint_DMS', 0),
              ('EternaFold 同時RNAgym', 'EternaFold_joint_rnagym', 0)]
    blocks.append(('EternaFold / RNAgym共通パネル', 'RNAgym_test', eterna,
                   ('2A3', 'DMS', '1M7', 'NMIA'), 'eternafold_rnagym'))
    blocks.append(('RNAgym / MXfold2・EternaFoldを同じ観測群で直接比較', 'RNAgym_test',
                   [('MXfold2 ' + label, model, epoch) for label, model, epoch in rn_entries] + eterna,
                   ('2A3', 'DMS'), 'mxfold2_eternafold_common'))
    blocks.append(('EternaFold / TrainSetB合成データ（訓練適合度）', 'TrainSetB_simulated_training',
                   [(label, model + '__TrainSetB', epoch) for label, model, epoch in eterna],
                   ('2A3', 'DMS'), 'eternafold_trainsetb'))
    joint_entries = [('TrainSetAのみ・同時学習対照', 'joint_traina-only-joint', 100)]
    joint_entries += [(method + ' / ' + probe, 'joint_' + name + '-' + probe + '-joint', 100)
                      for method, name in [('MW Rank', 'imle-rank'), ('IMLE-NLL', 'imle-nll'), ('SHAPE-fy', 'shape-fy')]
                      for probe in ('2A3', 'DMS')]
    blocks.append(('MXfold2同時学習 / RNAgym共通パネル・100 epochs（初期値はランダム）',
                   'RNAgym_test', joint_entries, ('2A3', 'DMS'), 'joint_rnagym'))
    for title, dataset, entries, modifiers, comparison_id in blocks:
        text, exported = comparison(index, title, dataset, entries, modifiers, comparison_id)
        lines.extend([text, ''])
        comparisons.extend(exported)
    full_entries = [('事前学習 TrainSetA', 'baseline', 0), ('MW Rank', 'rank_replay20_epoch3', 3),
                    ('IMLE-NLL', 'nll_replay20_epoch3', 3), ('SHAPE-fy', 'shape_fy_replay20_epoch3', 3)]
    for percent in (20, 10):
        entries = full_entries if percent == 20 else [(label, model.replace('replay20', 'replay10'), epoch)
                                                     for label, model, epoch in full_entries]
        text, exported = full_comparison(f'RNAgym / 全テスト・FT 3 epochs・replay {percent}%',
                                         entries, f'rnagym_full_replay{percent}')
        lines.extend([text, ''])
        comparisons.extend(exported)
    # Preserve matched-epoch trajectories as a separate, machine-readable table.
    trajectory = []
    for epoch in range(1, 6):
        entries = [e_entries[0]] + [(label, model, epoch) for label, model, _ in e_entries[1:]]
        _, exported = comparison(index, f'eFold FT {epoch}', 'eFold_viral', entries,
                                  ('SHAPE', 'DMS'), f'efold_epoch{epoch}')
        trajectory.extend(exported)
    for epoch in range(1, 4):
        entries = [rn_entries[0]] + [(label, model, epoch) for label, model, _ in rn_entries[1:]]
        _, exported = comparison(index, f'RNAgym FT {epoch}', 'RNAgym_test', entries,
                                  ('2A3', 'DMS', '1M7', 'NMIA'), f'rnagym_epoch{epoch}')
        trajectory.extend(exported)
    atomic_tsv(OUT / 'comparisons.tsv', pd.DataFrame(comparisons))
    atomic_tsv(OUT / 'matched_epoch_trajectories.tsv', pd.DataFrame(trajectory))
    lines.extend(['全条件・全保存エポックの数値と定義可能件数は all_results.tsv、データセット別は tables/、'
                  '共通対象での差分は comparisons.tsv、途中エポックは matched_epoch_trajectories.tsv。', ''])
    old_nll = all_rows[(all_rows.model == 'original_nll') & (all_rows.epoch == 5)]
    if not old_nll.empty:
        lines.extend(['旧IMLE-NLL（初回設定、FT 5 epochs）の退化確認：', '',
                      '| 試薬 | 定義可能AUROC RNA | 定義可能Spearman RNA | 全RNA | MW中立拡張AUROC |',
                      '|---|---:|---:|---:|---:|'])
        for row in old_nll.itertuples(index=False):
            lines.append(f'| {row.modifier} | {row.n_units_auroc_defined} | '
                         f'{row.n_units_spearman_defined} | {row.units} | '
                         f'{fmt(row.auroc_mw_neutral_sequence_macro)} |')
        lines.extend(['', 'この条件の未定義値を通常のAUROC=0.5と解釈しない。現在の復元設定によるIMLE-NLLとは別条件。', ''])
    path = OUT / 'REPORT_ja.md'
    temporary = OUT / f'.report.{os.getpid()}.tmp'
    temporary.write_text('\n'.join(lines))
    temporary.replace(path)
    print(json.dumps(status, indent=2), flush=True)
    require_complete = os.environ.get('MW_MAP_REQUIRE_COMPLETE') == '1' or os.environ.get('JOB_NAME') == 'mw_map_report'
    if require_complete and not status['all_available_evaluations_complete']:
        raise SystemExit('Some available evaluations failed or are incomplete; see status.json and scheduler logs')


if __name__ == '__main__':
    main()
