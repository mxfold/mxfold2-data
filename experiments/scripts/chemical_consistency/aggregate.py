#!/usr/bin/env python3
"""Aggregate completed AUROC/Spearman results and report coverage."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.chemical_consistency.prepare import OUT


def row_for(frame, dataset, model, epoch, modifier, kind='BPP'):
    rows = frame[(frame.dataset == dataset) & (frame.model == model) & (frame.epoch == epoch) &
                 (frame.modifier == modifier) & (frame.kind == kind)]
    if len(rows) > 1:
        raise ValueError(f'Ambiguous aggregate row: {dataset}/{model}/{epoch}/{modifier}/{kind}')
    return None if rows.empty else rows.iloc[0]


def fmt(value):
    return '—' if value is None or pd.isna(value) else f'{float(value):.4f}'


def metric(row, spearman='spearman_sequence_macro'):
    if row is None:
        return '— / —'
    return fmt(row.auroc_sequence_macro) + ' / ' + fmt(row[spearman])


def make_table(frame, dataset, entries, modifiers, kind='BPP', spearman='spearman_sequence_macro'):
    header = '| モデル | Epoch | ' + ' | '.join(p + ' AUROC / ρ' for p in modifiers) + ' |'
    lines = [header, '|---|---:|' + '---:|' * len(modifiers)]
    for label, model, epoch in entries:
        values = [metric(row_for(frame, dataset, model, epoch, modifier, kind), spearman) for modifier in modifiers]
        display_epoch = 10 if kind == 'BPP_common' and model == 'pretrain_common' else epoch
        lines.append('| ' + label + ' | ' + ('—' if epoch == 0 and model.startswith('EternaFold') else str(display_epoch)) +
                     ' | ' + ' | '.join(values) + ' |')
    return '\n'.join(lines)


def main():
    info = json.loads((OUT / 'panel_manifest.json').read_text())
    rn_plan = json.loads((OUT / 'rnagym_tasks.json').read_text())
    ef_tasks = json.loads((OUT / 'efold_tasks.json').read_text())
    historical_path = OUT / 'historical_tasks.json'
    historical = json.loads(historical_path.read_text()) if historical_path.exists() else {'tasks': [], 'missing': []}
    historical_epochs = {item['model']: int(Path(item['checkpoint']).name.removeprefix('epoch-'))
                         for item in historical['tasks']}
    rows, completed = [], set()
    for path in sorted((OUT / 'metrics').glob('*/*/epoch-*/summary.json')):
        summary = json.loads(path.read_text())
        if summary['kind'] == 'SMOKE':
            continue
        aliases = summary.get('provenance', {}).get('aliases', [summary['model']])
        for alias in aliases:
            for item in summary['summary']:
                # Historical tasks use an internal epoch-000 folder because
                # each label identifies one evaluation. In the exported table,
                # retain the actual checkpoint's local training epoch.
                rows.append({**item, 'model': alias,
                             'epoch': historical_epochs.get(alias, item['epoch'])})
            completed.add((summary['kind'], alias, summary['epoch']))
    # When only the historic Spearman CSV survived, preserve its actual score
    # and make the unavailable AUROC explicit rather than fabricating it.
    for item in rn_plan['missing']:
        p = Path(item['evaluation'])
        run = p.parent.parent.parent.parent
        rel = run.relative_to(ROOT / 'qsub_logs/rnagym')
        label = rel.parts[0] + ('__' + '__'.join(rel.parts[1:]) if len(rel.parts) > 1 else '')
        old = pd.read_csv(p)
        for modifier, group in old.groupby('modifier'):
            rows.append({'dataset': 'RNAgym_test', 'model': label, 'epoch': item['epoch'],
                         'kind': 'BPP_missing_checkpoint', 'modifier': modifier,
                         'profiles': len(group), 'units': len(group),
                         'spearman_sequence_macro': float(group.spearman.mean()),
                         'spearman_cluster_macro': float(group.groupby('cluster').spearman.mean().mean()),
                         'auroc_micro': None, 'auroc_sequence_macro': None,
                         'n_units_spearman_defined': len(group), 'n_units_auc_defined': 0})
    full_dir = OUT / 'rnagym_full_saved'
    full_shards = sorted(full_dir.glob('shard-*.json')) if full_dir.exists() else []
    if len(full_shards) == 16:
        pieces = [json.loads(p.read_text()) for p in full_shards]
        source = ROOT / 'qsub_logs/rnagym/replay20-full-test-20261023'
        labels = list(pieces[0]['stats'])
        for label in labels:
            official = json.loads((source / 'evaluation' / label / 'metrics.json').read_text())['chemical']
            for modifier in ('DMS', '2A3'):
                parts = [p['stats'][label][modifier] for p in pieces]
                total = lambda k: sum(p[k] for p in parts)
                auc_n, rho_n, rho_ac_n = (total('auc_defined'), total('spearman_defined'),
                                            total('spearman_ac_defined'))
                rows.append({'dataset': 'RNAgym_full_official_test', 'model': label,
                             'epoch': 0 if label == 'baseline' else 3, 'kind': 'BPP_full_legacy_mask',
                             'modifier': modifier, 'threshold': official[modifier]['true_median'],
                             'profiles': total('profiles'), 'units': total('profiles'),
                             'n_bases': total('bases'), 'auroc_micro': official[modifier]['auc'],
                             'auroc_sequence_macro': total('auc_sum') / auc_n if auc_n else None,
                             'n_units_auc_defined': auc_n,
                             'spearman_sequence_macro': total('spearman_sum') / rho_n if rho_n else None,
                             'spearman_all_sequence_macro': total('spearman_sum') / rho_n if rho_n else None,
                             'spearman_ac_sequence_macro': total('spearman_ac_sum') / rho_ac_n if rho_ac_n else None,
                             'n_units_spearman_defined': rho_n,
                             'n_prediction_constant': total('constant_predictions')})
    ac_dir = OUT / 'rnagym_full_ac'
    ac_shards = sorted(ac_dir.glob('shard-*.json')) if ac_dir.exists() else []
    ac_threshold_path = OUT / 'rnagym_full_ac_threshold.json'
    ac_threshold = (json.loads(ac_threshold_path.read_text())['threshold']
                    if ac_threshold_path.exists() else None)
    if len(ac_shards) == 16:
        ac_pieces = [json.loads(p.read_text()) for p in ac_shards]
        cut = ac_threshold
        if any(p['threshold'] != cut for p in ac_pieces):
            raise RuntimeError('DMS A/C shard thresholds differ')
        for label in ac_pieces[0]['stats']:
            parts = [p['stats'][label] for p in ac_pieces]
            total = lambda k: sum(p[k] for p in parts)
            auc_n, rho_n, ac_n = (total('auc_defined'), total('spearman_defined'),
                                  total('spearman_ac_defined'))
            rows.append({'dataset': 'RNAgym_full_test_AC', 'model': label,
                         'epoch': 0 if label == 'baseline' else 3, 'kind': 'BPP_full_AC_mask',
                         'modifier': 'DMS', 'threshold': cut, 'profiles': total('profiles'),
                         'units': total('profiles'), 'n_bases': total('bases'), 'auroc_micro': None,
                         'auroc_sequence_macro': total('auc_sum') / auc_n if auc_n else None,
                         'n_units_auc_defined': auc_n,
                         'spearman_sequence_macro': total('spearman_ac_sum') / ac_n if ac_n else None,
                         'spearman_all_sequence_macro': total('spearman_sum') / rho_n if rho_n else None,
                         'spearman_ac_sequence_macro': total('spearman_ac_sum') / ac_n if ac_n else None,
                         'n_units_spearman_defined': ac_n,
                         'n_prediction_constant': total('constant_predictions')})
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError('No scores yet')
    tmp = OUT / f'.summary.{os.getpid()}.tmp'
    frame.sort_values(['dataset', 'kind', 'model', 'epoch', 'modifier']).to_csv(
        tmp, sep='\t', index=False, float_format='%.8f')
    tmp.replace(OUT / 'summary.tsv')
    e_expected = [(x['model'], x['epoch']) for task in ef_tasks for x in task['entries']]
    r_expected = [(alias, e['epoch']) for task in rn_plan['tasks'] for e in task['entries']
                  for alias in e.get('aliases', [e['model']])]
    e_done = sum(('BPP', m, e) in completed for m, e in e_expected)
    r_done = sum(('BPP', m, e) in completed for m, e in r_expected)
    historical_done = sum(('BPP', item['model'], 0) in completed for item in historical['tasks'])
    e_rows = [('bpRNA事前学習', 'pretrain', 10), ('MW Rank', 'mw_rank_restored', 5),
              ('IMLE-NLL', 'nll_restored', 5), ('SHAPE-fy', 'shape_fy_restored', 5),
              ('assisted label（初回）', 'original_assisted', 5)]
    e_common = [('bpRNA事前学習', 'pretrain_common', 5),
                ('MW Rank', 'mw_rank_restored_common', 5),
                ('IMLE-NLL', 'nll_restored_common', 5),
                ('SHAPE-fy', 'shape_fy_restored_common', 5)]
    e_map = e_rows + [('著者eFold', 'author_eFold', 0), ('著者EternaFold', 'author_EternaFold', 0),
                      ('著者MXFold2', 'author_MXFold2', 0), ('旧IMLE-NLL', 'original_nll', 5)]
    eterna = [('EternaFold pretrain', 'EternaFold_pretrain', 0),
              ('EternaFold synthetic 2A3', 'EternaFold_branch_2A3', 0),
              ('EternaFold synthetic DMS', 'EternaFold_branch_DMS', 0),
              ('EternaFold joint 2A3', 'EternaFold_joint_2A3', 0),
              ('EternaFold joint DMS', 'EternaFold_joint_DMS', 0),
              ('EternaFold joint RNAgym', 'EternaFold_joint_rnagym', 0)]
    joint_names = [('MXfold2 joint ' + suffix,
                    'mxfold2-trainsetab-joint-20261022__conditions__' + suffix, 100)
                   for suffix in ('traina-only-joint', 'imle-rank-2A3-joint', 'imle-rank-DMS-joint',
                                  'imle-nll-2A3-joint', 'imle-nll-DMS-joint',
                                  'shape-fy-2A3-joint', 'shape-fy-DMS-joint')]
    synth = [('TrainSetB ' + suffix,
              'trainsetb-synthetic-20261020__conditions__' + suffix, 100)
             for suffix in ('imle-rank-2A3-replay20', 'imle-rank-DMS-replay20',
                            'shape-fy-2A3-replay20', 'shape-fy-DMS-replay20')]
    synth += [('TrainSetB ' + suffix,
               'trainsetb-synthetic-nll-20261021__conditions__' + suffix, 100)
              for suffix in ('imle-nll-2A3-replay20', 'imle-nll-DMS-replay20')]
    esynth = [(name + ' / TrainSetB', model + '__TrainSetB', epoch) for name, model, epoch in eterna]
    status = {'efold_bpp_completed': e_done, 'efold_bpp_expected': len(e_expected),
              'rnagym_bpp_completed': r_done, 'rnagym_bpp_expected': len(r_expected),
              'rnagym_old_checkpoints_missing': len(rn_plan['missing']),
              'eternafold_rnagym_bpp_completed': sum(('BPP', m, e) in completed for _, m, e in eterna),
              'eternafold_trainsetb_bpp_completed': sum(('BPP', m, e) in completed for _, m, e in esynth),
              'historical_panel_completed': historical_done,
              'historical_panel_expected': len(historical['tasks']),
              'historical_panel_missing': len(historical['missing']),
              'full_rnagym_saved_shards_completed': len(full_shards), 'full_rnagym_saved_shards_expected': 16,
              'full_rnagym_ac_shards_completed': len(ac_shards), 'full_rnagym_ac_shards_expected': 16}
    tmp = OUT / f'.status.{os.getpid()}.tmp'
    tmp.write_text(json.dumps(status, indent=2) + '\n')
    tmp.replace(OUT / 'status.json')
    lines = ['# 化学反応性との一致：AUROCとSpearman', '',
             '## 定義と注意', '',
             'AUROCは実測または合成反応性を正解として、データセット・試薬別の観測値を[0,1]にclipした中央値より大きい塩基を陽性とする。'
             'モデル側のスコアは配列のみから得た非対合確率 `1−Σj P(i,j)`。閾値はモデルごとに変えない。',
             'RNAgymとTrainSetBのMXfold2確率は従来評価と同様に非対合確率を[0,1]へclip。'
             'eFoldでは範囲から0.001を超えて外れたBPPを持つ配列を除外し、件数を記録する。',
             '主表は各RNAのAUROCの平均。陽性・陰性のどちらかがないRNAはAUROC未定義として件数を記録し、'
             '全位置をpoolしたAUROCは `auroc_micro` に保存する。',
             'Spearmanはclip前の反応性と非対合確率をRNA別に比較する。DMSはA/C別に相関を求め平均した値が主表で、'
             'A/Cをまとめた値も `spearman_all_sequence_macro` に保存。一定値や観測不足で未定義の相関は除外し件数を記録する。',
             'RNAgym公式に似た中央値二値化だが、DMSのA/Cマスク、反復測定、sequence平均、予測確率の定義が公式scorerと異なるため、'
             '公式値の再現とは呼ばない。単一構造の結果はBPPの結果と別表にする。',
             'TrainSetB合成反応性はTrainSetBの参照構造から生成されて学習に使われた。'
             'TrainSetBの表は独立テストではなく学習適合度であり、43件の配列重複があっても1,094サンプルを独立に集計する。', '',
             '## 進捗', '',
             f'- eFold BPP: {e_done} / {len(e_expected)} checkpoint。',
             f'- 既存RNAgym評価BPP: {r_done} / {len(r_expected)} 評価箇所。',
             f'- 古いRNAgymの{len(rn_plan["missing"])}箇所は対応するcheckpointが見つからず、保存済みSpearmanのみ収録。'
             '従来の定数予測の扱いや集計単位が異なるため、新しいSpearmanと厳密には同じ定義ではない。',
             f'- EternaFold: RNAgym {status["eternafold_rnagym_bpp_completed"]}/6、'
             f'TrainSetB {status["eternafold_trainsetb_bpp_completed"]}/6。', '',
             f'- 保存済みRNAgym全テスト比較: {len(full_shards)}/16 shard。',
             f'- 同じ全テストのDMS A/C限定: {len(ac_shards)}/16 shard。',
             f'- より古い実験の共通パネル再評価: {historical_done}/{len(historical["tasks"])}条件。'
             '当時の全テストまたは1/100テストそのものの再計算ではなく、現在のRNAgym共通パネルとTrainSetBを使う。'
             'これらのEpochは当時のcheckpointファイル名に記録された各学習セグメント内の番号。', '',
             '## eFoldテスト：Viral mRNA、BPP非対合確率', '',
             'DMS 25配列、SHAPE 15配列。FTは5 epoch。DMSのρは以前のeFold評価に合わせA/Cをまとめて計算。', '',
             make_table(frame, 'eFold_viral', e_rows, ['SHAPE', 'DMS'], spearman='spearman_all_sequence_macro'), '',
             '## eFoldテスト：同一配列に揃えた比較', '',
             '非有限または範囲外のBPPが出た配列は、そのepochの全モデルから除き、共通の測定配列で再集計する。'
             '陽性ラベルの閾値は元の全テストデータで定めた中央値を維持。'
             'DMSのρはAとCを別々に計算した後の平均であり、上の表のA/Cをまとめたρと定義が異なる。', '',
             make_table(frame, 'eFold_viral', e_common, ['SHAPE', 'DMS'], kind='BPP_common'), '',
             '## eFoldテスト：Viral mRNA、単一構造の非対合状態', '',
             '著者モデルは論文の保存済み単一構造を利用。単一構造の0/1予測とBPP確率を同じ性能として比較しない。', '',
             make_table(frame, 'eFold_viral', e_map, ['SHAPE', 'DMS'], kind='MAP',
                        spearman='spearman_all_sequence_macro'), '',
             '## RNAgym実測テスト、BPP非対合確率', '',
             '各プローブで2A3 825、DMS 765、1M7 61、NMIA 5配列。'
             'EternaFoldのBPPは保存済みテキストの丸めを含み、MXfold2はcutoff 0のnative BPPを再生成。', '',
             make_table(frame, 'RNAgym_test', eterna + joint_names, ['2A3', 'DMS', '1M7', 'NMIA']), '',
             '## RNAgym全テスト：保存済み7モデル', '',
             '元の全テスト評価はDMSもG/Uを含む観測全位置を用いている。'
             'その既存の定義・中央値のままRNA別AUROCを再計算し、元の非clip反応性を復元してSpearmanを計算した。'
             'A/Cのみのpanelとは分けて解釈する。', '',
             make_table(frame, 'RNAgym_full_official_test',
                        [('baseline', 'baseline', 0)] + [(x, x, 3) for x in
                         ('rank_replay10_epoch3', 'rank_replay20_epoch3', 'nll_replay10_epoch3',
                          'nll_replay20_epoch3', 'shape_fy_replay10_epoch3', 'shape_fy_replay20_epoch3')],
                        ['2A3', 'DMS'], kind='BPP_full_legacy_mask'), '',
             '## RNAgym全テスト：DMS A/C限定', '',
             'A/Cの有効反応性だけから計算した全体中央値を固定閾値とする。'
             'Spearmanは各RNA内でAとCの相関を別々に計算し、利用可能な値を平均する。'
             f'閾値: {fmt(ac_threshold)}。', '',
             make_table(frame, 'RNAgym_full_test_AC',
                        [('baseline', 'baseline', 0)] + [(x, x, 3) for x in
                         ('rank_replay10_epoch3', 'rank_replay20_epoch3', 'nll_replay10_epoch3',
                          'nll_replay20_epoch3', 'shape_fy_replay10_epoch3', 'shape_fy_replay20_epoch3')],
                        ['DMS'], kind='BPP_full_AC_mask'), '',
             '## TrainSetB合成反応性、BPP非対合確率', '',
             'この表は学習に使用した合成データへの適合度。', '',
             make_table(frame, 'TrainSetB_simulated_training', esynth + synth + joint_names,
                        ['2A3', 'DMS']), '',
             '計算済みの各条件・途中epoch・coverage・塩基別集計・pool版AUROCは `summary.tsv` に記録。'
             'モデルごとの `metrics/` に測定単位の結果、`predictions/` に非対合確率を保存する。',
             '固定閾値: `' + json.dumps(info['thresholds_clipped_median'], ensure_ascii=False) + '`', '']
    tmp = OUT / f'.report.{os.getpid()}.tmp'
    tmp.write_text('\n'.join(lines))
    tmp.replace(OUT / 'REPORT_ja.md')
    print(json.dumps(status, indent=2))


if __name__ == '__main__':
    main()
