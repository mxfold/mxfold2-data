#!/usr/bin/env python3
"""Summarize observed learning curves, dataset shifts, and frozen-score controls."""
from __future__ import annotations
from collections import defaultdict
import csv
from datetime import datetime
import json
from pathlib import Path
import statistics
import sys
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).parent))
import pretrain_diagnosis as experiment

ROOT, OUT, OLD = experiment.ROOT, experiment.OUT, experiment.OLD
base = experiment.base


def tsv(path, rows):
    with path.open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter='\t')
        writer.writeheader(); writer.writerows(rows)


def fmean(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def summarize(rows, cohort, dataset):
    common = [r for r in rows if r['ref_pairs'] > 0]
    matched_similarity = [r for r in rows if r['bpRNA_nearest_edit_similarity'] is not None]
    return {'dataset': dataset, 'cohort': cohort, 'n': len(rows),
        'bpRNA_f1': fmean(r['ours_f1'] for r in rows),
        'TrainSetAB_f1': fmean(r['public_f1'] for r in rows),
        'gap': fmean(r['gap_public_minus_ours'] for r in rows),
        'canonical_noncrossing_ceiling_f1': fmean(r['canonical_noncrossing_ceiling_f1'] for r in rows),
        'n_reference_nonempty': len(common),
        'bpRNA_empty_with_nonempty_reference': sum(not r['ours_pred_pairs'] for r in common),
        'TrainSetAB_empty_with_nonempty_reference': sum(not r['public_pred_pairs'] for r in common),
        'bpRNA_recall_nonempty_reference': fmean(r['ours_recall'] for r in common),
        'TrainSetAB_recall_nonempty_reference': fmean(r['public_recall'] for r in common),
        'TrainSetAB_exact_n': sum(bool(r['TrainSetAB_exact_ids']) for r in rows),
        'bpRNA_exact_n': sum(r['bpRNA_exact_id'] is not None for r in rows),
        'similarity_comparison_n': len(matched_similarity),
        'bpRNA_mean_nearest_edit_similarity': fmean(r['bpRNA_nearest_edit_similarity'] for r in matched_similarity),
        'TrainSetAB_mean_nearest_edit_similarity': fmean(r['TrainSetAB_nearest_edit_similarity'] for r in matched_similarity)}


def main():
    import numpy as np
    sys.path.append(str(ROOT / '.cache/bprna-benchmark-deps'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plan = json.loads((OUT / 'plan.json').read_text())
    panel = {r['id']: r for r in base.read_jsonl(OUT / 'diagnostic_panel.jsonl')}
    reference = {r['id']: r for r in base.read_jsonl(experiment.FULL / 'panel.jsonl')}
    observed = base.read_jsonl(OUT / 'per_test_diagnostics.jsonl')
    observed_by_id = {r['id']: r for r in observed}
    all_predictions, native_validation = {}, {}
    for spec in plan['models']:
        name = spec['model']; rows = []
        for shard in range(experiment.SHARDS):
            path = OUT / 'predictions' / name / f'shard-{shard}.jsonl'
            done = json.loads(path.with_name(f'shard-{shard}-complete.json').read_text())
            part = base.read_jsonl(path)
            if not done['success'] or done['predictions_sha256'] != base.sha(path) or done['n'] != len(part):
                raise ValueError('Incomplete or changed diagnostic shard')
            rows.extend(part)
        if len(rows) != len(panel) or {r['id'] for r in rows} != set(panel):
            raise ValueError('Incomplete diagnostic coverage')
        for r in rows:
            if r['sequence_sha256'] != panel[r['id']]['sequence_sha256'] or not r['constraint_exact']:
                raise ValueError('Sequence/target constraint mismatch')
            if r['id'] in observed_by_id:
                expected = observed_by_id[r['id']]['ours_f1' if name == 'mx_bprna10' else 'public_f1']
                if abs(r['default']['f1'] - expected) > 1e-12:
                    raise ValueError('Native default prediction differs from existing evaluation')
        native_validation[name] = {'n': len(rows), 'existing_test_scores_match_n': sum(r['id'] in observed_by_id for r in rows),
                                   'constraints_exact_n': sum(r['constraint_exact'] for r in rows)}
        all_predictions[name] = rows
    strata = []
    bins = [('1-30', 1, 30), ('31-50', 31, 50), ('51-100', 51, 100),
            ('101-200', 101, 200), ('201-500', 201, 500), ('501+', 501, 10000)]
    for dataset in ('PDB', 'archiveII'):
        a = [r for r in observed if r['source'] == dataset]
        strata.append(summarize(a, 'all', dataset))
        for label, lo, hi in bins:
            b = [r for r in a if lo <= r['length'] <= hi]
            if b: strata.append(summarize(b, 'length:' + label, dataset))
        for exact in (True, False):
            b = [r for r in a if bool(r['TrainSetAB_exact_ids']) == exact]
            if b: strata.append(summarize(b, 'TrainSetAB_exact:' + str(exact), dataset))
        for lo, hi in [(0, .7), (.7, .8), (.8, .9), (.9, 1.000001)]:
            b = [r for r in a if lo <= r['TrainSetAB_nearest_edit_similarity'] < hi]
            if b: strata.append(summarize(b, f'TrainSetAB_similarity:{lo}-{min(hi,1)}', dataset))
        if dataset == 'archiveII':
            for family in sorted({r['family'] for r in a}):
                strata.append(summarize([r for r in a if r['family'] == family], 'family:' + family, dataset))
    tsv(OUT / 'test_strata.tsv', strata)
    dataset_summary = json.loads((OUT / 'training_dataset_summary.json').read_text())
    curve = json.loads((OUT / 'training_curve.json').read_text())
    tsv(OUT / 'epoch_loss.tsv', [{k: r[k] for k in ['epoch', 'updates', 'mean_loss', 'primary_examples']}
                                for r in curve['epochs']])
    tsv(OUT / 'online_loss_blocks.tsv', curve['blocks'])
    ablations, fit, noise = [], [], []
    for name, rows in all_predictions.items():
        for dataset in ('bpRNA_train_sample', 'PDB', 'ArchiveII_sample'):
            a = [r for r in rows if r['sample_set'] == dataset]
            fit.append({'model': name, 'sample_set': dataset, 'n': len(a),
                'f1_original_labels': fmean(r['default']['f1'] for r in a),
                'f1_projected_labels': fmean(r['projected_target_f1'] for r in a),
                'clean_gap_per_nt': fmean(r['clean_gap_per_nt'] for r in a),
                'thermal_penalty_per_nt': fmean(r['thermal_penalty_per_nt'] for r in a),
                'signed_reference_calibration_error_per_nt': fmean((r['reference_score']-r['reference_turner_score'])/r['length'] for r in a),
                'negative_clean_gap_n': sum(r['clean_gap_per_nt'] < -1e-6 for r in a)})
            if dataset != 'bpRNA_train_sample':
                for variant in ['default'] + list(a[0]['ablations']):
                    values = [r['default'] if variant == 'default' else r['ablations'][variant] for r in a]
                    ablations.append({'model': name, 'dataset': dataset, 'variant': variant, 'n': len(a),
                        'f1': fmean(v['f1'] for v in values),
                        'changed_n': 0 if variant == 'default' else sum(not v['same_as_default'] for v in values),
                        'empty_prediction_n': sum(not r['predicted_pairs'] for r in a) if variant == 'default' else
                                               sum(v['n_predicted_pairs'] == 0 for v in values)})
        chosen = [r for r in rows if 'noise_dropout' in r]
        noise.append({'model': name, 'n_paired_examples': len(chosen),
            'clean_eval_loss': fmean(r['clean_gap_per_nt']+r['thermal_penalty_per_nt'] for r in chosen),
            'noisy_eval_loss': fmean(v['loss_per_nt'] for r in chosen for v in r['noise_dropout'] if v['mode'] == 'eval'),
            'noisy_train_loss': fmean(v['loss_per_nt'] for r in chosen for v in r['noise_dropout'] if v['mode'] == 'train')})
    tsv(OUT / 'score_ablation.tsv', ablations)
    tsv(OUT / 'fit_and_calibration.tsv', fit)
    tsv(OUT / 'noise_dropout_probe.tsv', noise)
    original = {r['id']: r for r in base.read_jsonl(OUT / 'official_TrainSetAB.jsonl')}
    exact_agreement = []
    for item in observed:
        if not item['TrainSetAB_exact_ids']: continue
        values = [experiment.pair_metrics(reference[item['id']]['pairs'], original[key]['pairs'])['f1']
                  for key in item['TrainSetAB_exact_ids']]
        exact_agreement.append({'id': item['id'], 'dataset': item['source'], 'best_label_f1': max(values),
                                'matching_training_ids': ','.join(item['TrainSetAB_exact_ids'])})
    tsv(OUT / 'exact_training_label_agreement.tsv', exact_agreement)
    # Plot measured curves, with no invented validation trajectory.
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'savefig.dpi': 180, 'font.family': 'DejaVu Sans'})
    colors = {'mx_bprna10': '#D55E00', 'MXfold2_public': '#0072B2'}
    labels = {'mx_bprna10': 'bpRNA pretrain10', 'MXfold2_public': 'Public TrainSetAB'}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.3), constrained_layout=True)
    ax = axes[0, 0]
    epochs = curve['epochs']
    ax.plot([r['epoch'] for r in epochs], [r['mean_loss'] for r in epochs], 'o-', color=colors['mx_bprna10'])
    ax.set(title='A. Observed pretraining loss', xlabel='Completed epoch', ylabel='Mean training objective / nt')
    ax.set_xticks(range(1, 11)); ax.grid(alpha=.2)
    ax.annotate(f"{epochs[-1]['mean_loss']:.4f}", (10, epochs[-1]['mean_loss']), xytext=(-40, 15), textcoords='offset points')
    ax.text(.03, .1, 'Dropout=0.5, noise SD=0.75\nValidation curve was not recorded', transform=ax.transAxes, fontsize=9)
    ax = axes[0, 1]
    lengths = [r for r in strata if r['dataset'] == 'PDB' and r['cohort'].startswith('length:')]
    x = np.arange(len(lengths)); width = .36
    ax.bar(x-width/2, [r['TrainSetAB_f1'] for r in lengths], width, color=colors['MXfold2_public'], label=labels['MXfold2_public'])
    ax.bar(x+width/2, [r['bpRNA_f1'] for r in lengths], width, color=colors['mx_bprna10'], label=labels['mx_bprna10'])
    ax.set_xticks(x, [r['cohort'].split(':')[1]+'\n(n='+str(r['n'])+')' for r in lengths], fontsize=8)
    ax.set(title='B. PDB accuracy by length (355 RNAs)', xlabel='Sequence length (nt)', ylabel='Per-RNA mean F1', ylim=(0, 1.03)); ax.legend(fontsize=8)
    ax = axes[1, 0]
    families = [r for r in strata if r['dataset'] == 'archiveII' and r['cohort'].startswith('family:')]
    x = np.arange(len(families))
    ax.barh(x-.18, [r['TrainSetAB_f1'] for r in families], .36, color=colors['MXfold2_public'])
    ax.barh(x+.18, [r['bpRNA_f1'] for r in families], .36, color=colors['mx_bprna10'])
    ax.set_yticks(x, [r['cohort'].split(':')[1]+f" (n={r['n']})" for r in families], fontsize=8)
    ax.set(title='C. ArchiveII accuracy by family (3370 RNAs)', xlabel='Per-RNA mean F1', xlim=(0, 1)); ax.invert_yaxis()
    ax = axes[1, 1]
    training = base.read_jsonl(OLD / 'data/bprna.jsonl')
    all_tests = base.read_jsonl(experiment.FULL / 'panel.jsonl')
    sets = [('bpRNA training', training, colors['mx_bprna10']), ('TrainSetAB training', list(original.values()), colors['MXfold2_public']),
            ('PDB test', [r for r in all_tests if r['source'] == 'PDB'], '#009E73'),
            ('ArchiveII test', [r for r in all_tests if r['source'] == 'archiveII'], '#CC79A7')]
    for label, rows, color in sets:
        values = np.sort([len(r['sequence']) for r in rows]); ax.step(values, np.arange(1, len(values)+1)/len(values), where='post', label=label, color=color)
    ax.set(title='D. Training / test length distribution', xlabel='Sequence length (nt, log scale)', ylabel='Cumulative fraction', xscale='log', xlim=(8, 2200), ylim=(0, 1.02)); ax.legend(fontsize=8); ax.grid(alpha=.2)
    fig.suptitle('bpRNA pretraining: observed learning curve and distribution shifts', fontsize=13)
    for suffix in ('png', 'pdf'): fig.savefig(OUT / f'learning_diagnostics.{suffix}')
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), constrained_layout=True)
    for ax, dataset, title in zip(axes, ('PDB', 'ArchiveII_sample'), ('PDB (all 355 RNAs)', 'ArchiveII (fixed sample, n=128)')):
        for name in ('MXfold2_public', 'mx_bprna10'):
            points = []
            for alpha in (0., .25, .5, 1., 2.):
                variant = 'default' if alpha == 1. else f'neural_alpha_{alpha:g}'
                value = next(r['f1'] for r in ablations if r['model'] == name and r['dataset'] == dataset and r['variant'] == variant)
                points.append(value)
            ax.plot([0, .25, .5, 1, 2], points, 'o-', color=colors[name], label=labels[name])
        ax.axvline(1, color='gray', linestyle=':', alpha=.7)
        ax.set(title=title, xlabel='Relative neural score weight (default = 1)', ylabel='Per-RNA mean F1', ylim=(.5, .97))
        ax.grid(alpha=.2); ax.legend(fontsize=9)
    fig.suptitle('Frozen-weight diagnostic: reducing the neural score helps PDB, not ArchiveII', fontsize=12)
    for suffix in ('png', 'pdf'): fig.savefig(OUT / f'score_ablation.{suffix}')
    plt.close(fig)
    lookup = {(r['dataset'], r['cohort']): r for r in strata}
    weight = json.loads((OUT / 'weight_audit.json').read_text())
    config_comparison = [
        {'setting': 'optimizer', 'bpRNA': 'AdaBelief', 'public_original': 'Adam (original paper)'},
        {'setting': 'structural objective', 'bpRNA': 'FY + Gaussian perturbation SD .75', 'public_original': 'Structured hinge; FN .5, FP .005'},
        {'setting': 'thermodynamic regularization', 'bpRNA': '.125', 'public_original': '.125'},
        {'setting': 'mix_type', 'bpRNA': 'average: (T+N)/2', 'public_original': 'add: T+N'},
        {'setting': 'paired_opt', 'bpRNA': 'symmetric', 'public_original': '0_1_1'},
        {'setting': 'additional parameters', 'bpRNA': '3 trainable scalar terms', 'public_original': 'disabled'},
        {'setting': 'training sequences', 'bpRNA': '45507', 'public_original': '4260 (official released TrainSetA + TrainSetB)'},
        {'setting': 'epochs', 'bpRNA': '10; 455070 updates', 'public_original': 'not inferred from released weight file'},
    ]
    tsv(OUT / 'config_comparison.tsv', config_comparison)
    lines = ['# bpRNA事前学習モデル：PDB・ArchiveII精度低下の診断', '',
        '元のチェックポイントと正式評価は変更していない。追加計算は、公式TrainSetAB訓練データとの照合と、固定重みの推論・スコア寄与の診断である。', '',
        '## 結論', '',
        'PDBの低下には、短いRNAへの分布のずれと、学習したニューラル成分による対合不足が強く関係する。公開TrainSetABの訓練集合には今回のテスト配列が多数含まれるため、公開モデルとの差を同じ独立テスト条件での学習性能差とは解釈できない。ArchiveIIではニューラル成分は有効で、PDBと同じスコア抑制では改善しない。原著とは損失・モデル設定も異なるため、データだけ、epoch数だけの効果は分離できない。', '',
        '## 観測された学習曲線', '',
        f"訓練損失はepoch1の{epochs[0]['mean_loss']:.6f}からepoch10の{epochs[-1]['mean_loss']:.6f}へ{100*(1-epochs[-1]['mean_loss']/epochs[0]['mean_loss']):.2f}%低下した。最終更新数は455,070。NaNによる停止・更新抜け・Turner表の意図しない更新は確認されなかった。LR=.001、weight decay=.01、AdaBelief、batch/accumulation=1、dropout=.5、FY perturb=.75、熱力学的正則化=.125である。", '',
        '1〜9epochの重みは保存されておらず、pretrainの評価もepoch10だけである。このため、検証F1の曲線、最良epoch、過学習の開始時期、10epoch以降の効果はこのログだけから確定できない。FYの訓練損失にはノイズ・dropout・熱力学的ペナルティが含まれ、F1の収束を表す量ではない。', '',
        '| Epoch | 平均訓練損失 | 更新回数 |', '|---:|---:|---:|']
    for r in epochs: lines.append(f"| {r['epoch']} | {r['mean_loss']:.6f} | {r['updates']:,} |")
    lines += ['', f'![学習曲線と分布差]({OUT}/learning_diagnostics.png)', '',
        '## 公開TrainSetABとのデータの違い', '',
        'MXfold2公式v0.1.0配布Rivas.tar.gz（TrainSetA 3,166件、TrainSetB 1,094件）を取得し、U/T・大文字小文字を正規化した配列を照合した。', '',
        '| テスト | 件数 | TrainSetABとの完全一致 | bpRNAとの完全一致 | bpRNA F1 | 公開TrainSetAB F1 |',
        '|---|---:|---:|---:|---:|---:|']
    for ds in ('PDB', 'archiveII'):
        r = lookup[(ds, 'all')]
        lines.append(f"| {ds} | {r['n']} | {r['TrainSetAB_exact_n']} | {r['bpRNA_exact_n']} | {r['bpRNA_f1']:.4f} | {r['TrainSetAB_f1']:.4f} |")
    lines += ['', '同一配列でも正解構造が異なることがあるため、配列一致を全て同一教師とは扱わない。完全一致群では、該当する公開訓練構造との最大F1も照合し、exact_training_label_agreement.tsvに保存した。完全一致を除いた結果は以下の通りで、残る差を完全一致だけで説明することもできない。', '',
              '| 完全一致を除いたテスト | 件数 | bpRNA F1 | 公開TrainSetAB F1 |', '|---|---:|---:|---:|']
    for ds in ('PDB', 'archiveII'):
        r = lookup[(ds, 'TrainSetAB_exact:False')]
        lines.append(f"| {ds} | {r['n']} | {r['bpRNA_f1']:.4f} | {r['TrainSetAB_f1']:.4f} |")
    lines += ['', '配列類似度は両訓練集合に対する全長編集距離の総当たり最近傍検索である。指標は1−Levenshtein距離/max(配列長)であり、アラインメントidentityやRNAファミリー相同性とは異なる。比較件数は既存bpRNA側の解析と揃え、PDB355・ArchiveII3359（1000nt以下）とした。', '',
              '| 集合 | 平均最近傍類似度：bpRNA | 平均最近傍類似度：TrainSetAB |', '|---|---:|---:|']
    for ds in ('PDB', 'archiveII'):
        r = lookup[(ds, 'all')]
        lines.append(f"| {ds} | {r['bpRNA_mean_nearest_edit_similarity']:.4f} | {r['TrainSetAB_mean_nearest_edit_similarity']:.4f} |")
    lengths = dataset_summary['lengths']
    lines += ['', '30nt以下の割合は、bpRNA訓練803/45,507（1.76%）、公開TrainSetAB訓練689/4,260（16.17%）、PDBテスト209/355（58.87%）。PDBの短さは今回のbpRNA訓練分布から大きく外れる。', '',
              '| PDBの長さ | 件数 | bpRNA F1 | 公開TrainSetAB F1 |', '|---|---:|---:|---:|']
    for r in (r for r in strata if r['dataset'] == 'PDB' and r['cohort'].startswith('length:')):
        lines.append(f"| {r['cohort'].split(':')[1]} | {r['n']} | {r['bpRNA_f1']:.4f} | {r['TrainSetAB_f1']:.4f} |")
    lines += ['', '## ニューラル成分の寄与を固定重みで切り分け', '',
        '同じRNA・同じモデル重みで、Turner成分に対するニューラル成分だけをα倍にした。α=1が既存結果、α=0がMXfold2内部のTurner-onlyである。PDBは全355件、ArchiveIIは事前指定したSHA256順の一様サンプル128件。これらの診断からモデルや係数を選び直して正式結果を置き換えることはしていない。', '',
        '| データ | モデル | α=0 | α=.25 | α=.5 | α=1（既定） | α=2 |', '|---|---|---:|---:|---:|---:|---:|']
    for ds in ('PDB', 'ArchiveII_sample'):
        for model in ('mx_bprna10', 'MXfold2_public'):
            vals = []
            for alpha in (0., .25, .5, 1., 2.):
                variant = 'default' if alpha == 1 else f'neural_alpha_{alpha:g}'
                vals.append(next(r['f1'] for r in ablations if r['model'] == model and r['dataset'] == ds and r['variant'] == variant))
            lines.append(f'| {ds} | {model} | ' + ' | '.join(f'{v:.4f}' for v in vals) + ' |')
    p = lookup[('PDB', 'all')]
    lines += ['',
        f"PDBでは、正解に塩基対がある{p['n_reference_nonempty']}配列のうち全未対合を予測したものがbpRNAモデル20件、公開モデル1件。共通の正解非空集合での感度は{p['bpRNA_recall_nonempty_reference']:.4f}対{p['TrainSetAB_recall_nonempty_reference']:.4f}で、低下の中心は対合の見逃しである。", '',
        'PDBでニューラル寄与を弱めるとF1が回復するため、ニューラル成分がこの分布で適切に較正されていないことが直接確認できる。データの短長比やFYとhingeの違いがその学習原因として考えられるが、それぞれの寄与率を確定するには条件を揃えた学習比較が必要である。ArchiveIIではTurner-onlyや弱いニューラル寄与は既定値を下回り、同じ補正では解決しない。', '',
        '追加3スカラー項だけを無効にしたbpRNAモデルのF1は、PDB0.8102、ArchiveII128例0.6140（既定0.6416）。追加項がPDB低下の主因とはいえず、ArchiveIIでは有益である。', '',
        f'![固定重みのスコア寄与診断]({OUT}/score_ablation.png)', '',
        '## 訓練への適合と、原著との設定差', '',
        '| モデル | 訓練bpRNAサンプル数 | 元教師へのF1 | 射影後の実学習教師へのF1 |', '|---|---:|---:|---:|']
    for r in (r for r in fit if r['sample_set'] == 'bpRNA_train_sample'):
        lines.append(f"| {r['model']} | {r['n']} | {r['f1_original_labels']:.4f} | {r['f1_projected_labels']:.4f} |")
    lines += ['', '訓練256例の推論F1も高い状態ではなく、訓練損失の低下だけで構造への適合が十分と判断する根拠はない。このサンプルは既に訓練に使った配列であり、validationとして扱っていない。また異なるデータ集合のF1を直接比較して過学習・適合不足を確定してはいない。', '',
        '実際の設定は、計画で参照したプロジェクトbaseline（TrainA_only_new_parameter/train-new.sh）のFY・AdaBelief・LR=.001・L2=.01・正則化=.125・perturb=.75・dropout=.5と一致する。原著の公開モデルとは以下の違いがある。', '',
        '| 項目 | 今回のbpRNA | 原著公開モデル |', '|---|---|---|']
    for r in config_comparison:
        lines.append(f"| {r['setting']} | {r['bpRNA']} | {r['public_original']} |")
    lines += ['',
        '原著hingeは偽陰性・偽陽性に異なるmarginを置くが、今回のFYではそのmarginを使わない。この違いは感度の差と整合的な候補であり、因果確定ではない。averageとaddは、固定された同一のT・Nなら正の定数倍なのでMAP構造を変えない。しかし学習時のλ(Sref−Tref)^2は、averageではλ(Nref−Tref)^2/4、addではλNref^2となり、同じ数値0.125でもNNに要求するスコアが異なる。推論時にaverageをaddへ切り替えるだけで改善するとは考えない。', '',
        'Turner表は公開モデルと全要素が一致しており、熱力学表が更新・破損した結果ではない。739配列×2モデルの全ターゲット制約が正しく再現され、既存テスト483配列×2モデルの既定F1も保存済み結果と一致した。', '',
        'bpRNA教師1,430,425対のうち139,316対（9.74%）を、canonical・非交差・最小hairpin制約へ射影する際に除いている。実学習教師が空の例は424/45,507。元教師と射影後教師のF1を分けて保存した。PDBとArchiveIIのcanonical非交差F1上限はどちらも約0.98であり、テスト正解のpseudoknot等の物理的制約だけで公開モデルとの差を説明できない。', '',
        '## 現時点で判定できないことと次の確認', '',
        '10epochで十分だったか、100epochまで回せば改善するか、過学習がどのepochで始まったかは未判定である。現在の結果だけを根拠に100epochへ延長する判断はしない。次の学習比較では、テストとは別のvalidationを最初から固定し、各epochの重み・訓練/validation F1・損失内訳を残す。そのうえで、同じbpRNAとモデル構造でFY対hingeを比較し、短いRNAの学習重みの調整や、学習率を下げた継続をvalidationで判定するのが具体的な切り分けになる。', '',
        'bpRNAモデルはbpRNAnewでは0.6386（公開TrainSetAB0.6328）、viralでは0.7901（同0.7095）であり、全ての集合で一様に低いわけではない。PDB/ArchiveIIの差をそのまま一般化性能全体の失敗と解釈しない。', '',
        '## 再現情報', '',
        'CPUデータ照合とGPU4タスクの固定重み診断をqsubで実行した。正式な再学習・チェックポイント選択は行っていない。', '',
        '- 指標・分割集計：test_strata.tsv、score_ablation.tsv、fit_and_calibration.tsv、epoch_loss.tsv。',
        '- 各RNA：per_test_diagnostics.jsonl、nearest_TrainSetAB.jsonl、predictions/以下。',
        '- 公開データ：公式Rivas.tar.gz、公式リリース情報、原著XMLをdata/efold_hf/pretrain_diagnosis_20261031に保存。',
        '- 原著：[Sato et al., Nature Communications 12:941 (2021)](https://www.nature.com/articles/s41467-021-21194-4)。',
        '- 公式データ：[MXfold2 v0.1.0 Rivas.tar.gz](https://github.com/mxfold/mxfold2/releases/download/v0.1.0/Rivas.tar.gz)。',
        '- 主要設定を照合した実行ファイル：/gs/bs/tga-satolab-gtex/yamauchi/TrainA_only_new_parameter/train-new.sh。',
        '- FY実装：' + str(OLD / 'source/mxfold2/loss/fy_loss.py') + '。',
        '- 診断中のnegative clean gapはbpRNA訓練サンプル1例（bpRNA_RFAM_30844）で観測した。通常推論の内部ループ上限30と参照制約評価の上限なしの差があり、loss内訳にはそのまま残した。これを主要な精度低下原因と断定していない。',
    ]
    (OUT / 'REPORT_ja.md').write_text('\n'.join(lines)+'\n')
    base.write_json(OUT / 'validation.json', {'success': True, 'models': native_validation,
        'checkpoints_unchanged': {spec['model']: base.sha(spec['checkpoint']) == spec['checkpoint_sha256'] for spec in plan['models']},
        'all_turner_tables_identical': weight['all_turner_tables_identical'],
        'no_validation_curve_invented': True})
    files = [Path(__file__), ROOT / 'scripts/efold_hf/pretrain_diagnosis.py',
             OLD / 'baseline.conf', OLD / 'source/mxfold2/loss/fy_loss.py',
             OLD / 'source/mxfold2/fold/mix.py', ROOT / 'mxfold2/models/TrainSetAB.conf',
             Path('/gs/bs/tga-satolab-gtex/yamauchi/TrainA_only_new_parameter/train-new.sh')]
    base.write_json(OUT / 'implementation.json', {str(p): base.sha(p) for p in files})
    base.write_json(OUT / 'status.json', {'complete': True, 'read_only_diagnosis': True,
        'test_n': {'PDB': 355, 'archiveII': 3370}, 'diagnostic_n_per_model': len(panel),
        'generated_at': datetime.now(ZoneInfo('Asia/Tokyo')).isoformat()})
    print(json.dumps({'complete': True, 'n_per_model': len(panel), 'plots': ['learning_diagnostics.png', 'score_ablation.png']}), flush=True)


if __name__ == '__main__':
    main()
