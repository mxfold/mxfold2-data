#!/usr/bin/env python3
"""Write a readable report from the OpenKnot audit's structured results."""
import json
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
OUT = PROJECT / 'qsub_logs/efold_hf/openknot_20261028'
JOINT = PROJECT / 'qsub_logs/rnagym/mxfold2-trainsetab-joint-20261022'


def number(row, key):
    return f'{float(row[key]):.2f}' if row is not None else '—'


def find(frame, dataset, model, epoch, modifier, normalization='dataset_q90'):
    found = frame[(frame.dataset == dataset) & (frame.model == model) &
                  (frame.epoch == epoch) & (frame.modifier == modifier) &
                  (frame.normalization == normalization)]
    if len(found) > 1:
        raise ValueError('Ambiguous summary')
    return None if found.empty else found.iloc[0]


def table(frame, dataset, models, modifiers):
    header = '| モデル | Epoch | ' + ' | '.join(f'{p} OKS / ECS' for p in modifiers) + ' |'
    lines = [header, '|---|---:|' + '---:|' * len(modifiers)]
    for name, model, epoch in models:
        cells = []
        for probe in modifiers:
            row = find(frame, dataset, model, epoch, probe)
            cells.append(number(row, 'openknot_observed_only') + ' / ' +
                         number(row, 'classic_observed_only'))
        label = '—' if epoch == 0 else str(epoch)
        lines.append(f'| {name} | {label} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def main():
    frame = pd.read_csv(OUT / 'summary.tsv', sep='\t')
    manifest = json.loads((OUT / 'manifest.json').read_text())
    grid = json.loads((JOINT / 'plan.json').read_text())
    expected = [Path(x['root']).name for x in grid['conditions']]
    complete = sorted(x.stem for x in (OUT / 'joint_predictions').glob('*.json'))
    scored = {model.removeprefix('MXfold2_') for model in frame.loc[
        (frame.dataset == 'RNAgym') & frame.model.str.startswith('MXfold2_'), 'model']}
    missing = sorted(set(expected) - scored)
    original = [('既存事前学習', 'pretrain', 10), ('著者 eFold', 'author_eFold', 0),
                ('著者 EternaFold', 'author_EternaFold', 0), ('著者 MXFold2', 'author_MXFold2', 0)]
    restored = [('MW Rank', 'mw_rank_restored'), ('IMLE-NLL', 'nll_restored'),
                ('SHAPE-fy', 'shape_fy_restored')]
    superseded = [('元のRank', 'original_rank'), ('元のNLL', 'original_nll'),
                  ('元のSHAPE-fy', 'original_shape_fy'), ('assisted label', 'original_assisted')]
    lines = [
        '# OpenKnot score の再集計', '',
        '反応性との全体的な一致には、OpenKnotの構成要素であるEterna Classic Score（ECS）を併記する。',
        '参照: [OpenKnot公式MATLAB実装](https://github.com/eternagame/OpenKnotScoreMATLAB)、'
        '[公式Python実装](https://github.com/eternagame/OpenKnotScorePipeline)。',
        'OpenKnot per-model score（OKS）は 0.5 × (ECS + crossed-pair quality: CPQ)。',
        '元のOpenKnotは候補構造のアンサンブルで配列の疑似結び目らしさを評価するものであり、ここではモデル別の予測を個別に評価する。',
        '疑似結び目を出さない予測のCPQは公式実装どおり0で、そのOKSはECSの半分になる。', '',
        '表は0–100点、配列ごとの平均。測定された塩基のみを分母に含む **observed-only適応版**。',
        '公式コードは指定された連続区間の欠測位置も分母に入れるため、その値はsummary.tsvの `*_official_span` 列に別途保存。',
        '単一塩基対は公式MATLABの既定に従い除去。各データ・試薬で測定値の全体90パーセンタイルが1になるように割った。',
        '元の値のままの結果も `normalization=input` として保存。DMSはA/Cを合わせ、SHAPE系は全塩基で評価する。'
        'DMSへのSHAPE由来閾値の適用は探索的である。',
        '0.125–0.5の領域はpairedでもunpairedでもECS上の一致になる。閾値のスケールや元データの正規化方法に依存する。', '',
        '## eFoldデータ：Viral mRNA', '',
        'SHAPEは15配列（Alphavirus 12、Hepatitis 2、HIV 1）、DMSはSARS由来25配列。PDB、ArchiveII、lncRNAには評価可能な反応性が同梱されていない。',
        '3種類のSHAPE実験を試薬別に一括正規化しているため、元の測定実験全体を用いたOpenKnot値と同一とは限らない。',
        '[Viralデータ作成コード](https://github.com/rouskinlab/efold_data/blob/f5c89cc26d743a8e5d09f4b7b29699912849396c/viral_fragments/viral_fragments.ipynb)'
        'では、参照構造と反応性の一致度によって領域を選別している。', '',
        table(frame, 'eFold viral', original + [(n, m, 5) for n, m in restored] +
              [(n, m, 5) for n, m in superseded], ['SHAPE', 'DMS']), '',
        '元の条件とassisted labelは初回実験で、FT開始時にoptimizerを新規作成した。'
        '再実験の3条件は事前学習時のoptimizer状態を復元している。', '',
        '### 途中エポック（事前学習後から復元した再実験）', '',
        table(frame, 'eFold viral', [(n, m, e) for e in range(1, 6) for n, m in restored],
              ['SHAPE', 'DMS']), '',
        '元の条件の1–5 epochと、元値を使った場合、CPQおよび公式区間ベースの値はsummary.tsvに収録している。',
        'DMSに対して全塩基を非対合としただけのECSは95.84点。'
        '実際、元のNLLの5 epoch目はViral 40配列すべてで予測塩基対が0だが、DMSのECSは95.84点となる。',
        'この例のように、ECSの高さをそれだけで構造精度と解釈できない。'
        'この対照値は `all_unpaired_classic` に収録した。', '',
        '## 以前のEternaFold / MXfold2比較：RNAgymテスト', '',
        '以前の比較と同じ1,024配列・複数測定を使用し、同一配列の繰り返し測定を先に平均する。',
        '試薬ごとの件数は2A3 825、DMS 765、1M7 61、NMIA 5配列。',
        'EternaFoldは保存済みMEA（γ=6）構造、MXfold2は保存チェックポイントから再生成したMAP構造。',
        '以前の相関評価は塩基対確率からの非対合確率を使い、DMSはAとCを分けて集計したため、今回の単一構造に対するスコアとは異なる。',
        'EternaFoldの「—」はL-BFGSによる学習であり、MXfold2のepoch数を適用できないことを示す。', '',
        table(frame, 'RNAgym', [('EternaFold: 事前学習', 'EternaFold_pretrain', 0),
                                ('EternaFold: 2A3合同', 'EternaFold_joint_synthetic_2A3', 0),
                                ('EternaFold: DMS合同', 'EternaFold_joint_synthetic_DMS', 0),
                                ('EternaFold: RNAgym合同', 'EternaFold_joint_rnagym_all', 0)] +
                                [(x, 'MXfold2_' + x, 100) for x in expected],
              ['2A3', 'DMS', '1M7', 'NMIA']), '',
        f'MXfold2予測再生成の完了: {len(complete)} / {len(expected)} 条件。'
        f'表への集計完了: {len(scored)} / {len(expected)} 条件。',
        ('未集計: ' + ', '.join(missing) if missing else 'すべて完了。'), '',
        '## ファイルと検証', '',
        '- summary.tsv: 全条件・全epoch・試薬別のモデル平均とcluster平均。',
        '- per_observation.jsonl: 反応性測定ごとのECS、CPQ、OKS、測定済み塩基数。',
        '- manifest.json: ソースと入力のハッシュ、正規化係数、公式実装のコミット。',
        '- joint_predictions/: 復元モデルの予測構造と、過去のSpearman評価との照合記録。',
        '- 公式Python実装と27ケースで区間・欠測・交差塩基対の一致を確認し、個別単体試験を5件実施。',
        '', '正規化の除数: `' + json.dumps(manifest['normalization'], ensure_ascii=False) + '`', '']
    (OUT / 'REPORT_ja.md').write_text('\n'.join(lines))
    print(f'Wrote report; {len(complete)}/{len(expected)} predictions and '
          f'{len(scored)}/{len(expected)} scores complete')


if __name__ == '__main__':
    main()
