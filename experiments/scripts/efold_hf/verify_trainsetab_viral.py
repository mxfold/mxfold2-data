#!/usr/bin/env python3
"""Verify the explicitly requested 2021 TrainSetAB on the eFold viral panel."""
from datetime import datetime
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(Path(__file__).parent)]
import bprnanew_benchmark as base
from paper_methods_mw import paired_union
from scripts.chemical_consistency import map_mw_metrics as metrics
from scripts.chemical_consistency.map_mw_report import common_frames, table, atomic_tsv

OUT = ROOT / 'qsub_logs/efold_hf/paper_trainsetab_viral_20261031'
PAPER = ROOT / 'qsub_logs/efold_hf/paper_methods_mw_20261031'
FULL = ROOT / 'qsub_logs/efold_hf/paper_full_20261030'
NOFAM = ROOT / 'qsub_logs/efold_hf/archive_nofam_20261031'


def main():
    import pandas as pd
    import pyarrow as pa
    import pyarrow.feather as feather
    pa.set_cpu_count(1); pa.set_io_thread_count(1)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [r for r in base.read_jsonl(FULL / 'panel.jsonl') if r['source'] == 'viral_fragments']
    if len(rows) != 40:
        raise ValueError('Expected all forty eFold chemical test entries')
    specs = json.loads((NOFAM / 'plan.json').read_text())['models']
    spec = next(x for x in specs if x['model'] == 'MXfold2_public')
    base.write_jsonl(OUT / 'panel.jsonl', rows)
    panel_sha = base.sha(OUT / 'panel.jsonl')
    base.write_json(OUT / 'plan.json', {'dataset': 'eFold viral test40', 'n': 40,
        'panel_sha256': panel_sha, 'shards_per_model': 1, 'models': [spec],
        'purpose': 'Verify explicitly requested public 2021 TrainSetAB against eFold author-saved MXFold2 predictions',
        'implementation_sha256': base.sha(Path(__file__))})
    base.write_json(OUT / 'prepare-complete.json', {'success': True, 'panel_sha256': panel_sha})
    base.OUT = OUT; base.SHARDS = 1
    base.work(1, smoke=True)
    base.work(1, smoke=False)
    predictions = base.read_jsonl(OUT / 'predictions/MXfold2_public/shard-0.jsonl')
    predicted = {r['id']: r['predicted_pairs'] for r in predictions}
    author_path = ROOT / 'data/efold_hf/paper_2026/results_benchmark_algos.feather'
    author = feather.read_table(author_path, columns=['reference', 'dataset', 'model', 'structure'], use_threads=False).to_pylist()
    saved = {'viral_fragments:' + r['reference']: sorted(map(tuple, r['structure'])) for r in author
             if r['dataset'] == 'viral_fragments' and r['model'] == 'MXFold2'}
    if set(saved) != set(predicted):
        raise ValueError('Author prediction coverage differs')
    different = [k for k, v in predicted.items() if sorted(map(tuple, v)) != saved[k]]
    profiles = [p for p in metrics.load_profiles() if p['dataset'] == 'eFold_viral']
    states = {metrics.sequence_hash(r['sequence']): paired_union(r['sequence'], predicted[r['id']])[0] for r in rows}
    current = metrics.profile_rows(profiles, states)
    current.to_parquet(OUT / 'profiles.parquet', index=False)
    group = ['author_eFold', 'paper_UFold', 'MXfold2_public', 'mx_bprna10',
             'mx_mw_rank_restored', 'paper_EternaFold', 'paper_RNAstructure']
    labels = ['eFold', 'UFold（eFold著者保存予測）', 'MXfold2 2021公開TrainSetAB',
              'MXfold2 bpRNA事前学習10ep', 'MXfold2 MW Rank FT5ep',
              'EternaFold（eFold著者保存予測）', 'RNAstructure（eFold著者保存予測）']
    frames = [current if m == 'MXfold2_public' else pd.read_parquet(PAPER / 'metrics' / m / 'profiles.parquet') for m in group]
    matched = common_frames(frames)
    summaries = [metrics.summarize(f) for f in matched]
    entries = [(label, model, 10 if model == 'mx_bprna10' else 5 if model == 'mx_mw_rank_restored' else 0)
               for label, model in zip(labels, group)]
    rendered, exported = table('指定7モデル：2021公開TrainSetABを直接推論', entries, summaries,
                               ('SHAPE', 'DMS'), 'requested_seven_verified_TrainSetAB', 'eFold_viral')
    rendered = rendered.replace('| 0 |', '| — |')
    atomic_tsv(OUT / 'comparisons.tsv', pd.DataFrame(exported))
    verification = {'complete': True, 'n': 40, 'checkpoint_sha256': spec['checkpoint_sha256'],
                    'author_saved_identical': not different, 'different_ids': different,
                    'author_saved_file_sha256': base.sha(author_path),
                    'generated_at': datetime.now(ZoneInfo('Asia/Tokyo')).isoformat()}
    base.write_json(OUT / 'status.json', verification)
    lines = ['# eFold化学評価：2021公開TrainSetABの確認', '',
        'ユーザー指定の2021年公開MXfold2 TrainSetABを40配列すべてで直接推論した。', '',
        f'eFold著者保存MXFold2予測との不一致は{len(different)}/40構造。', '',
        '新AUROC・Spearmanは既存の定義で計算し、全7モデルで定義できる共通対象に揃えた。塩基対確率は使わない。', '',
        rendered, '',
        '論文の他の比較手法を含む再採点一覧は、../paper_methods_mw_20261031/REPORT_ja.md を参照。']
    (OUT / 'REPORT_ja.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(verification, indent=2), flush=True)


if __name__ == '__main__':
    main()
