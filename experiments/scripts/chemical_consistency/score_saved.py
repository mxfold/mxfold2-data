#!/usr/bin/env python3
"""Score already-saved MAP structures and EternaFold base-pair probabilities."""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as feather

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.chemical_consistency.metrics import auc_binary_truth, score_profile, valid_mask
from scripts.chemical_consistency.prepare import OUT, seq_hash, sha
from scripts.rnagym.evaluate_rnagym import unpaired_from_bpp


def load_profiles():
    profiles = [json.loads(line) for line in (OUT / 'profiles.jsonl').open()]
    info = json.loads((OUT / 'panel_manifest.json').read_text())
    if sha(OUT / 'profiles.jsonl') != info['profiles_sha256']:
        raise ValueError('Panel hash changed')
    return profiles, info['thresholds_clipped_median']


def from_pairs(sequence, pairs):
    prediction = np.ones(len(sequence), dtype=float)
    occupied = set()
    for a, b in pairs:
        a, b = int(a), int(b)
        if not 0 <= a < b < len(sequence) or a in occupied or b in occupied:
            raise ValueError('Invalid predicted base pair')
        prediction[a] = prediction[b] = 0.
        occupied.update((a, b))
    return prediction


def score_predictions(model, epoch, kind, predictions, profiles, cuts, provenance):
    records, pooled = [], defaultdict(lambda: ([], []))
    for item in profiles:
        key = item['sequence_sha256']
        if key not in predictions:
            continue
        prediction = np.asarray(predictions[key], dtype=float)
        dataset, modifier = item['dataset'], item['modifier']
        cutoff = cuts[dataset + '/' + modifier]
        metrics = score_profile(item, prediction, cutoff)
        row = {'dataset': dataset, 'model': model, 'epoch': epoch, 'kind': kind,
               'sample_id': item['sample_id'], 'sequence_sha256': key,
               'cluster': item['cluster'], 'modifier': modifier, 'threshold': cutoff, **metrics}
        records.append(row)
        mask = valid_mask(item['sequence'], modifier, item['values'], prediction)
        values = np.asarray(item['values'], dtype=float)[mask]
        truth, pred = pooled[(dataset, modifier)]
        truth.extend((np.clip(values, 0, 1) > cutoff).tolist())
        pred.extend(np.clip(prediction[mask], 0, 1).tolist())
    if not records:
        raise ValueError('No matching profiles for ' + model)
    frame = pd.DataFrame(records)
    summaries = []
    for (dataset, modifier), group in frame.groupby(['dataset', 'modifier'], sort=True):
        # RNAgym replicates share one sequence, while TrainSetB has distinct
        # annotated samples even when two structures use the same sequence.
        unit = 'sample_id' if dataset == 'TrainSetB_simulated_training' else 'sequence_sha256'
        values = group.groupby([unit, 'cluster'], dropna=False)[['auroc', 'spearman_all',
            'spearman_probe_balanced']].mean().reset_index()
        clusters = values.groupby('cluster')[['auroc', 'spearman_all', 'spearman_probe_balanced']].mean()
        truth, pred = pooled[(dataset, modifier)]
        summary = {'dataset': dataset, 'model': model, 'epoch': epoch, 'kind': kind,
                   'modifier': modifier, 'threshold': cuts[dataset + '/' + modifier],
                   'profiles': len(group), 'units': len(values), 'n_bases': len(truth),
                   'auroc_micro': auc_binary_truth(truth, pred),
                   'auroc_sequence_macro': float(values.auroc.mean()) if values.auroc.notna().any() else None,
                   'auroc_cluster_macro': float(clusters.auroc.mean()) if clusters.auroc.notna().any() else None,
                   'n_units_auc_defined': int(values.auroc.notna().sum()),
                   'spearman_sequence_macro': float(values.spearman_probe_balanced.mean()) if values.spearman_probe_balanced.notna().any() else None,
                   'spearman_cluster_macro': float(clusters.spearman_probe_balanced.mean()) if clusters.spearman_probe_balanced.notna().any() else None,
                   'spearman_all_sequence_macro': float(values.spearman_all.mean()) if values.spearman_all.notna().any() else None,
                   'n_units_spearman_defined': int(values.spearman_probe_balanced.notna().sum()),
                   'n_prediction_constant': int(group.prediction_constant.sum())}
        summaries.append(summary)
    dest = OUT / 'metrics' / kind / model / f'epoch-{epoch:03d}'
    dest.mkdir(parents=True, exist_ok=True)
    temp = dest / 'profiles.jsonl.tmp'
    with temp.open('w') as f:
        for row in records:
            f.write(json.dumps(row, allow_nan=False) + '\n')
    temp.replace(dest / 'profiles.jsonl')
    info = {'model': model, 'epoch': epoch, 'kind': kind, 'provenance': provenance,
            'profiles_sha256': sha(dest / 'profiles.jsonl'), 'summary': summaries}
    tmp = dest / 'summary.json.tmp'
    tmp.write_text(json.dumps(info, indent=2, allow_nan=False) + '\n')
    tmp.replace(dest / 'summary.json')
    return info


def efold_map(profiles, cuts):
    relevant = [p for p in profiles if p['dataset'] == 'eFold_viral']
    seqs = {p['sample_id']: p['sequence'] for p in relevant}
    old = ROOT / 'qsub_logs/efold_hf/bprna100_ft5_replay20_20261024'
    new = ROOT / 'qsub_logs/efold_hf/bprna10_restored_ft5_replay20_20261026'
    aligned = ROOT / 'qsub_logs/efold_hf/paper_aligned_20261028'
    paths = [('pretrain', 10, old / 'evaluation/pretrain/epoch-010/per_sequence.jsonl')]
    for arm in ('mw_rank_restored', 'nll_restored', 'shape_fy_restored'):
        paths += [(arm, e, new / f'evaluation/{arm}/epoch-{e:03d}/per_sequence.jsonl') for e in range(1, 6)]
    for arm in ('rank', 'nll', 'shape_fy', 'assisted'):
        paths += [('original_' + arm, e, old / f'evaluation/{arm}/epoch-{e:03d}/per_sequence.jsonl') for e in range(1, 6)]
    for model, epoch, path in paths:
        pred = {}
        with path.open() as f:
            for line in f:
                row = json.loads(line)
                if row['id'] in seqs:
                    sequence = seqs[row['id']]
                    pred[seq_hash(sequence)] = from_pairs(sequence, row['predicted_pairs'])
        if len(pred) != len(seqs):
            raise ValueError('Incomplete eFold MAP predictions: ' + str(path))
        score_predictions(model, epoch, 'MAP', pred, relevant, cuts, {'path': str(path), 'sha256': sha(path)})
    pred = {}
    for path in (aligned / 'author_efold_paper_gpu').glob('shard-*.jsonl'):
        for line in path.open():
            row = json.loads(line)
            if row['id'] in seqs:
                sequence = seqs[row['id']]
                pred[seq_hash(sequence)] = from_pairs(sequence, row['predicted_pairs'])
    if len(pred) != len(seqs):
        raise ValueError('Incomplete author eFold predictions')
    score_predictions('author_eFold', 0, 'MAP', pred, relevant, cuts, {'source': 'author paper-protocol GPU'})
    path = ROOT / 'data/efold_hf/paper_2026/results_benchmark_algos.feather'
    t = feather.read_table(path, columns=['reference', 'dataset', 'model', 'structure'])
    t = t.filter(pc.and_(pc.equal(t['dataset'], 'viral_fragments'),
                         pc.is_in(t['model'], value_set=pa.array(['EternaFold', 'MXFold2']))))
    for model in ('EternaFold', 'MXFold2'):
        pred = {}
        for row in t.to_pylist():
            item_id = 'viral_fragments:' + row['reference']
            if row['model'] == model and item_id in seqs:
                sequence = seqs[item_id]
                pred[seq_hash(sequence)] = from_pairs(sequence, row['structure'])
        if len(pred) != len(seqs):
            raise ValueError('Incomplete author comparator: ' + model)
        score_predictions('author_' + model, 0, 'MAP', pred, relevant, cuts, {'path': str(path), 'sha256': sha(path)})


def joint_map(profiles, cuts):
    relevant = [p for p in profiles if p['dataset'] == 'RNAgym_test']
    seqs = {p['sequence_sha256']: p['sequence'] for p in relevant}
    folder = ROOT / 'qsub_logs/efold_hf/openknot_20261028/joint_predictions'
    for meta in sorted(folder.glob('*.json')):
        info = json.loads(meta.read_text())
        path = folder / (info['condition'] + '.jsonl')
        if sha(path) != info['output_sha256']:
            raise ValueError('Predictions changed: ' + str(path))
        pred = {}
        for line in path.open():
            row = json.loads(line)
            pred[row['sequence_sha256']] = from_pairs(seqs[row['sequence_sha256']], row['predicted_pairs'])
        if set(pred) != set(seqs):
            raise ValueError('Incomplete joint MAP: ' + str(path))
        score_predictions('joint_' + info['condition'], 100, 'MAP', pred, relevant, cuts, info)


def eternafold_bpp(profiles, cuts):
    relevant = [p for p in profiles if p['dataset'] == 'RNAgym_test']
    joint = ROOT / 'qsub_logs/eternafold/eternafold-joint-20261022'
    branch = ROOT / 'qsub_logs/eternafold/eternafold-traina-branches-20261022'
    entries = json.loads((joint / 'data/evaluation_entries.json').read_text())
    entries = [e for e in entries if e['kind'] == 'chemical' and len(e['sequence']) <= 1000]
    conditions = [('pretrain', branch, 'pretrain'), ('branch_2A3', branch, 'synthetic_2A3'),
                  ('branch_DMS', branch, 'synthetic_DMS'), ('joint_2A3', joint, 'joint_synthetic_2A3'),
                  ('joint_DMS', joint, 'joint_synthetic_DMS'), ('joint_rnagym', joint, 'joint_rnagym_all')]
    for name, root, condition in conditions:
        folder = root / 'evaluation' / condition / 'bpp'
        pred = {seq_hash(e['sequence']): unpaired_from_bpp(folder / e['name'], len(e['sequence'])) for e in entries}
        info = score_predictions('EternaFold_' + name, 0, 'BPP', pred, relevant, cuts,
                                 {'root': str(root), 'condition': condition,
                                  'prediction_format': 'native text BPP, rounded; threshold 0'})
        # Compare directly against earlier probe-balanced Spearman; original
        # evaluation assigns zero for constant predictions, recorded separately.
        earlier = pd.read_csv(root / 'evaluation' / condition / 'RNAgym.csv')
        saved = pd.read_json(OUT / 'metrics/BPP' / ('EternaFold_' + name) / 'epoch-000/profiles.jsonl', lines=True)
        grouped = saved.groupby(['sequence_sha256', 'modifier']).spearman_probe_balanced.mean()
        diffs = []
        for row in earlier.itertuples(index=False):
            key = (seq_hash(row.sequence), row.modifier)
            if key in grouped and pd.notna(grouped[key]):
                diffs.append(abs(float(grouped[key]) - float(row.spearman)))
        if not diffs or max(diffs) > 1e-6:
            raise RuntimeError(f'EternaFold BPP Spearman mismatch: {name}, max={max(diffs) if diffs else None}')
        print(f'{name}: {len(diffs)} previous Spearman profiles matched; max difference {max(diffs):.3g}', flush=True)


def main():
    profiles, cuts = load_profiles()
    efold_map(profiles, cuts)
    joint_map(profiles, cuts)
    eternafold_bpp(profiles, cuts)


if __name__ == '__main__':
    main()
