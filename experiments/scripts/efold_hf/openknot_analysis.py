#!/usr/bin/env python3
"""Compare model structures against probing data using OpenKnot components.

The official OpenKnot per-structure formula is 0.5 * (Eterna classic + CPQ).
This script preserves the official contiguous-region denominator and also
reports an explicitly labeled observed-only adaptation for sparse measurements.
It does not ensemble different models, since those are being compared.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.feather as feather
import pyarrow as pa

PROJECT = Path(__file__).resolve().parents[2]
OLD = PROJECT / 'qsub_logs/efold_hf/bprna100_ft5_replay20_20261024'
RESTORED = PROJECT / 'qsub_logs/efold_hf/bprna10_restored_ft5_replay20_20261026'
ALIGNED = PROJECT / 'qsub_logs/efold_hf/paper_aligned_20261028'
ETERNA = PROJECT / 'qsub_logs/eternafold/eternafold-joint-20261022'
ETERNA_OLD = PROJECT / 'qsub_logs/eternafold/eternafold-traina-branches-20261022'
OUT = PROJECT / 'qsub_logs/efold_hf/openknot_20261028'
SUPPORTED_BASES = {'DMS': 'AC', 'CMCT': 'GU', 'SHAPE': 'ACGU', '2A3': 'ACGU',
                   '1M7': 'ACGU', 'NMIA': 'ACGU'}


def read_jsonl(path):
    with path.open() as f:
        for line in f:
            yield json.loads(line)


def valid_values(sequence, modifier, values):
    values = np.asarray(values, dtype=float)
    if len(sequence) != len(values):
        raise ValueError('sequence/reactivity length mismatch')
    if modifier not in SUPPORTED_BASES:
        raise ValueError('unknown modifier ' + modifier)
    valid = np.isfinite(values) & (values > -900)
    valid &= np.array([base in SUPPORTED_BASES[modifier] for base in sequence])
    return valid


def crossing_residues(pairs):
    # Most prediction methods here never produce pseudoknots. Detect their
    # properly nested pairing in linear time before the general comparison.
    stack = []
    nested = True
    for a, b in sorted(pairs):
        while stack and stack[-1] < a:
            stack.pop()
        if stack and b > stack[-1]:
            nested = False
            break
        stack.append(b)
    if nested:
        return set()
    result = set()
    for i, (a, b) in enumerate(pairs):
        for c, d in pairs[i + 1:]:
            if a < c < b < d or c < a < d < b:
                result.update((a, b, c, d))
    return result


def sanitize_pairs(pairs, length, remove_singlets=True):
    pairs = {(int(a), int(b)) for a, b in pairs}
    if any(not (0 <= a < b < length) for a, b in pairs):
        raise ValueError('invalid base pair')
    sites = [i for pair in pairs for i in pair]
    if len(sites) != len(set(sites)):
        raise ValueError('multiple pairing partners')
    if remove_singlets:
        pairs = {(a, b) for a, b in pairs
                 if (a - 1, b + 1) in pairs or (a + 1, b - 1) in pairs}
    return sorted(pairs)


def score(sequence, pairs, values, valid, remove_singlets=True):
    """Apply MATLAB/official Python per-model formula with two denominators.

    Missing sites count as failures in official_span, matching both reference
    implementations. observed_only excludes missing sites from both component
    denominators. The scoring span is the first through last eligible site.
    """
    if not np.any(valid):
        return None
    pairs = sanitize_pairs(pairs, len(sequence), remove_singlets)
    paired = np.zeros(len(sequence), dtype=bool)
    for a, b in pairs:
        paired[a] = paired[b] = True
    start, end = int(np.flatnonzero(valid)[0]), int(np.flatnonzero(valid)[-1])
    span = np.zeros(len(sequence), dtype=bool)
    span[start:end + 1] = True
    hit = valid & ((paired & (values < .5)) | (~paired & (values > .125)))
    crossed = crossing_residues(pairs)
    within = [(a, b) for a, b in pairs if start <= a and b <= end]
    internal_crossed = crossing_residues(within)
    crossed_span = [i for i in crossed if span[i]]
    crossed_valid = [i for i in crossed_span if valid[i]]
    quality_hits = sum((1. if i in internal_crossed else .5)
                       for i in crossed_valid if values[i] < .25)
    classic_span = 100 * int(hit.sum()) / int(span.sum())
    classic_observed = 100 * int(hit.sum()) / int(valid.sum())
    cpq_span = 100 * quality_hits / len(crossed_span) if crossed_span else 0.
    cpq_observed = 100 * quality_hits / len(crossed_valid) if crossed_valid else 0.
    return {'scored_bases': int(valid.sum()), 'span_bases': int(span.sum()),
            'paired_fraction_observed': float((paired & valid).sum() / valid.sum()),
            'ambiguous_fraction_observed': float((valid & (values > .125) & (values < .5)).sum() / valid.sum()),
            'all_unpaired_classic': float(100 * (valid & (values > .125)).sum() / valid.sum()),
            'all_paired_classic': float(100 * (valid & (values < .5)).sum() / valid.sum()),
            'crossed_bases': len(crossed_span), 'observed_crossed_bases': len(crossed_valid),
            'classic_official_span': classic_span, 'cpq_official_span': cpq_span,
            'openknot_official_span': .5 * (classic_span + cpq_span),
            'classic_observed_only': classic_observed, 'cpq_observed_only': cpq_observed,
            'openknot_observed_only': .5 * (classic_observed + cpq_observed)}


def bpseq_pairs(path, sequence):
    pairs = []
    lines = path.read_text().splitlines()
    if len(lines) != len(sequence):
        raise ValueError('BPSEQ length mismatch: ' + str(path))
    partners = []
    for i, line in enumerate(lines):
        idx, base, j = line.split()[:3]
        if int(idx) != i + 1 or base.upper().replace('T', 'U') != sequence[i]:
            raise ValueError('BPSEQ sequence mismatch: ' + str(path))
        partners.append(int(j) - 1)
    for i, j in enumerate(partners):
        if j >= 0 and (j >= len(partners) or partners[j] != i):
            raise ValueError('asymmetric BPSEQ: ' + str(path))
        if j > i:
            pairs.append((i, j))
    return pairs


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def add_scores(output, dataset, model, epoch, sample_id, sequence, pairs,
               modifier, raw_values, cluster, scale):
    values = np.asarray(raw_values, dtype=float)
    valid = valid_values(sequence, modifier, values)
    if valid.sum() < 2:
        return
    for normalization, factor in [('input', 1.), ('dataset_q90', scale)]:
        result = score(sequence, pairs, values / factor, valid)
        output.append({'dataset': dataset, 'model': model, 'epoch': epoch,
                       'sample_id': sample_id, 'sequence_sha256': hashlib.sha256(sequence.encode()).hexdigest(),
                       'cluster': cluster, 'modifier': modifier, 'normalization': normalization,
                       'normalization_divisor': factor, **result})


def empirical_q90(observations):
    values = defaultdict(list)
    for obs in observations:
        arr = np.asarray(obs['values'], dtype=float)
        mask = valid_values(obs['sequence'], obs['modifier'], arr)
        values[obs['modifier']].extend(arr[mask].tolist())
    result = {k: float(np.percentile(v, 90)) for k, v in values.items()}
    if any(v <= 0 or not math.isfinite(v) for v in result.values()):
        raise ValueError('invalid 90th percentile')
    return result


def efold_scores():
    tests = {x['id']: x for x in read_jsonl(OLD / 'data/test.jsonl') if x['observations']}
    observations = [{'sequence': item['sequence'], 'modifier': o['probe'].upper(), 'values': o['values']}
                    for item in tests.values() for o in item['observations']]
    scales = empirical_q90(observations)
    output, predicted = [], {}
    paths = [('pretrain', 10, OLD / 'evaluation/pretrain/epoch-010/per_sequence.jsonl')]
    for arm in ('mw_rank_restored', 'nll_restored', 'shape_fy_restored'):
        paths += [(arm, e, RESTORED / f'evaluation/{arm}/epoch-{e:03d}/per_sequence.jsonl')
                  for e in range(1, 6)]
    # Retain the original, subsequently superseded experiments as separate runs.
    for arm in ('rank', 'nll', 'shape_fy', 'assisted'):
        paths += [('original_' + arm, e, OLD / f'evaluation/{arm}/epoch-{e:03d}/per_sequence.jsonl')
                  for e in range(1, 6)]
    for model, epoch, path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        selected = {r['id']: r['predicted_pairs'] for r in read_jsonl(path) if r['id'] in tests}
        if set(selected) != set(tests):
            raise ValueError('Incomplete evaluation coverage: ' + str(path))
        predicted[(model, epoch)] = selected
    author = {}
    for path in sorted((ALIGNED / 'author_efold_paper_gpu').glob('shard-*.jsonl')):
        author.update({r['id']: r['predicted_pairs'] for r in read_jsonl(path) if r['id'] in tests})
    if set(author) != set(tests):
        raise ValueError('Incomplete eFold author predictions')
    predicted[('author_eFold', 0)] = author
    t = feather.read_table(PROJECT / 'data/efold_hf/paper_2026/results_benchmark_algos.feather',
                           columns=['reference', 'dataset', 'model', 'structure'])
    t = t.filter(pc.and_(pc.equal(t['dataset'], 'viral_fragments'),
                         pc.is_in(t['model'], value_set=pa.array(['EternaFold', 'MXFold2']))))
    for model in ('EternaFold', 'MXFold2'):
        selected = {'viral_fragments:' + row['reference']: row['structure'] for row in t.to_pylist()
                    if row['model'] == model and 'viral_fragments:' + row['reference'] in tests}
        if set(selected) != set(tests):
            raise ValueError('Incomplete author baseline predictions: ' + model)
        predicted[('author_' + model, 0)] = selected
    for (model, epoch), predictions in predicted.items():
        for item_id, item in tests.items():
            for o in item['observations']:
                modifier = o['probe'].upper()
                add_scores(output, 'eFold viral', model, epoch, item_id, item['sequence'],
                           predictions[item_id], modifier, o['values'], item_id, scales[modifier])
    return output, scales


def rnagym_scores():
    frame = pd.read_parquet(ETERNA / 'data/test_panel.parquet')
    entries = json.loads((ETERNA / 'data/evaluation_entries.json').read_text())
    by_sequence = {x['sequence']: x['name'] for x in entries
                   if x['kind'] == 'chemical' and len(x['sequence']) <= 1000}
    frame = frame[frame.sequence.isin(by_sequence)].copy()
    observations = []
    for row in frame.itertuples(index=False):
        values = np.fromstring(row.reactivity.strip().strip('[]'), sep=',')
        observations.append({'sequence': row.sequence, 'modifier': row.modifier, 'values': values,
                             'sample_id': row.seqID, 'cluster': row.cluster})
    scales = empirical_q90(observations)
    conditions = {'pretrain': ETERNA_OLD / 'evaluation/pretrain/bpseq',
                  **{c: ETERNA / f'evaluation/{c}/bpseq' for c in
                     ('joint_synthetic_2A3', 'joint_synthetic_DMS', 'joint_rnagym_all')}}
    output = []
    for condition, folder in conditions.items():
        predictions = {seq: bpseq_pairs(folder / name, seq) for seq, name in by_sequence.items()}
        for obs in observations:
            # EternaFold was optimized by L-BFGS, so MXfold2 epochs do not apply.
            add_scores(output, 'RNAgym', 'EternaFold_' + condition, 0, obs['sample_id'],
                       obs['sequence'], predictions[obs['sequence']], obs['modifier'], obs['values'],
                       obs['cluster'], scales[obs['modifier']])
    # A prior MXfold2 comparison stored probabilities but not structures. These
    # optional, verified reconstructions are appended as each job completes.
    joint = OUT / 'joint_predictions'
    expected = {hashlib.sha256(seq.encode()).hexdigest() for seq in by_sequence}
    for metadata_path in sorted(joint.glob('*.json')):
        metadata = json.loads(metadata_path.read_text())
        condition = metadata['condition']
        path = joint / (condition + '.jsonl')
        if (metadata['panel_sha256'] != sha256(ETERNA / 'data/test_panel.parquet') or
                metadata['output_sha256'] != sha256(path) or metadata['verified_spearman_cases'] < 5):
            raise ValueError('Unverified joint predictions: ' + condition)
        predictions = {row['sequence_sha256']: row['predicted_pairs'] for row in read_jsonl(path)}
        if set(predictions) != expected:
            raise ValueError('Incomplete joint predictions: ' + condition)
        for obs in observations:
            seq_hash = hashlib.sha256(obs['sequence'].encode()).hexdigest()
            add_scores(output, 'RNAgym', 'MXfold2_' + condition, 100, obs['sample_id'],
                       obs['sequence'], predictions[seq_hash], obs['modifier'], obs['values'],
                       obs['cluster'], scales[obs['modifier']])
    return output, scales


def summarize(rows):
    frame = pd.DataFrame(rows)
    keys = ['dataset', 'model', 'epoch', 'modifier', 'normalization']
    metrics = ['classic_official_span', 'cpq_official_span', 'openknot_official_span',
               'classic_observed_only', 'cpq_observed_only', 'openknot_observed_only']
    metrics += ['paired_fraction_observed', 'ambiguous_fraction_observed',
                'all_unpaired_classic', 'all_paired_classic']
    sequence_keys = keys + ['cluster', 'sequence_sha256']
    per_sequence = frame.groupby(sequence_keys, dropna=False)[metrics].mean().reset_index()
    means = per_sequence.groupby(keys, dropna=False)[metrics].mean()
    counts = per_sequence.groupby(keys, dropna=False).size().rename('sequences')
    clusters = per_sequence.groupby(keys + ['cluster'], dropna=False)[metrics].mean()
    cluster_means = clusters.groupby(keys, dropna=False).mean().add_prefix('cluster_macro_')
    return means.join(counts).join(cluster_means).reset_index()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    e, e_scales = efold_scores()
    r, r_scales = rnagym_scores()
    rows = e + r
    with (OUT / 'per_observation.jsonl').open('w') as f:
        for row in rows:
            f.write(json.dumps(row, allow_nan=False) + '\n')
    summarize(rows).to_csv(OUT / 'summary.tsv', sep='\t', index=False, float_format='%.6f')
    info = {'official_python_commit': '23e24fe21a3ebe50046b0cce29bcb1c4a1a571b8',
            'official_matlab_commit': '6d1c413baa571be06e8dbb1d3ea9a544c36083d4',
            'official_matlab_singlet_filter': True,
            'normalization': {'eFold viral': e_scales, 'RNAgym': r_scales},
            'counts': dict(Counter((x['dataset'] + '/' + x['modifier']) for x in rows)),
            'length_limit': 1000,
            'metrics': 'per-model OpenKnot=(Eterna classic+crossed-pair quality)/2; '
                       'not the multi-model ensemble; both official contiguous-span and observed-only adaptation',
            'source_sha256': sha256(Path(__file__)),
            'rnagym_panel_sha256': sha256(ETERNA / 'data/test_panel.parquet'),
            'efold_test_sha256': sha256(OLD / 'data/test.jsonl')}
    (OUT / 'manifest.json').write_text(json.dumps(info, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(info, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
