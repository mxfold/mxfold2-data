#!/usr/bin/env python3
"""Add AUROC/Spearman to the saved 700k-profile RNAgym comparison.

The stored NPZ contains clipped truth and predictions but no per-RNA offsets.
Reconstruct those offsets in the exact sorted group order of the original
inference script and recover raw, unclipped reactivity from its frozen parquet.
"""
from __future__ import annotations

from collections import defaultdict
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.chemical_consistency.metrics import auc_binary_truth, correlation
from scripts.chemical_consistency.prepare import OUT, sha

SOURCE = ROOT / 'qsub_logs/rnagym/replay20-full-test-20261023'
LABELS = ('baseline', 'rank_replay10_epoch3', 'rank_replay20_epoch3',
          'nll_replay10_epoch3', 'nll_replay20_epoch3',
          'shape_fy_replay10_epoch3', 'shape_fy_replay20_epoch3')
PROBES = ('DMS', '2A3')


def empty_stats():
    return {'profiles': 0, 'bases': 0, 'auc_sum': 0., 'auc_defined': 0,
            'spearman_sum': 0., 'spearman_defined': 0, 'spearman_ac_sum': 0.,
            'spearman_ac_defined': 0, 'constant_predictions': 0}


def run(shard):
    meta = json.loads((SOURCE / 'test_data/complete.json').read_text())
    input_path = SOURCE / 'test_data' / f'shard-{shard}.parquet'
    if sha(input_path) != meta['shards'][str(shard)]['sha256']:
        raise RuntimeError('Full RNAgym input shard changed')
    frame = pd.read_parquet(input_path)
    arrays, cutoffs = {}, {}
    for label in LABELS:
        folder = SOURCE / 'evaluation' / label
        marker = json.loads((folder / f'shard-{shard}.json').read_text())
        if not marker['success'] or marker['unique_sequences'] != int(frame.sequence.nunique()):
            raise RuntimeError('Incomplete original prediction shard: ' + label)
        with np.load(folder / f'shard-{shard}.npz') as d:
            arrays[label] = {k: d[k] for k in d.files}
        metrics = json.loads((folder / 'metrics.json').read_text())
        cutoffs[label] = {p: float(metrics['chemical'][p]['true_median']) for p in PROBES}
    for label in LABELS[1:]:
        for probe in PROBES:
            if not np.array_equal(arrays[label][probe + '_true'], arrays['baseline'][probe + '_true']):
                raise RuntimeError('Original model truth arrays differ')
            if cutoffs[label][probe] != cutoffs['baseline'][probe]:
                raise RuntimeError('Original truth medians differ')
    offsets = dict.fromkeys(PROBES, 0)
    stats = {label: {probe: empty_stats() for probe in PROBES} for label in LABELS}
    for count, (sequence, group) in enumerate(frame.groupby('sequence', sort=True), 1):
        for row in group.itertuples(index=False):
            probe = row.modifier
            values = np.fromstring(row.reactivity.strip('[]'), sep=',', dtype=np.float64)
            if len(values) != len(sequence) or np.isinf(values).any():
                raise ValueError('Unexpected full RNAgym reactivity')
            valid = ~np.isnan(values)
            values = values[valid]
            bases = np.asarray(list(sequence))[valid]
            begin, end = offsets[probe], offsets[probe] + len(values)
            if not np.array_equal(np.clip(values, 0., 1.), arrays['baseline'][probe + '_true'][begin:end]):
                raise RuntimeError('Stored truth does not match parquet reconstruction')
            labels = np.clip(values, 0., 1.) > cutoffs['baseline'][probe]
            for label in LABELS:
                prediction = arrays[label][probe + '_pred'][begin:end]
                if len(prediction) != len(values):
                    raise RuntimeError('Stored prediction length mismatch')
                s = stats[label][probe]
                s['profiles'] += 1
                s['bases'] += len(values)
                if len(prediction) and np.ptp(prediction) == 0:
                    s['constant_predictions'] += 1
                auc = auc_binary_truth(labels, prediction)
                if auc is not None:
                    s['auc_sum'] += auc
                    s['auc_defined'] += 1
                rho = correlation(values, prediction)
                if rho is not None:
                    s['spearman_sum'] += rho
                    s['spearman_defined'] += 1
                if probe == 'DMS':
                    base_rhos = [correlation(values[bases == b], prediction[bases == b]) for b in 'AC']
                    base_rhos = [x for x in base_rhos if x is not None]
                    if base_rhos:
                        s['spearman_ac_sum'] += float(np.mean(base_rhos))
                        s['spearman_ac_defined'] += 1
            offsets[probe] = end
        if count % 5000 == 0:
            print(f'shard {shard}: {count} sequences', flush=True)
    for label in LABELS:
        for probe in PROBES:
            if offsets[probe] != len(arrays[label][probe + '_true']):
                raise RuntimeError('Not all original observations were consumed')
    target = OUT / 'rnagym_full_saved' / f'shard-{shard:02d}.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + '.tmp')
    tmp.write_text(json.dumps({'shard': shard, 'test_shard_sha256': sha(input_path),
                               'thresholds': cutoffs['baseline'], 'stats': stats}, indent=2) + '\n')
    tmp.replace(target)
    print(f'Completed full RNAgym shard {shard}', flush=True)


if __name__ == '__main__':
    run(int(os.environ['SGE_TASK_ID']))
