#!/usr/bin/env python3
"""Re-score full RNAgym DMS A/C observations from saved BPP arrays."""
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
from scripts.chemical_consistency.full_rnagym_saved import SOURCE, LABELS, empty_stats


def main(shard):
    threshold = json.loads((OUT / 'rnagym_full_ac_threshold.json').read_text())['threshold']
    manifest = json.loads((SOURCE / 'test_data/complete.json').read_text())
    input_path = SOURCE / 'test_data' / f'shard-{shard}.parquet'
    if sha(input_path) != manifest['shards'][str(shard)]['sha256']:
        raise RuntimeError('Full RNAgym input shard changed')
    frame = pd.read_parquet(input_path)
    arrays = {}
    for label in LABELS:
        folder = SOURCE / 'evaluation' / label
        marker = json.loads((folder / f'shard-{shard}.json').read_text())
        if not marker['success'] or marker['unique_sequences'] != int(frame.sequence.nunique()):
            raise RuntimeError('Incomplete original prediction shard')
        with np.load(folder / f'shard-{shard}.npz') as d:
            arrays[label] = {'truth': d['DMS_true'], 'pred': d['DMS_pred']}
    for label in LABELS[1:]:
        if not np.array_equal(arrays[label]['truth'], arrays['baseline']['truth']):
            raise RuntimeError('Original model truth arrays differ')
    stats = {label: empty_stats() for label in LABELS}
    offset = 0
    for count, (sequence, group) in enumerate(frame.groupby('sequence', sort=True), 1):
        for row in group.itertuples(index=False):
            if row.modifier != 'DMS':
                continue
            values = np.fromstring(row.reactivity.strip('[]'), sep=',', dtype=np.float64)
            if len(values) != len(sequence) or np.isinf(values).any():
                raise ValueError('Unexpected DMS profile')
            old_valid = ~np.isnan(values)
            observed = values[old_valid]
            bases = np.asarray(list(sequence))[old_valid]
            begin, end = offset, offset + len(observed)
            if not np.array_equal(np.clip(observed, 0, 1), arrays['baseline']['truth'][begin:end]):
                raise RuntimeError('Stored truth does not match source parquet')
            ac = np.isfinite(observed) & (observed > -900) & np.isin(bases, ('A', 'C'))
            v, b = observed[ac], bases[ac]
            labels = np.clip(v, 0, 1) > threshold
            for label in LABELS:
                prediction = arrays[label]['pred'][begin:end][ac]
                if len(prediction) != len(v) or not np.isfinite(prediction).all():
                    raise RuntimeError('Invalid stored prediction')
                s = stats[label]
                s['profiles'] += 1
                s['bases'] += len(v)
                if len(prediction) and np.ptp(prediction) == 0:
                    s['constant_predictions'] += 1
                auc = auc_binary_truth(labels, prediction)
                if auc is not None:
                    s['auc_sum'] += auc
                    s['auc_defined'] += 1
                rho = correlation(v, prediction)
                if rho is not None:
                    s['spearman_sum'] += rho
                    s['spearman_defined'] += 1
                rhos = [correlation(v[b == base], prediction[b == base]) for base in 'AC']
                rhos = [r for r in rhos if r is not None]
                if rhos:
                    s['spearman_ac_sum'] += float(np.mean(rhos))
                    s['spearman_ac_defined'] += 1
            offset = end
        if count % 5000 == 0:
            print(f'shard {shard}: {count} sequences', flush=True)
    if any(offset != len(arrays[label]['truth']) for label in LABELS):
        raise RuntimeError('Not all DMS observations consumed')
    target = OUT / 'rnagym_full_ac' / f'shard-{shard:02d}.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix('.tmp')
    tmp.write_text(json.dumps({'shard': shard, 'threshold': threshold,
                               'test_shard_sha256': sha(input_path), 'stats': stats}, indent=2) + '\n')
    tmp.replace(target)
    print(f'Completed full RNAgym DMS A/C shard {shard}', flush=True)


if __name__ == '__main__':
    main(int(os.environ['SGE_TASK_ID']))
