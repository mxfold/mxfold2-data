#!/usr/bin/env python3
"""Find the exact DMS A/C-only global median for the full RNAgym test."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.chemical_consistency.prepare import OUT, sha
from scripts.chemical_consistency.full_rnagym_saved import SOURCE


def main():
    manifest = json.loads((SOURCE / 'test_data/complete.json').read_text())
    selected = []
    total = 0
    for shard in range(1, 17):
        path = SOURCE / 'test_data' / f'shard-{shard}.parquet'
        if sha(path) != manifest['shards'][str(shard)]['sha256']:
            raise RuntimeError('Full RNAgym input shard changed')
        frame = pd.read_parquet(path)
        for row in frame[frame.modifier == 'DMS'].itertuples(index=False):
            values = np.fromstring(row.reactivity.strip('[]'), sep=',', dtype=np.float64)
            if len(values) != len(row.sequence):
                raise ValueError('Unexpected reactivity length')
            bases = np.asarray(list(row.sequence))
            mask = np.isfinite(values) & (values > -900) & np.isin(bases, ('A', 'C'))
            selected.append(np.clip(values[mask], 0, 1))
            total += int(mask.sum())
        print(f'shard {shard}: {total} DMS A/C observations', flush=True)
    values = np.concatenate(selected)
    if len(values) != total or not len(values):
        raise RuntimeError('No DMS A/C observations')
    info = {'threshold': float(np.median(values)), 'bases': total, 'shards': 16,
            'definition': 'finite DMS A/C reactivity > -900; clip to [0,1]; global median'}
    target = OUT / 'rnagym_full_ac_threshold.json'
    tmp = target.with_suffix('.tmp')
    tmp.write_text(json.dumps(info, indent=2) + '\n')
    tmp.replace(target)
    print(json.dumps(info), flush=True)


if __name__ == '__main__':
    main()
