#!/usr/bin/env python3
"""Prepare fixed eFold, RNAgym and TrainSetB simulated chemical panels."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.chemical_consistency.metrics import thresholds, valid_mask

OUT = ROOT / 'qsub_logs/chemical_consistency/auroc_spearman_20261028'
EFOLD = ROOT / 'qsub_logs/efold_hf/bprna100_ft5_replay20_20261024/data/test.jsonl'
RNAGYM = ROOT / 'qsub_logs/eternafold/eternafold-joint-20261022/data/test_panel.parquet'
SIM = ROOT / 'qsub_logs/rnagym/trainsetb-synthetic-20261020/data'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def seq_hash(sequence):
    return hashlib.sha256(sequence.encode()).hexdigest()


def make_profiles():
    profiles = []
    for line in EFOLD.open():
        item = json.loads(line)
        for obs in item['observations']:
            profiles.append({'dataset': 'eFold_viral', 'sample_id': item['id'],
                             'sequence': item['sequence'], 'cluster': item['id'],
                             'modifier': obs['probe'].upper(), 'values': obs['values']})
    frame = pd.read_parquet(RNAGYM)
    for row in frame.itertuples(index=False):
        if len(row.sequence) > 1000:
            continue
        values = np.fromstring(row.reactivity.strip().strip('[]'), sep=',')
        if len(values) != len(row.sequence):
            raise ValueError('RNAgym reactivity length mismatch')
        values[~np.isfinite(values)] = -999.
        profiles.append({'dataset': 'RNAgym_test', 'sample_id': row.seqID,
                         'sequence': row.sequence, 'cluster': row.cluster,
                         'modifier': row.modifier, 'values': values.tolist()})
    for probe in ('2A3', 'DMS'):
        folder = SIM / probe
        with (folder / 'sequences.tsv').open() as f:
            meta = list(csv.DictReader(f, delimiter='\t'))
        seqoffs = np.load(folder / 'sequence_offsets.npy', mmap_mode='r')
        obsoffs = np.load(folder / 'observation_offsets.npy', mmap_mode='r')
        vals = np.load(folder / 'values.npy', mmap_mode='r')
        for i, row in enumerate(meta):
            if len(row['sequence']) > 1000:
                continue
            if int(seqoffs[i + 1] - seqoffs[i]) != 1:
                raise ValueError('Expected one synthetic profile per sequence/probe')
            idx = int(seqoffs[i])
            values = np.asarray(vals[int(obsoffs[idx]):int(obsoffs[idx + 1])], dtype=float)
            if len(values) != len(row['sequence']):
                raise ValueError('Synthetic profile length mismatch')
            values[~np.isfinite(values)] = -999.
            profiles.append({'dataset': 'TrainSetB_simulated_training',
                             'sample_id': row['cluster'], 'sequence': row['sequence'],
                             'cluster': row['cluster'], 'modifier': probe, 'values': values.tolist()})
    return profiles


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    profiles = make_profiles()
    cuts = thresholds(profiles)
    stats = {}
    for p in profiles:
        key = p['dataset'] + '/' + p['modifier']
        row = stats.setdefault(key, {'profiles': 0, 'measured_bases': 0, 'sequences': set()})
        row['profiles'] += 1
        row['sequences'].add(p['sequence'])
        row['measured_bases'] += int(valid_mask(p['sequence'], p['modifier'], p['values']).sum())
    for row in stats.values():
        row['sequences'] = len(row['sequences'])
    path = OUT / 'profiles.jsonl'
    with path.open('w') as f:
        for p in profiles:
            p['sequence_sha256'] = seq_hash(p['sequence'])
            f.write(json.dumps(p, allow_nan=False) + '\n')
    unique = {p['sequence_sha256']: p['sequence'] for p in profiles}
    (OUT / 'sequences.json').write_text(json.dumps(unique, indent=2) + '\n')
    manifest = {'thresholds_clipped_median': {'/'.join(k): v for k, v in cuts.items()},
                'counts': stats, 'max_length': 1000,
                'reactivity_as_truth': 'High-reactivity label = clipped observed reactivity > fixed dataset/probe median; ties at median belong to low class.',
                'prediction': 'Sequence-only BPP unpaired probability 1-sum_j(P_ij), or single-structure binary unpaired state labeled separately.',
                'spearman': 'Raw reactivity versus prediction; DMS A/C and CMCT G/U also reported base-balanced. Constant or insufficient profiles are undefined and counted.',
                'probe_masks': {'DMS': 'AC', 'CMCT': 'GU', '2A3': 'ACGU', 'SHAPE': 'ACGU', '1M7': 'ACGU', 'NMIA': 'ACGU'},
                'rnagym_caveat': 'RNAgym-style median labels, but probe-base masks and replicate aggregation differ from the official scorer.',
                'trainsetb_caveat': 'Synthetic profiles generated from TrainSetB reference structures and used during training; training fit, not independent test.',
                'input_sha256': {str(EFOLD.relative_to(ROOT)): sha(EFOLD), str(RNAGYM.relative_to(ROOT)): sha(RNAGYM)},
                'profiles_sha256': sha(path), 'n_unique_sequences': len(unique)}
    (OUT / 'panel_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
