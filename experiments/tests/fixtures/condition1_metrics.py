"""Single-structure reactivity consistency, using the MW-rank orientation.

The binary AUROC label is predicted *unpaired* status; its continuous score is
raw measured reactivity. No probabilities, clipping, or median thresholds enter
these metrics. Spearman uses the same binary unpaired state and raw reactivity.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / 'qsub_logs/chemical_consistency/auroc_spearman_20261028'
OUT = ROOT / 'qsub_logs/chemical_consistency/map_mw_20261028'
BASES = {'DMS': 'AC', 'CMCT': 'GU', '2A3': 'ACGU', 'SHAPE': 'ACGU',
         '1M7': 'ACGU', 'NMIA': 'ACGU'}
METRICS = ('auroc', 'spearman', 'auroc_mw_neutral')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def sequence_hash(sequence):
    return hashlib.sha256(sequence.encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def load_profiles():
    info = json.loads((INPUT / 'panel_manifest.json').read_text())
    if sha(INPUT / 'profiles.jsonl') != info['profiles_sha256']:
        raise ValueError('Fixed input profiles changed')
    profiles = []
    for index, line in enumerate((INPUT / 'profiles.jsonl').open()):
        item = json.loads(line)
        if len(item['sequence']) > 1000:
            raise ValueError('Sequence exceeds the requested length limit')
        item['profile_id'] = index
        profiles.append(item)
    return profiles


def binary_metrics(reactivity, unpaired):
    """MW effect size and rank correlation for one observed nucleotide group.

    Conventional AUROC and Spearman are undefined for a collapsed prediction.
    The additional neutral extension equals 1 - the training MW loss whenever
    there are >=2 observations, including collapsed binary states.
    """
    r = np.asarray(reactivity, dtype=np.float64)
    u = np.asarray(unpaired)
    if r.ndim != 1 or u.shape != r.shape or not np.isfinite(r).all():
        raise ValueError('Expected finite, equal-length one-dimensional arrays')
    if not np.isin(u, (0, 1)).all():
        raise ValueError('A single binary structure is required, not probabilities')
    n = len(r)
    nu = int(u.sum())
    npair = n - nu
    answer = {'n_observed': n, 'n_unpaired': nu, 'n_paired': npair,
              'auroc': None, 'spearman': None, 'auroc_mw_neutral': None,
              'prediction_constant': bool(n and (nu == 0 or npair == 0)),
              'reactivity_constant': bool(n and np.ptp(r) == 0)}
    if n < 2:
        return answer
    if not nu or not npair:
        answer['auroc_mw_neutral'] = 0.5
        return answer
    ranks = rankdata(r, method='average')
    auc = float((ranks[u == 1].sum() - nu * (nu + 1) / 2) / (nu * npair))
    answer['auroc'] = answer['auroc_mw_neutral'] = auc
    centered = ranks - (n + 1) / 2
    variance = float(np.dot(centered, centered))
    if variance > 0:
        answer['spearman'] = float(centered[u == 1].sum() /
                                   np.sqrt(variance * nu * npair / n))
    return answer


def score_profile(profile, unpaired):
    sequence, probe = profile['sequence'], profile['modifier']
    if probe not in BASES:
        raise ValueError('Unknown chemical probe: ' + probe)
    values = np.asarray(profile['values'], dtype=np.float64)
    unpaired = np.asarray(unpaired)
    if values.shape != (len(sequence),) or unpaired.shape != values.shape:
        raise ValueError('Sequence/structure/reactivity length mismatch')
    if not np.isin(unpaired, (0, 1)).all():
        raise ValueError('Expected binary single-structure unpaired status')
    # The > -10 convention matches the frozen MW-rank training implementation.
    valid = np.isfinite(values) & (values > -10)
    bases = np.asarray(list(sequence))
    valid &= np.isin(bases, list(BASES[probe]))
    groups = tuple(BASES[probe]) if probe in ('DMS', 'CMCT') else ('all',)
    rows = []
    for group in groups:
        mask = valid if group == 'all' else valid & (bases == group)
        rows.append({'stratum': group, **binary_metrics(values[mask], unpaired[mask])})
    return rows


def from_pairs(sequence, pairs):
    answer = np.ones(len(sequence), dtype=np.uint8)
    occupied = set()
    for a, b in pairs:
        a, b = int(a), int(b)
        if not 0 <= a < b < len(sequence) or a in occupied or b in occupied:
            raise ValueError('Invalid/disjointness-violating predicted base pair')
        answer[a] = answer[b] = 0
        occupied.update((a, b))
    return answer


def from_partners(sequence, partners):
    partners = np.asarray(partners, dtype=np.int64)
    if len(partners) != len(sequence) + 1:
        raise ValueError('Expected the native one-based pairing array with dummy 0')
    for i, j in enumerate(partners[1:], 1):
        if j < 0 or j > len(sequence) or j == i or (j and partners[j] != i):
            raise ValueError('Invalid native base-pair partner')
    return (partners[1:] == 0).astype(np.uint8)


def from_bpseq(path, sequence):
    lines = [line.split() for line in Path(path).read_text().splitlines() if line.strip()]
    if len(lines) != len(sequence):
        raise ValueError('BPSEQ sequence length mismatch: ' + str(path))
    partners = [0]
    for i, (row, nucleotide) in enumerate(zip(lines, sequence), 1):
        observed = row[1].upper().replace('T', 'U') if len(row) == 3 else ''
        compatible = observed == nucleotide or (nucleotide not in 'ACGU' and observed == 'N')
        if len(row) != 3 or int(row[0]) != i or not compatible:
            raise ValueError('BPSEQ sequence/coordinate mismatch: ' + str(path))
        partners.append(int(row[2]))
    return from_partners(sequence, partners)


def profile_rows(profiles, predictions):
    rows = []
    for profile in profiles:
        key = profile['sequence_sha256']
        if key not in predictions:
            continue
        unit = profile['sample_id'] if profile['dataset'] == 'TrainSetB_simulated_training' else key
        for scores in score_profile(profile, predictions[key]):
            rows.append({'dataset': profile['dataset'], 'modifier': profile['modifier'],
                         'profile_id': profile['profile_id'], 'sample_id': str(profile['sample_id']),
                         'unit_id': str(unit), 'sequence_sha256': key,
                         'cluster': str(profile['cluster']), **scores})
    if not rows:
        raise ValueError('No profiles matched the predicted structures')
    return pd.DataFrame(rows)


def summarize(frame):
    result = []
    for (dataset, modifier), group in frame.groupby(['dataset', 'modifier'], sort=True):
        profiles = group.groupby(['profile_id', 'unit_id', 'cluster'])[list(METRICS)].mean().reset_index()
        units = profiles.groupby(['unit_id', 'cluster'])[list(METRICS)].mean().reset_index()
        clusters = units.groupby('cluster')[list(METRICS)].mean()
        row = {'dataset': dataset, 'modifier': modifier, 'profiles': len(profiles),
               'units': len(units), 'strata': len(group), 'n_bases': int(group.n_observed.sum()),
               'n_collapsed_strata': int(group.prediction_constant.sum()),
               'n_insufficient_strata': int((group.n_observed < 2).sum())}
        for metric in METRICS:
            row[metric + '_sequence_macro'] = (float(units[metric].mean())
                                               if units[metric].notna().any() else None)
            row[metric + '_cluster_macro'] = (float(clusters[metric].mean())
                                              if clusters[metric].notna().any() else None)
            row['n_units_' + metric + '_defined'] = int(units[metric].notna().sum())
            row['n_strata_' + metric + '_defined'] = int(group[metric].notna().sum())
        result.append(row)
    return result


def destination(model, epoch):
    return OUT / 'metrics' / model / f'epoch-{epoch:03d}'


def save_scores(model, epoch, frame, provenance):
    dest = destination(model, epoch)
    dest.mkdir(parents=True, exist_ok=True)
    temporary = dest / f'.profiles.{os.getpid()}.tmp'
    frame.to_parquet(temporary, index=False)
    temporary.replace(dest / 'profiles.parquet')
    info = {'definition': 'single_structure_MW_rank_AUROC', 'model': model, 'epoch': epoch,
            'provenance': provenance, 'profiles_sha256': sha(dest / 'profiles.parquet'),
            'summary': summarize(frame)}
    write_json(dest / 'summary.json', info)
    return info


def save_predictions(model, epoch, predictions):
    keys = sorted(predictions)
    offsets = np.cumsum([0] + [len(predictions[k]) for k in keys], dtype=np.int64)
    values = np.concatenate([predictions[k] for k in keys]).astype(np.uint8)
    dest = OUT / 'predictions' / model / f'epoch-{epoch:03d}.npz'
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = dest.with_name(f'.{dest.name}.{os.getpid()}.tmp')
    with temp.open('wb') as handle:
        np.savez_compressed(handle, keys=np.asarray(keys, dtype='U64'), offsets=offsets, values=values)
    temp.replace(dest)
    return dest
