#!/usr/bin/env python3
"""Recompute RNAgym and simulated TrainSetB BPP for archived experiments."""
from collections import defaultdict
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.chemical_consistency.metrics import correlation, valid_mask
from scripts.chemical_consistency.prepare import OUT, sha
from scripts.chemical_consistency.predict_mx import create_model, restore, save_predictions, unpaired
from scripts.chemical_consistency.score_saved import load_profiles, score_predictions


def legacy_spearman(item, prediction):
    values = np.asarray(item['values'], dtype=float)
    sequence, modifier = item['sequence'], item['modifier']
    valid = valid_mask(sequence, modifier, values, prediction)
    bases = np.asarray(list(sequence))
    groups = ('A', 'C') if modifier == 'DMS' else (('G', 'U') if modifier == 'CMCT' else (None,))
    rhos = []
    for base in groups:
        mask = valid if base is None else valid & (bases == base)
        if mask.sum() < 2 or np.ptp(values[mask]) == 0:
            continue
        rho = correlation(values[mask], prediction[mask])
        rhos.append(0. if rho is None else rho)
    return float(np.mean(rhos)) if rhos else None


def check_existing(sequence_hash, prediction, profiles, previous):
    groups = defaultdict(list)
    for item in profiles:
        rho = legacy_spearman(item, prediction)
        if rho is not None:
            groups[item['modifier']].append(rho)
    checks = 0
    for modifier, rhos in groups.items():
        key = (sequence_hash, modifier)
        if key not in previous:
            raise RuntimeError('Reference Spearman profile missing: ' + str(key))
        if abs(float(np.mean(rhos)) - previous[key]) > 1e-6:
            raise RuntimeError(f'Restored BPP differs from recorded Spearman: {key}; '
                               f'new {np.mean(rhos)} vs prior {previous[key]}')
        checks += 1
    return checks


def main():
    plan = json.loads((OUT / 'rnagym_tasks.json').read_text())
    task = plan['tasks'][int(os.environ['SGE_TASK_ID']) - 1]
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required')
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    profiles, cuts = load_profiles()
    profiles = [p for p in profiles if p['dataset'] in ('RNAgym_test', 'TrainSetB_simulated_training')]
    sequences = {p['sequence_sha256']: p['sequence'] for p in profiles}
    rnagym = defaultdict(list)
    for p in profiles:
        if p['dataset'] == 'RNAgym_test':
            rnagym[p['sequence_sha256']].append(p)
    order = list(rnagym) + sorted(set(sequences) - set(rnagym))
    model = create_model(task['source'], task['config'])
    for entry in task['entries']:
        name, epoch = entry['model'], entry['epoch']
        summary = OUT / 'metrics/BPP' / name / f'epoch-{epoch:03d}/summary.json'
        if summary.exists():
            print(f'{name} epoch {epoch}: already complete', flush=True)
            continue
        checkpoint, reference = Path(entry['checkpoint']), Path(entry['reference'])
        payload = restore(model, checkpoint)
        if entry['expected_state_epoch'] is not None:
            state = payload.get('training_state', payload.get('coverage_state', {}))
            if state.get('completed_epoch') != epoch:
                raise RuntimeError('Latest checkpoint does not match evaluated epoch')
        old = pd.read_csv(reference)
        previous = {(row.sequence_hash, row.modifier): float(row.spearman)
                    for row in old.itertuples(index=False)}
        predictions, checks = {}, 0
        with torch.no_grad():
            for index, key in enumerate(order, 1):
                sequence = sequences[key]
                result = model([sequence], return_partfunc=True)
                prediction = unpaired(result[4][0], len(sequence), strict=False)
                predictions[key] = prediction
                if checks < 6 and key in rnagym:
                    checks += check_existing(key, prediction, rnagym[key], previous)
                if index % 500 == 0:
                    print(f'{name} epoch {epoch}: {index}/{len(order)} sequences', flush=True)
        if checks < 5:
            raise RuntimeError('Insufficient previous Spearman checks')
        output = OUT / 'predictions/BPP' / name / f'epoch-{epoch:03d}.npz'
        save_predictions(output, predictions)
        score_predictions(name, epoch, 'BPP', predictions, profiles, cuts,
                          {'checkpoint': str(checkpoint), 'checkpoint_sha256': sha(checkpoint),
                           'prediction_sha256': sha(output), 'prior_csv': str(reference),
                           'prior_csv_sha256': sha(reference), 'prior_spearman_cases_checked': checks,
                           'aliases': entry.get('aliases', [name])})
        print(json.dumps({'model': name, 'epoch': epoch, 'predictions': len(predictions),
                          'previous_spearman_verified': checks}), flush=True)


if __name__ == '__main__':
    main()
