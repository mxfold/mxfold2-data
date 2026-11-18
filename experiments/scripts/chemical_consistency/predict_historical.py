#!/usr/bin/env python3
"""Evaluate historical checkpoints on the fixed RNAgym and TrainSetB panels."""
import csv
import json
import os
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.chemical_consistency.prepare import OUT, sha
from scripts.chemical_consistency.predict_mx import create_model, restore, save_predictions, unpaired
from scripts.chemical_consistency.score_saved import load_profiles, score_predictions
from scripts.chemical_consistency.validate_historical import pairs_from_dotbracket, predicted


def verify(model, task):
    ref = Path(task['reference'])
    if task['reference_type'] == 'subset_MAP':
        lines = ref.read_text().splitlines()
        for index in range(0, 15, 3):
            sequence, structure = lines[index + 1], lines[index + 2].split()[0]
            if predicted(model, sequence) != pairs_from_dotbracket(structure):
                raise RuntimeError('Historical structure mismatch: ' + lines[index])
        return 5
    rows = list(csv.DictReader(ref.open()))[:5]
    for row in rows:
        data = [line.split() for line in Path(row['name'].strip()).read_text().splitlines() if line.strip()]
        sequence = ''.join(a[1] for a in data)
        truth = {(int(i), int(j)) for i, _, j in data if int(j) > int(i)}
        pred = predicted(model, sequence)
        actual = (len(pred & truth), len(pred - truth), len(truth - pred))
        expected = tuple(int(row[x].strip()) for x in ('TP', 'FP', 'FN'))
        if actual != expected:
            raise RuntimeError('Historical TestSetB mismatch: ' + row['name'] +
                               ' ' + str(actual) + ' vs ' + str(expected))
    return len(rows)


def main():
    plan = json.loads((OUT / 'historical_tasks.json').read_text())
    task = plan['tasks'][int(os.environ['SGE_TASK_ID']) - 1]
    name = task['model']
    if (OUT / 'metrics/BPP' / name / 'epoch-000/summary.json').exists():
        print(name, 'already complete', flush=True)
        return
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required')
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    profiles, cuts = load_profiles()
    profiles = [p for p in profiles if p['dataset'] in ('RNAgym_test', 'TrainSetB_simulated_training')]
    sequences = {p['sequence_sha256']: p['sequence'] for p in profiles}
    model = create_model(OUT / 'historical_source', task['config'])
    restore(model, task['checkpoint'])
    predictions = {}
    with torch.no_grad():
        checked = verify(model, task)
        print(f'{name}: verified {checked} historical structures', flush=True)
        for index, (key, sequence) in enumerate(sorted(sequences.items()), 1):
            result = model([sequence], return_partfunc=True)
            predictions[key] = unpaired(result[4][0], len(sequence), strict=False)
            if index % 500 == 0:
                print(f'{name}: {index}/{len(sequences)} sequences', flush=True)
    output = OUT / 'predictions/BPP' / name / 'epoch-000.npz'
    save_predictions(output, predictions)
    score_predictions(name, 0, 'BPP', predictions, profiles, cuts,
                      {'checkpoint': task['checkpoint'], 'checkpoint_sha256': sha(Path(task['checkpoint'])),
                       'config': task['config'], 'config_sha256': sha(Path(task['config'])),
                       'prediction_sha256': sha(output), 'historical_metrics': task['evaluation'],
                       'historical_reference': task['reference'], 'historical_structures_checked': checked,
                       'recorded_git_revision': task['recorded_git_revision'],
                       'native_interface_sha256': sha(Path(plan['native_interface_source'])),
                       'original_dataset': task['original_dataset'],
                       'note': 'New fixed panel, not the original full/1-of-100 test'})
    print(json.dumps({'model': name, 'structures_verified': checked,
                      'predictions': len(predictions)}), flush=True)


if __name__ == '__main__':
    main()
