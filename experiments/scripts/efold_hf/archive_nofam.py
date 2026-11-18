#!/usr/bin/env python3
"""Archive-NoFam inference with the previously validated bpRNAnew adapters."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(Path(__file__).parent)]
import bprnanew_benchmark as base
import bprnanew_external as external

DATA = ROOT / 'data/efold_hf/archive_nofam_20261031'
OUT = ROOT / 'qsub_logs/efold_hf/archive_nofam_20261031'
PRIOR = ROOT / 'qsub_logs/efold_hf/bprnanew_20261029'
base.OUT = external.OUT = OUT
base.SHARDS = external.SHARDS = 1


def ct_record(text, identifier, *, first_structure=False):
    """Read one CT structure, preserving its supplied reference pairs."""
    lines = [line.split() for line in text.splitlines() if line.strip()]
    n = int(lines[0][0])
    if n < 1 or len(lines) < n + 1 or (not first_structure and len(lines) != n + 1):
        raise ValueError(f'Expected exactly one complete CT structure: {identifier}')
    sequence, partners = [], [0]
    for i, row in enumerate(lines[1:n+1], 1):
        if len(row) < 6 or int(row[0]) != i:
            raise ValueError(f'Invalid CT indices: {identifier}')
        sequence.append(row[1].upper())
        partners.append(int(row[4]))
    sequence = ''.join(sequence)
    if set(sequence) - set('ACGU'):
        raise ValueError(f'Unexpected unmodified alphabet: {identifier}')
    for i, j in enumerate(partners[1:], 1):
        if j < 0 or j > n or j == i or (j and partners[j] != i):
            raise ValueError(f'Inconsistent CT pair: {identifier}')
    return {'id': identifier, 'source': 'Archive-NoFam', 'sequence': sequence,
            'sequence_sha256': hashlib.sha256(sequence.encode()).hexdigest(),
            'pairs': [[i-1, j-1] for i, j in enumerate(partners) if i and j > i]}


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    archive = DATA / 'media-5.zip'
    downloads = json.loads((DATA / 'downloads.json').read_text())
    metadata = next(x for x in downloads if x['name'] == archive.name)
    if base.sha(archive) != metadata['sha256']:
        raise RuntimeError('Author supplement checksum changed')
    rows = []
    provided = {}
    with zipfile.ZipFile(archive) as bundle:
        for filename in sorted(bundle.namelist()):
            if filename.startswith('supplemental-file-2/ground-truth/') and filename.endswith('.ct'):
                rows.append(ct_record(bundle.read(filename).decode(), Path(filename).stem))
            elif filename.startswith('supplemental-file-2/preds/') and filename.endswith('.ct'):
                model = filename.split('/')[2]
                provided.setdefault(model, []).append(ct_record(bundle.read(filename).decode(), Path(filename).stem,
                                                                first_structure=True))
    if len(rows) != 89 or len({x['id'] for x in rows}) != 89:
        raise ValueError('Archive-NoFam must have exactly 89 distinct target identifiers')
    if min(len(x['sequence']) for x in rows) != 21 or max(len(x['sequence']) for x in rows) != 339:
        raise ValueError('Unexpected official target lengths')
    base.write_jsonl(OUT / 'panel.jsonl', rows)
    reference = {r['id']: r for r in rows}
    for model, predictions in provided.items():
        if len({r['id'] for r in predictions}) != len(predictions):
            raise ValueError('Duplicate author prediction')
        for prediction in predictions:
            if prediction['sequence'] != reference[prediction['id']]['sequence']:
                raise ValueError(f'Author prediction sequence mismatch: {model}, {prediction["id"]}')
        base.write_jsonl(OUT / 'provided_predictions' / f'{model}.jsonl', predictions)
    old_plan = json.loads((PRIOR / 'plan.json').read_text())
    core_order = ['author_eFold', 'mx_bprna10', 'mx_mw_rank_restored', 'MXfold2_public']
    by_name = {m['model']: m for m in old_plan['models']}
    models = [by_name[name] for name in core_order]
    old_external = json.loads((PRIOR / 'external_models.json').read_text())
    external_order = ['EternaFold_public', 'RNAstructure_Fold', 'UFold_public']
    by_name = {m['model']: m for m in old_external}
    external_models = [by_name[name] for name in external_order]
    base.write_json(OUT / 'external_models.json', external_models)
    adapters = [Path(__file__), Path(base.__file__), Path(external.__file__),
                ROOT / 'scripts/chemical_consistency/predict_mx.py']
    plan = {'dataset': 'Archive-NoFam', 'n': 89, 'structural_groups_expected': 47,
            'length_range': [21, 339], 'shards_per_model': 1,
            'paper_url': 'https://rnajournal.cshlp.org/content/32/4/428',
            'paper_doi': '10.1261/rna.080846.125',
            'supplement_source': metadata,
            'supplement_version': 'Author preprint v1 in NIH PMC (PMC12637522.1); final article PMC12990807 has no deposited supplementary media',
            'panel_sha256': base.sha(OUT / 'panel.jsonl'), 'models': models,
            'external_models': external_models,
            'provided_predictions': {k: len(v) for k, v in provided.items()},
            'provided_prediction_policy': 'First CT structure only (RNAstructure may also include higher-energy suboptimal candidates)',
            'metric': 'exact base-pair PPV/SEN/F1; per-target means within structural group, then equal group weighting; also full89 per-RNA mean',
            'reference_policy': 'Author curated CT contacts, no reference pruning, no base-pair tolerance, no segmentation',
            'checkpoint_selection': 'Fixed public checkpoints; project bpRNA pretrain10 and optimizer-restored MW Rank FT5',
            'adapters_sha256': {str(p): base.sha(p) for p in adapters}}
    base.write_json(OUT / 'plan.json', plan)
    # Exact sequence matches do not establish or disprove structural homology.
    training_paths = {'actual_bpRNA': base.OLD / 'data/bprna.jsonl',
                      'actual_chemical_FT': base.OLD / 'data/chemical.jsonl'}
    overlap = {}
    for name, path in training_paths.items():
        sequences = {base.normalize(x['sequence']) for x in base.read_jsonl(path)}
        overlap[name] = {'n_training_unique': len(sequences),
                         'ids': [x['id'] for x in rows if x['sequence'] in sequences],
                         'training_file_sha256': base.sha(path)}
    path = base.RAW / 'efold_train.json'
    training = json.loads(path.read_text())
    sequences = {base.normalize(x['sequence']) for x in training.values()}
    overlap['author_efold_train_public'] = {'n_training_unique': len(sequences),
        'ids': [x['id'] for x in rows if x['sequence'] in sequences],
        'training_file_sha256': base.sha(path)}
    base.write_json(OUT / 'training_exact_overlap.json', overlap)
    base.write_json(OUT / 'prepare-complete.json', {'success': True, 'n': 89,
                                                   'panel_sha256': plan['panel_sha256']})
    print(json.dumps({'n': 89, 'length_range': [21, 339], 'provided': plan['provided_predictions']}, indent=2), flush=True)


def check_adapters():
    plan = base.checked_plan()
    for filename, expected in plan['adapters_sha256'].items():
        if base.sha(filename) != expected:
            raise RuntimeError('Frozen inference adapter changed: ' + filename)


def external_smoke(index):
    check_adapters()
    spec = json.loads((OUT / 'external_models.json').read_text())[index-1]
    items = sorted(base.read_jsonl(OUT / 'panel.jsonl'), key=lambda r: (-len(r['sequence']), r['id']))
    items = [items[0], items[len(items)//2], items[-1]]
    folder = OUT / 'smoke' / spec['model']
    native = OUT / 'native-smoke' / spec['model']
    native.mkdir(parents=True, exist_ok=True)
    predict = external.predictor(spec, native)
    result = []
    start = time.monotonic()
    for item in items:
        pairs = predict(item)
        result.append({'id': item['id'], 'sequence_sha256': item['sequence_sha256'],
                       'length': len(item['sequence']), 'predicted_pairs': pairs,
                       **base.pair_metrics(item['pairs'], pairs)})
    path = folder / 'shard-0.jsonl'
    base.write_jsonl(path, result)
    base.write_json(folder / 'shard-0-complete.json', {'success': True, 'n': len(result),
        'model': spec['model'], 'predictions_sha256': base.sha(path), 'seconds': time.monotonic()-start})
    print('Smoke completed:', spec['model'], flush=True)


if __name__ == '__main__':
    action = sys.argv[1]
    raw = os.environ.get('SGE_TASK_ID', '1')
    task = int(raw) if raw.isdigit() else 1
    if action == 'prepare':
        prepare()
    elif action in ('gpu-smoke', 'gpu-predict'):
        check_adapters()
        if task <= 4:
            base.work(task, smoke=action == 'gpu-smoke')
        elif task == 5:
            if action == 'gpu-smoke':
                external_smoke(3)
            else:
                external.work(3, smoke=False)
        else:
            raise ValueError('Unexpected GPU task')
    elif action in ('cpu-smoke', 'cpu-predict'):
        check_adapters()
        if task not in (1, 2):
            raise ValueError('Unexpected CPU task')
        if action == 'cpu-smoke':
            external_smoke(task)
        else:
            external.work(task, smoke=False)
    else:
        raise ValueError(action)
