#!/usr/bin/env python3
"""Sequence-only prediction of one MAP structure, never requesting BPP."""
from collections import defaultdict
import csv
import json
import inspect
import os
from pathlib import Path
import sys
import time
import textwrap
import types

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.chemical_consistency.map_mw_metrics import (
    INPUT, OUT, destination, from_partners, load_profiles, profile_rows, save_scores,
    save_predictions, sequence_hash, sha, summarize, write_json,
)
from scripts.chemical_consistency.predict_mx import create_model, restore


def prepare_model(task):
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required')
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model = create_model(task['source'], task['config'])
    payload = restore(model, task['checkpoint'])
    if task.get('expected_state_epoch') is not None:
        state = payload.get('training_state', payload.get('coverage_state', {}))
        if state.get('completed_epoch') != task['epoch']:
            raise RuntimeError('Checkpoint epoch differs from its archived evaluation')
    if task.get('checkpoint_sha256') and sha(task['checkpoint']) != task['checkpoint_sha256']:
        raise RuntimeError('Checkpoint hash changed')
    return model


def validate_reference(model, task):
    """Verify restored MAP predictions using earlier per-sequence TP/FP/FN."""
    if task['family'] == 'historical':
        from scripts.chemical_consistency.predict_historical import verify
        return verify(model, task)
    source = Path(task['reference']).parent / 'TestSetB.csv'
    if not source.is_file():
        raise FileNotFoundError('Archived structural check: ' + str(source))
    rows = list(csv.DictReader(source.open()))[:5]
    for row in rows:
        reference = Path(row['name'].strip())
        data = [line.split() for line in reference.read_text().splitlines() if line.strip()]
        sequence = ''.join(a[1] for a in data)
        truth = {(int(i), int(j)) for i, _, j in data if int(j) > int(i)}
        result = model([sequence], return_partfunc=False)  # no chemical evidence
        partners = result[2][0]
        pred = {(i, int(j)) for i, j in enumerate(partners) if i > 0 and j > i}
        actual = (len(pred & truth), len(pred - truth), len(truth - pred))
        expected = tuple(int(row[k].strip()) for k in ('TP', 'FP', 'FN'))
        if actual != expected:
            raise RuntimeError(f'Restored MAP differs from archived evaluation: {reference}: {actual} != {expected}')
    return len(rows)


def provenance(task):
    return {'checkpoint': task['checkpoint'], 'checkpoint_sha256': sha(task['checkpoint']),
            'source': task['source'], 'config': task['config'], 'config_sha256': sha(task['config']),
            'inference': 'single MAP structure; return_partfunc=False; no reactivity input',
            'aliases': task.get('aliases', [task['model']]),
            'original_evaluation': task.get('evaluation', task.get('reference')),
            'original_dataset': task.get('original_dataset', 'RNAgym common panel'),
            'family': task.get('family', 'full_test')}


def full_predictions(model, sequences, label, shard, smoke=False):
    """Batch only equal lengths, validating each length against singleton MAP.

    Equal lengths avoid padding changes to the bidirectional encoder. If any
    binary structure in a length's first batch differs from singleton inference,
    that entire length uses singleton inference. No probabilities are computed.
    """
    # The archived symmetric head uses view after permute, which PyTorch only
    # accepts for singleton batches. For multi-sequence batches, reshape has
    # identical values and can make a contiguous copy. Keep the original path
    # for every singleton, including all validation calls.
    layout_patches = 0
    for module in model.modules():
        original = module.forward
        if getattr(original, '__name__', '') != 'forward_symmetric':
            continue
        source = textwrap.dedent(inspect.getsource(original.__func__))
        if '.view(' not in source:
            continue
        namespace = dict(original.__func__.__globals__)
        exec(compile(source.replace('.view(', '.reshape('), '<MAP_batch_layout>', 'exec'), namespace)
        batched = types.MethodType(namespace['forward_symmetric'], module)
        def forward(x, _single=original, _batch=batched):
            return _single(x) if x.shape[0] == 1 else _batch(x)
        module.forward = forward
        layout_patches += 1
    by_length = defaultdict(list)
    for sequence in sequences:
        by_length[len(sequence)].append(sequence)
    predictions = {}
    checks, fallback_lengths = 0, []
    start = time.monotonic()
    last_progress = 0
    for length, same_length in sorted(by_length.items()):
        size = 8 if length <= 256 else (4 if length <= 512 else 1)
        size = min(size, len(same_length))
        first = same_length[:size]
        result = model(first, return_partfunc=False)
        initial = [from_partners(s, p) for s, p in zip(first, result[2])]
        if size > 1:
            singleton = [from_partners(s, model([s], return_partfunc=False)[2][0]) for s in first]
            checks += size
            if any(not np.array_equal(a, b) for a, b in zip(initial, singleton)):
                fallback_lengths.append(length)
                size = 1
                initial = singleton
        for sequence, states in zip(first, initial):
            predictions[sequence_hash(sequence)] = states
        for begin in range(len(first), len(same_length), size):
            batch = same_length[begin:begin+size]
            result = model(batch, return_partfunc=False)
            for sequence, partners in zip(batch, result[2]):
                predictions[sequence_hash(sequence)] = from_partners(sequence, partners)
            if len(predictions) - last_progress >= 1000:
                last_progress = len(predictions)
                print(f'{label} shard {shard}: {last_progress}/{len(sequences)}; {time.monotonic()-start:.1f}s', flush=True)
        if len(predictions) - last_progress >= 1000:
            last_progress = len(predictions)
            print(f'{label} shard {shard}: {last_progress}/{len(sequences)}; {time.monotonic()-start:.1f}s', flush=True)
    return predictions, {'equal_length_batch_max': 8, 'view_to_reshape_heads_for_batches': layout_patches,
                         'singleton_structure_checks': checks,
                         'singleton_fallback_lengths': fallback_lengths}


def panel(index, smoke=False):
    plan = json.loads((OUT / 'panel_tasks.json').read_text())
    task = plan['tasks'][index - 1]
    model_name, epoch = task['model'], task['epoch']
    if not smoke and (destination(model_name, epoch) / 'summary.json').exists():
        print('Already complete:', model_name, epoch, flush=True)
        return
    profiles = [p for p in load_profiles() if p['dataset'] in ('RNAgym_test', 'TrainSetB_simulated_training')]
    model = prepare_model(task)
    start = time.monotonic()
    with torch.no_grad():
        checked = validate_reference(model, task)
        print(f'{model_name} epoch {epoch}: {checked} archived structures verified', flush=True)
        sequences = {p['sequence_sha256']: p['sequence'] for p in profiles}
        order = sorted(sequences, key=lambda k: (len(sequences[k]), k))
        if smoke:
            order = order[:8] + order[-2:]
        pred = {}
        for count, key in enumerate(order, 1):
            sequence = sequences[key]
            result = model([sequence], return_partfunc=False)
            pred[key] = from_partners(sequence, result[2][0])
            if count % 500 == 0:
                print(f'{model_name} epoch {epoch}: {count}/{len(order)}; {time.monotonic()-start:.1f}s', flush=True)
    if smoke:
        write_json(OUT / 'smoke' / f'panel-{index}.json', {'success': True, 'model': model_name,
                   'epoch': epoch, 'archived_structures_verified': checked, 'new_sequences': len(pred),
                   'seconds': time.monotonic() - start})
        return
    path = save_predictions(model_name, epoch, pred)
    info = provenance(task)
    info.update({'prediction_sha256': sha(path), 'archived_structures_verified': checked,
                 'profiles_sha256': plan['profiles_sha256'], 'seconds': time.monotonic() - start})
    save_scores(model_name, epoch, profile_rows(profiles, pred), info)
    print('Completed:', model_name, epoch, 'seconds', time.monotonic() - start, flush=True)


def full(index, smoke=False):
    plan = json.loads((OUT / 'full_tasks.json').read_text())
    task = plan['tasks'][index - 1]
    label, shard = task['model'], task['shard']
    dest = OUT / 'full' / label / f'shard-{shard:02d}'
    if not smoke and (dest / 'summary.json').exists():
        print('Already complete:', label, shard, flush=True)
        return
    if not smoke:
        smoke_path = OUT / 'smoke/full-1.json'
        if not smoke_path.exists() or not json.loads(smoke_path.read_text()).get('success'):
            raise RuntimeError('Full-test MAP smoke validation has not succeeded')
    source = Path(plan['source'])
    meta_path = source / 'test_data/complete.json'
    if sha(meta_path) != plan['input_meta_sha256']:
        raise ValueError('Full-test metadata changed')
    meta = json.loads(meta_path.read_text())
    input_path = source / 'test_data' / f'shard-{shard}.parquet'
    if sha(input_path) != meta['shards'][str(shard)]['sha256']:
        raise ValueError('Full-test shard changed')
    frame = pd.read_parquet(input_path)
    if frame.sequence.str.len().max() > 1000:
        raise ValueError('Length limit exceeded')
    model = prepare_model(task)
    # The full-test checkpoint also has an archived small-panel structural check.
    candidates = json.loads((OUT / 'panel_tasks.json').read_text())['tasks']
    matching = next((t for t in candidates if t['checkpoint'] == task['checkpoint']), None)
    start = time.monotonic()
    with torch.no_grad():
        checked = validate_reference(model, matching) if matching else 0
        sequences = sorted(frame.sequence.unique(), key=lambda s: (len(s), s))
        if smoke:
            sequences = sequences[:8] + sequences[-2:]
            frame = frame[frame.sequence.isin(sequences)]
        predictions, batching = full_predictions(model, sequences, label, shard, smoke)
    profiles = []
    for row_index, row in frame.iterrows():
        values = np.fromstring(row.reactivity.strip('[]'), sep=',', dtype=np.float64)
        if len(values) != len(row.sequence) or np.isinf(values).any():
            raise ValueError('Unexpected reactivity values in full-test shard')
        key = sequence_hash(row.sequence)
        # The frozen full-test parquet retains sequence/modifier/reactivity,
        # but not seqID or cluster. A shard+row ID preserves every measurement;
        # its sequence hash identifies the RNA for replicate averaging.
        profiles.append({'dataset': 'RNAgym_full_test', 'modifier': row.modifier,
                         'profile_id': f'{shard}:{row_index}', 'sample_id': f'{shard}:{row_index}',
                         'sequence': row.sequence, 'sequence_sha256': key,
                         'cluster': key, 'values': values})
    scores = profile_rows(profiles, predictions)
    if smoke:
        write_json(OUT / 'smoke' / f'full-{index}.json', {'success': True, 'model': label,
                   'archived_structures_verified': checked, 'sequences': len(sequences),
                   'profiles': len(profiles), 'summary': summarize(scores), 'batching': batching})
        return
    dest.mkdir(parents=True, exist_ok=True)
    temporary = dest / f'.profiles.{os.getpid()}.tmp'
    scores.to_parquet(temporary, index=False)
    temporary.replace(dest / 'profiles.parquet')
    # Keep the full binary structures, including unobserved bases, for reuse.
    prediction_path = save_predictions('full__' + label, shard, predictions)
    info = {'model': label, 'epoch': task['epoch'], 'shard': shard,
            'unique_sequences': len(sequences), 'measured_profiles': len(profiles),
            'provenance': {**provenance(task), 'test_shard_sha256': sha(input_path),
                           'prediction_sha256': sha(prediction_path),
                           'archived_structures_verified': checked,
                           'batching': batching,
                           'seconds': time.monotonic() - start},
            'profiles_sha256': sha(dest / 'profiles.parquet'), 'summary': summarize(scores)}
    write_json(dest / 'summary.json', info)
    print('Completed full:', label, shard, 'seconds', time.monotonic()-start, flush=True)


def main():
    action = os.environ['MW_MAP_ACTION']
    index = int(os.environ.get('SGE_TASK_ID', '1'))
    if action in ('panel', 'smoke-panel'):
        panel(index, action.startswith('smoke'))
    elif action in ('full', 'smoke-full'):
        full(index, action.startswith('smoke'))
    else:
        raise ValueError('Unknown MAP re-evaluation action: ' + action)


if __name__ == '__main__':
    main()
