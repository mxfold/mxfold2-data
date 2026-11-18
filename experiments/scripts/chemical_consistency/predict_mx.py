#!/usr/bin/env python3
"""Restore frozen MXfold2 checkpoints and save sequence-only BPP predictions."""
from __future__ import annotations

import argparse
import hashlib
from importlib.machinery import PathFinder
import json
import os
from pathlib import Path
import sys

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.chemical_consistency.prepare import OUT, sha
from scripts.chemical_consistency.score_saved import load_profiles, score_predictions


def frozen_mxfold(source):
    source = Path(source).resolve()
    if not (source / 'mxfold2').is_dir():
        raise FileNotFoundError(source / 'mxfold2')
    class FrozenFinder:
        def find_spec(self, fullname, path=None, target=None):
            if fullname == 'mxfold2':
                return PathFinder.find_spec(fullname, [str(source)])
            if fullname.startswith('mxfold2.'):
                return PathFinder.find_spec(fullname, path)
            return None
    sys.meta_path.insert(0, FrozenFinder())
    sys.path.insert(0, str(source))
    import mxfold2
    if not Path(mxfold2.__file__).resolve().is_relative_to(source):
        raise RuntimeError('Did not load the frozen MXfold2 snapshot')
    return mxfold2


def create_model(source, config):
    frozen_mxfold(source)
    import torch
    from mxfold2.predict import Predict
    from mxfold2 import interface
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    Predict.add_args(parser.add_subparsers())
    args = parser.parse_args(['predict', '@' + str(config), 'unused.fasta'])
    model, _ = Predict().build_model(args)
    for module in model.modules():
        if isinstance(module, torch.nn.LSTM):
            original = module.forward
            def forward(*args, _original=original, **kwargs):
                with torch.backends.cudnn.flags(enabled=False):
                    return _original(*args, **kwargs)
            module.forward = forward
    interface.set_bpp_cutoff(0.)
    return model.to('cuda:0').eval()


def restore(model, checkpoint):
    import torch
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    state = payload.get('ema_state_dict', payload.get('model_state_dict', payload))
    allowed = {'turner.score_bulge', 'turner.score_hairpin', 'turner.score_internal'}
    for key, value in state.items():
        if torch.is_tensor(value):
            valid = torch.isfinite(value)
            if key in allowed:
                valid |= torch.isneginf(value)
            if not bool(valid.all()):
                raise ValueError('Nonfinite checkpoint tensor: ' + key)
    missing, extra = model.load_state_dict(state, strict=False)
    if extra or any('.shape_head_' not in name for name in missing):
        raise ValueError(f'Checkpoint architecture mismatch: {missing}, {extra}')
    return payload


def unpaired(native, length, *, strict):
    paired = np.zeros(length, dtype=np.float64)
    seen = {}
    for i, row in enumerate(native):
        for j, probability in row:
            j, p = int(j), float(probability)
            if not (1 <= i <= length and 1 <= j <= length and i != j and np.isfinite(p) and 0 <= p <= 1):
                raise FloatingPointError('Invalid BPP edge')
            key = min(i, j), max(i, j)
            if key in seen:
                if abs(seen[key] - p) > 1e-6:
                    raise FloatingPointError('Conflicting reverse BPP edges')
                continue
            seen[key] = p
            paired[i - 1] += p
            paired[j - 1] += p
    v = 1. - paired
    if not np.isfinite(v).all() or (strict and (v.min() < -.001 or v.max() > 1.001)):
        raise FloatingPointError('Unpaired probability outside [0,1]')
    return np.clip(v, 0, 1)


def save_predictions(path, predictions):
    keys = sorted(predictions)
    lengths = [len(predictions[k]) for k in keys]
    offsets = np.cumsum([0] + lengths, dtype=np.int64)
    vals = np.concatenate([predictions[k] for k in keys]).astype(np.float64, copy=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f, keys=np.asarray(keys, dtype='U64'), offsets=offsets, values=vals)
    tmp.replace(path)


def verify_efold(reference, item, prediction, predicted_pairs):
    old = reference[item['sample_id']]
    if predicted_pairs != old['predicted_pairs']:
        raise RuntimeError('Restored structure differs from prior evaluation: ' + item['sample_id'])
    for obs in old['chemical']:
        if obs['probe'].upper() != item['modifier']:
            continue
        values = np.asarray(item['values'], dtype=float)
        valid = values != -999
        rho = float(spearmanr(values[valid], prediction[valid]).statistic)
        if obs['spearman'] is None:
            if np.isfinite(rho):
                raise RuntimeError('Previously undefined Spearman is now finite')
        elif not np.isfinite(rho) or abs(rho - obs['spearman']) > 1e-6:
            raise RuntimeError('Restored BPP differs from prior Spearman: ' + item['sample_id'])


def run_efold(spec):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required')
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    profiles, cuts = load_profiles()
    profiles = [p for p in profiles if p['dataset'] == 'eFold_viral']
    model = create_model(spec['source'], spec['config'])
    for entry in spec['entries']:
        name, epoch = entry['model'], entry['epoch']
        summary = OUT / 'metrics/BPP' / name / f'epoch-{epoch:03d}/summary.json'
        if summary.exists():
            continue
        checkpoint = Path(entry['checkpoint'])
        marker = json.loads(Path(entry['marker']).read_text())
        if marker.get('checkpoint_sha256') != sha(checkpoint):
            raise RuntimeError('Checkpoint does not match completion marker')
        restore(model, checkpoint)
        reference = {}
        for line in Path(entry['reference']).open():
            row = json.loads(line)
            if row['id'].startswith('viral_fragments:'):
                reference[row['id']] = row
        predictions, failed = {}, []
        with torch.no_grad():
            for item in profiles:
                if item['sequence_sha256'] in predictions:
                    continue
                result = model([item['sequence']], return_partfunc=True)
                bp = result[2][0]
                pairs = [[i - 1, int(j) - 1] for i, j in enumerate(bp) if i > 0 and j > i]
                try:
                    prediction = unpaired(result[4][0], len(item['sequence']), strict=True)
                except FloatingPointError as exc:
                    if 'chemical_bpp_error' not in reference[item['sample_id']]:
                        raise
                    failed.append({'sample_id': item['sample_id'], 'reason': str(exc)})
                    continue
                verify_efold(reference, item, prediction, pairs)
                predictions[item['sequence_sha256']] = prediction
        if not predictions:
            raise RuntimeError('No usable BPP predictions: ' + name)
        output = OUT / 'predictions/BPP' / name / f'epoch-{epoch:03d}.npz'
        save_predictions(output, predictions)
        score_predictions(name, epoch, 'BPP', predictions, profiles, cuts,
                          {'checkpoint': str(checkpoint), 'checkpoint_sha256': sha(checkpoint),
                           'prediction_sha256': sha(output), 'prior_map_spearman_checked': True,
                           'invalid_bpp': failed})
        print(json.dumps({'model': name, 'epoch': epoch, 'predictions': len(predictions),
                          'invalid_bpp': len(failed)}), flush=True)


def main():
    plan = json.loads((OUT / 'efold_tasks.json').read_text())
    task = int(os.environ['SGE_TASK_ID']) - 1
    run_efold(plan[task])


if __name__ == '__main__':
    main()
