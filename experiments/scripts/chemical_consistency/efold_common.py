#!/usr/bin/env python3
"""Compare eFold models on the exact same observed RNAs at each FT epoch."""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.chemical_consistency.prepare import OUT, sha
from scripts.chemical_consistency.score_saved import load_profiles, score_predictions


def load_npz(path):
    with np.load(path) as d:
        keys, offsets, values = d['keys'], d['offsets'], d['values']
        return {str(k): values[int(offsets[i]):int(offsets[i + 1])].copy()
                for i, k in enumerate(keys)}


def main():
    profiles, cuts = load_profiles()
    profiles = [p for p in profiles if p['dataset'] == 'eFold_viral']
    for epoch in range(1, 6):
        choices = [('pretrain_common', 'pretrain', 10)] + [(m + '_common', m, epoch) for m in
                   ('mw_rank_restored', 'nll_restored', 'shape_fy_restored')]
        paths = [OUT / 'predictions/BPP' / m / f'epoch-{e:03d}.npz' for _, m, e in choices]
        if not all(path.exists() for path in paths):
            print(f'epoch {epoch}: waiting for predictions', flush=True)
            continue
        predictions = [load_npz(path) for path in paths]
        common = set.intersection(*(set(p) for p in predictions))
        subset = [p for p in profiles if p['sequence_sha256'] in common]
        for (label, model, trained_epoch), path, pred in zip(choices, paths, predictions):
            score_predictions(label, epoch, 'BPP_common', pred, subset, cuts,
                              {'source_model': model, 'source_epoch': trained_epoch,
                               'prediction_sha256': sha(path), 'common_sequences': len(common),
                               'comparison_epoch': epoch})
        print(json.dumps({'epoch': epoch, 'common_sequences': len(common),
                          'DMS': sum(p['modifier'] == 'DMS' for p in subset),
                          'SHAPE': sum(p['modifier'] == 'SHAPE' for p in subset)}), flush=True)


if __name__ == '__main__':
    main()
