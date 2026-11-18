"""Reactivity-as-reference AUROC and Spearman with explicit coverage.

The AUROC labels are created using a fixed dataset/probe median of clipped
reactivity. Predictions are unpaired probabilities or binary unpaired states.
This follows the RNAgym median-label idea, while the probe-specific base masks
and replicate handling are stated separately in the run manifest.
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
from scipy.stats import rankdata

BASES = {'DMS': 'AC', 'CMCT': 'GU', '2A3': 'ACGU', '1M7': 'ACGU',
         'NMIA': 'ACGU', 'SHAPE': 'ACGU'}


def valid_mask(sequence, probe, values, prediction=None):
    values = np.asarray(values, dtype=np.float64)
    if probe not in BASES or len(values) != len(sequence):
        raise ValueError('Unknown probe or length mismatch')
    mask = np.isfinite(values) & (values > -900)
    mask &= np.fromiter((base in BASES[probe] for base in sequence), bool, len(sequence))
    if prediction is not None:
        prediction = np.asarray(prediction, dtype=np.float64)
        if len(prediction) != len(sequence):
            raise ValueError('Prediction length mismatch')
        mask &= np.isfinite(prediction)
        if np.any((prediction[mask] < -1e-5) | (prediction[mask] > 1.00001)):
            raise ValueError('Predicted unpaired probability outside [0,1]')
    return mask


def auc_binary_truth(labels, prediction):
    labels = np.asarray(labels, dtype=bool)
    prediction = np.asarray(prediction, dtype=float)
    npos = int(labels.sum())
    nneg = len(labels) - npos
    if not npos or not nneg:
        return None
    ranks = rankdata(prediction, method='average')
    return float((ranks[labels].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def correlation(values, prediction):
    if len(values) < 2 or np.ptp(values) == 0 or np.ptp(prediction) == 0:
        return None
    a, b = rankdata(values), rankdata(prediction)
    a -= a.mean()
    b -= b.mean()
    result = float(np.dot(a, b) / math.sqrt(np.dot(a, a) * np.dot(b, b)))
    return result if math.isfinite(result) else None


def score_profile(profile, prediction, threshold):
    sequence, probe = profile['sequence'], profile['modifier']
    values = np.asarray(profile['values'], dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    mask = valid_mask(sequence, probe, values, prediction)
    v, p = values[mask], np.clip(prediction[mask], 0, 1)
    label = np.clip(v, 0, 1) > threshold
    bases = np.array(list(sequence))[mask]
    overall = correlation(v, p)
    groups = ('A', 'C') if probe == 'DMS' else (('G', 'U') if probe == 'CMCT' else ())
    stratified = [] if groups else ([overall] if overall is not None else [])
    for base in groups:
        bm = bases == base
        r = correlation(v[bm], p[bm])
        if r is not None:
            stratified.append(r)
    return {'n_observed': int(len(v)), 'n_positive': int(label.sum()),
            'n_negative': int(len(label) - label.sum()),
            'auroc': auc_binary_truth(label, p), 'spearman_all': overall,
            'spearman_probe_balanced': float(np.mean(stratified)) if stratified else None,
            'prediction_constant': bool(len(p) and np.ptp(p) == 0),
            'truth_constant': bool(len(v) and np.ptp(v) == 0)}


def thresholds(profiles):
    values = defaultdict(list)
    for item in profiles:
        a = np.asarray(item['values'], dtype=float)
        mask = valid_mask(item['sequence'], item['modifier'], a)
        values[(item['dataset'], item['modifier'])].extend(np.clip(a[mask], 0, 1).tolist())
    answer = {}
    for key, items in values.items():
        if not items:
            raise ValueError('No measured data for ' + str(key))
        answer[key] = float(np.median(items))
    return answer
