#!/usr/bin/env python3
"""Validate restored historical checkpoints against archived predictions."""
from pathlib import Path
import csv
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.chemical_consistency.prepare import OUT
from scripts.chemical_consistency.predict_mx import create_model, restore


def pairs_from_dotbracket(structure):
    stack, pairs = [], set()
    for i, char in enumerate(structure, 1):
        if char == '(':
            stack.append(i)
        elif char == ')':
            pairs.add((stack.pop(), i))
    if stack:
        raise ValueError('Unbalanced dot bracket')
    return pairs


def predicted(model, sequence):
    output = model([sequence], return_partfunc=True)
    bp = output[2][0]
    return {(i, int(j)) for i, j in enumerate(bp) if i > 0 and j > i}


def main():
    root = ROOT / 'qsub_logs/rnagym'
    subset = root / 'rnagym-ckpt-compare-1of100-20260915-142838/checkpoint-comparison/epoch-1'
    ref = subset / 'eval/after/3-2A3_rnafold/rnagym_test/predict.stdout'
    cp = subset / 'finetune/segment-5/3-2A3_rnafold/train/epoch-100'
    config = cp.parent.parent / 'model.conf'
    torch.set_num_threads(4)
    model = create_model(OUT / 'historical_source', config)
    restore(model, cp)
    lines = ref.read_text().splitlines()
    checked = 0
    with torch.no_grad():
        for index in range(0, min(30, len(lines)), 3):
            sequence, structure = lines[index + 1], lines[index + 2].split()[0]
            if predicted(model, sequence) != pairs_from_dotbracket(structure):
                raise RuntimeError('Historical subset structure mismatch: ' + lines[index])
            checked += 1
    print('Historical 1/100 test: validated', checked, 'structures', flush=True)
    exp = root / 'rnagym-benchmark-replay-tuned-20260921-154847/trainseta_then_rnagym/imle_nll_dms_replay50'
    cp = exp / 'rnagym_posttrain/segment-5/train/epoch-5'
    restore(model, cp)
    rows = list(csv.DictReader((exp / 'evaluation/TestSetB/result.csv').open()))[:10]
    with torch.no_grad():
        for row in rows:
            data = [line.split() for line in Path(row['name']).read_text().splitlines() if line.strip()]
            sequence = ''.join(a[1] for a in data)
            truth = {(int(i), int(j)) for i, _, j in data if int(j) > int(i)}
            pred = predicted(model, sequence)
            numbers = (len(pred & truth), len(pred - truth), len(truth - pred))
            expected = tuple(int(row[x].strip()) for x in ('TP', 'FP', 'FN'))
            if numbers != expected:
                raise RuntimeError('Historical full-test model structure mismatch: ' + row['name'] +
                                   ' ' + str(numbers) + ' vs ' + str(expected))
    print('Historical full-test condition: validated', len(rows), 'TestSetB structures', flush=True)


if __name__ == '__main__':
    main()
