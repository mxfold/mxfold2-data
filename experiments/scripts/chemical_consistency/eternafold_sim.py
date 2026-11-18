#!/usr/bin/env python3
"""Evaluate EternaFold BPP on TrainSetB synthetic profiles, without evidence."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.chemical_consistency.prepare import OUT, seq_hash, sha
from scripts.chemical_consistency.score_saved import load_profiles, score_predictions
from scripts.eternafold.workflow import PackWriter, pack_list
from scripts.rnagym.evaluate_rnagym import unpaired_from_bpp


def conditions():
    a = ROOT / 'qsub_logs/eternafold/eternafold-traina-branches-20261022'
    b = ROOT / 'qsub_logs/eternafold/eternafold-joint-20261022'
    return [('pretrain', a, 'pretrain'), ('branch_2A3', a, 'synthetic_2A3'),
            ('branch_DMS', a, 'synthetic_DMS'), ('joint_2A3', b, 'joint_synthetic_2A3'),
            ('joint_DMS', b, 'joint_synthetic_DMS'), ('joint_rnagym', b, 'joint_rnagym_all')]


def prepare():
    profiles, _ = load_profiles()
    sequences = sorted({p['sequence'] for p in profiles if p['dataset'] == 'TrainSetB_simulated_training'})
    folder = OUT / 'eternafold_sim'
    folder.mkdir(parents=True, exist_ok=True)
    writer = PackWriter(folder / 'sequence_only.efpack')
    entries = []
    for i, sequence in enumerate(sequences):
        name = f'SIM_{i:05d}'
        writer.add(name, sequence)  # no structure or chemical evidence
        entries.append({'name': name, 'sequence': sequence, 'sequence_sha256': seq_hash(sequence)})
    info = writer.finish()
    pack_list(writer.path)
    (folder / 'entries.json').write_text(json.dumps(entries, indent=2) + '\n')
    print(json.dumps({'sequences': len(entries), 'pack_sha256': info['sha256'],
                      'ambiguous_sequences': sum(bool(set(s) - set('ACGU')) for s in sequences)}))


def predict(task):
    name, root, condition = conditions()[task - 1]
    folder = OUT / 'eternafold_sim'
    entries = json.loads((folder / 'entries.json').read_text())
    profiles, cuts = load_profiles()
    profiles = [p for p in profiles if p['dataset'] == 'TrainSetB_simulated_training']
    result = folder / name
    bpp, bpseq = result / 'bpp', result / 'bpseq'
    bpp.mkdir(parents=True, exist_ok=True)
    bpseq.mkdir(parents=True, exist_ok=True)
    params = root / 'runs' / condition / 'model.params'
    marker = json.loads((root / 'runs' / condition / 'complete.json').read_text())
    if marker['model_sha256'] != sha(params):
        raise RuntimeError('EternaFold parameters changed')
    command = [str(root / 'bin/contrafold'), 'predict', '--numdatasources', '5', '--params',
               str(params), '--gamma', '6', '--posteriors', '0', str(bpp), '--bpseq', str(bpseq),
               '--examplefile', str(folder / 'sequence_only.efpack.lst')]
    subprocess.run(command, check=True)
    predictions = {e['sequence_sha256']: unpaired_from_bpp(bpp / e['name'], len(e['sequence'])) for e in entries}
    score_predictions('EternaFold_' + name + '__TrainSetB', 0, 'BPP', predictions, profiles, cuts,
                      {'params': str(params), 'params_sha256': sha(params),
                       'sequence_only_pack_sha256': sha(folder / 'sequence_only.efpack'),
                       'inference_reactivity': False,
                       'ambiguous_nucleotides': 'native EternaFold maps alphabetic ambiguity to N'})
    print(f'{name}: completed {len(entries)} sequence-only predictions', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['prepare', 'predict'])
    args = p.parse_args()
    if args.action == 'prepare':
        prepare()
    else:
        predict(int(os.environ['SGE_TASK_ID']))


if __name__ == '__main__':
    main()
