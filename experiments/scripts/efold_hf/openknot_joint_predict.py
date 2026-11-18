#!/usr/bin/env python3
"""Regenerate MAP structures for the earlier MXfold2 joint-model comparison."""
import hashlib
import json
import os
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
JOINT = PROJECT / 'qsub_logs/rnagym/mxfold2-trainsetab-joint-20261022'
OUT = PROJECT / 'qsub_logs/efold_hf/openknot_20261028/joint_predictions'
sys.path.insert(0, str(PROJECT))
from scripts.rnagym import trainsetab_joint as joint


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    import pandas as pd
    task = int(os.environ['SGE_TASK_ID']) - 1
    grid = json.loads((JOINT / 'plan.json').read_text())
    condition = grid['conditions'][task]
    run = Path(condition['root'])
    name = run.name
    ckpt = run / 'runs/epoch-100.pt'
    previous = pd.read_csv(run / 'runs/evaluation/epoch-100/RNAgym_official_test_panel.csv')
    panel_path = JOINT / 'data/2A3/test_panel.parquet'
    if digest(panel_path) != digest(JOINT / 'data/DMS/test_panel.parquet'):
        raise RuntimeError('Joint condition test panels differ')
    panel = pd.read_parquet(panel_path)
    m = joint.runtime(JOINT)
    joint.verify(run)
    torch = m['torch']
    if not torch.cuda.is_available():
        raise RuntimeError('GPU is required')
    model = joint.fresh_model(JOINT, m)
    payload = torch.load(ckpt, map_location='cpu', weights_only=False)
    model.load_state_dict(payload['model_state_dict'], strict=True)
    model.to('cuda:0').eval()
    from scripts.rnagym import full_coverage_experiments as full
    full.interface.set_bpp_cutoff(0.)
    OUT.mkdir(parents=True, exist_ok=True)
    temp = OUT / (name + '.jsonl.tmp')
    finished = OUT / (name + '.jsonl')
    verified = 0
    with temp.open('w') as out, torch.no_grad():
        for index, (sequence, group) in enumerate(panel.groupby('sequence', sort=False), 1):
            if len(sequence) > 1000:
                continue
            # Recompute the previously reported marginal Spearman for a few
            # examples, ensuring the restored model and data match that run.
            need_bpp = verified < 5
            result = model([sequence], return_partfunc=need_bpp)
            bp = result[2][0]
            pairs = [[i - 1, int(j) - 1] for i, j in enumerate(bp) if i > 0 and j > i]
            seq_hash = hashlib.sha256(sequence.encode()).hexdigest()
            if need_bpp:
                unpaired = full.unpaired_from_edges(full.revision.edges_from_native(result[4][0]), len(sequence))
                reference = previous[previous.sequence_hash == seq_hash]
                for modifier, observations in group.groupby('modifier', sort=False):
                    scores = [full.probe_correlation(sequence, modifier,
                              full.parse_reactivity(row.reactivity, len(sequence)), unpaired)
                              for row in observations.itertuples(index=False)]
                    scores = [x for x in scores if x is not None]
                    expected = reference[reference.modifier == modifier]
                    if scores:
                        if len(expected) != 1 or abs(sum(scores) / len(scores) - float(expected.spearman.iloc[0])) > 1e-6:
                            raise RuntimeError('Existing probe score does not match restored model')
                        verified += 1
            out.write(json.dumps({'sequence_sha256': seq_hash, 'predicted_pairs': pairs}) + '\n')
            if index % 100 == 0:
                out.flush()
                print(f'{name}: {index} sequences processed', flush=True)
    if verified < 5:
        raise RuntimeError('Could not verify restored model against original evaluation')
    temp.replace(finished)
    metadata = {'condition': name, 'epoch': 100, 'checkpoint_sha256': digest(ckpt),
                'panel_sha256': digest(panel_path),
                'output_sha256': digest(finished), 'verified_spearman_cases': verified}
    (OUT / (name + '.json')).write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata), flush=True)


if __name__ == '__main__':
    main()
