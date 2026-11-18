#!/usr/bin/env python3
"""Re-score saved eFold, joint MXfold2, and EternaFold single structures."""
import json
from pathlib import Path
import sys

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as feather

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.chemical_consistency.map_mw_metrics import (
    INPUT, OUT, from_pairs, from_bpseq, load_profiles, profile_rows, save_scores,
    save_predictions, sequence_hash, sha, write_json,
)
from scripts.chemical_consistency.eternafold_sim import conditions


def emit(model, epoch, predictions, profiles, provenance):
    required = {p['sequence_sha256'] for p in profiles}
    if set(predictions) != required:
        raise ValueError(f'Incomplete saved structures: {model}; missing {len(required-set(predictions))}')
    path = save_predictions(model, epoch, predictions)
    save_scores(model, epoch, profile_rows(profiles, predictions),
                {**provenance, 'prediction_sha256': sha(path), 'inference_reactivity': False})
    print(f'{model} epoch {epoch}: {len(predictions)} saved structures scored', flush=True)


def efold(profiles):
    relevant = [p for p in profiles if p['dataset'] == 'eFold_viral']
    sequences = {p['sample_id']: p['sequence'] for p in relevant}
    old = ROOT / 'qsub_logs/efold_hf/bprna100_ft5_replay20_20261024'
    new = ROOT / 'qsub_logs/efold_hf/bprna10_restored_ft5_replay20_20261026'
    aligned = ROOT / 'qsub_logs/efold_hf/paper_aligned_20261028'
    paths = [('pretrain', 10, old / 'evaluation/pretrain/epoch-010/per_sequence.jsonl')]
    paths += [(arm, e, new / f'evaluation/{arm}/epoch-{e:03d}/per_sequence.jsonl')
              for arm in ('mw_rank_restored', 'nll_restored', 'shape_fy_restored') for e in range(1, 6)]
    paths += [('original_' + arm, e, old / f'evaluation/{arm}/epoch-{e:03d}/per_sequence.jsonl')
              for arm in ('rank', 'nll', 'shape_fy', 'assisted') for e in range(1, 6)]
    for model, epoch, path in paths:
        predictions = {}
        for line in path.open():
            item = json.loads(line)
            if item['id'] in sequences:
                seq = sequences[item['id']]
                predictions[sequence_hash(seq)] = from_pairs(seq, item['predicted_pairs'])
        emit(model, epoch, predictions, relevant, {'path': str(path), 'sha256': sha(path)})
    predictions = {}
    files = sorted((aligned / 'author_efold_paper_gpu').glob('shard-*.jsonl'))
    for path in files:
        for line in path.open():
            row = json.loads(line)
            if row['id'] in sequences:
                seq = sequences[row['id']]
                predictions[sequence_hash(seq)] = from_pairs(seq, row['predicted_pairs'])
    emit('author_eFold', 0, predictions, relevant,
         {'files': {str(p): sha(p) for p in files}, 'decode': 'author paper protocol'})
    path = ROOT / 'data/efold_hf/paper_2026/results_benchmark_algos.feather'
    table = feather.read_table(path, columns=['reference', 'dataset', 'model', 'structure'])
    table = table.filter(pc.and_(pc.equal(table['dataset'], 'viral_fragments'),
                                pc.is_in(table['model'], value_set=pa.array(['EternaFold', 'MXFold2']))))
    for model in ('EternaFold', 'MXFold2'):
        predictions = {}
        for row in table.to_pylist():
            item_id = 'viral_fragments:' + row['reference']
            if row['model'] == model and item_id in sequences:
                seq = sequences[item_id]
                predictions[sequence_hash(seq)] = from_pairs(seq, row['structure'])
        emit('author_' + model, 0, predictions, relevant,
             {'path': str(path), 'sha256': sha(path), 'decode': 'author saved comparator structure'})


def joint(profiles):
    relevant = [p for p in profiles if p['dataset'] == 'RNAgym_test']
    seqs = {p['sequence_sha256']: p['sequence'] for p in relevant}
    folder = ROOT / 'qsub_logs/efold_hf/openknot_20261028/joint_predictions'
    for metadata in sorted(folder.glob('*.json')):
        info = json.loads(metadata.read_text())
        path = folder / (info['condition'] + '.jsonl')
        if sha(path) != info['output_sha256']:
            raise ValueError('Saved joint MAP changed')
        pred = {}
        for line in path.open():
            row = json.loads(line)
            pred[row['sequence_sha256']] = from_pairs(seqs[row['sequence_sha256']], row['predicted_pairs'])
        emit('joint_' + info['condition'], 100, pred, relevant, info)


def eternafold(profiles):
    rn = [p for p in profiles if p['dataset'] == 'RNAgym_test']
    sim = [p for p in profiles if p['dataset'] == 'TrainSetB_simulated_training']
    entries = json.loads((ROOT / 'qsub_logs/eternafold/eternafold-joint-20261022/data/evaluation_entries.json').read_text())
    entries = [e for e in entries if e['kind'] == 'chemical' and len(e['sequence']) <= 1000]
    sim_entries = json.loads((INPUT / 'eternafold_sim/entries.json').read_text())
    for name, root, condition in conditions():
        params = root / 'runs' / condition / 'model.params'
        for relevant, items, folder, suffix in (
            (rn, entries, root / 'evaluation' / condition / 'bpseq', ''),
            (sim, sim_entries, INPUT / 'eternafold_sim' / name / 'bpseq', '__TrainSetB'),
        ):
            pred = {sequence_hash(e['sequence']): from_bpseq(folder / e['name'], e['sequence']) for e in items}
            emit('EternaFold_' + name + suffix, 0, pred, relevant,
                 {'params': str(params), 'params_sha256': sha(params), 'bpseq_folder': str(folder),
                  'decode': 'saved EternaFold structure, gamma=6 as in earlier experiments'})


def main():
    profiles = load_profiles()
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / 'definition.json', {
        'AUROC': 'label=predicted unpaired (1) vs paired (0); score=raw measured reactivity; ties=0.5',
        'Spearman': 'raw reactivity versus the same binary predicted unpaired state',
        'no_BPP': True, 'no_reactivity_threshold': True, 'no_reactivity_clipping': True,
        'max_length': 1000, 'observation_mask': 'finite reactivity > -10, matching frozen MW implementation',
        'DMS': 'A and C scored separately and averaged within each measured profile',
        'CMCT': 'G and U scored separately and averaged within each measured profile',
        'undefined': 'collapsed predictions: AUROC/Spearman undefined; constant reactivity: Spearman undefined',
        'auroc_mw_neutral': 'extra diagnostic, 1 - training MW loss for groups with >=2 observations; collapsed states assigned 0.5',
        'aggregation': 'strata mean -> measured profile -> RNA mean over replicates -> sequence macro; TrainSetB unit=annotated sample',
        'common_comparison': 'matched epochs; intersection of profile/stratum keys with both AUROC and Spearman defined in every compared model',
        'input_profiles_sha256': sha(INPUT / 'profiles.jsonl'),
        'trainsetb': 'simulation training fit, not independent test',
    })
    efold(profiles)
    joint(profiles)
    eternafold(profiles)


if __name__ == '__main__':
    main()
