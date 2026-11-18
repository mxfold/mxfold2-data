#!/usr/bin/env python3
"""Repeat the frozen MXfold2 joint grid with the original 3,166 BPSEQs."""

import argparse
import json
from pathlib import Path
import shutil

from trainsetab_joint import PROBES, digest, now, runtime, write_json


def read_json(path):
    return json.loads(path.read_text())


def bpseq_paths(path):
    return [Path(line.split()[0]) for line in path.read_text().splitlines() if line.strip()]


def check_hash(path, expected):
    if digest(path) != expected:
        raise ValueError(f'Frozen input changed: {path}')


def prepare(root, previous, full_traina):
    grid = read_json(previous / 'plan.json')
    status = read_json(previous / 'run_status.json')
    if grid['schema'] != 'mxfold2-trainsetab-joint-grid-v1' or status['status'] != 'complete':
        raise ValueError('Expected the completed MXfold2 joint grid')
    previous_plans = []
    for entry in grid['conditions']:
        path = Path(entry['root']) / 'plan.json'
        check_hash(path, entry['plan_sha256'])
        previous_plans.append(read_json(path))
    old_hashes = previous_plans[0]['source_sha256']
    if any(plan['source_sha256'] != old_hashes or plan['structural_sequences_per_epoch'] != 2713
           for plan in previous_plans):
        raise ValueError('The previous grid does not share a frozen 2,713-record protocol')

    reference_grid = read_json(full_traina / 'plan.json')
    reference_plan = Path(reference_grid['conditions'][0]['root']) / 'plan.json'
    reference_hashes = read_json(reference_plan)['source_sha256']
    original_list = full_traina / 'TrainSetA.lst'
    check_hash(original_list, reference_hashes[str(original_list)])
    originals = bpseq_paths(original_list)
    old_list = previous / 'TrainSetA.lst'
    check_hash(old_list, old_hashes[str(old_list)])
    if len(originals) != 3166 or len({path.name for path in originals}) != 3166:
        raise ValueError('Expected all 3,166 distinct original TrainSetA records')
    if not set(bpseq_paths(old_list)) < set(originals):
        raise ValueError('The previous TrainSetA records are not a subset of the full source')

    root.mkdir(parents=True)
    (root / 'scheduler').mkdir()
    shutil.copytree(previous / 'source', root / 'source', symlinks=True,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copytree(previous / 'labels', root / 'labels', ignore=shutil.ignore_patterns('*.lock'))
    for probe in PROBES:
        shutil.copytree(previous / 'data' / probe, root / 'data' / probe)
    for name in ('model.conf', 'initial_random.pt', 'initial_baseline.pth',
                 'TrainSetB.lst', 'TestSetB.lst', 'teacher_summary.json'):
        shutil.copyfile(previous / name, root / name)

    other_bpseq = set(bpseq_paths(root / 'TrainSetB.lst') + bpseq_paths(root / 'TestSetB.lst'))
    hashes = {}
    for name, expected in old_hashes.items():
        path = Path(name)
        if path == old_list:
            continue
        if path.is_relative_to(previous):
            path = root / path.relative_to(previous)
        elif path not in other_bpseq:
            continue
        check_hash(path, expected)
        hashes[str(path)] = expected
    if any(str(path) not in hashes for path in other_bpseq):
        raise ValueError('A TrainSetB or TestSetB record was not frozen in the previous grid')

    structural_data = root / 'data/TrainSetA_original'
    structural_data.mkdir()
    new_list = root / 'TrainSetA.lst'
    with new_list.open('w') as handle:
        for source in originals:
            destination = structural_data / source.name
            expected = reference_hashes[str(source)]
            shutil.copyfile(source, destination)
            check_hash(destination, expected)
            hashes[str(destination)] = expected
            handle.write(str(destination) + '\n')
    hashes[str(new_list)] = digest(new_list)

    m = runtime(root)
    a, b, test = [m['BPseqDataset'](str(root / f'{name}.lst'))
                  for name in ('TrainSetA', 'TrainSetB', 'TestSetB')]
    if (len(a), len(b), len(test)) != (3166, 1094, 430):
        raise ValueError('Unexpected coverage in the archived MXfold2 dataset reader')
    aseq, bseq, tseq = ({s.upper().replace('T', 'U') for _, s, _ in ds} for ds in (a, b, test))
    overlap = dict(grid['data_overlap_audit'], TrainSetA_TrainSetB_exact_sequences=len(aseq & bseq),
                   TrainSetA_TestSetB_exact_sequences=len(aseq & tseq))
    if overlap['TrainSetA_TestSetB_exact_sequences']:
        raise ValueError('The full TrainSetA overlaps TestSetB')

    for label in read_json(root / 'teacher_summary.json')['labels']:
        key = label['key']
        marker = m['controls'].require_labels(root, key)
        if marker['records'] != len(b) or marker['archive_sha256'] != label['archive_sha256']:
            raise ValueError(f'Unexpected previous labels: {key}')
        for suffix in ('complete.json', 'npz', 'tsv'):
            path = root / 'labels' / f'{key}.{suffix}'
            hashes[str(path)] = digest(path)

    structure = {
        'source_list': str(original_list), 'source_list_sha256': reference_hashes[str(original_list)],
        'source_manifest': str(reference_plan), 'source_manifest_sha256': digest(reference_plan),
        'records': len(a), 'added_records_vs_previous_joint': len(a) - len(bpseq_paths(old_list)),
        'original_pairs': sum(sum(j > i for i, j in enumerate(values['target'].tolist())) for _, _, values in a),
        'all_bpseq_files_byte_identical_to_originals': True,
        'bpseq_pairs_removed_in_preprocessing': 0, 'data_overlap_audit': overlap,
    }
    audit = root / 'structure_input_audit.json'
    write_json(audit, structure)
    hashes[str(audit)] = digest(audit)

    for entry, plan in zip(grid['conditions'], previous_plans, strict=True):
        run = root / 'conditions' / entry['name']
        (run / 'runs').mkdir(parents=True)
        for name in ('TrainSetA.lst', 'TrainSetB.lst', 'TestSetB.lst'):
            (run / name).symlink_to(root / name)
        data_probe = entry['probe'] if entry['probe'] != 'none' else '2A3'
        (run / 'data').symlink_to(root / 'data' / data_probe, target_is_directory=True)
        nb, accumulation = plan['auxiliary_sequences_per_epoch'], plan['gradient_accumulation']
        slots = len(a) + nb
        updates = (slots + accumulation - 1) // accumulation
        plan.update(created_local=now(), experiment_root=str(root), source_sha256=hashes,
                    structural_sequences_per_epoch=len(a), total_sample_slots_per_epoch=slots,
                    optimizer_updates_per_epoch=updates, structural_backward_coefficient=updates / len(a),
                    auxiliary_backward_coefficient=.5 * updates / nb if nb else 0., data_overlap_audit=overlap)
        write_json(run / 'plan.json', plan)
        entry.update(root=str(run), plan_sha256=digest(run / 'plan.json'))

    training_inputs = grid.pop('matched_to_eternafold')
    training_inputs['TrainSetA_sequences'] = len(a)
    grid.update(created_local=now(), teachers=[], training_inputs=training_inputs,
                previous_joint=str(previous), previous_joint_plan_sha256=digest(previous / 'plan.json'),
                structural_input_audit=structure, data_overlap_audit=overlap,
                pseudo_label_teacher='The exact label archives from the previous joint grid are reused.')
    grid['scheduler']['dependencies'] = '12 preflights -> 10 x 12 ten-epoch segments -> summary'
    write_json(root / 'plan.json', grid)
    write_json(root / 'run_status.json', {'status': 'prepared', 'conditions': len(grid['conditions']),
                                         'updated_local': now()})
    print(json.dumps(structure, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ('root', 'previous', 'full-traina'):
        parser.add_argument(f'--{argument}', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.root.resolve(), args.previous.resolve(), args.full_traina.resolve())
