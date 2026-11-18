#!/usr/bin/env python3
"""Re-run EternaFold with every TrainSetA record and pair-pruned labels."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd

SOURCE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import workflow as ef
import joint_workflow as joint
from scripts.eternafold.project_structure import prune_record, supported
from scripts.eternafold import fulltest_new_metrics as fulltest
from scripts.chemical_consistency import map_mw_metrics as metrics

PROJECT = Path(os.environ.get('EF_PROJECT_ROOT', str(SOURCE)))
OLD = PROJECT / 'qsub_logs/eternafold/eternafold-joint-20261022'
TEMPLATE = PROJECT / 'qsub_logs/eternafold/eternafold-rnagym-fulltest-20261101-r2'
CONDITIONS = ('pretrain', 'joint_synthetic_2A3', 'joint_rnagym_all')
FILES = ('pruned_traina_workflow.py', 'project_structure.py', 'workflow.py', 'joint_workflow.py',
         'fulltest_new_metrics.py', 'qsub_fulltest_new_metrics.sh', 'qsub_pruned_traina.sh', 'submit_pruned_traina.py')


def initialize(root):
    if root.exists():
        raise FileExistsError(root)
    old = json.loads((OLD / 'plan.json').read_text())
    validation = json.loads((OLD / 'validation.json').read_text())
    root.mkdir(parents=True)
    for name in ('bin', 'source/scripts/eternafold', 'source/scripts/chemical_consistency', 'scheduler', 'data'):
        (root / name).mkdir(parents=True)
    for name in ('contrafold', 'contrafold-mpi'):
        if ef.digest(OLD / 'bin' / name) != validation['binary_sha256'][name]:
            raise ValueError('Validated native binary changed')
        shutil.copy2(OLD / 'bin' / name, root / 'bin' / name)
    for name in FILES:
        shutil.copyfile(PROJECT / 'scripts/eternafold' / name, root / 'source/scripts/eternafold' / name)
    shutil.copyfile(PROJECT / 'scripts/chemical_consistency/map_mw_metrics.py',
                    root / 'source/scripts/chemical_consistency/map_mw_metrics.py')
    shutil.copyfile(OLD / 'native/Config.hpp', root / 'native_Config.hpp')
    bootstrap = {'created_jst': fulltest.now(), 'conditions': list(CONDITIONS),
                 'old_root': str(OLD), 'old_plan_sha256': ef.digest(OLD / 'plan.json'),
                 'old_audit_sha256': ef.digest(OLD / 'data/structure_audit.json'),
                 'old_record_audit_sha256': ef.digest(OLD / 'data/TrainSetA_audit.tsv'),
                 'original_list': str(ef.PACKED / 'TrainSetA.lst'),
                 'original_list_sha256': ef.digest(ef.PACKED / 'TrainSetA.lst'),
                 'full_test_template': str(TEMPLATE), 'template_plan_sha256': ef.digest(TEMPLATE / 'plan.json'),
                 'maximum_iterations': 200, 'slice_seconds': 72000,
                 'native_config_sha256': ef.digest(root / 'native_Config.hpp'),
                 'binary_sha256': validation['binary_sha256'],
                 'source_sha256': {str(p.relative_to(root)): ef.digest(p)
                                   for p in ef.source_files(root)}}
    ef.write_json(root / 'bootstrap.json', bootstrap)
    ef.emit(event='pruned_experiment_initialized', root=str(root))


def prepare(root, ranks):
    boot = json.loads((root / 'bootstrap.json').read_text())
    for name, expected in boot['source_sha256'].items():
        if ef.digest(root / name) != expected:
            raise ValueError('Frozen source changed before preparation')
    for name, expected in boot['binary_sha256'].items():
        if ef.digest(root / 'bin' / name) != expected:
            raise ValueError('Validated native binary changed before preparation')
    if ef.digest(root / 'native_Config.hpp') != boot['native_config_sha256']:
        raise ValueError('Pinned native configuration changed')
    if (root / 'prepared.json').exists():
        raise FileExistsError('The pruned experiment is already prepared')
    old = json.loads((OLD / 'plan.json').read_text())
    if ef.digest(OLD / 'plan.json') != boot['old_plan_sha256']:
        raise ValueError('Original experiment plan changed')
    if ef.digest(OLD / 'data/structure_audit.json') != boot['old_audit_sha256']:
        raise ValueError('Original audit changed')
    if ef.digest(OLD / 'data/TrainSetA_audit.tsv') != boot['old_record_audit_sha256']:
        raise ValueError('Original record audit changed')
    if ef.digest(boot['original_list']) != boot['original_list_sha256']:
        raise ValueError('Original TrainSetA list changed')
    config = (root / 'native_Config.hpp').read_text()
    if 'C_MIN_HAIRPIN_LENGTH = 3;' not in config or 'C_MAX_SINGLE_LENGTH = 30;' not in config:
        raise ValueError('Projection grammar differs from the validated native configuration')
    joint.preflight(root, ranks=4)
    audit = json.loads((OLD / 'data/structure_audit.json').read_text())
    old_records = {r['id']: r for r in csv.DictReader((OLD / 'data/TrainSetA_audit.tsv').open(), delimiter='\t')}
    data = root / 'data'
    labels = data / 'TrainSetA_3166'
    labels.mkdir()
    original_paths = [Path(line.split()[0]) for line in Path(boot['original_list']).read_text().splitlines() if line.strip()]
    if len(original_paths) != 3166:
        raise ValueError('Expected every one of the original 3166 TrainSetA records')
    writer = ef.PackWriter(data / 'TrainSetA.efpack')
    records, removed, paths, reasons = [], [], [], Counter()
    for index, path in enumerate(original_paths):
        if ef.digest(path) != audit['input_bpseq_sha256'][str(path)]:
            raise ValueError('Original BPSEQ annotation changed: ' + str(path))
        sequence, partners = ef.read_bpseq(path)
        result, deletions = prune_record(sequence, partners)
        before = {(i, j) for i, j in enumerate(partners, 1) if j > i}
        after = {(i, j) for i, j in enumerate(result, 1) if j > i}
        if not after.issubset(before) or not supported(after):
            raise ValueError('Projection added pairs or produced an invalid structure')
        label = f'A{index:05d}'
        previous = old_records[label]
        if previous['native_parsable'] == 'True' and result != partners:
            raise ValueError('A previously representable annotation was unnecessarily changed')
        destination = labels / (label + '.bpseq')
        destination.write_text(''.join(f'{i} {base} {j}\n' for i, (base, j) in enumerate(zip(sequence, result), 1)))
        check_sequence, check_partners = ef.read_bpseq(destination)
        if check_sequence != sequence or check_partners != result:
            raise ValueError('Projected BPSEQ roundtrip changed a sequence/label')
        writer.add(label, sequence, pairs=result, weight=1. / 3166)
        paths.append(str(destination))
        records.append({'id': label, 'source': str(path), 'projected': str(destination),
                        'length': len(sequence), 'sequence_sha256': metrics.sequence_hash(sequence),
                        'original_pairs': len(before), 'retained_pairs': len(after),
                        'removed_pairs': len(deletions), 'changed': bool(deletions),
                        'old_exclusion_reasons': previous['reasons'],
                        'old_native_parsable': previous['native_parsable'],
                        'source_sha256': ef.digest(path), 'projected_sha256': ef.digest(destination)})
        for (i, j), reason in sorted(deletions.items()):
            removed.append({'id': label, 'i': i, 'j': j, 'base_i': sequence[i-1],
                            'base_j': sequence[j-1], 'reason': reason})
            reasons.update(reason.split(';'))
    structure = writer.finish()
    ef.pack_list(writer.path)
    (data / 'TrainSetA_3166.lst').write_text('\n'.join(paths) + '\n')
    ef.native(root, ['train', '--numdatasources', 5, '--examplefile', str(writer.path) + '.lst'],
              root / 'native_audit', ranks=ranks, env={'EF_AUDIT_FILE': str(data / 'native_parsability.tsv')})
    native = dict(line.split('\t') for line in (data / 'native_parsability.tsv').read_text().splitlines())
    if len(native) != 3166 or any(native.get(r['id']) != '1' for r in records):
        bad = [r['id'] for r in records if native.get(r['id']) != '1']
        raise ValueError('Projected structures failed native parsability: ' + str(bad))
    for record in records:
        record['native_parsable'] = True
    pd.DataFrame(records).to_csv(data / 'structure_records.tsv', sep='\t', index=False)
    pd.DataFrame(removed, columns=['id', 'i', 'j', 'base_i', 'base_j', 'reason']).to_csv(data / 'removed_pairs.tsv', sep='\t', index=False)
    summary = {'input_sequences': 3166, 'retained_sequences': 3166, 'excluded_sequences': 0,
               'native_parsable_sequences': 3166, 'changed_annotations': sum(r['changed'] for r in records),
               'unchanged_annotations': sum(not r['changed'] for r in records),
               'original_pairs': sum(r['original_pairs'] for r in records),
               'retained_pairs': sum(r['retained_pairs'] for r in records), 'removed_pairs': len(removed),
               'overlapping_removal_reasons': dict(reasons),
               'empty_structures_retained': sum(r['retained_pairs'] == 0 for r in records),
               'sequence_policy': 'Every original record retained; same normalized sequence and ID; no deduplication or trimming',
               'pair_policy': 'No new pairs. Maximum-cardinality subset of original canonical pairs satisfying the pinned native hairpin, single-loop and noncrossing grammar',
               'tie_policy': 'Deterministic span/end-point ordering; keep earlier DP solution on equal scores',
               'original_2713_native_parsable_annotations_unchanged': True}
    ef.write_json(data / 'structure_audit.json', summary)
    structure.update({'examplefile': str(writer.path) + '.lst', 'sequences': 3166})
    datasets = {'pretrain': structure}
    for condition in CONDITIONS[1:]:
        previous = old['datasets'][condition]
        target = data / (condition + '.efpack')
        if ef.digest(previous['path']) != previous['sha256']:
            raise ValueError('Original positive-only chemical input changed')
        shutil.copyfile(previous['path'], target)
        inputs = data / (condition + '.lst')
        inputs.write_text('@EFPACK:' + str(writer.path) + '\n@EFPACK:' + str(target) + '\n')
        datasets[condition] = {**previous, 'path': str(target), 'examplefile': str(inputs),
                               'examplefile_sha256': ef.digest(inputs), 'structure_sequences': 3166}
    for name in ('evaluation.efpack', 'evaluation_entries.json', 'test_panel.parquet', 'TrainSetB_test_screen.json', 'zero.params'):
        shutil.copyfile(OLD / 'data' / name, data / name)
    ef.pack_list(data / 'evaluation.efpack')
    # Preserve the already-fixed TrainSetB simulation targets for fit diagnostics.
    profile_source = fulltest.PREVIOUS.parent / 'auroc_spearman_20261028/profiles.jsonl'
    profile_meta = json.loads(profile_source.with_name('panel_manifest.json').read_text())
    if metrics.sha(profile_source) != profile_meta['profiles_sha256']:
        raise ValueError('Original synthetic evaluation targets changed')
    simulation = []
    for index, line in enumerate(profile_source.open()):
        item = json.loads(line)
        if item['dataset'] == 'TrainSetB_simulated_training':
            item['profile_id'] = index
            simulation.append(item)
    with (data / 'simulation_profiles.jsonl').open('w') as handle:
        for item in simulation:
            handle.write(json.dumps(item, allow_nan=False) + '\n')
    entries = [{'name': 'S' + key, 'sequence_sha256': key, 'sequence': seq}
               for key, seq in sorted({p['sequence_sha256']: p['sequence'] for p in simulation}.items())]
    pack = ef.PackWriter(data / 'simulation.efpack')
    for entry in entries:
        pack.add(entry['name'], entry['sequence'])
    pack.finish()
    ef.pack_list(pack.path)
    ef.write_json(data / 'simulation_entries.json', entries)
    hashes = {str(p): ef.digest(p) for p in [*ef.source_files(root), root / 'bootstrap.json',
              root / 'validation.json', root / 'native_Config.hpp',
              *(p for p in data.iterdir() if p.is_file()),
              root / 'bin/contrafold', root / 'bin/contrafold-mpi']}
    plan = {'schema': 'eternafold-pruned-traina3166-v1', 'created_jst': fulltest.now(),
            'datasets': datasets, 'branches': list(CONDITIONS), 'structural_audit': summary,
            'zero_parameters': str(data / 'zero.params'), 'zero_sha256': ef.digest(data / 'zero.params'),
            'initialization': 'Every condition starts independently from the same explicit zero parameters',
            'objective': 'mean_A structural NLL + 0.5 * hierarchical_mean_C chemical NLL + ||w||^2/(2*3166); C omitted for pretrain',
            'regularize': 1. / 3166, 'hyperparam_data': .5,
            'optimizer': {'name': 'native L-BFGS', 'maximum_iterations': boot['maximum_iterations'],
                          'stopping': 'Native stopping; 200 accepted-iteration safety cap, no test selection'},
            'chemical_preprocessing': old['preprocessing'],
            'full_test_template': boot['full_test_template'], 'template_plan_sha256': boot['template_plan_sha256'],
            'evaluation': 'Unchanged original TestSetB references, old RNAgym panel, every original full-test measurement with new AUROC/Spearman, fixed TrainSetB simulations (training fit)',
            'source_and_input_sha256': hashes}
    ef.write_json(root / 'plan.json', plan)
    ef.write_json(root / 'prepared.json', {'plan_sha256': ef.digest(root / 'plan.json'), 'datasets': list(datasets)})
    ef.emit(event='pruned_preparation_complete', **summary)


def train(root, condition, segment, ranks):
    plan = ef.verify(root)
    folder = root / 'runs' / condition
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / '.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (folder / 'complete.json').exists():
            return
        if segment > 1 and not (folder / f'segment-{segment-1:02d}.json').exists():
            raise ValueError('The preceding training segment did not complete safely')
        initial = Path(plan['zero_parameters'])
        signature = hashlib.sha256((ef.digest(root / 'plan.json') + condition + plan['zero_sha256']).encode()).hexdigest()
        start = time.monotonic()
        ef.native(root, ['train', '--numdatasources', 5, '--regularize', plan['regularize'],
                  '--maxiter', plan['optimizer']['maximum_iterations'], '--hyperparam_data', .5,
                  '--examplefile', plan['datasets'][condition]['examplefile'], '--initweights', initial],
                  folder, ranks=ranks, env={'EF_LBFGS_STATE': str(folder / 'lbfgs.state'),
                         'EF_SIGNATURE': signature, 'EF_SLICE_SECONDS': '72000'})
        ef.read_params(folder / 'optimize.params.final')
        complete = (folder / 'lbfgs.state.done').exists()
        result = {'condition': condition, 'segment': segment, 'seconds': time.monotonic() - start,
                  'complete': complete, 'initial_checkpoint_sha256': plan['zero_sha256'],
                  'structure_dataset_sha256': plan['datasets']['pretrain']['sha256'],
                  'chemical_dataset_sha256': None if condition == 'pretrain' else plan['datasets'][condition]['sha256'],
                  'plan_sha256': ef.digest(root / 'plan.json')}
        if complete:
            if (folder / 'lbfgs.state.done').read_text().strip() != signature:
                raise ValueError('Completion signature differs')
            shutil.copyfile(folder / 'optimize.params.final', folder / 'model.params')
            result['model_sha256'] = ef.digest(folder / 'model.params')
            messages = [s for s in (folder / 'optimize.log').read_text().splitlines() if 'Termination' in s]
            result['stop_reason'] = messages[-1] if messages else 'Native completed; inspect optimize.log'
            ef.write_json(folder / 'complete.json', result)
        ef.write_json(folder / f'segment-{segment:02d}.json', result)
        ef.emit(event='pruned_training_segment', **result)


def evaluate(root, condition, ranks):
    ef.evaluate(root, condition, ranks)
    folder = root / 'evaluation' / condition / 'simulation'
    if (folder / 'summary.json').exists():
        return
    bpseq = folder / 'bpseq'
    bpseq.mkdir(parents=True, exist_ok=True)
    ef.native(root, ['predict', '--numdatasources', 5, '--params', root / 'runs' / condition / 'model.params',
              '--gamma', 6, '--bpseq', bpseq, '--examplefile', root / 'data/simulation.efpack.lst'], folder, ranks=ranks)
    entries = json.loads((root / 'data/simulation_entries.json').read_text())
    prediction = {e['sequence_sha256']: metrics.from_bpseq(bpseq / e['name'], e['sequence']) for e in entries}
    profiles = [json.loads(line) for line in (root / 'data/simulation_profiles.jsonl').open()]
    frame = metrics.profile_rows(profiles, prediction)
    frame.to_parquet(folder / 'profiles.parquet', index=False)
    fulltest.save_predictions(folder / 'predictions.npz', prediction)
    ef.write_json(folder / 'summary.json', {'condition': condition, 'training_fit': True,
                 'model_sha256': ef.digest(root / 'runs' / condition / 'model.params'),
                 'all_available': metrics.summarize(frame),
                 'joint_defined': metrics.summarize(frame[frame.auroc.notna() & frame.spearman.notna()])})


def full_setup(root, condition):
    plan = ef.verify(root)
    model = root / 'runs' / condition / 'model.params'
    marker = json.loads(model.with_name('complete.json').read_text())
    if not marker['complete'] or ef.digest(model) != marker['model_sha256']:
        raise ValueError('Full test requires a completed, verified model')
    template = Path(plan['full_test_template'])
    if ef.digest(template / 'plan.json') != plan['template_plan_sha256']:
        raise ValueError('Full-test template changed')
    base_plan = fulltest.verify(template)
    meta = fulltest.prepared(template, base_plan)
    destination = root / 'full_test' / condition
    if not (destination / 'plan.json').exists():
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copytree(template / 'source', destination / 'source', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        shutil.copytree(root / 'bin', destination / 'bin')
        shutil.copyfile(model, destination / 'model.params')
        shutil.copyfile(model.with_name('complete.json'), destination / 'training_complete.json')
        entries = [e for e in json.loads((root / 'data/evaluation_entries.json').read_text()) if e['kind'] == 'chemical']
        bpseq = root / 'evaluation' / condition / 'bpseq'
        known = {metrics.sequence_hash(e['sequence']): metrics.from_bpseq(bpseq / e['name'], e['sequence']) for e in entries}
        known_path = destination / 'archived_panel_predictions.npz'
        fulltest.save_predictions(known_path, known)
        new_plan = copy.deepcopy(base_plan)
        new_plan.update({'created_jst': fulltest.now(), 'training_root': str(root), 'training_condition': condition,
                         'model_sha256': marker['model_sha256'], 'saved_panel': str(known_path),
                         'saved_panel_sha256': ef.digest(known_path),
                         'shared_input_root': str(template / 'data'),
                         'shared_input_metadata_sha256': ef.digest(template / 'data/complete.json'),
                         'template_plan_sha256': plan['template_plan_sha256']})
        ef.write_json(destination / 'plan.json', new_plan)
        for item in meta['shards'].values():
            for relative in item['files_sha256']:
                target = destination / relative
                target.parent.mkdir(exist_ok=True)
                target.symlink_to(template / relative)
        shared = copy.deepcopy(meta)
        shared['plan_sha256'] = ef.digest(destination / 'plan.json')
        ef.write_json(destination / 'data/complete.json', shared)
    script = destination / 'source/scripts/eternafold/fulltest_new_metrics.py'
    subprocess.run([sys.executable, str(script), 'preflight', '--root', str(destination)], check=True)


def full_predict(root, task):
    condition_index, shard = divmod(task - 1, 16)
    condition = CONDITIONS[condition_index]
    destination = root / 'full_test' / condition
    subprocess.run([sys.executable, str(destination / 'source/scripts/eternafold/fulltest_new_metrics.py'),
                    'predict', '--root', str(destination), '--shard', str(shard + 1),
                    '--ranks', '128', '--workers', '32'], check=True)


def summarize(root):
    plan = ef.verify(root)
    results = []
    for condition in CONDITIONS:
        dest = root / 'full_test' / condition
        subprocess.run([sys.executable, str(dest / 'source/scripts/eternafold/fulltest_new_metrics.py'),
                        'aggregate', '--root', str(dest)], check=True)
        results.append({'condition': condition,
                        'training': json.loads((root / 'runs' / condition / 'complete.json').read_text()),
                        'structure_and_old_panel': json.loads((root / 'evaluation' / condition / 'metrics.json').read_text()),
                        'simulation_fit': json.loads((root / 'evaluation' / condition / 'simulation/summary.json').read_text()),
                        'full_test': json.loads((dest / 'results.json').read_text())})
    pieces = [[] for _ in CONDITIONS]
    for shard in range(1, 17):
        frames = []
        for condition in CONDITIONS:
            frame = pd.read_parquet(root / 'full_test' / condition / 'evaluation' / f'shard-{shard:02d}/profiles.parquet')
            frames.append(frame[frame.dataset == fulltest.RAW_DATASET])
        for output, frame in zip(pieces, fulltest.common_frames(frames)):
            output.append(metrics.summarize(frame))
    matched = [{'condition': name, **row} for name, p in zip(CONDITIONS, pieces) for row in fulltest.merge_summaries(p)]
    pd.DataFrame(matched).to_csv(root / 'new_metrics_matched.tsv', sep='\t', index=False, float_format='%.10f')
    rows = [{'condition': r['condition'], **r['structure_and_old_panel']['TestSetB']} for r in results]
    pd.DataFrame(rows).to_csv(root / 'TestSetB.tsv', sep='\t', index=False, float_format='%.10f')
    ef.write_json(root / 'results.json', {'success': True, 'completed_jst': fulltest.now(),
                  'protocol': plan['schema'], 'structural_audit': plan['structural_audit'],
                  'results': results, 'matched_new_metrics': matched})
    ef.emit(event='pruned_experiment_complete', conditions=list(CONDITIONS))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('initialize', 'prepare', 'train', 'evaluate', 'full_setup', 'full_predict', 'summarize'))
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--condition', choices=CONDITIONS)
    p.add_argument('--segment', type=int, default=1)
    p.add_argument('--ranks', type=int, default=32)
    a = p.parse_args()
    root = a.root.resolve()
    if a.action == 'initialize': initialize(root)
    elif a.action == 'prepare': prepare(root, a.ranks)
    elif a.action == 'train': train(root, a.condition, a.segment, a.ranks)
    elif a.action == 'evaluate': evaluate(root, a.condition, a.ranks)
    elif a.action == 'full_setup': full_setup(root, a.condition)
    elif a.action == 'full_predict': full_predict(root, int(os.environ['SGE_TASK_ID']))
    else: summarize(root)


if __name__ == '__main__':
    main()
