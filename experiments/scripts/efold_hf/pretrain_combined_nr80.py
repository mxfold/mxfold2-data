#!/usr/bin/env python3
"""Add the frozen 60,065-sequence NR80 dataset as an isolated 20-epoch run.

Reuse the existing variants' frozen controller and training source unchanged.
Only setup and data preparation differ; jobs, restoration and evaluation use
the same engine. Never modify the three already-running experiment roots.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil

PROJECT = Path('/gs/bs/tga-satolab-gtex/satoken/repos/mxfold2-ryntaro')
REFERENCE = PROJECT / 'qsub_logs/efold_hf/pretrain_variants_20261031'
ROOT = Path(os.environ.get('VARIANTS_ROOT', str(
    PROJECT / 'qsub_logs/efold_hf/pretrain_combined_nr80_20_20261031')))
INPUT = PROJECT / 'qsub_logs/efold_hf/combined_nr80_size_20261031/priority_refined'
ARM = 'combined_nr80_20'
EXPECTED_N = 60065
EXPECTED_SHA = '9bd0853ef707921a4f6ff69a615812e318be4d691086798cc448d0cac80ae1d1'
EXPECTED_SOURCES = {'bpRNA-1m': 12684, 'TrainSetA': 1925,
                    'TrainSetB': 222, 'rnacentral_synthetic': 45234}
if ROOT.resolve() == REFERENCE.resolve():
    raise RuntimeError('The new run must be isolated from the existing variants')


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def engine():
    path = ROOT / 'engine.py'
    spec = importlib.util.spec_from_file_location('combined_nr80_frozen_engine', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = ROOT
    module.ARMS = {ARM: 20}
    return module


def init():
    if (ROOT / 'setup-complete.json').exists():
        engine().verify_code()
        print('Setup already frozen', ROOT, flush=True)
        return
    reference_manifest = json.loads((REFERENCE / 'implementation.json').read_text())
    for path, checksum in reference_manifest.items():
        if sha(path) != checksum:
            raise RuntimeError('Reference implementation changed: ' + path)
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT / 'source').exists():
        raise RuntimeError('Inspect incomplete existing setup before reuse')
    shutil.copytree(REFERENCE / 'source', ROOT / 'source', symlinks=True,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ('source_hashes.json', 'baseline.conf', 'test_full.jsonl',
                 'test_monitor.jsonl', 'common_bp_monitor.jsonl'):
        shutil.copyfile(REFERENCE / name, ROOT / name)
    shutil.copyfile(REFERENCE / 'controller.py', ROOT / 'engine.py')
    shutil.copyfile(REFERENCE / 'job.sh', ROOT / 'job.sh')
    shutil.copyfile(Path(__file__), ROOT / 'controller.py')
    module = engine()
    folder = module.branch(ARM)
    (folder / 'scheduler').mkdir(parents=True, exist_ok=True)
    (folder / 'source').symlink_to(ROOT / 'source', target_is_directory=True)
    os.link(ROOT / 'baseline.conf', folder / 'baseline.conf')
    os.link(ROOT / 'job.sh', folder / 'job.sh')
    source_hashes = json.loads((ROOT / 'source_hashes.json').read_text())
    for relative, checksum in source_hashes.items():
        if sha(ROOT / 'source' / relative) != checksum:
            raise RuntimeError('Frozen source copy differs: ' + relative)
    files = [ROOT / name for name in ('controller.py', 'engine.py', 'job.sh',
        'baseline.conf', 'source_hashes.json', 'test_full.jsonl',
        'test_monitor.jsonl', 'common_bp_monitor.jsonl')]
    module.write_json(ROOT / 'implementation.json', {str(p): sha(p) for p in files})
    module.write_json(ROOT / 'reference_equivalence.json', {
        'reference_root': str(REFERENCE), 'reference_manifest': reference_manifest,
        'controller_engine_identical': sha(ROOT / 'engine.py') == sha(REFERENCE / 'controller.py'),
        'all_model_and_worker_sources_identical': True,
        'baseline_identical': sha(ROOT / 'baseline.conf') == sha(REFERENCE / 'baseline.conf'),
        'test_panels_identical': all(sha(ROOT / p) == sha(REFERENCE / p) for p in
            ('test_full.jsonl', 'test_monitor.jsonl', 'common_bp_monitor.jsonl'))})
    module.write_json(ROOT / 'setup-complete.json', {'success': True, 'arms': {ARM: 20},
        'training_max_length': 1000, 'expected_n': EXPECTED_N,
        'hard_wall_seconds': module.HARD_SECONDS, 'worker_soft_hours': module.SOFT_HOURS,
        'initialization': 'Fresh model and optimizer with seed1234; independent condition',
        'monitor_policy': 'Same fixed test monitor as existing variants; every epoch. Full tests at epochs0/10/20. No early stopping or test-driven checkpoint selection.'})
    module.verify_code()
    print('Frozen isolated run', ROOT, flush=True)


def prepare(module):
    module.verify_code()
    folder = module.branch(ARM)
    wf = module.legacy()
    if (folder / 'prepare-complete.json').exists():
        wf.verify(folder)
        print('Prepared already', ARM, flush=True)
        return
    complete = json.loads((INPUT / 'complete.json').read_text())
    summary = json.loads((INPUT / 'summary.json').read_text())
    labels = json.loads((INPUT / 'label_preservation_verification.json').read_text())
    native = json.loads((INPUT / 'combined_verification/result.json').read_text())
    if not all(x['success'] for x in (complete, summary, labels, native)):
        raise RuntimeError('NR80 dataset verification is incomplete')
    if (summary['total_n'] != EXPECTED_N or sha(INPUT / 'train.jsonl') != EXPECTED_SHA
            or summary['train_sha256'] != EXPECTED_SHA or labels['train_sha256'] != EXPECTED_SHA
            or not complete['combined_native_verification_passed'] or native['merged_clusters_n'] != 0):
        raise RuntimeError('Verified 60,065-sequence dataset changed')
    rows = module.read_jsonl(INPUT / 'train.jsonl')
    sources = dict(Counter(r['source'] for r in rows))
    if len(rows) != EXPECTED_N or sources != EXPECTED_SOURCES:
        raise RuntimeError('Unexpected source counts')
    if len({r['id'] for r in rows}) != len(rows) or len({r['sequence'] for r in rows}) != len(rows):
        raise RuntimeError('Duplicate training IDs or sequences')
    if any(not 1 <= len(r['sequence']) <= 1000 for r in rows):
        raise RuntimeError('Invalid training sequence length')
    data = folder / 'data'
    data.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(INPUT / 'train.jsonl', data / 'train.jsonl')
    if sha(data / 'train.jsonl') != EXPECTED_SHA:
        raise RuntimeError('Training copy differs from the approved dataset')
    os.link(data / 'train.jsonl', data / 'bprna.jsonl')  # legacy loader name
    module.write_jsonl(data / 'train_monitor.jsonl', module.stable_sample(
        rows, 128, namespace='actual-training-monitor'))
    train_sequences = {r['sequence'] for r in rows}
    tests = module.read_jsonl(ROOT / 'test_full.jsonl')
    overlaps = [r for r in tests if r['sequence'] in train_sequences]
    module.write_json(folder / 'test_exact_overlap.json', {'ids': [r['id'] for r in overlaps],
        'n': len(overlaps), 'by_source': dict(Counter(r['source'] for r in overlaps)),
        'note': 'Same overlap reporting and per-condition exclusion metrics as existing variants.'})
    common_bp = module.read_jsonl(ROOT / 'common_bp_monitor.jsonl')
    module.write_json(folder / 'common_bp_membership.json', {
        r['id']: r['sequence'] in train_sequences for r in common_bp})
    module.write_json(folder / 'dataset.json', {'n': len(rows), 'sources': sources,
        'max_length': 1000, 'length_range': [min(len(r['sequence']) for r in rows),
                                           max(len(r['sequence']) for r in rows)],
        'original_pairs': sum(len(r['pairs']) for r in rows),
        'train_pairs': sum(len(r['train_pairs']) for r in rows),
        'original_dataset': str(INPUT / 'train.jsonl'), 'dataset_sha256': EXPECTED_SHA,
        'labels_unchanged': True, 'priority': summary['priority'],
        'RNAcentral_labels': 'Supplied RNAstructure Fold sequence-only pseudo-labels; duplicates removed in favor of bpRNA/TrainSetA/B'})
    provenance = ROOT / 'dataset_origin'
    provenance.mkdir(exist_ok=True)
    for source, name in [(INPUT / 'summary.json', 'summary.json'),
                         (INPUT / 'complete.json', 'complete.json'),
                         (INPUT / 'label_preservation_verification.json', 'labels.json'),
                         (INPUT / 'combined_verification/result.json', 'native_verification.json')]:
        shutil.copyfile(source, provenance / name)
    reference_plan = json.loads((REFERENCE / 'bprna_trainab_nr80_20/plan.json').read_text())
    plan = {**reference_plan, 'arm': ARM, 'n': len(rows), 'pretrain_epochs': 20,
        'finetune_epochs': 0, 'arms': [], 'seed': 1234,
        'source_sha256': json.loads((ROOT / 'source_hashes.json').read_text()),
        'data_sha256': {name: sha(data / name) for name in
                       ('train.jsonl', 'bprna.jsonl', 'train_monitor.jsonl')},
        'config_sha256': sha(folder / 'baseline.conf'), 'job_sha256': sha(folder / 'job.sh'),
        'initialization': 'Fresh initialization with seed1234',
        'initial_checkpoint_sha256': None, 'original_epoch10_sha256': None,
        'hard_wall_seconds': module.HARD_SECONDS, 'soft_wall_seconds': module.SOFT_HOURS * 3600,
        'worker_hours': module.SOFT_HOURS, 'checkpoint_every_examples': 100,
        'dataset_provenance': str(INPUT / 'summary.json')}
    fields = ('optimizer', 'lr', 'weight_decay', 'score_loss_weight', 'loss_func',
              'perturb', 'dropout', 'accumulation', 'amp', 'max_length', 'seed',
              'config_sha256', 'job_sha256', 'source_sha256', 'checkpoint_every_examples',
              'hard_wall_seconds', 'soft_wall_seconds', 'worker_hours')
    differences = {key: (reference_plan[key], plan[key]) for key in fields
                   if reference_plan[key] != plan[key]}
    if differences or plan['score_loss_weight'] != .125:
        raise RuntimeError('Baseline training settings differ: ' + str(differences))
    run = folder / 'runs/pretrain'
    run.mkdir(parents=True, exist_ok=True)
    if any(run.glob('*.pt')):
        raise RuntimeError('An independent run must start without training checkpoints')
    module.write_json(folder / 'plan.json', plan)
    wf.verify(folder)
    module.write_json(ROOT / 'baseline_equivalence.json', {
        'success': True, 'reference_arm': 'bprna_trainab_nr80_20',
        'compared_fields': list(fields), 'differences': differences,
        'dataset_identical_to_approved_60065_file': True, 'pretrain_epochs': 20,
        'fresh_model_and_optimizer': True})
    module.write_json(folder / 'prepare-complete.json', {
        'success': True, 'n': len(rows), 'train_sha256': EXPECTED_SHA})
    print(json.dumps({'arm': ARM, 'n': len(rows), 'sources': sources, 'epochs': 20}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['init', 'submit', 'prepare', 'preflight',
                                         'work', 'evaluate', 'summarize'])
    parser.add_argument('--arm', choices=[ARM], default=ARM)
    parser.add_argument('--epoch', type=int, default=int(os.environ.get('EVALUATION_EPOCH', '0')))
    args = parser.parse_args()
    if args.action == 'init':
        init()
        return
    module = engine()
    if args.action == 'prepare':
        prepare(module)
    elif args.action == 'submit':
        module.submit_all()
    elif args.action == 'preflight':
        module.preflight(ARM)
    elif args.action == 'work':
        module.work(ARM)
    elif args.action == 'evaluate':
        module.evaluate(ARM, args.epoch)
    else:
        module.summarize_all()


if __name__ == '__main__':
    main()
