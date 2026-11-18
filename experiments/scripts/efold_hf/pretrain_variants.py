#!/usr/bin/env python3
"""Three isolated pretraining variants, with resumable <=24-hour SGE jobs."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import csv
from datetime import datetime
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time

PROJECT = Path(os.environ.get('VARIANT_PROJECT', '/gs/bs/tga-satolab-gtex/satoken/repos/mxfold2-ryntaro'))
ROOT = Path(os.environ.get('VARIANTS_ROOT', str(PROJECT / 'qsub_logs/efold_hf/pretrain_variants_20261031')))
OLD = PROJECT / 'qsub_logs/efold_hf/bprna100_ft5_replay20_20261024'
RAW = PROJECT / 'data/efold_hf/raw'
FULL = PROJECT / 'qsub_logs/efold_hf/paper_full_20261030'
DIAG = PROJECT / 'qsub_logs/efold_hf/pretrain_diagnosis_20261031'
CDHIT = PROJECT / '.cache/cdhit-4.8.1/cd-hit-v4.8.1-2019-0228'
ARMS = {'bprna20': 20, 'bprna_rnacentral10': 10, 'bprna_trainab_nr80_20': 20}
SOFT_HOURS = 22
HARD_SECONDS = 23*3600 + 50*60
MAX_LENGTH = 1000
QSUB_BINARY = '/apps/t4/rhel9/uge/latest/bin/lx-amd64/qsub'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''): h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def read_jsonl(path):
    return [json.loads(s) for s in Path(path).open() if s.strip()]


def write_jsonl(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with temp.open('w') as handle:
        for row in rows: handle.write(json.dumps(row, separators=(',', ':'), allow_nan=False)+'\n')
    temp.replace(path)


def link_atomic(source, target):
    target = Path(target); target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(f'.{target.name}.{os.getpid()}.tmp')
    os.link(source, temp); temp.replace(target)


def stable_sample(rows, n, namespace='pretrain-variants-20261031'):
    return sorted(rows, key=lambda r: hashlib.sha256((namespace+':'+r['id']).encode()).digest())[:n]


@contextmanager
def lock(path, nonblocking=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0))
        try: yield
        finally: fcntl.flock(handle, fcntl.LOCK_UN)


def branch(arm):
    if arm not in ARMS: raise ValueError(arm)
    return ROOT / arm


def legacy():
    frozen = ROOT / 'source'
    sys.path[:0] = [str(frozen), str(frozen / 'scripts/efold_hf')]
    sys.path.append(str(PROJECT / '.cache/bprna-benchmark-deps'))
    import workflow
    if Path(workflow.__file__).resolve() != (frozen / 'scripts/efold_hf/workflow.py').resolve():
        raise RuntimeError('Wrong frozen workflow imported')
    return workflow


def verify_code():
    manifest = json.loads((ROOT / 'implementation.json').read_text())
    for filename, expected in manifest.items():
        if sha(filename) != expected: raise RuntimeError('Frozen implementation changed: '+filename)


def init():
    if (ROOT / 'setup-complete.json').exists():
        verify_code(); print('Setup already frozen', ROOT); return
    ROOT.mkdir(parents=True, exist_ok=True)
    frozen = ROOT / 'source'
    if frozen.exists(): raise RuntimeError('Incomplete existing setup must be inspected before reuse')
    shutil.copytree(OLD / 'source', frozen, symlinks=True, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    path = frozen / 'scripts/efold_hf/workflow.py'
    before = path.read_text()
    trigger = "if arm != 'pretrain' or epoch == total_epochs:"
    if before.count(trigger) != 1: raise ValueError('Unexpected original checkpoint policy')
    path.write_text(before.replace(trigger, 'if True:  # retain every completed pretraining epoch'))
    write_json(ROOT / 'legacy_checkpoint_patch.json', {'original_sha256': hashlib.sha256(before.encode()).hexdigest(),
        'patched_sha256': sha(path), 'change': 'Save every completed pretrain epoch; all forward/backward/optimizer and RNG logic unchanged.'})
    shutil.copyfile(Path(__file__), ROOT / 'controller.py')
    shutil.copyfile(PROJECT / 'scripts/efold_hf/pretrain_variants_job.sh', ROOT / 'job.sh')
    shutil.copyfile(OLD / 'baseline.conf', ROOT / 'baseline.conf')
    tests = read_jsonl(FULL / 'panel.jsonl')
    monitor = [r for r in tests if r['source'] != 'archiveII']
    monitor += stable_sample([r for r in tests if r['source'] == 'archiveII'], 512)
    write_jsonl(ROOT / 'test_full.jsonl', tests)
    write_jsonl(ROOT / 'test_monitor.jsonl', monitor)
    common_bp = stable_sample(read_jsonl(OLD / 'data/bprna.jsonl'), 256)
    write_jsonl(ROOT / 'common_bp_monitor.jsonl', common_bp)
    source_sha = {str(p.relative_to(frozen)): sha(p) for p in frozen.rglob('*')
                  if p.is_file() and '__pycache__' not in p.parts}
    write_json(ROOT / 'source_hashes.json', source_sha)
    for arm in ARMS:
        folder = branch(arm); (folder / 'scheduler').mkdir(parents=True, exist_ok=True)
        (folder / 'source').symlink_to(frozen, target_is_directory=True)
        os.link(ROOT / 'baseline.conf', folder / 'baseline.conf')
        os.link(ROOT / 'job.sh', folder / 'job.sh')
    files = [ROOT / 'controller.py', ROOT / 'job.sh', ROOT / 'baseline.conf', ROOT / 'source_hashes.json',
             ROOT / 'test_full.jsonl', ROOT / 'test_monitor.jsonl', ROOT / 'common_bp_monitor.jsonl']
    write_json(ROOT / 'implementation.json', {str(p): sha(p) for p in files})
    write_json(ROOT / 'setup-complete.json', {'success': True, 'project': str(PROJECT), 'arms': ARMS,
        'hard_wall_seconds': HARD_SECONDS, 'worker_soft_hours': SOFT_HOURS,
        'training_max_length': 1000, 'monitor_test_n': len(monitor), 'full_test_n': len(tests),
        'monitor_policy': 'Fixed test monitor, no early stopping or test-driven checkpoint choice. Full tests at epoch10 and final; checkpoints and losses every epoch.'})
    print('Frozen setup', ROOT, 'monitor=', len(monitor), 'full=', len(tests), flush=True)


def exact_deduplicate(rows):
    """Prefer curated TrainSetAB labels on exact ties; retain full provenance."""
    priority = {'TrainSetA': 0, 'TrainSetB': 1, 'bpRNA-1m': 2, 'rnacentral_synthetic': 3}
    chosen, aliases = {}, {}
    for row in sorted(rows, key=lambda r: (priority.get(r['source'], 4), r['id'])):
        sequence = row['sequence']
        if sequence not in chosen: chosen[sequence] = row
        aliases[row['id']] = chosen[sequence]['id']
    return list(chosen.values()), aliases


def parse_clusters(path):
    clusters, members = [], []
    for line in Path(path).read_text().splitlines():
        if line.startswith('>Cluster'):
            if members: clusters.append(members)
            members = []
        else:
            match = re.search(r'>(S\d+)\.\.\.', line)
            if not match: raise ValueError('Unexpected CD-HIT cluster row: '+line)
            members.append((match[1], line.rstrip().endswith('*')))
    if members: clusters.append(members)
    result = {}
    for cluster in clusters:
        representative = [identifier for identifier, is_rep in cluster if is_rep]
        if len(representative) != 1: raise ValueError('Invalid cluster representative count')
        for identifier, _ in cluster:
            if identifier in result: raise ValueError('Duplicate cluster membership')
            result[identifier] = representative[0]
    return result


def cluster_trainab(rows, folder):
    unique, aliases = exact_deduplicate(rows)
    folder.mkdir(parents=True, exist_ok=True)
    binary = CDHIT / 'cd-hit-est'
    if not binary.exists():
        with (folder / 'build.log').open('w') as log:
            subprocess.run(['make', '-j8', 'cd-hit-est'], cwd=CDHIT, stdout=log, stderr=subprocess.STDOUT, check=True)
    by_token = {f'S{i:07d}': r for i, r in enumerate(unique)}
    main = {k: r for k, r in by_token.items() if len(r['sequence']) >= 4}
    short = {k: r for k, r in by_token.items() if len(r['sequence']) < 4}
    fasta = folder / 'input.fasta'
    with fasta.open('w') as handle:
        for key, r in main.items(): handle.write('>'+key+'\n'+r['sequence'].replace('U', 'T')+'\n')
    output = folder / 'representatives.fasta'
    command = [str(binary), '-i', str(fasta), '-o', str(output), '-c', '.8', '-n', '4',
               '-G', '1', '-g', '1', '-s', '.8', '-aS', '.8', '-aL', '.8', '-r', '0',
               '-l', '3', '-d', '0', '-p', '1', '-T', '8', '-M', '64000']
    with (folder / 'cd-hit-est.log').open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=5*3600)
    mapping = parse_clusters(str(output)+'.clstr')
    if set(mapping) != set(main): raise ValueError('CD-HIT silently dropped input sequences')
    # At >=80% length ratio, lengths1..3 can only match equal lengths;
    # >=80% identity then requires exact identity. Exact dedup already handled
    # those cases, so retaining them individually obeys the same thresholds.
    mapping.update({k: k for k in short})
    tokens_by_id = {r['id']: k for k, r in by_token.items()}
    representatives = {mapping[k] for k in mapping}
    selected = [by_token[k] for k in sorted(representatives)]
    with (folder / 'membership.tsv').open('w') as handle:
        writer = csv.writer(handle, delimiter='\t')
        writer.writerow(['original_id', 'source', 'exact_representative', 'cluster_representative', 'retained'])
        for r in rows:
            exact = aliases[r['id']]; rep = by_token[mapping[tokens_by_id[exact]]]['id']
            writer.writerow([r['id'], r['source'], exact, rep, r['id'] == rep])
    metadata = {'command': command, 'binary_sha256': sha(binary), 'input_n': len(rows),
        'exact_unique_n': len(unique), 'retained_n': len(selected), 'shorter_than_word_n': len(short),
        'short_policy': 'Keep exact-unique lengths1..3; at length ratio>=.8 and identity>=.8 no other pair can cluster with these lengths.',
        'orientation': 'RNA forward strand only (-r0); U->T only for clustering; training RNA/labels unchanged',
        'identity': .8, 'coverage_short': .8, 'coverage_long': .8, 'length_ratio': .8,
        'representative_policy': 'Exact ties prefer TrainSetA, then TrainSetB, then bpRNA; CD-HIT chooses cluster representatives from the original labeled sequences.',
        'source_counts_retained': dict(Counter(r['source'] for r in selected))}
    write_json(folder / 'summary.json', metadata)
    return selected, metadata


def prepare(arm):
    verify_code(); wf = legacy(); folder = branch(arm)
    if (folder / 'prepare-complete.json').exists(): wf.verify(folder); print('Prepared already', arm); return
    data = folder / 'data'; data.mkdir(parents=True, exist_ok=True)
    bp = wf.read_jsonl(OLD / 'data/bprna.jsonl')
    metadata = {'bpRNA_n': len(bp), 'max_length': MAX_LENGTH}
    rows = bp
    if arm == 'bprna_rnacentral10':
        source_path = RAW / 'rnacentral_synthetic.json'
        files = json.loads((RAW / 'rnacentral_synthetic.files.json').read_text())
        published = next(x for x in files if x['path'] == 'data.json')
        if sha(source_path) != published['lfs']['oid']: raise ValueError('RNAcentral source checksum changed')
        source = json.loads(source_path.read_text())
        final = json.loads((RAW / 'efold_train.json').read_text())
        added, label_differences = [], 0
        for reference, item in source.items():
            sequence = wf.normalize(item['sequence'])
            retained = final.get(reference)
            if retained is None or wf.normalize(retained['sequence']) != sequence: continue
            if len(sequence) > MAX_LENGTH: continue
            pairs = wf.structure_pairs(retained['structure'], len(sequence))
            original_pairs = wf.structure_pairs(item['structure'], len(sequence))
            label_differences += pairs != original_pairs
            added.append({'id': 'rnacentral_synthetic:'+reference, 'source': 'rnacentral_synthetic',
                          'sequence': sequence, 'pairs': pairs, 'train_pairs': wf.project_pairs(sequence, pairs)})
        if len(added) != 211191: raise ValueError('Expected frozen eFold RNAcentral subset211191')
        rows, aliases = exact_deduplicate(bp+added)
        metadata.update({'rnacentral_source_n': len(source), 'rnacentral_efold_retained_n': len(added),
            'labels': 'Supplied RNAstructure Fold sequence-only pseudo-labels; use the corresponding efold_train structure',
            'raw_vs_efold_label_differences': label_differences,
            'exact_duplicates_removed': len(bp)+len(added)-len(rows),
            'rnacentral_source_sha256': sha(source_path), 'efold_train_sha256': sha(RAW / 'efold_train.json')})
        write_json(folder / 'exact_aliases.json', aliases)
        del source, final
    elif arm == 'bprna_trainab_nr80_20':
        original = read_jsonl(DIAG / 'official_TrainSetAB.jsonl')
        added = [{**r, 'train_pairs': wf.project_pairs(r['sequence'], r['pairs'])}
                 for r in original if len(r['sequence']) <= MAX_LENGTH]
        rows, clustering = cluster_trainab(bp+added, folder / 'clustering')
        metadata.update({'TrainSetAB_input_n': len(added), 'clustering': clustering})
    if not rows or len({r['id'] for r in rows}) != len(rows): raise ValueError('Empty/duplicate training IDs')
    if any(not r['sequence'] or len(r['sequence']) > MAX_LENGTH for r in rows): raise ValueError('Invalid training length')
    if arm == 'bprna20':
        shutil.copyfile(OLD / 'data/bprna.jsonl', data / 'train.jsonl')
        if sha(data / 'train.jsonl') != sha(OLD / 'data/bprna.jsonl'): raise ValueError('Continuation data changed')
    else: write_jsonl(data / 'train.jsonl', rows)
    os.link(data / 'train.jsonl', data / 'bprna.jsonl')  # legacy loader alias
    samples = stable_sample(rows, 128, namespace='actual-training-monitor')
    write_jsonl(data / 'train_monitor.jsonl', samples)
    train_sequences = {r['sequence'] for r in rows}
    tests = read_jsonl(ROOT / 'test_full.jsonl')
    overlaps = [r['id'] for r in tests if r['sequence'] in train_sequences]
    write_json(folder / 'test_exact_overlap.json', {'ids': overlaps, 'n': len(overlaps),
        'by_source': dict(Counter(r['source'] for r in tests if r['id'] in set(overlaps))),
        'note': 'Training additions are retained as requested. Test overlap is reported, never silently removed from the training set.'})
    common_bp = read_jsonl(ROOT / 'common_bp_monitor.jsonl')
    write_json(folder / 'common_bp_membership.json', {r['id']: r['sequence'] in train_sequences for r in common_bp})
    metadata.update({'n': len(rows), 'sources': dict(Counter(r['source'] for r in rows)),
        'length_range': [min(len(r['sequence']) for r in rows), max(len(r['sequence']) for r in rows)],
        'original_pairs': sum(len(r['pairs']) for r in rows), 'train_pairs': sum(len(r['train_pairs']) for r in rows)})
    write_json(folder / 'dataset.json', metadata)
    run = folder / 'runs/pretrain'; run.mkdir(parents=True, exist_ok=True)
    if arm == 'bprna20':
        for name in ('latest.pt', 'epoch-010.pt'):
            os.link(OLD / 'runs/pretrain' / name, run / name)
        os.link(OLD / 'runs/pretrain/latest.pt', run / 'initial.pt')
        for name in (OLD / 'runs/pretrain').glob('epoch-*-complete.json'):
            shutil.copyfile(name, run / name.name)
    plan = {'arm': arm, 'pretrain_epochs': ARMS[arm], 'finetune_epochs': 0, 'arms': [],
        'seed': 1234, 'worker_hours': SOFT_HOURS, 'checkpoint_every_examples': 100,
        'max_length': MAX_LENGTH, 'n': len(rows), 'source_sha256': json.loads((ROOT / 'source_hashes.json').read_text()),
        'data_sha256': {name: sha(data / name) for name in ('train.jsonl', 'bprna.jsonl', 'train_monitor.jsonl')},
        'config_sha256': sha(folder / 'baseline.conf'), 'job_sha256': sha(folder / 'job.sh'),
        'optimizer': 'AdaBelief', 'lr': .001, 'weight_decay': .01, 'score_loss_weight': .125,
        'loss_func': 'fy', 'perturb': .75, 'dropout': .5, 'accumulation': 1, 'amp': False,
        'initialization': 'Restore old epoch10 model, optimizer and all RNG; start epoch11' if arm == 'bprna20' else 'Fresh initialization with seed1234',
        'initial_checkpoint_sha256': sha(run / 'initial.pt') if arm == 'bprna20' else None,
        'original_epoch10_sha256': sha(OLD / 'runs/pretrain/epoch-010.pt') if arm == 'bprna20' else None,
        'hard_wall_seconds': HARD_SECONDS, 'soft_wall_seconds': SOFT_HOURS*3600,
        'evaluation': 'Every epoch on a fixed test monitor + common bpRNA monitor + actual training sample; full published tests at epoch10/final. No checkpoint selection on tests.'}
    write_json(folder / 'plan.json', plan)
    wf.verify(folder)
    write_json(folder / 'prepare-complete.json', {'success': True, 'n': len(rows), 'train_sha256': sha(data / 'train.jsonl')})
    print(json.dumps({'arm': arm, 'n': len(rows), 'sources': metadata['sources'], 'epochs': ARMS[arm]}), flush=True)


def qsub_job(arm, action, epoch=0, hold=None):
    folder = branch(arm)
    resource = 'cpu_8=1' if action == 'prepare' and arm.endswith('nr80_20') else 'cpu_4=1' if action == 'prepare' else 'gpu_h=1'
    duration = '06:00:00' if action == 'prepare' else '00:45:00' if action == 'preflight' else '23:50:00'
    environment = f'VARIANTS_ROOT={ROOT},VARIANT_ARM={arm},VARIANT_ACTION={action},EVALUATION_EPOCH={epoch}'
    args = [QSUB_BINARY, '-terse', '-cwd', '-j', 'y', '-A', 'tga-satolab-gtex', '-r', 'y',
            '-N', 'pv_'+arm[:17]+'_'+action[:3]+(str(epoch) if action == 'evaluate' else ''),
            '-l', resource, '-l', 'h_rt='+duration, '-o', str(folder / 'scheduler'), '-v', environment]
    if resource == 'gpu_h=1': args += ['-l', 'hostname=!r22n6']
    if hold: args += ['-hold_jid', str(hold)]
    args.append(str(ROOT / 'job.sh'))
    command = ['sg', 'tga-satolab-gtex', '-c', shlex.join(args)]
    for attempt in range(3):
        run = subprocess.run(command, cwd=PROJECT, capture_output=True, text=True)
        if run.returncode == 0: break
        if attempt == 2: raise RuntimeError('qsub failed: '+run.stderr)
        time.sleep(2**attempt)
    match = re.fullmatch(r'(\d+)(?:\..+)?', run.stdout.strip())
    if not match: raise ValueError('Unexpected qsub reply: '+run.stdout)
    job_id = match[1]
    record = {'time': time.time(), 'arm': arm, 'action': action, 'epoch': epoch, 'job_id': job_id,
              'hold_jid': hold, 'command': args}
    with lock(ROOT / 'submission.lock'):
        with (ROOT / 'submissions.jsonl').open('a') as handle: handle.write(json.dumps(record)+'\n')
    print('submitted', arm, action, epoch, job_id, flush=True)
    return job_id


def submit_once(arm, key, action, epoch=0, hold=None):
    folder = branch(arm); path = folder / 'dispatch' / (key+'.json')
    with lock(folder / 'dispatch.lock'):
        if path.exists(): return json.loads(path.read_text())['job_id']
        job_id = qsub_job(arm, action, epoch, hold)
        write_json(path, {'job_id': job_id, 'action': action, 'epoch': epoch, 'hold_jid': hold})
        return job_id


def submit_all():
    verify_code(); jobs = {}
    for arm in ARMS:
        prep = submit_once(arm, 'prepare', 'prepare')
        preflight = submit_once(arm, 'preflight', 'preflight', hold=prep)
        work = submit_once(arm, 'work-initial', 'work', hold=preflight)
        jobs[arm] = {'prepare': prep, 'preflight': preflight, 'work': work}
    write_json(ROOT / 'initial_jobs.json', jobs)


def preflight(arm):
    import torch
    verify_code(); wf = legacy(); folder = branch(arm); plan = wf.verify(folder)
    if not json.loads((folder / 'prepare-complete.json').read_text())['success']: raise RuntimeError('Preparation failed')
    rows = sorted(wf.read_jsonl(folder / 'data/bprna.jsonl'), key=lambda r: (len(r['sequence']), r['id']))
    examples = [rows[0], rows[len(rows)//2], rows[-1]]
    wf.seed_all(1234)
    model, opt, loss, _, _, device = wf.build(folder, 'pretrain', 0)
    initial_step = 0
    if arm == 'bprna20':
        state = wf.restore(folder / 'runs/pretrain/latest.pt', model, opt)
        initial_step = opt.param_groups[0].get('step', 0)
        if state['epoch'] != 11 or state['cursor'] != 0 or initial_step != 455070:
            raise RuntimeError('Continuation is not the completed epoch10 state')
    # Verify the actual legacy worker across forced one-update segments,
    # including checkpoints at epoch boundaries. All tests live in scratch
    # subdirectories and are never used as a training starting checkpoint.
    del model, opt, loss
    def smoke_root(name):
        target = folder / 'preflight-tests' / name
        if target.exists(): shutil.rmtree(target)
        (target / 'data').mkdir(parents=True); (target / 'runs/pretrain').mkdir(parents=True)
        (target / 'source').symlink_to(ROOT / 'source', target_is_directory=True)
        os.link(folder / 'baseline.conf', target / 'baseline.conf'); os.link(folder / 'job.sh', target / 'job.sh')
        write_jsonl(target / 'data/bprna.jsonl', examples)
        mini = {**plan, 'pretrain_epochs': 2, 'data_sha256': {'bprna.jsonl': sha(target / 'data/bprna.jsonl')}}
        write_json(target / 'plan.json', mini); write_json(target / 'preflight.json', {'success': True})
        return target
    unbroken, segmented = smoke_root('unbroken'), smoke_root('segmented')
    auto_calls = []
    wf.submit_job = lambda *args, **kwargs: auto_calls.append((str(args[0]), args[1]))
    wf.dispatch_completed = lambda *args: None
    wf.train_worker(unbroken, 'pretrain', 0, hours=1)
    for _ in range(10):
        wf.train_worker(segmented, 'pretrain', 0, hours=0)
        if (segmented / 'runs/pretrain/complete.json').exists(): break
    if not (segmented / 'runs/pretrain/complete.json').exists(): raise RuntimeError('Segmented preflight did not finish')
    a = torch.load(unbroken / 'runs/pretrain/latest.pt', map_location='cpu', weights_only=False)
    b = torch.load(segmented / 'runs/pretrain/latest.pt', map_location='cpu', weights_only=False)
    max_delta = 0.
    for key, first in a['model_state_dict'].items():
        second = b['model_state_dict'][key]
        torch.testing.assert_close(first, second, rtol=1e-6, atol=1e-7)
        valid = torch.isfinite(first) & torch.isfinite(second)
        if valid.any(): max_delta = max(max_delta, float((first[valid]-second[valid]).abs().max()))
    if a['state'] != b['state'] or a['state']['updates'] != 6: raise RuntimeError('Cursor/update/loss state diverged')
    def compare_tree(first, second):
        import numpy as np
        if torch.is_tensor(first):
            torch.testing.assert_close(first, second, rtol=1e-6, atol=1e-7)
        elif isinstance(first, np.ndarray):
            if not np.array_equal(first, second): raise RuntimeError('NumPy state diverged')
        elif isinstance(first, dict):
            if first.keys() != second.keys(): raise RuntimeError('Optimizer state keys diverged')
            for key in first: compare_tree(first[key], second[key])
        elif isinstance(first, (tuple, list)):
            if len(first) != len(second): raise RuntimeError('State lengths diverged')
            for x, y in zip(first, second, strict=True): compare_tree(x, y)
        elif first != second: raise RuntimeError('State metadata diverged')
    compare_tree(a['optimizer_state_dict'], b['optimizer_state_dict'])
    compare_tree(a['rng']['numpy'], b['rng']['numpy'])
    compare_tree(a['rng']['python'], b['rng']['python'])
    if not torch.equal(a['rng']['torch'], b['rng']['torch']): raise RuntimeError('Torch RNG diverged')
    if not all(torch.equal(x, y) for x, y in zip(a['rng']['cuda'], b['rng']['cuda'], strict=True)):
        raise RuntimeError('CUDA RNG diverged')
    for root in (unbroken, segmented):
        if not all((root / f'runs/pretrain/epoch-{e:03d}.pt').exists() for e in (1, 2)):
            raise RuntimeError('Per-epoch checkpoint retention failed')
    if not auto_calls: raise RuntimeError('Budget-triggered resubmission was not exercised')
    if not all(x.get('step') == 6 for x in b['optimizer_state_dict']['param_groups']):
        raise RuntimeError('Optimizer update count diverged')
    write_json(folder / 'preflight.json', {'success': True, 'arm': arm, 'initial_optimizer_step': initial_step,
        'train_n': len(rows), 'test_lengths': [len(r['sequence']) for r in examples],
        'segmented_vs_unbroken_max_parameter_delta': max_delta, 'cursor_optimizer_rng_preserved': True,
        'forced_segment_submissions': len(auto_calls), 'every_epoch_checkpoint_verified': True,
        'hard_wall_seconds': HARD_SECONDS, 'soft_wall_seconds': SOFT_HOURS*3600})
    print('Preflight passed', arm, flush=True)


def dispatch(arm):
    folder = branch(arm)
    available = []
    for path in (folder / 'runs/pretrain').glob('epoch-*.pt'):
        match = re.fullmatch(r'epoch-(\d+)\.pt', path.name)
        if match and path.with_name(f'epoch-{int(match[1]):03d}-complete.json').exists(): available.append(int(match[1]))
    previous = None
    for epoch in sorted(available):
        previous = submit_once(arm, f'evaluate-{epoch:03d}', 'evaluate', epoch, hold=previous)


def work(arm):
    verify_code(); wf = legacy(); folder = branch(arm); wf.verify(folder)
    if not json.loads((folder / 'preflight.json').read_text())['success']: raise RuntimeError('Preflight failed')
    with lock(folder / 'worker.lock', nonblocking=True):
        actual_save = wf.save_checkpoint
        def save(path, model, optimizer, shapes, state):
            actual_save(path, model, optimizer, shapes, state)
            if Path(path).name == 'latest.pt' and state['updates'] == 0:
                run = folder / 'runs/pretrain'
                if not (run / 'initial.pt').exists(): link_atomic(path, run / 'initial.pt')
                if not (run / 'epoch-000.pt').exists(): link_atomic(path, run / 'epoch-000.pt')
                write_json(run / 'epoch-000-complete.json', {'success': True, 'epoch': 0,
                    'checkpoint_sha256': sha(run / 'epoch-000.pt'), 'initialization': 'fresh seed1234'})
                dispatch(arm)
        wf.save_checkpoint = save
        wf.dispatch_completed = lambda *args: dispatch(arm)
        def resume(*args, **kwargs):
            job = os.environ.get('JOB_ID', 'local')
            return submit_once(arm, 'resume-after-'+job, 'work', hold=job if job.isdigit() else None)
        wf.submit_job = resume
        dispatch(arm)
        wf.train_worker(folder, 'pretrain', 0, hours=SOFT_HOURS)
        if (folder / 'runs/pretrain/complete.json').exists(): summarize_all()


def pair_metrics(reference, predicted):
    ref, pred = {tuple(p) for p in reference}, {tuple(p) for p in predicted}
    tp = len(ref & pred); both = not ref and not pred
    return {'tp': tp, 'fp': len(pred)-tp, 'fn': len(ref)-tp,
        'precision': tp/len(pred) if pred else 1. if both else None,
        'recall': tp/len(ref) if ref else 1. if both else None,
        'f1': 2*tp/(len(ref)+len(pred)) if ref or pred else 1.}


def aggregate(rows):
    result = {}
    for source in sorted({r['source'] for r in rows}):
        selected = [r for r in rows if r['source'] == source]
        record = {'n': len(selected)}
        for name in ('precision', 'recall', 'f1'):
            values = [r[name] for r in selected if r[name] is not None]
            record[name] = sum(values)/len(values) if values else None
            record['n_'+name] = len(values)
        tp, fp, fn = (sum(r[k] for r in selected) for k in ('tp', 'fp', 'fn'))
        record['micro_f1'] = 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.
        result[source] = record
    return result


def evaluate(arm, epoch):
    import torch
    verify_code(); wf = legacy(); folder = branch(arm); plan = wf.verify(folder)
    checkpoint = folder / f'runs/pretrain/epoch-{epoch:03d}.pt'
    marker = json.loads(checkpoint.with_name(f'epoch-{epoch:03d}-complete.json').read_text())
    if not marker['success'] or marker['checkpoint_sha256'] != sha(checkpoint): raise RuntimeError('Checkpoint changed')
    full = epoch in (0, 10, ARMS[arm])
    tests = read_jsonl(ROOT / ('test_full.jsonl' if full else 'test_monitor.jsonl'))
    monitor_ids = {r['id'] for r in read_jsonl(ROOT / 'test_monitor.jsonl')}
    common = read_jsonl(ROOT / 'common_bp_monitor.jsonl')
    actual = read_jsonl(folder / 'data/train_monitor.jsonl')
    items = {r['id']: {**r, 'groups': ['test']} for r in tests}
    for name, rows in [('common_bpRNA', common), ('actual_train', actual)]:
        for row in rows:
            if row['id'] in items: items[row['id']]['groups'].append(name)
            else: items[row['id']] = {**row, 'groups': [name]}
    destination = folder / f'evaluation/epoch-{epoch:03d}'
    destination.mkdir(parents=True, exist_ok=True)
    with lock(destination / 'worker.lock', nonblocking=True):
        if (destination / 'summary.json').exists(): return
        wf.seed_all(1234)
        model, _, _, _, _, _ = wf.build(folder, 'pretrain', 0)
        wf.restore(checkpoint, model); model.eval()
        path = destination / 'per_sequence.jsonl'
        done = {r['id']: r for r in read_jsonl(path)} if path.exists() else {}
        if set(done)-set(items): raise RuntimeError('Unexpected cached evaluation IDs')
        deadline = time.monotonic()+SOFT_HOURS*3600
        with torch.inference_mode(), path.open('a') as handle:
            for item in sorted(items.values(), key=lambda r: (-len(r['sequence']), r['id'])):
                if item['id'] in done: continue
                result = model([item['sequence']], return_partfunc=False)
                partners = result[2][0]
                pairs = [[i-1, int(j)-1] for i, j in enumerate(partners) if i > 0 and j > i]
                row = {'id': item['id'], 'source': item['source'], 'groups': item['groups'],
                    'length': len(item['sequence']), 'predicted_pairs': pairs,
                    **pair_metrics(item['pairs'], pairs)}
                if 'train_pairs' in item: row['projected_target_f1'] = pair_metrics(item['train_pairs'], pairs)['f1']
                handle.write(json.dumps(row, allow_nan=False)+'\n'); handle.flush(); done[row['id']] = row
                if len(done) % 100 == 0: print(arm, 'eval', epoch, len(done), '/', len(items), flush=True)
                if time.monotonic() >= deadline:
                    job = os.environ.get('JOB_ID', 'local')
                    submit_once(arm, f'evaluate-{epoch:03d}-resume-after-{job}', 'evaluate', epoch,
                                hold=job if job.isdigit() else None)
                    return
        values = list(done.values())
        excluded = set(json.loads((folder / 'test_exact_overlap.json').read_text())['ids'])
        summary = {'success': True, 'arm': arm, 'epoch': epoch, 'checkpoint_sha256': sha(checkpoint),
            'test_scope': 'full' if full else 'monitor',
            'test': aggregate([r for r in values if 'test' in r['groups']]),
            'common_test_monitor': aggregate([r for r in values if r['id'] in monitor_ids]),
            'test_monitor_without_exact_training_overlap': aggregate([r for r in values if r['id'] in monitor_ids and r['id'] not in excluded]),
            'common_bpRNA_monitor': aggregate([r for r in values if 'common_bpRNA' in r['groups']]),
            'actual_train_monitor': aggregate([r for r in values if 'actual_train' in r['groups']]),
            'predictions_sha256': sha(path)}
        write_json(destination / 'summary.json', summary)
        summarize_all()


def summarize_all():
    with lock(ROOT / 'summary.lock'):
        curves, statuses = [], {}
        for arm, total in ARMS.items():
            folder = branch(arm); training = {}
            for path in (folder / 'runs/pretrain').glob('epoch-*-complete.json'):
                data = json.loads(path.read_text()); training[data['epoch']] = data
            for path in (folder / 'evaluation').glob('epoch-*/summary.json'):
                data = json.loads(path.read_text()); epoch = data['epoch']
                for scope in ('test', 'common_test_monitor', 'test_monitor_without_exact_training_overlap', 'common_bpRNA_monitor', 'actual_train_monitor'):
                    for source, values in data[scope].items():
                        curves.append({'arm': arm, 'epoch': epoch, 'scope': scope, 'dataset': source,
                            'n': values['n'], 'f1': values['f1'],
                            'training_mean_loss': training.get(epoch, {}).get('mean_loss')})
            complete = folder / 'runs/pretrain/complete.json'
            statuses[arm] = {'training_complete': complete.exists(), 'target_epochs': total,
                            'completed_training_epochs': max(training, default=0),
                            'final_evaluation_complete': (folder / f'evaluation/epoch-{total:03d}/summary.json').exists()}
        if curves:
            temp = ROOT / '.evaluation_curve.tsv.tmp'
            with temp.open('w') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(curves[0]), delimiter='\t')
                writer.writeheader(); writer.writerows(sorted(curves, key=lambda r: (r['arm'], r['epoch'], r['scope'], r['dataset'])))
            temp.replace(ROOT / 'evaluation_curve.tsv')
        write_json(ROOT / 'status.json', {'time': datetime.now().astimezone().isoformat(), 'arms': statuses,
            'complete': all(r['training_complete'] and r['final_evaluation_complete'] for r in statuses.values())})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['init', 'submit', 'prepare', 'preflight', 'work', 'evaluate', 'summarize'])
    parser.add_argument('--arm', choices=list(ARMS), default=os.environ.get('VARIANT_ARM', 'bprna20'))
    parser.add_argument('--epoch', type=int, default=int(os.environ.get('EVALUATION_EPOCH', '0')))
    args = parser.parse_args()
    if args.action == 'init': init()
    elif args.action == 'submit': submit_all()
    elif args.action == 'prepare': prepare(args.arm)
    elif args.action == 'preflight': preflight(args.arm)
    elif args.action == 'work': work(args.arm)
    elif args.action == 'evaluate': evaluate(args.arm, args.epoch)
    else: summarize_all()
