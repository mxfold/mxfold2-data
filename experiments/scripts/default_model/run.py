"""Freeze, submit, resume and inspect a default-model training run."""
from __future__ import annotations

import argparse
import fcntl
import io
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile

from scripts.default_model import prepare
from scripts.efold_hf import workflow as wf

PROJECT = Path(__file__).resolve().parents[2]
GROUP = 'tga-satolab-gtex'
QSUB = '/apps/t4/rhel9/uge/latest/bin/lx-amd64/qsub'
SOURCE_REF = '02fd1b1f8e8a66374ad818da9812d09f3a20912e'


def freeze(root, wheel=None, reference_run=None):
    reference_plan = verify(reference_run) if reference_run is not None else None
    if reference_plan is not None and reference_plan['source_commit'] != SOURCE_REF:
        raise ValueError('Reference run was built from a different MXfold2 source')
    root.mkdir(parents=True)
    source = root / 'source'
    source.mkdir()
    archive = subprocess.check_output([
        'git', 'archive', SOURCE_REF, 'mxfold2', 'pyproject.toml', 'uv.lock',
        'scripts/efold_hf/workflow.py', 'scripts/efold_hf/matched_losses.py',
    ], cwd=PROJECT)
    with tarfile.open(fileobj=io.BytesIO(archive)) as handle:
        handle.extractall(source, filter='data')
    if reference_plan is not None:
        interfaces = [name for name in reference_plan['frozen_sha256']
                      if name.startswith('source/mxfold2/interface.') and name.endswith('.so')]
        if len(interfaces) != 1:
            raise RuntimeError('Reference run must include one native MXfold2 interface')
        shutil.copy2(reference_run / interfaces[0], root / interfaces[0])
        wheel_sha256 = reference_plan['wheel_sha256']
    else:
        with zipfile.ZipFile(wheel) as handle:
            metadata = handle.read('mxfold2-0.3.0.dist-info/METADATA')
            if b'\nVersion: 0.3.0\n' not in metadata:
                raise RuntimeError('Expected the 0.3.0 wheel built from the pinned source')
            for name in handle.namelist():
                if name.startswith('mxfold2/') and name.endswith('.py'):
                    if (source / name).read_bytes() != handle.read(name):
                        raise RuntimeError(f'Wheel source differs: {name}')
            interfaces = [name for name in handle.namelist()
                          if name.startswith('mxfold2/interface.') and name.endswith('.so')]
            if len(interfaces) != 1:
                raise RuntimeError('Wheel must include one native MXfold2 interface')
            (source / interfaces[0]).write_bytes(handle.read(interfaces[0]))
        wheel_sha256 = wf.digest(wheel)

    for name in ('prepare.py', 'train.py', 'run.py', 'evaluate.py'):
        target = source / 'scripts/default_model' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(__file__).with_name(name), target)
    efold = source / 'scripts/efold_hf'
    shutil.copy2(PROJECT / 'scripts/efold_hf/condition5_nr80_data.py', efold)
    shutil.copy2(PROJECT / 'scripts/efold_hf/combined_nr80_size.py', efold / 'audit.py')
    shutil.copy2(Path(__file__).with_name('job.sh'), root)
    shutil.copy2(PROJECT / prepare.LOG / 'pretrain_variants_20261031/baseline.conf', root)
    binaries = root / 'bin'
    binaries.mkdir()
    for name in ('cd-hit-est', 'cd-hit-est-2d'):
        shutil.copy2(prepare.nr80.CDHIT / name, binaries)
    (root / 'scheduler').mkdir()
    frozen = [path for path in source.rglob('*') if path.is_file()]
    frozen += [root / 'job.sh', root / 'baseline.conf', *binaries.iterdir()]
    reference_ft = json.loads((PROJECT / prepare.PROFILE_RUNS['ribonanza'] / 'plan.json').read_text())
    rank_settings = reference_ft['arm_settings']['mw_rank_restored']
    plan = {
        'version': '0.3.0', 'source_commit': SOURCE_REF, 'wheel_sha256': wheel_sha256,
        'native_reference_run': str(reference_run) if reference_run is not None else None,
        'project': str(PROJECT), 'python': str(PROJECT / '.venv/bin/python'),
        'created': time.time(), 'pretrain_epochs': 20, 'ribonanza_epochs': 3, 'mammalian_epochs': 3,
        'max_length': prepare.MAX_LENGTH, 'structure_replay_fraction': .2,
        'structure_split': 'All NR80 representatives of the ten eligible structure sources.',
        'RNAcentral_policy': 'Exclude sequences that match any eligible non-RNAcentral sequence at NR80.',
        'profile_reference_runs': {phase: str(path) for phase, path in prepare.PROFILE_RUNS.items()},
        'scheduler': {'group': GROUP, 'worker_hours': 22, 'cpu': 'cpu_8=1', 'gpu': 'gpu_h=1'},
        'arm_settings': {'pretrain': wf.training_settings(root, 'pretrain'),
                         'mw_rank_restored': rank_settings},
        'input_sha256': {str(path): wf.digest(PROJECT / path) for path in prepare.INPUTS},
        'frozen_sha256': {str(path.relative_to(root)): wf.digest(path) for path in sorted(frozen)},
    }
    wf.write_json(root / 'plan.json', plan)
    print(f'Frozen MXfold2 0.3.0 run at {root}', flush=True)


def verify(root):
    plan = wf.plan_for(root)
    for name, expected in plan['frozen_sha256'].items():
        if wf.digest(root / name) != expected:
            raise RuntimeError(f'Frozen implementation changed: {name}')
    return plan


def submit(root, action, *, hold=None):
    plan = wf.plan_for(root)
    settings = plan['scheduler']
    group = settings['group']
    cpu = action in ('prepare', 'evaluate')
    duration = '08:00:00' if action == 'evaluate' else '23:50:00'
    command = [QSUB, '-terse', '-cwd', '-j', 'y', '-A', group, '-r', 'y',
               '-N', 'mx030_' + action, '-l', settings['cpu' if cpu else 'gpu'],
               '-l', 'h_rt='+duration, '-o', str(root / 'scheduler')]
    if not cpu:
        command += ['-l', 'hostname=!r22n6']
    if hold:
        command += ['-hold_jid', str(hold)]
    command += [str(root / 'job.sh'), str(root), action, plan['python']]
    submitted = subprocess.run(['sg', group, '-c', shlex.join(command)], cwd=root,
                               capture_output=True, text=True)
    if submitted.returncode != 0:
        raise RuntimeError(f'qsub failed: {submitted.stderr.strip()}')
    job = submitted.stdout.strip().split('.')[0]
    if not job.isdecimal():
        raise RuntimeError(f'Unexpected qsub response: {submitted.stdout}')
    with (root / 'submissions.jsonl').open('a') as handle:
        handle.write(json.dumps({'action': action, 'job_id': job, 'hold': hold, 'time': time.time()}) + '\n')
    print(f'Submitted {action} job {job}', flush=True)
    return job


def run_job(root, action, gpu, hours):
    plan = verify(root)
    with (root / 'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if action == 'prepare':
            if not (root / 'data-complete.json').exists():
                prepare.nr80.CDHIT = root / 'bin'
                prepare.prepare(root)
            return
        if action == 'evaluate':
            from scripts.default_model import evaluate
            deadline = time.monotonic() + 3600 * (hours if hours is not None else 7)
            if not evaluate.work(root, deadline) and os.environ.get('JOB_ID'):
                submit(root, 'evaluate', hold=os.environ['JOB_ID'])
            return
        from scripts.default_model import train
        deadline = time.monotonic() + 3600 * (hours if hours is not None else plan['scheduler']['worker_hours'])
        completed = train.work(root, gpu, deadline)
        job = os.environ.get('JOB_ID')
        if job and not completed:
            submit(root, 'train', hold=job)
        elif job and plan.get('evaluate'):
            ledger = root / 'submissions.jsonl'
            if not ledger.exists() or not any(row['action'] == 'evaluate' for row in wf.read_jsonl(ledger)):
                submit(root, 'evaluate', hold=job)


def status(root):
    for name in ('preparation.json', 'submissions.jsonl', 'preflight.json', 'training.json', 'complete.json'):
        path = root / name
        if path.exists():
            print(name, path.read_text().strip(), sep='\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze', 'submit', 'prepare', 'train', 'evaluate', 'status'))
    parser.add_argument('root', type=Path)
    parser.add_argument('--wheel', type=Path)
    parser.add_argument('--reference-run', type=Path, help='Reuse a verified native extension from a previous run')
    parser.add_argument('--step', choices=('prepare', 'train', 'evaluate'))
    parser.add_argument('--hold', help='TSUBAME predecessor job ID for an individual submission')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--hours', type=float)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == 'freeze':
        if (args.wheel is None) == (args.reference_run is None):
            parser.error('freeze requires one of --wheel and --reference-run')
        freeze(root, args.wheel, args.reference_run.resolve() if args.reference_run else None)
    elif args.action == 'submit':
        verify(root)
        if args.hold and not re.fullmatch(r'\d+', args.hold):
            parser.error('--hold requires a numeric job ID')
        if args.step:
            submit(root, args.step, hold=args.hold)
        else:
            if (root / 'submissions.jsonl').exists():
                parser.error('run already submitted, use status or --step to resume')
            prep = submit(root, 'prepare')
            submit(root, 'train', hold=prep)
    elif args.action == 'status':
        status(root)
    else:
        run_job(root, args.action, args.gpu, args.hours)


if __name__ == '__main__':
    main()
