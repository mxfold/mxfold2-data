#!/usr/bin/env python3
"""Evaluate the two completed 3166-record models before RNAgym training ends.

The original full-test array and its final summary must still be pending.
No training, model, metric, or frozen source is changed. Full-test inference
retains the global limit of four cpu_160/MPI128 tasks at once.
"""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--repo', type=Path, required=True)
    args = parser.parse_args()
    root, repo = args.root.resolve(), args.repo.resolve()
    ledger = root / 'evaluation_reschedule.jsonl'
    with (root / '.evaluation_reschedule.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if ledger.exists():
            raise FileExistsError('Evaluation was already rescheduled; inspect the ledger')
        for condition in ('pretrain', 'joint_synthetic_2A3'):
            model = root / 'runs' / condition / 'model.params'
            marker = json.loads(model.with_name('complete.json').read_text())
            folder = root / 'full_test' / condition
            plan = json.loads((folder / 'plan.json').read_text())
            check = json.loads((folder / 'preflight/complete.json').read_text())
            if not marker['complete'] or sha(model) != marker['model_sha256']:
                raise ValueError('A model is not complete')
            if plan['model_sha256'] != marker['model_sha256'] or not check['success']:
                raise ValueError('Full-test preflight/model mismatch')
            if check['plan_sha256'] != sha(folder / 'plan.json'):
                raise ValueError('Preflight belongs to another plan')
        original = [json.loads(line) for line in (root / 'submissions.jsonl').open()]
        old_array = next(j for j in original if j['action'] == 'full_predict')
        old_summary = next(j for j in original if j['action'] == 'summarize')
        rn_check = next(j for j in original if j['action'] == 'full_setup' and j['condition'] == 'joint_rnagym_all')

        def record(item):
            with ledger.open('a') as handle:
                handle.write(json.dumps(item) + '\n')
            print(json.dumps({k: v for k, v in item.items() if k != 'command'}), flush=True)

        # Delete the final summary first so deleting its dependency cannot
        # release it while the remaining RNAgym model is still unfinished.
        for job in (old_summary, old_array):
            command = ['qdel', str(job['job_id'])]
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            record({'action': 'superseded_pending_job', 'job_id': job['job_id'],
                    'command': command, 'response': result.stdout.strip()})

        def submit(name, action, *, holds=(), array=None, condition=None):
            variables = {'EF_PROJECT_ROOT': str(repo)}
            if condition is None:
                variables.update(EF_ROOT=str(root), EF_ACTION=action, EF_RANKS='128')
                script = root / 'source/scripts/eternafold/qsub_pruned_traina.sh'
            else:
                variables.update(EF_FULL_ROOT=str(root / 'full_test' / condition), EF_ACTION='aggregate')
                script = root / 'full_test' / condition / 'source/scripts/eternafold/qsub_fulltest_new_metrics.sh'
            command = ['qsub', '-terse', '-g', 'tga-satolab-gtex', '-cwd', '-S', '/bin/bash',
                       '-j', 'y', '-o', str(root / 'scheduler'), '-N', name,
                       '-l', 'cpu_160=1' if array else 'cpu_8=1',
                       '-l', 'h_rt=01:00:00']
            if holds:
                command += ['-hold_jid', ','.join(map(str, holds))]
            if array:
                command += ['-t', array, '-tc', '4']
            command += ['-v', ','.join(f'{k}={v}' for k, v in variables.items()), str(script)]
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            job_id = result.stdout.strip().split('.')[0]
            if not job_id.isdigit():
                raise ValueError('Unexpected qsub response: ' + result.stdout)
            record({'action': action, 'job_id': int(job_id), 'name': name, 'holds': list(holds),
                    'array_range': array, 'condition': condition, 'command': command})
            return int(job_id)

        early = submit('ef3166_full_AS', 'full_predict', array='1-32')
        sums = [submit('ef3166_sum_' + label, 'aggregate', holds=[early], condition=condition)
                for label, condition in [('A', 'pretrain'), ('AS', 'joint_synthetic_2A3')]]
        later = submit('ef3166_full_AR', 'full_predict', array='33-48',
                       holds=[rn_check['job_id'], early])
        submit('ef3166_summary', 'summarize', holds=[later, *sums])


if __name__ == '__main__':
    main()
