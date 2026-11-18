#!/usr/bin/env python3
"""Submit the guarded full-test evaluation DAG once."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--repo', type=Path, required=True)
    a = p.parse_args()
    root, repo = a.root.resolve(), a.repo.resolve()
    plan = json.loads((root / 'plan.json').read_text())
    ledger = root / 'submissions.jsonl'
    with (root / '.submission.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if ledger.exists():
            raise FileExistsError('This evaluation DAG was already submitted')
        def submit(name, action, resource, walltime, hold=None, array=False):
            command = ['qsub', '-terse', '-g', 'tga-satolab-gtex', '-cwd', '-S', '/bin/bash',
                       '-j', 'y', '-o', str(root / 'scheduler'), '-N', name, '-l', resource,
                       '-l', 'h_rt=' + walltime]
            if hold:
                command += ['-hold_jid', str(hold)]
            if array:
                command += ['-t', f'1-{plan["shards"]}', '-tc', str(plan['array_concurrency'])]
            command += ['-v', f'EF_PROJECT_ROOT={repo},EF_FULL_ROOT={root},EF_ACTION={action},EF_RANKS={plan["ranks"]},EF_WORKERS={plan["score_workers"]}',
                        str(root / 'source/scripts/eternafold/qsub_fulltest_new_metrics.sh')]
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            job = result.stdout.strip().split('.')[0]
            if not job.isdigit():
                raise ValueError('Unexpected qsub response: ' + result.stdout)
            record = {'job_id': int(job), 'action': action, 'name': name, 'resource': resource,
                      'hold_jid': hold, 'array': array, 'command': command}
            with ledger.open('a') as handle:
                handle.write(json.dumps(record) + '\n')
            print(json.dumps({k:v for k,v in record.items() if k != 'command'}), flush=True)
            return job
        prep = submit('ef_full_prep', 'prepare', 'cpu_40=1', '01:00:00')
        check = submit('ef_full_check', 'preflight', 'cpu_8=1', '00:30:00', prep)
        run = submit('ef_full_test', 'predict', 'cpu_160=1', '06:00:00', check, array=True)
        submit('ef_full_sum', 'aggregate', 'cpu_8=1', '00:30:00', run)


if __name__ == '__main__':
    main()
