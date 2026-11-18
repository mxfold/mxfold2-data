#!/usr/bin/env python3
"""Submit the pair-pruned 3166-record training/evaluation comparison once."""
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
    ledger = root / 'submissions.jsonl'
    with (root / '.submission.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if ledger.exists():
            raise FileExistsError('This training/evaluation DAG was already submitted')
        def job(name, action, *, resource, ranks, hold=(), condition='', segment=1, walltime='23:50:00', array=False):
            command = ['qsub', '-terse', '-g', 'tga-satolab-gtex', '-cwd', '-S', '/bin/bash', '-j', 'y',
                       '-o', str(root / 'scheduler'), '-N', name, '-l', resource, '-l', 'h_rt=' + walltime]
            if hold: command += ['-hold_jid', ','.join(map(str, hold))]
            if array: command += ['-t', '1-48', '-tc', '4']
            command += ['-v', f'EF_PROJECT_ROOT={repo},EF_ROOT={root},EF_ACTION={action},EF_RANKS={ranks},EF_CONDITION={condition},EF_SEGMENT={segment}',
                        str(root / 'source/scripts/eternafold/qsub_pruned_traina.sh')]
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            jid = result.stdout.strip().split('.')[0]
            if not jid.isdigit(): raise ValueError('Unexpected qsub response: ' + result.stdout)
            info = {'job_id': int(jid), 'action': action, 'condition': condition, 'segment': segment,
                    'name': name, 'resource': resource, 'ranks': ranks, 'dependencies': list(hold),
                    'array_tasks': 48 if array else 1, 'command': command}
            with ledger.open('a') as handle: handle.write(json.dumps(info) + '\n')
            print(json.dumps({k:v for k,v in info.items() if k != 'command'}), flush=True)
            return int(jid)
        prep = job('ef3166_prep', 'prepare', resource='cpu_40=1', ranks=32, walltime='02:00:00')
        checks = []
        for condition, label, segments, resource, ranks in (
            ('pretrain', 'A', 2, 'cpu_80=1', 64),
            ('joint_synthetic_2A3', 'AS', 2, 'cpu_80=1', 64),
            ('joint_rnagym_all', 'AR', 6, 'cpu_160=1', 128),
        ):
            previous = prep
            for segment in range(1, segments + 1):
                previous = job(f'ef3166_{label}_{segment}', 'train', resource=resource, ranks=ranks,
                               hold=[previous], condition=condition, segment=segment)
            evaluation = job(f'ef3166_eval_{label}', 'evaluate', resource='cpu_16=1', ranks=16,
                             hold=[previous], condition=condition, walltime='03:00:00')
            checks.append(job(f'ef3166_check_{label}', 'full_setup', resource='cpu_8=1', ranks=4,
                              hold=[evaluation], condition=condition, walltime='00:30:00'))
        full = job('ef3166_full', 'full_predict', resource='cpu_160=1', ranks=128,
                   hold=checks, array=True, walltime='06:00:00')
        job('ef3166_summary', 'summarize', resource='cpu_8=1', ranks=1, hold=[full], walltime='01:00:00')


if __name__ == '__main__':
    main()
