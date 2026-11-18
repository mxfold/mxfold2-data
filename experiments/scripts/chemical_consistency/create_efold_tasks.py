#!/usr/bin/env python3
"""Create one resumable inference task per eFold experiment arm."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.chemical_consistency.prepare import OUT


def main():
    old = ROOT / 'qsub_logs/efold_hf/bprna100_ft5_replay20_20261024'
    new = ROOT / 'qsub_logs/efold_hf/bprna10_restored_ft5_replay20_20261026'
    groups = [(old, 'pretrain', 'pretrain', [10])]
    groups += [(new, arm, arm, list(range(1, 6))) for arm in
               ('mw_rank_restored', 'nll_restored', 'shape_fy_restored')]
    groups += [(old, arm, 'original_' + arm, list(range(1, 6))) for arm in
               ('rank', 'nll', 'shape_fy', 'assisted')]
    tasks = []
    for root, arm, label, epochs in groups:
        entries = []
        for epoch in epochs:
            run = root / 'runs' / arm
            evaluation = root / 'evaluation' / arm / f'epoch-{epoch:03d}/per_sequence.jsonl'
            entries.append({'model': label, 'epoch': epoch,
                            'checkpoint': str(run / f'epoch-{epoch:03d}.pt'),
                            'marker': str(run / f'epoch-{epoch:03d}-complete.json'),
                            'reference': str(evaluation)})
        for entry in entries:
            for key in ('checkpoint', 'marker', 'reference'):
                if not Path(entry[key]).is_file():
                    raise FileNotFoundError(entry[key])
        tasks.append({'arm': label, 'source': str(root / 'source'),
                      'config': str(root / 'baseline.conf'), 'entries': entries})
    (OUT / 'efold_tasks.json').write_text(json.dumps(tasks, indent=2) + '\n')
    print(f'{len(tasks)} tasks, {sum(len(t["entries"]) for t in tasks)} evaluated checkpoints')


if __name__ == '__main__':
    main()
