#!/usr/bin/env python3
"""Inventory earlier chemical evaluations and build resumable inference tasks."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.chemical_consistency.prepare import OUT, sha


def main():
    top = ROOT / 'qsub_logs/rnagym'
    files = subprocess.check_output(['rg', '--files', str(top)], text=True).splitlines()
    paths = sorted(Path(x) for x in files if x.endswith('/RNAgym_official_test_panel.csv'))
    runs, missing, baselines = {}, [], defaultdict(list)
    for path in paths:
        epoch = int(path.parent.name.removeprefix('epoch-'))
        run = path.parent.parent.parent.parent
        group = path.relative_to(top).parts[0]
        group_root = top / group
        relative = run.relative_to(group_root)
        label = group + ('__' + '__'.join(relative.parts) if relative.parts else '')
        source = run / 'source' if (run / 'source/mxfold2').is_dir() else group_root / 'source'
        config = (run / 'baseline.conf' if (run / 'baseline.conf').is_file() else
                  group_root / 'model.conf' if (group_root / 'model.conf').is_file() else
                  group_root / 'baseline.conf')
        if not (source / 'mxfold2').is_dir() or not config.is_file():
            missing.append({'evaluation': str(path), 'epoch': epoch, 'reason': 'frozen source/config missing'})
            continue
        if epoch == 0:
            # This experiment trained a randomly initialized student; its
            # initial_baseline.pth is instead the pseudo-label teacher.
            if group == 'mxfold2-trainsetab-joint-20261022':
                choices = [group_root / 'initial_random.pt']
            else:
                choices = [run / 'initial_baseline.pth', group_root / 'initial_baseline.pth',
                           group_root / 'initial_random.pt']
        else:
            choices = [run / 'runs' / f'epoch-{epoch:03d}.pt', run / 'runs' / f'epoch-{epoch}.pt']
            if epoch in (3, 5, 100):
                choices.append(run / 'runs/latest.pt')
        checkpoint = next((c for c in choices if c.is_file()), None)
        if checkpoint is None:
            missing.append({'evaluation': str(path), 'epoch': epoch, 'reason': 'checkpoint unavailable'})
            continue
        entry = {'model': label, 'epoch': epoch, 'checkpoint': str(checkpoint),
                 'reference': str(path), 'expected_state_epoch': epoch if checkpoint.name == 'latest.pt' else None}
        if epoch == 0:
            # Reuse an identical initial checkpoint only when its archived
            # source, model config, AND prior evaluation CSV are identical.
            key = (str(source), sha(config), sha(checkpoint), sha(path))
            baselines[key].append((entry, source, config))
        else:
            k = (str(source), str(config), label)
            task = runs.setdefault(k, {'model': label, 'source': str(source),
                                       'config': str(config), 'entries': []})
            task['entries'].append(entry)
    for candidates in baselines.values():
        representative, source, config = candidates[0]
        representative['aliases'] = [entry['model'] for entry, _, _ in candidates]
        key = (str(source), str(config), representative['model'] + '__baseline')
        runs[key] = {'model': representative['model'] + '__baseline', 'source': str(source),
                     'config': str(config), 'entries': [representative]}
    tasks = []
    for task in runs.values():
        task['entries'].sort(key=lambda x: -x['epoch'])
        # Run every final checkpoint before the longer intermediate-epoch
        # sweep, so completed comparisons appear promptly.
        first, rest = task['entries'][:1], task['entries'][1:]
        tasks.append({**task, 'entries': first, 'phase': 'final_or_baseline'})
        if rest:
            tasks.append({**task, 'entries': rest, 'phase': 'intermediate'})
    tasks.sort(key=lambda t: (t['phase'] == 'intermediate',
                              t['entries'][0]['epoch'] != 0,
                              -t['entries'][0]['epoch'], t['model']))
    plan = {'tasks': tasks, 'missing': missing, 'evaluations': len(paths),
            'planned_unique_checkpoints': sum(len(t['entries']) for t in tasks),
            'represented_evaluations': sum(sum(len(e.get('aliases', [e['model']])) for e in t['entries']) for t in tasks)}
    (OUT / 'rnagym_tasks.json').write_text(json.dumps(plan, indent=2) + '\n')
    print(json.dumps({'tasks': len(tasks), 'evaluations': len(paths), 'missing': len(missing),
                      'planned_unique_checkpoints': plan['planned_unique_checkpoints'],
                      'represented_evaluations': plan['represented_evaluations'],
                      'missing_by_reason': dict(Counter(x['reason'] for x in missing))}, indent=2))


if __name__ == '__main__':
    main()
