#!/usr/bin/env python3
"""Freeze a complete, resumable inventory for MAP-only re-evaluation."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.chemical_consistency.map_mw_metrics import INPUT, OUT, sha, write_json


def config_for(source):
    folder = Path(source).parent
    for path in (folder / 'baseline.conf', folder / 'model.conf'):
        if path.is_file():
            return str(path)
    raise FileNotFoundError('Frozen configuration: ' + str(folder))


def main():
    previous = json.loads((INPUT / 'rnagym_tasks.json').read_text())
    tasks = []
    for task in previous['tasks']:
        for entry in task['entries']:
            tasks.append({'source': task['source'], 'config': task['config'], **entry,
                          'phase': task['phase'], 'family': 'modern'})
    old = json.loads((INPUT / 'historical_tasks.json').read_text())
    # These older labels can point to the very same checkpoint. Restore and
    # infer it once only when its checkpoint, configuration, and source match.
    old_unique = {}
    for entry in old['tasks']:
        key = (sha(entry['checkpoint']), sha(entry['config']))
        model = entry['model']
        if key in old_unique:
            old_unique[key]['aliases'].append(model)
        else:
            old_unique[key] = {**entry, 'source': str(INPUT / 'historical_source'),
                               'epoch': int(Path(entry['checkpoint']).name.removeprefix('epoch-')),
                               'aliases': [model], 'family': 'historical', 'phase': 'final_or_baseline'}
    tasks.extend(old_unique.values())
    def priority(task):
        name = task['model']
        main = ('replay20-full-test' in name or 'trainsetb-synthetic' in name)
        return (not main, task['phase'] == 'intermediate', task['family'] == 'historical',
                task['epoch'] != 0, name, -task['epoch'])
    tasks.sort(key=priority)
    panel = {'tasks': tasks, 'missing': previous['missing'] + old['missing'],
             'modern_represented_evaluations': previous['represented_evaluations'],
             'historical_represented_evaluations': len(old['tasks']),
             'historical_unique_checkpoints': len(old_unique),
             'profiles_sha256': sha(INPUT / 'profiles.jsonl'),
             'note': 'Common RNAgym panel and TrainSetB simulated profiles; historical original full/subsample scopes are recorded separately'}
    write_json(OUT / 'panel_tasks.json', panel)
    full_source = ROOT / 'qsub_logs/rnagym/replay20-full-test-20261023'
    plan = json.loads((full_source / 'comparison_plan.json').read_text())
    full_tasks = []
    for label, entry in plan['models'].items():
        expected_sha = json.loads((full_source / 'evaluation' / label / 'metrics.json').read_text())['checkpoint_sha256']
        if sha(entry['checkpoint']) != expected_sha:
            raise ValueError('Original full-test checkpoint changed: ' + label)
        for shard in range(1, 17):
            full_tasks.append({**entry, 'model': label, 'config': config_for(entry['source']),
                               'shard': shard, 'checkpoint_sha256': expected_sha})
    full_tasks.sort(key=lambda t: (t['model'] != 'baseline', 'replay20' not in t['model'], t['shard'], t['model']))
    write_json(OUT / 'full_tasks.json', {'source': str(full_source), 'tasks': full_tasks,
                                       'input_meta_sha256': sha(full_source / 'test_data/complete.json')})
    print(json.dumps({'panel_unique_checkpoints': len(tasks), 'full_tasks': len(full_tasks),
                      'represented_evaluations': previous['represented_evaluations'] + len(old['tasks']),
                      'missing': len(panel['missing'])}, indent=2))


if __name__ == '__main__':
    main()
