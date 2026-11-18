#!/usr/bin/env python3
"""Inventory older RNAgym chemical evaluations for panel re-evaluation."""
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.chemical_consistency.prepare import OUT


def main():
    top = ROOT / 'qsub_logs/rnagym'
    paths = sorted(Path(x) for x in subprocess.check_output(['rg', '--files', str(top)], text=True).splitlines()
                   if x.endswith('/chemical_metrics.csv'))
    tasks, missing = [], []
    for path in paths:
        info = next((p for p in (path.parent / 'run_info.txt', path.parent.parent / 'run_info.txt')
                     if p.exists()), None)
        if info is None:
            missing.append({'evaluation': str(path), 'reason': 'run_info missing'})
            continue
        metadata = dict(line.split('=', 1) for line in info.read_text().splitlines() if '=' in line)
        checkpoint = Path(metadata['checkpoint'])
        candidates = [checkpoint.parent / 'model.conf', checkpoint.parent.parent / 'model.conf',
                      checkpoint.parent.parent.parent / 'model.conf']
        if metadata.get('label') == 'baseline' or 'baseline/final/' in str(path):
            candidates.append(Path('/gs/bs/tga-satolab-gtex/yamauchi/TrainA_only_new_parameter/TrainSetA.conf'))
        config = next((p for p in candidates if p.is_file()), None)
        map_prediction = path.parent / 'rnagym_test/predict.stdout'
        structure_metrics = path.parent.parent / 'TestSetB/result.csv'
        if not checkpoint.is_file() or config is None or not (map_prediction.is_file() or structure_metrics.is_file()):
            missing.append({'evaluation': str(path), 'reason': 'checkpoint, config or verification reference missing'})
            continue
        if metadata.get('git') != '33fc667d7c7512e855affb96e3e6652589fbff19':
            missing.append({'evaluation': str(path), 'reason': 'unexpected historical source revision'})
            continue
        label = 'historical__' + '__'.join(path.relative_to(top).parts[:-1])
        tasks.append({'model': label, 'checkpoint': str(checkpoint), 'config': str(config),
                      'evaluation': str(path), 'recorded_git_revision': metadata['git'],
                      'reference_type': 'subset_MAP' if map_prediction.is_file() else 'TestSetB_counts',
                      'reference': str(map_prediction if map_prediction.is_file() else structure_metrics),
                      'original_dataset': 'RNAgym_1of100' if map_prediction.is_file() else 'RNAgym_full_test'})
    plan = {'tasks': tasks, 'missing': missing, 'source_revision':
            '33fc667d7c7512e855affb96e3e6652589fbff19',
            'native_interface_source': str(ROOT / 'qsub_logs/rnagym/mxfold2-trainsetab-joint-20261022/source/mxfold2/interface.cpython-313-x86_64-linux-gnu.so')}
    (OUT / 'historical_tasks.json').write_text(json.dumps(plan, indent=2) + '\n')
    print(json.dumps({'tasks': len(tasks), 'missing': len(missing),
                      'original_datasets': dict(Counter(t['original_dataset'] for t in tasks))}, indent=2))


if __name__ == '__main__':
    main()
