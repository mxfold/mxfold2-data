"""Evaluate pretraining and fine-tuning checkpoints on four fixed panels."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import fcntl
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean
import time

from scripts.default_model import prepare
from scripts.efold_hf import workflow as wf

PANEL_COUNTS = {'TestSetB': 430, 'bpRNAnew': 5401, 'viral_fragments': 40, 'lncRNA': 30}
STAGES = ('pretrain', 'ribonanza', 'mammalian')
ORIGINS = (5, 10, 15, 20)
BPRNA_PANEL = prepare.STRUCTURES['bpRNAnew']
EFOLD_PANEL = prepare.LOG / 'paper_full_20261030/panel.jsonl'


def panel_rows(project):
    for source, key, sequence, pairs in prepare.rivas_records(project / prepare.RIVAS):
        if source == 'TestSetB':
            yield {'id': f'{source}:{key}', 'source': source,
                   'sequence': wf.normalize(sequence), 'pairs': pairs}
    for path, sources in ((BPRNA_PANEL, ('bpRNAnew',)), (EFOLD_PANEL, ('viral_fragments', 'lncRNA'))):
        parent = (project / path).parent
        if wf.digest(parent / 'panel.jsonl') != wf.plan_for(parent)['panel_sha256']:
            raise RuntimeError(f'Evaluation panel changed: {path}')
        for item in prepare.nr80.rows(project / path):
            if item['source'] in sources:
                yield {key: item[key] for key in ('id', 'source', 'sequence', 'pairs')}


def prepare_panels(parent, plan):
    project = Path(plan['project'])
    data = parent / 'data'
    manifest = json.loads((parent / 'data-complete.json').read_text())
    for name in ('structure.jsonl', 'ribonanza.jsonl', 'mammalian.jsonl', 'structure-membership.tsv'):
        if wf.digest(data / name) != manifest['sha256'][f'data/{name}']:
            raise RuntimeError(f'Training data changed: {name}')
    for path in (prepare.RIVAS, BPRNA_PANEL):
        if wf.digest(project / path) != plan['input_sha256'][str(path)]:
            raise RuntimeError(f'Evaluation input differs from the pretraining source: {path}')

    rows = list(panel_rows(project))
    counts = Counter(item['source'] for item in rows)
    if counts != PANEL_COUNTS or len({row['id'] for row in rows}) != len(rows):
        raise RuntimeError(f'Unexpected evaluation panel counts: {counts}')
    membership = {}
    with (data / 'structure-membership.tsv').open() as handle:
        for row in csv.DictReader(handle, delimiter='\t'):
            source = row['source']
            if source == 'TestSetB':
                membership[row['id']] = row['status']
            elif source == 'bpRNAnew':
                membership[row['id'].removeprefix('bpRNAnew:')] = row['status']

    index = defaultdict(list)
    for row in rows:
        sequence = row['sequence']
        if not prepare.IUPAC.fullmatch(sequence):
            raise ValueError(f'Unsupported evaluation sequence: {row["id"]}')
        row['pairs'] = wf.structure_pairs(row['pairs'], len(sequence))
        row['sequence'] = sequence.translate(prepare.AMBIGUOUS)
        row['sequence_sha256'] = hashlib.sha256(row['sequence'].encode()).hexdigest()
        row['structure_nr80_status'] = membership.get(row['id'])
        if row['source'] in ('TestSetB', 'bpRNAnew') and row['structure_nr80_status'] is None:
            raise RuntimeError(f'Missing structure membership for {row["id"]}')
        row['exact_match_training'] = []
        index[row['sequence']].append(row)
    for name in ('structure', 'ribonanza', 'mammalian'):
        for training in prepare.nr80.rows(data / f'{name}.jsonl'):
            for row in index.get(training['sequence'], ()):
                if name not in row['exact_match_training']:
                    row['exact_match_training'].append(name)

    output = parent / 'evaluation'
    output.mkdir(exist_ok=True)
    path = output / 'panel.jsonl'
    temporary = path.with_suffix('.tmp')
    wf.write_jsonl(temporary, rows)
    temporary.replace(path)
    summary = {
        'path': str(path.resolve()), 'sha256': wf.digest(path), 'counts': counts,
        'exact_matches_any_training': {
            source: sum(row['source'] == source and bool(row['exact_match_training']) for row in rows)
            for source in PANEL_COUNTS},
        'length_max': {source: max(len(row['sequence']) for row in rows if row['source'] == source)
                       for source in PANEL_COUNTS},
        'metric': 'Macro F1 over exact base pairs, including all reference RNAs.',
        'source_sha256': {str(p): wf.digest(project / p)
                          for p in (prepare.RIVAS, BPRNA_PANEL, EFOLD_PANEL)},
    }
    wf.write_json(output / 'panel-manifest.json', summary)
    return summary


def checkpoints(root, plan):
    for phase in STAGES:
        epochs = plan[f'{phase}_epochs']
        for epoch in ([epochs] if phase == 'pretrain' else range(1, epochs+1)):
            path = root / 'runs' / phase / f'epoch-{epoch:03d}.pt'
            yield phase, epoch, path


def summarize(rows):
    result = {}
    for source in PANEL_COUNTS:
        group = [row for row in rows if row['source'] == source]
        if len(group) != PANEL_COUNTS[source]:
            raise RuntimeError(f'Incomplete evaluation coverage: {source}')
        tp, fp, fn = (sum(row[key] for row in group) for key in ('tp', 'fp', 'fn'))
        denominator = 2*tp + fp + fn
        result[source] = {'sequences': len(group), 'f1': fmean(row['f1'] for row in group),
                          'micro_f1': 2*tp/denominator if denominator else 1.,
                          'exact_match_any_training': sum(bool(row['exact_match_training']) for row in group)}
    return result


def score_checkpoint(root, plan, panel, phase, epoch, checkpoint, deadline):
    import torch

    folder = root / 'evaluation' / phase / f'epoch-{epoch:03d}'
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / 'per_sequence.jsonl'
    receipt = folder / 'summary.json'
    checksum = wf.digest(checkpoint)
    binding = {'checkpoint_sha256': checksum, 'panel_sha256': plan['evaluate']['sha256']}
    if receipt.exists():
        result = json.loads(receipt.read_text())
        if result['binding'] != binding or wf.digest(output) != result['predictions_sha256']:
            raise RuntimeError(f'Completed evaluation files changed: {folder}')
        return result
    binding_path = folder / 'input.json'
    if binding_path.exists() and json.loads(binding_path.read_text()) != binding:
        raise RuntimeError(f'Inputs changed during evaluation: {folder}')
    wf.write_json(binding_path, binding)
    model, optimizer, structural, chemical, _, _ = wf.build(root, 'pretrain', -1)
    del optimizer, structural, chemical
    state = wf.restore(checkpoint, model)
    if state['phase'] != phase or state['epoch'] != epoch:
        raise RuntimeError(f'Wrong evaluation checkpoint: {checkpoint}')
    model.eval()

    done = wf.read_jsonl(output) if output.exists() else []
    expected = {row['id'] for row in panel}
    ids = {row['id'] for row in done}
    if len(ids) != len(done) or not ids <= expected:
        raise RuntimeError(f'Invalid cached predictions: {output}')
    with torch.inference_mode(), output.open('a') as handle:
        for item in panel:
            if item['id'] in ids:
                continue
            sequence = item['sequence']
            score, _, partners = model([sequence], return_partfunc=False)
            bp = partners[0]
            if len(bp) != len(sequence)+1 or not math.isfinite(float(score[0])):
                raise RuntimeError(f'Invalid prediction: {item["id"]}')
            pairs = [(i-1, int(j)-1) for i, j in enumerate(bp) if i and j > i]
            pairs = wf.structure_pairs(pairs, len(sequence))
            row = {k: item[k] for k in ('id', 'source', 'sequence_sha256', 'exact_match_training',
                                        'structure_nr80_status')}
            row.update(wf.pair_metrics(item['pairs'], pairs), predicted_pairs=pairs)
            handle.write(json.dumps(row, allow_nan=False)+'\n')
            done.append(row)
            if len(done) % 500 == 0:
                handle.flush()
                print(phase, epoch, len(done), '/', len(panel), flush=True)
            if time.monotonic() >= deadline:
                return None
    if len(done) != len(panel):
        raise RuntimeError('Incomplete predictions')
    result = {'phase': phase, 'epoch': epoch, 'binding': binding,
              'predictions_sha256': wf.digest(output), 'datasets': summarize(done)}
    wf.write_json(receipt, result)
    return result


def setup(parent):
    """Create independent workers for the four origins and the needed rerun."""
    import shutil

    from scripts.default_model import run

    plan = run.verify(parent)
    if plan.get('mode') != 'pretrain' or plan['pretrain_milestones'] != [15, 20]:
        raise RuntimeError('Parent must retain original pretraining epochs 15 and 20')
    base = parent / 'branches'
    if base.exists():
        raise FileExistsError(base)
    panel = prepare_panels(parent, plan)
    stage = parent / 'branches.pending'
    frozen = stage / 'source'
    shutil.copytree(parent / 'source', frozen, ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('train.py', 'run.py', 'evaluate.py'):
        shutil.copy2(Path(__file__).with_name(name), frozen / 'scripts/default_model' / name)
    shared = plan['frozen_sha256'].copy()
    for name in ('train.py', 'run.py', 'evaluate.py'):
        label = f'source/scripts/default_model/{name}'
        shared[label] = wf.digest(stage / label)
    for epoch in (*ORIGINS, None):
        target = stage / (f'epoch-{epoch:03d}' if epoch else 'rerun')
        target.mkdir()
        (target / 'scheduler').mkdir()
        # This link is relative so the entire tree can be published atomically.
        (target / 'source').symlink_to('../source', target_is_directory=True)
        (target / 'data').symlink_to(parent / 'data', target_is_directory=True)
        for name in ('baseline.conf', 'job.sh', 'data-complete.json'):
            shutil.copy2(parent / name, target)
        (target / 'bin').symlink_to(parent / 'bin', target_is_directory=True)
        branch_plan = {**plan, 'frozen_sha256': shared,
                       'pretrain_epochs': epoch or 10,
                       'mode': 'finetune' if epoch else 'pretrain',
                       'pretrain_milestones': [] if epoch else [5, 10],
                       'comparison_root': str(parent),
                       'checkpoint_origin': 'rerun' if epoch in (None, 5, 10) else 'original'}
        if epoch:
            branch_plan['evaluate'] = panel
            branch_plan['scheduler'] = {**branch_plan['scheduler'], 'cpu': 'cpu_4=1'}
        else:
            (target / 'branches').symlink_to('..', target_is_directory=True)
        wf.write_json(target / 'plan.json', branch_plan)
        run.verify(target)
    stage.rename(base)
    return panel


def update_comparison(parent):
    folder = parent / 'evaluation'
    with (folder / 'comparison.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = []
        complete = []
        for origin in ORIGINS:
            branch = parent / 'branches' / f'epoch-{origin:03d}'
            marker = branch / 'evaluation/complete.json'
            files = sorted((branch / 'evaluation').glob('*/epoch-*/summary.json'))
            if marker.exists():
                complete.append(origin)
            for path in files:
                result = json.loads(path.read_text())
                for source, score in result['datasets'].items():
                    rows.append({'pretrain_epoch': origin,
                                 'checkpoint_origin': 'rerun' if origin <= 10 else 'original',
                                 'phase': result['phase'], 'epoch': result['epoch'],
                                 'dataset': source, **score})
        rows.sort(key=lambda row: (row['pretrain_epoch'], STAGES.index(row['phase']),
                                   row['epoch'], row['dataset']))
        temporary = folder / 'comparison.tsv.tmp'
        with temporary.open('w') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter='\t', lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(folder / 'comparison.tsv')
        wf.write_json(folder / 'status.json', {'completed_origins': complete,
                                                'expected_origins': ORIGINS, 'score_rows': len(rows)})


def work(root, deadline):
    from mxfold2 import interface

    plan = wf.plan_for(root)
    eval_plan = plan['evaluate']
    path = Path(eval_plan['path'])
    if wf.digest(path) != eval_plan['sha256']:
        raise RuntimeError('Evaluation panel changed')
    panel = wf.read_jsonl(path)
    if Counter(row['source'] for row in panel) != PANEL_COUNTS:
        raise RuntimeError('Evaluation panel counts changed')
    interface.set_num_threads(4)
    wf.seed_all(wf.SEED)
    parent = Path(plan['comparison_root'])
    for phase, epoch, checkpoint in checkpoints(root, plan):
        if score_checkpoint(root, plan, panel, phase, epoch, checkpoint, deadline) is None:
            return False
        update_comparison(parent)
    wf.write_json(root / 'evaluation/complete.json', {'pretrain_epoch': plan['pretrain_epochs'],
                                                      'models': 1 + plan['ribonanza_epochs'] + plan['mammalian_epochs']})
    update_comparison(parent)
    return True
