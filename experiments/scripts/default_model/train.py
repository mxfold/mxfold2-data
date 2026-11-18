"""FY pretraining, then IMLE-Rank on Ribonanza and mammalian profiles."""
from __future__ import annotations

import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

import numpy as np
import torch

from scripts.efold_hf import workflow as wf
from scripts.efold_hf.matched_losses import make_replay_stream

PHASES = ('pretrain', 'ribonanza', 'mammalian')
RANK_ARM = 'mw_rank_restored'
CHECKPOINT_INTERVAL = 400


def build(root, arm, gpu):
    model, optimizer, structural, chemical, shapes, device = wf.build(root, arm, gpu)
    model.train()
    return model, optimizer, structural, chemical, shapes, device


def label_for(row, epoch):
    alternatives = row['alternatives']
    index = (epoch - 1) % (1 + len(alternatives))
    if index:
        return {**row, **alternatives[index-1]}
    return row


def backpropagate(row, structural, chemical, arm, device, scale):
    loss = wf.sequence_loss(row, structural, chemical, arm, device)
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite training loss')
    (loss * scale).backward()
    return float(loss.detach())


def replay_per_epoch(primary_count):
    return (primary_count + 2) // 4


def epoch_order(phase, primary_count, structure_count, epoch, seed, earlier_replay, earlier_epochs):
    if phase == 'pretrain':
        return wf.schedule(primary_count, structure_count, epoch, seed=seed, replay=False)
    replay_count = replay_per_epoch(primary_count)
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch+earlier_epochs, 9823]))
    order = [('primary', int(i)) for i in rng.permutation(primary_count)]
    replay = make_replay_stream(structure_count, earlier_replay + (epoch-1)*replay_count, replay_count, seed)
    order += [('replay', i) for i in replay]
    rng.shuffle(order)
    return order


def preflight(root, structures, ribonanza, mammalian, gpu):
    if (root / 'preflight.json').exists():
        return
    checked = []
    for phase, rows in zip(PHASES, (structures, ribonanza, mammalian), strict=True):
        arm = 'pretrain' if phase == 'pretrain' else RANK_ARM
        settings = wf.training_settings(root, arm)
        wf.seed_all(settings['seed'])
        model, optimizer, structural, chemical, _, device = build(root, arm, gpu)
        row = max(rows, key=lambda item: len(item['sequence']))
        print(f'{phase} preflight {row["id"]} ({len(row["sequence"])} nt)', flush=True)
        value = backpropagate(row, structural, chemical, arm, device, settings['chemical_weight'])
        wf.optimizer_step(optimizer)
        checked.append({'phase': phase, 'id': row['id'], 'length': len(row['sequence']), 'loss': value})
        del model, optimizer, structural, chemical
    result = {'success': True, 'python': sys.version, 'torch': torch.__version__, 'cuda': torch.version.cuda,
              'checked': checked, 'peak_host_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    if device.type == 'cuda':
        result['gpu'] = torch.cuda.get_device_name(device)
        result['device_memory_bytes'] = torch.cuda.get_device_properties(device).total_memory
        result['peak_allocated_bytes'] = torch.cuda.max_memory_allocated(device)
    wf.write_json(root / 'preflight.json', result)
    print(json.dumps(result), flush=True)


def final_checkpoint(root, phase, epochs):
    folder = root / 'runs' / phase
    marker = json.loads((folder / 'complete.json').read_text())
    path = folder / f'epoch-{epochs:03d}.pt'
    if marker['epochs'] != epochs or marker['checkpoint_sha256'] != wf.digest(path):
        raise RuntimeError(f'Invalid completed checkpoint: {path}')
    return path


def publish_branch(root, checkpoint):
    """Expose a saved pretraining milestone to its own fine-tuning worker."""
    from scripts.default_model import run

    branch = root / 'branches' / checkpoint.stem
    if not branch.is_dir():
        return
    destination = branch / 'runs/pretrain' / checkpoint.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        destination.hardlink_to(checkpoint)
    if not destination.samefile(checkpoint):
        raise RuntimeError(f'Branch already points at a different checkpoint: {destination}')
    if os.environ.get('JOB_ID') and not (branch / 'submissions.jsonl').exists():
        run.submit(branch, 'train')


def train_phase(root, phase, epochs, primary, structures, previous, earlier_replay, earlier_epochs, gpu, deadline):
    folder = root / 'runs' / phase
    plan = wf.plan_for(root)
    milestones = set(plan.get('pretrain_milestones', ())) if phase == 'pretrain' else set()
    if phase != 'pretrain' and plan.get('evaluate'):
        milestones.update(range(1, epochs+1))
    if phase == 'pretrain':
        for epoch in sorted(milestones):
            checkpoint = folder / f'epoch-{epoch:03d}.pt'
            if checkpoint.exists():
                publish_branch(root, checkpoint)
    if (folder / 'complete.json').exists():
        return final_checkpoint(root, phase, epochs)
    arm = 'pretrain' if phase == 'pretrain' else RANK_ARM
    settings = wf.training_settings(root, arm)
    wf.seed_all(settings['seed'])
    model, optimizer, structural, chemical, shapes, device = build(root, arm, gpu)
    latest = folder / 'latest.pt'
    state = {'phase': phase, 'epoch': 1, 'cursor': 0, 'updates': 0, 'loss_sum': 0.}
    if latest.exists():
        state = wf.restore(latest, model, optimizer, shapes)
        if state['phase'] != phase:
            raise RuntimeError(f'Wrong phase in {latest}')
    else:
        if phase == 'ribonanza':
            wf.initialize_finetuning(root, arm, model, optimizer, shapes, folder / 'initialization.json')
            wf.seed_all(settings['seed'])
        elif phase == 'mammalian':
            wf.restore(previous, model, optimizer, shapes)
        wf.save_checkpoint(latest, model, optimizer, shapes, state)

    final = folder / f'epoch-{epochs:03d}.pt'
    while state['epoch'] <= epochs:
        epoch = state['epoch']
        order = epoch_order(phase, len(primary), len(structures), epoch, settings['seed'],
                            earlier_replay, earlier_epochs)
        while state['cursor'] < len(order):
            batch = order[state['cursor']:state['cursor']+settings['accumulation']]
            for kind, index in batch:
                use_profile = kind == 'primary' and phase != 'pretrain'
                if use_profile:
                    row = primary[index]
                else:
                    row = label_for(structures[index], epoch)
                weight = settings['chemical_weight'] if use_profile else 1.
                try:
                    value = backpropagate(row, structural, chemical if use_profile else None,
                                          arm, device, weight / len(batch))
                except (RuntimeError, FloatingPointError) as error:
                    raise RuntimeError(f'{phase} epoch {epoch} slot {state["cursor"]} '
                                       f'{kind} {row.get("id", index)} ({len(row["sequence"])} nt)') from error
                state['loss_sum'] += value * weight
            wf.optimizer_step(optimizer)
            state['cursor'] += len(batch)
            state['updates'] += 1
            stopping = time.monotonic() >= deadline
            if state['updates'] % CHECKPOINT_INTERVAL == 0 or stopping:
                wf.save_checkpoint(latest, model, optimizer, shapes, state)
                progress = {**state, 'total_slots': len(order), 'epochs': epochs,
                            'checkpoint': str(latest), 'job_id': os.environ.get('JOB_ID'),
                            'updated_at': time.time()}
                wf.write_json(root / 'training.json', progress)
                print(json.dumps(progress), flush=True)
            if stopping:
                return None

        summary = {**state, 'primary_sequences': len(primary), 'replay_sequences': len(order)-len(primary),
                   'mean_loss': state['loss_sum']/len(order)}
        if epoch == epochs or epoch in milestones:
            checkpoint = folder / f'epoch-{epoch:03d}.pt'
            if not checkpoint.exists():
                wf.save_checkpoint(latest, model, optimizer, shapes, state)
                checkpoint.hardlink_to(latest)
            summary['checkpoint_sha256'] = wf.digest(checkpoint)
            if phase == 'pretrain' and epoch in milestones:
                publish_branch(root, checkpoint)
        wf.write_json(folder / f'epoch-{epoch:03d}.json', summary)
        print(json.dumps(summary), flush=True)
        state = {**state, 'epoch': epoch+1, 'cursor': 0, 'loss_sum': 0.}
        wf.save_checkpoint(latest, model, optimizer, shapes, state)

    wf.write_json(folder / 'complete.json', {'epochs': epochs, 'updates': state['updates'],
                                             'checkpoint_sha256': wf.digest(final)})
    return final


def export(root, final):
    folder = root / 'models'
    folder.mkdir(exist_ok=True)
    weights = folder / 'all_cdhit.pth'
    config = folder / 'all_cdhit.conf'
    saved = torch.load(final, map_location='cpu', weights_only=False)
    temporary = weights.with_suffix('.tmp')
    torch.save(saved['model_state_dict'], temporary)
    temporary.replace(weights)
    config.write_text((root / 'baseline.conf').read_text().rstrip() + '\n--param\nall_cdhit.pth\n')
    fasta = folder / 'smoke.fasta'
    sequence = 'GGGCGAAAGCCCUUCGGGCUUCGCCC'
    fasta.write_text(f'>export-smoke\n{sequence}\n')
    output = subprocess.check_output([sys.executable, '-m', 'mxfold2', 'predict', '@'+str(config), str(fasta)],
                                     cwd=root / 'source', text=True)
    lines = output.strip().splitlines()
    if len(lines) != 3 or lines[1] != sequence or len(lines[2].split()[0]) != len(sequence):
        raise RuntimeError(f'Exported model inference failed: {output}')
    (folder / 'smoke.out').write_text(output)
    wf.write_json(root / 'complete.json', {
        'version': '0.3.0', 'checkpoint': str(final), 'checkpoint_sha256': wf.digest(final),
        'model_sha256': {str(path.relative_to(root)): wf.digest(path) for path in (weights, config)},
        'prediction_smoke_test': 'passed',
    })


def work(root, gpu, deadline):
    plan = wf.plan_for(root)
    if (root / 'complete.json').exists():
        complete = json.loads((root / 'complete.json').read_text())
        for path, expected in complete.get('model_sha256', {}).items():
            if wf.digest(root / path) != expected:
                raise RuntimeError(f'Completed model changed: {path}')
        if plan.get('mode') == 'pretrain':
            train_phase(root, 'pretrain', plan['pretrain_epochs'], None, None, None, 0, 0, gpu, deadline)
        return True
    ready = json.loads((root / 'data-complete.json').read_text())
    for name, expected in ready['sha256'].items():
        if wf.digest(root / name) != expected:
            raise RuntimeError(f'Prepared data changed: {name}')
    structures = wf.read_jsonl(root / 'data/structure.jsonl')
    ribonanza = wf.read_jsonl(root / 'data/ribonanza.jsonl')
    mammalian = wf.read_jsonl(root / 'data/mammalian.jsonl')
    mode = plan.get('mode', 'all')
    if mode not in ('all', 'pretrain', 'finetune'):
        raise ValueError(f'Unsupported training mode: {mode}')
    if mode == 'finetune':
        checkpoint = wf.pretrain_checkpoint(root, plan)
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        epoch = plan['pretrain_epochs']
        updates = epoch * len(structures)
        state = saved['state']
        step = saved['optimizer_state_dict']['param_groups'][0]['step']
        if (state['phase'], state['epoch'], state['cursor'], state['updates'], step) != (
                'pretrain', epoch, len(structures), updates, updates):
            raise RuntimeError(f'Checkpoint is not the completed pretraining epoch {epoch}')
        del saved
    preflight(root, structures, ribonanza, mammalian, gpu)
    previous, replay, ft_epochs = None, 0, 0
    for phase, primary in zip(PHASES, (structures, ribonanza, mammalian), strict=True):
        if phase == 'pretrain' and mode == 'finetune':
            continue
        if phase != 'pretrain' and mode == 'pretrain':
            break
        epochs = plan[f'{phase}_epochs']
        previous = train_phase(root, phase, epochs, primary, structures, previous,
                               replay, ft_epochs, gpu, deadline)
        if previous is None:
            return False
        if phase != 'pretrain':
            replay += epochs * replay_per_epoch(len(primary))
            ft_epochs += epochs
    if mode == 'pretrain':
        wf.write_json(root / 'complete.json', {
            'mode': mode, 'checkpoint': str(previous), 'checkpoint_sha256': wf.digest(previous),
        })
    else:
        export(root, previous)
    return True
