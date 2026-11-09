"""MXfold2 training and checkpoint code from the primary experiment."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random

ARMS = ('rank', 'nll', 'shape_fy', 'assisted')
RESTORED_ARMS = ('mw_rank_restored', 'nll_restored', 'shape_fy_restored')
SUPPORTED_ARMS = (*ARMS, 'mw_rank', *RESTORED_ARMS)
SEED = 1234


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def read_jsonl(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle]


def write_jsonl(path, rows):
    with Path(path).open('w') as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False, separators=(',', ':')) + '\n')


def normalize(sequence):
    return sequence.upper().replace('T', 'U')


def structure_pairs(structure, length):
    pairs = sorted({tuple(sorted(map(int, p))) for p in structure if len(p) == 2})
    used = set()
    for i, j in pairs:
        if not 0 <= i < j < length or i in used or j in used:
            raise ValueError('Invalid or multiply paired reference')
        used.update((i, j))
    return pairs


def project_pairs(sequence, pairs):
    """Largest noncrossing canonical subset, for training constraints only."""
    candidates = [(i, j) for i, j in pairs if j-i > 3 and sequence[i]+sequence[j] in
                  ('AU', 'UA', 'CG', 'GC', 'GU', 'UG')]
    stack = []
    crossing = False
    for i, j in candidates:
        while stack and stack[-1] < i:
            stack.pop()
        if stack and j > stack[-1]:
            crossing = True
            break
        stack.append(j)
    if not crossing:
        return candidates
    import numpy as np
    n = len(sequence)
    dp = np.zeros((n+1, n+1), dtype=np.int16)
    partner = dict(candidates)
    # Intervals are half-open; each base has at most one reference partner.
    for i in range(n-1, -1, -1):
        dp[i, i+1:] = dp[i+1, i+1:]
        k = partner.get(i)
        if k is not None:
            dp[i, k+1:] = np.maximum(dp[i, k+1:], 1+dp[i+1, k]+dp[k+1, k+1:])
    answer, pending = [], [(0, n)]
    while pending:
        i, end = pending.pop()
        if end-i <= 1:
            continue
        k = partner.get(i)
        if k is not None and k < end and 1+dp[i+1, k]+dp[k+1, end] > dp[i+1, end]:
            answer.append((i, k))
            pending.extend(((i+1, k), (k+1, end)))
        else:
            pending.append((i+1, end))
    return sorted(answer)


def clean_observations(row):
    sequence = normalize(row['sequence'])
    output = []
    for probe in ('dms', 'shape'):
        values = row.get(probe)
        if not isinstance(values, list):
            continue
        if len(values) != len(sequence):
            raise ValueError('Reactivity length mismatch')
        # Source -1000 and nonfinite entries are missing, never low reactivity.
        # Preserve finite negative background-corrected values for rank/FY.
        values = [float(v) if v is not None and math.isfinite(v) and v > -10
                  and (probe != 'dms' or base in 'AC') else -999.
                  for base, v in zip(sequence, values)]
        if sum(v != -999 for v in values) >= 2:
            output.append({'probe': probe, 'values': values})
    return output


def plan_for(root):
    return json.loads((root / 'plan.json').read_text())


def pretrain_checkpoint(root, plan=None):
    plan = plan if plan is not None else plan_for(root)
    return root / 'runs/pretrain' / f"epoch-{int(plan['pretrain_epochs']):03d}.pt"


def verify(root):
    plan = plan_for(root)
    paths = [(root / 'source' / p, h) for p, h in plan['source_sha256'].items()]
    paths += [(root / 'data' / p, h) for p, h in plan['data_sha256'].items()]
    paths += [(root / 'baseline.conf', plan['config_sha256'])]
    for path, expected in paths:
        if digest(path) != expected:
            raise RuntimeError('Frozen input changed: '+str(path))
    return plan


def schedule(primary_count, replay_count, epoch, seed=SEED, replay=True, replay_mode='replacement'):
    import numpy as np
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, 9823]))
    primary = [('primary', int(i)) for i in rng.permutation(primary_count)]
    if replay:
        if not replay_count:
            raise ValueError('Replay pool is empty')
        n = int(math.floor(primary_count/4+.5))
        if replay_mode == 'cycle':
            from scripts.efold_hf.matched_losses import make_replay_stream
            replay_indices = make_replay_stream(replay_count, (epoch-1)*n, n, seed)
        else:
            replay_indices = rng.integers(replay_count, size=n)
        primary += [('replay', int(i)) for i in replay_indices]
        rng.shuffle(primary)
    return primary


def training_settings(root, arm):
    settings = dict(lr=.001, weight_decay=.01, accumulation=1, chemical_weight=1.,
                    restore_optimizer=False, freeze_turner=False, seed=SEED, replay_mode='replacement',
                    normalization='rms' if arm in ('rank', 'mw_rank') else 'none')
    if arm in RESTORED_ARMS:
        settings.update(lr=.0001, weight_decay=.0001, accumulation=4, chemical_weight=.5,
                        restore_optimizer=True, freeze_turner=True, normalization='rms', seed=0,
                        replay_mode='cycle')
    if (root/'plan.json').exists():
        settings.update(plan_for(root).get('arm_settings', {}).get(arm, {}))
    return settings


def build(root, arm, gpu):
    import torch
    from mxfold2.train import Train
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    sub = parser.add_subparsers()
    Train.add_args(sub)
    settings = training_settings(root, arm)
    argv = ['train', '@'+str(root / 'baseline.conf'), 'unused.lst', '--gpu', str(gpu),
            '--optimizer', 'AdaBelief', '--lr', '.001', '--l2-weight', '.01',
            '--score-loss-weight', '.125', '--loss-func', 'fy', '--perturb', '.75',
            '--shape-perturb', '.1', '--shape-nu', '.1', '--shape-imle-difference', 'central',
            '--shape-rank-tau', '.2', '--shape-rank-delta', '.002',
            '--shape-rank-teacher-temperature', '.03', '--shape-rank-teacher-normalization', 'percentile']
    if arm in ('mw_rank', 'mw_rank_restored'):
        argv += ['--shape-rank-objective', 'mann_whitney', '--shape-margin', '0',
                 '--shape-rank-weighting', 'uniform', '--shape-rank-delta', '0',
                 '--shape-rank-targets', 'raw', '--shape-rank-teacher-temperature', '0',
                 '--shape-rank-teacher-normalization', 'none']
    args = parser.parse_args(argv)
    trainer = Train()
    model, _ = trainer.build_model(args)
    # Native LSTM dropout uses the saved Torch RNG; cuDNN keeps opaque state.
    class NativeLSTM(torch.nn.LSTM):
        def forward(self, *inputs, **kwargs):
            with torch.backends.cudnn.flags(enabled=False):
                return super().forward(*inputs, **kwargs)
    for module in model.modules():
        if isinstance(module, torch.nn.LSTM):
            module.__class__ = NativeLSTM
    device, _ = trainer.get_device(gpu)
    model.to(device)
    if settings['freeze_turner'] and hasattr(model, 'turner'):
        for parameter in model.turner.parameters():
            parameter.requires_grad_(False)
    shape_models = None
    if arm in ('nll', 'nll_restored'):
        # Separate distributions for DMS/SHAPE and for each experimental source.
        shape_models = [trainer.build_shape_model(args).to(device) for _ in range(6)]
    optimizer = trainer.build_optimizer('AdaBelief', model, settings['lr'],
                                        settings['weight_decay'], shape_model=shape_models)
    structural = trainer.build_loss_function('fy', model, args)
    chemical = None
    if arm in RESTORED_ARMS:
        from scripts.efold_hf.matched_losses import (
            FullRNAGymRankLoss, FullRNAGymNLLLoss, FullRNAgymShapeFYLoss, ReplayFYLoss)
        structural = ReplayFYLoss(model)
        if arm == 'mw_rank_restored':
            chemical = FullRNAGymRankLoss(model)
        elif arm == 'nll_restored':
            distributions = torch.nn.ModuleDict({source+':'+probe: shape_models[2*i+j]
                for i,source in enumerate(('ribo500-blast','pri_miRNA','human_mRNA'))
                for j,probe in enumerate(('dms','shape'))})
            chemical = FullRNAGymNLLLoss(model, distributions)
        else:
            chemical = FullRNAgymShapeFYLoss(model)
    elif arm in ('rank', 'mw_rank', 'nll', 'shape_fy'):
        args.shape_imle_normalization = settings['normalization']
        loss_name = {'rank': 'shape_rank', 'mw_rank': 'shape_rank', 'mw_rank_restored': 'shape_rank',
                     'nll': 'shape_nll', 'nll_restored': 'shape_nll', 'shape_fy': 'shape_fy'}[arm]
        chemical = trainer.build_shape_loss_function(loss_name, model, args, shape_models)
    return model, optimizer, structural, chemical, shape_models, device


def seed_all(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(4)


def save_checkpoint(path, model, optimizer, shapes, state):
    import numpy as np
    import torch
    payload = {'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
               'state': state, 'shape_model_state_dict': [m.state_dict() for m in shapes] if shapes else None,
               'rng': {'python': random.getstate(), 'numpy': np.random.get_state(),
                       'torch': torch.get_rng_state(),
                       'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}}
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.tmp')
    torch.save(payload, temp)
    temp.replace(path)


def restore(path, model, optimizer=None, shapes=None):
    import numpy as np
    import torch
    payload = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(payload['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(payload['optimizer_state_dict'])
        if shapes:
            for m, saved in zip(shapes, payload['shape_model_state_dict'], strict=True):
                m.load_state_dict(saved)
        rng = payload['rng']
        random.setstate(rng['python'])
        np.random.set_state(rng['numpy'])
        torch.set_rng_state(rng['torch'])
        if rng['cuda'] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng['cuda'])
    return payload['state']


def initialize_finetuning(root, arm, model, optimizer, shapes, audit_path=None):
    """Restore named network AdaBelief moments; new Wu parameters start fresh."""
    import torch
    initial = pretrain_checkpoint(root)
    expected_hash = plan_for(root).get('pretrain_checkpoint_sha256')
    if expected_hash and digest(initial) != expected_hash:
        raise RuntimeError('Pretraining checkpoint changed')
    payload = torch.load(initial, map_location='cpu', weights_only=False)
    model.load_state_dict(payload['model_state_dict'])
    settings = training_settings(root, arm)
    if not settings['restore_optimizer']:
        return
    source = payload['optimizer_state_dict']
    if len(source['param_groups']) != 1:
        raise RuntimeError('Expected one pretraining optimizer group')
    source_group = source['param_groups'][0]
    named = dict(model.named_parameters())
    if list(named) != list(payload['model_state_dict']):
        raise RuntimeError('Pretraining parameter names/order do not match')
    if len(named) != len(source_group['params']):
        raise RuntimeError('Pretraining optimizer parameter mapping is not one-to-one')
    source_ids = dict(zip(named, source_group['params']))
    current = optimizer.state_dict()
    python_names = {id(parameter): name for name, parameter in named.items()}
    restored, fresh_network = [], []
    for parameter, target_id in zip(optimizer.param_groups[0]['params'], current['param_groups'][0]['params'], strict=True):
        name = python_names[id(parameter)]
        saved = source['state'].get(source_ids[name])
        if saved is None:
            if not name.startswith('zuker.net.shape_head_'):
                raise RuntimeError('Missing network optimizer moments: '+name)
            current['state'][target_id] = {'exp_avg': torch.zeros_like(parameter),
                                           'exp_avg_var': torch.zeros_like(parameter)}
            fresh_network.append(name)
            continue
        for value in saved.values():
            if torch.is_tensor(value) and value.ndim and value.shape != parameter.shape:
                raise RuntimeError('Optimizer moment shape mismatch: '+name)
        current['state'][target_id] = copy.deepcopy(saved)
        restored.append(name)
    migrated = copy.deepcopy(source_group)
    migrated.update(params=current['param_groups'][0]['params'], lr=settings['lr'],
                    weight_decay=settings['weight_decay'])
    current['param_groups'][0] = migrated
    for group, serialized in zip(optimizer.param_groups[1:], current['param_groups'][1:], strict=True):
        serialized['step'] = 0
        for parameter, target_id in zip(group['params'], serialized['params'], strict=True):
            current['state'][target_id] = {'exp_avg': torch.zeros_like(parameter),
                                         'exp_avg_var': torch.zeros_like(parameter)}
    optimizer.load_state_dict(current)
    if optimizer.param_groups[0].get('step') != source_group.get('step'):
        raise RuntimeError('Network optimizer step was not restored')
    for name in restored:
        for key, value in source['state'][source_ids[name]].items():
            if torch.is_tensor(value) and not torch.equal(value, optimizer.state[named[name]][key].cpu()):
                raise RuntimeError('Optimizer moment differs after restoration: '+name+'/'+key)
    audit = {'checkpoint': str(initial.resolve()), 'checkpoint_sha256': digest(initial),
             'source_step': source_group['step'], 'target_step': optimizer.param_groups[0]['step'],
             'restored_network_states': len(restored), 'fresh_unused_network_parameters': fresh_network,
             'fresh_Wu_parameters': sum(len(g['params']) for g in optimizer.param_groups[1:]),
             'settings': settings, 'mapping': 'exact name/order and tensor equality checks'}
    write_json(audit_path or root/'runs'/arm/'initialization.json', audit)


def sequence_loss(item, structural, chemical, arm, device):
    import torch
    sequence = item['sequence']
    if chemical is None:
        target = torch.zeros(len(sequence)+1, dtype=torch.long)
        for i, j in item['train_pairs']:
            target[i+1], target[j+1] = j+1, i+1
        return structural([sequence], [target], fname=[item['id']]).mean()
    if arm in RESTORED_ARMS:
        observations = []
        for obs in item['observations']:
            values = torch.tensor(obs['values'], device=device)
            modifier = (item['source']+':'+obs['probe'] if arm == 'nll_restored'
                        else 'DMS' if obs['probe'] == 'dms' else 'SHAPE')
            observations.append({'values': values, 'mask': torch.isfinite(values) & (values > -10),
                                 'modifier': modifier, 'condition_id': 0})
        return chemical(sequence, observations)
    values = []
    for obs in item['observations']:
        dataset_id = 2*('ribo500-blast', 'pri_miRNA', 'human_mRNA').index(item['source']) + (obs['probe'] == 'shape')
        target = torch.tensor([-999.]+obs['values'], device=device)
        values.append(chemical([sequence], [target], fname=[item['id']], dataset_id=dataset_id).mean())
    return torch.stack(values).mean()


def optimizer_step(optimizer):
    import torch
    for group in optimizer.param_groups:
        for p in group['params']:
            if p.grad is not None:
                if not torch.isfinite(p.grad).all():
                    raise FloatingPointError('Nonfinite gradient')
                # Same lazy-state compatibility fix as the project trainer.
                if len(optimizer.state[p]) == 0:
                    optimizer.state[p]['exp_avg'] = torch.zeros_like(p)
                    optimizer.state[p]['exp_avg_var'] = torch.zeros_like(p)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def train_worker(root, arm, gpu):
    import torch
    plan = verify(root)
    settings = training_settings(root, arm)
    seed_all(settings['seed'])
    model, optimizer, structural, chemical, shapes, device = build(root, arm, gpu)
    bprna = read_jsonl(root / 'data/bprna.jsonl')
    primary = bprna if arm == 'pretrain' else read_jsonl(root / 'data' / ('assisted.jsonl' if arm == 'assisted' else 'chemical.jsonl'))
    run = root / 'runs' / arm
    latest = run / 'latest.pt'
    total_epochs = int(plan['pretrain_epochs'] if arm == 'pretrain' else plan['finetune_epochs'])
    if latest.exists():
        state = restore(latest, model, optimizer, shapes)
    else:
        if arm != 'pretrain':
            initialize_finetuning(root, arm, model, optimizer, shapes)
            seed_all(settings['seed'])
        state = {'epoch': 1, 'cursor': 0, 'updates': 0, 'loss_sum': 0.}
        save_checkpoint(latest, model, optimizer, shapes, state)
    model.train()
    if shapes:
        for m in shapes:
            m.train()
    while state['epoch'] <= total_epochs:
        epoch = state['epoch']
        order = schedule(len(primary), len(bprna), epoch, seed=settings['seed'],
                         replay=arm != 'pretrain', replay_mode=settings['replay_mode'])
        if chemical is not None and hasattr(chemical, 'set_epoch_info'):
            chemical.set_epoch_info(epoch, int(plan['finetune_epochs']))
        while state['cursor'] < len(order):
            batch = order[state['cursor']:state['cursor']+int(settings['accumulation'])]
            optimizer.zero_grad(set_to_none=True)
            for kind, index in batch:
                item = (bprna if kind == 'replay' else primary)[index]
                loss = sequence_loss(item, structural, chemical if kind == 'primary' else None, arm, device)
                loss = loss * (settings['chemical_weight'] if kind == 'primary' and arm != 'pretrain' else 1.)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite loss')
                (loss/len(batch)).backward()
                state['loss_sum'] += float(loss.detach())
            optimizer_step(optimizer)
            state['cursor'] += len(batch)
            state['updates'] += 1
            if state['cursor'] % 100 == 0:
                save_checkpoint(latest, model, optimizer, shapes, state)
                write_json(run / 'progress.json', {**state, 'total_primary': len(primary),
                           'total_slots': len(order), 'latest_checkpoint': str(latest)})
                print(arm, state, flush=True)
        metrics = {**state, 'success': True, 'primary_examples': len(primary),
                   'replay_examples': len(order)-len(primary), 'mean_loss': state['loss_sum']/len(order)}
        # Commit completed-epoch snapshot and marker before advancing latest.
        # An interruption anywhere here can replay the commit without retraining.
        save_checkpoint(latest, model, optimizer, shapes, state)
        if arm != 'pretrain' or epoch == total_epochs:
            checkpoint = run / f'epoch-{epoch:03d}.pt'
            if not checkpoint.exists():
                os.link(latest, checkpoint)
            metrics['checkpoint_sha256'] = digest(checkpoint)
        write_json(run / f'epoch-{epoch:03d}-complete.json', metrics)
        state = {**state, 'epoch': epoch+1, 'cursor': 0, 'loss_sum': 0.}
        save_checkpoint(latest, model, optimizer, shapes, state)
    write_json(run / 'complete.json', {'success': True, 'epochs': total_epochs, 'updates': state['updates']})
