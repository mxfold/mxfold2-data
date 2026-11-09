"""Frozen chemical objectives and sample scheduling from the RNAgym experiments."""
from collections import defaultdict
import copy
import math

import numpy as np
import torch

from mxfold2 import param_turner2004
from mxfold2.fold.rnafold import RNAFold
from mxfold2.loss.fy_loss import FenchelYoungLoss
from mxfold2.loss.shape_fy_loss import ShapeFenchelYoungLoss
from mxfold2.loss.implicit_mle import estimate_implicit_mle
from mxfold2.loss.nonparametric_rank import mann_whitney_rank_loss
from mxfold2.rnagym import coverage_order
from scripts.efold_hf.workflow import write_json

BASELINE_OPT_STEP = 316600
LR = WEIGHT_DECAY = .0001
THERMO_WEIGHT = .125


def restore_baseline_optimizer(out, model, optimizer):
    checkpoint = torch.load(out / 'initial_baseline.pth', map_location='cpu', weights_only=False)
    if checkpoint.get('epoch') != 100 or 'optimizer_state_dict' not in checkpoint:
        raise RuntimeError('TrainSetA baseline checkpoint lacks epoch-100 AdaBelief state')
    source_model = checkpoint['model_state_dict']
    source_opt = checkpoint['optimizer_state_dict']
    if len(source_opt['param_groups']) != 1:
        raise RuntimeError('Unexpected source optimizer group layout')
    source_group = source_opt['param_groups'][0]
    if (source_group.get('step') != BASELINE_OPT_STEP
            or tuple(source_group.get('betas', ())) != (0.9, 0.999)
            or not math.isclose(float(source_group.get('eps')), 1e-16, rel_tol=1e-10)
            or not source_group.get('weight_decouple', False)):
        raise RuntimeError('Unexpected baseline AdaBelief settings')

    named = list(model.named_parameters())
    source_names = [name for name, _ in named if name in source_model]
    if list(source_model) != source_names:
        raise RuntimeError('Baseline model parameter ordering changed')
    source_ids = dict(zip(source_names, source_group['params']))
    if len(source_ids) != len(source_group['params']):
        raise RuntimeError('Baseline optimizer parameter mapping is not one-to-one')
    for name, parameter in named:
        if name in source_model and not torch.equal(parameter.detach().cpu(), source_model[name]):
            raise RuntimeError('Model value differs from the optimizer source: ' + name)

    current = optimizer.state_dict()
    python_names = {id(parameter): name for name, parameter in named}
    target_name_by_id = {}
    for group, serialized in zip(optimizer.param_groups, current['param_groups']):
        if len(group['params']) != len(serialized['params']):
            raise RuntimeError('Target optimizer parameter mapping changed')
        for parameter, parameter_id in zip(group['params'], serialized['params']):
            target_name_by_id[parameter_id] = python_names[id(parameter)]
    restored = []
    for target_id, name in target_name_by_id.items():
        source_id = source_ids.get(name)
        if source_id is None or source_id not in source_opt['state']:
            continue
        source_state = source_opt['state'][source_id]
        parameter = dict(named)[name]
        for key, value in source_state.items():
            if torch.is_tensor(value) and value.ndim and value.shape != parameter.shape:
                raise RuntimeError(f'Optimizer state shape mismatch: {name}/{key}')
        current['state'][target_id] = copy.deepcopy(source_state)
        restored.append(name)
    target_ids = current['param_groups'][0]['params']
    migrated = copy.deepcopy(source_group)
    migrated['params'] = target_ids
    migrated['lr'], migrated['weight_decay'] = LR, WEIGHT_DECAY
    current['param_groups'] = [migrated]
    optimizer.load_state_dict(current)
    if optimizer.param_groups[0].get('step') != BASELINE_OPT_STEP:
        raise RuntimeError('AdaBelief global step was not restored')
    missing = [name for name in target_name_by_id.values() if name not in restored]
    if any(not name.startswith('zuker.net.shape_head_') for name in missing):
        raise RuntimeError('Unexpected trainable parameters lack optimizer moments: ' + str(missing))
    for name in restored:
        source_id = source_ids[name]
        for key, value in source_opt['state'][source_id].items():
            if torch.is_tensor(value) and not torch.equal(
                    optimizer.state[dict(named)[name]][key].detach().cpu(), value):
                raise RuntimeError(f'AdaBelief moment restore failed: {name}/{key}')
    audit = {'source_step': BASELINE_OPT_STEP, 'target_step': BASELINE_OPT_STEP,
             'source_parameters': len(source_names), 'target_trainable_parameters': len(target_name_by_id),
             'restored_states': len(restored), 'new_parameters_without_source_state': missing,
             'target_lr': LR, 'target_weight_decay': WEIGHT_DECAY,
             'betas': list(migrated['betas']), 'eps': migrated['eps'],
             'mapping': 'exact parameter-name mapping with exact initial tensor and moment verification'}
    write_json(out / 'optimizer_restore_audit.json', audit)

class FullRNAGymRankLoss(torch.nn.Module):
    """Multi-probe Mann-Whitney IMLE loss plus the existing Turner penalty."""
    def __init__(self, model, thermo_weight=THERMO_WEIGHT):
        super().__init__()
        self.model = model
        self.perturb, self.nu = 0.1, 0.1
        self.thermo_weight = thermo_weight
        self.turner = RNAFold(param_turner2004).to(next(model.parameters()).device)
        self.last = {}

    def rank_objective(self, sequence, paired, observations):
        if paired.numel() == len(sequence) + 1:
            paired = paired[1:]
        if paired.numel() != len(sequence):
            raise ValueError('Pairedness length mismatch')
        by_probe = defaultdict(lambda: defaultdict(list))
        for obs in observations:
            values = obs['values'].to(paired.device)
            mask = obs['mask'].to(paired.device)
            if obs['modifier'] == 'DMS':
                base_groups = ('A', 'C')
            elif obs['modifier'] == 'CMCT':
                base_groups = ('G', 'U')
            else:
                base_groups = (None,)
            groups = []
            for base in base_groups:
                base_mask = mask if base is None else mask & torch.tensor(
                    [nt == base for nt in sequence], dtype=torch.bool, device=paired.device)
                if int(base_mask.sum()) >= 2:
                    groups.append(mann_whitney_rank_loss(values[base_mask], paired[base_mask]))
            profile = torch.stack(groups).mean() if groups else paired.sum() * 0
            by_probe[obs['modifier']][obs.get('condition_id', 0)].append(profile)
        if not by_probe:
            return paired.sum() * 0
        # Replicates -> condition means -> equal probe means; adding a replicate
        # cannot multiply a sequence's weight.
        probe_means = []
        for conditions in by_probe.values():
            condition_means = [torch.stack(profiles).mean() for profiles in conditions.values()]
            probe_means.append(torch.stack(condition_means).mean())
        return torch.stack(probe_means).mean()

    def forward(self, sequence, observations):
        objective = lambda paired: self.rank_objective(sequence, paired[0], observations).reshape(1)
        estimate = estimate_implicit_mle(
            self.model, [sequence], objective, perturb=self.perturb, nu=self.nu,
            alignment='aligned', normalization='rms', estimator_scale='divide_nu',
            difference_mode='central', num_samples=1)
        raw = estimate.loss.mean()
        with torch.no_grad():
            turner_score, _, _ = self.turner([sequence], constraint=estimate.target_structures,
                                              max_internal_length=None)
        thermo = self.thermo_weight * (estimate.target_score - turner_score).square() / len(sequence)
        loss = raw + thermo.mean()
        self.last = {'rank_raw': float(raw.detach()), 'thermo': float(thermo.mean().detach()),
                     'flip_fraction': float(np.mean(estimate.flip_fraction))}
        return loss

class ReplayFYLoss(FenchelYoungLoss):
    def __init__(self, model):
        super().__init__(model, perturb=0.75, sl_weight=THERMO_WEIGHT)
        self.last = {}

    def forward(self, seq, pairs, fname=None):
        if self.l1_weight:
            raise ValueError('Unexpected L1 regularization in replay FY')
        pred, _, _, _, parameters = self.model(seq, return_param=True, perturb=self.perturb)
        ref, _, _ = self.model(seq, param=parameters, constraint=pairs, max_internal_length=None)
        lengths = torch.tensor([len(s) for s in seq], device=pred.device)
        fy = ((pred - ref) / lengths).sum()
        if self.sl_weight > 0:
            with torch.no_grad():
                ref_turner, _, _ = self.turner(seq, constraint=pairs, max_internal_length=None)
            thermo = self.sl_weight * (ref - ref_turner).square() / lengths
        else:
            thermo = torch.zeros_like(fy)
        self.last = {'fy_raw': float(fy.detach()), 'thermo': float(thermo.sum().detach())}
        return fy + thermo.sum()

def make_replay_stream(replay_length, start, count, seed):
    values = []
    cached_cycle, order = None, None
    for position in range(start, start + count):
        cycle, cursor = divmod(position, replay_length)
        if cycle != cached_cycle:
            order = coverage_order(np.arange(replay_length), seed + 107, cycle)
            cached_cycle = cycle
        values.append(int(order[cursor]))
    return values

def epoch_slots(chemical_indices, replay_length, epoch, replay_per_epoch, seed):
    chemical_order = coverage_order(np.arange(len(chemical_indices)), seed, epoch - 1)
    chemical = [('chemical', int(chemical_indices[i])) for i in chemical_order]
    replay = [('replay', i) for i in make_replay_stream(
        replay_length, (epoch - 1) * replay_per_epoch, replay_per_epoch, seed)]
    slots = chemical + replay
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, 9163]))
    return [slots[int(i)] for i in rng.permutation(len(slots))]

class FullRNAGymNLLLoss(torch.nn.Module):
    """Multi-probe Wu-NLL IMLE loss plus the frozen Turner penalty."""
    def __init__(self, model, shape_models, thermo_weight=THERMO_WEIGHT):
        super().__init__()
        self.model = model
        self.shape_models = shape_models
        self.perturb, self.nu = 0.1, 0.1
        self.thermo_weight = thermo_weight
        self.turner = RNAFold(param_turner2004).to(next(model.parameters()).device)
        self.last = {}

    def nll_objective(self, sequence, paired, observations):
        if paired.numel() == len(sequence) + 1:
            paired = paired[1:]
        if paired.numel() != len(sequence):
            raise ValueError('Pairedness length mismatch')
        by_probe = defaultdict(lambda: defaultdict(list))
        for obs in observations:
            values = obs['values'].to(paired.device)
            mask = obs['mask'].to(paired.device)
            target = values.clone()
            target[~mask] = -999.0
            profile = self.shape_models[obs['modifier']]([sequence], [paired], [target])[0]
            by_probe[obs['modifier']][obs.get('condition_id', 0)].append(profile)
        if not by_probe:
            return paired.sum() * 0
        # Replicates -> condition means -> equal probe means. Each modifier has
        # its own Wu distribution so assay scales do not share nuisance params.
        probe_means = []
        for conditions in by_probe.values():
            condition_means = [torch.stack(profiles).mean() for profiles in conditions.values()]
            probe_means.append(torch.stack(condition_means).mean())
        return torch.stack(probe_means).mean()

    def forward(self, sequence, observations):
        objective = lambda paired: self.nll_objective(sequence, paired[0], observations).reshape(1)
        estimate = estimate_implicit_mle(
            self.model, [sequence], objective, perturb=self.perturb, nu=self.nu,
            alignment='aligned', normalization='rms', estimator_scale='divide_nu',
            difference_mode='central', num_samples=1, propagate_value_gradient=True)
        raw = estimate.loss.mean()
        with torch.no_grad():
            turner_score, _, _ = self.turner([sequence], constraint=estimate.target_structures,
                                              max_internal_length=None)
        thermo = self.thermo_weight * (estimate.target_score - turner_score).square() / len(sequence)
        loss = raw + thermo.mean()
        self.last = {'nll_raw': float(raw.detach()), 'thermo': float(thermo.mean().detach()),
                     'flip_fraction': float(np.mean(estimate.flip_fraction))}
        return loss

class FullRNAgymShapeFYLoss(torch.nn.Module):
    """Probe/condition-balanced SHAPE-FY, including its Turner score penalty."""
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.base = ShapeFenchelYoungLoss(
            model, perturb=0.1, shape_slope=2.6, shape_intercept=-0.8,
            sl_weight=THERMO_WEIGHT, weight_schedule='none')
        self.last = {}

    def profile_components(self, sequence, target):
        # Match ShapeFenchelYoungLoss.forward: observations not measured for a
        # modifier are represented as missing, so they create no pseudoenergy.
        if target.numel() == len(sequence) + 1:
            target = target[1:]
        if target.numel() != len(sequence):
            raise ValueError('Chemical target length mismatch')
        pred, _, _, _, parameters = self.model(
            [sequence], return_param=True, perturb=self.base.perturb)
        pseudoenergy = self.base.calc_pseudoenergy(target)
        _, _, ref_structure = self.model(
            [sequence], param=parameters, pseudoenergy=[pseudoenergy])
        ref, _, ref_structure = self.model(
            [sequence], param=parameters, constraint=ref_structure)
        lengths = torch.tensor([len(sequence)], device=pred.device)
        fy = ((pred - ref) / lengths).sum()
        with torch.no_grad():
            ref_turner, _, _ = self.base.turner([sequence], constraint=ref_structure)
        thermo = (self.base.sl_weight * (ref - ref_turner).square() / lengths).sum()
        return fy, thermo

    def forward(self, sequence, observations):
        by_probe = defaultdict(lambda: defaultdict(list))
        valid_bases = 0
        for obs in observations:
            target = obs['values'].to(next(self.model.parameters()).device).clone()
            mask = obs['mask'].to(target.device)
            valid_bases += int(mask.sum())
            target[~mask] = -999.0
            raw, thermo = self.profile_components(sequence, target)
            by_probe[obs['modifier']][obs.get('condition_id', 0)].append((raw, thermo))
        if not by_probe:
            raise ValueError('RNAgym training sequence has no chemical profiles')
        probe_raw, probe_thermo = [], []
        for conditions in by_probe.values():
            condition_raw, condition_thermo = [], []
            for profiles in conditions.values():
                condition_raw.append(torch.stack([x[0] for x in profiles]).mean())
                condition_thermo.append(torch.stack([x[1] for x in profiles]).mean())
            probe_raw.append(torch.stack(condition_raw).mean())
            probe_thermo.append(torch.stack(condition_thermo).mean())
        raw = torch.stack(probe_raw).mean()
        thermo = torch.stack(probe_thermo).mean()
        self.last = {'shape_fy_raw': float(raw.detach()), 'thermo': float(thermo.detach()),
                     'profiles': len(observations), 'valid_bases': valid_bases}
        return raw + thermo
