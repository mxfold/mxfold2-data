#!/usr/bin/env python3
"""Continue each condition1/3 Ribonanza FT arm on pri-miRNA and human mRNA."""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import time

PROJECT = Path('/gs/bs/tga-satolab-gtex/satoken/repos/mxfold2-ryntaro')
CONDITION = int(os.environ.get('FT_CONDITION', '3'))
if CONDITION not in (1,2,3,4,5):
    raise ValueError('Unsupported pretraining condition')
PRETRAIN_EPOCH = int(os.environ.get('FT_PRETRAIN_EPOCH',str(10 if CONDITION==2 else 20)))
RUN_TAG = f'condition{CONDITION}' + (f'_pt{PRETRAIN_EPOCH:03d}' if 'FT_PRETRAIN_EPOCH' in os.environ else '')
RUN_DATE = '20261103' if CONDITION==5 else '20261101'
ROOT = Path(os.environ.get('CURRICULUM_ROOT', str(PROJECT /
    f'qsub_logs/efold_hf/{RUN_TAG}_pri_mrna3_after_ribonanza3_replay20_{RUN_DATE}')))
STAGE1 = Path(os.environ.get('FT_STAGE1_ROOT',str(PROJECT/f'qsub_logs/efold_hf/{RUN_TAG}_ribonanza3_replay20_{RUN_DATE}')))
REFERENCE = PROJECT/'qsub_logs/efold_hf/bprna10_restored_ft5_replay20_20261026'
ARMS = ('mw_rank_restored', 'nll_restored', 'shape_fy_restored')
QSUB = '/apps/t4/rhel9/uge/latest/bin/lx-amd64/qsub'
SOFT_HOURS = 22
HARD_SECONDS = 85800
PARENT_UPDATES = 42873
REPLAY_START = 34296
EPOCH_OFFSET = 3


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_shared_path = ROOT/'stage1_helper.py'
_bootstrap_helper = STAGE1/'controller.py'
if not _bootstrap_helper.exists():
    _bootstrap_helper = PROJECT/'scripts/efold_hf/condition3_ribonanza_ft.py'
_shared = load_module('condition3_ft_shared', _shared_path if _shared_path.exists()
                      else _bootstrap_helper)
if getattr(_shared,'CONDITION',3) != CONDITION:
    raise RuntimeError('Wrong condition in frozen first-stage helper')
if getattr(_shared,'PRETRAIN_EPOCH',20) != PRETRAIN_EPOCH:
    raise RuntimeError('Wrong pretraining epoch in first-stage helper')
REPLAY_POOL_COUNT = getattr(_shared,'REPLAY_POOL_COUNT',
    {1:45507,2:255767,3:14831,4:60065}.get(CONDITION))
if REPLAY_POOL_COUNT is None:
    raise RuntimeError('Condition5 requires its frozen replay pool size')
PRETRAIN_NETWORK_STEP = REPLAY_POOL_COUNT*PRETRAIN_EPOCH
PARENT_NETWORK_STEP = PRETRAIN_NETWORK_STEP+PARENT_UPDATES
sha, write_json, rows, write_rows, lock = (
    _shared.sha, _shared.write_json, _shared.rows, _shared.write_rows, _shared.lock)


def shared():
    _shared.ROOT = ROOT
    _shared.submit_once = submit_once
    return _shared


def verify():
    plan = shared().verify()
    marker = ROOT/'prepare-complete.json'
    if marker.exists() and sha(ROOT/'plan.json') != json.loads(marker.read_text())['plan_sha256']:
        raise RuntimeError('Frozen stage2 plan changed')
    return plan


def assert_same(x, y, *, approximate=False):
    """Compare nested checkpoints, including optimizer groups and all RNGs."""
    import numpy as np
    import torch
    if torch.is_tensor(x):
        torch.testing.assert_close(x, y, rtol=1e-6 if approximate else 0,
                                   atol=1e-7 if approximate else 0, equal_nan=True)
    elif isinstance(x, np.ndarray):
        if not np.array_equal(x, y):
            raise AssertionError('NumPy RNG differs')
    elif isinstance(x, dict):
        if x.keys() != y.keys():
            raise AssertionError('Checkpoint keys differ')
        for key in x:
            assert_same(x[key], y[key], approximate=approximate)
    elif isinstance(x, (list, tuple)):
        if type(x) is not type(y) or len(x) != len(y):
            raise AssertionError('Checkpoint sequence differs')
        for a, b in zip(x, y, strict=True):
            assert_same(a, b, approximate=approximate)
    elif x != y:
        raise AssertionError('Checkpoint value differs')


def initial_payload(parent):
    """Only the stage-local counters restart; learned state and RNG stay intact."""
    return {**parent, 'state': {'epoch': 1, 'cursor': 0, 'updates': 0, 'loss_sum': 0.}}


def save_payload(path, payload):
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.'+path.name+f'.{os.getpid()}.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def audit_restore(arm, payload, directory, gpu=-1):
    """Exercise the actual frozen build/restore/save without any model update."""
    import torch
    directory.mkdir(parents=True, exist_ok=True)
    source = directory/'input.pt'
    output = directory/'restored.pt'
    save_payload(source, payload)
    wf = shared().workflow()
    wf.seed_all(0)
    model, optimizer, _, _, shapes, _ = wf.build(ROOT, arm, gpu)
    state = wf.restore(source, model, optimizer, shapes)
    wf.save_checkpoint(output, model, optimizer, shapes, state)
    restored = torch.load(output, map_location='cpu', weights_only=False)
    for key in ('model_state_dict', 'optimizer_state_dict', 'shape_model_state_dict', 'state'):
        assert_same(payload[key], restored[key])
    for key in ('python', 'numpy', 'torch'):
        assert_same(payload['rng'][key], restored['rng'][key])
    if gpu >= 0:
        assert_same(payload['rng']['cuda'], restored['rng']['cuda'])
    result = {'success': True, 'gpu': gpu,
              'model_optimizer_Wu_exact': True, 'CPU_rng_exact': True,
              'CUDA_rng_exact': True if gpu >= 0 else 'preserved in payload; GPU audit pending',
              'network_optimizer_step': payload['optimizer_state_dict']['param_groups'][0]['step'],
              'state': state, 'input_sha256': sha(source), 'roundtrip_sha256': sha(output)}
    write_json(directory/'audit.json', result)
    return result


def prepare():
    if (ROOT/'prepare-complete.json').exists():
        verify()
        return
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT/'source').exists():
        raise RuntimeError('Inspect incomplete stage2 preparation before reuse')
    parent = json.loads((STAGE1/'plan.json').read_text())
    for path, digest in json.loads((STAGE1/'implementation.json').read_text()).items():
        if sha(path) != digest:
            raise RuntimeError('Stage1 implementation changed: '+path)
    for relative, digest in parent['source_sha256'].items():
        if sha(STAGE1/'source'/relative) != digest:
            raise RuntimeError('Stage1 loss/model source changed: '+relative)
    for relative, digest in parent['data_sha256'].items():
        if sha(STAGE1/'data'/relative) != digest:
            raise RuntimeError('Stage1 input changed: '+relative)
    shutil.copytree(STAGE1/'source', ROOT/'source', symlinks=True,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for src, name in [(STAGE1/'baseline.conf', 'baseline.conf'),
                      (STAGE1/'metrics.py', 'metrics.py'),
                      (STAGE1/'controller.py', 'stage1_helper.py')]:
        shutil.copyfile(src, ROOT/name)
    if Path(__file__).resolve() != (ROOT/'controller.py').resolve():
        shutil.copyfile(__file__, ROOT/'controller.py')
    if not (ROOT/'job.sh').exists():
        shutil.copyfile(PROJECT/'scripts/efold_hf/condition3_pri_mrna_ft_job.sh', ROOT/'job.sh')
    data = ROOT/'data'
    data.mkdir()
    source = REFERENCE/'data/chemical.jsonl'
    reference_plan = json.loads((REFERENCE/'plan.json').read_text())
    if sha(source) != reference_plan['data_sha256']['chemical.jsonl']:
        raise RuntimeError('Frozen chemical source changed')
    primary = [r for r in rows(source) if r['source'] in ('pri_miRNA', 'human_mRNA')]
    replay = rows(STAGE1/'data/bprna.jsonl')
    ribo = rows(STAGE1/'data/chemical.jsonl')
    tests = rows(STAGE1/'data/test.jsonl')
    counts = dict(Counter(r['source'] for r in primary))
    if counts != {'pri_miRNA': 1096, 'human_mRNA': 1375} or len(replay) != REPLAY_POOL_COUNT:
        raise RuntimeError('Unexpected stage2 training sizes')
    sequences = {r['sequence'] for r in primary}
    if any(not 1 <= len(r['sequence']) <= 1000 for r in primary+replay):
        raise RuntimeError('Training length exceeds1000')
    if sequences & {r['sequence'] for r in tests+ribo}:
        raise RuntimeError('Unexpected stage2 overlap with test or Ribonanza')
    shared_sequences = sequences & {r['sequence'] for r in replay}
    if shared_sequences and CONDITION in (1,3):
        raise RuntimeError('Unexpected stage2 overlap with replay')
    write_json(ROOT/'primary_replay_overlap.json', {'n':len(shared_sequences),
        'primary_ids':[r['id'] for r in primary if r['sequence'] in shared_sequences],
        'policy':'Use the complete pretraining set for replay; within-training overlap is recorded'})
    if len({r['id'] for r in primary}) != len(primary) or not all(r['observations'] for r in primary):
        raise RuntimeError('Invalid primary IDs or observations')
    write_rows(data/'chemical.jsonl', primary)
    shutil.copyfile(STAGE1/'data/bprna.jsonl', data/'bprna.jsonl')
    shutil.copyfile(STAGE1/'data/test.jsonl', data/'test.jsonl')
    write_rows(data/'assisted.jsonl', [])
    all_training = {r['sequence'] for r in primary+replay+ribo}
    excluded = [r for r in tests if r['sequence'] in all_training]
    write_json(ROOT/'test_exact_overlap.json', {'ids': [r['id'] for r in excluded],
        'n': len(excluded), 'by_source': dict(Counter(r['source'] for r in excluded))})
    for name in ('scheduler', 'dispatch', 'validation'):
        (ROOT/name).mkdir(exist_ok=True)
    for arm in ARMS:
        (ROOT/'runs'/arm).mkdir(parents=True)
    n_replay = int(math.floor(len(primary)/4+.5))
    slots = len(primary)+n_replay
    plan = {**parent, 'root': str(ROOT), 'parent_experiment': str(STAGE1),
        'stage': 'pri-miRNA plus human mRNA after matching Ribonanza arm epoch3',
        'finetune_epochs': 3, 'global_ft_epoch_offset': EPOCH_OFFSET,
        'cumulative_ft_epochs': 6, 'replay_stream_start': REPLAY_START,
        'counts': {'primary': len(primary), 'by_source': counts, 'replay_pool': len(replay),
                   'primary_per_epoch': len(primary), 'replay_per_epoch': n_replay,
                   'slots_per_epoch': slots, 'optimizer_updates_per_epoch': math.ceil(slots/4)},
        'epoch_definition': 'Each pri-miRNA/human mRNA sequence once, plus618 cyclic replay draws; 3089slots/epoch, 773optimizer updates/epoch',
        'initialization': f'Restore matching Ribonanza epoch3 model, full AdaBelief groups/moments/step{PARENT_NETWORK_STEP}, all6 Wu distributions and all RNG; reset only stage-local counters',
        'optimizer_initialization': 'Full same-arm FT optimizer restoration; no pretraining migration and no parameter reinitialization',
        'replay_sampling': f'Continue the same shuffled condition{CONDITION} replay stream after34296 preceding draws; global FT epochs4..6 for schedule shuffling',
        'data_sha256': {p.name: sha(p) for p in data.glob('*.jsonl')},
        'config_sha256': sha(ROOT/'baseline.conf'), 'job_sha256': sha(ROOT/'job.sh'),
        'parent_inputs_sha256': {str(p): sha(p) for p in
            (STAGE1/'plan.json', STAGE1/'implementation.json', source)},
        'filtering': 'Frozen eFold train chemical data, pri_miRNA/human_mRNA only; lengths<=1000; zero exact test or Ribonanza overlap; replay overlap audited',
        'test_limitations': 'Full published PDB355/ArchiveII3370/viral40/lncRNA30; pretraining overlaps are reported and excluded in parallel; no new homology exclusion',
        'next_stage': 'None; this completes the requested3+3 FT epochs',
        'scheduler_hard_seconds': HARD_SECONDS,
        'user_clarification': '2026-11-01: 後半3エポックはpri-miRNA＋ヒトmRNAです。'}
    plan.pop('ribonanza_structure_provenance', None)
    write_json(ROOT/'plan.json', plan)
    files = [ROOT/n for n in ('controller.py', 'job.sh', 'stage1_helper.py', 'metrics.py', 'baseline.conf')]
    write_json(ROOT/'implementation.json', {str(p): sha(p) for p in files})
    verify()
    # Validate the real FT optimizer layout now; production initialization still
    # strictly waits for epoch3, even if only epoch1/2 is available at preparation.
    import torch
    early = {}
    for arm in ARMS:
        candidates = sorted((STAGE1/'runs'/arm).glob('epoch-*-complete.json'))
        if not candidates:
            continue
        marker = json.loads(candidates[-1].read_text())
        checkpoint = STAGE1/'runs'/arm/f"epoch-{marker['epoch']:03d}.pt"
        if sha(checkpoint) != marker['checkpoint_sha256']:
            raise RuntimeError('Early source checkpoint changed')
        payload = initial_payload(torch.load(checkpoint, map_location='cpu', weights_only=False))
        audit = audit_restore(arm, payload, ROOT/'validation'/'early-handoff'/arm)
        early[arm] = {'source': str(checkpoint), 'source_epoch': marker['epoch'],
                      'production_initialization': False, **audit}
    write_json(ROOT/'validation/early_handoff.json', early)
    write_json(ROOT/'prepare-complete.json', {'success': True, 'plan_sha256': sha(ROOT/'plan.json'),
        'primary_n': len(primary), 'replay_n': n_replay, 'slots_per_epoch': slots,
        'early_CPU_audits': list(early), 'production_initialization': 'await matching Ribonanza epoch3'})
    print(json.dumps(plan['counts']), flush=True)


def submit_once(key, action, arm=ARMS[0], epoch=0, hold=None, waited_job=None):
    if arm not in ARMS or action not in ('prepare', 'bridge', 'preflight', 'work', 'evaluate'):
        raise ValueError((arm, action))
    marker = ROOT/'dispatch'/(key+'.json')
    with lock(ROOT/'submission.lock'):
        if marker.exists():
            return json.loads(marker.read_text())['job_id']
        duration = {'prepare': '02:00:00', 'bridge': '01:00:00', 'preflight': '01:30:00'}.get(action, '23:50:00')
        resource = 'cpu_8=1' if action in ('prepare', 'bridge') else 'gpu_h=1'
        env = f'CURRICULUM_ROOT={ROOT},FT_CONDITION={CONDITION},FT_PRETRAIN_EPOCH={PRETRAIN_EPOCH},FT_STAGE1_ROOT={STAGE1},FT_ACTION={action},FT_ARM={arm},FT_EPOCH={epoch},WAITED_STAGE1_JOB={waited_job or ""}'
        args = [QSUB, '-terse', '-cwd', '-j', 'y', '-A', 'tga-satolab-gtex', '-r', 'y',
                '-N', f'c{CONDITION}p{PRETRAIN_EPOCH}h_'+arm[:13]+'_'+action[:3]+str(epoch), '-l', resource,
                '-l', 'h_rt='+duration, '-l', 'hostname=!r22n6',
                '-o', str(ROOT/'scheduler'), '-v', env]
        if hold:
            args += ['-hold_jid', str(hold)]
        args.append(str(ROOT/'job.sh'))
        command = ['sg', 'tga-satolab-gtex', '-c', shlex.join(args)]
        for attempt in range(3):
            result = subprocess.run(command, cwd=PROJECT, capture_output=True, text=True)
            if result.returncode == 0:
                break
            if attempt == 2:
                raise RuntimeError(result.stderr)
            time.sleep(2**attempt)
        if not re.fullmatch(r'\d+(?:\..+)?', result.stdout.strip()):
            raise RuntimeError('Unexpected qsub output: '+result.stdout)
        job = result.stdout.strip().split('.')[0]
        record = {'job_id': job, 'action': action, 'arm': arm, 'epoch': epoch,
                  'hold_jid': hold, 'waited_stage1_job': waited_job, 'time': time.time(), 'command': args}
        write_json(marker, record)
        with (ROOT/'submissions.jsonl').open('a') as handle:
            handle.write(json.dumps(record)+'\n')
        print('submitted', arm, action, epoch, job, flush=True)
        return job


def latest_parent_work(arm):
    candidates = [r for r in rows(STAGE1/'submissions.jsonl')
                  if r['action'] == 'work' and r['arm'] == arm]
    if not candidates:
        raise RuntimeError('No upstream training job for '+arm)
    return str(candidates[-1]['job_id'])


def parent_completed(arm):
    path = STAGE1/'runs'/arm/'complete.json'
    if not path.exists():
        return False
    data = json.loads(path.read_text())
    if not data.get('success') or data['epochs'] != 3 or data['updates'] != PARENT_UPDATES:
        raise RuntimeError('Unexpected upstream completion state')
    return True


def handoff(arm):
    import torch
    if not parent_completed(arm):
        raise RuntimeError('Ribonanza3 has not completed')
    source = STAGE1/'runs'/arm/'epoch-003.pt'
    metadata = json.loads(source.with_name('epoch-003-complete.json').read_text())
    checksum = sha(source)
    if not metadata['success'] or checksum != metadata['checkpoint_sha256']:
        raise RuntimeError('Upstream checkpoint hash mismatch')
    payload = torch.load(source, map_location='cpu', weights_only=False)
    if (payload['state']['epoch'] != 3 or payload['state']['cursor'] != 57161 or
            payload['state']['updates'] != PARENT_UPDATES or
            payload['optimizer_state_dict']['param_groups'][0]['step'] != PARENT_NETWORK_STEP):
        raise RuntimeError('Upstream optimizer/cursor does not represent Ribonanza3')
    target = ROOT/'runs'/arm
    with lock(target/'handoff.lock', nonblocking=True):
        audit_path = target/'initialization.json'
        if audit_path.exists():
            audit = json.loads(audit_path.read_text())
            if audit['source_sha256'] != checksum or sha(target/'initial.pt') != audit['initial_sha256']:
                raise RuntimeError('Existing handoff changed')
            return audit
        initial = initial_payload(payload)
        if (target/'initial.pt').exists():
            assert_same(initial, torch.load(target/'initial.pt', map_location='cpu', weights_only=False))
        else:
            save_payload(target/'initial.pt', initial)
        if (target/'latest.pt').exists():
            assert_same(initial, torch.load(target/'latest.pt', map_location='cpu', weights_only=False))
        else:
            os.link(target/'initial.pt', target/'latest.pt')
        audit = audit_restore(arm, initial, ROOT/'validation'/'handoff'/arm)
        audit.update(source=str(source), source_sha256=checksum,
                     source_epoch=3, source_updates=PARENT_UPDATES,
                     source_state=payload['state'], initial_sha256=sha(target/'initial.pt'),
                     global_ft_epoch_offset=EPOCH_OFFSET, replay_stream_start=REPLAY_START,
                     initialization='Full same-arm FT restore; only stage-local counters reset')
        write_json(audit_path, audit)
        return audit


def bridge(arm, waited_job):
    verify()
    with lock(ROOT/'runs'/arm/'bridge.lock', nonblocking=True):
        if not parent_completed(arm):
            following = latest_parent_work(arm)
            if not waited_job or following == waited_job:
                write_json(ROOT/'runs'/arm/'bridge-status.json', {'success': False,
                    'reason': 'Upstream job ended without completion or registered continuation',
                    'waited_job': waited_job, 'last_registered_work': following})
                raise RuntimeError('Upstream incomplete; refusing to start stage2 or spin on a finished job')
            job = submit_once(f'bridge-{arm}-after-{following}', 'bridge', arm,
                              hold=following, waited_job=following)
            write_json(ROOT/'runs'/arm/'bridge-status.json', {'waiting_for': following, 'bridge_job': job})
            return
        audit = handoff(arm)
        pf = submit_once('preflight-'+arm, 'preflight', arm, hold=os.environ.get('JOB_ID'))
        work = submit_once('work-initial-'+arm, 'work', arm, hold=pf)
        write_json(ROOT/'runs'/arm/'bridge-status.json', {'success': True,
            'source_sha256': audit['source_sha256'], 'preflight_job': pf, 'work_job': work})


def stage_schedule(primary_count, replay_count, epoch, seed=0, replay=True, replay_mode='cycle'):
    import numpy as np
    if not replay or replay_mode != 'cycle' or not replay_count:
        raise ValueError('Stage2 requires nonempty cyclic replay')
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch+EPOCH_OFFSET, 9823]))
    order = [('primary', int(i)) for i in rng.permutation(primary_count)]
    count = int(math.floor(primary_count/4+.5))
    from scripts.efold_hf.matched_losses import make_replay_stream
    indices = make_replay_stream(replay_count, REPLAY_START+(epoch-1)*count, count, seed)
    order += [('replay', int(i)) for i in indices]
    rng.shuffle(order)
    return order


def prohibit_reinitialization(*args, **kwargs):
    raise RuntimeError('Stage2 must restore its same-arm Ribonanza handoff; initialization is forbidden')


def dispatch(arm):
    previous = None
    for epoch in range(1, 4):
        if (ROOT/'runs'/arm/f'epoch-{epoch:03d}-complete.json').exists():
            previous = submit_once(f'evaluate-{arm}-{epoch}', 'evaluate', arm, epoch, hold=previous)


def train(arm):
    verify()
    audit = json.loads((ROOT/'runs'/arm/'initialization.json').read_text())
    if sha(ROOT/'runs'/arm/'initial.pt') != audit['initial_sha256']:
        raise RuntimeError('Initial handoff changed')
    if not json.loads((ROOT/'validation'/f'preflight-{arm}.json').read_text())['success']:
        raise RuntimeError('GPU stage2 restore validation missing')
    if not (ROOT/'runs'/arm/'latest.pt').exists():
        raise RuntimeError('Missing same-arm FT continuation checkpoint')
    with lock(ROOT/'runs'/arm/'worker.lock', nonblocking=True):
        wf = shared().workflow()
        wf.schedule = stage_schedule
        wf.initialize_finetuning = prohibit_reinitialization
        wf.dispatch_completed = lambda *_: dispatch(arm)
        def resume(*args, **kwargs):
            job = os.environ.get('JOB_ID')
            return submit_once(f'resume-{arm}-after-{job}', 'work', arm, hold=job)
        wf.submit_job = resume
        dispatch(arm)
        wf.train_worker(ROOT, arm, 0, hours=SOFT_HOURS)


def preflight(arm):
    import torch
    plan = verify()
    audit = json.loads((ROOT/'runs'/arm/'initialization.json').read_text())
    initial_path = ROOT/'runs'/arm/'initial.pt'
    if sha(initial_path) != audit['initial_sha256']:
        raise RuntimeError('Changed initial checkpoint')
    payload = torch.load(initial_path, map_location='cpu', weights_only=False)
    if payload['optimizer_state_dict']['param_groups'][0]['step'] != PARENT_NETWORK_STEP:
        raise RuntimeError('Wrong stage2 initial optimizer step')
    audit_restore(arm, payload, ROOT/'validation'/'GPU-handoff'/arm, gpu=0)
    primary = sorted(rows(ROOT/'data/chemical.jsonl'), key=lambda r: (r['source'], len(r['sequence']), r['id']))
    by_source = {s: [r for r in primary if r['source'] == s] for s in ('pri_miRNA', 'human_mRNA')}
    primary = [by_source['pri_miRNA'][i] for i in (0, len(by_source['pri_miRNA'])//2, -1)]
    primary += [by_source['human_mRNA'][i] for i in (0, len(by_source['human_mRNA'])//3,
                                                  2*len(by_source['human_mRNA'])//3, -1)]
    replay = sorted(rows(ROOT/'data/bprna.jsonl'), key=lambda r: (len(r['sequence']), r['id']))
    replay = [replay[i] for i in (0, len(replay)//2, -1)]
    def mini(name):
        directory = ROOT/'preflight-tests'/arm/name
        if directory.exists():
            shutil.rmtree(directory)
        (directory/'data').mkdir(parents=True)
        (directory/'runs'/arm).mkdir(parents=True)
        (directory/'source').symlink_to(ROOT/'source', target_is_directory=True)
        os.link(ROOT/'baseline.conf', directory/'baseline.conf')
        os.link(ROOT/'job.sh', directory/'job.sh')
        os.link(initial_path, directory/'runs'/arm/'latest.pt')
        os.link(initial_path, directory/'runs'/arm/'initial.pt')
        write_rows(directory/'data/chemical.jsonl', primary)
        write_rows(directory/'data/bprna.jsonl', replay)
        write_rows(directory/'data/test.jsonl', [])
        write_rows(directory/'data/assisted.jsonl', [])
        write_json(directory/'plan.json', {**plan, 'finetune_epochs': 2,
            'data_sha256': {p.name: sha(p) for p in (directory/'data').glob('*.jsonl')}})
        write_json(directory/f'preflight-{arm}.json', {'success': True})
        return directory
    unbroken, segmented = mini('unbroken'), mini('segmented')
    wf = shared().workflow()
    wf.schedule = stage_schedule
    wf.initialize_finetuning = prohibit_reinitialization
    calls = []
    wf.submit_job = lambda *a, **k: calls.append((str(a), k))
    wf.dispatch_completed = lambda *_: None
    wf.train_worker(unbroken, arm, 0, hours=1)
    for _ in range(12):
        wf.train_worker(segmented, arm, 0, hours=0)
        if (segmented/'runs'/arm/'complete.json').exists():
            break
    if not (segmented/'runs'/arm/'complete.json').exists():
        raise RuntimeError('Segmented smoke did not finish')
    a = torch.load(unbroken/'runs'/arm/'latest.pt', map_location='cpu', weights_only=False)
    b = torch.load(segmented/'runs'/arm/'latest.pt', map_location='cpu', weights_only=False)
    for key in ('model_state_dict', 'optimizer_state_dict', 'shape_model_state_dict', 'rng', 'state'):
        assert_same(a[key], b[key], approximate=True)
    if a['state']['updates'] != 6 or not calls:
        raise RuntimeError('Partial-batch resume not exercised')
    delta = max(float((x[torch.isfinite(x)]-b['model_state_dict'][k][torch.isfinite(x)]).abs().max())
                if torch.isfinite(x).any() else 0. for k, x in a['model_state_dict'].items())
    write_json(ROOT/'validation'/f'preflight-{arm}.json', {'success': True,
        'source_optimizer_step': PARENT_NETWORK_STEP, 'source_sha256': audit['source_sha256'],
        'GPU_full_handoff_exact': True, 'continuous_vs_segmented_max_parameter_delta': delta,
        'model_optimizer_Wu_RNG_cursor_restored': True,
        'epochs': 2, 'smoke_optimizer_updates': 6, 'forced_resumptions': len(calls),
        'accumulation_batch_sizes_per_epoch': [4, 4, 1],
        'both_pri_miRNA_and_human_mRNA_tested': True, 'replay_stream_start': REPLAY_START})
    write_json(ROOT/f'preflight-{arm}.json', {'success': True})
    print('Stage2 GPU handoff and resume validation passed', arm, flush=True)


def evaluate(arm, epoch):
    verify()
    shared().evaluate(arm, epoch)
    path = ROOT/'evaluation'/arm/f'epoch-{epoch:03d}'/'summary.json'
    if path.exists():
        value = json.loads(path.read_text())
        write_json(path, {**value, 'stage': 'pri_miRNA plus human_mRNA',
            'global_ft_epoch': EPOCH_OFFSET+epoch, 'stage_local_epoch': epoch,
            'parent_experiment': str(STAGE1)})


def submit():
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT/'scheduler').mkdir(exist_ok=True)
    frozen=Path(__file__).resolve()==(ROOT/'controller.py').resolve()
    job_source=ROOT/'job.sh' if frozen else PROJECT/'scripts/efold_hf/condition3_pri_mrna_ft_job.sh'
    for src, name in [(Path(__file__), 'controller.py'),(job_source, 'job.sh')]:
        target = ROOT/name
        if not target.exists():
            shutil.copyfile(src, target)
        elif sha(src) != sha(target):
            raise RuntimeError('Submitted stage2 bootstrap changed: '+name)
    upstream = None
    if not (STAGE1/'prepare-complete.json').exists():
        upstream = json.loads((STAGE1/'dispatch/prepare.json').read_text())['job_id']
    prep = submit_once('prepare', 'prepare', hold=upstream)
    methods = {}
    for arm in ARMS:
        parent_job = latest_parent_work(arm)
        held = prep if parent_completed(arm) else f'{prep},{parent_job}'
        bridge_job = submit_once('bridge-initial-'+arm, 'bridge', arm, hold=held, waited_job=parent_job)
        methods[arm] = {'stage1_work': parent_job, 'bridge': bridge_job}
    write_json(ROOT/'initial_jobs.json', {'prepare': prep, 'methods': methods})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'submit', 'bridge', 'preflight', 'work', 'evaluate'])
    parser.add_argument('--arm', choices=ARMS, default=os.environ.get('FT_ARM', ARMS[0]))
    parser.add_argument('--epoch', type=int, default=int(os.environ.get('FT_EPOCH', '0')))
    parser.add_argument('--waited-job', default=os.environ.get('WAITED_STAGE1_JOB'))
    args = parser.parse_args()
    if args.action == 'prepare': prepare()
    elif args.action == 'submit': submit()
    elif args.action == 'bridge': bridge(args.arm, args.waited_job)
    elif args.action == 'preflight': preflight(args.arm)
    elif args.action == 'work': train(args.arm)
    else: evaluate(args.arm, args.epoch)


if __name__ == '__main__':
    main()
