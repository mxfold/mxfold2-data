#!/usr/bin/env python3
"""Evaluate the frozen TrainSetA+RNAgym EternaFold on every RNAgym test row."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import fcntl
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pyarrow as pa

SOURCE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SOURCE))
from scripts.chemical_consistency.map_mw_metrics import (
    METRICS, from_bpseq, score_profile, sequence_hash, sha, summarize, write_json,
)
from scripts.eternafold.workflow import PackWriter, pack_list, launcher

PROJECT = Path(os.environ.get('EF_PROJECT_ROOT', str(SOURCE)))
TRAINING = PROJECT / 'qsub_logs/eternafold/eternafold-joint-20261022'
BENCHMARK = PROJECT / 'qsub_logs/rnagym/replay20-full-test-20261023/test_data'
PREVIOUS = PROJECT / 'qsub_logs/chemical_consistency/map_mw_20261028'
EXPECTED_MODEL = '9f9ff688676d1e6f7f5c92c2eeb3e6bbe9b7077b376b9fcd225695dea80c075a'
RAW_DATASET = 'RNAgym_test_all_profiles'
BENCH_DATASET = 'RNAgym_full_test'
PROBES = ('2A3', 'DMS', '1M7', 'NMIA', 'CMCT', 'BzCN')
TRAINING_LABELS = {
    'pretrain': ('TrainSetAのみ', 'EternaFold_TrainSetA_only'),
    'joint_synthetic_2A3': ('TrainSetA＋TrainSetB合成2A3', 'EternaFold_TrainSetA_TrainSetB_synthetic_2A3'),
    'joint_rnagym_all': ('TrainSetA＋RNAgym train', 'EternaFold_TrainSetA_RNAgym'),
}


def now():
    return datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(timespec='seconds')


def task_index(value):
    # SGE exports the literal "undefined" for ordinary, non-array jobs.
    return 1 if value in (None, '', 'undefined') else int(value)


def emit(**items):
    print(json.dumps({'time_jst': now(), **items}, ensure_ascii=False), flush=True)


def initialize(root):
    if root.exists():
        raise FileExistsError(root)
    model = TRAINING / 'runs/joint_rnagym_all/model.params'
    marker = json.loads(model.with_name('complete.json').read_text())
    if not marker['complete'] or sha(model) != marker['model_sha256']:
        raise ValueError('Training completion/model mismatch')
    if marker['model_sha256'] != EXPECTED_MODEL:
        raise ValueError('This evaluation requires the frozen joint RNAgym model')
    root.mkdir(parents=True)
    for part in ('bin', 'scheduler', 'source/scripts/eternafold', 'source/scripts/chemical_consistency'):
        (root / part).mkdir(parents=True)
    for name in ('contrafold', 'contrafold-mpi'):
        shutil.copy2(TRAINING / 'bin' / name, root / 'bin' / name)
    shutil.copyfile(model, root / 'model.params')
    shutil.copyfile(model.with_name('complete.json'), root / 'training_complete.json')
    files = ['scripts/eternafold/fulltest_new_metrics.py', 'scripts/eternafold/qsub_fulltest_new_metrics.sh',
             'scripts/eternafold/workflow.py', 'scripts/chemical_consistency/map_mw_metrics.py']
    for name in files:
        shutil.copyfile(PROJECT / name, root / 'source' / name)
    raw = PROJECT / 'data/rnagym/secondary_structure_prediction/RMDB_s40_test.parquet'
    old_plan = json.loads((BENCHMARK.parent / 'comparison_plan.json').read_text())
    known = PREVIOUS / 'predictions/EternaFold_joint_rnagym/epoch-000.npz'
    known_info = json.loads((PREVIOUS / 'metrics/EternaFold_joint_rnagym/epoch-000/summary.json').read_text())
    if known_info['provenance']['params_sha256'] != EXPECTED_MODEL:
        raise ValueError('Saved panel uses a different model')
    if sha(known) != known_info['provenance']['prediction_sha256']:
        raise ValueError('Saved panel predictions changed')
    plan = {
        'schema': 'eternafold-full-rnagym-new-metrics-v1', 'created_jst': now(),
        'training_root': str(TRAINING), 'model_sha256': EXPECTED_MODEL,
        'raw_test': str(raw), 'raw_test_sha256': old_plan['test_parquet_sha256'],
        'benchmark_data': str(BENCHMARK), 'benchmark_meta_sha256': sha(BENCHMARK / 'complete.json'),
        'saved_panel': str(known), 'saved_panel_sha256': sha(known),
        'baseline_metrics': str(PREVIOUS / 'full/baseline'),
        'shards': 16, 'ranks': 128, 'array_concurrency': 4, 'score_workers': 32,
        'inference': {'decoder': 'native MEA', 'gamma': 6, 'chemical_evidence': False,
                      'export_bpp': False, 'length_filter': None},
        'metric_definition': {
            'auroc': 'label=binary predicted unpaired state; score=raw measured reactivity; ties=0.5',
            'spearman': 'raw measured reactivity versus the same binary unpaired state',
            'mask': 'finite reactivity > -10 and probe-appropriate nucleotide',
            'strata': 'DMS A/C separately, CMCT G/U separately, other probes ACGU',
            'BzCN': 'SHAPE reagent; use the existing all-nucleotide SHAPE scorer',
            'BzCN_reference': 'https://www.nature.com/articles/nprot.2009.126',
            'aggregation': 'base strata mean -> measurement -> sequence mean over replicates -> sequence macro',
            'primary': 'Only strata with both AUROC and Spearman defined; all input measurements retained in coverage counts',
            'undefined': 'Constant structure: AUROC/Spearman undefined; constant reactivity: Spearman undefined',
            'neutral_auc': 'Diagnostic only; never used in the primary AUROC',
        },
        'cohorts': {
            RAW_DATASET: 'Every row and every probe in the original official test parquet; no SNR/length filtering',
            BENCH_DATASET: 'Exact frozen 433539-sequence DMS/2A3 highest-SNR cohort, for comparison with earlier full-test results',
        },
        'source_sha256': {name: sha(root / 'source' / name) for name in files},
        'binary_sha256': {name: sha(root / 'bin' / name) for name in ('contrafold', 'contrafold-mpi')},
    }
    write_json(root / 'plan.json', plan)
    emit(event='initialized', root=str(root), model_sha256=EXPECTED_MODEL)


def verify(root):
    plan = json.loads((root / 'plan.json').read_text())
    if sha(root / 'model.params') != plan['model_sha256']:
        raise ValueError('Frozen model changed')
    for name, expected in plan['source_sha256'].items():
        if sha(root / 'source' / name) != expected:
            raise ValueError('Frozen source changed: ' + name)
    for name, expected in plan['binary_sha256'].items():
        if sha(root / 'bin' / name) != expected:
            raise ValueError('Frozen binary changed: ' + name)
    return plan


def attach_benchmark_ids(raw, frozen, shard):
    """Identify the exact historical selected rows, validating their raw values."""
    selected = raw[raw.modifier.isin(('2A3', 'DMS'))].sort_values(
        ['modifier', 'sequence', 'SNR'], ascending=[True, True, False], kind='mergesort')
    selected = selected.drop_duplicates(['modifier', 'sequence']).set_index(['sequence', 'modifier'])
    keys = pd.MultiIndex.from_frame(frozen[['sequence', 'modifier']])
    chosen = selected.loc[keys]
    if not np.array_equal(chosen.reactivity.to_numpy(), frozen.reactivity.to_numpy()):
        raise ValueError('Highest-SNR raw rows differ from the frozen full-test cohort')
    if not np.allclose(chosen.SNR.to_numpy(float), frozen.SNR.to_numpy(float), equal_nan=True):
        raise ValueError('Highest-SNR SNR values differ from the frozen cohort')
    answer = raw.copy()
    answer['benchmark_profile_id'] = ''
    answer.loc[chosen.raw_row_id.to_numpy(), 'benchmark_profile_id'] = [
        f'{shard}:{i}' for i in range(len(frozen))]
    return answer


def prepare(root):
    plan = verify(root)
    if (root / 'data/complete.json').exists():
        raise FileExistsError('Inputs are already prepared')
    pa.set_cpu_count(8)
    pa.set_io_thread_count(4)
    raw_path = Path(plan['raw_test'])
    emit(event='checking_raw_input')
    if sha(raw_path) != plan['raw_test_sha256']:
        raise ValueError('Original test parquet changed')
    meta_path = Path(plan['benchmark_data']) / 'complete.json'
    if sha(meta_path) != plan['benchmark_meta_sha256']:
        raise ValueError('Frozen benchmark metadata changed')
    meta = json.loads(meta_path.read_text())
    assignment, loads, benchmark_paths = {}, np.zeros(plan['shards'], dtype=np.float64), {}
    for shard in range(1, plan['shards'] + 1):
        path = Path(plan['benchmark_data']) / f'shard-{shard}.parquet'
        if sha(path) != meta['shards'][str(shard)]['sha256']:
            raise ValueError('Frozen benchmark shard changed')
        sequences = pd.read_parquet(path, columns=['sequence']).sequence.unique()
        for sequence in sequences:
            if sequence in assignment:
                raise ValueError('The same sequence appears in multiple shards')
            assignment[sequence] = shard
            loads[shard - 1] += len(sequence) ** 3
        benchmark_paths[shard] = path
    emit(event='reading_all_test_rows')
    frame = pd.read_parquet(raw_path, columns=['seqID', 'sequence', 'modifier', 'SNR', 'reactivity']).reset_index(drop=True)
    frame['sequence'] = frame.sequence.str.upper().str.replace('T', 'U', regex=False)
    frame['raw_row_id'] = np.arange(len(frame), dtype=np.int64)
    if set(frame.modifier.unique()) != set(PROBES):
        raise ValueError('Unexpected test probes: ' + str(frame.modifier.unique()))
    seqs = set(frame.sequence.unique())
    if not set(assignment).issubset(seqs):
        raise ValueError('Historical benchmark is not a subset of the original test')
    extra = sorted(seqs - set(assignment), key=lambda s: (-len(s), s))
    for sequence in extra:
        shard = int(np.argmin(loads)) + 1
        assignment[sequence] = shard
        loads[shard - 1] += len(sequence) ** 3
    if any(not sequence.isalpha() or not sequence.isascii() for sequence in seqs):
        raise ValueError('Unsupported non-alphabetic nucleotide in test input')
    frame['shard'] = frame.sequence.map(assignment)
    data = root / 'data'
    data.mkdir(exist_ok=True)
    info = {'plan_sha256': sha(root / 'plan.json'), 'raw_rows': len(frame),
            'unique_sequences': len(seqs), 'historical_benchmark_sequences': len(assignment) - len(extra),
            'additional_sequences': len(extra), 'length_min': min(map(len, seqs)),
            'length_max': max(map(len, seqs)),
            'ambiguous_sequences': sum(bool(set(s) - set('ACGU')) for s in seqs),
            'probe_rows': {str(k): int(v) for k, v in frame.modifier.value_counts().items()}, 'shards': {}}
    for shard in range(1, plan['shards'] + 1):
        part = frame[frame.shard == shard].copy().set_index('raw_row_id', drop=False)
        frozen = pd.read_parquet(benchmark_paths[shard])
        part = attach_benchmark_ids(part, frozen, shard)
        part = part.drop(columns='shard').reset_index(drop=True)
        sequences = sorted(part.sequence.unique(), key=lambda s: (len(s), s))
        entries = pd.DataFrame([{'sequence': s, 'sequence_sha256': sequence_hash(s),
                                'name': 'R' + sequence_hash(s)} for s in sequences])
        prefix = data / f'shard-{shard:02d}'
        entries.to_parquet(prefix.with_suffix('.sequences.parquet'), index=False)
        part.to_parquet(prefix.with_suffix('.profiles.parquet'), index=False)
        writer = PackWriter(prefix.with_suffix('.efpack'))
        for row in entries.itertuples(index=False):
            writer.add(row.name, row.sequence)  # no structure or reactivity
        pack = writer.finish()
        pack_list(writer.path)
        paths = [prefix.with_suffix(s) for s in ('.sequences.parquet', '.profiles.parquet', '.efpack', '.efpack.lst')]
        info['shards'][str(shard)] = {
            'unique_sequences': len(sequences), 'raw_rows': len(part),
            'benchmark_rows': int(part.benchmark_profile_id.ne('').sum()),
            'probe_rows': {str(k): int(v) for k, v in part.modifier.value_counts().items()},
            'files_sha256': {str(p.relative_to(root)): sha(p) for p in paths},
            'sequence_only_pack_sha256': pack['sha256'],
        }
        emit(event='prepared_shard', shard=shard, **{k: info['shards'][str(shard)][k] for k in ('unique_sequences', 'raw_rows', 'benchmark_rows')})
    if sum(s['benchmark_rows'] for s in info['shards'].values()) != meta['deduplicated_rows']:
        raise ValueError('Historical cohort coverage mismatch')
    write_json(data / 'complete.json', info)
    emit(event='preparation_complete', rows=len(frame), unique_sequences=len(seqs))


def prepared(root, plan):
    meta = json.loads((root / 'data/complete.json').read_text())
    if meta['plan_sha256'] != sha(root / 'plan.json'):
        raise ValueError('Prepared data belongs to a different plan')
    return meta


def predict_native(root, entries, folder, ranks):
    folder.mkdir(parents=True, exist_ok=True)
    local = folder / 'runtime'
    (local / 'bin').mkdir(parents=True)
    for name in ('contrafold', 'contrafold-mpi'):
        shutil.copy2(root / 'bin' / name, local / 'bin' / name)
    shutil.copyfile(root / 'model.params', local / 'model.params')
    writer = PackWriter(local / 'input.efpack')
    for row in entries.itertuples(index=False):
        writer.add(row.name, row.sequence)
    writer.finish()
    inputs = pack_list(writer.path)
    bpseq = local / 'bpseq'
    bpseq.mkdir()
    command = launcher(local, ranks) + ['predict', '--numdatasources', '5', '--params', str(local / 'model.params'),
              '--gamma', '6', '--bpseq', str(bpseq), '--examplefile', str(inputs)]
    start = time.monotonic()
    with (folder / 'stdout.log').open('w') as out, (folder / 'stderr.log').open('w') as err:
        subprocess.run(command, cwd=local, stdout=out, stderr=err, check=True)
    return bpseq, {'ranks': ranks, 'seconds': time.monotonic() - start, 'command': command}


def probe_scores(sequence, modifier, values, unpaired):
    # BzCN is a SHAPE reagent. Its raw label is preserved in every output row.
    scorer_modifier = 'SHAPE' if modifier == 'BzCN' else modifier
    return score_profile({'sequence': sequence, 'modifier': scorer_modifier, 'values': values}, unpaired)


def preflight(root):
    plan = verify(root)
    prepared(root, plan)
    entries = pd.concat([pd.read_parquet(root / 'data' / f'shard-{s:02d}.sequences.parquet')
                         for s in range(1, plan['shards'] + 1)], ignore_index=True)
    if sha(plan['saved_panel']) != plan['saved_panel_sha256']:
        raise ValueError('Saved panel predictions changed')
    with np.load(plan['saved_panel']) as saved:
        known = {str(k): saved['values'][saved['offsets'][i]:saved['offsets'][i+1]].copy()
                 for i, k in enumerate(saved['keys'])}
    panel = entries[entries.sequence_sha256.isin(known)].sort_values('sequence_sha256').head(16)
    if len(panel) != 16:
        raise ValueError('Could not identify enough archived panel checks')
    # Include the longest/shortest sequences and at least one RNA from each probe.
    samples = [panel, entries.sort_values('sequence', key=lambda x: x.str.len()).iloc[[0, -1]]]
    profiles = pd.read_parquet(root / 'data/shard-01.profiles.parquet', columns=['sequence', 'modifier'])
    selected = profiles.groupby('modifier', sort=False).head(1).sequence
    samples.append(entries[entries.sequence.isin(selected)])
    sample = pd.concat(samples).drop_duplicates('sequence_sha256').reset_index(drop=True)
    work = root / 'preflight'
    work.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='ef-full-smoke-', dir=os.environ.get('TMPDIR')) as tmp:
        tmp = Path(tmp)
        try:
            serial, serial_info = predict_native(root, sample, tmp / 'serial', 1)
            parallel, mpi_info = predict_native(root, sample, tmp / 'mpi', 4)
            checked = 0
            for row in sample.itertuples(index=False):
                a, b = from_bpseq(serial / row.name, row.sequence), from_bpseq(parallel / row.name, row.sequence)
                if not np.array_equal(a, b):
                    raise ValueError('Serial/MPI structure disagreement')
                if row.sequence_sha256 in known:
                    if not np.array_equal(a, known[row.sequence_sha256]):
                        raise ValueError('New inference differs from the archived gamma=6 structure')
                    checked += 1
        finally:
            for label in ('serial', 'mpi'):
                for stream in ('stdout.log', 'stderr.log'):
                    if (tmp / label / stream).exists():
                        shutil.copyfile(tmp / label / stream, work / f'{label}.{stream}')
    write_json(work / 'complete.json', {'success': True, 'plan_sha256': sha(root / 'plan.json'),
               'sequences': len(sample), 'archived_panel_structures_verified': checked,
               'serial': serial_info, 'mpi': mpi_info, 'completed_jst': now()})
    emit(event='preflight_complete', sequences=len(sample), archived_checks=checked)


def score_sequence(task):
    sequence, key, name, bpseq, profiles = task
    states = from_bpseq(Path(bpseq) / name, sequence)
    rows = []
    for raw_id, sample_id, modifier, text, benchmark_id in profiles:
        values = np.fromstring(text.strip('[]'), sep=',', dtype=np.float64)
        if len(values) != len(sequence):
            raise ValueError('Test reactivity length mismatch')
        for scores in probe_scores(sequence, modifier, values, states):
            common = {'modifier': modifier, 'unit_id': key, 'sequence_sha256': key, 'cluster': key, **scores}
            rows.append({'dataset': RAW_DATASET, 'profile_id': f'raw:{raw_id}',
                         'sample_id': str(sample_id), **common})
            if benchmark_id:
                rows.append({'dataset': BENCH_DATASET, 'profile_id': benchmark_id,
                             'sample_id': benchmark_id, **common})
    return key, states, rows


def save_predictions(path, predictions):
    keys = sorted(predictions)
    offsets = np.cumsum([0] + [len(predictions[k]) for k in keys], dtype=np.int64)
    values = np.concatenate([predictions[k] for k in keys]).astype(np.uint8)
    temp = path.with_suffix('.tmp')
    with temp.open('wb') as handle:
        np.savez_compressed(handle, keys=np.asarray(keys, dtype='U64'), offsets=offsets, values=values)
    temp.replace(path)


def predict(root, shard, ranks, workers):
    plan = verify(root)
    meta = prepared(root, plan)
    smoke = json.loads((root / 'preflight/complete.json').read_text())
    if not smoke['success'] or smoke['plan_sha256'] != sha(root / 'plan.json'):
        raise ValueError('Preflight has not passed for this run')
    dest = root / 'evaluation' / f'shard-{shard:02d}'
    dest.mkdir(parents=True, exist_ok=True)
    with (dest / '.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (dest / 'complete.json').exists():
            emit(event='already_complete', shard=shard)
            return
        for name, expected in meta['shards'][str(shard)]['files_sha256'].items():
            if sha(root / name) != expected:
                raise ValueError('Prepared input shard changed')
        pa.set_cpu_count(1)
        pa.set_io_thread_count(1)
        entries = pd.read_parquet(root / 'data' / f'shard-{shard:02d}.sequences.parquet', use_threads=False)
        profiles = pd.read_parquet(root / 'data' / f'shard-{shard:02d}.profiles.parquet', use_threads=False)
        start = time.monotonic()
        emit(event='prediction_start', shard=shard, sequences=len(entries), ranks=ranks)
        with tempfile.TemporaryDirectory(prefix=f'ef-full-{shard:02d}-', dir=os.environ.get('TMPDIR')) as tmp:
            tmp = Path(tmp)
            try:
                bpseq, native = predict_native(root, entries, tmp, ranks)
            finally:
                for stream in ('stdout.log', 'stderr.log'):
                    if (tmp / stream).exists():
                        shutil.copyfile(tmp / stream, dest / stream)
            emit(event='native_prediction_complete', shard=shard, seconds=native['seconds'])
            lookup = entries.set_index('sequence')
            def tasks():
                columns = ['raw_row_id', 'seqID', 'modifier', 'reactivity', 'benchmark_profile_id']
                for sequence, group in profiles.groupby('sequence', sort=False):
                    entry = lookup.loc[sequence]
                    yield sequence, entry.sequence_sha256, entry['name'], str(bpseq), list(group[columns].itertuples(index=False, name=None))
            predictions, scores = {}, []
            # Workers do only NumPy/SciPy scoring and text parsing, never Arrow I/O.
            with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('fork')) as pool:
                for count, (key, states, rows) in enumerate(pool.map(score_sequence, tasks(), chunksize=64), 1):
                    predictions[key] = states
                    scores.extend(rows)
                    if count % 5000 == 0:
                        emit(event='score_progress', shard=shard, sequences=count, total=len(entries))
            if len(predictions) != len(entries):
                raise ValueError('Prediction coverage mismatch')
            save_predictions(dest / 'predictions.npz', predictions)
        scored = pd.DataFrame(scores)
        for dataset, expected in ((RAW_DATASET, len(profiles)), (BENCH_DATASET, int(profiles.benchmark_profile_id.ne('').sum()))):
            if scored.loc[scored.dataset == dataset, 'profile_id'].nunique() != expected:
                raise ValueError('Not all input measurements were scored')
        temporary = dest / 'profiles.parquet.tmp'
        scored.to_parquet(temporary, index=False)
        temporary.replace(dest / 'profiles.parquet')
        joint = scored[scored.auroc.notna() & scored.spearman.notna()]
        info = {'success': True, 'shard': shard, 'plan_sha256': sha(root / 'plan.json'),
                'model_sha256': plan['model_sha256'], 'unique_sequences': len(entries),
                'raw_profiles': len(profiles), 'benchmark_profiles': int(profiles.benchmark_profile_id.ne('').sum()),
                'native': native, 'scoring_workers': workers, 'seconds': time.monotonic() - start,
                'all_available': summarize(scored), 'joint_defined': summarize(joint),
                'files_sha256': {name: sha(dest / name) for name in ('profiles.parquet', 'predictions.npz')},
                'completed_jst': now()}
        write_json(dest / 'complete.json', info)
        emit(event='shard_complete', shard=shard, seconds=info['seconds'])


def merge_summaries(pieces):
    answer = []
    for dataset, modifier in sorted({(s['dataset'], s['modifier']) for p in pieces for s in p}):
        rows = [s for p in pieces for s in p if (s['dataset'], s['modifier']) == (dataset, modifier)]
        row = {'dataset': dataset, 'modifier': modifier}
        for key in ('profiles', 'units', 'strata', 'n_bases', 'n_collapsed_strata', 'n_insufficient_strata'):
            row[key] = sum(s[key] for s in rows)
        for metric in METRICS:
            count = 'n_units_' + metric + '_defined'
            n = sum(s[count] for s in rows)
            row[count] = n
            row['n_strata_' + metric + '_defined'] = sum(s['n_strata_' + metric + '_defined'] for s in rows)
            row[metric + '_sequence_macro'] = (sum(s[metric + '_sequence_macro'] * s[count]
                                                  for s in rows if s[count]) / n if n else None)
        answer.append(row)
    return answer


def common_frames(frames):
    indexed = [f.set_index(['profile_id', 'stratum'], verify_integrity=True) for f in frames]
    common = None
    for frame in indexed:
        good = frame.index[frame.auroc.notna() & frame.spearman.notna()]
        common = good if common is None else common.intersection(good)
    return [f.loc[common].reset_index() for f in indexed]


def aggregate(root):
    plan = verify(root)
    meta = prepared(root, plan)
    pieces, paired = [], [[], []]
    unique = set()
    baseline_sources = {}
    for shard in range(1, plan['shards'] + 1):
        dest = root / 'evaluation' / f'shard-{shard:02d}'
        info = json.loads((dest / 'complete.json').read_text())
        if not info['success'] or info['plan_sha256'] != sha(root / 'plan.json'):
            raise ValueError('Incomplete/different shard')
        for name, expected in info['files_sha256'].items():
            if sha(dest / name) != expected:
                raise ValueError('Completed output changed')
        with np.load(dest / 'predictions.npz') as arr:
            keys = set(arr['keys'].tolist())
        if unique & keys:
            raise ValueError('A sequence was predicted in more than one shard')
        unique.update(keys)
        pieces.append(info)
        baseline = Path(plan['baseline_metrics']) / f'shard-{shard:02d}'
        baseline_meta = json.loads((baseline / 'summary.json').read_text())
        if sha(baseline / 'profiles.parquet') != baseline_meta['profiles_sha256']:
            raise ValueError('Archived MXfold2 baseline metrics changed')
        ours = pd.read_parquet(dest / 'profiles.parquet')
        ours = ours[ours.dataset == BENCH_DATASET]
        old = pd.read_parquet(baseline / 'profiles.parquet')
        if set(zip(ours.profile_id, ours.stratum)) != set(zip(old.profile_id, old.stratum)):
            raise ValueError('Historical benchmark profile/stratum coordinates differ')
        for output, frame in zip(paired, common_frames([old, ours])):
            output.append(summarize(frame))
        baseline_sources[str(shard)] = sha(baseline / 'profiles.parquet')
    if len(unique) != meta['unique_sequences'] or sum(p['raw_profiles'] for p in pieces) != meta['raw_rows']:
        raise ValueError('Not every original test sequence/profile was evaluated')
    available = merge_summaries([p['all_available'] for p in pieces])
    primary = merge_summaries([p['joint_defined'] for p in pieces])
    counts = {s['modifier']: s['profiles'] for s in available if s['dataset'] == RAW_DATASET}
    if counts != meta['probe_rows']:
        raise ValueError('Original probe coverage mismatch')
    training_label, model_name = TRAINING_LABELS[plan.get('training_condition', 'joint_rnagym_all')]
    comparison = [{'model': name, **s} for name, item in zip(('MXfold2_TrainSetA_baseline', model_name), paired)
                  for s in merge_summaries(item)]
    result = {'success': True, 'completed_jst': now(), 'plan_sha256': sha(root / 'plan.json'),
              'model_sha256': plan['model_sha256'], 'unique_sequences': len(unique), 'raw_profiles': meta['raw_rows'],
              'all_available': available, 'primary_joint_defined': primary,
              'matched_historical_baseline_comparison': comparison,
              'baseline_profiles_sha256': baseline_sources, 'shards_complete': len(pieces)}
    for name, rows in (('all_available.tsv', available), ('primary.tsv', primary), ('matched_baseline.tsv', comparison)):
        pd.DataFrame(rows).to_csv(root / name, sep='\t', index=False, float_format='%.10f')
    lines = [f'# EternaFold（{training_label}）のRNAgym test全体評価', '',
             f'完了：{now()}。モデルSHA256：`{plan["model_sha256"]}`。', '',
             f'元test全体：{len(unique):,}配列・{meta["raw_rows"]:,}測定。全試薬・全反復測定を保持。', '',
             '単一MEA構造（gamma=6）を配列だけから予測。新AUROCは未対合状態を1、raw反応性をscoreとする。',
             '新Spearmanも同じbinary状態とraw反応性。有限かつr>−10。DMS A/C、CMCT G/Uを別計算。',
             'BzCNはSHAPEとして全4塩基を採点。塩基種層→測定→配列の順に平均する。',
             '主表は両指標が定義できる同一stratumで計算。未定義例はcoverageに残し、0.5補完しない。', '',
             '| 試薬 | 入力測定 | 有効RNA | 新AUROC | 新Spearman |', '|---|---:|---:|---:|---:|']
    primary_raw = {r['modifier']: r for r in primary if r['dataset'] == RAW_DATASET}
    for probe in PROBES:
        r = primary_raw.get(probe)
        if r is None:
            lines.append(f'| {probe} | {counts[probe]:,} | 0 | — | — |')
        else:
            lines.append(f'| {probe} | {counts[probe]:,} | {r["units"]:,} | {r["auroc_sequence_macro"]:.6f} | {r["spearman_sequence_macro"]:.6f} |')
    lines += ['', '2A3/DMS最高SNRの旧全test集合は別cohortとして保存。`matched_baseline.tsv`では保存MXfold2 baselineと共通の定義可能stratumで直接比較する。',
              '', f'CPU：cpu_160×最大4ジョブ、各MPI 128プロセス（最大512 MPI）。全{len(pieces)} shard完了。',
              '', 'raw測定の個別結果とbinary構造は各`evaluation/shard-XX/`に保存。入力・コード・重み・出力のSHA256と完了マーカーを確認して集計した。', '']
    (root / 'REPORT_ja.md').write_text('\n'.join(lines))
    write_json(root / 'results.json', result)
    emit(event='evaluation_complete', sequences=len(unique), profiles=meta['raw_rows'], shards=len(pieces))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('initialize', 'prepare', 'preflight', 'predict', 'aggregate'))
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--shard', type=int, default=task_index(os.environ.get('SGE_TASK_ID')))
    p.add_argument('--ranks', type=int, default=128)
    p.add_argument('--workers', type=int, default=32)
    a = p.parse_args()
    root = a.root.resolve()
    if a.action == 'initialize':
        initialize(root)
    elif a.action == 'prepare':
        prepare(root)
    elif a.action == 'preflight':
        preflight(root)
    elif a.action == 'predict':
        predict(root, a.shard, a.ranks, a.workers)
    else:
        aggregate(root)


if __name__ == '__main__':
    main()
