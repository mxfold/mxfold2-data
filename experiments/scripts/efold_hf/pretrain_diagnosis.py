#!/usr/bin/env python3
"""Read-only diagnosis of the frozen bpRNA pretraining experiment."""
from __future__ import annotations
import ast
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(Path(__file__).parent)]
import bprnanew_benchmark as base
from workflow import project_pairs
from paper_evaluation import pair_metrics

OUT = ROOT / 'qsub_logs/efold_hf/pretrain_diagnosis_20261031'
DATA = ROOT / 'data/efold_hf/pretrain_diagnosis_20261031'
FULL = ROOT / 'qsub_logs/efold_hf/paper_full_20261030'
PRIOR = ROOT / 'qsub_logs/efold_hf/bprnanew_20261029'
OLD = base.OLD
SHARDS = 2


def stable_sample(rows, n):
    return sorted(rows, key=lambda r: hashlib.sha256(('diagnosis20261031:' + r['id']).encode()).hexdigest())[:n]


def prepare():
    import torch
    import numpy as np
    from rapidfuzz import process
    from rapidfuzz.distance import Levenshtein
    import pyarrow as pa
    import pyarrow.feather as feather
    pa.set_cpu_count(1); pa.set_io_thread_count(1)
    OUT.mkdir(parents=True, exist_ok=True)
    train = base.read_jsonl(OLD / 'data/bprna.jsonl')
    panel = [r for r in base.read_jsonl(FULL / 'panel.jsonl') if r['source'] in ('PDB', 'archiveII')]
    original = []
    with tarfile.open(DATA / 'Rivas.tar.gz') as bundle:
        for source in ('TrainSetA', 'TrainSetB'):
            names = bundle.extractfile(f'data/{source}.lst').read().decode().splitlines()
            for name in names:
                filename = f'data/{source}/' + Path(name).name
                item = base.bpseq(bundle.extractfile(filename).read().decode(), f'{source}:{Path(name).stem}')
                item['source'] = source
                original.append(item)
    base.write_jsonl(OUT / 'official_TrainSetAB.jsonl', original)
    originals_by_seq = defaultdict(list)
    for row in original:
        originals_by_seq[row['sequence']].append(row['id'])
    actual_by_seq = {r['sequence']: r['id'] for r in train}
    base.write_json(OUT / 'training_dataset_summary.json', {
        'bpRNA_n': len(train), 'official_TrainSetAB_n': len(original),
        'official_counts': dict(Counter(x['source'] for x in original)),
        'training_source_counts': dict(Counter(x['id'].split(':')[-1].split('_')[1] for x in train)),
        'bpRNA_labels': {'original_pairs': sum(len(x['pairs']) for x in train),
                         'projected_pairs': sum(len(x['train_pairs']) for x in train),
                         'empty_projected_targets': sum(not x['train_pairs'] for x in train)},
        'lengths': {name: {'n': len(rows), 'mean': statistics.mean(len(x['sequence']) for x in rows),
            'median': statistics.median(len(x['sequence']) for x in rows),
            'bins': {label: sum(lo <= len(x['sequence']) <= hi for x in rows) for label, lo, hi in
                     [('1-30', 1, 30), ('31-50', 31, 50), ('51-100', 51, 100),
                      ('101-200', 101, 200), ('201-500', 201, 500), ('501-1000', 501, 1000), ('1001+', 1001, 10000)]}}
            for name, rows in [('bpRNA', train), ('TrainSetAB', original)] +
                              [(source, [x for x in panel if x['source'] == source]) for source in ('PDB', 'archiveII')]},
        'official_archive_sha256': base.sha(DATA / 'Rivas.tar.gz')})
    epochs = [json.loads(p.read_text()) for p in sorted((OLD / 'runs/pretrain').glob('epoch-*-complete.json'))]
    progress = {}
    for filename in sorted((OLD / 'scheduler').glob('ef_wor_pretrain.o*')):
        for line in filename.open():
            if line.startswith('pretrain {'):
                row = ast.literal_eval(line[len('pretrain '):].strip())
                row['log_file'] = str(filename)
                progress[(row['epoch'], row['cursor'])] = row
    local = []
    for epoch in epochs:
        previous_cursor, previous_sum = 0, 0.
        for row in sorted((v for (e, _), v in progress.items() if e == epoch['epoch']), key=lambda v: v['cursor']):
            local.append({**row, 'block_n': row['cursor']-previous_cursor,
                          'block_mean_loss': (row['loss_sum']-previous_sum)/(row['cursor']-previous_cursor)})
            previous_cursor, previous_sum = row['cursor'], row['loss_sum']
    base.write_json(OUT / 'training_curve.json', {'epochs': epochs, 'blocks': local,
        'intermediate_weights_available': [p.name for p in (OLD / 'runs/pretrain').glob('epoch-*.pt')],
        'validation_curve_available': False,
        'note': 'Only final epoch10 weights were retained; per-epoch losses are observed, validation F1 curves cannot be reconstructed.'})
    # Reuse all existing per-test predictions rather than rerunning the benchmarks.
    current = {}
    for path in sorted((FULL / 'predictions/mx_bprna10').glob('shard-*.jsonl')):
        current.update({r['id']: r for r in base.read_jsonl(path)})
    author = feather.read_table(ROOT / 'data/efold_hf/paper_2026/results_benchmark_algos.feather',
                                columns=['dataset', 'reference', 'model', 'structure'], use_threads=False).to_pylist()
    published = {r['dataset']+':'+r['reference']: r['structure'] for r in author if r['model'] == 'MXFold2'}
    existing_similarity = {}
    for filename in sorted((PRIOR / 'similarity').glob('global-shard-*.jsonl')):
        existing_similarity.update({r['id']: r for r in base.read_jsonl(filename)})
    nearest_path = OUT / 'nearest_TrainSetAB.jsonl'
    nearest = {r['id']: r for r in base.read_jsonl(nearest_path)} if nearest_path.exists() else {}
    choices = [r['sequence'] for r in original]
    for start in range(0, len(panel), 32):
        batch = [r for r in panel[start:start+32] if r['id'] not in nearest]
        if not batch:
            continue
        similarities = process.cdist([r['sequence'] for r in batch], choices,
                                     scorer=Levenshtein.normalized_similarity, workers=4, dtype=np.float64)
        with nearest_path.open('a') as handle:
            for item, values in zip(batch, similarities):
                best = int(values.argmax())
                row = {'id': item['id'], 'best_id': original[best]['id'], 'similarity': float(values[best])}
                handle.write(json.dumps(row)+'\n'); nearest[item['id']] = row
        if start % 320 == 0:
            print('TrainSetAB nearest:', len(nearest), '/', len(panel), flush=True)
    diagnostics = []
    for item in panel:
        pairs = project_pairs(item['sequence'], item['pairs'])
        ours = current[item['id']]
        pub_pairs = published[item['id']]
        pub = pair_metrics(item['pairs'], pub_pairs)
        diagnostics.append({'id': item['id'], 'source': item['source'], 'length': len(item['sequence']),
            'family': item['reference'].split('_')[0] if item['source'] == 'archiveII' else 'PDB',
            'ref_pairs': len(item['pairs']), 'projected_ref_pairs': len(pairs),
            'canonical_noncrossing_ceiling_f1': pair_metrics(item['pairs'], pairs)['f1'],
            'ours_pred_pairs': len(ours['predicted_pairs']), 'public_pred_pairs': len(pub_pairs),
            'ours_f1': ours['f1'], 'public_f1': pub['f1'], 'gap_public_minus_ours': pub['f1']-ours['f1'],
            'ours_precision': ours['precision'], 'ours_recall': ours['recall'],
            'public_precision': pub['precision'], 'public_recall': pub['recall'],
            'bpRNA_nearest_edit_similarity': existing_similarity.get(item['id'], {}).get('global_edit_similarity'),
            'TrainSetAB_nearest_edit_similarity': nearest[item['id']]['similarity'],
            'TrainSetAB_exact_ids': originals_by_seq.get(item['sequence'], []),
            'bpRNA_exact_id': actual_by_seq.get(item['sequence'])})
    base.write_jsonl(OUT / 'per_test_diagnostics.jsonl', diagnostics)
    # Fixed diagnostic controls; these do not replace the official full-test scores.
    selected_train = stable_sample(train, 256)
    samples = [{**r, 'sample_set': 'bpRNA_train_sample'} for r in selected_train]
    samples += [{**r, 'train_pairs': project_pairs(r['sequence'], r['pairs']), 'sample_set': 'PDB'}
                for r in panel if r['source'] == 'PDB']
    archive = [r for r in panel if r['source'] == 'archiveII' and len(r['sequence']) <= 1000]
    samples += [{**r, 'train_pairs': project_pairs(r['sequence'], r['pairs']), 'sample_set': 'ArchiveII_sample'}
                for r in stable_sample(archive, 128)]
    for r in samples:
        r['sequence_sha256'] = hashlib.sha256(r['sequence'].encode()).hexdigest()
    base.write_jsonl(OUT / 'diagnostic_panel.jsonl', samples)
    prior_models = json.loads((PRIOR / 'plan.json').read_text())['models']
    models = [next(m for m in prior_models if m['model'] == name) for name in ('mx_bprna10', 'MXfold2_public')]
    base.write_json(OUT / 'plan.json', {'models': models, 'shards': SHARDS, 'n': len(samples),
        'sample_counts': dict(Counter(r['sample_set'] for r in samples)),
        'panel_sha256': base.sha(OUT / 'diagnostic_panel.jsonl'),
        'selection': 'Prespecified SHA256 sample ordering; all PDB355, uniform ArchiveII128 <=1000nt, actual bpRNA256. No checkpoints or weights selected on tests.',
        'analyses': ['default prediction', 'relative neural score alpha=0/.25/.5/2', 'additional scalar parameters disabled',
                     'clean FY/thermodynamic loss decomposition', 'noise/dropout loss on a small fixed training subset'],
        'training_changes': 'none; all original checkpoints and results remain untouched'})
    # Turner tables should stay fixed: MixedFold differentiates positional scores.
    actual = torch.load(models[0]['checkpoint'], map_location='cpu', weights_only=False)
    public = torch.load(models[1]['checkpoint'], map_location='cpu', weights_only=False)
    actual_state = actual['model_state_dict']
    public_state = public.get('model_state_dict', public)
    comparisons = []
    for key, a in actual_state.items():
        if key.startswith('turner.'):
            b = public_state[key]
            mask = torch.isfinite(a) & torch.isfinite(b)
            comparisons.append({'name': key, 'identical': torch.equal(a, b),
                                'max_abs_finite_delta': float((a[mask]-b[mask]).abs().max()) if mask.any() else 0.})
    groups = actual['optimizer_state_dict']['param_groups']
    base.write_json(OUT / 'weight_audit.json', {'turner': comparisons,
        'all_turner_tables_identical': all(x['identical'] for x in comparisons),
        'additional_scalars': {k: v.tolist() for k, v in actual_state.items() if 'fc_additional' in k},
        'optimizer_groups': [{k: v for k, v in g.items() if k != 'params'} for g in groups],
        'network_parameter_count': sum(v.numel() for k, v in actual_state.items() if not k.startswith('turner.'))})
    base.write_json(OUT / 'prepare-complete.json', {'success': True, 'n': len(samples),
                                                   'panel_sha256': base.sha(OUT / 'diagnostic_panel.jsonl')})
    print('Prepared', len(samples), 'diagnostic sequences', flush=True)


def edges(result):
    return [[i-1, int(j)-1] for i, j in enumerate(result[2][0]) if i > 0 and j > i]


def work(index):
    import torch
    from scripts.chemical_consistency.predict_mx import create_model, restore
    plan = json.loads((OUT / 'plan.json').read_text())
    if base.sha(OUT / 'diagnostic_panel.jsonl') != plan['panel_sha256']:
        raise RuntimeError('Diagnostic panel changed')
    model_index, shard = divmod(index-1, SHARDS)
    spec = plan['models'][model_index]
    if base.sha(spec['checkpoint']) != spec['checkpoint_sha256']:
        raise RuntimeError('Model checkpoint changed')
    torch.set_num_threads(4)
    torch.manual_seed(20261031); torch.cuda.manual_seed_all(20261031)
    model = create_model(spec['source'], spec['config'])
    restore(model, spec['checkpoint'])
    samples = sorted(base.read_jsonl(OUT / 'diagnostic_panel.jsonl'), key=lambda r: (-len(r['sequence']), r['id']))
    samples = [r for i, r in enumerate(samples) if i % SHARDS == shard]
    folder = OUT / 'predictions' / spec['model']; folder.mkdir(parents=True, exist_ok=True)
    path = folder / f'shard-{shard}.jsonl'
    previous = base.read_jsonl(path) if path.exists() else []
    done = {r['id'] for r in previous}
    w_t, w_n = model.score_weight_turner, model.score_weight_positional
    started = time.monotonic()
    with torch.inference_mode(), path.open('a') as handle:
        for item in samples:
            if item['id'] in done:
                continue
            sequence = item['sequence']; n = len(sequence)
            model.eval()
            model.score_weight_turner = model.count_weight_turner = w_t
            model.score_weight_positional = model.count_weight_positional = w_n
            result = model([sequence], return_param=True)
            native_pairs, parameters = edges(result), result[4]
            default = pair_metrics(item['pairs'], native_pairs)
            target = torch.zeros(n+1, dtype=torch.long)
            for a, b in item['train_pairs']:
                target[a+1], target[b+1] = b+1, a+1
            constrained = model([sequence], param=parameters, constraint=[target], max_internal_length=None)
            thermal = model.turner([sequence], constraint=[target], max_internal_length=None)
            ref_s, thermal_s = float(constrained[0][0]), float(thermal[0][0])
            record = {'id': item['id'], 'sample_set': item['sample_set'], 'length': n,
                'sequence_sha256': item['sequence_sha256'], 'predicted_pairs': native_pairs,
                'default': default, 'projected_target_f1': pair_metrics(item['train_pairs'], native_pairs)['f1'],
                'clean_gap_per_nt': (float(result[0][0])-ref_s)/n,
                'thermal_penalty_per_nt': .125*(ref_s-thermal_s)**2/n,
                'reference_score': ref_s, 'reference_turner_score': thermal_s,
                'reference_neural_score': (ref_s-w_t*thermal_s)/w_n,
                'constraint_exact': edges(constrained) == [list(p) for p in item['train_pairs']],
                'ablations': {}}
            if item['sample_set'] != 'bpRNA_train_sample':
                for alpha in (0., .25, .5, 2.):
                    model.score_weight_positional = model.count_weight_positional = w_n*alpha
                    prediction = edges(model([sequence], param=parameters))
                    record['ablations'][f'neural_alpha_{alpha:g}'] = {
                        **pair_metrics(item['pairs'], prediction), 'n_predicted_pairs': len(prediction),
                        'same_as_default': prediction == native_pairs}
                model.score_weight_positional = model.count_weight_positional = w_n
                if spec['model'] == 'mx_bprna10':
                    replacement = [{'turner': parameters[0]['turner'], 'positional': dict(parameters[0]['positional'])}]
                    for name in ('score_multi_base', 'score_multi_paired', 'score_external_paired'):
                        replacement[0]['positional'][name] = torch.zeros_like(replacement[0]['positional'][name])
                    prediction = edges(model([sequence], param=replacement))
                    record['ablations']['additional_off'] = {
                        **pair_metrics(item['pairs'], prediction), 'n_predicted_pairs': len(prediction),
                        'same_as_default': prediction == native_pairs}
            if item['sample_set'] == 'bpRNA_train_sample' and len(done) % 16 == 0:
                record['noise_dropout'] = []
                for mode in ('eval', 'train'):
                    model.train(mode == 'train')
                    noisy = model([sequence], perturb=.75, return_param=True)
                    ref = model([sequence], param=noisy[4], constraint=[target], max_internal_length=None)
                    gap = (float(noisy[0][0])-float(ref[0][0]))/n
                    penalty = .125*(float(ref[0][0])-thermal_s)**2/n
                    record['noise_dropout'].append({'mode': mode, 'gap_per_nt': gap,
                        'thermal_penalty_per_nt': penalty, 'loss_per_nt': gap+penalty})
                model.eval()
            handle.write(json.dumps(record, allow_nan=False)+'\n'); handle.flush(); done.add(item['id'])
            if len(done) % 25 == 0:
                print(spec['model'], shard, len(done), '/', len(samples), round(time.monotonic()-started, 1), flush=True)
    base.write_json(folder / f'shard-{shard}-complete.json', {'success': True, 'n': len(samples),
        'model': spec['model'], 'predictions_sha256': base.sha(path), 'seconds': time.monotonic()-started})
    print('Completed', spec['model'], shard, len(samples), flush=True)


if __name__ == '__main__':
    if sys.argv[1] == 'prepare':
        prepare()
    elif sys.argv[1] == 'work':
        work(int(os.environ.get('SGE_TASK_ID', '1')))
    else:
        raise ValueError(sys.argv[1])
