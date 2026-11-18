#!/usr/bin/env python3
"""Frozen bpRNAnew evaluation and train/test sequence-similarity audit."""
from __future__ import annotations

import argparse
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
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT / '.cache/bprna-benchmark-deps'))
DATA = ROOT / 'data/efold_hf/bprnanew_20261029'
OUT = ROOT / 'qsub_logs/efold_hf/bprnanew_20261029'
OLD = ROOT / 'qsub_logs/efold_hf/bprna100_ft5_replay20_20261024'
FT = ROOT / 'qsub_logs/efold_hf/bprna10_restored_ft5_replay20_20261026'
PAPER = ROOT / 'qsub_logs/efold_hf/paper_aligned_20261028'
RAW = ROOT / 'data/efold_hf/raw'
SHARDS = 4


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for b in iter(lambda: handle.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def read_jsonl(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with temp.open('w') as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False) + '\n')
    temp.replace(path)


def normalize(sequence):
    return sequence.upper().replace('T', 'U')


def bpseq(text, identifier):
    rows = [s.split() for s in text.splitlines() if s.strip() and not s.startswith('#')]
    sequence, partners = [], [0]
    for i, row in enumerate(rows, 1):
        if len(row) != 3 or int(row[0]) != i:
            raise ValueError('Invalid BPSEQ: ' + identifier)
        sequence.append(row[1]); partners.append(int(row[2]))
    sequence = normalize(''.join(sequence))
    if not sequence or set(sequence) - set('ACGUNRYMKSWBDHVX'):
        raise ValueError('Unexpected nucleotide in official bpRNAnew: ' + identifier)
    for i, j in enumerate(partners[1:], 1):
        if j < 0 or j > len(sequence) or j == i or (j and partners[j] != i):
            raise ValueError('Inconsistent base pairs: ' + identifier)
    pairs = [[i-1, j-1] for i, j in enumerate(partners) if i > 0 and j > i]
    return {'id': identifier, 'source': 'bpRNAnew', 'sequence': sequence, 'pairs': pairs,
            'sequence_sha256': hashlib.sha256(sequence.encode()).hexdigest()}


def pair_metrics(reference, prediction):
    ref = {tuple(map(int, p)) for p in reference}
    pred = {tuple(map(int, p)) for p in prediction}
    tp = len(ref & pred)
    return {'tp': tp, 'fp': len(pred)-tp, 'fn': len(ref)-tp,
            'precision': tp/len(pred) if pred else (1. if not ref else None),
            'recall': tp/len(ref) if ref else (1. if not pred else None),
            'f1': 2*tp/(len(ref)+len(pred)) if ref or pred else 1.}


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    archive = DATA / 'bpRNAnew.tar.gz'
    rows, local_matches = [], 0
    local = Path('/gs/bs/tga-satolab-gtex/yamauchi/data/bpRNAnew_dataset/bpRNAnew.nr500.canonicals')
    with tarfile.open(archive) as tar:
        for member in sorted(tar.getmembers(), key=lambda m: m.name):
            if not member.isfile() or not member.name.endswith('.bpseq'):
                continue
            content = tar.extractfile(member).read()
            item = bpseq(content.decode(), Path(member.name).stem)
            if len(item['sequence']) <= 1000:
                rows.append(item)
            other = local / Path(member.name).name
            if other.is_file() and other.read_bytes() == content:
                local_matches += 1
    if len(rows) != 5401 or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Official bpRNAnew coverage differs from expected 5401')
    write_jsonl(OUT / 'panel.jsonl', rows)
    bprna = read_jsonl(OLD / 'data/bprna.jsonl')
    train = [{'id': r['id'], 'sequence': r['sequence']} for r in bprna]
    if len(train) != 45507:
        raise ValueError('Actual bpRNA training cohort changed')
    write_jsonl(OUT / 'bprna_actual_training.jsonl', train)
    tests = [r for r in read_jsonl(PAPER / 'panel.jsonl') if r['source'] in ('PDB', 'archiveII')]
    legacy = {r['id'] for r in read_jsonl(OLD / 'data/test.jsonl')}
    for r in tests:
        r['in_original_conservative_panel'] = r['id'] in legacy
    write_jsonl(OUT / 'similarity_queries.jsonl', tests)
    # Keep the complete public bpRNA source separate from the training subset.
    public = json.loads((RAW / 'bpRNA-1m.json').read_text())
    public_rows = [{'id': key, 'sequence': normalize(item['sequence'])}
                   for key, item in public.items() if len(item['sequence']) <= 1000]
    write_jsonl(OUT / 'bprna_public_le1000.jsonl', public_rows)
    model_source = ROOT / 'data/efold_hf/paper_2026/author_efold_source'
    models = [{'model': 'author_eFold', 'label': '著者公開eFold', 'backend': 'efold',
               'source': str(model_source),
               'checkpoint': str(model_source / 'efold/resources/efold_weights.pt')}]
    models.append({'model': 'mx_bprna10', 'label': 'MXfold2 bpRNA事前学習10ep', 'backend': 'mx',
                   'source': str(OLD / 'source'), 'config': str(OLD / 'baseline.conf'),
                   'checkpoint': str(OLD / 'runs/pretrain/epoch-010.pt')})
    for arm, label in [('mw_rank_restored', 'MW Rank'), ('nll_restored', 'IMLE-NLL'),
                       ('shape_fy_restored', 'SHAPE-fy')]:
        models.append({'model': 'mx_' + arm, 'label': 'MXfold2 ' + label + ' FT5ep', 'backend': 'mx',
                       'source': str(FT / 'source'), 'config': str(FT / 'baseline.conf'),
                       'checkpoint': str(FT / f'runs/{arm}/epoch-005.pt')})
    models.append({'model': 'mx_assisted_original', 'label': 'MXfold2 assisted FT5ep（初回設定）',
                   'backend': 'mx', 'source': str(OLD / 'source'), 'config': str(OLD / 'baseline.conf'),
                   'checkpoint': str(OLD / 'runs/assisted/epoch-005.pt')})
    models.append({'model': 'MXfold2_public', 'label': '公開MXfold2 TrainSetAB', 'backend': 'mx',
                   'source': str(ROOT), 'config': str(ROOT / 'mxfold2/models/TrainSetAB.conf'),
                   'checkpoint': str(ROOT / 'mxfold2/models/TrainSetAB.pth')})
    for m in models:
        m['checkpoint_sha256'] = sha(m['checkpoint'])
        if 'config' in m:
            m['config_sha256'] = sha(m['config'])
    plan = {'dataset': 'MXfold2 v0.1.1 official bpRNAnew.nr500.canonicals',
            'dataset_url': 'https://github.com/mxfold/mxfold2/releases/download/v0.1.1/bpRNAnew.tar.gz',
            'archive_sha256': sha(archive), 'panel_sha256': sha(OUT / 'panel.jsonl'),
            'n': len(rows), 'unique_sequences': len({r['sequence'] for r in rows}),
            'length_range': [min(len(r['sequence']) for r in rows), max(len(r['sequence']) for r in rows)],
            'ambiguous_entries': [r['id'] for r in rows if set(r['sequence']) - set('ACGU')],
            'ambiguity_policy': 'Retain all 5401 entries; non-ACGU uses model unknown token (eFold X=0, MXfold2 native unknown); also report ACGU-only 5388 cohort',
            'local_files_identical_to_official': local_matches, 'max_length': 1000,
            'models': models, 'shards_per_model': SHARDS,
            'similarity_queries': dict(Counter(r['source'] for r in tests)),
            'actual_training_sequences': len(train), 'public_bprna_le1000': len(public_rows),
            'metric': 'exact-pair per-RNA precision/recall/F1 and micro F1; original reference retained; no tolerance or reference pruning',
            'selection': 'pre-specified pretrain10 and FT5; no checkpoint choice on bpRNAnew',
            'input_sha256': {str(p): sha(p) for p in [OLD / 'data/bprna.jsonl', PAPER / 'panel.jsonl',
                                                    RAW / 'bpRNA-1m.json', OLD / 'data/chemical.jsonl']}}
    write_json(OUT / 'plan.json', plan)
    # Distinguish family novelty relative to bpRNA from overlap with eFold's
    # newer composite training data, including RNAcentral pseudo-label sources.
    new_sequences = {r['sequence'] for r in rows}
    sources = {'actual_bpRNA': {r['sequence'] for r in train},
               'actual_chemical_FT': {r['sequence'] for r in read_jsonl(OLD / 'data/chemical.jsonl')}}
    full = json.loads((RAW / 'efold_train.json').read_text())
    sources['author_efold_train_public'] = {normalize(r['sequence']) for r in full.values()}
    overlap = {}
    for name, sequences in sources.items():
        match = [r['id'] for r in rows if r['sequence'] in sequences]
        overlap[name] = {'training_unique_sequences': len(sequences), 'test_entries_exact': len(match),
                         'test_unique_sequences_exact': len(new_sequences & sequences), 'ids': match}
    write_json(OUT / 'bprnanew_training_exact_overlap.json', overlap)
    write_json(OUT / 'prepare-complete.json', {'success': True, 'panel_sha256': plan['panel_sha256']})
    print(json.dumps({k: v for k, v in plan.items() if k not in ('models', 'input_sha256')}, indent=2), flush=True)


def checked_plan():
    marker = OUT / 'prepare-complete.json'
    if not marker.exists() or not json.loads(marker.read_text()).get('success'):
        raise RuntimeError('Preparation did not complete successfully')
    plan = json.loads((OUT / 'plan.json').read_text())
    if sha(OUT / 'panel.jsonl') != plan['panel_sha256']:
        raise ValueError('bpRNAnew panel changed')
    return plan


def work(task_index, smoke=False):
    import torch
    plan = checked_plan()
    model_index, shard = divmod(task_index-1, SHARDS)
    spec = plan['models'][model_index]
    if not smoke:
        marker = OUT / 'smoke' / spec['model'] / 'shard-0-complete.json'
        if not marker.exists() or not json.loads(marker.read_text()).get('success'):
            raise RuntimeError('Model smoke validation has not succeeded')
    if sha(spec['checkpoint']) != spec['checkpoint_sha256']:
        raise ValueError('Checkpoint changed')
    items = sorted(read_jsonl(OUT / 'panel.jsonl'), key=lambda r: (-len(r['sequence']), r['id']))
    items = [r for i, r in enumerate(items) if i % SHARDS == shard]
    if smoke:
        items = items[:1] + items[len(items)//2:len(items)//2+1] + items[-1:]
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required')
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if spec['backend'] == 'efold':
        sys.path[:0] = [str(ROOT / 'data/efold_hf/paper_2026/author_dependencies'), spec['source']]
        from efold.core import batch
        from efold.core.embeddings import sequence_to_int
        from efold.models import create_model
        model = create_model(model='efold', ntoken=5, d_model=64, c_z=32, d_cnn=64,
                             num_blocks=4, no_recycles=0, dropout=0, lr=3e-4,
                             weight_decay=0, gamma=0.995)
        model.load_state_dict(torch.load(spec['checkpoint'], map_location='cpu', weights_only=True), strict=True)
        model.eval().to('cuda')
        def predict(item):
            sequence = item['sequence']
            encoded = ''.join(nt if nt in 'ACGU' else 'X' for nt in sequence)
            data = batch.Batch(sequence=sequence_to_int(encoded).unsqueeze(0), reference=[item['id']],
                               length=[len(sequence)], L=len(sequence), use_error=False,
                               batch_size=1, data_types=['sequence'], dt_count={'sequence': 1}).to('cuda')
            matrix = model.postprocesser.run(model(data)['structure'], data.get('sequence'))[0]
            return torch.nonzero(torch.triu(matrix, diagonal=1), as_tuple=False).cpu().tolist()
    else:
        if sha(spec['config']) != spec['config_sha256']:
            raise ValueError('Configuration changed')
        from scripts.chemical_consistency.predict_mx import create_model, restore
        model = create_model(spec['source'], spec['config'])
        restore(model, spec['checkpoint'])
        def predict(item):
            partners = model([item['sequence']], return_partfunc=False)[2][0]
            return [[i-1, int(j)-1] for i, j in enumerate(partners) if i > 0 and j > i]
    folder = OUT / ('smoke' if smoke else 'predictions') / spec['model']
    path = folder / f'shard-{shard}.jsonl'
    folder.mkdir(parents=True, exist_ok=True)
    previous = read_jsonl(path) if path.exists() else []
    done = {r['id'] for r in previous}
    if len(done) != len(previous) or done - {r['id'] for r in items}:
        raise ValueError('Invalid cached predictions')
    start = time.monotonic()
    with torch.inference_mode(), path.open('a') as handle:
        for item in items:
            if item['id'] in done:
                continue
            pairs = predict(item)
            row = {'id': item['id'], 'sequence_sha256': item['sequence_sha256'],
                   'length': len(item['sequence']), 'predicted_pairs': pairs,
                   **pair_metrics(item['pairs'], pairs)}
            handle.write(json.dumps(row, allow_nan=False)+'\n'); handle.flush()
            done.add(item['id'])
            if len(done) % 100 == 0:
                print(spec['model'], shard, len(done), '/', len(items), f'{time.monotonic()-start:.1f}s', flush=True)
    if len(done) != len(items):
        raise RuntimeError('Incomplete prediction shard')
    write_json(folder / f'shard-{shard}-complete.json', {'success': True, 'n': len(items),
               'model': spec['model'], 'checkpoint_sha256': spec['checkpoint_sha256'],
               'predictions_sha256': sha(path), 'seconds': time.monotonic()-start})
    print('Completed', spec['model'], shard, len(items), flush=True)


def global_similarity(index):
    """Exhaustive nearest training sequence under normalized Levenshtein."""
    import numpy as np
    from rapidfuzz import process
    from rapidfuzz.distance import Levenshtein
    checked_plan()
    queries = read_jsonl(OUT / 'similarity_queries.jsonl')
    target = read_jsonl(OUT / 'bprna_actual_training.jsonl')
    # Four CPU array tasks per dataset split the exhaustive all-pairs search.
    selected = [q for i, q in enumerate(queries) if i % 8 == index-1]
    sequences = [q['sequence'] for q in target]
    result = []
    start = time.monotonic()
    for begin in range(0, len(selected), 16):
        batch = selected[begin:begin+16]
        similarities = process.cdist([q['sequence'] for q in batch], sequences,
                                     scorer=Levenshtein.normalized_similarity, dtype=np.float64, workers=4)
        for query, sims in zip(batch, similarities):
            best = int(np.argmax(sims)); score = float(sims[best])
            distance = Levenshtein.distance(query['sequence'], sequences[best])
            if abs(score - (1-distance/max(len(query['sequence']), len(sequences[best])))) > 1e-12:
                raise AssertionError('Normalized edit similarity mismatch')
            result.append({'id': query['id'], 'source': query['source'],
                           'length': len(query['sequence']), 'best_training_id': target[best]['id'],
                           'best_training_length': len(sequences[best]), 'edit_distance': distance,
                           'global_edit_similarity': score, 'ties_at_maximum': int((sims == score).sum()),
                           'in_original_conservative_panel': query['in_original_conservative_panel']})
        print('global shard', index, min(begin+16, len(selected)), '/', len(selected),
              f'{time.monotonic()-start:.1f}s', flush=True)
    write_jsonl(OUT / 'similarity' / f'global-shard-{index}.jsonl', result)
    write_json(OUT / 'similarity' / f'global-shard-{index}-complete.json',
               {'success': True, 'n': len(result), 'target_n': len(target),
                'target_sha256': sha(OUT / 'bprna_actual_training.jsonl'),
                'seconds': time.monotonic()-start})


def local_similarity():
    checked_plan()
    import subprocess
    folder = OUT / 'similarity/local'
    folder.mkdir(parents=True, exist_ok=True)
    queries = read_jsonl(OUT / 'similarity_queries.jsonl')
    targets = read_jsonl(OUT / 'bprna_actual_training.jsonl')
    for name, rows, prefix in [('queries', queries, 'Q'), ('targets', targets, 'T')]:
        with (folder / f'{name}.fasta').open('w') as handle:
            for i, row in enumerate(rows):
                handle.write(f'>{prefix}{i:06d}\n' + row['sequence'].replace('U','T') + '\n')
    binary = ROOT / '.cache/rnagym-tools/mmseqs/bin/mmseqs'
    fields = 'query,target,pident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits,qcov,tcov'
    command = [str(binary), 'easy-search', str(folder/'queries.fasta'), str(folder/'targets.fasta'),
               str(folder/'hits.tsv'), str(folder/'tmp'), '--search-type','3', '--min-seq-id','0.6',
               '-c','0.8','--cov-mode','2','--alignment-mode','3','--max-seqs','50000',
               '--threads','4','--strand','1','--mask','0','-e','1000','-s','7.5',
               '--format-output',fields]
    subprocess.run(command, check=True)
    write_json(folder/'complete.json',{'success':True,'command':command,'format':fields,
               'mmseqs_sha256':sha(binary),'hits_sha256':sha(folder/'hits.tsv'),
               'note':'Forward nucleotide local search; qcov>=0.8; report tcov separately. Heuristic search: no hit does not imply a zero global similarity.'})


def summarize():
    plan = checked_plan()
    panel = {r['id']: r for r in read_jsonl(OUT / 'panel.jsonl')}
    results = []
    external = OUT / 'external_models.json'
    models = plan['models'] + (json.loads(external.read_text()) if external.exists() else [])
    for spec in models:
        folder = OUT / 'predictions' / spec['model']
        if not all((folder / f'shard-{i}-complete.json').exists() for i in range(SHARDS)):
            continue
        rows = []
        for shard in range(SHARDS):
            path = folder / f'shard-{shard}.jsonl'
            marker = json.loads((folder / f'shard-{shard}-complete.json').read_text())
            if not marker['success'] or sha(path) != marker['predictions_sha256']:
                raise ValueError('Prediction shard changed')
            rows.extend(read_jsonl(path))
        if len(rows) != len(panel) or {r['id'] for r in rows} != set(panel):
            raise ValueError('Not the same complete evaluation panel')
        for row in rows:
            if any(row[k] != v for k, v in pair_metrics(panel[row['id']]['pairs'], row['predicted_pairs']).items()):
                raise ValueError('Stored structural metric differs from reference')
        for cohort in ('all_5401', 'ACGU_only_5388'):
            subset = rows if cohort == 'all_5401' else [r for r in rows if not set(panel[r['id']]['sequence'])-set('ACGU')]
            record = {'model': spec['model'], 'label': spec['label'], 'cohort': cohort, 'n': len(subset)}
            for metric in ('precision', 'recall', 'f1'):
                values = [r[metric] for r in subset if r[metric] is not None]
                record[metric] = statistics.fmean(values) if values else None
                record[metric + '_defined'] = len(values)
            tp, fp, fn = (sum(r[k] for r in subset) for k in ('tp', 'fp', 'fn'))
            record['micro_f1'] = 2*tp/(2*tp+fp+fn) if tp+fp+fn else 1.
            results.append(record)
    write_json(OUT / 'results.json', {'results': results, 'n_expected': len(panel),
                                     'models_expected': len(models)})
    print(json.dumps(results, indent=2, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'predict', 'smoke', 'global', 'local', 'summarize'])
    parser.add_argument('--index', type=int)
    args = parser.parse_args()
    raw_index = os.environ.get('SGE_TASK_ID', '1')
    index = args.index or (int(raw_index) if raw_index.isdigit() else 1)
    if args.action == 'prepare': prepare()
    elif args.action in ('predict', 'smoke'): work(index, args.action == 'smoke')
    elif args.action == 'global': global_similarity(index)
    elif args.action == 'local': local_similarity()
    else: summarize()


if __name__ == '__main__':
    main()
