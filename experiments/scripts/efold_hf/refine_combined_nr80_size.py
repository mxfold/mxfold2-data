#!/usr/bin/env python3
"""Complete protected NR80 filtering in both native CD-HIT directions.

CD-HIT-EST-2D can miss a boundary match when db2 is longer (see the saved
80 nt / 100 nt, 80% identity fixture). Reverse the databases, use native
short-query/long-reference matching, and remove only RNAcentral references.
Repeat because each experimental query reports only one matching reference.
"""
import argparse
from collections import Counter
import csv
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import combined_nr80_size as audit


def fasta(path):
    result, identifier, sequence = {}, None, []
    with Path(path).open() as handle:
        for line in handle:
            if line.startswith('>'):
                if identifier is not None:
                    result[identifier] = ''.join(sequence)
                identifier, sequence = line[1:].strip(), []
            else:
                sequence.append(line.strip())
    if identifier is not None:
        result[identifier] = ''.join(sequence)
    return result


def write_fasta(path, tokens, sequences):
    with Path(path).open('w') as handle:
        for token in sorted(tokens):
            handle.write(f'>{token}\n{sequences[token]}\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    args = parser.parse_args()
    original = args.input.resolve()
    if (original / 'summary.json').exists():
        initial = json.loads((original / 'summary.json').read_text())
    else:
        # The completed forward filter is sufficient; its preliminary internal
        # clustering can finish independently while the reverse filter runs.
        cross = json.loads((original / 'cross_filter_summary.json').read_text())
        exp = json.loads((audit.RUN / 'bprna_trainab_nr80_20/clustering/summary.json').read_text())
        central = json.loads((audit.RUN / 'bprna_rnacentral10/dataset.json').read_text())
        if cross['audited_matches_n'] != cross['removed_n']:
            raise ValueError('Forward match audit incomplete')
        initial = {'success': True, 'total_n': None,
            'input_hashes': json.loads((original / 'input_hashes.json').read_text()),
            'experimental_input_n': exp['input_n'],
            'experimental_exact_unique_n': exp['exact_unique_n'],
            'experimental_retained_n': exp['retained_n'],
            'rnacentral_efold_pool_n': central['rnacentral_efold_retained_n'],
            'rnacentral_previously_exact_removed_n': central['exact_duplicates_removed'],
            'rnacentral_candidates_n': cross['input_n'],
            'identity': .8, 'coverage_short': .8, 'coverage_long': .8, 'length_ratio': .8,
            'max_length': 1000, 'strand': 'forward only; U->T for clustering only',
            'cross_execution': cross['execution'],
            'binary_sha256': {n: audit.sha(audit.CDHIT / n)
                              for n in ('cd-hit-est', 'cd-hit-est-2d')},
            'priority': 'Keep existing experimental NR80 representatives; remove RNAcentral homologs of ALL original experimental sequences before internal RNAcentral clustering.',
            'algorithm': 'CD-HIT-EST 4.8.1 greedy clustering; native integer rounding and heuristic alignment',
            'labels': 'Preserve original pairs and training pairs; supplied RNAstructure pseudo-labels never replace experimental labels.'}
    if not initial['success']:
        raise ValueError('Initial forward pass incomplete')
    output = original / 'priority_refined'
    output.mkdir(exist_ok=True)
    if (output / 'summary.json').exists():
        raise ValueError('Refinement output already exists; inspect before rerunning')
    started = time.time()
    for path, checksum in initial['input_hashes'].items():
        if audit.sha(path) != checksum:
            raise ValueError('Frozen source changed: ' + path)
    protected_path = original / 'protected_all_experimental.fasta'
    protected = fasta(protected_path)
    sequences = {**protected, **fasta(original / 'rnacentral_candidates.fasta')}
    central_tokens = {t for t in sequences if t.startswith('C')}
    central_input = audit.RUN / 'bprna_rnacentral10/data/train.jsonl'
    central_ids = {}
    for row in audit.rows(central_input):
        if row['source'] == 'rnacentral_synthetic':
            central_ids[f'C{len(central_ids):07d}'] = row['id']
    if set(central_ids) != central_tokens:
        raise ValueError('RNAcentral tokens changed')
    remaining_path = original / 'rnacentral_without_experimental_homologs.fasta'
    remaining = audit.fasta_ids(remaining_path)
    cross_mapping = audit.parse_clusters(str(remaining_path) + '.clstr')
    if set(cross_mapping) != set(protected) | (central_tokens - remaining):
        raise ValueError('Initial cross mapping incomplete')
    # Recover the original IDs for all S tokens from the identical frozen order.
    inputs = [audit.PROJECT / 'qsub_logs/efold_hf/bprna100_ft5_replay20_20261024/data/bprna.jsonl',
              audit.PROJECT / 'qsub_logs/efold_hf/pretrain_diagnosis_20261031/official_TrainSetAB.jsonl']
    raw = [r for path in inputs for r in audit.rows(path) if len(r['sequence']) <= 1000]
    priority = {'TrainSetA': 0, 'TrainSetB': 1, 'bpRNA-1m': 2}
    unique = {}
    for row in sorted(raw, key=lambda r: (priority[r['source']], r['id'])):
        unique.setdefault(row['sequence'], row)
    protected_ids = {f'S{i:07d}': r['id'] for i, r in enumerate(unique.values())
                     if len(r['sequence']) >= 4}
    if set(protected_ids) != set(protected):
        raise ValueError('Experimental token mapping changed')
    reverse_runs, extra_removed = [], 0
    active_queries = set(protected)
    for iteration in range(1, 1001):
        folder = output / f'reverse-{iteration:03d}'
        folder.mkdir(exist_ok=True)
        unmatched_experimental = folder / 'unmatched_experimental.fasta'
        cached = folder / 'summary.json'
        if cached.exists():
            previous = json.loads(cached.read_text())
            execution = previous['execution']
            command = execution['command']
            query_path = Path(command[command.index('-i2')+1])
            query_tokens = audit.fasta_ids(query_path)
            if not active_queries <= query_tokens <= set(protected):
                raise ValueError('Cached pass does not cover the active queries')
            print('Reusing completed reverse iteration', iteration, flush=True)
        else:
            query_tokens = active_queries
            query_path = folder / 'experimental_queries.fasta'
            write_fasta(query_path, query_tokens, sequences)
            execution = audit.execute([str(audit.CDHIT / 'cd-hit-est-2d'),
                '-i', str(remaining_path), '-i2', str(query_path),
                '-o', str(unmatched_experimental), *audit.PARAMETERS], folder / 'run.log')
        mapping = audit.parse_clusters(str(unmatched_experimental) + '.clstr')
        unmatched = audit.fasta_ids(unmatched_experimental)
        if set(mapping) != remaining | (query_tokens-unmatched):
            raise ValueError('Reverse match mapping incomplete')
        audit.audit_matches(mapping, sequences)
        removed, matched_queries = set(), set()
        for token, (reference, alignment) in mapping.items():
            if not token.startswith('S'):
                continue
            if not reference.startswith('C') or reference not in remaining:
                raise ValueError('Unexpected reverse reference')
            removed.add(reference)
            matched_queries.add(token)
            if reference not in cross_mapping:
                match = re.search(r'at (\d+):(\d+):(\d+):(\d+)/\+/([\d.]+)%', alignment)
                a, b, c, d, identity = match.groups()
                inverse = (f'{len(sequences[reference])}nt, >{reference}... '
                           f'at {c}:{d}:{a}:{b}/+/{identity}%')
                cross_mapping[reference] = (token, inverse)
        if not removed <= remaining:
            raise ValueError('Attempted removal outside RNAcentral pool')
        remaining -= removed
        extra_removed += len(removed)
        result = {'iteration': iteration, 'rnacentral_removed_n': len(removed),
            'rnacentral_remaining_n': len(remaining), 'experimental_query_n': len(query_tokens),
            'execution': execution}
        if cached.exists() and (previous['rnacentral_removed_n'] != len(removed)
                or previous['rnacentral_remaining_n'] != len(remaining)):
            raise ValueError('Cached reverse pass count changed')
        reverse_runs.append(result)
        audit.write_json(folder / 'summary.json', result)
        audit.write_json(output / 'progress.json', {'stage': 'reverse_cross_filter',
            'iteration': iteration, 'additional_rnacentral_removed': extra_removed,
            'rnacentral_remaining_n': len(remaining)})
        print('Reverse iteration', iteration, 'removed', len(removed),
              'RNAcentral; remaining', len(remaining), flush=True)
        if not removed:
            if query_tokens == set(protected):
                break
            # A final full scan checks that reducing the active query set did
            # not hide a match because of a native heuristic/context effect.
            active_queries = set(protected)
            continue
        # Only queries with a hit can have another previously masked hit.
        # Still verify ALL experimental sequences once more before stopping.
        active_queries = matched_queries
        remaining_path = folder / 'remaining_rnacentral.fasta'
        write_fasta(remaining_path, remaining, sequences)
    else:
        raise ValueError('Reverse filter did not converge')
    if set(cross_mapping) != set(protected) | (central_tokens - remaining):
        raise ValueError('Final experimental removal provenance incomplete')
    audit_n = audit.audit_matches(cross_mapping, sequences)
    representatives_path = output / 'rnacentral_nr80.fasta'
    within_parameters = list(audit.PARAMETERS)
    # CD-HIT's documented -g0/-g1 modes select the same representatives;
    # -g0 stops at the first eligible hit instead of searching for the best.
    within_parameters[within_parameters.index('-g')+1] = '0'
    within_execution = audit.execute([str(audit.CDHIT / 'cd-hit-est'),
        '-i', str(remaining_path), '-o', str(representatives_path), *within_parameters],
        output / 'within_rnacentral.log')
    within_mapping = audit.parse_clusters(str(representatives_path) + '.clstr')
    retained = audit.fasta_ids(representatives_path)
    if set(within_mapping) != remaining or retained != {r for r, _ in within_mapping.values()}:
        raise ValueError('Final internal clustering membership incomplete')
    within_audit_n = audit.audit_matches(within_mapping, sequences)
    retained_ids = {central_ids[t] for t in retained}
    experimental_train = audit.RUN / 'bprna_trainab_nr80_20/data/train.jsonl'
    train = output / 'train.jsonl'
    counts, ids, seen_sequences = Counter(), set(), set()
    with train.open('w') as handle:
        for path in (experimental_train, central_input):
            for row in audit.rows(path):
                if path == central_input and row['id'] not in retained_ids:
                    continue
                if row['id'] in ids or row['sequence'] in seen_sequences:
                    raise ValueError('Duplicate output ID or sequence')
                ids.add(row['id'])
                seen_sequences.add(row['sequence'])
                counts[row['source']] += 1
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    if sum(counts.values()) != initial['experimental_retained_n'] + len(retained):
        raise ValueError('Final merged count mismatch')
    with (output / 'rnacentral_membership.tsv').open('w') as handle:
        writer = csv.writer(handle, delimiter='\t')
        writer.writerow(['id', 'status', 'matched_or_retained_id', 'alignment'])
        for token, identifier in central_ids.items():
            if token in cross_mapping:
                reference, alignment = cross_mapping[token]
                writer.writerow([identifier, 'removed_experimental_homolog', protected_ids[reference], alignment])
            else:
                reference, alignment = within_mapping[token]
                writer.writerow([identifier, 'retained' if token == reference else 'removed_rnacentral_homolog',
                                 central_ids[reference], alignment])
    for path, checksum in initial['input_hashes'].items():
        if audit.sha(path) != checksum:
            raise ValueError('Input changed during refinement: ' + path)
    if (original / 'summary.json').exists():
        initial['total_n'] = json.loads((original / 'summary.json').read_text())['total_n']
    result = {**initial, 'job_id': os.environ.get('JOB_ID'), 'seconds': time.time()-started,
        'forward_only_preliminary_total_n': initial['total_n'],
        'forward_only_summary': str(original / ('summary.json' if (original / 'summary.json').exists()
                                              else 'cross_filter_summary.json')),
        'reverse_extra_rnacentral_removed_n': extra_removed,
        'reverse_passes_n': len(reverse_runs), 'reverse_runs': reverse_runs,
        'rnacentral_removed_experimental_homolog_n': len(central_tokens)-len(remaining),
        'rnacentral_after_experimental_filter_n': len(remaining),
        'rnacentral_removed_internal_homolog_n': len(remaining)-len(retained),
        'rnacentral_retained_n': len(retained), 'total_n': sum(counts.values()),
        'total_nt': sum(map(len, seen_sequences)), 'sources_retained': dict(counts),
        'cross_audited_matches_n': audit_n, 'within_audited_matches_n': within_audit_n,
        'within_execution': within_execution, 'train_jsonl': str(train),
        'train_sha256': audit.sha(train),
        'reverse_policy': 'Repeat native default short-query/long-reference matching with RNAcentral in db1 and experimental sequences in db2. Remove only matching RNAcentral references, never experimental queries. Requery matched experimental sequences, then require a final full scan of ALL original experimental sequences with zero matches.',
        'within_mode': 'Native -g0 first eligible hit; CD-HIT documents identical representatives to -g1. Final combined verification uses -g1.',
        'boundary_regression_fixture': str(original / 'boundary_check/result.json'),
        'new_training_submitted': False}
    audit.write_json(output / 'summary.json', result)
    print('Final data size before independent verification:', result['total_n'], flush=True)
    subprocess.run([sys.executable, str(audit.PROJECT / 'scripts/efold_hf/verify_combined_nr80_size.py'),
                    '--output', str(output)], check=True, timeout=3600)
    audit.write_json(output / 'complete.json', {'success': True, 'total_n': result['total_n'],
        'sources_retained': dict(counts), 'combined_native_verification_passed': True})
    audit.write_json(output / 'progress.json', {'stage': 'complete', 'total_n': result['total_n']})


if __name__ == '__main__':
    main()
