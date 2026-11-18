#!/usr/bin/env python3
"""Count a protected experimental NR80 set plus lower-priority RNAcentral.

Run on a CPU compute node; leave the three existing training runs untouched.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import re
import subprocess
import time

PROJECT = Path(__file__).resolve().parents[2]
RUN = PROJECT / 'qsub_logs/efold_hf/pretrain_variants_20261031'
CDHIT = PROJECT / '.cache/cdhit-4.8.1/cd-hit-v4.8.1-2019-0228'
PARAMETERS = ['-c', '.8', '-n', '4', '-G', '1', '-g', '1', '-s', '.8',
              '-aS', '.8', '-aL', '.8', '-r', '0', '-l', '3', '-d', '0',
              '-p', '1', '-T', '8', '-M', '64000']


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def rows(path):
    with Path(path).open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_json(path, value):
    temporary = Path(str(path) + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def fasta_ids(path):
    with Path(path).open() as handle:
        return {line[1:].strip() for line in handle if line.startswith('>')}


def parse_clusters(path):
    result, members = {}, []

    def finish():
        if not members:
            return
        representatives = [token for token, is_rep, _ in members if is_rep]
        if len(representatives) != 1:
            raise ValueError('Invalid representative count: ' + str(path))
        representative = representatives[0]
        for token, _, raw in members:
            if token in result:
                raise ValueError('Repeated cluster token: ' + token)
            result[token] = (representative, raw)

    with Path(path).open() as handle:
        for line in handle:
            if line.startswith('>Cluster'):
                finish()
                members = []
            else:
                match = re.search(r'>([SC]\d+)\.\.\.', line)
                if not match:
                    raise ValueError('Unexpected cluster line: ' + line)
                members.append((match[1], line.rstrip().endswith('*'), line.strip()))
    finish()
    return result


def execute(command, log, timeout=20 * 3600):
    start = time.time()
    print('Running', ' '.join(command), flush=True)
    with Path(log).open('w') as handle:
        subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT,
                       check=True, timeout=timeout)
    return {'command': command, 'seconds': time.time() - start,
            'log': str(log)}


def synthetic_check(output):
    """Prove that a longer pseudo-label cannot displace a real reference."""
    folder = output / 'priority_check'
    folder.mkdir(exist_ok=True)
    rng = random.Random(20261031)
    reference = ''.join(rng.choices('ACGT', k=80))
    independent = ''.join(rng.choices('ACGT', k=80))
    reverse = reference.translate(str.maketrans('ACGT', 'TGCA'))[::-1]
    variants = [reference, 'ACGT' + reference + 'TGCA', reference[4:76],
                'ACGTACGTAC' + reference + 'TGCATGCATG',
                'A' * 15 + reference + 'C' * 15, independent, reverse]
    protected = folder / 'protected.fasta'
    candidates = folder / 'candidates.fasta'
    unmatched = folder / 'unmatched.fasta'
    protected.write_text('>S0000000\n' + reference + '\n')
    candidates.write_text(''.join(f'>C{i:07d}\n{sequence}\n'
                                  for i, sequence in enumerate(variants)))
    execution = execute([str(CDHIT / 'cd-hit-est-2d'), '-i', str(protected),
                         '-i2', str(candidates), '-o', str(unmatched),
                         *PARAMETERS, '-s2', '.8'], folder / 'run.log', 120)
    expected = {f'C{i:07d}' for i in (4, 5, 6)}
    observed = fasta_ids(unmatched)
    if observed != expected:
        raise ValueError(f'Priority/length/orientation check failed: {observed} != {expected}')
    write_json(folder / 'result.json', {'success': True,
        'protected_reference_length': 80,
        'removed_candidate_lengths': [80, 88, 72, 100],
        'retained_controls': ['length ratio below .8', 'unrelated', 'reverse complement'],
        'execution': execution})


def audit_matches(mapping, sequences):
    checked = 0
    for token, (representative, raw) in mapping.items():
        if token == representative:
            continue
        length, reference_length = len(sequences[token]), len(sequences[representative])
        if min(length, reference_length) < int(.8 * max(length, reference_length)):
            raise ValueError('Length ratio violation: ' + raw)
        alignment = re.search(r'at (\d+):(\d+):(\d+):(\d+)/\+/([\d.]+)%', raw)
        if not alignment:
            raise ValueError('Missing forward alignment: ' + raw)
        a, b, c, d = map(int, alignment.groups()[:4])
        identity = float(alignment.group(5))
        if (abs(b-a)+1 < int(.8 * length)
                or abs(d-c)+1 < int(.8 * reference_length) or identity < 79.99):
            raise ValueError('Alignment cutoff violation: ' + raw)
        checked += 1
    return checked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    if (output / 'summary.json').exists():
        print((output / 'summary.json').read_text(), flush=True)
        return
    experimental_folder = RUN / 'bprna_trainab_nr80_20'
    experimental_train = experimental_folder / 'data/train.jsonl'
    protected_fasta = experimental_folder / 'clustering/input.fasta'
    central_input = RUN / 'bprna_rnacentral10/data/train.jsonl'
    bp_input = PROJECT / 'qsub_logs/efold_hf/bprna100_ft5_replay20_20261024/data/bprna.jsonl'
    trainab_input = PROJECT / 'qsub_logs/efold_hf/pretrain_diagnosis_20261031/official_TrainSetAB.jsonl'
    input_paths = [experimental_train, protected_fasta, central_input, bp_input, trainab_input]
    input_hashes = {str(path): sha(path) for path in input_paths}
    write_json(output / 'input_hashes.json', input_hashes)
    if not (CDHIT / 'cd-hit-est-2d').exists():
        execute(['make', '-j8', 'cd-hit-est-2d'], output / 'build.log', 600)
    synthetic_check(output)

    # Reconstruct the original S tokens; all original experimental sequences
    # protect against pseudo-labels, including members dropped by greedy NR80.
    experimental = [r for path in (bp_input, trainab_input) for r in rows(path)
                    if len(r['sequence']) <= 1000]
    priority = {'TrainSetA': 0, 'TrainSetB': 1, 'bpRNA-1m': 2}
    unique = {}
    for row in sorted(experimental, key=lambda r: (priority[r['source']], r['id'])):
        unique.setdefault(row['sequence'], row)
    protected = {f'S{i:07d}': row for i, row in enumerate(unique.values())
                 if len(row['sequence']) >= 4}
    reconstructed = output / 'protected_all_experimental.fasta'
    with reconstructed.open('w') as handle:
        for token, row in protected.items():
            handle.write(f'>{token}\n' + row['sequence'].replace('U', 'T') + '\n')
    if sha(reconstructed) != sha(protected_fasta):
        raise ValueError('Existing protected experimental FASTA differs from reconstruction')
    representatives = list(rows(experimental_train))
    if len(representatives) != 14831 or len(experimental) != 49767:
        raise ValueError('Frozen experimental counts changed')
    if any(r['sequence'] not in unique for r in representatives):
        raise ValueError('Unrecognized experimental representative')

    candidates = output / 'rnacentral_candidates.fasta'
    central, sequences = {}, {token: row['sequence'] for token, row in protected.items()}
    with candidates.open('w') as handle:
        for row in rows(central_input):
            if row['source'] != 'rnacentral_synthetic':
                continue
            token = f'C{len(central):07d}'
            sequence = row['sequence']
            if not 4 <= len(sequence) <= 1000 or set(sequence) - set('ACGU'):
                raise ValueError('Unexpected RNAcentral length/alphabet')
            central[token] = row['id']
            sequences[token] = sequence
            handle.write(f'>{token}\n' + sequence.replace('U', 'T') + '\n')
    if len(central) != 210260:
        raise ValueError('Frozen RNAcentral candidate count changed')
    write_json(output / 'progress.json', {'stage': 'cross_filter', 'central_candidates': len(central),
        'protected_experimental_unique': len(unique), 'experimental_retained': len(representatives)})

    unmatched = output / 'rnacentral_without_experimental_homologs.fasta'
    cross_execution = execute([str(CDHIT / 'cd-hit-est-2d'), '-i', str(reconstructed),
        '-i2', str(candidates), '-o', str(unmatched), *PARAMETERS, '-s2', '.8'],
        output / 'cross_filter.log')
    remaining = fasta_ids(unmatched)
    cross_mapping = parse_clusters(str(unmatched) + '.clstr')
    if set(cross_mapping) != set(protected) | (set(central) - remaining):
        raise ValueError('Cross-filter membership is incomplete')
    if any(not cross_mapping[t][0].startswith('S') for t in cross_mapping):
        raise ValueError('Pseudo-label displaced an experimental reference')
    cross_audit_n = audit_matches(cross_mapping, sequences)
    write_json(output / 'cross_filter_summary.json', {'input_n': len(central),
        'removed_n': len(central)-len(remaining), 'retained_n': len(remaining),
        'audited_matches_n': cross_audit_n, 'execution': cross_execution})
    print('RNAcentral after protecting experimental data:', len(remaining), flush=True)
    write_json(output / 'progress.json', {'stage': 'within_rnacentral_clustering',
        'central_after_cross_filter': len(remaining)})

    central_representatives = output / 'rnacentral_nr80.fasta'
    within_execution = execute([str(CDHIT / 'cd-hit-est'), '-i', str(unmatched),
        '-o', str(central_representatives), *PARAMETERS], output / 'within_rnacentral.log')
    within_mapping = parse_clusters(str(central_representatives) + '.clstr')
    retained = fasta_ids(central_representatives)
    if set(within_mapping) != remaining or retained != {r for r, _ in within_mapping.values()}:
        raise ValueError('Within-RNAcentral membership is incomplete')
    within_audit_n = audit_matches(within_mapping, sequences)
    retained_ids = {central[t] for t in retained}
    train = output / 'train.jsonl'
    source_counts = Counter()
    length_sum = 0
    with train.open('w') as handle:
        for row in representatives:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
            source_counts[row['source']] += 1
            length_sum += len(row['sequence'])
        for row in rows(central_input):
            if row['id'] in retained_ids:
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')
                source_counts[row['source']] += 1
                length_sum += len(row['sequence'])
    if sum(source_counts.values()) != len(representatives) + len(retained):
        raise ValueError('Output count mismatch')
    with (output / 'rnacentral_membership.tsv').open('w') as handle:
        writer = csv.writer(handle, delimiter='\t')
        writer.writerow(['id', 'status', 'matched_or_retained_id', 'alignment'])
        for token, identifier in central.items():
            if token in cross_mapping:
                reference, alignment = cross_mapping[token]
                writer.writerow([identifier, 'removed_experimental_homolog', protected[reference]['id'], alignment])
            else:
                reference, alignment = within_mapping[token]
                writer.writerow([identifier, 'retained' if token == reference else 'removed_rnacentral_homolog',
                                 central[reference], alignment])
    if any(sha(path) != checksum for path, checksum in input_hashes.items()):
        raise ValueError('A frozen input changed during computation')
    central_metadata = json.loads((RUN / 'bprna_rnacentral10/dataset.json').read_text())
    summary = {
        'success': True, 'job_id': os.environ.get('JOB_ID'), 'seconds': time.time()-started,
        'experimental_input_n': len(experimental), 'experimental_exact_unique_n': len(unique),
        'experimental_retained_n': len(representatives),
        'rnacentral_efold_pool_n': central_metadata['rnacentral_efold_retained_n'],
        'rnacentral_previously_exact_removed_n': central_metadata['exact_duplicates_removed'],
        'rnacentral_candidates_n': len(central),
        'rnacentral_removed_experimental_homolog_n': len(central)-len(remaining),
        'rnacentral_after_experimental_filter_n': len(remaining),
        'rnacentral_removed_internal_homolog_n': len(remaining)-len(retained),
        'rnacentral_retained_n': len(retained), 'total_n': sum(source_counts.values()),
        'sources_retained': dict(source_counts), 'total_nt': length_sum,
        'identity': .8, 'coverage_short': .8, 'coverage_long': .8, 'length_ratio': .8,
        'max_length': 1000, 'strand': 'forward only; U->T for clustering only',
        'priority': 'Keep existing experimental NR80 representatives. Remove RNAcentral homologs of ALL original experimental sequences, then cluster remaining RNAcentral internally.',
        'algorithm': 'CD-HIT-EST 4.8.1 greedy clustering (native integer rounding and heuristic alignment)',
        'labels': 'Preserve original pairs and training pairs; RNAcentral supplied RNAstructure pseudo-labels never replace experimental labels.',
        'cross_audited_matches_n': cross_audit_n, 'within_audited_matches_n': within_audit_n,
        'cross_execution': cross_execution, 'within_execution': within_execution,
        'binary_sha256': {name: sha(CDHIT / name) for name in ('cd-hit-est', 'cd-hit-est-2d')},
        'train_jsonl': str(train), 'train_sha256': sha(train), 'input_hashes': input_hashes,
        'new_training_submitted': False,
    }
    write_json(output / 'summary.json', summary)
    write_json(output / 'progress.json', {'stage': 'complete', 'total_n': summary['total_n']})
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    # Build only the missing 2D target, without rebuilding the running jobs' binary.
    os.chdir(CDHIT)
    main()
