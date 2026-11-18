"""Select RNAgym representatives by measured-core identity.

The result indexes the original packed sequence groups. Their full-length
sequences and aligned observations stay together when used for training.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pyarrow.parquet as pq

from mxfold2.rnagym import MODIFIERS
from scripts.default_model import prepare
from scripts.efold_hf import workflow as wf
from scripts.rnagym.make_rnagym_test_subset import read_fasta


def measured_span(full_sequence, values, offsets, first, end):
    span = values[offsets[first]:offsets[end]].reshape(-1, len(full_sequence))
    positions = np.flatnonzero((np.isfinite(span) & (span > -900)).any(axis=0))
    if not positions.size:
        raise ValueError('No measured bases for RNAgym core fallback')
    return full_sequence[positions[0]:positions[-1]+1]


def measured_cores(root, project, sequences, sequence_offsets):
    folder = project / prepare.RNAGYM
    index = {seq: i for i, seq in enumerate(sequences)}
    cores = [''] * len(sequences)
    snr = np.zeros(len(sequences))
    seen = np.zeros(len(sequences), dtype=np.int32)
    used = Counter()
    alternate_cores = set()
    malformed = set()
    inputs = {}
    plan = wf.plan_for(root)
    for split in prepare.RNAGYM_SPLITS:
        fasta = folder / f'RMDB_s40_{split}.fasta'
        parquet = folder / f'RMDB_s40_{split}.parquet'
        for path in (fasta, parquet):
            inputs[str(path.relative_to(project))] = wf.digest(path)
        parquet_key = str(parquet.relative_to(project))
        if inputs[parquet_key] != plan['input_sha256'][parquet_key]:
            raise RuntimeError(f'RNAgym input differs from packing: {parquet}')
        by_alias = {}
        for header, core in read_fasta(fasta):
            for alias in header.split('+'):
                if by_alias.setdefault(alias, core) != core:
                    raise ValueError(f'Conflicting FASTA cores for {alias}')
        file = pq.ParquetFile(parquet)
        for batch in file.iter_batches(columns=['seqID', 'sequence', 'modifier', 'SNR']):
            for alias, full, probe, signal in zip(*(col.to_pylist() for col in batch.columns), strict=True):
                if probe not in MODIFIERS:
                    continue
                i = index[wf.normalize(full)]
                seen[i] += 1
                used[split] += 1
                if signal is not None and np.isfinite(signal):
                    snr[i] = max(snr[i], signal)
                core = by_alias[alias]
                if core not in sequences[i]:
                    malformed.add(i)
                    continue
                previous = cores[i]
                if previous and previous != core:
                    alternate_cores.add(i)
                if (len(core), core) > (len(previous), previous):
                    cores[i] = core
        print(f'{split} aligned profiles: {used[split]}', flush=True)
    if not np.array_equal(seen, np.diff(sequence_offsets)):
        raise RuntimeError('Raw RNAgym observations disagree with packed sequence offsets')

    values = np.load(root / 'data/rnagym/values.npy', mmap_mode='r')
    offsets = np.load(root / 'data/rnagym/observation_offsets.npy', mmap_mode='r')
    missing = {i for i, core in enumerate(cores) if not core}
    if missing - malformed:
        raise RuntimeError('Missing RNAgym measured cores')
    for i in missing:
        cores[i] = measured_span(sequences[i], values, offsets, sequence_offsets[i], sequence_offsets[i+1])
    details = {'sources': inputs, 'profile_rows_by_split': dict(used),
               'sequences_with_multiple_valid_cores': len(alternate_cores),
               'sequences_with_malformed_fasta_core': sorted(malformed),
               'sequences_using_profile_span': sorted(missing)}
    return cores, snr, details


def collapse_cores(cores, snr, probe_counts):
    best = {}
    for i, core in enumerate(cores):
        priority = (int(probe_counts[i]), float(snr[i]), -i)
        if core not in best or priority > best[core][0]:
            best[core] = (priority, i)
    return {core: i for core, (_, i) in best.items()}


def run(root, output, threads):
    if not 1 <= threads <= 64:
        raise ValueError('CD-HIT threads must be between 1 and 64')
    output.mkdir(parents=True, exist_ok=True)
    packed = root / 'data/rnagym'
    data_manifest = json.loads((root / 'data-complete.json').read_text())
    for name, expected in data_manifest['sha256'].items():
        if name.startswith('data/rnagym/') and wf.digest(root / name) != expected:
            raise RuntimeError(f'Packed data changed: {name}')
    project = Path(wf.plan_for(root)['project'])
    with (packed / 'sequences.tsv').open() as handle:
        sequences = []
        for i, row in enumerate(csv.DictReader(handle, delimiter='\t')):
            if int(row['index']) != i:
                raise ValueError('Unexpected packed sequence indices')
            sequences.append(row['sequence'])
    offsets = np.load(packed / 'sequence_offsets.npy')
    modifiers = np.load(packed / 'modifiers.npy')
    probes = np.bitwise_or.reduceat(1 << modifiers, offsets[:-1])
    counts = np.array([mask.bit_count() for mask in probes.tolist()])
    cores, snr, metadata = measured_cores(root, project, sequences, offsets)
    best = collapse_cores(cores, snr, counts)
    print(f'{len(sequences)} unique full sequences and {len(best)} unique measured cores', flush=True)

    ranked = sorted(best.values(), key=lambda i: (-len(cores[i]), -counts[i], -snr[i], i))
    fasta = output / 'measured-cores.fasta'
    with fasta.open('w') as handle:
        for i in ranked:
            handle.write(f'>S{i:07d}\n{cores[i].replace("U", "T")}\n')
    config = prepare.nr80.audit.PARAMETERS.copy()
    config[config.index('-T')+1] = str(threads)
    config[config.index('-M')+1] = '32000'
    representatives = output / 'representatives.fasta'
    binary = root / 'bin/cd-hit-est'
    command = [str(binary), '-i', str(fasta), '-o', str(representatives), *config]
    with (output / 'cd-hit.log').open('w') as handle:
        subprocess.run(command, check=True, stdout=handle, stderr=subprocess.STDOUT)
    mapping = prepare.nr80.audit.parse_clusters(str(representatives)+'.clstr')
    short = {f'S{i:07d}' for i in ranked if len(cores[i]) < 4}
    expected = {f'S{i:07d}' for i in ranked}
    if mapping.keys() != expected - short:
        raise RuntimeError('CD-HIT membership differs from the measured-core inputs')
    prepare.nr80.audit.audit_matches(mapping, {f'S{i:07d}': cores[i] for i in ranked})
    kept_tokens = prepare.nr80.audit.fasta_ids(representatives) | short
    if kept_tokens != short | {token for token, (rep, _) in mapping.items() if token == rep}:
        raise RuntimeError('CD-HIT representatives disagree with cluster membership')
    selected = np.array(sorted(int(token[1:]) for token in kept_tokens), dtype=np.int64)
    selected_mask = np.zeros(len(sequences), dtype=bool)
    selected_mask[selected] = True
    observation_mask = np.repeat(selected_mask, np.diff(offsets))
    counts_before = np.bincount(modifiers, minlength=len(MODIFIERS))
    counts_after = np.bincount(modifiers[observation_mask], minlength=len(MODIFIERS))
    index_path = output / 'indices.npy'
    np.save(index_path, selected)
    membership = output / 'membership.tsv'
    with membership.open('w') as handle:
        writer = csv.writer(handle, delimiter='\t', lineterminator='\n')
        writer.writerow(['packed_index', 'core_length', 'exact_core_representative', 'nr80_representative'])
        for i, core in enumerate(cores):
            exact = best[core]
            token = f'S{exact:07d}'
            representative = token if token in short else mapping[token][0]
            writer.writerow((i, len(core), exact, int(representative[1:])))
    result = {'source_dataset': str(packed), 'source_manifest_sha256': wf.digest(packed / 'manifest.json'),
              'packed_files_sha256': {k: v for k, v in data_manifest['sha256'].items()
                                      if k.startswith('data/rnagym/')},
              'input_sequences': len(sequences), 'unique_measured_cores': len(best),
              'selected_sequences': len(selected), 'input_profiles': len(modifiers),
              'selected_profiles': int(observation_mask.sum()), 'job_id': os.environ.get('JOB_ID'),
              'profile_counts': {m: {'before': int(a), 'after': int(b)}
                                 for m, a, b in zip(MODIFIERS, counts_before, counts_after, strict=True)},
              'measured_cores': metadata, 'cdhit_command': command, 'cdhit_sha256': wf.digest(binary),
              'tie_break': 'More distinct probes, higher maximum SNR, lower packed index.',
              'selection_sha256': wf.digest(index_path), 'membership_sha256': wf.digest(membership)}
    wf.write_json(output / 'manifest.json', result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--threads', type=int, default=16)
    args = parser.parse_args()
    run(args.run.resolve(), args.output.resolve(), args.threads)


if __name__ == '__main__':
    main()
