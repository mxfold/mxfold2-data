"""Prepare all-structure NR80 and profile training data for MXfold2 0.3.0."""
from __future__ import annotations

from collections import Counter
import csv
import json
from pathlib import Path
import re
import tarfile

from scripts.efold_hf import condition5_nr80_data as nr80, workflow as wf

RAW = Path('data/efold_hf/raw')
LOG = Path('qsub_logs/efold_hf')
STRUCTURES = {
    'PDB': RAW / 'PDB.json',
    'Archive-NoFam': LOG / 'archive_nofam_20261031/panel.jsonl',
    'ArchiveII': RAW / 'archiveII.json',
    'bpRNAnew': LOG / 'bprnanew_20261029/panel.jsonl',
    'bpRNA-1m': RAW / 'bpRNA-1m.json',
    'RNAcentral': RAW / 'rnacentral_synthetic.json',
}
RIVAS = Path('data/efold_hf/pretrain_diagnosis_20261031/Rivas.tar.gz')
RIVAS_SPLITS = {'TrainSetA': 3166, 'TrainSetB': 1094, 'TestSetA': 592, 'TestSetB': 430}
# Used by the earlier standalone RNAgym selection script.
RNAGYM = Path('data/rnagym/secondary_structure_prediction')
RNAGYM_SPLITS = ('train', 'test')
PROFILE_RUNS = {
    'ribonanza': LOG / 'condition5_pt020_ribonanza3_replay20_20261103',
    'mammalian': LOG / 'condition5_pt020_pri_mrna3_after_ribonanza3_replay20_20261103',
}
PROFILE_COUNTS = {
    'ribonanza': {'ribo500-blast': 45729},
    'mammalian': {'pri_miRNA': 1096, 'human_mRNA': 1375},
}
PROFILE_FILES = {phase: run / 'data/chemical.jsonl' for phase, run in PROFILE_RUNS.items()}
INPUTS = [*STRUCTURES.values(), RIVAS, *PROFILE_FILES.values(),
          *(run / 'plan.json' for run in PROFILE_RUNS.values())]
MAX_LENGTH = 1000
IUPAC = re.compile('[ACGUNXRYSWKMBDHV]+')
AMBIGUOUS = str.maketrans(dict.fromkeys('XRYSWKMBDHV', 'N'))


def bpseq(handle):
    entries = [line.decode().split() for line in handle if line.split() and line.split()[0].isdigit()]
    sequence = ''.join(base for _, base, _ in entries)
    partners = [int(other) for _, _, other in entries]
    if any(int(entry[0]) != i for i, entry in enumerate(entries, 1)):
        raise ValueError('Nonconsecutive BPSEQ indices')
    if any(j < 0 or j > len(partners) or j == i or (j and partners[j-1] != i)
           for i, j in enumerate(partners, 1)):
        raise ValueError('Inconsistent BPSEQ pairs')
    return sequence, [(i-1, j-1) for i, j in enumerate(partners, 1) if i < j]


def rivas_records(path):
    with tarfile.open(path, 'r:gz') as archive:
        expected = {}
        for source, count in RIVAS_SPLITS.items():
            with archive.extractfile(f'data/{source}.lst') as handle:
                members = handle.read().decode().splitlines()
            if len(set(members)) != count or any(not member.startswith(f'data/{source}/') for member in members):
                raise ValueError(f'Unexpected {source} membership')
            expected.update(dict.fromkeys(members, source))
        for member in archive:
            if member.name in expected:
                source = expected.pop(member.name)
                with archive.extractfile(member) as handle:
                    sequence, pairs = bpseq(handle)
                yield source, Path(member.name).stem, sequence, pairs
        if expected:
            raise ValueError(f'Missing Rivas BPSEQ entries: {list(expected)[:5]}')


def structural_records(project):
    for source, path in STRUCTURES.items():
        if source == 'ArchiveII':
            yield from rivas_records(project / RIVAS)
        if path.suffix == '.jsonl':
            for row in nr80.rows(project / path):
                yield source, row['id'], row['sequence'], row['pairs']
        else:
            for identifier, row in json.loads((project / path).read_bytes()).items():
                yield source, identifier, row['sequence'], row['structure']


def cluster(folder, records, tokens, *, input_path=None, greedy=False):
    if not tokens:
        return {}, set()
    folder.mkdir(parents=True, exist_ok=True)
    if input_path is None:
        input_path = folder / 'input.fasta'
        nr80.write_fasta(input_path, tokens, records)
    output = folder / 'representatives.fasta'
    nr80.native(folder, 'cluster', 'cd-hit-est', [('-i', input_path)], output, greedy=greedy)
    mapping = nr80.audit.parse_clusters(str(output) + '.clstr')
    if set(mapping) != tokens:
        raise RuntimeError(f'Clustering did not account for every input in {folder}')
    nr80.audit.audit_matches(mapping, {key: row['sequence'] for key, row in records.items()})
    return mapping, nr80.audit.fasta_ids(output)


def prepare_structures(root, project, max_length):
    data = root / 'data'
    work = data / 'nr80'
    records, token_by_sequence, membership = {}, {}, []
    audit = {name: Counter() for name in (*STRUCTURES, *RIVAS_SPLITS)}
    for source, key, raw_sequence, raw_pairs in structural_records(project):
        counts = audit[source]
        counts['input'] += 1
        identifier = f'{source}:{key}'
        sequence = wf.normalize(raw_sequence)
        if not IUPAC.fullmatch(sequence):
            raise ValueError(f'Unsupported structural sequence: {identifier}')
        counts['ambiguous_bases'] += sum(base not in 'ACGU' for base in sequence)
        sequence = sequence.translate(AMBIGUOUS)
        if len(sequence) > max_length:
            counts['excluded_over_max_length'] += 1
            membership.append((source, identifier, len(sequence), None))
            continue
        pairs = wf.structure_pairs(raw_pairs, len(sequence))
        token = token_by_sequence.get(sequence)
        if token is None:
            prefix = 'C' if source == 'RNAcentral' else 'S'
            token = f'{prefix}{len(records):07d}'
            token_by_sequence[sequence] = token
            records[token] = {'id': identifier, 'source': source, 'sequence': sequence,
                              'pairs': pairs, 'alternatives': []}
            counts['exact_unique'] += 1
        elif source != 'RNAcentral':
            entry = records[token]
            if pairs != entry['pairs'] and all(pairs != alt['pairs'] for alt in entry['alternatives']):
                entry['alternatives'].append({'id': identifier, 'pairs': pairs})
        elif token.startswith('S'):
            counts['exact_curated_overlap'] += 1
        membership.append((source, identifier, len(sequence), token))
    if len({row[1] for row in membership}) != len(membership):
        raise RuntimeError('Duplicate source identifiers')
    short = {k for k, row in records.items() if len(row['sequence']) < 4}
    curated = {k for k in records if k.startswith('S')} - short
    pseudo = set(records) - curated - short
    wf.write_json(root / 'preparation.json', {'stage': 'NR80', 'input': audit})
    curated_map, curated_kept = cluster(work / 'curated', records, curated)
    cross_map = {}
    if curated and pseudo:
        pseudo, pseudo_path, cross_map, _ = nr80.filter_protected(
            work / 'RNAcentral_vs_curated', records, curated, pseudo)
    else:
        pseudo_path = None
    pseudo_map, pseudo_kept = cluster(work / 'RNAcentral', records, pseudo,
                                      input_path=pseudo_path, greedy=True)
    kept = curated_kept | pseudo_kept | short
    _, checked = cluster(work / 'combined_check', records, kept - short)
    if checked != kept - short:
        raise RuntimeError('Combined CD-HIT check found residual redundancy')
    mapping = {**curated_map, **pseudo_map, **cross_map, **{k: (k, '') for k in short}}

    output = data / 'structure.jsonl'
    temporary = output.with_suffix('.tmp')
    for token in sorted(kept):
        row = records[token]
        distinct = {}
        for label in [row, *row['alternatives']]:
            label['train_pairs'] = wf.project_pairs(row['sequence'], label['pairs'])
            distinct.setdefault(tuple(label['train_pairs']), label)
        row['alternatives'] = list(distinct.values())[1:]
        audit[row['source']]['retained'] += 1
        audit[row['source']]['alternative_train_labels'] += len(distinct) - 1
    wf.write_jsonl(temporary, (records[token] for token in sorted(kept)))
    temporary.replace(output)

    with (data / 'structure-membership.tsv').open('w') as handle:
        writer = csv.writer(handle, delimiter='\t')
        writer.writerow(['source', 'id', 'length', 'status', 'representative_id'])
        for source, identifier, length, token in membership:
            if token is None:
                writer.writerow((source, identifier, length, 'excluded_over_max_length', ''))
                continue
            representative = mapping[token][0]
            if token in cross_map:
                representative = curated_map[representative][0]
            status = 'cluster_member'
            if token in kept:
                status = 'exact_match'
                if identifier == records[token]['id']:
                    status = 'retained'
                elif any(identifier == alt['id'] for alt in records[token]['alternatives']):
                    status = 'alternative_label'
            audit[source][f'membership_{status}'] += 1
            writer.writerow((source, identifier, length, status, records[representative]['id']))
    result = {'sequences': len(kept), 'sources': audit,
              'longest': max(len(records[k]['sequence']) for k in kept),
              'max_length': max_length, 'rna_central_removed_near_curated': len(cross_map),
              'clustering': nr80.audit.PARAMETERS, 'combined_check': 'passed',
              'RNAcentral_labels': 'RNAstructure Fold pseudo-labels',
              'RNAcentral_priority': 'Exclude matches to all eligible non-RNAcentral inputs, including members of curated clusters.',
              'alternative_labels': 'Distinct curated exact-sequence conformers rotate by epoch.'}
    wf.write_json(data / 'structure-summary.json', result)
    print(json.dumps(result), flush=True)
    return result


def prepare_profiles(root, project, plan, phase):
    path = PROFILE_FILES[phase]
    reference = json.loads((project / PROFILE_RUNS[phase] / 'plan.json').read_text())
    if plan['input_sha256'][str(path)] != reference['data_sha256']['chemical.jsonl']:
        raise RuntimeError(f'Profile snapshot differs from the historical {phase} experiment')
    counts = Counter()
    ids, sequences = set(), set()
    observation_count = 0

    def rows():
        nonlocal observation_count
        for row in nr80.rows(project / path):
            identifier, source, sequence, observations = (
                row['id'], row['source'], row['sequence'], row['observations'])
            if identifier in ids or not 1 <= len(sequence) <= plan['max_length'] or not observations:
                raise ValueError(f'Invalid profile record: {identifier}')
            for obs in observations:
                if obs['probe'] not in ('dms', 'shape') or len(obs['values']) != len(sequence):
                    raise ValueError(f'Misaligned profile: {identifier}')
            ids.add(identifier)
            sequences.add(sequence)
            counts[source] += 1
            observation_count += len(observations)
            yield {'id': identifier, 'source': source, 'sequence': sequence, 'observations': observations}

    output = root / 'data' / f'{phase}.jsonl'
    temporary = output.with_suffix('.tmp')
    wf.write_jsonl(temporary, rows())
    if counts != PROFILE_COUNTS[phase] or len(sequences) != len(ids):
        raise RuntimeError(f'Unexpected {phase} training cohort: {dict(counts)}')
    temporary.replace(output)
    return {'sequences': len(ids), 'sources': counts, 'observations': observation_count}, sequences


def prepare(root):
    plan = wf.plan_for(root)
    project = Path(plan['project'])
    data = root / 'data'
    data.mkdir(exist_ok=True)
    wf.write_json(root / 'preparation.json', {'stage': 'structure_input'})
    for name, expected in plan['input_sha256'].items():
        if wf.digest(project / name) != expected:
            raise RuntimeError(f'Input changed: {name}')
    structures = prepare_structures(root, project, plan['max_length'])
    wf.write_json(root / 'preparation.json', {'stage': 'profile_packing', 'structures': structures})
    ribonanza, ribo_sequences = prepare_profiles(root, project, plan, 'ribonanza')
    mammalian, mammal_sequences = prepare_profiles(root, project, plan, 'mammalian')
    structure_sequences = {row['sequence'] for row in nr80.rows(data / 'structure.jsonl')}
    tracked = ('structure.jsonl', 'structure-summary.json', 'structure-membership.tsv',
               'ribonanza.jsonl', 'mammalian.jsonl')
    result = {
        'structures': structures, 'ribonanza': ribonanza, 'mammalian': mammalian,
        'exact_sequence_overlap': {
            'ribonanza_structure': len(ribo_sequences & structure_sequences),
            'mammalian_structure': len(mammal_sequences & structure_sequences),
            'mammalian_ribonanza': len(mammal_sequences & ribo_sequences),
        },
        'sha256': {f'data/{name}': wf.digest(data / name) for name in tracked},
    }
    wf.write_json(root / 'data-complete.json', result)
    wf.write_json(root / 'preparation.json', {'stage': 'complete', 'structures': structures})
    print(json.dumps({k: result[k] for k in ('ribonanza', 'mammalian', 'exact_sequence_overlap')}), flush=True)
