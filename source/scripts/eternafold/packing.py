"""The native EternaFold pack format and input readers."""
from collections import Counter
import csv
import hashlib
import math
from pathlib import Path
import struct

import numpy as np

from mxfold2.rnagym import MODIFIERS, PROBE_BASES as BASES


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

def read_bpseq(path):
    rows = [s.split() for s in Path(path).read_text().splitlines() if s.strip() and not s.startswith('#')]
    if [int(r[0]) for r in rows] != list(range(1, len(rows) + 1)):
        raise ValueError(f'Nonconsecutive BPSEQ rows: {path}')
    sequence = ''.join(r[1] for r in rows).upper().replace('T', 'U')
    pairs = [int(r[2]) for r in rows]
    if any(j < 0 or j > len(rows) or j == i or (j and pairs[j - 1] != i)
           for i, j in enumerate(pairs, 1)):
        raise ValueError(f'Invalid/symmetric BPSEQ mapping: {path}')
    return sequence, pairs

class PackWriter:
    def __init__(self, path):
        self.path = Path(path)
        self.file = self.path.with_name(self.path.name + '.tmp').open('w+b')
        self.file.write(struct.pack('<8sIQ', b'EFPACK01', 5, 0))
        self.count, self.weight_sum = 0, 0.

    def add(self, name, sequence, *, weight=1., pairs=None, channel=None, values=None):
        encoded = name.encode('ascii')
        kind = 1 if pairs is not None else (2 if channel is not None else 0)
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError('Invalid record weight')
        self.file.write(struct.pack('<IIdii', len(sequence), len(encoded), weight, kind,
                                    -1 if channel is None else channel))
        self.file.write(encoded)
        self.file.write(sequence.encode('ascii'))
        if pairs is not None:
            self.file.write(np.asarray(pairs, dtype='<i4').tobytes())
        if channel is not None:
            if values is None or len(values) != len(sequence):
                raise ValueError('Missing evidence values')
            self.file.write(np.asarray(values, dtype='<f4').tobytes())
        self.count += 1
        self.weight_sum += weight

    def finish(self):
        self.file.seek(12)
        self.file.write(struct.pack('<Q', self.count))
        self.file.close()
        self.path.with_name(self.path.name + '.tmp').replace(self.path)
        return {'path': str(self.path), 'observations': self.count,
                'weight_sum': self.weight_sum, 'sha256': digest(self.path)}

def pack_list(pack):
    path = Path(str(pack) + '.lst')
    path.write_text('@EFPACK:' + str(Path(pack).resolve()) + '\n')
    return path

def packed_arrays(folder):
    with (folder / 'sequences.tsv').open() as f:
        sequences = list(csv.DictReader(f, delimiter='\t'))
    arrays = {name: np.load(folder / (name + '.npy'), mmap_mode='r')
              for name in ('values', 'modifiers', 'conditions', 'sequence_offsets', 'observation_offsets')}
    return sequences, arrays

def read_params(path):
    values = {k: float(v) for k, v in map(str.split, Path(path).read_text().splitlines())}
    if len(values) != 788 or not all(math.isfinite(v) for v in values.values()):
        raise ValueError('Invalid EternaFold model parameters')
    return values

def native_evidence(sequence, modifier, values):
    """Native positive Gamma observations; do not shift or squash the data."""
    values = np.asarray(values, dtype=np.float64)
    applicable = np.fromiter((b in BASES[modifier] for b in sequence), bool, count=len(sequence))
    mask = np.isfinite(values) & (values > 1e-5) & applicable
    out = np.full(len(sequence), -1., dtype='<f4')
    out[mask] = values[mask]
    return out, mask

def export_native_chemical(folder, output, *, modifier=None):
    sequences, a = packed_arrays(folder)
    selected = []
    for i in range(len(sequences)):
        left, right = map(int, a['sequence_offsets'][i:i + 2])
        observations = [o for o in range(left, right)
                        if modifier is None or MODIFIERS[int(a['modifiers'][o])] == modifier]
        if observations:
            selected.append((i, observations))
    writer = PackWriter(output)
    totals, probe_counts = Counter(), Counter()
    audit = output.with_suffix('.observations.tsv')
    with audit.open('w') as f:
        f.write('sequence_index\tobservation_index\tmodifier\tcondition\tweight\tused_positive\tmasked_nonpositive\tmasked_small_positive\n')
        for i, observations in selected:
            sequence = sequences[i]['sequence']
            groups = Counter((int(a['modifiers'][o]), int(a['conditions'][o])) for o in observations)
            conditions = Counter(p for p, c in groups)
            sequence_weight = 0.
            for o in observations:
                probe, condition = int(a['modifiers'][o]), int(a['conditions'][o])
                name = MODIFIERS[probe]
                left, right = map(int, a['observation_offsets'][o:o + 2])
                raw = np.asarray(a['values'][left:right])
                evidence, mask = native_evidence(sequence, name, raw)
                observed = np.isfinite(raw) & (raw > -900) & np.fromiter(
                    (b in BASES[name] for b in sequence), bool, count=len(sequence))
                nonpositive = int((observed & (raw <= 0)).sum())
                small = int((observed & (raw > 0) & ~mask).sum())
                weight = 1. / (len(selected) * len(conditions) * conditions[probe] * groups[probe, condition])
                writer.add(f's{i}_o{o}', sequence, weight=weight, channel=probe, values=evidence)
                f.write(f'{i}\t{o}\t{name}\t{condition}\t{weight:.17g}\t{mask.sum()}\t{nonpositive}\t{small}\n')
                sequence_weight += weight
                totals['used_positive_bases'] += int(mask.sum())
                totals['masked_nonpositive_bases'] += nonpositive
                totals['masked_small_positive_bases'] += small
                totals['empty_profiles_retained_with_zero_likelihood'] += int(not mask.any())
                probe_counts[name] += 1
            if not math.isclose(sequence_weight, 1. / len(selected), rel_tol=1e-10):
                raise ValueError('Sequence normalization mismatch')
    result = {**writer.finish(), 'sequences': len(selected), 'modifiers': dict(probe_counts),
              'counts': dict(totals), 'source': str(folder),
              'source_manifest_sha256': digest(folder / 'manifest.json'),
              'observation_audit_sha256': digest(audit)}
    if not math.isclose(result['weight_sum'], 1., rel_tol=1e-8):
        raise ValueError('Chemical weights do not sum to one')
    return result
