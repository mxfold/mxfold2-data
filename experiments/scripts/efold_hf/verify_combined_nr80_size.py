#!/usr/bin/env python3
"""Check the assembled representatives with an independent combined clustering."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import combined_nr80_size as audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    summary = json.loads((output / 'summary.json').read_text())
    if not summary['success'] or audit.sha(output / 'train.jsonl') != summary['train_sha256']:
        raise ValueError('Main computation is incomplete or changed')
    folder = output / 'combined_verification'
    folder.mkdir(exist_ok=True)
    records = {}
    fasta = folder / 'input.fasta'
    with fasta.open('w') as handle:
        for i, row in enumerate(audit.rows(output / 'train.jsonl')):
            token = ('C' if row['source'] == 'rnacentral_synthetic' else 'S') + f'{i:07d}'
            records[token] = row
            if len(row['sequence']) >= 4:
                handle.write(f'>{token}\n' + row['sequence'].replace('U', 'T') + '\n')
    clustered = folder / 'representatives.fasta'
    execution = audit.execute([str(audit.CDHIT / 'cd-hit-est'), '-i', str(fasta),
        '-o', str(clustered), *audit.PARAMETERS], folder / 'run.log')
    mapping = audit.parse_clusters(str(clustered) + '.clstr')
    expected = {t for t, r in records.items() if len(r['sequence']) >= 4}
    if set(mapping) != expected:
        raise ValueError('Validation omitted input sequences')
    clusters = defaultdict(list)
    for token, (representative, alignment) in mapping.items():
        clusters[representative].append({'id': records[token]['id'],
            'source': records[token]['source'], 'token': token, 'alignment': alignment})
    merged = [v for v in clusters.values() if len(v) > 1]
    result = {'success': not merged, 'input_n': len(records),
        'clustered_n': len(clusters) + len(records)-len(mapping),
        'merged_clusters_n': len(merged), 'merged_clusters': merged, 'execution': execution,
        'binary_sha256': audit.sha(audit.CDHIT / 'cd-hit-est')}
    audit.write_json(folder / 'result.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'merged_clusters'}, indent=2), flush=True)
    if merged:
        raise ValueError('Combined clustering found additional redundancy; inspect result.json')


if __name__ == '__main__':
    main()
