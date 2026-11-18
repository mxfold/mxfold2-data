#!/usr/bin/env python3
"""NR80 of condition2, protecting all original bpRNA sequences from pseudo-labels.

Run on a CPU compute node. Completed native clustering stages are reusable;
the final marker requires label preservation and independent combined NR80.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import time

PROJECT = Path('/gs/bs/tga-satolab-gtex/satoken/repos/mxfold2-ryntaro')
INPUT = PROJECT/'qsub_logs/efold_hf/pretrain_variants_20261031/bprna_rnacentral10/data/train.jsonl'
INPUT_SHA = '612e034325a43882ad1ff066046b28e97d311dfc75e695724f0d01ae95831bd9'
INPUT_SOURCES = {'bpRNA-1m':45507, 'rnacentral_synthetic':210260}
DEFAULT_OUTPUT = PROJECT/'qsub_logs/efold_hf/condition5_nr80_dataset_20261103'
CDHIT = PROJECT/'.cache/cdhit-4.8.1/cd-hit-v4.8.1-2019-0228'
helper = Path(__file__).with_name('audit.py')
if not helper.exists():
    helper = PROJECT/'scripts/efold_hf/combined_nr80_size.py'
spec = importlib.util.spec_from_file_location('condition5_native_audit', helper)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
audit.PROJECT, audit.CDHIT = PROJECT, CDHIT
sha, rows, write_json = audit.sha, audit.rows, audit.write_json


def read(path):
    return json.loads(Path(path).read_text())


def write_fasta(path, tokens, records):
    path = Path(path)
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('w') as handle:
        for token in sorted(tokens):
            handle.write('>'+token+'\n'+records[token]['sequence'].replace('U','T')+'\n')
    temporary.replace(path)


def native(folder, name, binary, inputs, output, extra=(), *, greedy=False):
    """Reuse only a committed stage whose command, inputs and outputs still match."""
    folder.mkdir(parents=True,exist_ok=True)
    command = [str(CDHIT/binary)]
    for flag,path in inputs:
        command += [flag,str(path)]
    parameters = list(audit.PARAMETERS)
    if greedy:
        parameters[parameters.index('-g')+1] = '0'
    command += ['-o',str(output),*parameters,*extra]
    marker = folder/(name+'-complete.json')
    binding = {str(path):sha(path) for _,path in inputs}
    binding[str(CDHIT/binary)] = sha(CDHIT/binary)
    if marker.exists():
        saved = read(marker)
        if saved['command']!=command or saved['input_sha256']!=binding:
            raise RuntimeError('Native stage inputs changed: '+str(marker))
        if any(sha(p)!=h for p,h in saved['output_sha256'].items()):
            raise RuntimeError('Native stage output changed: '+str(marker))
        return saved
    started=time.monotonic()
    with (folder/(name+'.log')).open('w') as handle:
        subprocess.run(command,stdout=handle,stderr=subprocess.STDOUT,check=True,timeout=5*3600)
    result={'success':True,'command':command,'input_sha256':binding,
        'output_sha256':{str(p):sha(p) for p in (output,Path(str(output)+'.clstr'))},
        'seconds':time.monotonic()-started,'job_id':os.environ.get('JOB_ID')}
    write_json(marker,result)
    return result


def filter_protected(folder, records, protected, candidates):
    """Forward filter plus repeated reverse native matching; delete only C tokens."""
    folder.mkdir(parents=True,exist_ok=True)
    protected_path,candidates_path=folder/'protected.fasta',folder/'candidates.fasta'
    write_fasta(protected_path,protected,records)
    write_fasta(candidates_path,candidates,records)
    sequences={k:r['sequence'] for k,r in records.items()}
    remaining_path=folder/'forward_unmatched.fasta'
    forward=native(folder,'forward','cd-hit-est-2d',
        [('-i',protected_path),('-i2',candidates_path)],remaining_path,('-s2','.8'))
    remaining=audit.fasta_ids(remaining_path)
    mapping=audit.parse_clusters(str(remaining_path)+'.clstr')
    if set(mapping)!=protected|(candidates-remaining):
        raise RuntimeError('Forward filter omitted records')
    audit.audit_matches(mapping,sequences)
    removed={k:v for k,v in mapping.items() if k in candidates}
    if any(v[0] not in protected for v in removed.values()):
        raise RuntimeError('RNAcentral displaced bpRNA')
    active=set(protected);reverse_runs=[]
    for iteration in range(1,1001):
        stage=folder/f'reverse-{iteration:03d}'
        stage.mkdir(exist_ok=True)
        query_path=stage/'queries.fasta'
        write_fasta(query_path,active,records)
        unmatched_path=stage/'unmatched.fasta'
        execution=native(stage,'reverse','cd-hit-est-2d',
            [('-i',remaining_path),('-i2',query_path)],unmatched_path)
        mapping=audit.parse_clusters(str(unmatched_path)+'.clstr')
        unmatched=audit.fasta_ids(unmatched_path)
        if set(mapping)!=remaining|(active-unmatched):
            raise RuntimeError('Reverse filter omitted records')
        audit.audit_matches(mapping,sequences)
        matched_queries=active-unmatched
        newly_removed=set()
        for token in matched_queries:
            reference,alignment=mapping[token]
            if reference not in remaining or not reference.startswith('C'):
                raise RuntimeError('Reverse filtering attempted to remove protected bpRNA')
            newly_removed.add(reference)
            match=re.search(r'at (\d+):(\d+):(\d+):(\d+)/\+/([\d.]+)%',alignment)
            a,b,c,d,identity=match.groups()
            removed[reference]=(token,f'{len(sequences[reference])}nt, >{reference}... at {c}:{d}:{a}:{b}/+/{identity}%')
        remaining-=newly_removed
        result={'iteration':iteration,'removed_n':len(newly_removed),'remaining_n':len(remaining),
            'query_n':len(active),'execution':execution}
        reverse_runs.append(result)
        write_json(stage/'summary.json',result)
        write_json(folder.parent/'progress.json',{'stage':'protect_all_bpRNA',**{k:v for k,v in result.items() if k!='execution'}})
        print('Protected bpRNA: reverse',iteration,'removed',len(newly_removed),'remaining',len(remaining),flush=True)
        if not newly_removed:
            if active==protected:
                break
            active=set(protected)
            continue
        active=matched_queries
        remaining_path=stage/'remaining.fasta'
        write_fasta(remaining_path,remaining,records)
    else:
        raise RuntimeError('Protected filtering did not converge')
    if set(removed)!=candidates-remaining:
        raise RuntimeError('Missing removal provenance')
    audit.audit_matches(removed,sequences)
    return remaining,remaining_path,removed,{'forward':forward,'reverse':reverse_runs,
        'all_original_bpRNA_protected':len(protected),'removed_n':len(removed),
        'final_full_reverse_scan_zero_matches':True}


def build(output, input_path=INPUT, expected_sha=INPUT_SHA, expected_sources=None):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    if sha(input_path)!=expected_sha:
        raise RuntimeError('Condition2 training input changed')
    for path,checksum in read(output/'bootstrap.json').get('files',{}).items() if (output/'bootstrap.json').exists() else []:
        if sha(path)!=checksum:
            raise RuntimeError('Frozen dataset implementation changed: '+path)
    if (output/'complete.json').exists():
        complete=read(output/'complete.json')
        if not complete['success'] or sha(output/'train.jsonl')!=complete['train_sha256']:
            raise RuntimeError('Completed dataset changed')
        return read(output/'summary.json')
    source=list(rows(input_path))
    counts=dict(Counter(r['source'] for r in source))
    if counts!=(INPUT_SOURCES if expected_sources is None else expected_sources):
        raise RuntimeError('Wrong condition2 source counts: '+str(counts))
    if len({r['id'] for r in source})!=len(source) or len({r['sequence'] for r in source})!=len(source):
        raise RuntimeError('Condition2 inputs must be exact unique')
    if any(not 1<=len(r['sequence'])<=1000 or set(r['sequence'])-set('ACGU') or 'train_pairs' not in r for r in source):
        raise RuntimeError('Invalid sequence, training length or structural label')
    by_source={s:sorted((r for r in source if r['source']==s),key=lambda r:r['id']) for s in counts}
    records={}
    for s,prefix in [('bpRNA-1m','S'),('rnacentral_synthetic','C')]:
        records.update({f'{prefix}{i:07d}':r for i,r in enumerate(by_source[s])})
    protected={k for k,r in records.items() if k.startswith('S') and len(r['sequence'])>=4}
    short={k for k,r in records.items() if len(r['sequence'])<4}
    if any(k.startswith('C') for k in short):
        raise RuntimeError('Unexpected short RNAcentral sequence')
    candidates={k for k in records if k.startswith('C')}
    bp=output/'bpRNA';bp.mkdir(exist_ok=True)
    bp_input,bp_representatives=bp/'input.fasta',bp/'representatives.fasta'
    write_fasta(bp_input,protected,records)
    write_json(output/'progress.json',{'stage':'bpRNA_clustering','input_sources':counts})
    native(bp,'cluster','cd-hit-est',[('-i',bp_input)],bp_representatives)
    bp_mapping=audit.parse_clusters(str(bp_representatives)+'.clstr')
    if set(bp_mapping)!=protected:
        raise RuntimeError('bpRNA clustering omitted input')
    audit.audit_matches(bp_mapping,{k:r['sequence'] for k,r in records.items()})
    bp_kept=audit.fasta_ids(bp_representatives)|short
    remaining,remaining_path,removed,cross=filter_protected(output/'cross',records,protected,candidates)
    central=output/'RNAcentral';central.mkdir(exist_ok=True)
    central_representatives=central/'representatives.fasta'
    write_json(output/'progress.json',{'stage':'RNAcentral_clustering','remaining_n':len(remaining)})
    native(central,'cluster','cd-hit-est',[('-i',remaining_path)],central_representatives,greedy=True)
    central_mapping=audit.parse_clusters(str(central_representatives)+'.clstr')
    if set(central_mapping)!=remaining:
        raise RuntimeError('RNAcentral clustering omitted input')
    audit.audit_matches(central_mapping,{k:r['sequence'] for k,r in records.items()})
    central_kept=audit.fasta_ids(central_representatives)
    kept=bp_kept|central_kept
    verification=output/'combined_verification';verification.mkdir(exist_ok=True)
    combined_input,combined_output=verification/'input.fasta',verification/'representatives.fasta'
    write_fasta(combined_input,kept-short,records)
    write_json(output/'progress.json',{'stage':'combined_independent_verification','candidate_n':len(kept)})
    native(verification,'cluster','cd-hit-est',[('-i',combined_input)],combined_output)
    combined_mapping=audit.parse_clusters(str(combined_output)+'.clstr')
    if set(combined_mapping)!=kept-short:
        raise RuntimeError('Combined verification omitted input')
    merged={k:v for k,v in combined_mapping.items() if k!=v[0]}
    write_json(verification/'result.json',{'success':not merged,'input_n':len(kept),
        'merged_n':len(merged),'merged':merged})
    if merged:
        raise RuntimeError('Independent combined clustering found additional redundancy')
    temporary=output/'train.jsonl.tmp'
    with temporary.open('w') as handle:
        for token in sorted(kept):
            handle.write(json.dumps(records[token],separators=(',',':'),allow_nan=False)+'\n')
    temporary.replace(output/'train.jsonl')
    original={r['id']:r for r in source}
    selected=list(rows(output/'train.jsonl'))
    if any(r!=original[r['id']] for r in selected) or len(selected)!=len(kept):
        raise RuntimeError('Selected sequences or structural labels changed')
    with (output/'membership.tsv').open('w') as handle:
        writer=csv.writer(handle,delimiter='\t')
        writer.writerow(['id','source','status','representative_id','alignment'])
        for token,row in sorted(records.items()):
            if token in short:
                representative,alignment=token,'exact-unique length<4'
            elif token.startswith('S'):
                representative,alignment=bp_mapping[token]
            elif token in removed:
                representative,alignment=removed[token]
            else:
                representative,alignment=central_mapping[token]
            status=('retained' if token in kept else 'removed_bpRNA_homolog' if token in removed else 'removed_within_source_homolog')
            writer.writerow([row['id'],row['source'],status,records[representative]['id'],alignment])
    summary={'success':True,'condition':5,'input_condition':2,'input':str(input_path),
        'input_sha256':expected_sha,'input_sources':counts,'total_n':len(selected),
        'sources_retained':dict(Counter(r['source'] for r in selected)),
        'train_sha256':sha(output/'train.jsonl'),'max_length':1000,
        'identity':.8,'coverage_short':.8,'coverage_long':.8,'length_ratio':.8,
        'strand':'forward only; U to T only during clustering','priority':'bpRNA before RNAcentral; protect ALL original bpRNA, including removed cluster members',
        'labels':'Every selected original sequence, pairs and train_pairs preserved; RNAcentral supplied RNAstructure Fold pseudo-labels',
        'short_policy':'Retain exact-unique lengths1..3; at length ratio>=.8 and identity>=.8 only exact matches can cluster',
        'cross_filter':cross,'combined_native_verification_passed':True,
        'binary_sha256':{n:sha(CDHIT/n) for n in ['cd-hit-est','cd-hit-est-2d']}}
    write_json(output/'summary.json',summary)
    write_json(output/'label_preservation_verification.json',{'success':True,'selected_n':len(selected),
        'train_sha256':summary['train_sha256'],'all_selected_rows_equal_original':True})
    if sha(input_path)!=expected_sha:
        raise RuntimeError('Input changed during clustering')
    write_json(output/'complete.json',{'success':True,'total_n':len(selected),
        'sources_retained':summary['sources_retained'],'train_sha256':summary['train_sha256'],
        'combined_native_verification_passed':True})
    write_json(output/'progress.json',{'stage':'complete','total_n':len(selected)})
    print(json.dumps({k:summary[k] for k in ['condition','total_n','sources_retained','train_sha256']},ensure_ascii=False),flush=True)
    return summary


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    args=parser.parse_args()
    build(args.output)


if __name__=='__main__':
    main()
