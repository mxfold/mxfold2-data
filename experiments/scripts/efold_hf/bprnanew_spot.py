#!/usr/bin/env python3
"""Run the official SPOT-RNA five-model ensemble on compute nodes."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).parent))
from bprnanew_benchmark import DATA,OUT,SHARDS,bpseq,checked_plan,pair_metrics,read_jsonl,sha,write_json

SOURCE=DATA/'SPOT-RNA'
ARCHIVE=DATA/'SPOT-RNA-models.tar.gz'


def input_basename(shard,smoke):
    # Upstream stores TFRecords in a shared source directory using only this
    # basename. It must be unique across every concurrently running shard.
    return f'bprnanew_{"smoke" if smoke else "full"}_shard-{shard}.fasta'


def prepare():
    with tarfile.open(ARCHIVE) as archive:
        archive.extractall(SOURCE,filter='data')
    if not (SOURCE/'SPOT-RNA-models/model0.meta').exists():
        raise FileNotFoundError('Official SPOT-RNA graph missing')
    registry=OUT/'external_models.json'
    models=json.loads(registry.read_text())
    if not any(m['model']=='SPOT_RNA_public' for m in models):
        models.append({'model':'SPOT_RNA_public','label':'公開SPOT-RNA 5モデルensemble',
                       'backend':'spotrna','source':str(SOURCE),'checkpoint':str(ARCHIVE),
                       'checkpoint_sha256':sha(ARCHIVE),
                       'protocol':'Official SPOT-RNA.py; five ensemble members; native postprocessing; TensorFlow 2.20 compat.v1 CPU inference'})
        write_json(registry,models)
    write_json(OUT/'spot-prepare-complete.json',{'success':True,'archive_sha256':sha(ARCHIVE)})


def work(shard,smoke):
    checked_plan()
    if not (OUT/'spot-prepare-complete.json').exists():raise RuntimeError('SPOT-RNA preparation incomplete')
    items=sorted(read_jsonl(OUT/'panel.jsonl'),key=lambda r:(-len(r['sequence']),r['id']))
    items=[r for i,r in enumerate(items) if i%SHARDS==shard]
    if smoke:
        amb=next(r for r in items if set(r['sequence'])-set('ACGU'))
        items=list({r['id']:r for r in [items[0],items[len(items)//2],items[-1],amb]}.values())
    elif not (OUT/'smoke/SPOT_RNA_public/shard-0-complete.json').exists():
        raise RuntimeError('SPOT-RNA smoke validation not complete')
    folder=OUT/('smoke' if smoke else 'predictions')/'SPOT_RNA_public'
    folder.mkdir(parents=True,exist_ok=True)
    marker=folder/f'shard-{shard}-complete.json'
    if marker.exists():print('Already complete SPOT-RNA',shard,flush=True);return
    native=OUT/'native/SPOT_RNA_public'/('smoke' if smoke else 'full')/f'shard-{shard}'
    native.mkdir(parents=True,exist_ok=True)
    fasta=native/input_basename(shard,smoke)
    with fasta.open('w') as handle:
        for item in items:handle.write('>'+item['id']+'\n'+item['sequence']+'\n')
    outputs=native/'outputs';outputs.mkdir(exist_ok=True)
    code="import tensorflow as tf; tf.compat.v1.disable_eager_execution(); import runpy; runpy.run_path("+repr(str(SOURCE/'SPOT-RNA.py'))+",run_name='__main__')"
    environment=dict(os.environ)
    environment['PYTHONPATH']=str(ROOT/'.cache/bprna-spot-tensorflow')+os.pathsep+str(ROOT/'.cache/bprna-benchmark-deps')
    environment['TF_CPP_MIN_LOG_LEVEL']='2'
    environment['CUDA_VISIBLE_DEVICES']='-1'
    command=[str(ROOT/'.venv/bin/python'),'-c',code,'--inputs',str(fasta),'--outputs',str(outputs)+'/',
             '--cpu','4','--gpu','-1']
    start=time.monotonic()
    with (native/'predict.log').open('w') as log:
        subprocess.run(command,cwd=SOURCE,env=environment,stdout=log,stderr=subprocess.STDOUT,check=True)
    records=[]
    for item in items:
        output=outputs/(item['id']+'.bpseq')
        result=bpseq(output.read_text(),item['id'])
        if len(result['sequence'])!=len(item['sequence']):raise RuntimeError('SPOT-RNA output length differs')
        pairs=result['pairs']
        records.append({'id':item['id'],'sequence_sha256':item['sequence_sha256'],'length':len(item['sequence']),
                        'predicted_pairs':pairs,**pair_metrics(item['pairs'],pairs)})
    path=folder/f'shard-{shard}.jsonl'
    with path.open('w') as handle:
        for row in records:handle.write(json.dumps(row,allow_nan=False)+'\n')
    write_json(marker,{'success':True,'model':'SPOT_RNA_public','n':len(records),
                      'predictions_sha256':sha(path),'archive_sha256':sha(ARCHIVE),
                      'input_fasta_sha256':sha(fasta),'upstream_tfrecord_basename':fasta.name+'.tfrecords',
                      'seconds':time.monotonic()-start})
    print('Completed SPOT-RNA',shard,len(records),time.monotonic()-start,flush=True)


if __name__=='__main__':
    action=sys.argv[1]
    raw=os.environ.get('SGE_TASK_ID','1');index=int(raw) if raw.isdigit() else 1
    if action=='prepare':prepare()
    else:work(index-1,action=='smoke')
