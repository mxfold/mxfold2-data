#!/usr/bin/env python3
"""Published comparison models on the same immutable bpRNAnew panel."""
from __future__ import annotations
import ast
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import textwrap

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(Path(__file__).parent)]
sys.path.append(str(ROOT / '.cache/bprna-benchmark-deps'))
from bprnanew_benchmark import DATA, OUT, SHARDS, bpseq, checked_plan, pair_metrics, read_jsonl, sha, write_json


def prepare():
    eterna = Path('/gs/bs/tga-satolab-gtex/yamauchi/EternaFold')
    rnastructure = Path('/gs/bs/tga-satolab-gtex/yamauchi/RNAstructure')
    models = [
        {'model': 'EternaFold_public', 'label': '公開EternaFold v1・MEA gamma=6', 'backend': 'eternafold',
         'binary': str(eterna / 'src/contrafold'), 'checkpoint': str(eterna / 'parameters/EternaFoldParams.v1')},
        {'model': 'RNAstructure_Fold', 'label': 'RNAstructure Fold・MFE', 'backend': 'rnastructure',
         'binary': str(rnastructure / 'exe/Fold'), 'datapath': str(rnastructure / 'data_tables')},
        {'model': 'RNAformer_bprna', 'label': 'RNAformer公開bpRNAモデル', 'backend': 'rnaformer',
         'source': str(DATA / 'RNAformer'), 'checkpoint': str(DATA / 'RNAformer/bprna.pth'),
         'config': str(DATA / 'RNAformer/bprna.yml')},
        {'model': 'CNNFold_600', 'label': 'CNNFold-600公開重み・公式argmax後処理', 'backend': 'cnnfold',
         'source': str(DATA / 'CNNFold'), 'checkpoint': str(DATA / 'CNNFold/models/cnnfold600.mdl')},
    ]
    ufold = DATA / 'UFold/models/ufold_train.pt'
    if ufold.exists():
        models.append({'model': 'UFold_public', 'label': 'UFold公開ufold_train.pt', 'backend': 'ufold',
                       'source': str(DATA / 'UFold'), 'checkpoint': str(ufold)})
    for spec in models:
        for key in ('binary', 'checkpoint', 'config'):
            if key in spec:
                spec[key + '_sha256'] = sha(spec[key])
    write_json(OUT / 'external_models.json', models)
    print(json.dumps(models, indent=2, ensure_ascii=False), flush=True)


def parse_ct(path, expected_length):
    lines = [line.split() for line in path.read_text().splitlines() if line.strip()]
    if int(lines[0][0]) != expected_length or len(lines) < expected_length+1:
        raise ValueError('Invalid CT output length')
    partners = [0]
    for i, row in enumerate(lines[1:expected_length+1], 1):
        if len(row) < 6 or int(row[0]) != i:
            raise ValueError('Invalid CT coordinates')
        partners.append(int(row[4]))
    for i, j in enumerate(partners[1:], 1):
        if j < 0 or j > expected_length or j == i or (j and partners[j] != i):
            raise ValueError('Invalid CT partner')
    return [[i-1, j-1] for i, j in enumerate(partners) if i and j > i]


def predictor(spec, folder):
    backend = spec['backend']
    if backend in ('rnastructure', 'eternafold'):
        for key in ('binary', 'checkpoint'):
            if key in spec and sha(spec[key]) != spec[key + '_sha256']:
                raise RuntimeError('External model/binary changed')
        def predict(item):
            prefix = folder / item['id']
            sequence = ''.join(c if c in 'ACGU' else 'N' for c in item['sequence'])
            fasta = Path(str(prefix)+'.fasta')
            fasta.write_text('>' + item['id'] + '\n' + sequence + '\n')
            if backend == 'eternafold':
                output = Path(str(prefix)+'.bpseq')
                cmd = [spec['binary'], 'predict', '--params', spec['checkpoint'], '--gamma', '6',
                       '--bpseq', str(output), str(fasta)]
                run = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
                if run.returncode:
                    raise RuntimeError(run.stderr[-2000:])
                result = bpseq(output.read_text(), item['id'])
                if len(result['sequence']) != len(sequence):
                    raise RuntimeError('EternaFold length changed')
                return result['pairs']
            output = Path(str(prefix)+'.ct')
            environment = dict(os.environ, DATAPATH=spec['datapath'], OMP_NUM_THREADS='1')
            run = subprocess.run([spec['binary'], str(fasta), str(output), '-m', '1'],
                                 env=environment, capture_output=True, text=True, timeout=180)
            if run.returncode:
                raise RuntimeError(run.stderr[-2000:] + run.stdout[-1000:])
            return parse_ct(output, len(sequence))
        return predict
    import torch
    import numpy as np
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required for neural comparison model')
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    sys.path.insert(0, spec['source'])
    if sha(spec['checkpoint']) != spec['checkpoint_sha256']:
        raise RuntimeError('Comparison checkpoint changed')
    if backend == 'rnaformer':
        from RNAformer.model.RNAformer import RiboFormer
        from RNAformer.utils.configuration import Config
        config = Config(config_file=spec['config'])
        # The official implementation automatically supports non-FlashAttention.
        # Keep the published six recycling steps and all trained parameters.
        model = RiboFormer(config.RNAformer)
        model.load_state_dict(torch.load(spec['checkpoint'], map_location='cpu', weights_only=True), strict=True)
        model.eval().cuda().bfloat16()
        mapping = {c: i for i, c in enumerate('ACGUN')}
        def predict(item):
            seq = torch.tensor([[mapping.get(c, 4) for c in item['sequence']]], device='cuda')
            length = torch.tensor([seq.shape[1]], device='cuda')
            pdb_sample = torch.zeros((1, 1), device='cuda', dtype=torch.bfloat16)
            with torch.no_grad():
                logits, mask = model(seq, length, pdb_sample)
                prediction = torch.sigmoid(logits[0, :, :, -1]) > 0.5
            return torch.nonzero(torch.triu(prediction, 1), as_tuple=False).cpu().tolist()
        return predict
    if backend == 'cnnfold':
        import models  # official classes required for the released serialized model
        scope = {'torch': torch, 'np': np, 'device': torch.device('cuda')}
        tree = ast.parse((Path(spec['source']) / 'utils.py').read_text())
        # Reuse the original inference functions without importing unused
        # torchvision training/plotting helpers. AST bodies remain unchanged.
        assignments = [n for n in tree.body if isinstance(n, ast.Assign) and
                       any(isinstance(t, ast.Name) and t.id == 'bases' for t in n.targets)]
        functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('create_matrix', 'binarize')]
        exec(compile(ast.Module(body=assignments+functions, type_ignores=[]), '<official_CNNFold_inference>', 'exec'), scope)
        model = torch.load(spec['checkpoint'], map_location='cuda', weights_only=False)
        # The official test script leaves the serialized module mode intact
        # (its cnn1.eval() call is commented out). Preserve that BN behavior.
        model.to('cuda')
        print('CNNFold serialized training flag:', model.training, flush=True)
        def predict(item):
            with torch.no_grad():
                features = scope['create_matrix'](item['sequence']).unsqueeze(0)
                raw = model(features, test=True, repeat=2)
                binary = scope['binarize'](raw, 0, use_blossom=False)[0, 0]
            return torch.nonzero(torch.triu(binary, 1), as_tuple=False).cpu().tolist()
        return predict
    if backend == 'ufold':
        from Network import U_Net
        from ufold.utils import seq_encoding
        from ufold.data_generator import Dataset_Cut_concat_new
        import ufold.data_generator as generator_module
        from ufold.postprocess import postprocess_new
        model=U_Net(img_ch=17).cuda()
        model.load_state_dict(torch.load(spec['checkpoint'],map_location='cpu',weights_only=True),strict=True)
        # Both official ufold_test.py and ufold_predict.py deliberately use
        # batch statistics during prediction; this architecture has no dropout.
        model.train()
        # The released feature builder fails on its own all-zero N encoding.
        # Extend only that missing case: an unknown has stacking energy zero.
        original=generator_module.creatmat
        source=textwrap.dedent(inspect.getsource(original))
        old="data = ''.join(['AUCG'[list(d).index(1)] for d in data])"
        new="data = ''.join(['AUCG'[list(d).index(1)] if 1 in list(d) else 'N' for d in data])"
        if old not in source:raise RuntimeError('Unexpected UFold stacking implementation')
        scope=dict(original.__globals__)
        exec(compile(source.replace(old,new),'<UFold_unknown_stacking>', 'exec'),scope)
        known=seq_encoding('AUGCCGUA')
        if not torch.equal(original(known),scope['creatmat'](known)):
            raise RuntimeError('UFold known-base stacking features changed')
        generator_module.creatmat=scope['creatmat']
        class OneInput:
            len=1
            def get_one_sample(self,index):return self.encoding,self.length,self.name
        source=OneInput()
        generator=Dataset_Cut_concat_new(source)
        def predict(item):
            source.encoding=seq_encoding(item['sequence']);source.length=len(item['sequence']);source.name=item['id']
            features,length,encoding,name=generator[0]
            matrix=torch.as_tensor(features,dtype=torch.float32,device='cuda').unsqueeze(0)
            seq=torch.zeros((1,features.shape[1],4),dtype=torch.float32,device='cuda')
            seq[0,:length]=torch.as_tensor(encoding[:length],dtype=torch.float32,device='cuda')
            with torch.no_grad():raw=model(matrix)
            processed=postprocess_new(raw,seq,0.01,0.1,100,1.6,True,1.5)
            binary=processed[0,:length,:length]>0.5
            return torch.nonzero(torch.triu(binary,1),as_tuple=False).detach().cpu().tolist()
        return predict
    raise ValueError('Neural comparison adapter not prepared: ' + backend)


def work(index, smoke):
    checked_plan()
    specs = json.loads((OUT / 'external_models.json').read_text())
    model_index, shard = divmod(index-1, SHARDS)
    spec = specs[model_index]
    items = sorted(read_jsonl(OUT / 'panel.jsonl'), key=lambda r: (-len(r['sequence']), r['id']))
    items = [r for i, r in enumerate(items) if i % SHARDS == shard]
    if smoke:
        ambiguous = next(r for r in items if set(r['sequence'])-set('ACGU'))
        items = [items[0], items[len(items)//2], items[-1], ambiguous]
        items = list({r['id']: r for r in items}.values())
    else:
        marker = OUT / 'smoke' / spec['model'] / 'shard-0-complete.json'
        if not marker.exists() or not json.loads(marker.read_text()).get('success'):
            raise RuntimeError('External model smoke check not complete')
    path = OUT / ('smoke' if smoke else 'predictions') / spec['model'] / f'shard-{shard}.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    native_folder = OUT / 'native' / spec['model'] / f'shard-{shard}'
    native_folder.mkdir(parents=True, exist_ok=True)
    predict = predictor(spec, native_folder)
    done = {r['id'] for r in read_jsonl(path)} if path.exists() else set()
    if done - {r['id'] for r in items}:
        raise ValueError('Unexpected cached comparison entries')
    start = time.monotonic()
    with path.open('a') as handle:
        for item in items:
            if item['id'] in done: continue
            pairs = predict(item)
            row = {'id': item['id'], 'sequence_sha256': item['sequence_sha256'],
                   'length': len(item['sequence']), 'predicted_pairs': pairs,
                   **pair_metrics(item['pairs'], pairs)}
            handle.write(json.dumps(row, allow_nan=False)+'\n');handle.flush();done.add(item['id'])
            if len(done) % 100 == 0:
                print(spec['model'], shard, len(done), '/', len(items), f'{time.monotonic()-start:.1f}s', flush=True)
    write_json(path.with_name(f'shard-{shard}-complete.json'), {'success': True, 'n': len(items),
               'model': spec['model'], 'predictions_sha256': sha(path), 'seconds': time.monotonic()-start})
    print('Completed', spec['model'], shard, len(items), flush=True)


if __name__ == '__main__':
    action = sys.argv[1]
    raw = os.environ.get('SGE_TASK_ID', '1')
    if action == 'prepare': prepare()
    else: work(int(raw) if raw.isdigit() else 1, action == 'smoke')
