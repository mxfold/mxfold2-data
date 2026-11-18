"""Prepare the archived inputs or retrain the MXfold2 0.3.0 default model."""

from __future__ import annotations

import argparse
import fcntl
import gzip
import json
import math
from pathlib import Path
import shutil
import time

from scripts.efold_hf.workflow import digest

REPO = Path(__file__).resolve().parent
RECIPE = REPO / 'default-model'


def copy_verified(source, target, expected):
    if target.exists():
        if digest(target) != expected:
            raise ValueError(f'Existing input differs from the archived recipe: {target}')
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + '.tmp')
    open_source = gzip.open if source.suffix == '.gz' else open
    try:
        with open_source(source, 'rb') as archived, temporary.open('wb') as ready:
            shutil.copyfileobj(archived, ready, length=1 << 20)
        if digest(temporary) != expected:
            raise ValueError(f'Incorrect SHA-256 for archived input: {source}')
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def prepare(root):
    for name in ('plan.json', 'baseline.conf', 'data-complete.json'):
        archive = RECIPE / name
        copy_verified(archive, root / name, digest(archive))
    ready = json.loads((RECIPE / 'data-complete.json').read_text())
    for name, expected in ready['sha256'].items():
        archive = RECIPE / name
        if not archive.is_file():
            archive = archive.with_name(archive.name + '.gz')
        copy_verified(archive, root / name, expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('train', 'prepare'), nargs='?', default='train')
    parser.add_argument('--work-dir', type=Path, default=REPO / 'runs/default-model-v030')
    parser.add_argument('--hours', type=float, help='Save and stop training after this many hours')
    args = parser.parse_args()
    if args.hours is not None and (args.mode != 'train' or not math.isfinite(args.hours) or args.hours <= 0):
        parser.error('--hours requires train mode and a positive, finite number')

    root = args.work_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'worker.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error(f'Another worker is using {root}')
        prepare(root)
        print(f'Verified the prepared training inputs in {root}', flush=True)
        if args.mode == 'prepare':
            return

        import mxfold2
        import torch

        from scripts.default_model import train

        if Path(mxfold2.__file__).resolve().parent != REPO / 'source/mxfold2':
            parser.error('Run with the frozen MXfold2 source. Use the environment created by reproduce-default.sh.')
        if not torch.cuda.is_available():
            parser.error('Training the default model requires a CUDA GPU')
        deadline = time.monotonic() + args.hours * 3600 if args.hours is not None else math.inf
        if train.work(root, gpu=0, deadline=deadline):
            print(f'Trained model and prediction configuration are in {root / "models"}')
        else:
            print(f'Checkpoint saved in {root}. Run the same command to resume.')


if __name__ == '__main__':
    main()
