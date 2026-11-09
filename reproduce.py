"""Command-line interface for data preparation, retraining and result replay."""

import argparse
import fcntl
import json
import os
from pathlib import Path

from data_sources import ROOT, digest, download
from panels import bpseq, efold_test, efold_train, external, rivas, rows, write_rows


def prepare(work, training, rnagym):
    raw = work / "downloads"
    prepared = work / "prepared"
    groups = {"efold-test", "external", "rivas"}
    if training:
        groups.add("efold")
    if rnagym:
        groups.add("rnagym")
    download(raw, groups)
    write_rows(prepared / "efold.jsonl", efold_test(raw / "efold"))
    write_rows(prepared / "external.jsonl", external(raw))
    rivas(raw / "Rivas.tar.gz", prepared / "rivas")
    if training:
        expected = json.loads((ROOT / "protocol/training-inputs.json").read_text())
        if not all((prepared / name).exists() and digest(prepared / name) == sha for name, sha in expected.items()):
            efold_train(raw / "efold", prepared)
        for name, sha in expected.items():
            if digest(prepared / name) != sha:
                raise ValueError(f"Reconstructed training input differs: {name}")
    return prepared


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "verify", "prepare"), nargs="?", default="train")
    parser.add_argument("--work-dir", type=Path, default=ROOT / "runs" / "round2")
    parser.add_argument("--suite", choices=("all", "primary", "rnagym", "synthetic", "joint", "eternafold"),
                        default="all", help="Select a training suite (default: all)")
    parser.add_argument("--eternafold-ranks", type=int, default=1,
                        help="CPU ranks for EternaFold. Use MPI when greater than 1.")

    args = parser.parse_args()
    if args.eternafold_ranks < 1:
        parser.error("--eternafold-ranks must be positive")
    if args.mode == "train" and args.suite != "eternafold":
        import torch

        if not torch.cuda.is_available():
            raise SystemExit("A CUDA GPU is required for train. Use verify to rescore archived predictions on CPU.")

    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    with (work / "run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"Another reproduction is active in {work}")
        primary = args.suite in ("all", "primary")
        prepared = prepare(work, training=args.mode != "verify" and primary,
                           rnagym=args.mode != "verify" and args.suite != "primary")
        if args.mode == "prepare":
            print(f"Verified and prepared study inputs in {prepared}")
            return
        from paper import PRIMARY, REF, regenerate
        from score import read_predictions

        archived = args.mode == "verify"
        out = work / ("archived-results" if archived else "retrained-results")
        os.environ.setdefault("MPLBACKEND", "Agg")
        if archived or primary:
            panel = {row["id"]: row for name in ("efold", "external")
                     for row in rows(prepared / f"{name}.jsonl")}
            if archived:
                models = {name: read_predictions(REF / f"predictions/{name}.jsonl.gz") for name in PRIMARY}
            else:
                from train_primary import predict, train

                paths = work / "checkpoints"
                output = work / "predictions"
                predict(train(prepared, paths), panel, paths, output)
                models = {name: read_predictions(output / f"{name}.jsonl") for name in PRIMARY}
            regenerate(panel, models, out, archived)
        if not archived and args.suite != "primary":
            from train_rnagym import run

            mx_suites = {"synthetic", "joint", "rnagym"}
            modes = mx_suites if args.suite == "all" else mx_suites & {args.suite}
            eternafold = args.eternafold_ranks if args.suite in ("all", "eternafold") else None
            run(prepared, work / "downloads/rnagym", work / "checkpoints", out, modes,
                gpu=-1 if args.suite == "eternafold" else 0, eternafold=eternafold)
        from score import table

        rivas_stats = []
        for name in ("TrainSetA", "TrainSetB", "TestSetB"):
            files = (prepared / "rivas" / f"{name}.lst").read_text().splitlines()
            lengths = [len(bpseq(Path(path).read_text())[0]) for path in files]
            rivas_stats.append(dict(dataset=name, rnas=len(lengths), nucleotides=sum(lengths),
                                    min_length=min(lengths), max_length=max(lengths)))
        table(out / "supp-table-2_rivas.tsv", rivas_stats)
        print(f"Numerical tables and plots written to {out}")


if __name__ == "__main__":
    main()
