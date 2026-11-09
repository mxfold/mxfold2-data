"""Rebuild the patched EternaFold CPU comparison on public input data."""

import hashlib
import os
from pathlib import Path
from statistics import fmean
import subprocess

from data_sources import ROOT, digest
from score import pair_counts, pairs_metric, table
from scripts.eternafold.packing import PackWriter, export_native_chemical, pack_list, read_bpseq, read_params
from scripts.eternafold.patch_upstream import patch
from scripts.eternafold.project_structure import prune_record

COMMIT = "87b9aac55cee14fd562049d08f7b92d3131f10ce"
REPO = "https://github.com/WaymentSteeleLab/EternaFold.git"
EXPECTED_PACKS = {
    "structure": "39fcbf01b698b70e4873c5610486a83a4d803cecbbc2f32ec96868ec436b2df2",
    "synthetic": "20888685686b15da69f9cd4753f15899ace5f2f5b4ad145aacdea35770c8fa4a",
    "rnagym": "ea1b6f45302b9dbf771058c4f168166c8a0cdca485ce00a90cdcb416a9dde413",
}


def build(work, ranks):
    upstream = work / "upstream"
    if not (upstream / ".git").exists():
        subprocess.run(["git", "init", str(upstream)], check=True)
    checked = subprocess.run(["git", "-C", str(upstream), "rev-parse", "HEAD"],
                             text=True, capture_output=True)
    if checked.stdout.strip() != COMMIT:
        subprocess.run(["git", "-C", str(upstream), "fetch", "--depth=1", REPO, COMMIT], check=True)
        subprocess.run(["git", "-C", str(upstream), "checkout", "--detach", COMMIT], check=True)
    tag = "serial" if ranks == 1 else "mpi"
    target = work / f"native-{tag}"
    done = target / "patch-complete"
    if not done.exists():
        import shutil

        if target.exists():
            shutil.rmtree(target)
        patch(upstream / "src", target)
        done.touch()
    flags = [] if ranks == 1 else ["CXX=mpic++", "OTHERFLAGS=-DMULTI"]
    subprocess.run(["make", "-C", str(target), "-j4", "contrafold", *flags], check=True)
    return target / "contrafold"


def data_packs(prepared, work):
    work.mkdir(parents=True, exist_ok=True)
    structure = work / "TrainSetA.efpack"
    trainseta = prepared / "rivas/TrainSetA.lst"
    if not structure.exists() or digest(structure) != EXPECTED_PACKS["structure"]:
        writer = PackWriter(structure)
        changed = before = after = 0
        for i, path in enumerate(trainseta.read_text().splitlines()):
            sequence, original = read_bpseq(path)
            projected, _ = prune_record(sequence, original)
            changed += projected != original
            before += sum(j > i for i, j in enumerate(original, 1))
            after += sum(j > i for i, j in enumerate(projected, 1))
            writer.add(f"A{i:05d}", sequence, pairs=projected, weight=1. / 3166)
        if (writer.count, changed, before, after) != (3166, 453, 167320, 166055):
            raise ValueError("Unexpected TrainSetA projection")
        writer.finish()
    syn = work / "Synthetic-2A3.efpack"
    gym = work / "RNAgym.efpack"
    for path, folder, key, modifier in (
        (syn, prepared / "synthetic/2A3", "synthetic", "2A3"),
        (gym, prepared / "rnagym", "rnagym", None),
    ):
        if not path.exists() or digest(path) != EXPECTED_PACKS[key]:
            export_native_chemical(folder, path, modifier=modifier)
    for key, path in (("structure", structure), ("synthetic", syn), ("rnagym", gym)):
        if digest(path) != EXPECTED_PACKS[key]:
            raise ValueError(f"EternaFold input {key} differs from the study")
    return structure, syn, gym


def native(binary, ranks, folder, args, env=None):
    folder.mkdir(parents=True, exist_ok=True)
    command = [str(binary)] if ranks == 1 else ["mpirun", "-np", str(ranks), str(binary)]
    with (folder / "stdout.log").open("a") as out, (folder / "stderr.log").open("a") as err:
        subprocess.run(command + [str(x) for x in args], cwd=folder, stdout=out, stderr=err,
                       env={**os.environ, **(env or {})}, check=True)


def train(binary, ranks, work, condition, structure, chemical):
    work.mkdir(parents=True, exist_ok=True)
    inputs = work / "train.lst"
    files = (structure, chemical) if chemical else (structure,)
    inputs.write_text("".join(f"@EFPACK:{p.resolve()}\n" for p in files))
    zero = ROOT / "protocol/eternafold-zero.params"
    signature = hashlib.sha256("".join(digest(p) for p in (*files, zero)).encode()).hexdigest()
    marker = work / "lbfgs.state.done"
    params = work / "optimize.params.final"
    if not marker.exists() or not params.exists():
        native(binary, ranks, work, [
            "train", "--numdatasources", 5, "--regularize", 1. / 3166,
            "--maxiter", 200, "--hyperparam_data", .5, "--examplefile", inputs,
            "--initweights", zero,
        ], {"EF_SIGNATURE": signature, "EF_LBFGS_STATE": str(work / "lbfgs.state")})
    if not marker.exists() or marker.read_text().strip() != signature:
        raise ValueError(f"EternaFold training did not complete for {condition}")
    read_params(params)
    return params


def prediction_pairs(binary, params, ranks, entries, work):
    work.mkdir(parents=True, exist_ok=True)
    target = work / "bpseq"
    target.mkdir(exist_ok=True)
    path = work / "queries.efpack"
    writer = PackWriter(path)
    for rid, sequence in entries:
        writer.add(rid, sequence)
    writer.finish()
    destination = target if len(entries) > 1 else target / entries[0][0]
    native(binary, ranks, work, [
        "predict", "--numdatasources", 5, "--params", params, "--gamma", 6,
        "--bpseq", destination, "--examplefile", pack_list(path),
    ])
    for rid, sequence in entries:
        predicted_sequence, partners = read_bpseq(target / rid)
        if predicted_sequence != sequence:
            raise ValueError(f"EternaFold prediction sequence differs: {rid}")
        yield [(i - 1, j - 1) for i, j in enumerate(partners, 1) if j > i]


def run(prepared, work, output, ranks):
    binary = build(work, ranks)
    structure, syn, gym = data_packs(prepared, work / "data")
    evaluation = prepared / "rivas/TestSetB.lst"
    references = [read_bpseq(p) for p in evaluation.read_text().splitlines()]
    entries = [(f"TestSetB_{i:05d}", seq) for i, (seq, _) in enumerate(references)]
    results = []
    for condition, chemical in (("pretrain", None), ("joint_synthetic_2A3", syn), ("joint_rnagym_all", gym)):
        params = train(binary, ranks, work / condition, condition, structure, chemical)
        predictions = prediction_pairs(binary, params, ranks, entries, work / condition / "TestSetB")
        scores = []
        for (_, partners), predicted in zip(references, predictions, strict=True):
            ref = [(i - 1, j - 1) for i, j in enumerate(partners, 1) if j > i]
            scores.append(pairs_metric(*pair_counts(ref, predicted), empty=0))
        results.append(dict(model=condition, n=len(scores),
                            PPV=fmean(r["precision"] for r in scores),
                            SEN=fmean(r["recall"] for r in scores),
                            F1=fmean(r["f1"] for r in scores)))
    table(output / "eternafold-TestSetB.tsv", results)
    return binary, params
