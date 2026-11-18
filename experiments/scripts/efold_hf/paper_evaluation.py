#!/usr/bin/env python3
"""Reproduce the eFold paper's structure evaluation on a fixed <=1000 nt panel.

The old experiment is read-only.  This script reuses its 711 saved predictions,
then predicts the 3,046 ArchiveII entries that its conservative filter omitted.
The author's public comparison predictions are checked against the same labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
RAW = PROJECT / "data/efold_hf/raw"
PAPER = PROJECT / "data/efold_hf/paper_2026"
OLD = PROJECT / "qsub_logs/efold_hf/bprna100_ft5_replay20_20261024"
FT = PROJECT / "qsub_logs/efold_hf/bprna10_restored_ft5_replay20_20261026"
OUTPUT = PROJECT / "qsub_logs/efold_hf/paper_aligned_20261028"
DATASETS = ("PDB", "archiveII", "viral_fragments", "lncRNA")
ARMS = ("pretrain", "mw_rank_restored", "nll_restored", "shape_fy_restored")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def pair_metrics(reference, prediction):
    """Table notebooks: undefined precision/recall are NaN and skipped in means."""
    ref = {tuple(sorted(map(int, p))) for p in reference}
    pred = {tuple(sorted(map(int, p))) for p in prediction}
    tp = len(ref & pred)
    both_empty = not ref and not pred
    return {
        "tp": tp, "fp": len(pred) - tp, "fn": len(ref) - tp,
        "precision": tp / len(pred) if pred else (1.0 if both_empty else None),
        "recall": tp / len(ref) if ref else (1.0 if both_empty else None),
        "f1": 2 * tp / (len(ref) + len(pred)) if ref or pred else 1.0,
    }


def panel():
    rows = []
    for source in DATASETS:
        data = json.loads((RAW / (source + ".json")).read_text())
        for ref, item in data.items():
            sequence = item["sequence"].upper().replace("T", "U")
            if len(sequence) > 1000:
                continue
            if not sequence or set(sequence) - set("ACGU"):
                raise ValueError(f"Invalid sequence: {source}:{ref}")
            pairs = sorted({tuple(sorted(map(int, p))) for p in item["structure"]})
            if any(i < 0 or j >= len(sequence) or i == j for i, j in pairs):
                raise ValueError(f"Invalid reference pairs: {source}:{ref}")
            rows.append({"id": source + ":" + ref, "source": source,
                         "reference": ref, "sequence": sequence, "pairs": pairs})
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate source/reference")
    return rows


def mean_defined(rows, key):
    values = [r[key] for r in rows if r[key] is not None]
    return (statistics.fmean(values) if values else None), len(values)


def aggregate(rows):
    ans = {}
    for source in DATASETS:
        selected = [r for r in rows if r["source"] == source]
        metrics = {key: mean_defined(selected, key) for key in ("precision", "recall", "f1")}
        ans[source] = {"n": len(selected)}
        for key, (value, count) in metrics.items():
            ans[source][key] = value
            ans[source]["n_" + key] = count
    return ans


def previous_paths(arm):
    if arm == "pretrain":
        root, epoch = OLD, 10
    else:
        root, epoch = FT, 5
    return root, root / "runs" / arm / f"epoch-{epoch:03d}.pt", root / "evaluation" / arm / f"epoch-{epoch:03d}" / "per_sequence.jsonl"


def load_old(arm, refs):
    _, ckpt, previous = previous_paths(arm)
    marker = ckpt.with_name(ckpt.stem + "-complete.json")
    if json.loads(marker.read_text())["checkpoint_sha256"] != digest(ckpt):
        raise RuntimeError("Checkpoint hash mismatch: " + str(ckpt))
    out = {}
    for row in read_jsonl(previous):
        ref = refs[row["id"]]
        metrics = pair_metrics(ref["pairs"], row["predicted_pairs"])
        if abs(metrics["f1"] - row["f1"]) > 1e-12:
            raise ValueError("Old prediction/reference mismatch: " + row["id"])
        if row["id"] in out:
            raise ValueError("Duplicate old prediction: " + row["id"])
        out[row["id"]] = {"id": row["id"], "source": ref["source"],
                          "predicted_pairs": row["predicted_pairs"], **metrics}
    return out


def prepare():
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.feather as feather

    if OUTPUT.exists():
        raise FileExistsError(f"Output already exists: {OUTPUT}")
    # This checks exact-match F1, including the eFold empty-structure case.
    assert pair_metrics([], [])["f1"] == 1
    assert pair_metrics([[0, 4]], [])["precision"] is None
    assert pair_metrics([], [[0, 4]])["recall"] is None
    assert pair_metrics([[0, 4]], [[0, 5]])["f1"] == 0
    assert pair_metrics([[0, 4]], [[4, 0]])["f1"] == 1

    items = panel()
    refs = {r["id"]: r for r in items}
    counts = Counter(r["source"] for r in items)
    assert dict(counts) == {"PDB": 355, "archiveII": 3359, "viral_fragments": 40, "lncRNA": 3}
    old = {arm: load_old(arm, refs) for arm in ARMS}
    existing_ids = set(old["pretrain"])
    if len(existing_ids) != 711 or any(set(x) != existing_ids for x in old.values()):
        raise ValueError("Original model predictions do not use the same 711 sequences")
    extra = [r for r in items if r["id"] not in existing_ids]
    assert len(extra) == 3046 and all(r["source"] == "archiveII" for r in extra)

    # Author eFold prediction files contain both reference ID and input sequence.
    # This is independent of the reported scalar F1 and detects changed inputs.
    for filename in ("test_results_PT.feather", "test_results_PT+FT.feather"):
        table = feather.read_table(PAPER / filename, columns=["reference", "sequence"])
        entries = table.to_pylist()
        author_seq = {e["reference"]: e["sequence"] for e in entries}
        if len(author_seq) != len(entries):
            raise ValueError("Duplicate author reference: " + filename)
        for item in items:
            if author_seq.get(item["reference"]) != item["sequence"]:
                raise ValueError("Author sequence mismatch: " + item["id"])

    author_table = feather.read_table(PAPER / "results_benchmark_algos.feather",
                                      columns=["reference", "dataset", "model", "length", "F1", "structure"])
    author_table = author_table.filter(pc.and_(
        pc.is_in(author_table["model"], value_set=pa.array(["EternaFold", "MXFold2"])),
        pc.less_equal(author_table["length"], 1000)))
    author = author_table.to_pylist()
    author_rows = []
    for row in author:
        if row["model"] not in ("EternaFold", "MXFold2") or row["length"] > 1000:
            continue
        rid = row["dataset"] + ":" + row["reference"]
        if rid not in refs:
            raise ValueError("Author test reference missing: " + rid)
        ref = refs[rid]
        if row["length"] != len(ref["sequence"]):
            raise ValueError("Author length mismatch: " + rid)
        metrics = pair_metrics(ref["pairs"], row["structure"])
        if abs(metrics["f1"] - row["F1"]) > 1e-6:
            raise ValueError("Author score mismatch: " + rid)
        author_rows.append({"model": row["model"], "id": rid, "source": ref["source"], **metrics})
    for model in ("EternaFold", "MXFold2"):
        ids = [r["id"] for r in author_rows if r["model"] == model]
        if len(ids) != len(refs) or set(ids) != set(refs):
            raise ValueError("Author predictions do not cover panel: " + model)

    train_sets = {name: {row["sequence"] for row in read_jsonl(OLD / "data" / (name + ".jsonl"))}
                  for name in ("bprna", "chemical")}
    overlap = {source: {name: sum(r["sequence"] in seqs for r in items if r["source"] == source)
                        for name, seqs in train_sets.items()} for source in DATASETS}
    if any(v for dataset in overlap.values() for v in dataset.values()):
        raise ValueError("Exact sequence overlap with this experiment's training data")

    OUTPUT.mkdir(parents=True)
    (OUTPUT / "predictions").mkdir()
    (OUTPUT / "scheduler").mkdir()
    with (OUTPUT / "panel.jsonl").open("w") as handle:
        for item in items:
            handle.write(json.dumps(item) + "\n")
    with (OUTPUT / "author_comparators.jsonl").open("w") as handle:
        for row in author_rows:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
    manifest = {
        "panel_counts": dict(counts), "old_panel_count": len(existing_ids),
        "new_archiveII_count": len(extra), "training_exact_overlap": overlap,
        "metric": "Mean per-sequence exact-pair precision/recall/F1; both empty=1; undefined precision/recall excluded from mean",
        "max_length": 1000,
        "inputs": {str(p.relative_to(PROJECT)): digest(p) for p in [
            *(RAW / (name + ".json") for name in DATASETS),
            PAPER / "results_benchmark_algos.feather", PAPER / "test_results_PT.feather",
            PAPER / "test_results_PT+FT.feather", *(previous_paths(arm)[1] for arm in ARMS),
            *(previous_paths(arm)[2] for arm in ARMS), Path(__file__)]},
        "author_eFold_sequence_matches": len(items),
        "author_structure_f1_matches": {m: sum(r["model"] == m for r in author_rows)
                                        for m in ("EternaFold", "MXFold2")},
        "warning": "No homology screening was performed; exact overlap is not proof of family independence.",
    }
    manifest["panel_sha256"] = digest(OUTPUT / "panel.jsonl")
    manifest["author_comparators_sha256"] = digest(OUTPUT / "author_comparators.jsonl")
    write_json(OUTPUT / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)


def verify_manifest():
    manifest = json.loads((OUTPUT / "manifest.json").read_text())
    if digest(OUTPUT / "panel.jsonl") != manifest["panel_sha256"]:
        raise RuntimeError("Panel changed")
    if digest(OUTPUT / "author_comparators.jsonl") != manifest["author_comparators_sha256"]:
        raise RuntimeError("Author comparison changed")
    for path, expected in manifest["inputs"].items():
        if digest(PROJECT / path) != expected:
            raise RuntimeError("Input changed: " + path)
    return manifest


def predict(arm):
    import torch

    verify_manifest()
    items = read_jsonl(OUTPUT / "panel.jsonl")
    refs = {r["id"]: r for r in items}
    previous = load_old(arm, refs)
    root, checkpoint, _ = previous_paths(arm)
    sys.path.insert(0, str(root / "source"))
    from scripts.efold_hf import workflow as wf
    wf.verify(root)
    wf.seed_all(wf.SEED)
    model, _, _, _, _, _ = wf.build(root, "pretrain", 0)
    wf.restore(checkpoint, model)
    model.eval()
    path = OUTPUT / "predictions" / (arm + ".jsonl")
    if path.exists():
        done = {r["id"]: r for r in read_jsonl(path)}
    else:
        done = {}
    expected = {r["id"] for r in items} - set(previous)
    if set(done) - expected:
        raise ValueError("Unexpected cached predictions")

    def infer(sequence):
        result = model([sequence], return_partfunc=False)
        bp = result[2][0]
        return [[i - 1, int(j) - 1] for i, j in enumerate(bp) if i > 0 and j > i]

    with torch.no_grad():
        # Replay cached predictions spanning all four test sets before inference.
        for source in DATASETS:
            sample = next(r for r in items if r["source"] == source and r["id"] in previous)
            if infer(sample["sequence"]) != previous[sample["id"]]["predicted_pairs"]:
                raise RuntimeError("Restored model differs from saved prediction: " + sample["id"])
        with path.open("a") as handle:
            for item in items:
                if item["id"] not in expected or item["id"] in done:
                    continue
                pred = infer(item["sequence"])
                row = {"id": item["id"], "source": item["source"], "predicted_pairs": pred,
                       **pair_metrics(item["pairs"], pred)}
                handle.write(json.dumps(row, allow_nan=False) + "\n")
                handle.flush()
                done[item["id"]] = row
                if len(done) % 100 == 0:
                    print(arm, len(done), "/", len(expected), flush=True)
    if set(done) != expected:
        raise RuntimeError("Evaluation incomplete")
    write_json(OUTPUT / "predictions" / (arm + "-complete.json"), {
        "success": True, "n": len(done), "checkpoint_sha256": digest(checkpoint),
        "predictions_sha256": digest(path), "panel_sha256": digest(OUTPUT / "panel.jsonl")})
    print(arm, "complete", len(done), flush=True)


def summarize():
    verify_manifest()
    items = read_jsonl(OUTPUT / "panel.jsonl")
    refs = {r["id"]: r for r in items}
    author = read_jsonl(OUTPUT / "author_comparators.jsonl")
    result = {"panel_counts": dict(Counter(r["source"] for r in items)), "models": {}}
    for model in ("EternaFold", "MXFold2"):
        result["models"]["paper_" + model] = aggregate([r for r in author if r["model"] == model])
    for arm in ARMS:
        previous = load_old(arm, refs)
        extra_path = OUTPUT / "predictions" / (arm + ".jsonl")
        extra = read_jsonl(extra_path) if extra_path.exists() else []
        if len({r["id"] for r in extra}) != len(extra):
            raise ValueError("Duplicate predictions: " + arm)
        if set(previous) & {r["id"] for r in extra}:
            raise ValueError("Old and new predictions overlap")
        for row in extra:
            expected = pair_metrics(refs[row["id"]]["pairs"], row["predicted_pairs"])
            if expected != {k: row[k] for k in expected}:
                raise ValueError("Stored score mismatch: " + row["id"])
        complete = OUTPUT / "predictions" / (arm + "-complete.json")
        if complete.exists():
            marker = json.loads(complete.read_text())
            if marker["predictions_sha256"] != digest(extra_path):
                raise RuntimeError("Completed predictions changed")
        result["models"][arm] = {"complete": complete.exists(),
                                  "datasets": aggregate(list(previous.values()) + extra)}
    write_json(OUTPUT / "results.json", result)
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "predict", "summarize"))
    parser.add_argument("--arm", choices=ARMS)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    elif args.action == "predict":
        if not args.arm:
            parser.error("predict requires --arm")
        predict(args.arm)
    else:
        summarize()


if __name__ == "__main__":
    main()
