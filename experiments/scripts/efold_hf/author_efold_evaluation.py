#!/usr/bin/env python3
"""Evaluate the author's public eFold model on the fixed paper comparison panel."""

import argparse
import json
import sys
from pathlib import Path

from paper_evaluation import OUTPUT, PROJECT, aggregate, digest, pair_metrics, read_jsonl, write_json

PAPER = PROJECT / "data/efold_hf/paper_2026"
SOURCE = PAPER / "author_efold_source"
DEPENDENCIES = PAPER / "author_dependencies"
DIRECT = OUTPUT / "author_efold_direct"
WEIGHTS = SOURCE / "efold/resources/efold_weights.pt"
SHARDS = 4


def items_for_shard(index):
    rows = read_jsonl(OUTPUT / "panel.jsonl")
    ordered = sorted(rows, key=lambda row: (-len(row["sequence"]), row["id"]))
    return [row for i, row in enumerate(ordered) if i % SHARDS == index]


def prepare():
    smoke = json.loads((OUTPUT / "author_efold_smoke.json").read_text())
    if not smoke["all_agree"] or smoke["n"] != 18 or smoke["missing_keys"] or smoke["unexpected_keys"]:
        raise RuntimeError("The public eFold smoke test did not reproduce the paper figure")
    if DIRECT.exists():
        raise FileExistsError("Author evaluation already prepared")
    DIRECT.mkdir()
    files = [p for p in (SOURCE / "efold").rglob("*.py") if p.is_file()]
    files += [WEIGHTS, OUTPUT / "panel.jsonl", OUTPUT / "author_efold_figure_inferred.jsonl", Path(__file__)]
    plan = {"source_commit": json.loads((SOURCE / "provenance.json").read_text())["commit"],
            "source_and_input_sha256": {str(p): digest(p) for p in files},
            "shards": SHARDS, "shard_sizes": [len(items_for_shard(i)) for i in range(SHARDS)],
            "dependencies": {"einops": "0.8.1", "lightning": "2.5.5", "torchmetrics": "1.7.4",
                             "huggingface_hub": "0.34.4", "rouskinhf": "0.4.8"},
            "output": "Sequence-only inference with the author's published model and default postprocessing; macro per-RNA exact-pair metrics."}
    write_json(DIRECT / "plan.json", plan)
    print(json.dumps(plan, indent=2), flush=True)


def verify():
    plan = json.loads((DIRECT / "plan.json").read_text())
    for p, h in plan["source_and_input_sha256"].items():
        if digest(Path(p)) != h:
            raise RuntimeError("Changed eFold input: " + p)
    return plan


def predict(index):
    import torch

    verify()
    if index not in range(SHARDS):
        raise ValueError("Invalid shard")
    sys.path[:0] = [str(DEPENDENCIES), str(SOURCE)]
    from efold.api.run import _predict_structure
    from efold.models import create_model

    figure = {r["id_inferred"]: r["f1"] for r in read_jsonl(OUTPUT / "author_efold_figure_inferred.jsonl")
              if r["model"] == "eFold"}
    data = items_for_shard(index)
    expected = {r["id"] for r in data}
    output = DIRECT / f"shard-{index}.jsonl"
    cached = read_jsonl(output) if output.exists() else []
    done = {r["id"]: r for r in cached}
    if len(done) != len(cached) or set(done) - expected:
        raise ValueError("Invalid existing shard data")
    model = create_model(model="efold", ntoken=5, d_model=64, c_z=32, d_cnn=64,
                         num_blocks=4, no_recycles=0, dropout=0, lr=3e-4, weight_decay=0, gamma=0.995)
    model.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=True), strict=True)
    model.eval().to("cuda")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    with output.open("a") as handle:
        for item in data:
            if item["id"] in done:
                continue
            pred = [[int(i)-1, int(j)-1] for i, j in _predict_structure(model, item["sequence"], device="cuda")]
            metrics = pair_metrics(item["pairs"], pred)
            row = {"id": item["id"], "source": item["source"], "predicted_pairs": pred, **metrics,
                   "inferred_figure_f1": figure[item["id"]]}
            handle.write(json.dumps(row, allow_nan=False) + "\n")
            handle.flush()
            done[item["id"]] = row
            if len(done) % 50 == 0:
                print(f"shard {index}: {len(done)}/{len(data)}", flush=True)
    if set(done) != expected:
        raise RuntimeError("Incomplete author eFold shard")
    write_json(DIRECT / f"shard-{index}-complete.json", {"success": True, "n": len(done),
               "predictions_sha256": digest(output), "model_sha256": digest(WEIGHTS)})
    print(f"shard {index} complete: {len(done)}", flush=True)


def summarize():
    verify()
    refs = {r["id"]: r for r in read_jsonl(OUTPUT / "panel.jsonl")}
    rows = []
    for i in range(SHARDS):
        path = DIRECT / f"shard-{i}.jsonl"
        marker = json.loads((DIRECT / f"shard-{i}-complete.json").read_text())
        if not marker["success"] or marker["predictions_sha256"] != digest(path):
            raise RuntimeError("Invalid eFold shard")
        part = read_jsonl(path)
        if len(part) != marker["n"] or {r["id"] for r in part} != {r["id"] for r in items_for_shard(i)}:
            raise RuntimeError("Missing or duplicate eFold shard records")
        rows.extend(part)
    if len(rows) != len(refs) or {r["id"] for r in rows} != set(refs):
        raise RuntimeError("eFold results do not cover the panel")
    for row in rows:
        metrics = pair_metrics(refs[row["id"]]["pairs"], row["predicted_pairs"])
        if any(row[k] != value for k, value in metrics.items()):
            raise RuntimeError("eFold stored metric mismatch: " + row["id"])
    different = [r for r in rows if abs(r["f1"] - r["inferred_figure_f1"]) > 1e-6]
    result = {"model": "author_public_eFold", "n": len(rows), "datasets": aggregate(rows),
              "figure_f1_mismatches": len(different),
              "figure_mismatch_examples": [{"id": r["id"], "current_f1": r["f1"],
                                            "inferred_figure_f1": r["inferred_figure_f1"]}
                                           for r in different[:20]]}
    write_json(DIRECT / "summary.json", result)
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "predict", "summarize"))
    parser.add_argument("--shard", type=int)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    elif args.action == "predict":
        if args.shard is None:
            parser.error("--shard is required")
        predict(args.shard)
    else:
        summarize()


if __name__ == "__main__":
    main()
