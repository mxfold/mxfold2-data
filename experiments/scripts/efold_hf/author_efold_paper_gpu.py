#!/usr/bin/env python3
"""Evaluate eFold as in its paper test_step, including GPU postprocessing."""

import argparse
import json
import sys
from pathlib import Path

from paper_evaluation import OUTPUT, PROJECT, aggregate, digest, pair_metrics, read_jsonl, write_json
from author_efold_evaluation import DEPENDENCIES, PAPER, SHARDS, SOURCE, WEIGHTS, items_for_shard, verify

DEST = OUTPUT / "author_efold_paper_gpu"


def prepare():
    old_plan = OUTPUT / "author_efold_direct/plan.json"
    verify()
    smoke_path = OUTPUT / "author_efold_gpu_postprocess_smoke.json"
    smoke = json.loads(smoke_path.read_text()) if smoke_path.exists() else None
    if smoke is not None and smoke["n"] != 12:
        raise RuntimeError("The paper postprocessing smoke test is incomplete")
    if DEST.exists():
        raise FileExistsError("Paper-protocol eFold evaluation already prepared")
    DEST.mkdir()
    write_json(DEST / "plan.json", {
        "old_plan_sha256": digest(old_plan), "script_sha256": digest(Path(__file__)),
        "weights_sha256": digest(WEIGHTS), "panel_sha256": digest(OUTPUT / "panel.jsonl"),
        "figure_sha256": digest(OUTPUT / "author_efold_figure_inferred.jsonl"),
        "shards": SHARDS, "shard_sizes": [len(items_for_shard(i)) for i in range(SHARDS)],
        "smoke_gpu_matches": smoke["gpu_matches"] if smoke else None,
        "smoke_cpu_matches": smoke["cpu_matches"] if smoke else None,
        "protocol": "Author eFold Model.test_step: forward and postprocess on the GPU; scores use 0.5 threshold and exact-pair macro metrics."})


def verify_plan():
    verify()
    p = json.loads((DEST / "plan.json").read_text())
    files = {"old_plan_sha256": OUTPUT / "author_efold_direct/plan.json", "script_sha256": Path(__file__),
             "weights_sha256": WEIGHTS, "panel_sha256": OUTPUT / "panel.jsonl",
             "figure_sha256": OUTPUT / "author_efold_figure_inferred.jsonl"}
    for key, file in files.items():
        if digest(file) != p[key]:
            raise RuntimeError("Paper-protocol input changed: " + str(file))


def predict(index):
    import torch

    verify_plan()
    if index not in range(SHARDS):
        raise ValueError("Invalid shard")
    sys.path[:0] = [str(DEPENDENCIES), str(SOURCE)]
    from efold.core import batch
    from efold.core.embeddings import sequence_to_int
    from efold.models import create_model

    figure = {r["id_inferred"]: r["f1"] for r in read_jsonl(OUTPUT / "author_efold_figure_inferred.jsonl")
              if r["model"] == "eFold"}
    items = items_for_shard(index)
    expected = {r["id"] for r in items}
    path = DEST / f"shard-{index}.jsonl"
    cached = read_jsonl(path) if path.exists() else []
    done = {r["id"]: r for r in cached}
    if len(cached) != len(done) or set(done) - expected:
        raise RuntimeError("Unexpected cached shard results")
    model = create_model(model="efold", ntoken=5, d_model=64, c_z=32, d_cnn=64,
                         num_blocks=4, no_recycles=0, dropout=0, lr=3e-4, weight_decay=0, gamma=0.995)
    model.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=True), strict=True)
    model.eval().to("cuda")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    with path.open("a") as handle, torch.inference_mode():
        for item in items:
            if item["id"] in done:
                continue
            sequence = item["sequence"]
            data = batch.Batch(sequence=sequence_to_int(sequence).unsqueeze(0), reference=[item["id"]],
                               length=[len(sequence)], L=len(sequence), use_error=False,
                               batch_size=1, data_types=["sequence"], dt_count={"sequence": 1}).to("cuda")
            raw = model(data)
            matrix = model.postprocesser.run(raw["structure"], data.get("sequence"))[0]
            pairs = torch.nonzero(torch.triu(matrix, diagonal=1), as_tuple=False).cpu().tolist()
            row = {"id": item["id"], "source": item["source"], "predicted_pairs": pairs,
                   **pair_metrics(item["pairs"], pairs), "inferred_figure_f1": figure[item["id"]]}
            handle.write(json.dumps(row, allow_nan=False) + "\n")
            handle.flush()
            done[item["id"]] = row
            if len(done) % 50 == 0:
                print(f"paper GPU shard {index}: {len(done)}/{len(items)}", flush=True)
    if set(done) != expected:
        raise RuntimeError("Paper GPU shard incomplete")
    write_json(DEST / f"shard-{index}-complete.json", {"success": True, "n": len(done),
               "predictions_sha256": digest(path), "model_sha256": digest(WEIGHTS)})
    print(f"paper GPU shard {index} complete: {len(done)}", flush=True)


def summarize():
    verify_plan()
    refs = {r["id"]: r for r in read_jsonl(OUTPUT / "panel.jsonl")}
    rows = []
    for i in range(SHARDS):
        path = DEST / f"shard-{i}.jsonl"
        marker = json.loads((DEST / f"shard-{i}-complete.json").read_text())
        if not marker["success"] or marker["predictions_sha256"] != digest(path):
            raise RuntimeError("Invalid paper GPU shard")
        part = read_jsonl(path)
        if len(part) != marker["n"] or {r["id"] for r in part} != {r["id"] for r in items_for_shard(i)}:
            raise RuntimeError("Paper GPU shard coverage mismatch")
        rows.extend(part)
    if len(rows) != len(refs) or {r["id"] for r in rows} != set(refs):
        raise RuntimeError("Paper GPU panel coverage mismatch")
    for row in rows:
        metrics = pair_metrics(refs[row["id"]]["pairs"], row["predicted_pairs"])
        if any(row[k] != value for k, value in metrics.items()):
            raise RuntimeError("Paper GPU score mismatch: " + row["id"])
    different = [r for r in rows if abs(r["f1"] - r["inferred_figure_f1"]) > 1e-6]
    result = {"model": "author_public_eFold_paper_gpu_postprocessing", "n": len(rows),
              "datasets": aggregate(rows), "figure_f1_mismatches": len(different),
              "figure_mismatch_examples": [{"id": r["id"], "current_f1": r["f1"],
                                            "inferred_figure_f1": r["inferred_figure_f1"]}
                                           for r in different[:20]]}
    write_json(DEST / "summary.json", result)
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
