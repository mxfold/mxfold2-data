#!/usr/bin/env python3
"""Check paper test_step's GPU postprocessing against mismatching figure scores."""

import json
import sys
from pathlib import Path

import torch

from paper_evaluation import OUTPUT, PROJECT, pair_metrics, read_jsonl, write_json

PAPER = PROJECT / "data/efold_hf/paper_2026"
SOURCE = PAPER / "author_efold_source"
sys.path[:0] = [str(PAPER / "author_dependencies"), str(SOURCE)]
from efold.core import batch
from efold.core.embeddings import sequence_to_int
from efold.models import create_model


def main():
    rows = []
    for i in range(4):
        rows.extend(read_jsonl(OUTPUT / f"author_efold_direct/shard-{i}.jsonl"))
    mismatch = [r for r in rows if abs(r["f1"] - r["inferred_figure_f1"]) > 1e-6]
    chosen = [r for r in mismatch if r["source"] == "PDB"][:2] + [r for r in mismatch if r["source"] == "archiveII"][:10]
    assert len(chosen) == 12
    refs = {r["id"]: r for r in read_jsonl(OUTPUT / "panel.jsonl")}
    model = create_model(model="efold", ntoken=5, d_model=64, c_z=32, d_cnn=64,
                         num_blocks=4, no_recycles=0, dropout=0, lr=3e-4, weight_decay=0, gamma=0.995)
    model.load_state_dict(torch.load(SOURCE / "efold/resources/efold_weights.pt", map_location="cpu", weights_only=True), strict=True)
    model.eval().to("cuda")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    results = []
    for old in chosen:
        item = refs[old["id"]]
        seq = item["sequence"]
        tensor = sequence_to_int(seq).unsqueeze(0)
        data = batch.Batch(sequence=tensor, reference=[item["id"]], length=[len(seq)], L=len(seq),
                           use_error=False, batch_size=1, data_types=["sequence"], dt_count={"sequence": 1}).to("cuda")
        with torch.inference_mode():
            raw = model(data)
            structure = model.postprocesser.run(raw["structure"], data.get("sequence"))[0]
            pairs = torch.nonzero(torch.triu(structure, diagonal=1), as_tuple=False).cpu().tolist()
        f1 = pair_metrics(item["pairs"], pairs)["f1"]
        row = {"id": item["id"], "cpu_postprocess_f1": old["f1"], "gpu_postprocess_f1": f1,
               "figure_f1_inferred": old["inferred_figure_f1"]}
        results.append(row)
        print(json.dumps(row), flush=True)
    write_json(OUTPUT / "author_efold_gpu_postprocess_smoke.json", {
        "rows": results,
        "n": len(results),
        "cpu_matches": sum(abs(r["cpu_postprocess_f1"] - r["figure_f1_inferred"]) <= 1e-6 for r in results),
        "gpu_matches": sum(abs(r["gpu_postprocess_f1"] - r["figure_f1_inferred"]) <= 1e-6 for r in results),
        "note": "Figure ID mapping is inferred from author notebook input order."})


if __name__ == "__main__":
    main()
