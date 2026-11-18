#!/usr/bin/env python3
"""Check the public eFold inference weights against the published figure data."""

import json
import sys
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[2]
PAPER = PROJECT / "data/efold_hf/paper_2026"
OUT = PROJECT / "qsub_logs/efold_hf/paper_aligned_20261028"
sys.path[:0] = [str(PAPER / "author_dependencies"), str(PAPER / "author_efold_source")]

from efold.models import create_model
from efold.api.run import _predict_structure


def main():
    nb = json.loads((PAPER / "Figure5_structure_performance.ipynb").read_text())
    plots = nb["cells"][6]["outputs"][0]["data"]["application/vnd.plotly.v1+json"]["data"]
    trace = next(t for t in plots if t["name"] == "eFold")
    paper_names = {"PDB": "PDB", "archiveII": "ArchiveII", "viral_fragments": "Viral mRNA", "lncRNA": "Long ncRNA"}
    entries = []
    for source in ("PDB", "archiveII", "viral_fragments", "lncRNA"):
        raw = json.loads((PROJECT / "data/efold_hf/raw" / (source + ".json")).read_text())
        entries += [(source, reference, value) for reference, value in raw.items()]
    if len(entries) != len(trace["y"]):
        raise ValueError("Published plot and current datasets differ in size")
    if any(paper_names[source] != x for (source, _, _), x in zip(entries, trace["x"])):
        raise ValueError("Published plot and current datasets differ in grouping")

    sample = []
    for source in ("PDB", "archiveII", "viral_fragments", "lncRNA"):
        eligible = [(i, ref, value) for i, (s, ref, value) in enumerate(entries)
                    if s == source and len(value["sequence"]) <= 1000]
        sample += [(i, source, ref, value) for i, ref, value in eligible[:5]]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(model="efold", ntoken=5, d_model=64, c_z=32, d_cnn=64,
                         num_blocks=4, no_recycles=0, dropout=0, lr=3e-4, weight_decay=0, gamma=0.995)
    weights = PAPER / "author_efold_source/efold/resources/efold_weights.pt"
    status = model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=False)
    print("load keys", status, flush=True)
    model.eval().to(device)
    results = []
    for idx, source, ref, value in sample:
        seq = value["sequence"]
        predicted = {(int(i)-1, int(j)-1) for i, j in _predict_structure(model, seq, device=device)}
        truth = {tuple(sorted(map(int, p))) for p in value["structure"]}
        f1 = 2*len(predicted & truth)/(len(predicted)+len(truth)) if predicted or truth else 1.0
        fig = float(trace["y"][idx])
        row = {"id": source + ":" + ref, "length": len(seq), "public_weight_f1": f1,
               "paper_figure_f1_assuming_input_order": fig, "difference": f1-fig}
        print(json.dumps(row), flush=True)
        results.append(row)
    result = {"rows": results, "n": len(results),
              "all_agree": all(abs(x["difference"]) <= 1e-6 for x in results),
              "missing_keys": list(status.missing_keys), "unexpected_keys": list(status.unexpected_keys)}
    (OUT / "author_efold_smoke.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
