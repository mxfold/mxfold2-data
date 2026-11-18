#!/usr/bin/env python3
"""Reconstruct eFold figure F1 values by the notebook's input ordering.

Plotly traces save F1 and dataset, but not reference IDs.  The mapping below is
therefore inferred from the notebook's left-merge order and the current data;
it is not a substitute for the missing V2 per-sequence prediction files.
"""

import json
import statistics
from collections import Counter
from pathlib import Path

from paper_evaluation import PROJECT, RAW, OUTPUT, digest, write_json


def main():
    source = PROJECT / "data/efold_hf/paper_2026/Figure5_structure_performance.ipynb"
    nb = json.loads(source.read_text())
    trace_data = nb["cells"][6]["outputs"][0]["data"]["application/vnd.plotly.v1+json"]["data"]
    names = {"PDB": "PDB", "archiveII": "ArchiveII", "viral_fragments": "Viral mRNA", "lncRNA": "Long ncRNA"}
    entries = []
    for source_name in ("PDB", "archiveII", "viral_fragments", "lncRNA"):
        obj = json.loads((RAW / (source_name + ".json")).read_text())
        entries.extend((source_name, key, value) for key, value in obj.items())
    assert Counter(x[0] for x in entries) == {"PDB": 355, "archiveII": 3370, "viral_fragments": 40, "lncRNA": 30}
    assert [x[1] for x in entries[:5]] == ["2N7X-2D", "8S95-2D", "2CD1-2D", "1WTT-2D", "2NCI-2D"]
    summary, records = {}, []
    for model in ("eFold (pre-trained)", "eFold"):
        trace = next(x for x in trace_data if x["name"] == model)
        if len(trace["y"]) != len(entries) or len(trace["x"]) != len(entries):
            raise ValueError("Figure trace and raw data lengths differ")
        if any(names[src] != x for (src, _, _), x in zip(entries, trace["x"])):
            raise ValueError("Figure trace dataset order differs")
        if model == "eFold (pre-trained)":
            # Published Table 3 notebook displays these ID/F1 pairs explicitly.
            anchors = [0.933333, 0.703704, 1.0, 1.0, 0.888889]
            assert all(abs(a-b) < 1e-6 for a, b in zip(anchors, trace["y"][:5]))
        all_scores, filtered = {}, {}
        for (src, ref, data), score in zip(entries, trace["y"]):
            all_scores.setdefault(src, []).append(float(score))
            if len(data["sequence"]) <= 1000:
                filtered.setdefault(src, []).append(float(score))
                records.append({"model": model, "id_inferred": src + ":" + ref,
                                "length": len(data["sequence"]), "f1": float(score)})
        summary[model] = {src: {"n_all": len(all_scores[src]), "f1_all": statistics.fmean(all_scores[src]),
                                "n_le1000": len(filtered[src]), "f1_le1000": statistics.fmean(filtered[src])}
                          for src in names}
    expected = {"PDB": .894044, "archiveII": .599568, "viral_fragments": .725702, "lncRNA": .442343}
    for src, value in expected.items():
        assert abs(summary["eFold"][src]["f1_all"] - value) < 1e-6
    out = OUTPUT / "author_efold_figure_inferred.jsonl"
    with out.open("w") as handle:
        for row in records:
            handle.write(json.dumps(row) + "\n")
    write_json(OUTPUT / "author_efold_figure_inferred.json", {
        "warning": "The published plot stores no reference IDs. Reference-to-score mapping is inferred from author notebook merge order; the missing V2 prediction file prevents full ID-level verification.",
        "source": str(source), "source_sha256": digest(source), "inferred_rows_sha256": digest(out),
        "counts": len(records), "models": summary})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
