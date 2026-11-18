#!/usr/bin/env python3
"""Exact-sequence overlap of the prior EternaFold TrainSetA and eFold panel."""

from collections import Counter
from pathlib import Path

from paper_evaluation import OUTPUT, PROJECT, digest, read_jsonl, write_json


def main():
    train = PROJECT / "qsub_logs/eternafold/eternafold-joint-20261022/data/TrainSetA_accepted.lst"
    paths = [Path(x) for x in train.read_text().splitlines() if x.strip()]
    sequences = set()
    for path in paths:
        seq = "".join(fields[1] for line in path.read_text().splitlines()
                      if line.strip() and not line.startswith("#") for fields in [line.split()])
        sequences.add(seq.upper().replace("T", "U"))
    panel = read_jsonl(OUTPUT / "panel.jsonl")
    matches = [row["id"] for row in panel if row["sequence"] in sequences]
    counts = Counter(x.split(":", 1)[0] for x in matches)
    result = {"training_list": str(train), "training_list_sha256": digest(train),
              "training_files": len(paths), "unique_training_sequences": len(sequences),
              "test_panel_sha256": digest(OUTPUT / "panel.jsonl"),
              "matched_counts": {name: counts[name] for name in ("PDB", "archiveII", "viral_fragments", "lncRNA")},
              "matched_ids": matches,
              "scope": "Exact sequence only; homology and RNAgym training overlaps were not assessed."}
    write_json(OUTPUT / "eternafold_traina_overlap.json", result)
    print(result["matched_counts"])


if __name__ == "__main__":
    main()
