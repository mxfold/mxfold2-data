#!/usr/bin/env python3
"""Audit actual test coverage in the previous EternaFold/MXfold2 comparison."""

import csv
import hashlib
import json
from pathlib import Path

from paper_evaluation import OUTPUT, PROJECT, digest, write_json


def main():
    ef = PROJECT / "qsub_logs/eternafold/eternafold-joint-20261022"
    mx = PROJECT / "qsub_logs/rnagym/mxfold2-trainsetab-joint-20261022"
    entries = [r for r in json.loads((ef / "data/evaluation_entries.json").read_text()) if r["kind"] == "structure"]
    paths = (mx / "TestSetB.lst").read_text().splitlines()
    if len(entries) != 430 or [r["source"] for r in entries] != paths:
        raise RuntimeError("TestSetB references differ")
    if any(digest(Path(r["source"])) != r["source_sha256"] for r in entries):
        raise RuntimeError("TestSetB reference file changed")
    empty = 0
    for row in entries:
        data = [line.split() for line in Path(row["source"]).read_text().splitlines()
                if line.strip() and not line.startswith("#")]
        empty += not any(int(value[2]) > 0 for value in data)
    ef_panel = ef / "data/test_panel.parquet"
    mx_panel = mx / "data/2A3/test_panel.parquet"
    if digest(ef_panel) != digest(mx_panel):
        raise RuntimeError("RNAgym panel differs")
    coverages = {}
    reference = None
    for path in sorted((ef / "evaluation").glob("*/RNAgym.csv")):
        with path.open() as f:
            rows = list(csv.DictReader(f))
        keys = {(hashlib.sha256(row["sequence"].encode()).hexdigest(), row["cluster"], row["modifier"]) for row in rows}
        if len(keys) != len(rows):
            raise RuntimeError("Duplicate EternaFold chemical result")
        reference = keys if reference is None else reference
        coverages["EternaFold:" + path.parent.name] = {"n": len(keys), "same_as_reference": keys == reference}
    for path in sorted((mx / "conditions").glob("*/runs/evaluation/epoch-100/RNAgym_official_test_panel.csv")):
        with path.open() as f:
            rows = list(csv.DictReader(f))
        keys = {(row["sequence_hash"], row["cluster"], row["modifier"]) for row in rows}
        if len(keys) != len(rows):
            raise RuntimeError("Duplicate MXfold2 chemical result")
        coverages["MXfold2:" + path.parents[3].name] = {"n": len(keys), "same_as_reference": keys == reference}
    if len(coverages) != 16 or any(not r["same_as_reference"] or r["n"] != 1656 for r in coverages.values()):
        raise RuntimeError("RNAgym coverage differs between conditions")
    result = {"testsetb_n": len(entries), "testsetb_same_paths_in_same_order": True,
              "testsetb_reference_hashes_match": True, "testsetb_empty_reference": empty,
              "rnagym_panel_sha256": digest(ef_panel), "rnagym_sequence_probe_cluster_coverage": coverages,
              "inference_difference": "EternaFold MEA gamma=6; MXfold2 native decoder",
              "scope": "Previous experiments use TestSetB and RNAgym, not eFold paper test datasets."}
    write_json(OUTPUT / "previous_eternafold_comparison_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
