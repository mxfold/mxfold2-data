"""Rebuild packed RNAgym train records and the public two-probe test panel."""

import argparse
from collections import Counter, defaultdict
import csv
import json
import re

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from data_sources import ROOT, digest
from mxfold2.rnagym import MODIFIERS, observation_confidence
from panels import normalize, sequence_hash
from scripts.chemical_consistency.simulate_trainsetb_reactivity import generate


def parse_reactivity(value, length):
    values = np.fromstring(str(value).strip().strip("[]"), sep=",", dtype=float)
    if values.size != length:
        raise ValueError(f"Reactivity length {values.size} differs from sequence length {length}")
    return values


def official_clusters(path):
    frame = pd.read_parquet(path)
    if frame.shape[1] != 2:
        raise ValueError("Expected two columns in official cluster table")
    clusters = {}
    if frame.columns[1] == frame.columns[0] + ".1":
        clusters.update((alias, frame.columns[0]) for alias in frame.columns[0].split("+"))
    for representative, member in frame.itertuples(index=False, name=None):
        for alias in str(member).split("+"):
            if clusters.setdefault(alias, str(representative)) != representative:
                raise ValueError(f"Conflicting official clusters for {alias}")
    return clusters


def sequence_clusters(meta, clusters):
    parent = {}

    def find(item):
        root = parent.setdefault(item, item)
        while root != parent[root]:
            root = parent[root]
        while item != root:
            item, parent[item] = parent[item], root
        return root

    for _, frame in meta.groupby("sequence", sort=False):
        keys = sorted({clusters[x] for x in frame.seqID})
        for key in keys[1:]:
            left, right = find(keys[0]), find(key)
            parent[max(left, right)] = min(left, right)
    return [find(clusters[x]) for x in meta.seqID]


def condition(row):
    temperature = re.search(r"\d+\s*C", str(row["temperature"]))
    temperature = temperature.group().replace(" ", "") if temperature else "unknown"
    chemical = "+".join(sorted(re.sub(r"\s+", "", str(row["chemical"])).split("+")))
    return temperature + "|" + chemical


def pack_train(raw, target):
    manifest = target / "manifest.json"
    if manifest.exists():
        hashes = json.loads(manifest.read_text())["packed_files_sha256"]
        if all((target / name).exists() and digest(target / name) == sha for name, sha in hashes.items()):
            return
        raise ValueError(f"Packed RNAgym cache is damaged: {target}")
    target.mkdir(parents=True, exist_ok=True)
    raw_train = raw / "RMDB_s40_train.parquet"
    meta = pd.read_parquet(raw_train, columns=["seqID", "sequence", "modifier", "temperature", "chemical"])
    meta["sequence"] = meta.sequence.map(normalize)
    meta["cluster"] = sequence_clusters(meta, official_clusters(raw / "cluster_40.parquet"))
    sequences = sorted(meta.sequence.unique(), key=sequence_hash)
    meta["index"] = meta.sequence.map({s: i for i, s in enumerate(sequences)})
    conditions = [condition(row) for row in meta.to_dict("records")]
    condition_index = {c: i for i, c in enumerate(sorted(set(conditions)))}
    order = np.argsort(meta["index"].to_numpy(), kind="stable")
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(meta))
    offsets = np.concatenate(([0], np.cumsum(meta.sequence.str.len().to_numpy(np.int64)[order])))
    seq_offsets = np.concatenate(([0], np.cumsum(np.bincount(meta["index"]))))
    modifiers = meta.modifier.map({m: i for i, m in enumerate(MODIFIERS)}).to_numpy(np.int8)
    for name, value in {
        "observation_offsets": offsets, "sequence_offsets": seq_offsets, "modifiers": modifiers[order],
        "conditions": np.array([condition_index[c] for c in conditions], np.int32)[order],
    }.items():
        np.save(target / f"{name}.npy", value)
    shape = (int(offsets[-1]),)
    values = np.lib.format.open_memmap(target / "values.npy", mode="w+", dtype=np.float32, shape=shape)
    confidence = np.lib.format.open_memmap(target / "confidence.npy", mode="w+", dtype=np.float32, shape=shape)
    index = 0
    for batch in pq.ParquetFile(raw_train).iter_batches(
        batch_size=2048, columns=["sequence", "modifier", "SNR", "reactivity", "reactivity_error"]
    ):
        for row in batch.to_pylist():
            sequence = normalize(row["sequence"])
            reactivity = parse_reactivity(row["reactivity"], len(sequence)).astype(np.float32)
            error = parse_reactivity(row["reactivity_error"], len(sequence)).astype(np.float32)
            packed = inverse[index]
            index += 1
            left, right = offsets[packed:packed + 2]
            values[left:right] = reactivity
            confidence[left:right] = observation_confidence(
                reactivity, error, row["SNR"], copied_error=np.array_equal(reactivity, error, equal_nan=True)
            )
    values.flush()
    confidence.flush()
    validation = set((ROOT / "protocol/rnagym-validation-clusters.txt").read_text().splitlines())
    seq_cluster = meta.groupby("index").cluster.first()
    n_train = n_train_obs = 0
    with (target / "sequences.tsv").open("w", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(("index", "sequence", "cluster", "split"))
        for i, sequence in enumerate(sequences):
            cluster = seq_cluster[i]
            train = cluster not in validation
            n_train += train
            n_train_obs += (seq_offsets[i + 1] - seq_offsets[i]) if train else 0
            writer.writerow((i, sequence, cluster, "train" if train else "validation"))
    if (len(sequences), len(meta), n_train, n_train_obs) != (117773, 198411, 112815, 189858):
        raise ValueError("Packed RNAgym split counts differ from the study")
    hashes = {p.name: digest(p) for p in [*target.glob("*.npy"), target / "sequences.tsv"]}
    manifest.write_text(json.dumps({"packed_files_sha256": hashes}, indent=2) + "\n")


def synthetic(prepared):
    target = prepared / "synthetic"
    if (target / "simulation_report.json").exists():
        return target
    temp = prepared / "synthetic.partial"
    if temp.exists():
        import shutil

        shutil.rmtree(temp)
    generate(argparse.Namespace(
        reference=prepared / "rnagym", bpseq_list=prepared / "rivas/TrainSetB.lst",
        output=temp, seed=20261020, coverage=None, coverage_2a3=None, coverage_dms=None,
    ))
    temp.rename(target)
    for listing in target.glob("*/TrainSetB.shape.lst"):
        listing.write_text(listing.read_text().replace(str(temp), str(target)))
    return target


def test_sequences(path):
    frame = pd.read_parquet(path, columns=["sequence", "modifier", "SNR", "reactivity"])
    frame = frame[frame.modifier.isin(("2A3", "DMS"))].copy()
    frame["sequence"] = frame.sequence.map(normalize)
    frame = frame.sort_values(["modifier", "sequence", "SNR"], ascending=[True, True, False], kind="mergesort")
    frame = frame.drop_duplicates(["modifier", "sequence"], keep="first")
    if (frame.sequence.nunique(), len(frame)) != (433539, 702609):
        raise ValueError("RNAgym test panel differs from the published selection")
    return frame
