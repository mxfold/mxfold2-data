"""Score sequence-only folds with the study's panel-specific conventions."""

from collections import defaultdict
import csv
import gzip
import json
from statistics import fmean

from data_sources import ROOT


def table(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    with temp.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, list(records[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(records)
    temp.replace(path)


def pairs_metric(tp, fp, fn, empty):
    return {
        "precision": tp / (tp + fp) if tp + fp else (empty if fn else 1),
        "recall": tp / (tp + fn) if tp + fn else (empty if fp else 1),
        "f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1,
    }


def pair_counts(reference, predicted):
    ref, pred = set(map(tuple, reference)), set(map(tuple, predicted))
    return len(ref & pred), len(pred - ref), len(ref - pred)


def mean_metrics(scored, field):
    values = [r[field] for r in scored if r[field] is not None]
    return fmean(values) if values else None


def structural(panel, models):
    cohorts = json.loads((ROOT / "protocol/nofam_cohorts.json").read_text())
    group_map = cohorts["group_map"]
    populations = defaultdict(set)
    for rid, row in panel.items():
        source = row["source"]
        if source == "Archive-NoFam":
            for name in ("full89", "paper_bprna_filtered"):
                if group_map[rid] in cohorts["cohorts"][name]["groups"]:
                    populations[source, name].add(rid)
        elif source == "bpRNAnew":
            populations[source, "all"].add(rid)
            if set(row["sequence"]) <= set("ACGU"):
                populations[source, "ACGU"].add(rid)
        else:
            populations[source, "all"].add(rid)

    for model, predictions in models.items():
        if set(predictions) != set(panel):
            raise ValueError(f"Missing or unexpected predictions for {model}")
        for (source, population), selected in populations.items():
            groups = defaultdict(list)
            scores = []
            for rid in sorted(selected):
                counts = pair_counts(panel[rid]["pairs"], predictions[rid])
                metrics = pairs_metric(*counts, empty=None)
                scores.append(metrics)
                if source == "Archive-NoFam":
                    groups[group_map[rid]].append({k: v if v is not None else 0
                                                  for k, v in metrics.items()})
            units = ([{k: fmean(r[k] for r in group) for k in scores[0]}
                      for group in groups.values()] if groups else scores)
            yield {
                "dataset": source, "population": population, "model": model,
                "n": len(scores), "groups": len(groups),
                "n_precision": sum(r["precision"] is not None for r in scores),
                "n_recall": sum(r["recall"] is not None for r in scores),
                **{k: mean_metrics(units, k) for k in scores[0]},
                "rna_f1": mean_metrics(scores, "f1"),
            }


def viral(panel, models):
    from scripts.chemical_consistency.map_mw_metrics import from_pairs, score_profile

    measurements = {}
    for rid, row in panel.items():
        for i, observation in enumerate(row.get("observations", [])):
            modifier = observation["probe"].upper()
            measurements[rid, i] = {
                "sequence": row["sequence"], "modifier": modifier, "values": observation["values"]
            }
    scores = {}
    for model, predictions in models.items():
        valid = {}
        for (rid, index), profile in measurements.items():
            states = from_pairs(panel[rid]["sequence"], predictions[rid])
            for stratum in score_profile(profile, states):
                if stratum["auroc"] is not None and stratum["spearman"] is not None:
                    valid[rid, index, stratum["stratum"]] = (stratum["auroc"], stratum["spearman"])
        scores[model] = valid
    common = sorted(set.intersection(*(set(values) for values in scores.values())))
    for model, values in scores.items():
        for probe in ("ALL", "DMS", "SHAPE"):
            profiles = defaultdict(list)
            strata = 0
            for rid, index, group in common:
                if probe == "ALL" or measurements[rid, index]["modifier"] == probe:
                    profiles[rid, index].append(values[rid, index, group])
                    strata += 1
            rnas = defaultdict(list)
            for (rid, _), groups in profiles.items():
                rnas[rid].append(tuple(map(fmean, zip(*groups))))
            units = [tuple(map(fmean, zip(*profiles))) for profiles in rnas.values()]
            yield {
                "model": model, "probe": probe, "rnas": len(units), "profiles": len(profiles),
                "strata": strata, "auroc": fmean(r[0] for r in units) if units else None,
                "spearman": fmean(r[1] for r in units) if units else None,
            }


def read_predictions(path):
    opener = gzip.open if path.suffix == ".gz" else open
    predictions = {}
    with opener(path, "rt") as stream:
        for line in stream:
            rid, pairs = json.loads(line)
            if rid in predictions:
                raise ValueError(f"Duplicate prediction for {rid}: {path}")
            predictions[rid] = pairs
    return predictions
