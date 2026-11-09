"""Regenerate numerical source tables and figures for the revised manuscript."""

from collections import defaultdict
import csv
import json
from statistics import fmean, median
import shutil

from data_sources import ROOT
from score import mean_metrics, pairs_metric, read_predictions, structural, table, viral

REF = ROOT / "reference"
PRIMARY = ("pretrain", "mw_rank_restored", "nll_restored", "shape_fy_restored")
LABELS = dict(zip(PRIMARY, ("Pretrained baseline", "IMLE-Rank", "IMLE-NLL", "SHAPE-FY")))
LABELS.update(author_eFold="eFold", paper_EternaFold="EternaFold", paper_MXFold2="MXfold2 (released)",
              MXfold2_public="MXfold2 (released)", EternaFold_public="EternaFold (released)",
              RNAstructure_Fold="RNAstructure", UFold_public="UFold", SPOT_RNA_public="SPOT-RNA")
PUBLIC = ("author_eFold", "MXfold2_public", "EternaFold_public", "RNAstructure_Fold", "UFold_public", "SPOT_RNA_public")


def records(path):
    with path.open() as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def comparator_counts(path, panel):
    scores = defaultdict(dict)
    for row in records(path):
        rid = row["id"]
        if rid not in panel or rid in scores[row["model"]]:
            raise ValueError(f"Invalid comparator ID {rid}")
        counts = tuple(int(row[k]) for k in ("tp", "fp", "fn"))
        if counts[0] + counts[2] != len(panel[rid]["pairs"]):
            raise ValueError(f"Reference differs for {rid}")
        scores[row["model"]][rid] = pairs_metric(*counts, empty=None)
    return scores


def comparator_summary(source, population, model, scores, group_map=None):
    units = list(scores.values())
    if group_map is not None:
        groups = defaultdict(list)
        for rid, score in scores.items():
            groups[group_map[rid]].append(score)
        units = [{k: fmean(s[k] or 0 for s in group) for k in group[0]} for group in groups.values()]
    return {
        "dataset": source, "population": population, "model": model, "n": len(scores),
        "groups": len(units) if group_map is not None else 0,
        "n_precision": len(scores) if group_map is not None else sum(r["precision"] is not None for r in scores.values()),
        "n_recall": len(scores) if group_map is not None else sum(r["recall"] is not None for r in scores.values()),
        **{k: mean_metrics(units, k) for k in ("precision", "recall", "f1")},
        "rna_f1": mean_metrics(list(scores.values()), "f1"),
    }


def comparators(panel):
    e = {rid: row for rid, row in panel.items() if row["source"] not in ("bpRNAnew", "Archive-NoFam")}
    preds = read_predictions(REF / "predictions/efold-released.jsonl.gz")
    yield from structural(e, {"author_eFold": preds})
    for model, scores in comparator_counts(REF / "efold-distributed.tsv", e).items():
        if set(scores) != set(e):
            raise ValueError(f"Incomplete comparator: {model}")
        for source in ("PDB", "archiveII", "viral_fragments", "lncRNA"):
            yield comparator_summary(source, "all", model,
                                     {rid: v for rid, v in scores.items() if e[rid]["source"] == source})
    b = {rid: row for rid, row in panel.items() if row["source"] == "bpRNAnew"}
    for model in PUBLIC:
        preds = read_predictions(REF / f"predictions/bprnanew-{model}.jsonl.gz")
        yield from (r for r in structural(b, {model: preds}) if r["population"] == "all")
    nf = {rid: row for rid, row in panel.items() if row["source"] == "Archive-NoFam"}
    cohorts = json.loads((ROOT / "protocol/nofam_cohorts.json").read_text())
    groups = cohorts["group_map"]
    chosen = set(cohorts["cohorts"]["paper_bprna_filtered"]["groups"])
    for model, scores in comparator_counts(REF / "nofam-released.tsv", nf).items():
        scores = {rid: row for rid, row in scores.items() if groups[rid] in chosen}
        if len(scores) != 79:
            raise ValueError(f"Archive-NoFam comparator lacks full primary panel: {model}")
        yield comparator_summary("Archive-NoFam", "paper_bprna_filtered", model, scores, groups)


def testsetb():
    groups = defaultdict(list)
    for row in records(REF / "testsetb-per-rna.tsv"):
        groups[row["block"], row["model"], row["probe"], row["replay"]].append(row)
    for (block, model, probe, replay), group in groups.items():
        if len(group) != 430 or len({r["id"] for r in group}) != 430:
            raise ValueError(f"TestSetB is incomplete: {model}")
        yield dict(block=block, model=model, probe=probe, replay=replay, n=430,
                   **{k: fmean(float(r[k]) for r in group) for k in ("PPV", "SEN", "F1")})


def similarity():
    globs = defaultdict(list)
    for r in records(REF / "similarity-global.tsv"):
        globs[r["cohort"], r["dataset"]].append(float(r["similarity"]))
    local = defaultdict(list)
    for r in records(REF / "similarity-local.tsv"):
        local[r["cohort"], r["dataset"]].append(r)
    for key, vals in globs.items():
        cohort, dataset = key
        n = 355 if dataset == "PDB" else 3359
        hits = local[key]
        if len(vals) != n or len(hits) != n:
            raise ValueError(f"Incomplete similarity cohort {key}")
        yield {"cohort": cohort, "dataset": dataset, "n": n,
               "mean": fmean(vals), "median": median(vals),
               **{f"global_ge_{t}": sum(v >= t / 100 for v in vals) for t in (80, 90, 95, 100)},
               "query_cov80_hits": sum(bool(r["identity"]) for r in hits),
               "query_cov80_ge_90": sum(float(r["identity"] or 0) >= 90 for r in hits),
               **{f"both_cov80_ge_{t}": sum(float(r["both_identity"] or 0) >= t for r in hits)
                  for t in (80, 90, 95)},
               "contained": sum(r["contained"] == "True" for r in hits)}


def regenerate(panel, predictions, out, archived):
    metrics = list(structural(panel, predictions))
    chemistry = list(viral(panel, predictions))
    published = list(comparators(panel))
    structural_rows = metrics + published
    external = {"Archive-NoFam", "bpRNAnew"}
    table(out / "main-table-3_viral-chemistry.tsv", chemistry)
    table(out / "supp-table-8_efold.tsv", [r for r in structural_rows if r["dataset"] not in external])
    table(out / "supp-table-9_external.tsv", [
        r for r in structural_rows if (r["dataset"], r["population"]) in
        {("Archive-NoFam", "paper_bprna_filtered"), ("bpRNAnew", "all")}
    ])
    table(out / "supp-table-10_nofam-populations.tsv",
          [r for r in metrics if r["dataset"] == "Archive-NoFam"])
    table(out / "supp-table-11_bprnanew-populations.tsv",
          [r for r in metrics if r["dataset"] == "bpRNAnew"])

    if archived:
        verify_primary(metrics, chemistry)
        test = list(testsetb())
        table(out / "supp-table-5_synthetic-grid.tsv", [r for r in test if r["block"] == "synthetic"])
        table(out / "supp-table-6_joint-grid.tsv", [r for r in test if r["block"] == "joint"])
        table(out / "main-table-1_testsetb-source.tsv", [
            r for r in test if (r["block"] == "synthetic" and r["replay"] == "20" and r["probe"] in ("2A3", "none"))
            or (r["block"] == "joint" and (r["model"] == "traina-only-joint" or r["model"] in ("imle-rank-2A3-joint", "imle-nll-2A3-joint", "shape-fy-2A3-joint")))
            or r["model"] in ("baseline", "pretrain", "joint_synthetic_2A3")
        ])
        table(out / "main-table-2_testsetb.tsv", [r for r in test if r["block"] == "rnagym" or r["model"] == "joint_rnagym_all"])
        table(out / "main-table-2_supp-table-7_rnagym-chemistry.tsv", records(REF / "rnagym-matched.tsv"))
        table(out / "supp-tables-12-13_similarity.tsv", list(similarity()))
        shutil.copyfile(REF / "simulation-counts.json", out / "supp-table-3_simulation-counts.json")
        plot_replay(out, test)
    plot_structures(out, structural_rows, 3, ["PDB", "archiveII", "viral_fragments", "lncRNA"])
    plot_structures(out, structural_rows, 4, ["Archive-NoFam", "bpRNAnew"])


def verify_primary(structures, chemistry):
    expected = {
        "PDB": (0.888961, 0.870791, 0.864837, 0.892977),
        "archiveII": (0.614761, 0.626114, 0.621270, 0.618514),
        "viral_fragments": (0.739236, 0.778022, 0.800083, 0.744664),
        "lncRNA": (0.444005, 0.467145, 0.443593, 0.448048),
        "Archive-NoFam": (0.752424, 0.759087, 0.765014, 0.750511),
        "bpRNAnew": (0.638185, 0.646957, 0.639305, 0.639626),
    }
    for source, values in expected.items():
        population = "paper_bprna_filtered" if source == "Archive-NoFam" else "all"
        scores = {r["model"]: r["f1"] for r in structures if r["dataset"] == source and r["population"] == population}
        if any(abs(scores[model] - ref) > 0.000001 for model, ref in zip(PRIMARY, values)):
            raise ValueError(f"Archived F1 differs from the reference for {source}")
    expected_auroc = (0.809926, 0.835545, 0.845109, 0.813749)
    for model, expected in zip(PRIMARY, expected_auroc):
        result = next(r for r in chemistry if r["model"] == model and r["probe"] == "ALL")
        if (result["rnas"], result["strata"]) != (40, 65) or abs(result["auroc"] - expected) > 0.000001:
            raise ValueError(f"Viral chemistry differs for {model}")


def plot_structures(out, records, figure, panels):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.3), squeeze=False)
    symbols = ["o", "^", "s", "D", "P", "X", "v", "*", "h", "p"]
    for ax, source in zip(axes[0], panels):
        population = "paper_bprna_filtered" if source == "Archive-NoFam" else "all"
        selected = [r for r in records if r["dataset"] == source and r["population"] == population]
        for i, row in enumerate(selected):
            label = LABELS[row["model"]]
            ax.scatter(row["recall"], row["precision"], marker=symbols[i % len(symbols)],
                       label=f"{label} ({row['f1']:.3g})")
        ax.set(title=source, xlabel="Sensitivity", ylabel="Precision")
        ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    fig.savefig(out / f"figure-{figure}.pdf")
    plt.close(fig)


def plot_replay(out, test):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    baseline = next(r["F1"] for r in test if r["model"] == "baseline")
    for ax, probe in zip(axes, ("2A3", "DMS")):
        selected = defaultdict(list)
        for row in test:
            if row["block"] == "synthetic" and (row["probe"] == probe or row["model"].startswith("truth-")):
                method = row["model"].split("-replay")[0].replace(f"-{probe}", "")
                selected[method].append((int(row["replay"]), row["F1"]))
        for method, points in selected.items():
            ax.plot(*zip(*sorted(points)), marker="o", label=method)
        ax.axhline(baseline, linestyle="--", color="black", label="Pretrained baseline")
        ax.set(title=probe, xlabel="Structural replay (%)", ylabel="TestSetB mean F1", xticks=[0, 5, 10, 20, 50])
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "figure-2.pdf")
    plt.close(fig)
