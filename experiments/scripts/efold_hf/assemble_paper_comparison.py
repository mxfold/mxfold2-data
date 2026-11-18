#!/usr/bin/env python3
"""Combine the independently checked eFold and MXfold2 evaluation results."""

import csv
import io
import json

from paper_evaluation import OUTPUT, write_json


def main():
    results = json.loads((OUTPUT / "results.json").read_text())
    gpu_summary = OUTPUT / "author_efold_paper_gpu/summary.json"
    efold = json.loads((gpu_summary if gpu_summary.exists() else OUTPUT / "author_efold_direct/summary.json").read_text())
    models = results["models"]
    if any(not models[arm]["complete"] for arm in ("pretrain", "mw_rank_restored", "nll_restored", "shape_fy_restored")):
        raise RuntimeError("MXfold2 results are incomplete")
    panel = results["panel_counts"]
    all_models = {"author_eFold": efold["datasets"]}
    for name, value in models.items():
        all_models[name] = value.get("datasets", value)
    if any(data[src]["n"] != count for data in all_models.values() for src, count in panel.items()):
        raise RuntimeError("Model evaluation panels have different counts")
    combined = {"panel_counts": panel, "models": all_models, "efold_model": efold["model"],
                "efold_figure_f1_mismatches": efold["figure_f1_mismatches"],
                "efold_figure_mismatch_examples": efold["figure_mismatch_examples"]}
    write_json(OUTPUT / "final_comparison.json", combined)
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, delimiter="\t")
    writer.writerow(["dataset", "model", "n", "precision", "n_precision", "recall", "n_recall", "f1", "n_f1"])
    for src in ("PDB", "archiveII", "viral_fragments", "lncRNA"):
        for name, data in all_models.items():
            row = data[src]
            writer.writerow([src, name, *(row[k] for k in ("n", "precision", "n_precision", "recall", "n_recall", "f1", "n_f1"))])
    (OUTPUT / "final_comparison.tsv").write_text(buffer.getvalue())
    label = "著者eFold（論文手順）" if gpu_summary.exists() else "著者eFold（公開API）"
    headers = [("author_eFold", label), ("paper_EternaFold", "著者EternaFold"),
               ("pretrain", "事前学習後"), ("mw_rank_restored", "MW Rank"),
               ("nll_restored", "IMLE-NLL"), ("shape_fy_restored", "SHAPE-fy")]
    names = {"PDB": "PDB", "archiveII": "ArchiveII", "viral_fragments": "Viral mRNA", "lncRNA": "Long ncRNA"}
    method = "eFoldの後処理は、論文用test_stepと同様にGPU上で実行した。" if gpu_summary.exists() else "eFoldの後処理は公開APIを使ってCPU上で実行した。論文用test_stepのGPU後処理による評価は別途実行中。"
    text = ["# 1000 nt以下の同一パネルでのF1", "",
            "著者の公開eFold・EternaFoldと今回のMXfold2について、各RNAの完全一致F1の平均を示す。" + method, "",
            "| データ | 件数 | " + " | ".join(label for _, label in headers) + " |",
            "|---|---:" + "|---:" * len(headers) + "|"]
    for source, label in names.items():
        vals = [f"{all_models[model][source]['f1']:.4f}" for model, _ in headers]
        text.append("| " + label + " | " + str(panel[source]) + " | " + " | ".join(vals) + " |")
    text += ["", f"著者の図から順序を復元した配列別F1と、公開eFoldモデルの直接評価でF1が異なった件数: {efold['figure_f1_mismatches']}。",
             "", "precision・recallと集計対象件数は `final_comparison.tsv` を参照。lncRNAは3件のみ。", ""]
    (OUTPUT / "final_comparison.md").write_text("\n".join(text))
    print("complete", sum(panel.values()), "figure mismatches", efold["figure_f1_mismatches"])


if __name__ == "__main__":
    main()
