# Archived numerical evidence

These files contain the source data needed by the CPU `verify` command. Prediction files store [RNA ID, zero-based base pairs] per line. The primary model identifiers are `pretrain` (PT5), `mw_rank_restored` (IMLE-Rank), `nll_restored` (IMLE-NLL), and `shape_fy_restored` (SHAPE-FY). Adaptation records use the fixed FT6 checkpoint. Raw reference annotations and probing profiles are fetched from the upstream datasets.

| Files | Evidence |
| --- | --- |
| `predictions/` | Base pairs for the four primary models and released eFold/bpRNAnew comparators |
| `efold-distributed.tsv`, `nofam-released.tsv` | Per-RNA TP/FP/FN for the other released comparators |
| `testsetb-per-rna.tsv` | Final per-RNA PPV, sensitivity and F1 for synthetic, joint, RNAgym, and EternaFold comparisons. The joint rows are from epoch 100 of the MXfold2 rerun on all 3,166 original TrainSetA records, completed on 2026-11-12. |
| `rnagym-matched.tsv` | RNA-weighted means and valid RNA/profile/stratum counts for the two distinct common-observation comparisons |
| `similarity-global.tsv`, `similarity-local.tsv` | Per-query nearest global similarity, best local identity, and containment for the three target collections |
| `simulation-counts.json` | Calibration and coverage records for the generated TrainSetB profiles |
