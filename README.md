# MXfold2 chemical-probing study reproducibility

Code, numerical source data, and run recipes for *Learning RNA secondary structure from chemical probing with MXfold2* by Rintaro Yamauchi, Sho Tsukiyama, and Kengo Sato. The recipes cover the 2026-11-08 round-2 manuscript and supplement and the MXfold2 joint training rerun completed on 2026-11-12.

## Run on Linux with an NVIDIA GPU

On Linux x86-64, install Git, curl, [uv](https://docs.astral.sh/uv/), CMake, GNU Make, GCC/G++ with C++17 support, and Perl. Use an NVIDIA driver compatible with PyTorch CUDA 12.6. Open MPI and an MPI C++ compiler are needed when choosing more than one EternaFold rank. A GPU with ample memory and at least 32 GB of host RAM are useful for the longest RNA and full RNAgym evaluations.

```bash
git clone https://github.com/mxfold/mxfold2-data.git
cd mxfold2-data
./reproduce.sh train --work-dir runs/round2
```

The command installs the pinned Python 3.13 environment, builds the MXfold2 extension, downloads and checks the public datasets, and runs the study suites sequentially. It requires sustained compute time. The primary MXfold2 pretraining alone uses 1,278,835 optimizer updates. The synthetic series includes 55 runs of 100 adaptation epochs, and EternaFold training runs on the CPU. On a multi-core workstation pass `--eternafold-ranks N` with the number of MPI ranks available to you. Jobs save their model, optimizer, and RNG state to the work directory. Run the same command with the same work directory to resume.

Run one suite at a time with `--suite primary`, `--suite synthetic`, `--suite joint`, `--suite rnagym`, or `--suite eternafold`. The EternaFold suite also works on a CPU-only machine. `./reproduce.sh prepare --suite primary` prepares the eFold and external panels and reconstructs the three primary training files. Set `CUDA_VISIBLE_DEVICES` to select the GPU. If you manage the environment yourself, point `MXFOLD2_PYTHON` at its Python executable.

The MXfold2 joint suite uses all 3,166 original TrainSetA BPSEQ records from the Rivas release. Each of the 11 joint conditions adds 1,094 TrainSetB examples per epoch and takes 1,065 optimizer updates. The TrainSetA-only control takes 792 updates per epoch. All 12 conditions run for 100 epochs from the same seed-0 neural initialization. The BPSEQ annotations enter the structural loss as constraints without preprocessing. In the archived Turner folding audit, the native decoder retained 166,178 of their 167,320 pairs. Joint checkpoints are stored under `checkpoints/joint-raw3166/` in the work directory.

## Recompute the paper results on a CPU

```bash
./reproduce.sh verify --work-dir runs/round2
```

This mode downloads the reference panels and rescores or aggregates the archived evidence. It checks the primary structural F1 values and 40-RNA viral chemistry results against the round-2 experiment and generates the numerical TSVs and Figures 2–4 under `runs/round2/archived-results/`. The plots reproduce the coordinates and values with a standalone Matplotlib layout. Figure 1 is a manuscript schematic.

| Round-2 result | Training suite or evidence used |
| --- | --- |
| Main Table 1 and Figure 2, Supplementary Tables 4–6 | `synthetic`, `joint`, and `eternafold`. The verification path averages the 430 saved TestSetB scores for each condition. The MXfold2 joint rows in Main Table 1 and Supplementary Table 6 use the 3,166-record rerun. The 2,713-record protocol in the November 8 draft is available at commit `3be8044`. |
| Main Table 2 and Supplementary Table 7 | `rnagym` and `eternafold`. The full RNAgym benchmark retains the highest-SNR profile per sequence and probe. The two comparisons apply separate common-observation masks. The verification path includes the two archived RNAgym aggregates. |
| Main Table 3 and Figure 3, Supplementary Table 8 | `primary`. The verification path rescores the four MXfold2 models' saved base pairs. Released predictors use saved predictions or per-RNA base-pair counts. |
| Figure 4 and Supplementary Tables 9–11 | `primary`. Archive-NoFam averages RNAs within each group and then groups equally. bpRNAnew averages RNAs and retains all 5,401 test records. |
| Supplementary Tables 12–13 | `verify` recomputes counts and summaries from saved per-query edit-distance and MMseqs2 results. |
| Supplementary Tables 1–3 | `protocol/optimization.tsv`, the MXfold2 release's BPSEQ lists, and the saved or regenerated synthetic calibration report. |

Fresh fits are written to `retrained-results/`. GPU, CUDA, library, and MPI differences can change individual predictions. Compare numerical values and populations against `archived-results/`. The original TrainSetA epoch-100 checkpoint, including AdaBelief moments, anchors the synthetic and RNAgym adaptation runs. Fixed RNAfold/MXfold2 teacher labels and the joint experiment's random initialization are preserved as small protocol artifacts. Released predictor comparisons use their archived outputs, and similarity tables use the archived search hits.

## Inputs and provenance

- Rivas TrainSetA, TrainSetB, and TestSetB are retrieved from [the MXfold2 v0.1.0 release](https://github.com/mxfold/mxfold2/releases/tag/v0.1.0). bpRNAnew is retrieved from [v0.1.1](https://github.com/mxfold/mxfold2/releases/tag/v0.1.1).
- The RNAgym train/test parquet files and sequence clusters come from [the official Hugging Face snapshot](https://huggingface.co/datasets/Marks-lab/RNAgym/tree/bc6ccf4c5bf445ca25991f2a74448ae7e153b9f6/secondary_structure_prediction). `protocol/rnagym-validation-clusters.txt` records the development holdout. The independent EternaFold RNAgym comparison uses the full official training split.
- The eFold datasets come from [Rouskin Lab's public collection](https://huggingface.co/rouskinlab/datasets). `protocol/downloads.json` pins each source and SHA-256. `protocol/training-inputs.json` binds the reconstructed 255,767-record pretraining/replay input and the two chemical training inputs to the original runs.
- Archive-NoFam references come from [the benchmark's archived preprint supplement](https://pmc.ncbi.nlm.nih.gov/articles/PMC12637522/). The 79-RNA primary and 89-RNA full populations use the same fixed group mapping.
- MXfold2 model, loss, and native folding code come from the study's frozen experiment source. The CPU EternaFold recipe fetches [the pinned upstream commit](https://github.com/WaymentSteeleLab/EternaFold/tree/87b9aac55cee14fd562049d08f7b92d3131f10ce), applies the study's packed-input and optimizer adapters, and builds the resulting executable.
- `reference/` contains study predictions, TestSetB scores, similarity records, and compact RNAgym aggregates. Each result file names its population and model. `protocol/historical-inputs.json` records checksums of the starting checkpoints and fixed teachers. Source datasets retain the terms of their upstream distributions.

The primary adaptation restores the PT5 optimizer at step 1,278,835 and carries each arm's state from Ribonanza into the mammalian stage, including its likelihood parameters and RNG state. Structural replay is 20% of sample slots. Every test prediction uses sequence alone. Precision, sensitivity, and F1 use exact base-pair matches, with the panel-specific zero-denominator rules implemented in `score.py`. Viral and RNAgym chemistry compare unpaired states with raw measurements, with DMS stratified by A and C.
