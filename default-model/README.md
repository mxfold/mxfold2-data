# Reproduce the MXfold2 0.3.0 default model

This recipe trains the default `nr80_imle_rank` model from random neural
weights. It runs ten epochs of structure pretraining, three epochs of
IMLE-Rank on Ribonanza, and three more on pri-miRNA and human mRNA.

## Run the training

Use Linux x86-64 with an NVIDIA GPU and a driver compatible with PyTorch CUDA
12.6. Install Git, [uv](https://docs.astral.sh/uv/), CMake 3.15 or newer, a
C++17 compiler, and GNU Make. The original run used one H100 MIG 3g.47gb.

```bash
git clone https://github.com/mxfold/mxfold2-data.git
cd mxfold2-data
./reproduce-default.sh train --work-dir runs/default-model-v030
```

The launcher installs the pinned Python 3.13.8 and PyTorch 2.7.1 CUDA 12.6
environment, builds the native folding extension, verifies the archived
inputs, and trains the three phases in order. The datasets are included in
this checkout. Select a physical GPU with `CUDA_VISIBLE_DEVICES` if needed.
To use a compatible Python environment you manage yourself, set
`MXFOLD2_PYTHON` to its Python executable. See [requirements.txt](requirements.txt)
for the other dependency versions.

The run saves model, optimizer, and random state every 400 optimizer updates
and at epoch boundaries. Use the same work directory and command to resume.
For a job with a wall time limit, pass `--hours 22` or another budget that
leaves time for setup and the last update. The trained weights and prediction
configuration go into `runs/default-model-v030/models/`, with a small
prediction smoke test and a SHA-256 record in the work directory.

To check and unpack the inputs without building MXfold2 or using a GPU:

```bash
./reproduce-default.sh prepare --work-dir runs/default-model-v030
```

## Training inputs

| Archive | Contents |
| --- | --- |
| [structure.jsonl.gz](data/structure.jsonl.gz) | 67,681 structure-labeled RNAs. Each record includes the training base pairs and any alternate conformers used in the run. |
| [ribonanza.jsonl.gz](data/ribonanza.jsonl.gz) | 45,729 Ribonanza sequences and 82,836 probing observations. |
| [mammalian.jsonl.gz](data/mammalian.jsonl.gz) | 1,096 pri-miRNAs and 1,375 human mRNAs, with one probing observation per sequence. |
| [structure-membership.tsv.gz](data/structure-membership.tsv.gz) | Source record IDs and their disposition during NR80 clustering. |

The archives preserve the exact prepared record order, base-pair labels,
and profile values. [data-complete.json](data-complete.json) records the
counts, sequence-overlap checks, and SHA-256 of the decompressed files.

The structure pool combines eFold's bpRNA-1m, RNAcentral with RNAstructure
Fold labels, PDB, and ArchiveII with the Rivas TrainSetA/B and TestSetA/B
splits, bpRNAnew, and Archive-NoFam. It includes all eligible representatives
up to 1,000 nt. CD-HIT-EST used 80% sequence identity, 80% coverage of both
sequences, and a minimum length ratio of 80%. Non-RNAcentral records took
priority when there was a match. The per-source counts and clustering flags
are in [structure-summary.json](data/structure-summary.json). The pinned
upstream URLs and SHA-256 values for the eFold files, Rivas release, bpRNAnew,
Archive-NoFam supplement, and profile collections are in
[protocol/downloads.json](../protocol/downloads.json). The datasets retain
the terms of their original distributions.

## Model and schedule

The architecture is in [baseline.conf](baseline.conf), and the epoch counts
and optimizer settings are in [plan.json](plan.json).

| Phase | Epochs | Primary sequences per epoch | Structure replay per epoch | Optimizer updates per epoch |
| --- | ---: | ---: | ---: | ---: |
| Structure pretraining | 10 | 67,681 | 0 | 67,681 |
| Ribonanza | 3 | 45,729 | 11,432 | 14,291 |
| pri-miRNA and human mRNA | 3 | 2,471 | 618 | 773 |

Pretraining uses the Fenchel–Young structure loss and AdaBelief with learning
rate `0.001`, weight decay `0.01`, and seed `1234`. Exact-sequence alternate
structures rotate by epoch. Fine-tuning restores the pretraining optimizer
moments, uses learning rate and weight decay `0.0001`, accumulates four
examples per update, and freezes the Turner parameters. The Mann–Whitney
IMLE-Rank objective uses one sample, central differences, RMS normalization,
perturbation `0.1`, and `nu=0.1`. Profile and replay losses have weights
`0.5` and `1.0`, respectively. The fine-tuning sequence order starts at seed
`0`; the replay stream, optimizer and RNG state carry over from Ribonanza
into the mammalian phase.

[reference.json](reference.json) records the original source commit,
environment, optimizer update counts, training losses, and checkpoint and
weight hashes. CUDA, compiler, and GPU differences can change floating-point
results even with the same inputs and seeds. The build uses this repository's
frozen MXfold2 runtime, and the launcher runs the phases in one process with
checkpoints instead of submitting jobs to a specific cluster.
