"""Recreate the MXfold2 TestSetB grid and the RNAgym adaptation endpoints."""

import argparse
from collections import defaultdict
from itertools import product
import json
import math
from pathlib import Path
import random
from statistics import fmean

import numpy as np
import torch

from data_sources import ROOT, digest
from mxfold2.dataset import BPseqDataset
from mxfold2.predict import Predict
from mxfold2.rnagym import MODIFIERS, PackedRNAGym
from mxfold2.train import Train
from panels import bpseq
from rnagym_data import pack_train, parse_reactivity, synthetic, test_sequences
from score import pair_counts, pairs_metric, table
from scripts.chemical_consistency.direct_wu import Wu
from scripts.chemical_consistency.map_mw_metrics import from_pairs, score_profile
from scripts.chemical_consistency.rnagym_losses import (
    FullRNAGymRankLoss, FullRNAGymNLLLoss, FullRNAgymShapeFYLoss, ReplayFYLoss,
    epoch_slots, restore_baseline_optimizer,
)
from scripts.efold_hf import workflow as wf

METHODS = ("imle-rank", "imle-nll", "shape-fy")
CONTROLS = ("rnafold", "mxfold2")
PROBES = ("2A3", "DMS")


class NativeLSTM(torch.nn.LSTM):
    def forward(self, *args, **kwargs):
        with torch.backends.cudnn.flags(enabled=False):
            return super().forward(*args, **kwargs)


def build_model(state, gpu, joint=False):
    wf.seed_all(0)
    parser = argparse.ArgumentParser(fromfile_prefix_chars="@")
    Predict.add_args(parser.add_subparsers())
    config = ROOT / "protocol" / ("joint-model.conf" if joint else "model.conf")
    args = parser.parse_args(["predict", f"@{config}", "unused.fasta"])
    model, _ = Predict().build_model(args)
    for module in model.modules():
        if isinstance(module, torch.nn.LSTM):
            module.__class__ = NativeLSTM
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or any(".shape_head_" not in key for key in missing):
        raise ValueError(f"Checkpoint does not fit the model: {missing}, {unexpected}")
    model.turner.requires_grad_(False)
    return model.to(f"cuda:{gpu}" if gpu >= 0 else "cpu")


class Targets:
    def __init__(self, block, key, dataset):
        path = ROOT / "protocol/teachers" / block / f"{key}.npz"
        with np.load(path, allow_pickle=False) as payload:
            self.offsets = payload["offsets"]
            self.targets = payload["targets"]
            if payload["sequences"].tolist() != [row["sequence"] for row in dataset.sequences]:
                raise ValueError(f"Frozen teacher sequence order differs: {key}")

    def __getitem__(self, index):
        left, right = self.offsets[index:index + 2]
        return torch.as_tensor(self.targets[left:right].copy(), dtype=torch.long)


def fit(run, train_a, packed, block, method, probe, replay_percent, gpu):
    joint = block == "joint"
    run.mkdir(parents=True, exist_ok=True)
    if joint:
        baseline = ROOT / "protocol/checkpoints/joint-seed0.pt"
        source = baseline
    else:
        source = ROOT / "protocol/checkpoints/traina-epoch100.pth"
        baseline = run / "initial_baseline.pth"
        if not baseline.exists():
            baseline.symlink_to(source)
    starting = torch.load(source, map_location="cpu", weights_only=False)
    state_dict = starting.get("ema_state_dict", starting["model_state_dict"])
    model = build_model(state_dict, gpu, joint)
    device = next(model.parameters()).device
    optimizer = Train().build_optimizer("AdaBelief", model, .0001, .0001)
    if not joint:
        restore_baseline_optimizer(run, model, optimizer)
    structural = ReplayFYLoss(model)
    shapes = None
    labels = None
    if method == "imle-rank":
        auxiliary = FullRNAGymRankLoss(model)
    elif method == "imle-nll":
        modules = torch.nn.ModuleDict({p: Wu() for p in MODIFIERS}).to(device)
        shapes = list(modules.values())
        optimizer.add_param_group({"params": modules.parameters(), "lr": .0001, "weight_decay": .0001})
        auxiliary = FullRNAGymNLLLoss(model, modules)
    elif method == "shape-fy":
        auxiliary = FullRNAgymShapeFYLoss(model)
    elif method != "traina-only":
        key = "truth" if method == "truth" else f"{method}-{probe}"
        labels = Targets(block, key, packed)
        auxiliary = ReplayFYLoss(model)
        auxiliary.perturb = .1
    else:
        auxiliary = None

    indices = packed.indices("train") if packed else []
    replay = math.ceil(len(indices) * replay_percent / (100 - replay_percent))
    epochs = 3 if block == "rnagym" else 100
    state = {"epoch": 1, "cursor": 0, "updates": 0}
    latest = run / "latest.pt"
    if latest.exists():
        state = wf.restore(latest, model, optimizer, shapes)
    else:
        wf.save_checkpoint(latest, model, optimizer, shapes, state)
    model.train()
    if auxiliary is not None:
        auxiliary.train()
    while state["epoch"] <= epochs:
        epoch = state["epoch"]
        if joint:
            slots = [("replay", i) for i in range(len(train_a))]
            slots += [("chemical", i) for i in indices]
            random.Random(epoch * 1009).shuffle(slots)
        else:
            slots = epoch_slots(indices, len(train_a), epoch, replay, 0)
        steps = math.ceil(len(slots) / 4)
        while state["cursor"] < len(slots):
            batch = slots[state["cursor"]:state["cursor"] + 4]
            optimizer.zero_grad(set_to_none=True)
            for kind, index in batch:
                if kind == "replay":
                    name, sequence, values = train_a[index]
                    loss = structural([sequence], [values["target"].to(device)], fname=[name]).sum()
                    weight = steps / len(train_a) if joint else 1 / len(batch)
                else:
                    item, observations = packed[index]
                    if labels is not None:
                        loss = auxiliary([item["sequence"]], [labels[index].to(device)]).sum()
                    else:
                        if joint:
                            for obs in observations:
                                obs["mask"] &= obs["values"] > 1e-5
                        loss = auxiliary(item["sequence"], observations)
                    weight = .5 * steps / len(indices) if joint else .5 / len(batch)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite loss in {run.name}, epoch {epoch}, cursor {state['cursor']}")
                (weight * loss).backward()
            wf.optimizer_step(optimizer)
            state["cursor"] += len(batch)
            state["updates"] += 1
            if state["updates"] % 250 == 0:
                wf.save_checkpoint(latest, model, optimizer, shapes, state)
        state.update(epoch=epoch + 1, cursor=0)
        wf.save_checkpoint(latest, model, optimizer, shapes, state)
        print(f"{block}/{run.name} epoch {epoch}/{epochs}, {state['updates']} updates", flush=True)
    return model.eval()


@torch.inference_mode()
def score_testsetb(model, listing):
    result = []
    for path in map(Path, listing.read_text().splitlines()):
        sequence, expected = bpseq(path.read_text())
        partners = model([sequence])[2][0]
        pairs = [(i - 1, int(j) - 1) for i, j in enumerate(partners) if i and j > i]
        result.append(pairs_metric(*pair_counts(expected, pairs), empty=0))
    if len(result) != 430:
        raise ValueError("Expected 430 TestSetB RNAs")
    return {"PPV": fmean(r["precision"] for r in result),
            "SEN": fmean(r["recall"] for r in result), "F1": fmean(r["f1"] for r in result)}


@torch.inference_mode()
def score_rnagym(models, frame, extra_predictions=None):
    # 2A3 and DMS share a sequence-only prediction. A nucleotide group enters
    # the comparison only if both metrics are defined for all four models.
    accumulators = defaultdict(lambda: [0, 0, 0., 0.])
    for sequence, rows in frame.groupby("sequence", sort=False):
        states = ({name: from_pairs(sequence, predictions[sequence])
                   for name, predictions in extra_predictions.items()} if extra_predictions else {})
        for name, model in models.items():
            partners = model([sequence])[2][0]
            pairs = [(i - 1, int(j) - 1) for i, j in enumerate(partners) if i and j > i]
            states[name] = from_pairs(sequence, pairs)
        for row in rows.itertuples(index=False):
            profile = {"sequence": sequence, "modifier": row.modifier,
                       "values": parse_reactivity(row.reactivity, len(sequence))}
            measures = {name: score_profile(profile, unpaired) for name, unpaired in states.items()}
            valid = [i for i in range(len(next(iter(measures.values()))))
                     if all(scores[i]["auroc"] is not None and scores[i]["spearman"] is not None
                            for scores in measures.values())]
            if valid:
                for name, scores in measures.items():
                    counts = accumulators[name, row.modifier]
                    counts[0] += 1
                    counts[1] += len(valid)
                    counts[2] += fmean(scores[i]["auroc"] for i in valid)
                    counts[3] += fmean(scores[i]["spearman"] for i in valid)
    return [
        dict(model=name, probe=probe, rnas=n, profiles=n, strata=strata,
             auroc=auc / n, spearman=rho / n)
        for (name, probe), (n, strata, auc, rho) in accumulators.items()
    ]


def run(prepared, raw, work, output, modes, gpu=0, eternafold=None):
    bound = json.loads((ROOT / "protocol/historical-inputs.json").read_text())
    for name, sha in bound.items():
        if digest(ROOT / name) != sha:
            raise ValueError(f"Historical input changed: {name}")
    pack_train(raw, prepared / "rnagym")
    simulated = synthetic(prepared) if eternafold or modes & {"synthetic", "joint"} else None
    listing = prepared / "rivas/TrainSetA.lst"
    train_a = BPseqDataset(str(listing))
    joint_list = prepared / "rivas/JointTrainSetA.lst"
    joint_list.write_text("".join(
        str(listing.parent / "TrainSetA" / name) + "\n"
        for name in (ROOT / "protocol/joint-traina.txt").read_text().splitlines()
    ))
    joint_a = BPseqDataset(str(joint_list))
    if (len(train_a), len(joint_a)) != (3166, 2713):
        raise ValueError("TrainSetA protocol lists differ")
    start = torch.load(ROOT / "protocol/checkpoints/traina-epoch100.pth", map_location="cpu", weights_only=False)
    models = {"baseline": build_model(start.get("ema_state_dict", start["model_state_dict"]), gpu).eval()}
    for block in sorted(modes):
        scores = [dict(block=block, model="baseline", probe="", replay=0, n=430,
                       **score_testsetb(models["baseline"], prepared / "rivas/TestSetB.lst"))] if block != "joint" else []
        if block == "rnagym":
            cases = [(method, "", 20) for method in METHODS]
        else:
            replay = [0, 5, 10, 20, 50] if block == "synthetic" else [0]
            cases = list(product(METHODS + CONTROLS, PROBES, replay))
            cases += [("truth", "2A3", p) for p in replay]
            if block == "joint":
                cases.append(("traina-only", "", 0))
        for method, probe, fraction in cases:
            tag = f"{method}-{probe or 'all'}-replay{fraction:02d}"
            data = (PackedRNAGym(prepared / "rnagym") if block == "rnagym"
                    else PackedRNAGym(simulated / probe) if probe else None)
            trained = fit(work / block / tag, joint_a if block == "joint" else train_a, data,
                          block, method, probe, fraction, gpu)
            scores.append(dict(block=block, model=tag, probe=probe, replay=fraction, n=430,
                               **score_testsetb(trained, prepared / "rivas/TestSetB.lst")))
            if block == "rnagym":
                models[method] = trained
            else:
                del trained
            table(output / f"mx-{block}-TestSetB.tsv", scores)
        if block == "synthetic":
            from paper import plot_replay

            plot_replay(output, scores)
    if "rnagym" in modes or eternafold:
        frame = test_sequences(raw / "RMDB_s40_test.parquet")
        if "rnagym" in modes:
            table(output / "rnagym-four-model-chemistry.tsv", score_rnagym(models, frame))
        if eternafold:
            from train_eternafold import run as train_eternafold, prediction_pairs
            native_work = work / "eternafold"
            binary, params = train_eternafold(prepared, native_work, output, eternafold)
            sequences = sorted(frame.sequence.unique())
            entries = [(f"R{i:06d}", seq) for i, seq in enumerate(sequences)]
            predictions = dict(zip(sequences, prediction_pairs(
                binary, params, eternafold, entries, native_work / "RNAgym-predictions"
            ), strict=True))
            scores = score_rnagym({"baseline": models["baseline"]}, frame, {"EternaFold": predictions})
            table(output / "rnagym-eternafold-matched-chemistry.tsv", scores)
