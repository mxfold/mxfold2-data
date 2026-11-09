"""Run the five-epoch pretrain and same-arm 3+3 curriculum on one GPU."""

import hashlib
import json
import math
import os
import shutil

from data_sources import ROOT, digest
from panels import rows, write_rows
from scripts.efold_hf import workflow as wf

ARMS = ("mw_rank_restored", "nll_restored", "shape_fy_restored")
PRETRAIN = "pretrain"
STAGES = (PRETRAIN, "ribonanza", "mammalian")


def checkpoint(root, arm, epoch, updates):
    folder = root / "runs" / arm
    checkpoint = folder / f"epoch-{epoch:03d}.pt"
    marker = json.loads(checkpoint.with_name(f"epoch-{epoch:03d}-complete.json").read_text())
    if not marker["success"] or marker["updates"] != updates or marker["checkpoint_sha256"] != digest(checkpoint):
        raise ValueError(f"Checkpoint does not match the completed stage: {checkpoint}")
    return checkpoint


def setup(root, train, primary, pretrain_epochs, finetune_epochs, starting_checkpoint=None):
    data = root / "data"
    data.mkdir(parents=True, exist_ok=True)
    inputs = {"bprna.jsonl": train}
    if primary:
        inputs["chemical.jsonl"] = primary
    source = ROOT / "source"
    config = ROOT / "protocol/model.conf"
    plan = {
        "pretrain_epochs": pretrain_epochs, "finetune_epochs": finetune_epochs,
        "data_sha256": {name: digest(path) for name, path in inputs.items()},
        "source_sha256": {str(p.relative_to(source)): digest(p) for p in sorted(source.rglob("*"))
                          if p.is_file() and p.suffix in {".py", ".so"}},
        "config_sha256": digest(config),
        "pretrain_checkpoint_sha256": digest(starting_checkpoint) if starting_checkpoint else None,
    }
    if (root / "plan.json").exists():
        if wf.plan_for(root) != plan:
            raise ValueError(f"Inputs changed for {root}. Choose a fresh work directory.")
        wf.verify(root)
        return
    for name, path in inputs.items():
        destination = data / name
        if not destination.is_symlink():
            destination.symlink_to(path.resolve())
        elif destination.resolve() != path.resolve():
            raise ValueError(f"Preparation already points to a different input: {destination}")
    if not (root / "source").is_symlink():
        (root / "source").symlink_to(source)
    shutil.copyfile(config, root / "baseline.conf")
    if starting_checkpoint:
        target = wf.pretrain_checkpoint(root, plan)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.is_symlink():
            target.symlink_to(starting_checkpoint.resolve())
        elif target.resolve() != starting_checkpoint.resolve():
            raise ValueError(f"Preparation already points to a different checkpoint: {target}")
    wf.write_json(root / "plan.json", plan)
    wf.verify(root)


def mammalian_schedule(primary_count, replay_count, epoch, seed=0, replay=True, replay_mode="cycle"):
    import numpy as np
    from scripts.efold_hf.matched_losses import make_replay_stream

    if not replay or replay_mode != "cycle":
        raise ValueError("Mammalian adaptation requires continuous cyclic replay")
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch + 3, 9823]))
    order = [("primary", int(i)) for i in rng.permutation(primary_count)]
    count = math.floor(primary_count / 4 + .5)
    replay_start = 34296 + (epoch - 1) * count
    order.extend(("replay", int(i)) for i in make_replay_stream(replay_count, replay_start, count, seed))
    rng.shuffle(order)
    return order


def handoff(root, arm, parent):
    import torch

    target = root / "runs" / arm / "latest.pt"
    if target.exists():
        return
    payload = torch.load(parent, map_location="cpu", weights_only=False)
    if payload["optimizer_state_dict"]["param_groups"][0]["step"] != 1321708:
        raise ValueError("Expected the Ribonanza epoch-3 optimizer step")
    payload["state"] = {"epoch": 1, "cursor": 0, "updates": 0, "loss_sum": 0.}
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(".tmp")
    torch.save(payload, temp)
    temp.replace(target)


def train(prepared, runs, gpu=0):
    train_path = prepared / "train.jsonl"
    pretrain, ribonanza, mammalian = (runs / name for name in STAGES)
    setup(pretrain, train_path, None, 5, 0)
    wf.train_worker(pretrain, PRETRAIN, gpu)
    pt = checkpoint(pretrain, PRETRAIN, 5, 1278835)
    setup(ribonanza, train_path, prepared / "ribonanza.jsonl", 5, 3, pt)
    setup(mammalian, train_path, prepared / "mammalian.jsonl", 5, 3, pt)
    initial_schedule = wf.schedule
    results = {PRETRAIN: pt}
    for arm in ARMS:
        wf.train_worker(ribonanza, arm, gpu)
        parent = checkpoint(ribonanza, arm, 3, 42873)
        handoff(mammalian, arm, parent)
        wf.schedule = mammalian_schedule
        try:
            wf.train_worker(mammalian, arm, gpu)
        finally:
            wf.schedule = initial_schedule
        results[arm] = checkpoint(mammalian, arm, 3, 2319)
    return results


def predict(checkpoints, panel, runs, output, gpu=0):
    import torch

    output.mkdir(parents=True, exist_ok=True)
    for name, path in checkpoints.items():
        destination = output / f"{name}.jsonl"
        marker = destination.with_suffix(".json")
        identity = {"checkpoint": digest(path), "panel": hashlib.sha256(
            json.dumps(panel, sort_keys=True).encode()
        ).hexdigest()}
        if destination.exists() and marker.exists():
            recorded = json.loads(marker.read_text())
            if recorded != {**identity, "predictions": digest(destination)}:
                raise ValueError(f"Predictions are stale or damaged: {destination}")
            continue
        wf.seed_all(1234)
        model, _, _, _, _, _ = wf.build(runs / PRETRAIN, PRETRAIN, gpu)
        wf.restore(path, model)
        model.eval()

        def predictions():
            with torch.inference_mode():
                for rid, row in panel.items():
                    partners = model([row["sequence"]])[2][0]
                    yield [rid, [(i - 1, int(j) - 1) for i, j in enumerate(partners) if i and j > i]]

        write_rows(destination, predictions())
        wf.write_json(marker, {**identity, "predictions": digest(destination)})
