"""Retrieve and verify the immutable inputs used by the workflows."""

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download(directory, groups):
    manifest = json.loads((ROOT / "protocol/downloads.json").read_text())
    for item in manifest:
        if item["group"] not in groups:
            continue
        target = directory / item["path"]
        if target.exists() and digest(target) == item["sha256"]:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        part = target.with_suffix(target.suffix + ".part")
        print(f"Downloading {item['path']}", flush=True)
        subprocess.run(
            ["curl", "--fail", "--silent", "--show-error", "--location", "--retry", "4", "--connect-timeout", "30",
             "--output", str(part), item["url"]],
            check=True,
        )
        if digest(part) != item["sha256"]:
            part.unlink()
            raise ValueError(f"Download SHA-256 differs from the study snapshot: {target}")
        part.replace(target)
