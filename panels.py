"""Build the study panels from the pinned public downloads."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import tarfile
import zipfile

from data_sources import ROOT

EFOLD_COUNTS = {"PDB": 355, "archiveII": 3370, "viral_fragments": 40, "lncRNA": 30}
CHEMICAL_COUNTS = {"ribo500-blast": 45729, "pri_miRNA": 1096, "human_mRNA": 1375}


def normalize(sequence):
    return sequence.upper().replace("T", "U")


def sequence_hash(sequence):
    return hashlib.sha256(sequence.encode()).hexdigest()


def rows(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def write_rows(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    with temp.open("w") as stream:
        for record in records:
            stream.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
    temp.replace(path)


def reference(identifier, source, sequence, pairs):
    return {
        "id": identifier, "source": source, "sequence": sequence,
        "pairs": pairs, "sequence_sha256": sequence_hash(sequence),
    }


def efold_test(raw):
    from scripts.efold_hf.workflow import clean_observations, structure_pairs

    for source, count in EFOLD_COUNTS.items():
        items = json.loads((raw / f"{source}.json").read_text())
        if len(items) != count:
            raise ValueError(f"Expected {count} sequences for {source}")
        for name, item in items.items():
            sequence = normalize(item["sequence"])
            pairs = structure_pairs(item["structure"], len(sequence))
            row = reference(f"{source}:{name}", source, sequence, pairs)
            row["observations"] = clean_observations(item) if source == "viral_fragments" else []
            yield row


def efold_train(raw, out):
    from scripts.efold_hf.workflow import clean_observations, project_pairs, structure_pairs

    final = json.loads((raw / "efold_train.json").read_text())
    membership = {normalize(row["sequence"]) for row in final.values()}
    structural, chemical = [], []
    for source in ("bpRNA-1m", *CHEMICAL_COUNTS):
        items = json.loads((raw / f"{source}.json").read_text())
        seen = set()
        for key, item in items.items():
            sequence = normalize(item["sequence"])
            if not 1 <= len(sequence) <= 1000 or set(sequence) - set("ACGU"):
                continue
            if sequence not in membership or sequence in seen:
                continue
            seen.add(sequence)
            pairs = structure_pairs(item["structure"], len(sequence))
            row = {
                "id": f"{source}:{key}", "source": source, "sequence": sequence,
                "pairs": pairs, "train_pairs": project_pairs(sequence, pairs),
            }
            if source == "bpRNA-1m":
                structural.append(row)
            else:
                row["observations"] = clean_observations(item)
                if row["observations"]:
                    chemical.append(row)
    chemical_sequences = {r["sequence"] for r in chemical}
    structural = [r for r in structural if r["sequence"] not in chemical_sequences]
    if len(structural) != 45507 or Counter(r["source"] for r in chemical) != CHEMICAL_COUNTS:
        raise ValueError("eFold training counts changed")

    central = json.loads((raw / "rnacentral_synthetic.json").read_text())
    added = []
    for key, item in central.items():
        sequence = normalize(item["sequence"])
        retained = final.get(key)
        if not retained or normalize(retained["sequence"]) != sequence or len(sequence) > 1000:
            continue
        pairs = structure_pairs(retained["structure"], len(sequence))
        added.append({
            "id": f"rnacentral_synthetic:{key}", "source": "rnacentral_synthetic",
            "sequence": sequence, "pairs": pairs, "train_pairs": project_pairs(sequence, pairs),
        })
    if len(central) != 226729 or len(added) != 211191:
        raise ValueError("RNAcentral source or retained count changed")
    by_sequence = {}
    for cohort in (structural, added):
        for row in sorted(cohort, key=lambda r: r["id"]):
            by_sequence.setdefault(row["sequence"], row)
    if len(by_sequence) != 255767:
        raise ValueError("The deduplicated replay pool changed")
    write_rows(out / "train.jsonl", by_sequence.values())
    write_rows(out / "ribonanza.jsonl", (r for r in chemical if r["source"] == "ribo500-blast"))
    write_rows(out / "mammalian.jsonl", (r for r in chemical if r["source"] != "ribo500-blast"))


def bpseq(content):
    entries = [line.split() for line in content.splitlines() if line and line[0].isdigit()]
    sequence = normalize("".join(entry[1] for entry in entries))
    pairs = [(i - 1, j - 1) for i, j in ((int(r[0]), int(r[2])) for r in entries) if j > i]
    return sequence, pairs


def rivas(archive, out):
    for name, count in {"TrainSetA": 3166, "TrainSetB": 1094, "TestSetB": 430}.items():
        with tarfile.open(archive) as bundle:
            listing = bundle.extractfile(f"data/{name}.lst").read().decode().splitlines()
            if len(listing) != count:
                raise ValueError(f"Unexpected size of {name}")
            paths = []
            for item in listing:
                path = out / name / Path(item).name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(bundle.extractfile(item).read())
                paths.append(str(path.resolve()))
            (out / f"{name}.lst").write_text("\n".join(paths) + "\n")


def external(raw):
    with tarfile.open(raw / "bpRNAnew.tar.gz") as bundle:
        listing = bundle.extractfile("data/bpRNAnew.nr500.canonicals.lst").read().decode()
        paths = listing.splitlines()
        if len(paths) != 5401:
            raise ValueError("Expected 5,401 bpRNAnew sequences")
        for path in paths:
            sequence, pairs = bpseq(bundle.extractfile(path).read().decode())
            yield reference(Path(path).stem, "bpRNAnew", sequence, pairs)
    with zipfile.ZipFile(raw / "nofam.zip") as bundle:
        paths = sorted(p for p in bundle.namelist()
                       if p.startswith("supplemental-file-2/ground-truth/") and p.endswith(".ct"))
        groups = json.loads((ROOT / "protocol/nofam_cohorts.json").read_text())["group_map"]
        if len(paths) != 89 or {Path(p).stem for p in paths} != set(groups):
            raise ValueError("Archive-NoFam IDs differ")
        for path in paths:
            lines = bundle.read(path).decode().splitlines()
            length = int(lines[0].split()[0])
            data = [line.split() for line in lines[1:1 + length]]
            sequence = normalize("".join(row[1] for row in data))
            pairs = [(int(row[0]) - 1, int(row[4]) - 1) for row in data if int(row[4]) > int(row[0])]
            yield reference(Path(path).stem, "Archive-NoFam", sequence, pairs)
