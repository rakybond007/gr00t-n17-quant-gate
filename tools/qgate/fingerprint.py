"""Identify the dataset a set of labels was built against.

Labels are stored apart from the dataset and joined at load time on
`(episode_index, frame_index)`. That join is what lets several labelling
generations share one dataset, but it is an inner join with nothing checking
that both sides describe the same thing. Two failures follow from it, and only
one is visible: if episodes disappear the match count drops and the training
log shows it, but if the dataset is *regenerated* with the same numbering and
different content, the join still succeeds at 100% and every label now points
at the wrong frame.

A fingerprint closes the invisible case. It is taken from `meta/` only — a few
small files — so it costs nothing and never walks the dataset.
"""
import hashlib
import json
import os
from pathlib import Path

FILENAME = "provenance.json"
# Rewriting any of these means the episode numbering or the frame content
# behind it may have moved.
_HASHED = ("info.json", "episodes.jsonl", "modality.json")


def _sha(path, limit=64 * 1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
            limit -= len(chunk)
            if limit <= 0:      # a truncated hash still detects regeneration
                break
    return h.hexdigest()[:16]


def compute(dataset_path):
    ds = Path(dataset_path).expanduser()
    meta = ds / "meta"
    if not meta.is_dir():
        raise FileNotFoundError(f"{ds} has no meta/ — not a LeRobot dataset")

    fp = {"dataset": str(ds), "files": {}}
    for name in _HASHED:
        p = meta / name
        if p.exists():
            fp["files"][name] = {"sha256_16": _sha(p), "bytes": p.stat().st_size}

    eps = meta / "episodes.jsonl"
    if eps.exists():
        n, frames = 0, 0
        for line in eps.read_text(errors="ignore").splitlines():
            if not line.strip():
                continue
            n += 1
            try:
                frames += int(json.loads(line).get("length", 0))
            except Exception:
                pass
        fp["episodes"] = n
        fp["frames"] = frames
    return fp


def write(target_dir, dataset_path, extra=None):
    """Stamp a cache or label directory with the dataset it came from."""
    fp = compute(dataset_path)
    if extra:
        fp.update(extra)
    out = Path(target_dir).expanduser() / FILENAME
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(fp, indent=1, ensure_ascii=False) + "\n")
    return out, fp


def read(target_dir):
    p = Path(target_dir).expanduser() / FILENAME
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def compare(a, b):
    """Differences between two fingerprints, as human-readable lines."""
    if a is None or b is None:
        return []
    diffs = []
    for k in ("episodes", "frames"):
        if k in a and k in b and a[k] != b[k]:
            diffs.append(f"{k}: {a[k]} vs {b[k]}")
    for name in _HASHED:
        x, y = a.get("files", {}).get(name), b.get("files", {}).get(name)
        if x and y and x["sha256_16"] != y["sha256_16"]:
            diffs.append(f"meta/{name} differs ({x['sha256_16']} vs {y['sha256_16']})")
    return diffs


def verify(target_dir, dataset_path, label=""):
    """Check a stamped directory against the dataset in front of us now.

    Returns (ok, message). An unstamped directory is not an error — most were
    built before stamping existed — but it is reported, because "no evidence"
    and "verified" must not read the same.
    """
    stored = read(target_dir)
    if stored is None:
        return True, f"{label or target_dir}: not stamped, so provenance is unverified"
    try:
        now = compute(dataset_path)
    except FileNotFoundError as e:
        return False, f"{label or target_dir}: {e}"
    diffs = compare(stored, now)
    if diffs:
        return False, (f"{label or target_dir}: built against a different dataset than "
                       f"the one given — " + "; ".join(diffs))
    return True, f"{label or target_dir}: matches {Path(dataset_path).name}"
