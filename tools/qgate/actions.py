"""Analyse raw action trajectories — the measurements that set gate thresholds.

Two things live here.  `layout` reports what the action vector actually
contains and how a K-step merge behaves on it; that is the measurement that
decides whether an embodiment is *summed* (end-effector deltas) or *skipped*
(absolute joint targets).  `sweep` runs a benchmark's deterministic descriptor
module over sampled chunks and reports how often each risk flag fires, which
is how the thresholds in those modules were calibrated in the first place.
"""
import importlib.util
import statistics
from pathlib import Path

import numpy as np

from . import paths

# Deterministic descriptor module per benchmark, all sharing the same interface:
# descriptors(chunk) -> dict, computed_risk(dict) -> {flag: severity}.
DESCRIPTOR_MODULES = {
    "robocasa": "robocasa_descriptors_soft.py",
    "libero": "libero_descriptors.py",
    "dexjoco": "dexjoco_descriptors.py",
}

# How compression is applied. Deltas accumulate, so merging must add them;
# absolute targets do not, so merging drops the intermediate target instead.
MERGE_OP = {"robocasa": "sum", "libero": "sum", "dexjoco": "skip", "allex": "skip"}


def load_descriptors(benchmark):
    fn = DESCRIPTOR_MODULES.get(benchmark)
    if not fn:
        raise ValueError(f"no descriptor module for {benchmark!r}; "
                         f"have {sorted(DESCRIPTOR_MODULES)}")
    # In the working tree the descriptor modules sit under vlm_gate/scripts/;
    # in the portable checkout they sit under gate/. Try both before failing.
    candidates = [paths.SCRIPTS / fn, paths.WS / "gate" / fn,
                  Path(__file__).resolve().parent.parent / "gate" / fn]
    p = next((c for c in candidates if c.exists()), None)
    if p is None:
        raise FileNotFoundError(
            f"{fn} not found; looked in " + ", ".join(str(c.parent) for c in candidates))
    spec = importlib.util.spec_from_file_location(p.stem, p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_episodes(root, limit=30, column="action"):
    """Read up to `limit` LeRobot episodes as (name, (T, D) float array)."""
    import pandas as pd

    root = Path(root).expanduser()
    files = sorted(root.glob("data/chunk-*/episode_*.parquet"))
    if not files:                      # a directory of tasks rather than one task
        files = sorted(root.glob("*/data/chunk-*/episode_*.parquet"))
    if not files:
        raise FileNotFoundError(f"no LeRobot episode parquet under {root}")
    out = []
    for f in files[:limit]:
        df = pd.read_parquet(f, columns=[column])
        out.append((f.stem, np.stack(df[column].to_numpy())))
    return out


def _pct(v, q):
    return float(np.percentile(v, q)) if len(v) else float("nan")


def layout(episodes, n=16, k=2, clip=1.0):
    """What the action vector holds, and what a K-step merge does to it.

    `merge_exceeds` is the fraction of merged commands that would leave the
    controller's range if the merge were a sum.  For a delta embodiment that
    excess is displacement the robot simply never travels; for an absolute
    embodiment the number is meaningless, which is itself the diagnostic.
    """
    arrs = [a for _, a in episodes]
    D = arrs[0].shape[1]
    allsteps = np.concatenate(arrs, axis=0)
    per_dim = [{"dim": i,
                "min": float(allsteps[:, i].min()), "max": float(allsteps[:, i].max()),
                "mean_abs": float(np.abs(allsteps[:, i]).mean()),
                "binary": bool(np.isin(np.unique(np.round(allsteps[:, i], 3)),
                                       [-1.0, 0.0, 1.0]).all())}
               for i in range(D)]

    single, merged = [], []
    for a in arrs:
        t = (a.shape[0] // k) * k
        if t < k:
            continue
        pairs = a[:t].reshape(-1, k, D).sum(axis=1)
        single.append(np.abs(a).max(axis=1))
        merged.append(np.abs(pairs).max(axis=1))
    single = np.concatenate(single) if single else np.zeros(0)
    merged = np.concatenate(merged) if merged else np.zeros(0)

    return {
        "episodes": len(arrs), "steps": int(allsteps.shape[0]), "action_dim": D,
        "chunk": n, "k": k,
        "per_dim": per_dim,
        "step_abs_p50": _pct(single, 50), "step_abs_p99": _pct(single, 99),
        "single_exceeds": float((single > clip).mean()) if len(single) else float("nan"),
        "merge_exceeds": float((merged > clip).mean()) if len(merged) else float("nan"),
    }


def sweep(benchmark, episodes, n=16, k=2, stride=None):
    """Fire rate of each deterministic risk flag over sampled chunks.

    A flag that never fires carries no information and should be replaced,
    not merely re-thresholded; a flag that fires almost always is equally
    useless.  Both show up here.
    """
    mod = load_descriptors(benchmark)
    stride = stride or n
    rows = []
    for _, a in episodes:
        for f in range(0, max(a.shape[0] - n, 0) + 1, stride):
            rows.append(mod.computed_risk(mod.descriptors(a, f, n)))
    if not rows:
        return {"chunks": 0, "flags": {}}
    flags = sorted(rows[0])
    return {
        "chunks": len(rows), "benchmark": benchmark, "merge_op": MERGE_OP.get(benchmark),
        "flags": {
            fl: {"fire_rate": float(np.mean([r[fl] > 0 for r in rows])),
                 "mean": float(np.mean([r[fl] for r in rows])),
                 "p90": _pct(np.array([r[fl] for r in rows]), 90)}
            for fl in flags
        },
    }
