"""Locate the checkpoints an experiment actually ran against."""
from pathlib import Path

from . import paths

# Base VLA policies, per benchmark. These are the controls every gate result
# is measured against, so they are listed explicitly rather than globbed.
POLICIES = {
    "robocasa": "robocasa/groot/groot_n1_5_bs64_baseline/checkpoint-60000",
    "libero": "libero/groot/groot_n1_5_bs64_baseline/checkpoint-60000",
    "dexjoco": "dexjoco/groot/groot_n1_5_bs64_single_arm_multitask_baseline/checkpoint-60000",
}


def _size(p):
    """Bytes under a directory, one level of files deep (checkpoints are flat)."""
    if not p.exists():
        return 0
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def policies():
    rows = []
    for bench, rel in POLICIES.items():
        p = paths.CKPT_ROOT / rel
        rows.append({"benchmark": bench, "kind": "policy", "name": Path(rel).parent.name,
                     "path": str(p), "exists": p.exists(),
                     "bytes": _size(p) if p.exists() else 0})
    return rows


def students():
    """Trained gate modules. `gate_module_best.pt` is the one to serve."""
    rows = []
    if not paths.MODULES.is_dir():
        return rows
    for d in sorted(paths.MODULES.iterdir()):
        if not d.is_dir() or "smoke" in d.name:
            continue
        best = d / "gate_module_best.pt"
        final = d / "gate_module.pt"
        pick = best if best.exists() else final
        if not pick.exists():
            continue
        rows.append({"benchmark": d.name.split("_")[0], "kind": "student",
                     "name": d.name, "path": str(pick), "exists": True,
                     "bytes": pick.stat().st_size,
                     "has_best": best.exists()})
    return rows


def inventory():
    return policies() + students()
