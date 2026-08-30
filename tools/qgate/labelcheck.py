"""Score a labelling generation against what compression actually costs.

The only external check on a label set is the closed loop we already ran
without any gate: how much success blanket K=2 compression costs each task.
A label that is any good must rank tasks the way that damage ranks them —
high confidence where compression is nearly free, low where it hurts.

This was not being done. A generation was accepted because its questions all
fired, which is a property of the questions and not of the labels. phase6 was
built that way: it revived a question that had gone dead in phase5 (mean 0.007
to 0.067, the stated goal, achieved) and its correlation with measured damage
fell from +0.420 to +0.019 — labels uncorrelated with the thing they exist to
predict. Twenty seconds of arithmetic, available the whole time, would have
caught it before 247,887 chunks were labelled and a student trained on them.
"""
import json
from pathlib import Path

from . import evalscan, paths

# Task class name is the one token in `tasks` that is neither the instruction
# sentence nor the "Valid" marker.
def task_classes(dataset_path):
    out = {}
    for line in open(Path(dataset_path).expanduser() / "meta" / "episodes.jsonl"):
        d = json.loads(line)
        c = [t for t in d.get("tasks", [])
             if isinstance(t, str) and " " not in t and t != "Valid"]
        if c:
            out[d["episode_index"]] = c[0]
    return out


def damage(benchmark, slow_run, fast_run, expected=50):
    """Per-task success cost of blanket compression: success(fast) - success(slow)."""
    slow = evalscan.load_run(benchmark, slow_run)
    fast = evalscan.load_run(benchmark, fast_run)
    common = slow.complete_tasks(expected) & fast.complete_tasks(expected)
    return {t: fast.tasks[t].success - slow.tasks[t].success for t in common}


def per_task_confidence(parquet, classes, column="p_yes"):
    import pandas as pd
    df = pd.read_parquet(parquet, columns=["episode_index", column])
    df["cls"] = df.episode_index.map(classes)
    return df.dropna(subset=["cls"]).groupby("cls")[column].mean().to_dict()


def score(parquet, benchmark, dataset_path, slow_run, fast_run,
          expected=50, column="p_yes"):
    from scipy.stats import spearmanr

    dmg = damage(benchmark, slow_run, fast_run, expected)
    conf = per_task_confidence(parquet, task_classes(dataset_path), column)
    tasks = sorted(set(dmg) & set(conf))
    if len(tasks) < 3:
        raise ValueError(f"only {len(tasks)} tasks in common between labels and evals")
    x = [dmg[t] for t in tasks]
    y = [conf[t] for t in tasks]
    rho = spearmanr(x, y).statistic
    rows = sorted(({"task": t, "delta_k2": dmg[t], "confidence": conf[t]} for t in tasks),
                  key=lambda r: r["delta_k2"])
    return {"parquet": str(parquet), "tasks": len(tasks), "spearman": float(rho),
            "rows": rows}


def compare(candidate, reference):
    """Is the candidate at least as well aligned with measured damage?"""
    d = candidate["spearman"] - reference["spearman"]
    return {"candidate": candidate["spearman"], "reference": reference["spearman"],
            "delta": d, "pass": d >= -0.02}
