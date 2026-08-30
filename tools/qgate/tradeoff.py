"""Rank runs that trade success against episode length.

A gate makes the robot faster and slightly worse, so success rate alone
cannot rank two gates: a gate that compresses nothing scores highest and is
worthless.  What matters is whether a run beats the trade you could get for
free.  Blanket compression and no compression are two points in the
(steps, success) plane; the straight line between them is the free trade.
A run's `excess` is how far above that line it sits — a positive excess means
the gate bought success that uniform compression at the same speed could not.
"""


def line(anchor_fast, anchor_slow):
    """(steps, success) of the two anchors -> f(steps) on the line joining them."""
    (x0, y0), (x1, y1) = anchor_fast, anchor_slow
    if x1 == x0:
        raise ValueError("anchors have the same step count; the line is undefined")
    slope = (y1 - y0) / (x1 - x0)
    return (lambda x: y0 + slope * (x - x0)), slope


def score(runs, anchor_fast, anchor_slow):
    """runs: [(name, steps, success)] -> rows sorted by excess, best first.

    `steps_saved` is measured against the uncompressed anchor, so it reads as
    "how much of the full episode this run removed".
    """
    f, slope = line(anchor_fast, anchor_slow)
    x1, y1 = anchor_slow
    rows = []
    for name, steps, succ in runs:
        rows.append({
            "run": name, "steps": steps, "success": succ,
            "on_line": f(steps), "excess": succ - f(steps),
            "steps_saved": x1 - steps,
            "steps_saved_frac": (x1 - steps) / x1 if x1 else float("nan"),
        })
    rows.sort(key=lambda r: -r["excess"])
    return {"slope_success_per_step": slope, "anchor_fast": anchor_fast,
            "anchor_slow": anchor_slow, "rows": rows}


def paired_tasks(run_a, run_b, expected):
    """Per-task deltas over tasks both runs finished.

    Restricting to complete tasks is not cosmetic: a task with 5 of 50
    episodes done has a success rate with a ±0.2 quantisation step, and
    comparing it against a finished task manufactures differences.
    """
    common = run_a.complete_tasks(expected) & run_b.complete_tasks(expected)
    rows = []
    for t in sorted(common):
        ta, tb = run_a.tasks[t], run_b.tasks[t]
        rows.append({"task": t, "a": ta.success, "b": tb.success,
                     "delta": tb.success - ta.success,
                     "episodes_a": ta.n, "episodes_b": tb.n})
    rows.sort(key=lambda r: r["delta"])
    dropped = sorted((set(run_a.tasks) | set(run_b.tasks)) - common)
    mean_a = sum(r["a"] for r in rows) / len(rows) if rows else float("nan")
    mean_b = sum(r["b"] for r in rows) / len(rows) if rows else float("nan")
    return {"tasks": len(rows), "expected_episodes": expected,
            "excluded_incomplete": dropped,
            "mean_a": mean_a, "mean_b": mean_b, "mean_delta": mean_b - mean_a,
            "rows": rows}
