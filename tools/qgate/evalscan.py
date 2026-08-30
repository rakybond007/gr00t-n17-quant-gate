"""Read closed-loop evaluation results off disk.

Every benchmark writes the same per-task file, `<run>/<task>/prediction.txt`,
one line per episode:

    episode 0 is_success: [ True] action_steps: 155

The brackets and inner spacing vary, and a run that was requeued mid-flight can
leave a line with no `action_steps` field at all, so the pattern below treats
the step count as optional rather than splitting on whitespace.
"""
import re
import statistics
from pathlib import Path

from . import paths

# `action_steps` optional: a preempted-and-resumed run can truncate the line.
_EP = re.compile(
    r"episode\s+(\d+)\s+is_success:\s*\[?\s*(True|False)\s*\]?"
    r"(?:\s+action_steps:\s*(\d+))?",
    re.IGNORECASE,
)

# Config echoed into the header of prediction.txt by the eval services.
_CFG_KEYS = ("compress_k", "tau", "judge_threshold", "compensate",
             "clip_scale", "vark_bound", "gate_k3_threshold")


class Task:
    """One task within a run."""

    def __init__(self, name, episodes):
        self.name = name
        self.episodes = episodes          # list of (idx, success, steps|None)

    @property
    def n(self):
        return len(self.episodes)

    @property
    def n_success(self):
        return sum(1 for _, ok, _ in self.episodes if ok)

    @property
    def success(self):
        return self.n_success / self.n if self.n else float("nan")

    @property
    def steps_on_success(self):
        """Mean steps over successful episodes only.

        Failures run to the environment's time limit, so including them would
        measure the timeout, not the policy.
        """
        v = [s for _, ok, s in self.episodes if ok and s is not None]
        return statistics.fmean(v) if v else None


class Run:
    """One evaluation run: a directory of per-task results."""

    def __init__(self, benchmark, name, path, tasks, config):
        self.benchmark = benchmark
        self.name = name
        self.path = path
        self.tasks = tasks                # dict name -> Task
        self.config = config

    @property
    def n(self):
        return sum(t.n for t in self.tasks.values())

    @property
    def success(self):
        n = self.n
        return sum(t.n_success for t in self.tasks.values()) / n if n else float("nan")

    @property
    def steps_on_success(self):
        v = [s for t in self.tasks.values() for _, ok, s in t.episodes
             if ok and s is not None]
        return statistics.fmean(v) if v else None

    def restricted(self, names):
        """Success and steps over a subset of tasks.

        Two runs are only comparable over tasks both of them finished, and the
        aggregate over everything quietly breaks that: a task with 2 of 50
        episodes still moves the mean.
        """
        sel = [t for n, t in self.tasks.items() if n in names]
        n = sum(t.n for t in sel)
        steps = [s for t in sel for _, ok, s in t.episodes if ok and s is not None]
        return {"tasks": len(sel), "episodes": n,
                "success": sum(t.n_success for t in sel) / n if n else float("nan"),
                "steps_on_success": statistics.fmean(steps) if steps else None}

    def complete_tasks(self, expected):
        """Task names with at least `expected` episodes.

        Comparing two runs task-by-task is only fair over tasks both of them
        actually finished; partial tasks otherwise shift the mean.
        """
        return {n for n, t in self.tasks.items() if t.n >= expected}

    def as_dict(self):
        return {
            "benchmark": self.benchmark, "run": self.name, "path": str(self.path),
            "success": round(self.success, 4), "episodes": self.n,
            "tasks": len(self.tasks), "steps_on_success": self.steps_on_success,
            "config": self.config,
            "per_task": {n: {"success": round(t.success, 4), "episodes": t.n,
                             "steps_on_success": t.steps_on_success}
                         for n, t in sorted(self.tasks.items())},
        }


def _parse_task(f):
    episodes, seen = [], set()
    for line in f.read_text(errors="ignore").splitlines():
        m = _EP.match(line.strip())
        if not m:
            continue
        idx = int(m.group(1))
        if idx in seen:      # a resumed run re-prints earlier episodes
            continue
        seen.add(idx)
        episodes.append((idx, m.group(2).lower() == "true",
                         int(m.group(3)) if m.group(3) else None))
    return episodes


def _parse_config(f):
    cfg, head = {}, f.read_text(errors="ignore")[:4000]
    for k in _CFG_KEYS:
        m = re.search(rf"^{k}:\s*(\S+)", head, re.M)
        if m:
            cfg[k] = m.group(1)
    return cfg


def load_run(benchmark, name):
    d = paths.eval_dir(benchmark) / name
    if not d.is_dir():
        raise FileNotFoundError(f"no such run: {d}")
    tasks, config = {}, {}
    for f in sorted(d.glob("*/prediction.txt")):
        eps = _parse_task(f)
        if eps:
            tasks[f.parent.name] = Task(f.parent.name, eps)
        config.update(_parse_config(f))
    return Run(benchmark, name, d, tasks, config)


def list_runs(benchmark, pattern=None):
    """Every run under a benchmark that holds at least one parsed episode."""
    root = paths.eval_dir(benchmark)
    if not root.is_dir():
        return []
    out = []
    for d in sorted(root.iterdir()):
        if not d.is_dir() or (pattern and pattern not in d.name):
            continue
        try:
            r = load_run(benchmark, d.name)
        except FileNotFoundError:
            continue
        if r.tasks:
            out.append(r)
    return out
