"""Every absolute path the toolkit needs, resolved in one place.

The workspace root is found from this file's own location, so a clone that
lands somewhere else still works with no edits.  Each root can be overridden
by an environment variable, which is what makes the commands runnable from a
one-shot ssh invocation with no shell setup.
"""
import os
from pathlib import Path


def _env(name, default):
    v = os.environ.get(name)
    return Path(v).expanduser().resolve() if v else Path(default)


def _find_ws():
    """Walk up from this file for the directory that holds `vlm_gate/`.

    Searching for the marker rather than counting `..` levels means the same
    package works whether it sits at `<ws>/vlm_gate/qgate/` in the working
    tree or at `<repo>/tools/qgate/` in a checkout on another machine.  When
    no marker is found — a checkout with no results beside it — QGATE_WS is
    the answer, and every command that needs a real path will say so.
    """
    here = Path(__file__).resolve()
    for cand in here.parents:
        if (cand / "vlm_gate").is_dir():
            return cand
    return here.parents[2] if len(here.parents) > 2 else here.parent


WS = _env("QGATE_WS", _find_ws())
VG = _env("QGATE_VLM_GATE", WS / "vlm_gate")

OUTPUT = _env("QGATE_OUTPUT", VG / "output")          # closed-loop eval results
ANALYSIS = _env("QGATE_ANALYSIS", VG / "analysis")    # labels, evolver state
SCRIPTS = VG / "scripts"
RUN_SCRIPTS = VG / "run_scripts"
DOCS = _env("QGATE_DOCS", WS / "docs")

ASSETS = _env("QGATE_ASSETS", WS / "assets")
MODULES = ASSETS / "modules_A"                        # trained gate students
DATASETS = ASSETS / "datasets"

# GR00T policy checkpoints live outside this workspace.
CKPT_ROOT = _env("QGATE_CKPT_ROOT", Path.home() / "multigpu_workspace/Isaac-GR00T/ckpt")

BENCHMARKS = ("robocasa", "libero", "dexjoco")


def eval_dir(benchmark):
    return OUTPUT / benchmark


def describe():
    """Rows of (name, path, exists) — what `qgate paths` prints."""
    items = [
        ("workspace", WS), ("vlm_gate", VG), ("eval output", OUTPUT),
        ("analysis", ANALYSIS), ("scripts", SCRIPTS), ("run_scripts", RUN_SCRIPTS),
        ("docs", DOCS), ("assets", ASSETS), ("gate modules", MODULES),
        ("datasets", DATASETS), ("policy checkpoints", CKPT_ROOT),
    ]
    return [(n, str(p), p.exists()) for n, p in items]
