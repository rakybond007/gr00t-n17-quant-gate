"""Toolkit for the quantizability-gate experiments.

Reads results and trained artefacts that other scripts in this workspace
produce; it writes nothing back into them, so it is safe to run against a
benchmark while jobs for it are still in flight.
"""
__all__ = ["paths", "evalscan", "tradeoff", "actions", "ckpt"]
