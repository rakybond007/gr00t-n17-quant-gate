"""Check a labelling run before anything is trained on it.

A labelling job that says COMPLETED has proved nothing. Shards get preempted
and requeued, a resumed shard can re-emit rows it already wrote, and a job
whose last command is `kill` exits 0 whatever happened before it. The verdict
has to come from the rows on disk, which is what this reads.
"""
import collections
import glob
import json
import statistics as st

from . import paths


# Shard files are named <tag>_s<N>_<i>.jsonl where N is how many shards that
# run used. A pilot and a full run of the same prompt therefore differ only in
# N, so the tag must carry it — globbing across N merges two distinct runs and
# reports their overlap as duplicate rows.
def _shards(tag):
    pat = f"{tag}_*.jsonl" if "_s" in tag.rsplit("_", 1)[-1] or tag[-1].isdigit() \
        else f"{tag}_s*_*.jsonl"
    return sorted(glob.glob(str(paths.OUTPUT / "_gate_distill" / pat)))


def runs_for(prefix):
    """Distinct runs sharing a prompt tag, as <prefix>_s<N> keys."""
    import re
    out = set()
    for f in glob.glob(str(paths.OUTPUT / "_gate_distill" / f"{prefix}_s*_*.jsonl")):
        m = re.search(rf"({re.escape(prefix)}_s\d+)_\d+\.jsonl$", f)
        if m:
            out.add(m.group(1))
    return sorted(out)


def scan(tag, expected=None):
    """Row counts, duplicates, field coverage and answer spread for one run."""
    files = _shards(tag)
    if not files:
        raise FileNotFoundError(
            f"no shards matching {tag}_s*_*.jsonl under {paths.OUTPUT / '_gate_distill'}")

    per_shard, seen, fields, bad = [], collections.Counter(), collections.Counter(), 0
    answers = collections.defaultdict(list)
    for f in files:
        n = 0
        for line in open(f):
            try:
                r = json.loads(line)
            except Exception:
                bad += 1
                continue
            n += 1
            seen[(r.get("ep"), r.get("f"))] += 1
            for k, v in r.items():
                fields[k] += 1
                if k in "ABCDE" and len(k) == 1 and isinstance(v, (int, float)):
                    answers[k].append(float(v))
        per_shard.append({"shard": f.rsplit("_", 1)[-1].replace(".jsonl", ""),
                          "path": f, "rows": n})

    total = sum(s["rows"] for s in per_shard)
    dups = sum(v - 1 for v in seen.values() if v > 1)
    qs = {}
    for k in sorted(answers):
        v = sorted(answers[k])
        # A question that never varies cannot separate one chunk from another,
        # whatever its mean happens to be.
        qs[k] = {"mean": st.fmean(v), "sd": st.pstdev(v),
                 "p50": v[len(v) // 2], "p90": v[int(len(v) * .9)],
                 "over_half": sum(1 for x in v if x > 0.5) / len(v)}

    empty = [s["shard"] for s in per_shard if s["rows"] == 0]
    problems = []
    if bad:
        problems.append(f"{bad} unparseable line(s)")
    if dups:
        problems.append(f"{dups} duplicate (episode, frame) row(s) — a requeued shard "
                        "re-emitted work it had already written, "
                        "or two runs were globbed together")
    if empty:
        problems.append(f"empty shard(s): {', '.join(empty)}")
    if expected and total != expected:
        problems.append(f"{total} rows, expected {expected}")
    flat = [k for k, v in qs.items() if v["sd"] < 0.01]
    if flat:
        problems.append(f"question(s) with no spread, so they separate nothing: "
                        f"{', '.join(flat)}")

    return {"tag": tag, "shards": len(files), "rows": total,
            "unique_chunks": len(seen), "duplicates": dups, "unparseable": bad,
            "per_shard": per_shard, "fields": dict(fields), "questions": qs,
            "problems": problems, "ok": not problems}


def agreement(tag_a, tag_b):
    """How far two labelling generations disagree, chunk by chunk.

    Two prompts that rank chunks the same way cannot produce different gates,
    so this says whether a closed-loop comparison is even worth running.
    """
    def load(tag):
        out = {}
        for f in _shards(tag):
            for line in open(f):
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                out[(r.get("ep"), r.get("f"))] = r
        return out

    A, B = load(tag_a), load(tag_b)
    common = sorted(set(A) & set(B))
    if not common:
        raise ValueError(f"{tag_a} and {tag_b} share no chunks")
    qs = sorted(set("ABCDE") & set(A[common[0]]) & set(B[common[0]]))
    per_q = {}
    for q in qs:
        a = [A[k][q] for k in common]
        b = [B[k][q] for k in common]
        per_q[q] = {"mean_a": st.fmean(a), "mean_b": st.fmean(b),
                    "mean_shift": st.fmean(b) - st.fmean(a)}
    return {"a": tag_a, "b": tag_b, "common_chunks": len(common),
            "shared_questions": qs, "per_question": per_q}
