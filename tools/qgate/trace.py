"""Draw one episode's labels against time, as a file you can carry away.

Written as a self-contained HTML page with hand-drawn SVG rather than a plot
library, for two reasons: it survives `scp` as a single file with nothing to
install at the other end, and the label schemes differ enough between
benchmarks that the fields have to be discovered rather than named. RoboCasa
asks four or five questions, LIBERO renames a flag, dexjoco renames two, and
allex asks eight across two calls and carries a variable compression ratio.
"""
import glob
import html
import json
from pathlib import Path

from . import paths

# Anything that is not a per-question answer or a plottable series.
_META = {"ep", "f", "task", "ep_local", "ans", "ans1", "ans2"}


def _is_slot(k):
    return (len(k) == 1 and k in "ABCDE") or (k[:3] in ("s1_", "s2_") and len(k) == 4)


def load_episode(source, episode, task=None):
    """Rows for one episode, in frame order, from a jsonl file or shard glob."""
    src = Path(source).expanduser()
    files = sorted(glob.glob(str(src))) if any(c in str(src) for c in "*?") else [str(src)]
    if not files:
        raise FileNotFoundError(f"no label file matching {source}")
    rows = []
    for f in files:
        for line in open(f, errors="ignore"):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("ep") != episode:
                continue
            if task and r.get("task") != task:
                continue
            rows.append(r)
    rows.sort(key=lambda r: r.get("f", 0))
    return rows


def series(rows):
    """Split the numeric fields into question answers and everything else."""
    if not rows:
        return {}, {}
    keys = [k for k, v in rows[0].items()
            if k not in _META and isinstance(v, (int, float))]
    slots = {k: [float(r.get(k, 0)) for r in rows] for k in keys if _is_slot(k)}
    others = {k: [float(r.get(k, 0)) for r in rows] for k in keys if not _is_slot(k)}
    return slots, others


_PALETTE = ["#0D6B59", "#B4522A", "#3B5FA8", "#8A5AA8", "#7A6A1E",
            "#2E8C7A", "#C4703A", "#5578C4", "#A473C4"]


def _path(xs, ys, x0, y0, w, h, lo, hi):
    if not xs:
        return ""
    span = (hi - lo) or 1.0
    n = max(len(xs) - 1, 1)
    pts = [(x0 + w * i / n, y0 + h * (1 - (y - lo) / span)) for i, y in enumerate(ys)]
    return "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)


_BANDS = ["#0D6B5914", "#B4522A14", "#3B5FA814", "#8A5AA814", "#7A6A1E14"]


def _segments(rows):
    """Contiguous runs of the same task, as (start_index, end_index, name).

    An allex episode interleaves several subtasks — episode 0 changes 15 times
    — and each carries its own compression ceiling. Filtering to one task
    would splice disconnected pieces into a line that looks continuous and is
    not, so the whole episode is drawn and the runs are shaded instead.
    """
    segs, start = [], 0
    for i in range(1, len(rows) + 1):
        if i == len(rows) or rows[i].get("task") != rows[start].get("task"):
            segs.append((start, i - 1, rows[start].get("task")))
            start = i
    return [s for s in segs if s[2] is not None]


def _panel(title, data, frames, x0, y0, w, h, lo, hi, note="", segs=None, names=None):
    out = [f'<text x="{x0}" y="{y0 - 12}" class="ttl">{html.escape(title)}</text>']
    if segs:
        n = max(len(frames) - 1, 1)
        for a, b, nm in segs:
            xa = x0 + w * a / n
            xb = x0 + w * min(b + 1, len(frames) - 1) / n
            c = _BANDS[(names.index(nm) if names and nm in names else 0) % len(_BANDS)]
            out.append(f'<rect x="{xa:.1f}" y="{y0}" width="{max(xb - xa, 0.5):.1f}" '
                       f'height="{h}" fill="{c}"/>')
    if note:
        out.append(f'<text x="{x0 + w}" y="{y0 - 12}" class="note" '
                   f'text-anchor="end">{html.escape(note)}</text>')
    out.append(f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" class="plot"/>')
    for frac in (0.0, 0.5, 1.0):
        yy = y0 + h * (1 - frac)
        out.append(f'<line x1="{x0}" y1="{yy:.1f}" x2="{x0 + w}" y2="{yy:.1f}" class="grid"/>')
        out.append(f'<text x="{x0 - 8}" y="{yy + 4:.1f}" class="ax" text-anchor="end">'
                   f'{lo + (hi - lo) * frac:g}</text>')
    for i, (name, ys) in enumerate(sorted(data.items())):
        c = _PALETTE[i % len(_PALETTE)]
        out.append(f'<path d="{_path(frames, ys, x0, y0, w, h, lo, hi)}" '
                   f'stroke="{c}" class="ln"/>')
        out.append(f'<text x="{x0 + w + 10}" y="{y0 + 14 + i * 16}" class="lg" '
                   f'fill="{c}">{html.escape(name)}</text>')
    return "\n".join(out)


MAX_SERIES = 6


def render(rows, episode, source, task=None, keep=None):
    slots, others = series(rows)
    frames = [r.get("f", 0) for r in rows]
    segs = _segments(rows)
    names = sorted({s[2] for s in segs})
    if not frames:
        raise ValueError(f"no rows for episode {episode}")

    W, H, L, R = 1120, 240, 70, 130
    pw = W - L - R
    panels, y = [], 46

    if slots:
        panels.append(_panel("VLM answers — P(YES) per question", slots, frames,
                             L, y, pw, H, 0, 1,
                             f"{len(slots)} questions", segs, names))
        y += H + 70

    risk = {k: v for k, v in others.items()
            if k not in ("K", "K_max", "K_pre", "K_pre_run", "p")}
    # A series that never moves cannot explain anything that happens, and a
    # dozen overlaid lines are unreadable. Drop the flat ones, keep the most
    # variable, and say what was left out rather than quietly truncating.
    moving = {k: v for k, v in risk.items() if max(v) - min(v) > 1e-9}
    flat = sorted(set(risk) - set(moving))
    if keep:
        moving = {k: v for k, v in risk.items() if k in keep}
        flat = []
    ranked = sorted(moving, key=lambda k: -(max(moving[k]) - min(moving[k])))
    shown = ranked if keep else ranked[:MAX_SERIES]
    dropped = [k for k in ranked if k not in shown]
    if shown:
        sel = {k: moving[k] for k in shown}
        hi = max(1.0, max(max(v) for v in sel.values()))
        note = "from the actions alone"
        if flat or dropped:
            bits = []
            if flat:
                bits.append(f"{len(flat)} constant")
            if dropped:
                bits.append(f"{len(dropped)} less variable")
            note += " — hiding " + " and ".join(bits)
        panels.append(_panel("Computed risk and motion", sel, frames, L, y, pw, H, 0, hi,
                             note, segs, names))
        y += H + 70

    final = {k: v for k, v in others.items() if k in ("p", "K", "K_max")}
    if final:
        hi = max(3.0, max(max(v) for v in final.values()))
        panels.append(_panel("Confidence and chosen ratio", final, frames, L, y, pw, H, 0, hi,
                             "p is 0-1; K is the compression ratio", segs, names))
        y += H + 70

    total_h = y
    ticks = []
    n = max(len(frames) - 1, 1)
    for i in range(0, len(frames), max(len(frames) // 10, 1)):
        xx = L + pw * i / n
        ticks.append(f'<text x="{xx:.1f}" y="{total_h - 24}" class="ax" '
                     f'text-anchor="middle">{frames[i]}</text>')
    ticks.append(f'<text x="{L + pw / 2:.1f}" y="{total_h - 4}" class="ax" '
                 f'text-anchor="middle">frame index</text>')

    head = f"episode {episode}" + (f" — {task}" if task else "")
    legend = ""
    if names:
        chips = "".join(
            f'<span class="chip" style="background:{_BANDS[i % len(_BANDS)]}">'
            f'{html.escape(n)} &times;{sum(1 for s_ in segs if s_[2] == n)}</span>'
            for i, n in enumerate(names))
        legend = (f'<p class="segs">subtask runs, shaded in order of appearance: {chips}</p>')
    return f"""<title>Label trace, episode {episode}</title>
<style>
:root{{--bg:#EFF2F3;--surface:#FFF;--ink:#111819;--muted:#6B7B7F;--line:#DBE2E3}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{
  --bg:#0E1416;--surface:#161E20;--ink:#E2EAEB;--muted:#7F9296;--line:#263134}}}}
:root[data-theme="dark"]{{--bg:#0E1416;--surface:#161E20;--ink:#E2EAEB;--muted:#7F9296;--line:#263134}}
body{{background:var(--bg);color:var(--ink);margin:0;padding:28px;
 font-family:"IBM Plex Sans",system-ui,sans-serif}}
h1{{font-size:20px;font-weight:600;margin:0 0 4px}}
p.sub{{color:var(--muted);font-size:13px;margin:0 0 20px;font-family:ui-monospace,monospace}}
.wrap{{overflow-x:auto;background:var(--surface);border:1px solid var(--line);
 border-radius:10px;padding:8px}}
.plot{{fill:none;stroke:var(--line)}}
.grid{{stroke:var(--line);stroke-dasharray:2 3}}
.ln{{fill:none;stroke-width:1.6;stroke-linejoin:round}}
.ttl{{font-size:13px;font-weight:600;fill:var(--ink)}}
.note,.ax{{font-size:11px;fill:var(--muted)}}
.lg{{font-size:11px;font-family:ui-monospace,monospace}}
p.segs{{font-size:12px;margin:0 0 14px;color:var(--muted)}}
.chip{{display:inline-block;padding:2px 9px;border-radius:999px;margin-right:6px;
 border:1px solid var(--line);color:var(--ink)}}
</style>
<h1>{html.escape(head)}</h1>
{legend}
<p class="sub">{len(rows)} chunks &middot; frames {frames[0]}&ndash;{frames[-1]} &middot; {html.escape(str(source))}</p>
<div class="wrap"><svg width="{W}" height="{total_h}" viewBox="0 0 {W} {total_h}">
{chr(10).join(panels)}
{chr(10).join(ticks)}
</svg></div>
"""


def default_source(benchmark):
    d = paths.OUTPUT / "_gate_distill"
    if benchmark == "allex":
        return paths.OUTPUT / "allex_v2" / "records.jsonl"
    tags = {"robocasa": "v6b_phase6_s16_*", "libero": "libero_v1*_s*_*",
            "dexjoco": "dexjoco_v1*_s*_*"}
    return d / tags.get(benchmark, f"{benchmark}_*")
