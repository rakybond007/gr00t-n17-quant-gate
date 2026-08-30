"""`qgate` — one entry point for reading results out of this workspace.

Every command works from any working directory and prints a table by default
or JSON with --json, so it is usable both from an interactive shell and from
a one-shot `ssh host qgate ...` invocation whose output something else parses.
Nothing here writes to the experiment directories.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from . import actions, ckpt, evalscan, labelcheck, labels, paths, trace, tradeoff


def _human(nbytes):
    for unit in ("B", "K", "M", "G", "T"):
        if nbytes < 1024 or unit == "T":
            return f"{nbytes:.0f}{unit}" if unit == "B" else f"{nbytes:.1f}{unit}"
        nbytes /= 1024


def _emit(args, payload, table):
    if args.json:
        json.dump(payload, sys.stdout, indent=1, default=str)
        sys.stdout.write("\n")
    else:
        table()


def cmd_paths(args):
    rows = paths.describe()
    _emit(args, [{"name": n, "path": p, "exists": e} for n, p, e in rows],
          lambda: [print(f"{'ok ' if e else 'MISSING'} {n:20s} {p}") for n, p, e in rows])


def cmd_jobs(args):
    fmt = "%.10i %.12P %.44j %.2t %.11M %R"
    out = subprocess.run(["squeue", "-u", args.user, "-o", fmt],
                         capture_output=True, text=True)
    if out.returncode != 0:
        sys.exit(out.stderr.strip() or "squeue failed")
    print(out.stdout.rstrip() or "no jobs queued or running")


def cmd_results(args):
    runs = evalscan.list_runs(args.benchmark, args.filter)
    if not runs:
        sys.exit(f"no parsed runs under {paths.eval_dir(args.benchmark)}"
                 + (f" matching {args.filter!r}" if args.filter else ""))
    runs.sort(key=lambda r: -r.success)
    payload = [{"run": r.name, "success": round(r.success, 4), "episodes": r.n,
                "tasks": len(r.tasks),
                "steps_on_success": r.steps_on_success, "config": r.config}
               for r in runs]

    def table():
        print(f"{'run':44s} {'success':>8s} {'steps':>8s} {'eps':>6s} {'tasks':>6s}")
        for r in runs:
            st = f"{r.steps_on_success:.1f}" if r.steps_on_success else "-"
            print(f"{r.name:44.44s} {r.success:8.4f} {st:>8s} {r.n:6d} {len(r.tasks):6d}")
        print(f"\n{len(runs)} runs.  steps = mean over successful episodes only; "
              "failures end at the time limit and would measure that instead.")
    _emit(args, payload, table)


def cmd_run(args):
    r = evalscan.load_run(args.benchmark, args.run)
    d = r.as_dict()

    def table():
        st = f"{r.steps_on_success:.1f}" if r.steps_on_success else "-"
        print(f"{args.benchmark}/{r.name}\n  {r.path}")
        print(f"  success {r.success:.4f} over {r.n} episodes in {len(r.tasks)} tasks"
              f" | steps {st}")
        if r.config:
            print("  config " + "  ".join(f"{k}={v}" for k, v in sorted(r.config.items())))
        print()
        for n, t in sorted(r.tasks.items(), key=lambda kv: kv[1].success):
            ts = f"{t.steps_on_success:.0f}" if t.steps_on_success else "-"
            print(f"  {n:44.44s} {t.success:6.2f} ({t.n_success:3d}/{t.n:3d}) steps {ts:>6s}")
    _emit(args, d, table)


def cmd_compare(args):
    a = evalscan.load_run(args.benchmark, args.a)
    b = evalscan.load_run(args.benchmark, args.b)
    res = tradeoff.paired_tasks(a, b, args.episodes)

    def table():
        print(f"{args.a}  ->  {args.b}   (tasks with >= {args.episodes} episodes in both)")
        print(f"{'task':44s} {args.a[:9]:>9s} {args.b[:9]:>9s} {'delta':>8s}")
        for row in res["rows"]:
            print(f"{row['task']:44.44s} {row['a']:9.2f} {row['b']:9.2f} {row['delta']:+8.2f}")
        print(f"\n{res['tasks']} tasks:  {res['mean_a']:.4f} -> {res['mean_b']:.4f}"
              f"  ({res['mean_delta']:+.4f})")
        if res["excluded_incomplete"]:
            print(f"excluded {len(res['excluded_incomplete'])} incomplete task(s): "
                  + ", ".join(res["excluded_incomplete"][:6])
                  + (" ..." if len(res["excluded_incomplete"]) > 6 else ""))
    _emit(args, res, table)


def cmd_tradeoff(args):
    fast = evalscan.load_run(args.benchmark, args.fast)
    slow = evalscan.load_run(args.benchmark, args.slow)
    names = args.runs or [r.name for r in evalscan.list_runs(args.benchmark, args.filter)]
    loaded = [evalscan.load_run(args.benchmark, n) for n in names]

    # Restrict every run to the tasks all of them finished. Without this an
    # aggregate silently mixes a task with 2 episodes into a 50-episode mean,
    # and the ranking turns on that noise.
    common, dropped = None, []
    if args.episodes:
        sets = [r.complete_tasks(args.episodes) for r in ([fast, slow] + loaded)]
        common = set.intersection(*sets) if sets else set()
        seen = set().union(*[set(r.tasks) for r in ([fast, slow] + loaded)])
        dropped = sorted(seen - common)

    def stat(r):
        if common is None:
            return r.steps_on_success, r.success
        d = r.restricted(common)
        return d["steps_on_success"], d["success"]

    runs = [(r.name,) + stat(r) for r in loaded if stat(r)[0]]
    res = tradeoff.score(runs, stat(fast), stat(slow))
    res["restricted_to_tasks"] = len(common) if common is not None else None
    res["excluded_incomplete"] = dropped

    def table():
        (fx, fy), (sx, sy) = res["anchor_fast"], res["anchor_slow"]
        print(f"free trade: {args.fast} ({fx:.0f} steps, {fy:.3f}) -- "
              f"{args.slow} ({sx:.0f}, {sy:.3f}), "
              f"{res['slope_success_per_step']:.5f} success/step")
        if res["restricted_to_tasks"] is not None:
            print(f"restricted to the {res['restricted_to_tasks']} tasks every run "
                  f"finished ({args.episodes}+ episodes)")
        print(f"\n{'run':40s} {'steps':>7s} {'success':>8s} {'on line':>8s} {'excess':>8s} {'saved':>7s}")
        for row in res["rows"]:
            print(f"{row['run']:40.40s} {row['steps']:7.0f} {row['success']:8.3f} "
                  f"{row['on_line']:8.3f} {row['excess']:+8.4f} {row['steps_saved_frac']:6.0%}")
        if res["excluded_incomplete"]:
            print("excluded as incomplete: " + ", ".join(res["excluded_incomplete"]))
        print("\nexcess > 0: the gate bought success uniform compression could not "
              "buy at the same speed.")
    _emit(args, res, table)


def cmd_actions(args):
    eps = actions.load_episodes(args.dataset, args.episodes)
    if args.sweep:
        res = actions.sweep(args.benchmark, eps, args.chunk, args.k)

        def table():
            print(f"{res['benchmark']}: {res['chunks']} chunks from {len(eps)} episodes"
                  f"  (merge op: {res['merge_op']})")
            print(f"\n{'flag':22s} {'fires':>8s} {'mean':>8s} {'p90':>8s}")
            for fl, v in res["flags"].items():
                print(f"{fl:22s} {v['fire_rate']:8.3f} {v['mean']:8.3f} {v['p90']:8.3f}")
            dead = [f for f, v in res["flags"].items() if v["fire_rate"] < 0.005]
            sat = [f for f, v in res["flags"].items() if v["fire_rate"] > 0.98]
            for label, fs in (("never fires", dead), ("always fires", sat)):
                if fs:
                    print(f"\n{label}, so it carries no information: {', '.join(fs)}")
    else:
        res = actions.layout(eps, args.chunk, args.k, args.clip)

        def table():
            print(f"{res['episodes']} episodes, {res['steps']} steps, "
                  f"action_dim {res['action_dim']}")
            print(f"\n{'dim':>4s} {'min':>9s} {'max':>9s} {'mean|.|':>9s}  kind")
            for d in res["per_dim"]:
                print(f"{d['dim']:4d} {d['min']:9.3f} {d['max']:9.3f} {d['mean_abs']:9.3f}"
                      f"  {'binary' if d['binary'] else ''}")
            print(f"\nstep magnitude p50 {res['step_abs_p50']:.3f} "
                  f"p99 {res['step_abs_p99']:.3f}")
            print(f"single step over +-{args.clip}: {res['single_exceeds']:.4f}   "
                  f"summed over k={args.k}: {res['merge_exceeds']:.4f}")
            op = actions.MERGE_OP.get(args.benchmark, "?")
            if res["single_exceeds"] > 0.5:
                print(f"\nSingle steps already sit outside +-{args.clip}, so these are "
                      "absolute targets, not deltas: the clip test above does not apply "
                      f"to them. Compression must drop intermediate targets ({op}).")
            else:
                print(f"\nSingle steps stay inside +-{args.clip} while merged ones leave it, "
                      "so these are deltas: compression adds adjacent steps "
                      f"({op}) and the excess is displacement the robot never travels.")
    _emit(args, res, table)


def cmd_labels(args):
    if args.against:
        res = labels.agreement(args.tag, args.against)

        def table():
            print(f"{res['a']}  vs  {res['b']}   ({res['common_chunks']} shared chunks)")
            print(f"\n{'question':10s} {res['a'][:12]:>12s} {res['b'][:12]:>12s} {'shift':>9s}")
            for q, v in res["per_question"].items():
                print(f"{q:10s} {v['mean_a']:12.3f} {v['mean_b']:12.3f} {v['mean_shift']:+9.3f}")
    else:
        res = labels.scan(args.tag, args.expected)

        def table():
            print(f"{res['tag']}: {res['rows']} rows in {res['shards']} shards, "
                  f"{res['unique_chunks']} unique chunks")
            if args.verbose:
                for s_ in res["per_shard"]:
                    print(f"   shard {s_['shard']:>3s} {s_['rows']:8d}")
            if res["questions"]:
                print(f"\n{'question':10s} {'mean':>8s} {'sd':>8s} {'p90':>8s} {'>0.5':>8s}")
                for q, v in res["questions"].items():
                    print(f"{q:10s} {v['mean']:8.3f} {v['sd']:8.3f} {v['p90']:8.3f} "
                          f"{v['over_half']:8.3f}")
            print()
            if res["ok"]:
                print("no integrity problems found; safe to aggregate")
            else:
                for p_ in res["problems"]:
                    print(f"PROBLEM: {p_}")
    _emit(args, res, table)


def cmd_trace(args):
    src = args.source or trace.default_source(args.benchmark)
    rows = trace.load_episode(src, args.episode, args.task)
    if not rows:
        sys.exit(f"no rows for episode {args.episode}"
                 + (f" / task {args.task}" if args.task else "") + f" in {src}")
    out = Path(args.out) if args.out else Path(
        f"trace_{args.benchmark}_ep{args.episode}.html")
    out.write_text(trace.render(rows, args.episode, src, args.task, args.series))
    slots, others = trace.series(rows)
    payload = {"episode": args.episode, "task": args.task, "source": str(src),
               "chunks": len(rows), "questions": sorted(slots),
               "series": sorted(others), "out": str(out.resolve())}

    def table():
        print(f"episode {args.episode}: {len(rows)} chunks, "
              f"frames {rows[0].get('f')}-{rows[-1].get('f')}")
        print(f"  questions: {', '.join(sorted(slots)) or 'none'}")
        print(f"  series:    {', '.join(sorted(others)) or 'none'}")
        print(f"\nwrote {out.resolve()}")
        print("open it locally with:  scp <host>:" + str(out.resolve()) + " .")
    _emit(args, payload, table)


def cmd_labelcheck(args):
    res = labelcheck.score(args.parquet, args.benchmark, args.dataset,
                           args.slow, args.fast, args.episodes, args.column)
    ref = None
    if args.reference:
        ref = labelcheck.score(args.reference, args.benchmark, args.dataset,
                               args.slow, args.fast, args.episodes, args.column)
        res["vs_reference"] = labelcheck.compare(res, ref)

    def table():
        print(f"{Path(args.parquet).name}: {res['tasks']} tasks")
        print(f"\n{'task':24s} {'dK2':>7s} {'conf':>7s}")
        for r in res["rows"]:
            print(f"{r['task']:24s} {r['delta_k2']:+7.2f} {r['confidence']:7.3f}")
        print(f"\nSpearman(dK2, confidence) = {res['spearman']:+.3f}")
        print("positive means the labels rank tasks the way measured damage does; "
              "near zero means they carry no information about it.")
        if ref:
            v = res["vs_reference"]
            print(f"\nreference {Path(args.reference).name}: {v['reference']:+.3f}"
                  f"   delta {v['delta']:+.3f}   "
                  + ("PASS" if v["pass"] else "FAIL — do not train on these labels"))
    _emit(args, res, table)


def cmd_ckpt(args):
    rows = ckpt.inventory()

    def table():
        print(f"{'benchmark':10s} {'kind':8s} {'name':44s} {'size':>7s}  path")
        for r in rows:
            mark = "" if r["exists"] else "  [MISSING]"
            print(f"{r['benchmark']:10s} {r['kind']:8s} {r['name']:44.44s} "
                  f"{_human(r['bytes']):>7s}  {r['path']}{mark}")
    _emit(args, rows, table)


def build_parser():
    p = argparse.ArgumentParser(
        prog="qgate", description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="every command accepts --json for machine-readable output")
    p.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("paths", help="resolved workspace paths").set_defaults(fn=cmd_paths)

    j = sub.add_parser("jobs", help="slurm queue for this workspace")
    j.add_argument("--user", default=None)
    j.set_defaults(fn=cmd_jobs)

    def bench(sp):
        sp.add_argument("benchmark", choices=paths.BENCHMARKS)

    r = sub.add_parser("results", help="all evaluation runs for a benchmark")
    bench(r); r.add_argument("--filter", help="substring the run name must contain")
    r.set_defaults(fn=cmd_results)

    d = sub.add_parser("run", help="one run in detail, per task")
    bench(d); d.add_argument("run")
    d.set_defaults(fn=cmd_run)

    c = sub.add_parser("compare", help="per-task delta between two runs")
    bench(c); c.add_argument("a"); c.add_argument("b")
    c.add_argument("--episodes", type=int, default=50,
                   help="episodes a task needs before it counts (default 50)")
    c.set_defaults(fn=cmd_compare)

    t = sub.add_parser("tradeoff", help="rank runs against the success/speed line")
    bench(t)
    t.add_argument("--fast", required=True, help="blanket-compression anchor run")
    t.add_argument("--slow", required=True, help="uncompressed anchor run")
    t.add_argument("--runs", nargs="+", default=None,
                   help="runs to rank (default: every run in the benchmark)")
    t.add_argument("--filter", help="substring filter when runs are not listed")
    t.add_argument("--episodes", type=int, default=50,
                   help="restrict to tasks every run finished with this many "
                        "episodes; 0 to use each run's full aggregate")
    t.set_defaults(fn=cmd_tradeoff)

    a = sub.add_parser("actions", help="action layout and risk-flag rates")
    bench(a); a.add_argument("dataset", help="LeRobot dataset root")
    a.add_argument("--episodes", type=int, default=30)
    a.add_argument("--chunk", type=int, default=16)
    a.add_argument("--k", type=int, default=2)
    a.add_argument("--clip", type=float, default=1.0)
    a.add_argument("--sweep", action="store_true",
                   help="run the descriptor module instead of reporting layout")
    a.set_defaults(fn=cmd_actions)

    lb = sub.add_parser("labels", help="verify a labelling run before training on it")
    lb.add_argument("tag", help="shard tag, e.g. v6b_phase6")
    lb.add_argument("--expected", type=int, help="row count the run should have")
    lb.add_argument("--against", help="another tag to compare answers against")
    lb.add_argument("-v", "--verbose", action="store_true", help="per-shard row counts")
    lb.set_defaults(fn=cmd_labels)

    tr = sub.add_parser("trace", help="plot one episode's labels over time")
    tr.add_argument("benchmark", choices=list(paths.BENCHMARKS) + ["allex"])
    tr.add_argument("--episode", type=int, required=True)
    tr.add_argument("--task", help="task name, for benchmarks whose rows carry one")
    tr.add_argument("--source", help="label jsonl or glob (default: this benchmark's)")
    tr.add_argument("--out", help="output html path")
    tr.add_argument("--series", nargs="+",
                    help="plot exactly these computed series instead of the default pick")
    tr.set_defaults(fn=cmd_trace)

    lc = sub.add_parser("labelcheck",
                        help="score a label set against measured compression damage")
    lc.add_argument("parquet")
    lc.add_argument("--benchmark", default="robocasa", choices=list(paths.BENCHMARKS))
    lc.add_argument("--dataset", required=True, help="LeRobot dataset root")
    lc.add_argument("--slow", default="baseline_full_v2_with_action_steps",
                    help="uncompressed run")
    lc.add_argument("--fast", default="baseline_compress_K2", help="blanket-compression run")
    lc.add_argument("--reference", help="a known-good label parquet to compare against")
    lc.add_argument("--episodes", type=int, default=50)
    lc.add_argument("--column", default="p_yes")
    lc.set_defaults(fn=cmd_labelcheck)

    k = sub.add_parser("ckpt", help="checkpoints on disk")
    k.set_defaults(fn=cmd_ckpt)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if getattr(args, "user", "sentinel") is None:
        import getpass
        args.user = getpass.getuser()
    try:
        args.fn(args)
    except (FileNotFoundError, ValueError) as e:
        sys.exit(str(e))
    except BrokenPipeError:
        pass


if __name__ == "__main__":
    main()
