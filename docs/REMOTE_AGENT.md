# Working from a laptop, with the work on a cluster

From now on the agent runs on the local machine and the compute is elsewhere.
This page is about what that changes. The short version: less than it looks,
but not nothing, and the parts that do change are worth designing for rather
than discovering.

## How a coding agent actually works

It is worth being concrete, because the mental picture matters here.

An agent does not watch a screen. Each step, it emits a command, something runs
it, and the **complete stdout and stderr come back as text** along with the
exit code. That text is what the agent reads. There is no terminal, no cursor,
no scrollback to squint at — the output of `ls` reaches the agent the same way
whether the agent is on the machine or not.

That is why the remote case is less exotic than it seems. When the agent runs

```bash
ssh cluster 'qgate results robocasa'
```

that is a **local command**. `ssh` runs locally, connects, runs the remote
program, and pipes its output back to local stdout. The agent gets the same
text it would have got on the cluster. Nothing is being scraped.

So the working loop is unchanged: emit a command, read the result, decide.
What changes is everything that depended on the agent *living* on the machine.

## What changes, honestly

**Latency, and therefore batching.** Each `ssh` pays a round trip. Ten small
commands cost ten round trips; one command with ten things in it costs one.
Use a shared connection so the handshake happens once:

```
# ~/.ssh/config
Host cluster
    HostName <login-node>
    User hojin2
    ControlMaster auto
    ControlPath ~/.ssh/cm-%r@%h:%p
    ControlPersist 10m
```

**No background-task notifications.** On the cluster the agent could start a
job and be woken when it finished. Remotely nothing wakes it. Long work is
submitted with `sbatch` and *asked about later* — which is why `qgate jobs`
and `qgate labels` exist, and why every verdict in this project is defined on
artefacts rather than on job state.

**Watching is gone.** Several real bugs in this project were caught by noticing
something odd while it ran: five labelling shards finishing in under a minute,
a training run using the wrong label column. Remotely nobody is looking. The
compensation is that those checks became commands — `qgate labels`, and the
dataset fingerprint — so they run whether or not anyone is watching.

**Interactive debugging needs a session.** A pdb prompt, a REPL, an
`srun --pty` allocation: these need a shell that stays alive between commands.
That is what `tools/dev` is for.

## The two modes

### One-shot: `ssh cluster '<command>'`

The default, and enough for nearly everything: reading results, submitting
jobs, checking files, running analysis.

```bash
ssh cluster '~/quantization_agent_workspace/bin/qgate results robocasa'
ssh cluster '~/quantization_agent_workspace/bin/qgate --json labels v6b_phase6_s16'
ssh cluster 'cd ~/quantization_agent_workspace/vlm_gate && sbatch run_scripts/label/sbatch_libero_label.sh'
ssh cluster 'tail -40 ~/quantization_agent_workspace/vlm_gate/out/144470-*.out'
```

**The catch:** this runs a non-login shell. No profile is sourced, `PATH` has
no conda in it, and on some setups `HOME` is not even set. Anything you invoke
this way has to tolerate that. `bin/qgate` was written to — it finds its own
interpreter and never dereferences an unset variable — and that was verified by
running it under `env -i`. A script that assumes an activated environment will
fail here in a way that looks like a bug in the script.

### Stateful: `tools/dev`

Keeps one tmux session on the cluster and drives the shell inside it, so `cd`,
`export`, `conda activate` and an `srun` allocation all persist between
commands. It is **not** screen-scraping: a command's output is redirected to a
file and the whole file plus its exit code is retrieved.

```bash
export DEV_HOST=hojin2@<login-node>
tools/dev up                    # ensure the session
tools/dev alloc --gpus=1        # hold a GPU node; later commands run inside it
tools/dev run 'python scripts/smoke_n17_gate_train.py'
tools/dev bg  'python scripts/long_thing.py'    # send without waiting
tools/dev tail 100              # peek at a running command
tools/dev cap                   # real screen capture — only for pdb and prompts
tools/dev free                  # release the allocation
```

`dev run` also rsyncs the local `scripts/`, `run_scripts/`, `analysis/`,
`tools/` and `docs/` to the cluster first, which is the answer to "I edited a
file locally, how does the cluster see it".

## Getting things back

**Numbers:** `--json` on any `qgate` command, read directly from stdout.

**A file:** `scp cluster:/path/to/file .`

**Something visual:** every `qgate` command that draws writes one
self-contained HTML file with no external dependencies, precisely so it
survives a single `scp`. It can also be published as an artifact and viewed in
a browser without copying anything.

```bash
ssh cluster '~/quantization_agent_workspace/bin/qgate trace allex --episode 0 --out /tmp/ep0.html'
scp cluster:/tmp/ep0.html .
```

**A running tmux session someone else started:**
`ssh cluster 'tmux capture-pane -p -t <session> -S -3000'` gives the scrollback
as text. Useful for looking in on something; not how you should run new work.

## Editing code

The repository is the source of truth for code and it is on both sides. Two
ways to move an edit:

- commit and `git pull` on the cluster — durable, and what to use for anything
  that will run more than once;
- `tools/dev run`, which rsyncs first — faster while iterating.

Do not edit files on the cluster directly and let them drift; the repo copy is
what the next person clones. When they have already drifted, the workspace is
what actually ran, so copy from it into the repo rather than the reverse.

## What to check before trusting a result

Remotely you cannot glance at a job and see it is wrong, so the checks are
explicit:

```bash
qgate jobs                              is it even running
qgate labels <tag> --expected <N>       did the labelling really finish
qgate results <bench>                   episode counts, not just success
qgate compare <a> <b>                   only tasks both runs completed
```

A slurm state of COMPLETED means nothing here: the labelling jobs end on a
`kill`, so their exit code is the exit code of `kill`. Judge on rows and files.
