# Handoff — state as of 2026-08-31

Written so an agent starting cold can continue without re-deriving anything.
Read `WORKSPACE_MAP.md` for where things are, `REMOTE_AGENT.md` for how to
reach the cluster, `STATUS.md` for the per-run table, and this for what is
decided, what is in flight, and what is still open.

## Read this first: the repository is the source of truth

Files created in `~/quantization_agent_workspace/vlm_gate/` have disappeared
twice today, and been silently rewritten by something outside this session a
third time. The LIBERO guidance and question files vanished entirely — never
committed, so nothing could recover them and the wording had to be rewritten.
`libero_task_index.py` and its JSON vanished the same way and were restored
from the repository. On the third occasion the files came back with a
different umask (`-rw-r--r--` where this session writes `-rw-rw-r--`), which
says another process edits this tree.

So: **commit anything you create before you rely on it**, and when the
workspace and the repository disagree, check the repository first rather than
assuming the workspace is current. The one uncommitted prompt in this project
is the one that was lost.

## What the project is

A gate that decides, per 16-step action chunk, whether a GR00T policy can
execute that stretch at half rate. Compressing everything uniformly costs too
much success; compressing the right chunks costs little. A VLM teacher labels
chunks offline, a 1.3 MB student is distilled from those labels, and the
student runs in the closed loop alongside the policy.

On RoboCasa's 24 tasks: uncompressed 0.656 success at 330 steps, blanket K=2
0.599 at 216, gate 0.627–0.642 at 252–276.

## Decisions that are settled — do not relitigate

**Success rate cannot rank gates.** Compression trades success for speed, so a
gate that compresses nothing wins on success and is worthless. Rank by
position above the line joining blanket compression to no compression, over
the tasks every run finished. `qgate tradeoff` does it.

**Never judge a labelling job by its slurm state.** These jobs end on `kill`;
the exit code is `kill`'s. A preempted, requeued shard can re-emit rows. Run
`qgate labels <tag> --expected <N>` before aggregating.

**The four RoboCasa phase5 students are statistically indistinguishable**
(+0.0099 to +0.0128 excess against a standard error near 0.014). What separates
them is speed at equal quality. An earlier claim that softB clearly won came
from mixing a 23-task run figure with 24-task anchors.

**Validation AUC does not compare across label sets.** Continuous labels are
easier to rank-match; 0.918 against 0.674 is not a better gate. Use AUC only to
detect distillation collapse.

**The VLM is asked only what the arithmetic cannot answer.** Computed values
are stated to it as facts and never asked about. An early version that asked
them collapsed to two distinct answers across the dataset.

**Score a label set against measured compression damage before training on
it.** `qgate labelcheck`. A generation that ranks tasks uncorrelated with what
K=2 actually costs them is not a labelling to train on, however lively its
questions look. This is how phase6 was lost.

**Do not drop a question because its answers are low.** The values are
continuous and contribute to noisy-OR regardless. Only the closed loop can
retire a question. This was learned by dropping two and having to restore them.

**Embodiment decides the compression operation.** End-effector deltas
(RoboCasa, LIBERO) are summed; absolute joint targets (dexjoco, allex) have
intermediate targets dropped. `qgate actions` diagnoses which.

## In flight right now

| what | id | state |
|---|---|---|
| N1.7 joint training, gate | 145335 | 60k / batch 64, continuous labels, loss split logged |
| N1.7 joint training, baseline | 145336 | matched |
| LIBERO dense labelling, 16 shards | 145246 | every frame, ~274k chunks |

When LIBERO labels arrive: `qgate labels libero_dense_s16` for integrity, then
`qgate labelcheck` against the measured per-task K=2 damage. The second one
gates training. All 40 LIBERO tasks join to evaluation, so it can be scored
before the full pass finishes.

The joint runs now log `gate_loss`, `action_loss_only` and `gate_valid_n`
separately, so whether attaching the gate hurt the policy is readable from the
log rather than from re-evaluating a checkpoint 25 hours later.

## Ready to launch, deliberately not launched

LIBERO and dexjoco labelling harnesses are built and smoke-tested; neither full
run has been submitted. Both were held back for two reasons: they cost
significant GPU time, and phase6's closed loop has not yet shown that the
five-axis prompt approach is worth propagating.

```bash
# LIBERO   (tiles first, then merge the manifest, then label)
sbatch vlm_gate/run_scripts/label/sbatch_libero_tiles.sh
python vlm_gate/scripts/gen_libero_tiles_shard.py merge 16
sbatch vlm_gate/run_scripts/label/sbatch_libero_label.sh
qgate labels libero_v1_s16 --expected 67298

# dexjoco
MODEL_OUTPUT_DIR=/rlwrld-unified-checkpoints/hojin2/gate_modules/dexjoco_v1 \
  sbatch vlm_gate/run_scripts/label/sbatch_dexjoco_tiles.sh
python vlm_gate/scripts/dexjoco_tiles_manifest.py
MODEL_OUTPUT_DIR=/rlwrld-unified-checkpoints/hojin2/gate_modules/dexjoco_v1 \
  sbatch vlm_gate/run_scripts/label/sbatch_dexjoco_label.sh
```

## Open questions, with what is known about each

**Labelling stride is inconsistent and nobody chose it.** RoboCasa labels every
8 steps (50% overlap with a 16-step chunk), allex every 16 (none), dexjoco 16,
LIBERO 4 (75%). LIBERO's 4 came from matching RoboCasa's wall-clock spacing of
0.40 s, but the gate decides per chunk, not per second, so the invariant that
matters is overlap, not seconds. Deciding before launching costs nothing;
after, it costs a re-run. Recommendation on the table: new benchmarks at 16,
RoboCasa left alone since three generations already exist at 8.

**The dataset join has no guard.** Labels are stored apart from the dataset and
joined on `(episode_index, frame_index)`. If a dataset is regenerated with the
same numbering and different content, the join still succeeds and every label
silently points at the wrong frame. `qgate/fingerprint.py` computes a
fingerprint from `meta/` and can verify a stamped directory; **it is written
but not yet wired into training or the labelling verifier.**

**The event-preserving aggregation cannot be reproduced.** The script that
generated phase5's softB variant is not in the tree. Its parquet survives so
the trained student is fine, but the variant cannot be regenerated for another
benchmark. phase6 was matched to softA for this reason.

**dexjoco questions C and E answered near zero in the smoke** (sd 0.010, 258
chunks). This has the shape of the RoboCasa D/E case, where the cause turned
out to be guidance that never described those axes rather than bad questions.
Not acted on: the sample is small and dexjoco has no gated closed loop to
judge with.

**allex `Bring Object` ceiling.** Not a defect — checked. The ceiling is 3.0 as
a prior and stage-2 question B pulls it toward 1.0 during the precise stop.
Verified: chunks that end at K=1 have mean B 0.819, chunks at K=3 have 0.426.

## Things that have gone wrong before

- A resume parser split on whitespace and silently dropped every success on a
  resumed run.
- An eval job selected zero tasks from an array index, started its servers, and
  "succeeded" with no client having run.
- Training unfroze the backbone inside a conditional branch, so the control run
  trained frozen and the comparison was void.
- `OUT_SUFFIX` was ignored and nearly overwrote a baseline checkpoint an
  evaluation was queued against.
- A per-task regression was reported from 1 of 5 episodes against 30 of 50.
- The gate head in every N1.7 joint run received no gradient at all. The label
  join read the episode index off a DataFrame that never carries it, so every
  sample was marked unlabelled and the gate loss was exactly zero for 10,000
  steps. The smoke that was supposed to catch this applied BCE to the head
  directly and never called the model's forward — it verified the component,
  not the wiring.
- `--max-steps` and `--global-batch-size` never reached python. Comment lines
  sat between backslash-continued arguments, so the joined line commented out
  everything after `--output-dir`, and every run silently used config defaults.
  A job named 60k trained 10,000 steps.
- A whole labelling generation (phase6) was accepted on question liveness and
  turned out to rank tasks uncorrelated with what compression costs them.
- `bin/qgate` dereferenced `$HOME` under `set -u`, so it died in exactly the
  bare shell `ssh host 'cmd'` provides — the mode it was written for.
- Joint training ran on the binary label variant whose 29.51% tied block takes
  arbitrary ranks, contradicting this project's own documented standard.

The pattern in all of them: the failure was invisible in the thing being
watched (a job state, a mean, an exit code) and visible only in the artefacts.
That is why the toolkit reports counts alongside every number.

## Conventions

Prompts and code comments in English; explanations to the user in Korean.
Terminology in prose, not filenames: say "binary / ratio / event-preserving
aggregation", not "softA / softB". Commit messages explain why, not what.
