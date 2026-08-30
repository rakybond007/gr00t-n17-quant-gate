# Where each benchmark stands

Four stages have to complete before a prompt can be judged: the prompt itself,
a labelling pass over the dataset, a distilled student, and a closed-loop
evaluation. A prompt is only *finished* when it is frozen and fully applied;
it is only *good* once the closed loop says so. Those are different claims and
this page keeps them apart.

Snapshot of 2026-08-30. Every number below was read off disk with `bin/qgate`,
not carried over from notes.

### Every run, one row each

`gen` is the labelling generation — the prompt that produced the targets.
`aggregation` is how the deterministic risk flags were turned into a number:
*binary* is the original 0/1, *ratio* makes all four continuous, *event-kept*
makes them continuous except gripper transition and direction reversal, which
stay at full strength because one such event in a window is already decisive.

| benchmark | gen | prompt | aggregation | encoder | trained | val AUC | closed loop | success | steps | excess |
|---|---|---|---|---|---|---|---|---|---|---|
| RoboCasa | v6b | ttl_aligned, 4Q | binary | — | no | — | no | | | |
| RoboCasa | phase5 | guidance v3, 4Q | binary | SmallGate 1.3M | yes, ep5 | 0.674 | 1200 ep / 24 tasks | 0.638 | 274.5 | +0.0100 |
| RoboCasa | phase5 | guidance v3, 4Q | ratio | SmallGate 1.3M | yes, ep7 | 0.918 | 1155 ep / 24 tasks | 0.635 | 266.6 | +0.0105 |
| RoboCasa | phase5 | guidance v3, 4Q | event-kept | SmallGate 1.3M | yes, ep6 | 0.906 | 1200 ep / 24 tasks | 0.627 | 252.0 | +0.0099 |
| RoboCasa | phase5 | guidance v3, 4Q | binary | DINOv3 ViT-S/16 87M | yes, ep9 | 0.639 | 1152 ep / 24 tasks | 0.642 | 276.4 | +0.0128 |
| RoboCasa | **phase6** | guidance v5, 5Q | binary | — | rejected | — | no | | | |
| RoboCasa | **phase6** | guidance v5, 5Q | ratio | — | **rejected** | — | not run | | | |
| RoboCasa | phase6 | guidance v5, 5Q | event-kept | — | blocked | — | no | | | |
| LIBERO | v1 | guidance v1, 5Q | — | — | no | — | no | | | |
| dexjoco | v1 | guidance v1, 5Q | — | — | no | — | no | | | |
| allex | v1 | two-stage, per-task ratio | — | — | n/a | — | **impossible** | | | |

Anchors the excess column is measured against, over the 23 tasks every run
finished: uncompressed 0.656 at 330 steps, blanket K=2 0.599 at 216. Success
and steps in this table are the full 24-task aggregates, which is why they
differ slightly from the 23-task figures the excess uses.

Two rows need reading carefully. **`phase6` × `event-kept` is blocked, not
skipped**: the script that generated that aggregation for phase5 is not in the
tree, so the variant cannot be regenerated. Its phase5 parquet survives, so
that student is fine, but nothing new can be matched to it. And **validation
AUC does not compare across rows with different aggregations** — 0.918 against
0.674 is a property of continuous labels being easier to rank-match, not a
better gate. It is only useful as a distillation-collapse detector, which is
why the closed-loop columns exist.

### phase6 was rejected

Its labels do not rank tasks the way compression damage does. Spearman between
per-task label confidence and the measured cost of blanket K=2 is **+0.019**,
against **+0.420** for phase5 — not inverted, uncorrelated.

It was built to fix a dead question: phase5's fourth question answered 0.007 on
average, and the diagnosis was that the guidance never described that axis. The
rewrite achieved exactly what it set out to — that question went to 0.067 — and
the labels stopped predicting the only thing they exist to predict. Question
liveness is a property of the questions, not of the labels, and it was never
checked against anything external.

The check now exists and takes twenty seconds:

    qgate labelcheck <parquet> --dataset <root> --reference <phase5 parquet>

Run it on a pilot before labelling a full pass. The student and its evaluation
were deleted; the label parquets are kept as the evidence for this.

### Stage completion

| | Prompt | Labelling | Student | Closed loop |
|---|---|---|---|---|
| **RoboCasa phase5** | final | 247,887 verified | 4 architectures | 4 of 4 evaluated |
| **RoboCasa phase6** | **rejected** | 247,887 verified | none | none |
| **LIBERO** | final | none | none | baselines only |
| **dexjoco** | final | none | none | baselines only |
| **allex** | final | 14,809 done | n/a | impossible — no policy |

---

## RoboCasa Kitchen — 24 tasks x 50 episodes

The only benchmark where the full chain is closed.

**Prompt.** `GUIDANCE=phase6` selects `robocasa_guidance_phase_v5.txt` and five
questions; `phase5` selects `v3` and four. Both live in
`vlm_gate/analysis/_evolver/_varkA/`, and the v1-v5 ladder is kept because
each version is the control for the next.

The five phase6 axes are one safe signal and four risks: empty space (safe);
holding a handle, knob, lever or door edge; the moment a load transfers;
having to lift over or go around an obstruction; having to drive the base
while the arm is still moving.

**Labelling.** Three full passes over the same 247,887 chunks, so they are
directly comparable: `v6b_s16`, `v6b_phase5_s16`, `v6b_phase6_s16`, plus a
2,998-chunk phase6 pilot at `v6b_phase6_s4`. All pass
`qgate labels <tag> --expected 247887` with no duplicates and no empty shards.

Answer spread over the phase6 pass:

| | A (safe) | B | C | D | E |
|---|---|---|---|---|---|
| mean | 0.237 | 0.154 | 0.056 | 0.067 | 0.020 |
| sd | 0.179 | 0.209 | 0.090 | 0.077 | 0.028 |

C, D and E answer low. They were **not** dropped for it: the values are
continuous and feed noisy-OR whatever their mean, and the only instrument that
can retire a question is the closed loop.

phase6 disagrees with phase5 on 32.8% of chunk decisions (Spearman 0.598 on
`p_yes`), so the two are materially different labellings and the comparison is
worth running.

**One finding that is not about prompts.** With binary computed flags, any
chunk where a flag fires collapses to exactly zero confidence regardless of
what the VLM answered: 29.51% of chunks, *identical* in phase5 and phase6. No
prompt change could ever have moved it. Recomputing the flags continuously
takes it to 0.08%. This is why every comparison runs on continuous flags.

**Students.** Four trained on phase5 labels; a fifth is training on phase6
(`robocasa_module_A_phase6_softA`, job 144470). It is matched to phase5's
softA in computed layer, aggregation and recipe — 10 epochs, batch 256,
lr 3e-4 — so the prompt is the only variable.

**Closed loop.** 99 runs. Anchors: uncompressed 0.656 at 330 steps, blanket
K=2 0.599 at 216, over the 23 tasks every run finished.

| student | success | steps | excess | steps saved |
|---|---|---|---|---|
| DINOv3 ViT-S/16 | 0.642 | 276.4 | +0.0128 | 16% |
| softA (ratio) | 0.635 | 266.6 | +0.0105 | 19% |
| binary | 0.638 | 274.5 | +0.0100 | 17% |
| softB | 0.627 | 252.0 | +0.0099 | 24% |

All four clear the line and **none is separable from the others**: the spread
is 0.003 against a standard error near 0.014. What separates them is speed at
equal quality.

**Blocking the next step:** nothing. Training then evaluation.

---

## LIBERO — 40 tasks x 50 episodes

Prompt written, nothing labelled.

**Prompt.** `_libero/libero_guidance_v1.txt` and `libero_questions_v1.txt`,
five questions. The deterministic descriptors are calibrated on this data
rather than copied from RoboCasa: position delta p50 0.518 / p99 1.155, a
single step never leaves the controller's +-1 but a K=2 merge does on 7.9%,
gripper transition rate 1.45%.

The `reversal` flag was removed rather than re-thresholded. LIBERO
trajectories are very smooth — adjacent-step cosine median 0.998, and
cos < 0 on 0.02% of steps — so reversal is not rare, it is absent. A
continuous `turn` flag cut at the data's own 5th percentile replaced it.

**What the questions are aimed at.** Blanket K=2 costs very different amounts
per suite:

| suite | uncompressed | blanket K=2 | delta |
|---|---|---|---|
| libero_spatial | 0.968 | 0.714 | -0.254 |
| libero_goal | 0.938 | 0.734 | -0.204 |
| libero_10 | 0.840 | 0.800 | -0.040 |
| libero_object | 0.984 | 0.954 | -0.030 |

The reading the questions encode — **destination forgiveness**, not the
motion — is a hypothesis, not a settled result. It fits the ordering:
`libero_object` places into a basket that catches the object and barely
suffers, while the two suites that place onto or against a referent lose a
quarter of their success. It has not been tested, and testing it is what the
labelling run is for.

Within-suite spread is wide enough that suite means hide a lot: across
`libero_10`'s ten tasks the K=2 delta runs from -0.28 to +0.20. Attributing
that to specific instructions is not currently possible — the result
directories are numbered (`libero_10_0`), and no task-index-to-instruction
mapping is stored alongside them.

**Closed loop.** 56 runs, all uncompressed or blanket-K baselines: raw
0.9325 at 159.9 steps, K=2 0.8005 at 109.4, and the K3/K4/clip/dyn/var
ladder. (Step means here pool all successful episodes; the benchmark notes
kept with these results macro-average per task instead and so read higher.) **No gated run exists**, because no student exists.

**Blocking the next step:** the tile manifest and the labelling job script.
The prompt and descriptors are ready; the harness that feeds frames to the
judge is not built for this benchmark.

---

## dexjoco single-arm — 6 tasks x 50 episodes

Prompt written, nothing labelled.

**Prompt.** `_dexjoco/dexjoco_guidance_v1.txt` and `dexjoco_questions_v1.txt`.

**Embodiment.** Absolute joint targets, not deltas: `qgate actions` shows
99.95% of single steps already outside +-1, which is the diagnostic. So
compression drops intermediate targets instead of summing adjacent ones, and
the controller-range test that governs RoboCasa and LIBERO does not apply.

**Descriptor sweep, per task** (fire rate, 30 episodes each):

| task | hand transition | infeasible merge | precise hold | reversal |
|---|---|---|---|---|
| fold_glasses | 0.673 | 0.488 | 0.159 | 0.000 |
| pick_bucket | 0.575 | 0.447 | 0.032 | 0.000 |
| pinch_tongs | 0.488 | 0.166 | 0.716 | 0.001 |
| click_mouse | 0.364 | 0.219 | 0.241 | 0.002 |
| water_plant | 0.360 | 0.174 | 0.322 | 0.002 |
| hammer_nail | 0.261 | 0.625 | 0.026 | 0.100 |

`reversal` looks dead on five of six tasks and fires on 10% of `hammer_nail`
chunks, because hammering *is* reversal. It is a task-specific detector, not a
broken one, and `hammer_nail` is among the tasks compression hurts most. A
benchmark-wide average would have hidden that in both directions — which is
the argument for sweeping per task.

**Closed loop.** Three runs, 300 episodes each:

| | success | steps |
|---|---|---|
| K=1 | 0.760 | 375.6 |
| K=2 | 0.593 | 317.6 |
| K=3 | 0.450 | 326.1 |

Worst hit by K=2: fold_glasses -0.30, click_mouse -0.26. K=3 shows *more*
steps than K=2 purely through selection — its surviving episodes are an easier
subset — which is a reminder that step means across runs of different
difficulty are not strictly comparable.

**Blocking the next step:** the labeller. The dataset reads correctly and the
descriptors are written; the judge harness is not wired for this embodiment.

---

## allex — real robot

Labelled and delivered; it can never have a closed loop.

**Prompt.** Two-stage, with a task-specific second stage that sets a per-task
compression ceiling rather than assuming K=2 everywhere: Pass Object 3x,
Rotate PolyBag 2.5x, Rotate Box 2x, Bring Object 1x (it has to stop precisely,
so it does not compress at all).

**Labelling.** 14,809 chunks across 6 shards, with the variable ratio applied
and a labelled video rendered.

**Why there is no closed loop.** No policy exists for this embodiment, so no
threshold can ever be fitted and no success rate can be measured. The
deliverable stops at qualitative labelled video, and that is a property of the
setup, not an unfinished step.

---

## What is actually missing

1. **RoboCasa phase6 closed loop.** Training, then evaluation. Nothing blocks it.
2. **LIBERO labelling harness** — tiles, manifest, job script.
3. **dexjoco labelling harness** — same.
4. **softB is not reproducible.** The script that generated the softB label
   variant is not in the tree. Its parquet survives, so the trained student is
   fine, but the variant cannot be regenerated for another benchmark. phase6
   was matched to softA for exactly this reason.
