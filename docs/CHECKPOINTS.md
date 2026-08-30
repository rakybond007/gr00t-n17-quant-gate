# Checkpoint catalogue

Every model checkpoint this project produced: the three GR00T-N1.5 base
policies, the quantizability-gate student modules distilled from VLM teacher
labels, and the policy variants that were trained but not published.

Published under the HuggingFace account **[`prehj`](https://huggingface.co/prehj)**,
all repos public.

Last updated: 2026-08-30.

---

## 1. Base policies — published

All three are GR00T-N1.5 (`architectures: ["GR00T_N1_5"]`) finetuned per
benchmark from the same recipe. Uploaded contents are the `checkpoint-60000`
directory: `config.json`, `model-0000{1,2}-of-00002.safetensors`,
`model.safetensors.index.json`, `trainer_state.json`, `experiment_cfg/metadata.json`.

**Shared recipe** (from `trainer_state.json` + `training_args.bin`): 60,000
steps; batch size **64** effective (32 per device x 2 GPUs, no grad accumulation);
lr 1e-4 cosine with warmup ratio 0.05; weight decay 1e-5; bf16.
Backbone Eagle `NVEagle/eagle_er-qwen3_1_7B-Siglip2_400M_stage1_5_128gpu_er_v7_1mlp_nops`,
tapped at `select_layer 12`, `tune_llm: false` / `tune_visual: true`;
flow-matching DiT action head (16 layers, 32 heads), `tune_diffusion_model: true`,
`tune_projector: true`; `action_dim 32`, `action_horizon 16`, 4 denoising steps.

| Benchmark | What it is | Local path | HuggingFace | Size | Recipe delta |
|---|---|---|---|---|---|
| RoboCasa Kitchen (24 tasks) | Baseline (non-MoE) policy; the reference policy the gate modules are measured against. 3 views (left/right/wrist) 256x256 @20fps, delta EEF actions. Embodiment tag `new_embodiment`. Data: LeRobot `kimtaey/robocasa_mg_gr00t_300` (MimicGen). | `~/multigpu_workspace/Isaac-GR00T/ckpt/robocasa/groot/groot_n1_5_bs64_baseline/checkpoint-60000` | [prehj/GR00T-N1.5-robocasa-baseline](https://huggingface.co/prehj/GR00T-N1.5-robocasa-baseline) | 7.1 GB | 1.85 epochs, final loss 0.0324, 32 dataloader workers |
| LIBERO (`libero_10` for closed loop) | Baseline (non-MoE) policy. 2 views (front/left-wrist) 256x256 @10fps, delta EEF + euler RPY actions. Embodiment tag `libero`. | `~/multigpu_workspace/Isaac-GR00T/ckpt/libero/groot/groot_n1_5_bs64_baseline/checkpoint-60000` | [prehj/GR00T-N1.5-libero-baseline](https://huggingface.co/prehj/GR00T-N1.5-libero-baseline) | 7.1 GB | 14.04 epochs, final loss 0.0231, 32 dataloader workers |
| DexJoCo single-arm multitask (6 tasks) | Baseline (non-MoE) policy. 2 views (front/wrist) 640x640 @30fps, **absolute** arm(3+4) + 16-DoF hand actions. Embodiment tag `new_embodiment`. Data: `dexjoco_lerobot_datasets` (LeRobot v3.0 packed). | `~/multigpu_workspace/Isaac-GR00T/ckpt/dexjoco/groot/groot_n1_5_bs64_single_arm_multitask_baseline/checkpoint-60000` | [prehj/GR00T-N1.5-dexjoco-single-arm-baseline](https://huggingface.co/prehj/GR00T-N1.5-dexjoco-single-arm-baseline) | 7.1 GB | 17.53 epochs, final loss 0.0069, 16 dataloader workers |

Exact weight sizes, identical across all three:
`model-00001-of-00002.safetensors` 4,999,367,120 B ·
`model-00002-of-00002.safetensors` 2,586,508,600 B ·
`model.safetensors.index.json` 104,603 B.

---

## 2. Gate student modules — published

All four are RoboCasa students distilled from **one** VLM labelling pass. No
variant required re-calling the teacher.

**Teacher:** `nvidia/Cosmos3-Nano`, served locally through HF `transformers`
(not vLLM), scored from the next-token distribution over answer slots.
**Labels:** 247,887 action chunks from `kimtaey/robocasa_mg_gr00t_300`,
stride-8, 16 parallel judge shards, prompting generation "phase5" (phase-based
guidance + four grasp/hold-axis questions `q_A..q_D`, one call per chunk).
Label parquets live at `~/quantization_agent_workspace/assets/labels/robocasa/`.

**Shared training recipe** (`vlm_gate/scripts/train_gate_module.py`): 3 views at
128x128 concatenated to 9 channels; 384-d MiniLM instruction embedding;
BCE on the soft teacher score `p_yes`; episode-wise split, last 25% of episodes
held out; Adam lr 3e-4; 1 GPU. **No action inputs** (`act_cols: []`) — by design,
so the gate can run concurrently with action-head denoising.

Each repo ships `gate_module_best.pt` (**the one to serve**), `gate_module.pt`
(final epoch), `summary.json`, and `robocasa_task_embeddings.npz`.

| What it is | Local path | HuggingFace | Size (best / final) | Recipe |
|---|---|---|---|---|
| **phase5 binary** — SmallGate CNN, teacher labels aggregated with the original 0/1 computed risk flags. Best epoch 5, val AUC 0.674. | `~/quantization_agent_workspace/assets/modules_A/robocasa_module_A_phase5` | [prehj/quantgate-student-robocasa-phase5-binary](https://huggingface.co/prehj/quantgate-student-robocasa-phase5-binary) | 1,283,782 B / 1,283,528 B | 30 epochs, bs 256, lr 3e-4; labels `v6b_phase5_1call_full.parquet` |
| **phase5 softA (ratio)** — same CNN; all four computed flags made continuous as "fraction of K=2 merge pairs actually harmed". Best epoch 7, val AUC 0.918. | `~/quantization_agent_workspace/assets/modules_A/robocasa_module_A_phase5_softA` | [prehj/quantgate-student-robocasa-phase5-softA](https://huggingface.co/prehj/quantgate-student-robocasa-phase5-softA) | 1,283,782 B / 1,283,528 B | 10 epochs, bs 256, lr 3e-4; labels `v6b_phase5_softA.parquet` |
| **phase5 softB (event-preserving)** — same CNN; cumulative flags continuous but event-type flags (gripper transition, direction reversal) held near-saturating. Best epoch 6, val AUC 0.906. **Recommended CNN student.** | `~/quantization_agent_workspace/assets/modules_A/robocasa_module_A_phase5_softB` | [prehj/quantgate-student-robocasa-phase5-softB](https://huggingface.co/prehj/quantgate-student-robocasa-phase5-softB) | 1,283,782 B / 1,283,528 B | 10 epochs, bs 256, lr 3e-4; labels `v6b_phase5_softB.parquet` |
| **phase5 DINOv3 ViT-S/16** — `facebook/dinov3-vits16-pretrain-lvd1689m` frozen (weights bundled) + per-view embedding + learned attention-pooling query + MLP head. ~7 ms inference vs ~0.6 ms for the CNN. Best epoch 9, val AUC 0.639. | `~/quantization_agent_workspace/assets/modules_A/robocasa_module_A_phase5_dinov3s` | [prehj/quantgate-student-robocasa-phase5-dinov3s](https://huggingface.co/prehj/quantgate-student-robocasa-phase5-dinov3s) | 90,286,667 B / 90,285,448 B | 10 epochs, bs 128, lr 3e-4; `--encoder dinov3s`; labels `v6b_phase5_1call_full.parquet` (binary) |

### Closed-loop numbers (RoboCasa)

Computed with `bin/qgate tradeoff` (see [TOOLKIT.md](TOOLKIT.md)), restricted to
the **23 tasks every run finished with 50 episodes**. `TurnOnSinkFaucet` is
excluded: two of the runs have 2 and 5 episodes for it, and letting a 2-episode
task into a 50-episode mean is how a difference gets manufactured rather than
measured.

| configuration | success | steps on success | excess over line | steps saved |
|---|---|---|---|---|
| uncompressed | 0.656 | 330.0 | (anchor) | 0% |
| blanket K=2 | 0.599 | 216.0 | (anchor) | 35% |
| phase5 DINOv3 | 0.642 | 276.4 | +0.0128 | 16% |
| phase5 softA (ratio) | 0.635 | 266.6 | +0.0105 | 19% |
| phase5 binary | 0.638 | 274.5 | +0.0100 | 17% |
| phase5 softB (event-preserving) | 0.627 | 252.0 | +0.0099 | 24% |

The two anchors define the trade available for free — the line between them is
what selecting chunks at random already gives you — and its slope here is
0.00050 success per step. **Success rate alone cannot rank these.** A gate that
almost never fires reproduces the uncompressed row, one that always fires
reproduces the blanket row, and both are trivial; only distance above the line
is evidence that the gate picks the *right* chunks.

All four students clear the line, and **they are not separable from each other**.
The spread is +0.0099 to +0.0128, while the standard error on a success rate
measured over 1,150 episodes is about 0.014. An earlier tabulation put softB
clearly ahead at +0.0111 against +0.0078 for binary; that gap came from scoring
23-task run figures against 24-task anchors, and it disappears once both sides
sit on the same task set. What does separate them is speed at equal quality:
softB reaches its excess having removed 24% of the episode, against 16-19% for
the others.

Validation AUC is **not** comparable across students trained on different label
sets — softA/softB's ~0.91 is a property of continuous labels being easier to
rank-match, not better gating. AUC is useful here only as a
distillation-collapse detector.

---|---|---|
| uncompressed | 0.657 | 327.0 |
| blanket K=2 | 0.598 | 214.0 |
| phase5 binary | 0.638 | 274.5 |
| phase5 softA (ratio) | 0.635 | 266.6 |
| phase5 softB (event-preserving) | 0.627 | 252.0 |
| phase5 DINOv3 | 0.642 | 276.6 |

**Success rate alone cannot rank these.** Compression trades success for steps:
a gate that almost never fires reproduces the uncompressed row, one that always
fires reproduces the blanket row, and both are trivial. The meaningful comparison
is each gate's position against the straight line joining blanket-K=2 to
uncompressed — that line is what random chunk selection already gives you, and
only distance above it is evidence the gate is picking the *right* chunks.
On the project's tabulated 23-task closed-loop subset the excesses over that
line were binary +0.0078, softA +0.0076, softB **+0.0111**.

Note also that validation AUC is **not** comparable across students trained on
different label sets (softA/softB's ~0.91 is a property of continuous labels
being easier to rank-match). AUC is useful here only as a distillation-collapse
detector.

---

## 3. Local only — policy variants not published

Same shared recipe as section 1 (60k steps, effective batch 64, lr 1e-4 cosine)
unless noted. All under `~/multigpu_workspace/Isaac-GR00T/ckpt/`, ~7.1 GB per
checkpoint directory. Not uploaded: these are MoE / architecture-ablation and
extra-embodiment runs outside the scope of the gate release.

| Benchmark | Variant | Local path (relative to `ckpt/`) | Checkpoints |
|---|---|---|---|
| RoboCasa | Fair-MoE, body+DiT, length cost 0.10 | `robocasa/groot/groot_n1_5_bs64_fair_moe_b_d_lc_0p10` | 60000 |
| RoboCasa | MoE-4, per-expert body | `robocasa/groot/groot_n1_5_bs64_moe4_per_expert_body` | 60000 |
| RoboCasa | MoE-4, per-expert + meta-Q n8, no length cost | `robocasa/groot/groot_n1_5_bs64_moe4_per_expert_metaq_n8_no_length_cost` | 60000 |
| RoboCasa | MoE pyramid K3 (raw16/m8/m4), body only, no meta-Q, no balance | `robocasa/groot/groot_n1_5_bs64_moe_pyramid_K3_raw16_m8_m4_b_only_no_metaq_no_balance` | 60000 |
| RoboCasa | MoE pyramid K4 (raw16/m8/m4/m2), body only, no meta-Q, no balance | `robocasa/groot/groot_n1_5_bs64_moe_pyramid_K4_raw16_m8_m4_m2_b_only_no_metaq_no_balance` | 60000 |
| LIBERO | Fair-MoE v1 K4, body only, no meta-Q | `libero/groot/groot_n1_5_bs64_fair_moe_v1_K4_b_only_no_metaq` | 50000, 60000 |
| DexJoCo | **Dual-arm** multitask baseline | `dexjoco/groot/groot_n1_5_bs64_dual_arm_multitask_baseline` | 60000 |
| DexJoCo | Dual-arm multitask MoE-4 v1, balance | `dexjoco/groot/groot_n1_5_bs64_dual_arm_multitask_moe4_v1_balance` | 60000 |
| DexJoCo | Single-arm multitask MoE-4 v1, balance | `dexjoco/groot/groot_n1_5_bs64_single_arm_multitask_moe4_v1_balance` | 50000, 60000 |
| DexJoCo | Single-arm multitask MoE-4 v1, no balance | `dexjoco/groot/groot_n1_5_bs64_single_arm_multitask_moe4_v1_no_balance` | 50000, 60000 |

**Intermediate steps of published runs, also local only:**

| Benchmark | Path | Note |
|---|---|---|
| DexJoCo single-arm baseline | `dexjoco/groot/groot_n1_5_bs64_single_arm_multitask_baseline/checkpoint-50000` | earlier step of a published run; only step 60000 was uploaded |

Each run directory additionally holds a **top-level copy** of the final weights
(`model-*.safetensors`, `config.json`, `trainer_state.json`, `training_args.bin`,
`experiment_cfg/`) alongside its `checkpoint-*/` directories. `training_args.bin`
exists only at the top level — it is not inside `checkpoint-60000` and therefore
is not in the HuggingFace repos; the recipe it encodes is transcribed into each
model card instead.

---

## 4. Local only — earlier gate modules

Under `~/quantization_agent_workspace/assets/modules_A/`. All are SmallGate CNNs
(1,283,528 B final / 1,283,718–1,283,782 B best) unless noted. These predate the
phase5 labelling generation, were trained against **different and in several
cases unidentified teacher label sets**, and have no closed-loop measurement, so
they were not published. Their per-directory training scripts are not in the
repo; the teacher named in the directory name is the only record.

| Directory | Best-epoch metadata | Reading of the name — treat as provisional |
|---|---|---|
| `robocasa_module_A_cosmos9k` | no best ckpt, final only | Cosmos teacher, 9k-label set |
| `robocasa_module_A_cosmos9k_ep900` | no best ckpt, final only | as above, 900-episode label pool |
| `robocasa_module_A_cosmos9k_act_ep900` | epoch 3, AUC 0.880 | as above **with action descriptors fed to the student** |
| `robocasa_module_A_f9k_act_ep900` | epoch 3, AUC 0.640 | "f9k" (frontier 9k?) teacher, action-fed |
| `robocasa_module_A_f9k_act_hint_ep900` | epoch 32, AUC 0.453 | as above plus a "hint"; AUC below chance — a failed run |
| `robocasa_module_A_frontier_full` | epoch 2, AUC 0.795 | frontier-model teacher, full label set |
| `robocasa_module_A_gemma_full_restored` | epoch 1, AUC 0.710 | Gemma teacher, full set, restored from backup |
| `robocasa_module_A_luna`, `..._luna_ep900` | no best ckpt, final only | "luna" teacher |
| `robocasa_module_A_sonnet`, `..._sonnet_ep900` | no best ckpt, final only | Claude Sonnet teacher |
| `robocasa_module_A_tc_full` | no best ckpt, final only | "tc" teacher — **expansion unknown** |
| `robocasa_module_A_full` | **directory is empty** | nothing to publish |
| `real_module_A_cosmos_25kstep` | epoch 4, AUC 0.831 | real-robot (not RoboCasa sim) labels, Cosmos teacher, 25k steps |
| `real_module_A_luna_25kstep` | epoch 5, AUC 0.722 | real-robot labels, "luna" teacher |
| `real_module_A_luna` | no best ckpt, final only | real-robot labels, "luna" teacher |
| `robocasa_module_A_phase5_smoke` | epoch 1, AUC 0.630 | 1-epoch smoke test of the phase5 run — deliberately skipped |
| `robocasa_module_A_phase5_dinov3s_smoke` | epoch 1, AUC 0.574 | 1-epoch smoke test of the DINOv3 run — deliberately skipped |

The `real_module_A_*` group points at a real-robot ("allex") label set. No policy
exists for that embodiment, so no closed-loop threshold could ever be fitted for
those modules — the project's own record notes the deliverable there stops at
qualitative label videos.
