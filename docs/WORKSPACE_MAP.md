# The server workspace, and what of it lives in this repository

This repository is code. The work happens in a workspace on the cluster that
cannot be cloned. Measured: **254 GB of evaluation output**, 723 GB of
checkpoints, a 37 GB frame cache, 13 GB of probe features. The repository is
about 70 MB. An agent working from a laptop has the repository and reaches
everything else over ssh.

So there are two questions to keep straight, and this page answers both: **what
is where**, and **which side of the line it is on**.

Workspace root on the cluster: `~/quantization_agent_workspace`
(`/sjw_alinlab/home/hojin2/quantization_agent_workspace`). Everything below is
relative to it unless stated.

---

## The line

| | In the repo | Server only |
|---|---|---|
| Scripts, job scripts, prompts, docs, tools | yes | mirrored |
| GR00T model code (`gr00t/`, `scripts/`) | yes | mirrored |
| Label parquets, frame cache, checkpoints | no | yes |
| Evaluation output, logs, wandb | no | yes |
| Datasets | no | yes, and mostly outside the workspace |
| Conda environments, venvs | no | yes |

The rule behind it: anything a human wrote belongs in the repo; anything a job
produced does not. Two consequences worth knowing before you go looking. The
repo has no `output/`, so `qgate` needs `QGATE_WS` pointing at the workspace to
read results. And the repo's copy of a script can drift from the workspace's —
the workspace is what actually ran, so when they disagree, believe the
workspace and sync.

---

## Directories that matter

### Code you will edit

```
vlm_gate/scripts/          137 python scripts — the whole pipeline
vlm_gate/run_scripts/      91 slurm job scripts, in label/ train/ eval/ cache/
vlm_gate/qgate/            the reading toolkit (also at tools/qgate in checkouts)
vlm_gate/analysis/_evolver/  the prompts: guidance texts and question sets
bin/qgate                  entry point; works from any directory, no env needed
docs/                      what you are reading
```

`vlm_gate/analysis/_evolver/` has more in it than you want. The versions
actually in use are `_varkA/robocasa_guidance_phase_v{1..5}.txt`,
`_libero/libero_{guidance,questions}_v1.txt` and
`_dexjoco/dexjoco_{guidance,questions}_v1.txt`. Everything else is the
evolution history of earlier cycles.

### Output you will read, never edit

```
vlm_gate/output/robocasa/     99 evaluation runs
vlm_gate/output/libero/       56 runs
vlm_gate/output/dexjoco/      3 runs
vlm_gate/output/allex_v2/     allex labels and their render
vlm_gate/output/_gate_distill/  labelling shards (*.jsonl) and tile directories
vlm_gate/out/                 slurm .out/.err, named <jobid>_<array>-<jobname>
```

A run directory is `<run>/<task>/prediction.txt`, one line per episode. Read
these with `qgate`, not by hand — the parsing has traps (see TOOLKIT.md).

**`output/_gate_distill/` holds tile directories with hundreds of thousands of
PNGs.** Never `ls`, `find` or `du` them. They are reached through their
manifest files, which is why the manifests exist.

### Artefacts

```
assets/labels/<benchmark>/*.parquet    teacher labels, joined by (episode_index, frame_index)
assets/modules_A/<name>/               trained students; gate_module_best.pt is the one to serve
assets/frame_cache_robocasa/           37 GB memmap; frames_shard*.u8 + index_shard*.parquet
assets/checkpoints/                    723 GB — older policy and gate checkpoints
assets/probe_features/                 13 GB backbone-feature probes
assets/robocasa_task_embeddings.npz    384-d instruction embeddings
assets/datasets/                       local copies (dexjoco, allex renders)
```

Policy checkpoints are **outside** this workspace, at
`~/multigpu_workspace/Isaac-GR00T/ckpt/<benchmark>/groot/<run>/checkpoint-60000`.
`docs/CHECKPOINTS.md` lists every one with its HuggingFace URL.

### Datasets

```
robocasa  /sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/kimtaey/robocasa_mg_gr00t_300
libero    /sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/kimtaey/libero_gr00t_delta
dexjoco   assets/datasets/dexjoco_lerobot/dexjoco_lerobot_datasets/<task>/
allex     /rlwrld2/home/david/action_quantization/v1/subtask_labeled_data_update_eef_256x256_hojin
```

The first two belong to a colleague and the last to another; **all four are
read-only to us in practice, and nothing in this project writes to them.**

### Environments

```
~/miniconda3/envs/quant_gate        general work: numpy 1.23.5, transformers 4.51.3
~/miniconda3/envs/quant_gate_eval   labelling clients and training
cosmos_judge_venv/                  the VLM judge: transformers 5.12.1, cu124
pylibs/tf4573                       PYTHONPATH overlay for N1.7 only
```

The home directory has ~35 conda environments from other projects; only the
four above belong to this one. These are pinned and they fight each other. The judge needs a transformers far
newer than the training stack tolerates, which is why it runs as a **separate
process behind a socket** rather than being imported. Do not try to unify them.

### Other repositories in the workspace

```
Isaac-GR00T/         the N1.5 tree, a colleague's fork — do not push to it
Isaac-GR00T-n17/     the N1.7 tree
quantization_label_gr00t/   this repo's checkout
groot-n17-quant-gate/       the N1.7 portable repo
```

---

## The pipeline, and where each stage leaves its output

```
1  tiles        gen_<bench>_tiles_shard.py     -> output/_gate_distill/<dir>/tiles/*.png
                                                  + a flat manifest .txt
2  label        cosmos_1call_v6.py (robocasa)  -> output/_gate_distill/<tag>_s<N>_<i>.jsonl
                <bench>_label_chunks.py         one judge server per shard, port 8900+i
3  verify       qgate labels <tag>              rows, duplicates, answer spread
4  aggregate    aggregate_*.py                 -> assets/labels/<bench>/*.parquet
5  continuous   recompute_soft_*.py             makes computed flags continuous, no VLM
6  train        train_gate_module.py           -> assets/modules_A/<name>/
7  evaluate     run_scripts/eval/eval_*_gated.sh -> output/<bench>/<run>/
8  judge        qgate tradeoff                  position against the free-trade line
```

Stage 3 is not optional. These jobs end on a `kill`, so they exit 0 whatever
happened, and a preempted shard that requeues can re-emit rows it already
wrote.

---

## Cluster rules that will bite you

- `--wckey=project-short-name:sub_fast` on every submission.
- `MODEL_OUTPUT_DIR` must be set **in the submitting environment** and start
  with `/rlwrld-unified-checkpoints/hojin2/`. A `#SBATCH --comment=` alone is
  rejected.
- Job names must exceed 50 characters. Short names are refused.
- `srun` works only on the `debug` partition. Training goes to `sjw_alinlab`,
  labelling and evaluation to `background` (which is preemptible, hence the
  resume requirements), CPU-only work to `cpu`.
- Judge servers bind `8900 + shard`; evaluation uses `10000+` and `20000+`
  offset by the array job id.
- Never run recursive `find`, `du` or `grep -r` on `/sjw_alinlab*`,
  `/rlwrld2` or the tile directories. `du` on the workspace root does not
  return in two minutes.

## Reading the state of things

```bash
bin/qgate paths                 where everything resolved, and what is missing
bin/qgate jobs                  what is queued or running
bin/qgate results <bench>       every evaluation run
bin/qgate labels <tag>          is a labelling run safe to train on
bin/qgate ckpt                  what weights exist
bin/qgate trace <bench> --episode N   one episode's labels as an HTML file
```

Add `--json` to any of them. `docs/STATUS.md` is the current per-benchmark
state; `docs/TOOLKIT.md` explains the measurements and their traps.
