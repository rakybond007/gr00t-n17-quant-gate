#!/bin/bash
#SBATCH --wckey=project-short-name:sub_fast
#SBATCH --job-name=label_robocasa_phase9_v7_questions_stride4_with_grade_probs_background_array
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=background
#SBATCH --time=3:00:00
#SBATCH --requeue
#SBATCH --exclude=worker-node100,worker-node1,worker-node104,worker-node3
#SBATCH --output=out/%A_%a-p9v7.out
#SBATCH --error=out/%A_%a-p9v7.err
#SBATCH --comment="robocasa phase9 v7 questions, stride 4 (511k frames), grade probabilities stored. Short walltime + 1 GPU per array task so the backfill scheduler can slot them in; resumable so preemption costs only the chunk in flight."
set -u
# background 를 쓰는 이유: TRESBillingWeights GPU=0.2 로 다른 파티션의 1/5 이고 노드가
# 25개로 가장 많다. PriorityTier=1 이라 선점당하지만 worklist 로 재개하므로 진행 중
# 청크만 잃는다. walltime 3시간은 backfill 용이다 -- 대기 잡의 136개가 2일을 요청한다.
# **허브를 안 본다.** 같은 노드의 여러 샤드가 동시에 모델을 해석하다가 깨졌다 --
# "does not appear to have a file named transformer/diffusion_pytorch_model-..." 로
# 여섯 태스크가 20분 대기 뒤 죽었다. 캐시는 33G 로 완전하므로 오프라인이면 안전하다.
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_XET=1
export MODEL_OUTPUT_DIR=/rlwrld-unified-checkpoints/hojin2/gate_modules/phase9_labels
JUDGE_PY="$HOME/quantization_agent_workspace/cosmos_judge_venv/bin/python"
RUN_PY="/sjw_alinlab/home/hojin2/miniconda3/envs/quant_gate_eval/bin/python"
cd "$HOME/quantization_agent_workspace/vlm_gate" || exit 1

export PHASE9_OUT="$PWD/output/_gate_distill/phase9_v7_stride4"
export PHASE9_BATCH=8
export PHASE9_STRIDE=4
export CHECKS=phase9_checks_v7
NW=16
: "${SLURM_ARRAY_TASK_ID:=0}"
mkdir -p "$PHASE9_OUT" out

# 배열 0번만 worklist 를 만들고 나머지는 기다린다. 동시에 쓰면 서로 덮어쓴다.
if [ "$SLURM_ARRAY_TASK_ID" = "0" ] && [ ! -f "$PHASE9_OUT/.worklist_ready" ]; then
  # 이미 있으면 다시 만들지 않는다 -- 0번이 선점 뒤 재큐되면 돌고 있는 샤드들과
  # 다른 분배가 만들어져 같은 프레임을 두 번 굽는다
  PHASE9_STRIDE=4 "$RUN_PY" -u scripts/phase9_worklist.py "$NW" "$PHASE9_OUT" || exit 1
  touch "$PHASE9_OUT/.worklist_ready"
else
  for _ in $(seq 1 180); do [ -f "$PHASE9_OUT/.worklist_ready" ] && break; sleep 10; done
  [ -f "$PHASE9_OUT/.worklist_ready" ] || { echo "worklist 가 안 생겼다"; exit 1; }
fi

# **포트는 배열 번호로 정한다.** 빈 포트를 스캔하면 같은 노드의 두 태스크가 동시에
# 스캔해 같은 포트를 고르고 뒤에 바인드하는 쪽이 Errno 98 로 죽는다(실제로 발생).
# 배열 번호는 유일하므로 이 방식은 배열 안에서 충돌이 없다.
# **배열 번호만으로는 부족하다.** 두 배열이 겹쳐 돌면 같은 포트를 쓴다 -- 옛 배열이
# 포트를 스캔하던 판과 새 배열의 고정 포트가 충돌해 Errno 98 로 죽었다. 배열 잡 번호를
# 섞어 배열끼리도 갈리게 한다.
PORT=$((18000 + (SLURM_ARRAY_JOB_ID % 60) * 16 + SLURM_ARRAY_TASK_ID))
LOG="$PHASE9_OUT/judge_${SLURM_ARRAY_TASK_ID}.log"
OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 "$JUDGE_PY" -u scripts/vlm_gate_cosmos.py \
    --serve --port "$PORT" > "$LOG" 2>&1 &
J=$!
trap 'kill $J 2>/dev/null' EXIT
for _ in $(seq 1 120); do grep -q "JUDGE READY" "$LOG" && break; sleep 10; done
grep -q "JUDGE READY" "$LOG" || { echo "judge 실패"; tail -20 "$LOG"; exit 1; }
echo "shard $SLURM_ARRAY_TASK_ID/$NW · port $PORT · CHECKS=$CHECKS · stride $PHASE9_STRIDE"

PHASE9_WORKLIST="$PHASE9_OUT/worklist_${NW}_${SLURM_ARRAY_TASK_ID}.json" \
  "$RUN_PY" -u scripts/phase9_label_full.py "$PORT" "$SLURM_ARRAY_TASK_ID" "$NW"
