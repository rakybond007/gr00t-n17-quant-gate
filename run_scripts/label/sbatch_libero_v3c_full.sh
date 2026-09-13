#!/bin/bash
#SBATCH --wckey=project-short-name:sub_fast
#SBATCH --job-name=label_libero_v3c_questions_stride1_all_frames_with_grade_probs_sixteen_shards
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=background
#SBATCH --time=3:00:00
#SBATCH --requeue
#SBATCH --exclude=worker-node100,worker-node1,worker-node104,worker-node3
#SBATCH --output=out/%A_%a-libv3c.out
#SBATCH --error=out/%A_%a-libv3c.err
#SBATCH --comment="LIBERO v3c questions, stride 1 (266,693 frames), grade probabilities stored. Resumable via per-shard output dedup."
set -u
# **허브를 안 본다.** 같은 노드의 여러 샤드가 동시에 모델을 해석하다 깨진 적이 있다
# (robocasa 배열에서 여섯 태스크가 20분 대기 뒤 죽었다). 캐시가 완전하므로 안전하다.
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_XET=1
export MODEL_OUTPUT_DIR=/rlwrld-unified-checkpoints/hojin2/gate_modules/phase9_labels
JUDGE_PY="$HOME/quantization_agent_workspace/cosmos_judge_venv/bin/python"
RUN_PY="/sjw_alinlab/home/hojin2/miniconda3/envs/quant_gate_eval/bin/python"
cd "$HOME/quantization_agent_workspace/vlm_gate" || exit 1
: "${SLURM_ARRAY_TASK_ID:=0}"
NSH=$((SLURM_ARRAY_TASK_MAX+1))

export TAG=libero_v3c_s1
export PROMPT_VER=v3c
export CHECKS=libero_v3c_checks
export TILES="$PWD/output/_gate_distill/libero_full/tiles"
export MANIFEST="$PWD/output/_gate_distill/libero_tiles_manifest.txt"
export LIBERO_BATCH=8

# **포트는 배열 잡 번호와 태스크 번호로 정한다.** 빈 포트를 스캔하면 같은 노드의 두
# 태스크가 동시에 스캔해 같은 포트를 고르고 뒤에 바인드하는 쪽이 Errno 98 로 죽는다.
PORT=$((20000 + (SLURM_ARRAY_JOB_ID % 60) * 16 + SLURM_ARRAY_TASK_ID))
LOG="output/_gate_distill/${TAG}_judge_${SLURM_ARRAY_TASK_ID}.log"
OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 "$JUDGE_PY" -u scripts/vlm_gate_cosmos.py \
    --serve --port "$PORT" > "$LOG" 2>&1 &
J=$!
trap 'kill $J 2>/dev/null' EXIT
for _ in $(seq 1 120); do grep -q "JUDGE READY" "$LOG" && break; sleep 10; done
grep -q "JUDGE READY" "$LOG" || { echo "judge 실패"; tail -20 "$LOG"; exit 1; }
echo "shard $SLURM_ARRAY_TASK_ID/$NSH · port $PORT · CHECKS=$CHECKS · 문항 $PROMPT_VER"

"$RUN_PY" -u scripts/libero_label_chunks.py "$PORT" "$SLURM_ARRAY_TASK_ID" "$NSH"

# **산출물로 판정한다.** 라벨러는 배치 예외를 삼키고 계속 돌다가 정상 종료한다 --
# judge_batch 가 mode 인자를 안 받던 판에서 16샤드가 전부 0행을 쓰고 COMPLETED 로
# 끝났다. 잡 상태만 보면 성공과 구별되지 않으므로 행 수를 세어 실패로 만든다.
OUTF="output/_gate_distill/${TAG}_s${NSH}_${SLURM_ARRAY_TASK_ID}.jsonl"
ROWS=$(wc -l < "$OUTF" 2>/dev/null || echo 0)
GPROWS=$(grep -c '"gp"' "$OUTF" 2>/dev/null || echo 0)
echo "샤드 $SLURM_ARRAY_TASK_ID: $ROWS 행 (gp $GPROWS)"
if [ "$ROWS" -lt 100 ]; then
    echo "실패: $ROWS 행뿐이다. 판정기 로그 끝:"; tail -5 "$LOG"; exit 1
fi
if [ "$GPROWS" -lt $((ROWS / 2)) ]; then
    echo "실패: 등급분포가 $GPROWS/$ROWS 행에만 있다 -- conf 가 계단이 된다"; exit 1
fi
