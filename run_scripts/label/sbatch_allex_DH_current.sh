#!/bin/bash
#SBATCH --wckey=project-short-name:sub_fast
#SBATCH --job-name=label_allex_balanced_dev_and_untouched_holdout_with_current_five_checks
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=background
#SBATCH --time=2:00:00
#SBATCH --requeue
#SBATCH --output=out/%j-allexDH.out
#SBATCH --error=out/%j-allexDH.err
#SBATCH --comment="allex D(750, 125/cell) + H(750, never labelled) with the current 5 checks, to settle whether conf is inverted."
set -u
# **왜 D 와 H 를 둘 다 도나.** 부호가 뒤집힌 것(-0.829)을 잡아낸 표본이 D 다
# (6층 x 125, 층별 검정력이 같게 뽑힌 집합). 그 실행의 가이던스는 되찾을 수
# 없다 -- allex 가이던스는 checks 파일에 박혀 있고 그 파일은 판올림됐다.
# 그래서 **지금 프롬프트로 D 를 다시 재고, 한 번도 안 쓴 H 로 확인한다.**
# H 를 같이 도는 것이 핵심이다: D 에서 나온 부호를 D 로 다시 확인하면 같은
# 표본을 두 번 쓰는 것이고, 그건 확인이 아니다.
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_XET=1
export MODEL_OUTPUT_DIR=/rlwrld-unified-checkpoints/hojin2/gate_modules/phase9_labels
export ALLEX_DS=/rlwrld2/home/david/action_quantization/v1/subtask_labeled_data_update_eef_256x256_hojin
export ALLEX_CHECKS="CLAMP,LOOSE,SHOVE,FLIP,FREE"
JUDGE_PY="$HOME/quantization_agent_workspace/cosmos_judge_venv/bin/python"
RUN_PY="/sjw_alinlab/home/hojin2/miniconda3/envs/quant_gate_eval/bin/python"
cd "$HOME/quantization_agent_workspace/vlm_gate" || exit 1

PORT=$((23000 + SLURM_JOB_ID % 900))
LOG="output/_gate_distill/allex_DH_judge_${SLURM_JOB_ID}.log"
OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 "$JUDGE_PY" -u scripts/vlm_gate_cosmos.py \
    --serve --port "$PORT" > "$LOG" 2>&1 &
J=$!
trap 'kill $J 2>/dev/null' EXIT
for _ in $(seq 1 120); do grep -q "JUDGE READY" "$LOG" && break; sleep 10; done
grep -q "JUDGE READY" "$LOG" || { echo "판정기 안 뜸"; tail -20 "$LOG"; exit 1; }

rc=0
for SET in D H; do
    export ALLEX_OUT="$PWD/output/allex_${SET}_current"
    mkdir -p "$ALLEX_OUT"
    echo "=== $SET 집합 라벨링 -> $ALLEX_OUT ==="
    "$RUN_PY" -u scripts/allex_v3_label.py "$PORT" \
        "$PWD/output/allex_sample/${SET}.json" || rc=1
    # **산출물로 판정한다.** 라벨러가 판정 실패를 삼키고 정상 종료한 적이 있다
    # (libero 16샤드가 0행으로 COMPLETED). 행 수와 등급 분포를 세어 실패로 만든다.
    F="$ALLEX_OUT/records.jsonl"
    N=$(wc -l < "$F" 2>/dev/null || echo 0)
    G=$(grep -c '"gp"' "$F" 2>/dev/null || echo 0)
    echo "$SET: $N 행 (gp $G) · 목표 750"
    [ "$N" -lt 700 ] && { echo "실패: $SET 이 $N 행뿐이다"; tail -5 "$LOG"; rc=1; }
    [ "$G" -lt $((N / 2)) ] && { echo "실패: $SET 등급분포가 $G/$N 행"; rc=1; }

    # **게이트 판정과 부호를 바로 낸다.** 사람이 따로 돌려야 하면 결과가 로그에만
    # 남고 판정이 미뤄진다. 158163 이 자기 부호 게이트에서 떨어져 있었는데도
    # 그 라벨로 분석이 진행된 것이 그렇게 생긴 일이다.
    echo "--- [$SET] 게이트 판정 ---"
    ALLEX_OUT="$ALLEX_OUT" "$RUN_PY" -u scripts/allex_v3_verify.py 2>&1 | tail -25
    echo "--- [$SET] 실측 유지율 대비 부호 ---"
    "$RUN_PY" -u scripts/allex_polarity.py "$F" 2>&1 | tail -22
done
echo "=== D 와 H 를 맞대어 ==="
"$RUN_PY" -u scripts/allex_polarity.py \
    output/allex_D_current/records.jsonl output/allex_H_current/records.jsonl 2>&1 | tail -14
exit $rc
