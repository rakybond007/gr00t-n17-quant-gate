#!/bin/bash
#SBATCH --wckey=project-short-name:sub_fast
#SBATCH --job-name=merge_libero_shard_manifests_and_verify_stride1_coverage_before_labelling
#SBATCH -p cpu
#SBATCH --cpus-per-task=2
#SBATCH --time=0:20:00
#SBATCH --output=/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate/out/%j-libmerge.out
#SBATCH --error=/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate/out/%j-libmerge.err
set -u
# **매니페스트를 합치는 단계가 따로 있다.** 타일 생성 샤드는 각자
# manifest_shard<n>.json 만 쓰고, 라벨러가 읽는 평문 매니페스트는 이 merge 가
# 만든다. 이걸 안 돌리면 라벨러는 **예전 stride 4 매니페스트를 그대로 읽어**
# 26.7만 프레임 중 6.7만만 라벨하고 정상 종료한다 -- 개수를 안 세면 성공과
# 구별되지 않는 실패다.
cd "$HOME/quantization_agent_workspace" || exit 1
PY=$HOME/miniconda3/envs/quant_gate/bin/python
$PY -u vlm_gate/scripts/gen_libero_tiles_shard.py merge 16 || exit 1

# 합친 결과가 정말 stride 1 인지 **매니페스트 자체로** 확인한다. 줄 수만 보면
# 샤드 하나가 예전 stride 4 JSON 을 남겨 섞여도 통과할 수 있다. stride 4 판은
# f%4!=0 인 이름이 하나도 없으므로 그 비율이 가장 날카로운 판별이다.
$PY - <<'PYEOF' || exit 1
import sys
M = ("/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate/"
     "output/_gate_distill/libero_tiles_manifest.txt")
names = open(M).read().split()
off = sum(1 for n in names if int(n.split("_f")[1].split(".")[0]) % 4)
eps = len({n[2:6] for n in names})
print(f"매니페스트 {len(names):,}개 · 에피 {eps} · f%4!=0 비율 {off/max(len(names),1):.1%}")
bad = []
if len(names) < 260000:
    bad.append(f"타일이 {len(names):,}개뿐이다 (stride 1 이면 27.3만 근처)")
if off / max(len(names), 1) < 0.5:
    bad.append(f"f%4!=0 이 {off/max(len(names),1):.1%} -- stride 4 매니페스트가 섞였다")
if eps < 1680:
    bad.append(f"에피가 {eps}개뿐이다 (1693 이어야)")
if bad:
    print("검증 실패:"); [print("  -", b) for b in bad]
    sys.exit(1)
print("검증 통과 -- 라벨링을 시작해도 된다")
PYEOF
echo "매니페스트 준비 완료"
