#!/usr/bin/env bash
# NVIDIA 공개 태그로 n1.7 트리를 만들고 게이트 패치를 적용한다.
# 업스트림 트리는 이 레포에 포함하지 않는다 — 공개 태그에서 재현되고,
# 포크로 끌고 오면 LFS 데모 데이터까지 딸려온다.
#
#   bash setup/01_apply_to_n17.sh ~/Isaac-GR00T-n17
set -euo pipefail
DST="${1:?사용법: 01_apply_to_n17.sh <n1.7 트리를 만들 경로>}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
if [ ! -d "$DST/.git" ]; then
  GIT_LFS_SKIP_SMUDGE=1 git clone --branch n1.7-release --depth 1 \
    https://github.com/NVIDIA/Isaac-GR00T.git "$DST"
fi
cp "$HERE/gate/quant_gate.py"        "$DST/gr00t/model/modules/"
cp "$HERE/gate/quant_gate_labels.py" "$DST/gr00t/data/dataset/"
git -C "$DST" apply --check "$HERE/gate/n17_gate.patch" \
  && git -C "$DST" apply "$HERE/gate/n17_gate.patch" \
  && echo "패치 적용 완료" \
  || echo "패치가 이미 적용돼 있거나 충돌 — git -C $DST diff 로 확인"
echo "n1.7 트리: $DST"
