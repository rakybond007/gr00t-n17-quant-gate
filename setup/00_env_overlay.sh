#!/usr/bin/env bash
# n1.7 은 Qwen3-VL 백본이라 transformers>=4.57 이 필요하다.
# 기존 학습/평가 환경의 transformers 를 올리면 robocasa 스택이 깨지므로
# (numpy 1.23.x 요구 · gr00t import), 별도 디렉터리에 얹어 PYTHONPATH 로만 쓴다.
#
#   bash setup/00_env_overlay.sh /path/to/python
#   export PYTHONPATH=$PWD/pylibs/tf4573:$PYTHONPATH
set -euo pipefail
PY="${1:?사용법: 00_env_overlay.sh <기존 환경의 python 경로>}"
OUT="$(cd "$(dirname "$0")/.." && pwd)/pylibs/tf4573"
mkdir -p "$OUT"
"$PY" -m pip install --no-warn-script-location --target="$OUT" "transformers==4.57.3" lmdb
# numpy 를 오버레이에 남기면 환경의 numpy 를 가려 학습이 조용히 깨진다. 반드시 제거.
rm -rf "$OUT"/numpy "$OUT"/numpy-*.dist-info "$OUT"/numpy.libs "$OUT"/hf_xet "$OUT"/hf_xet-*.dist-info
echo "오버레이 완료: $OUT"
PYTHONPATH="$OUT" "$PY" -c "
import numpy, transformers
print('numpy', numpy.__version__, '(1.23.x 여야 함)')
print('transformers', transformers.__version__)
from transformers import Qwen3VLForConditionalGeneration; print('Qwen3VL OK')
"
