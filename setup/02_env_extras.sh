#!/usr/bin/env bash
# 오버레이에 추가로 필요한 것들.
#
# transformers 4.57.3 만 얹으면 학습 루프 진입 직전에 accelerate 버전 때문에 막힌다
# (Accelerator.unwrap_model 에 keep_torch_compile 인자가 없다는 오류).
# 그런데 accelerate 를 의존성까지 함께 설치하면 torch 2.13 · numpy 2 · huggingface_hub 1.x
# 가 딸려와 오히려 환경을 부순다. 반드시 --no-deps 로 넣고, 딸려온 것이 있으면 지운다.
set -euo pipefail
OUT="$(cd "$(dirname "$0")/.." && pwd)/pylibs/tf4573"
PY="${1:?사용법: 02_env_extras.sh <기존 환경의 python 경로>}"
[ -d "$OUT" ] || { echo "먼저 00_env_overlay.sh 를 실행하세요"; exit 1; }
"$PY" -m pip install --no-warn-script-location --no-deps --target="$OUT" \
  "accelerate==1.6.0" "huggingface_hub==0.36.2"
# 환경의 numpy/torch 를 가리면 학습이 조용히 깨진다
rm -rf "$OUT"/numpy "$OUT"/numpy-*.dist-info "$OUT"/numpy.libs \
       "$OUT"/torch "$OUT"/torch-*.dist-info "$OUT"/nvidia "$OUT"/triton* \
       "$OUT"/sympy* "$OUT"/networkx* "$OUT"/hf_xet*
PYTHONPATH="$OUT" "$PY" - <<'PYCODE'
import inspect, numpy, torch, accelerate, transformers
from accelerate import Accelerator
print("numpy", numpy.__version__, "(1.23.x 여야 함)")
print("torch", torch.__version__, "(환경 것이 유지돼야 함)")
print("accelerate", accelerate.__version__)
print("transformers", transformers.__version__)
print("keep_torch_compile 지원:",
      "keep_torch_compile" in inspect.signature(Accelerator.unwrap_model).parameters)
PYCODE
