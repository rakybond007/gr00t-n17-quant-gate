# GR00T-N1.7 quantization-confidence gate

행동 청크(16스텝)마다 **이 구간을 절반 속도로 실행해도 되는가**를 판단하는 게이트를
GR00T-N1.7 위에 붙이고, 액션 헤드와 **함께 파인튜닝**한다.

되는 구간만 K2로 압축하면 성공률을 지키면서 스텝을 줄인다. RoboCasa Kitchen 기준
OpenDrawer를 82% 압축했을 때 성공률 0.68 / 성공시 178스텝이 나왔다 —
무압축 0.72 / 229스텝, 무지성 K2 0.44 / 136스텝. **얼마나가 아니라 어디를 압축하냐**가
성능을 가른다.

배경과 이 설계에 이르기까지의 측정 근거는 [docs/CONTEXT.md](docs/CONTEXT.md)에 있다.

## 이 레포가 하는 일

게이트가 백본의 **다른 레이어**를 읽고 액션 헤드와 함께 학습된다.

```
Qwen3-VL 백본 (Cosmos-Reason2-2B, 상위 N층 unfreeze)
  ├─ hidden_states[16]  → vlln → self-attn → 액션 헤드 (DiT)   ← 기존 경로
  └─ hidden_states[10]  → image_mask 로 이미지 토큰만
                        → attention 풀링 → 게이트 헤드 → P(quantize)
```

두 손실이 서로 다른 깊이에 걸리므로 unfreeze 된 상위 레이어가 각 목적에 특화된다.
백본 forward 는 정책이 어차피 수행하므로 **추가 비전 연산은 0이고 헤드만 늘어난다.**

이미지 토큰만 보는 이유는 앞선 실패에서 나왔다. 전체 토큰 평균 풀링은 텍스트까지 섞어
공간·객체 정보를 잃고, 그 구조로 만든 게이트가 독립 모듈보다 성능이 낮았다.
n1.7 백본은 `image_mask` 를 내보내므로 골라낼 수 있다.

## 구성

| 경로 | 내용 |
|---|---|
| `gate/quant_gate.py` | 백본 탭 + `QuantGateHead` (이미지 토큰 attention 풀링) |
| `gate/quant_gate_labels.py` | 학습 배치에 게이트 라벨을 싣는 데이터셋 래핑 |
| `gate/gr00t_n1d7_gate.patch` | `Gr00tN1d7` 에 `attach_quant_gate` + 결합 손실 |
| `gate/smoke_n17_gate_train.py` | 배선 스모크 (forward·backward·그래디언트 범위) |
| `gate/robocasa_descriptors*.py` | 라벨 생성에 쓰인 결정론적 위험 기술자 |
| `labels/*.parquet` | VLM 교사 라벨 247,887 청크 |

## 준비

**1. 환경.** n1.7 은 transformers 4.57.3 이 필요한데, 기존 robocasa 학습/평가 환경의
transformers 를 올리면 스택이 깨진다(numpy 1.23.x 요구, gr00t import 실패).
별도 디렉터리에 얹어 `PYTHONPATH` 로만 쓴다.

```bash
bash setup/00_env_overlay.sh /path/to/env/bin/python
export PYTHONPATH=$PWD/pylibs/tf4573:$PYTHONPATH
```

오버레이에 numpy 가 남으면 환경의 numpy 를 가려 **학습이 조용히 깨진다.**
스크립트가 지우지만, 직접 설치했다면 반드시 확인할 것.

**2. n1.7 트리.** 업스트림은 공개 태그에서 받고 패치를 얹는다.

```bash
bash setup/01_apply_to_n17.sh ~/Isaac-GR00T-n17
export PYTHONPATH=~/Isaac-GR00T-n17:$PYTHONPATH
```

**3. 가중치.** 둘 다 HuggingFace 승인이 필요하다.

- `nvidia/GR00T-N1.7-3B`
- `nvidia/Cosmos-Reason2-2B` — 백본. 체크포인트에 포함돼 있지 않으므로 **따로 받아야 한다.**
  승인 없이는 `Access to model ... is restricted` 로 막힌다.

**4. 데이터.** RoboCasa Kitchen LeRobot 데이터셋. 라벨은 `(episode_index, frame_index)`
로 조인되므로 라벨을 만든 것과 **같은 데이터셋**이어야 한다.

## 확인

```bash
python gate/smoke_n17_gate_train.py
```

네 가지를 본다 — 게이트 부착, 중간 레이어 탭과 `image_mask` 정합, `gate_loss` 역전파,
그리고 **그래디언트가 의도한 범위에만 흐르는지**. 동결한 비전 타워에 그래디언트가
잡히면 동결이 새고 있다는 뜻이고, 그대로 학습하면 정책을 망가뜨린다.

## 학습 배선

```python
from gr00t.data.dataset.quant_gate_labels import GateLabelLookup, patch_dataset_gate_labels

model.backbone.set_trainable_parameters(False, False, tune_top_llm_layers=4)
model.attach_quant_gate(gate_layer=10, loss_weight=1.0)

lookup = GateLabelLookup("labels/v6b_phase5_1call_full.parquet")
patch_dataset_gate_labels(dataset, lookup)
```

라벨은 `get_datapoint` 에서 실린다. 콜레이터가 샘플 딕셔너리의 모든 키를 스택하므로
**셔딩된 데이터셋을 다시 굽지 않아도** 모델 `forward` 까지 전달된다.

교사 라벨이 없는 스텝은 `gate_valid=0` 으로 손실에서 **제외**한다. 0.5 를 타겟으로 주면
모델이 "모르는 구간은 애매하다"를 학습해버린다.

손실은 `loss = action_loss + λ · gate_loss` 이고, `action_loss_only` 와 `gate_loss` 가
출력에 함께 담기므로 둘을 따로 볼 수 있다.

## 비교 기준이 달라진다는 점

이 프로젝트의 기존 전제는 "정책은 고정하고 추론 시점에만 개입한다"였다.
공동 파인튜닝은 정책 자체를 바꾸므로 **게이트 없는 n1.7 RoboCasa 베이스라인을 따로 뽑아야**
개선폭을 귀속할 수 있다. N1.5 기반 결과와는 직접 비교되지 않는다.

## 라벨

`labels/v6b_phase5_1call_full.parquet` — 247,887 청크. 컬럼은
`episode_index, frame_index, task, p_yes, p_raw, quantize, c_*(계산 플래그 4), q_*(VLM 문항 4)`.

`p_yes` 가 학습 타겟이다(순위정규화 값이라 τ 를 차단율로 바로 해석할 수 있다).
`v6b_phase5_soft.parquet` 은 계산 플래그를 연속값으로 바꾼 변형 두 가지를 함께 담고 있다
(`p_yes_soft`, `p_yes_softB`) — 이유는 CONTEXT 참고.
