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
  └─ hidden_states[14]  → image_mask 로 이미지 토큰만
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
| `gate/quant_gate.py` | 백본 탭 + `QuantGateHead` + 체크포인트에서 붙이는 로더 |
| `gate/quant_gate_labels.py` | 학습 배치에 게이트 라벨을 싣는 데이터셋 래핑 |
| `gate/robocasa_modality_config.py` | N1.7 레지스트리에 RoboCasa 등록 (자기 등록형) |
| `gate/n17_policy_client.py` | N1.7 서버와 통신하는 클라이언트 (gr00t 임포트 없음) |
| `gate/n17_gate.patch` | 위 조각들을 잇는 N1.7 트리 수정 일체 |
| `setup/02_env_extras.sh` | accelerate 오버레이 (의존성 없이) |
| `gate/smoke_n17_gate_train.py` | 배선 스모크 (forward·backward·그래디언트 범위) |
| `gate/robocasa_descriptors*.py` | 라벨 생성에 쓰인 결정론적 위험 기술자 |
| `labels/*.parquet` | VLM 교사 라벨 247,887 청크 |

## 준비

**1. 환경.** n1.7 은 transformers 4.57.3 이 필요한데, 기존 robocasa 학습/평가 환경의
transformers 를 올리면 스택이 깨진다(numpy 1.23.x 요구, gr00t import 실패).
별도 디렉터리에 얹어 `PYTHONPATH` 로만 쓴다.

```bash
bash setup/00_env_overlay.sh /path/to/env/bin/python
bash setup/02_env_extras.sh  /path/to/env/bin/python   # accelerate — 아래 참고
export PYTHONPATH=$PWD/pylibs/tf4573:$PYTHONPATH
```

transformers 만 얹으면 학습 루프 진입 직전에 `Accelerator.unwrap_model() got an
unexpected keyword argument 'keep_torch_compile'` 로 막힌다. accelerate 를 의존성까지
함께 설치하면 torch 2.13 · numpy 2 가 딸려와 더 크게 망가지므로 `--no-deps` 로만 넣는다.
`02_env_extras.sh` 가 그 처리와 사후 정리를 한다.

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

`GATE_LAYER` / `TUNE_TOP` 환경변수로 조합을 바꿔 확인할 수 있다.
통과 기준은 **게이트 탭 이하의 학습가능 레이어에 그래디언트가 잡히는 것**이다.

```
$ GATE_LAYER=14 TUNE_TOP=4 python gate/smoke_n17_gate_train.py
   그래디언트 있는 텐서 — 게이트헤드 13 / 상위층(>=12) 22 / 하위층 0 / 비전타워 0
```

`상위층 0` 이 나오면 게이트가 백본에 닿지 않는 것이다. 위의 탭 레이어 제약을 볼 것.

## 데이터

**변환이 필요 없다.** N1.7 의 `lerobot_episode_loader` 가 LeRobot 형식(`meta/` +
청크 parquet + mp4)을 그대로 읽는다. `sharded_*_dataset.py` 의 "sharding" 은 저장
포맷이 아니라 에피소드를 워커에 나누는 방식이고, `lmdb` 는 데이터 경로와 무관하다.

필요한 것은 임베디먼트 모달리티 설정 하나뿐이다 — `gate/robocasa_modality_config.py`.
`--modality-config-path` 로 넘기면 임포트되면서 스스로 등록한다.

두 가지 주의:

- **영상 백엔드.** N1.7 기본값은 `torchcodec` 이고 없으면 즉시 실패한다. `decord` 가
  있으면 `VIDEO_BACKEND=decord` 로 지정한다(패치가 이 환경변수를 읽는다).
- **`action_configs` 를 비워 둔다.** 비우면 전 키가 `ABSOLUTE / NON_EEF / DEFAULT` 로
  채워져 액션 값을 변환 없이 쓴다. RoboCasa 액션은 이미 컨트롤러가 소비하는 델타
  명령이라 이게 맞고, `RELATIVE` 로 두면 상태 기준 델타로 다시 계산해 이중 변환이 된다.

확인된 값: 1,965,457 스텝 / 1920 샤드, 액션 지평 16 × 12 차원, 인덱스 레이아웃
`0:4 base_motion · 4:5 control_mode · 5:8 EE 위치 · 8:11 EE 회전 · 11:12 그리퍼`.

## 평가

학습된 정책을 우리 RoboCasa 클라이언트로 평가하려면 두 세대의 환경이 충돌한다.
클라이언트는 numpy 1.23.5 / transformers 4.51.3 고정이 필요하고 N1.7 서버는 4.57 이
필요하다. 그래서 **한 잡 안에서 프로세스마다 환경을 다르게 준다.**

```
GPU0  N1.7 정책 서버   오버레이 ON  · 온라인 (아래 참고)
GPU1  판정기           오버레이 OFF · 오프라인
      클라이언트       오버레이 OFF · 오프라인 · gr00t 임포트 0
```

`gate/n17_policy_client.py` 가 N1.7 의 msgpack/ZeroMQ 프로토콜을 직접 말한다.
`zmq`, `msgpack`, `numpy` 만 쓰고 N1.7 패키지를 임포트하지 않으므로 클라이언트 환경이
그대로 유지된다. `--selftest` 로 직렬화 왕복을 서버 없이 검증할 수 있다.

**서버 프로세스만 온라인이어야 한다.** transformers 4.57 은 캐시가 있어도 토크나이저
패치 경로에서 HF API 를 조회한다. 판정기는 게이트드 리포 때문에 오프라인을 유지해야
하므로 둘을 분리한다.

## 평가 실행 절차

```bash
# 스모크 — 태스크 하나, 2 에피소드
export MODEL_OUTPUT_DIR=/rlwrld-unified-checkpoints/hojin2/quant_gate_modules
RUN=gate N_EPISODES=2 MAX_STEPS=300 SMOKE_TASK=OpenDrawer \
srun --gpus=2 --job-name=smoke_eval_n17_<설명> \
     --wckey=project-short-name:sub_fast \
     --exclude=worker-node100,worker-node1,worker-node104,worker-node3 \
  bash run_scripts/eval/eval_robocasa_n17_gated.sh

# 본평가 — 24 태스크 × 50 에피소드
sbatch --export=ALL,RUN=gate,TAU=0.5,N_EPISODES=50,OUTPUT_BASE=...,\
MODEL_OUTPUT_DIR=$MODEL_OUTPUT_DIR \
  --job-name=eval_robocasa_n17_<설명> run_scripts/eval/eval_robocasa_n17_gated.sh
```

`RUN=gate|baseline` 이 체크포인트를 고른다. **판정은 잡 상태가 아니라 산출물로 한다.**

```bash
for t in $OUTPUT_BASE/*/; do echo "$(basename $t): $(grep -c '^episode' $t/prediction.txt)"; done
grep "Server is ready" $OUTPUT_BASE/server-*.log
```

에피소드 줄이 하나도 없으면 클라이언트가 돌지 않은 것이다. 어레이 인덱스가 24 이상이면
태스크 매핑이 전부 탈락해 **서버만 뜨고 정상 종료**하므로, 런처가 그 경우를 거부하도록
해 두었다. 스모크에는 `SMOKE_TASK` 로 태스크 하나를 직접 지정하는 편이 안전하다.

`prediction.txt` 를 읽을 때는 대괄호 안의 공백을 허용해야 한다 — numpy 가 `[ True]` 로
쓰기 때문에, 공백을 빼먹으면 성공한 에피소드만 조용히 사라진다.

```python
re.match(r"episode\s+(\d+)\s+is_success:\s*\[\s*(True|False)\s*\]\s*action_steps:\s*(\d+)", line)
```

## 학습 스모크

본학습 전에 2 스텝만 돌려 배선을 확인한다.

```bash
export PYTHONPATH=$PWD/pylibs/tf4573:$HOME/Isaac-GR00T-n17
export HF_HUB_OFFLINE=0 VIDEO_BACKEND=decord TUNE_TOP_LLM_LAYERS=4
export GATE_LABELS=$PWD/labels/v6b_phase5_1call_full.parquet GATE_LAYER=14
srun --gpus=1 --job-name=smoke_n17_finetune_<설명> ... \
  python gr00t/experiment/launch_finetune.py \
    --base-model-path nvidia/GR00T-N1.7-3B --dataset-path <데이터셋> \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path <이 레포>/gate/robocasa_modality_config.py \
    --num-gpus 1 --output-dir <출력> --max-steps 2 --global-batch-size 2
```

통과 기준은 로그가 아니라 **체크포인트에 게이트 가중치가 들어갔는지**다.

```bash
python -c "
import json; idx=json.load(open('<출력>/model.safetensors.index.json'))['weight_map']
print('게이트 텐서', len([k for k in idx if 'quant_gate' in k]))"
```

13 개가 나와야 한다. 0 이면 게이트가 붙지 않은 채 학습된 것이고, 로그로는 알 수 없다.

## 내부 게이트 노출

추론 경로는 `attach_quant_gate` 를 부르지 않는다. 그래서 학습된 게이트 가중치가
체크포인트에 있어도 **모듈이 없어 조용히 버려지고**, 게이트 없는 정책으로 평가하면서
게이트를 쟀다고 착각하게 된다. 패치가 세 곳을 잇는다.

```
모델   get_action 에서 게이트를 돌려 gate_confidence 산출 (백본 forward 재사용, 추가 연산 0)
정책   그 값을 info 딕셔너리로 반환 (기존엔 빈 dict)
로더   attach_from_checkpoint() 가 quant_gate.* 가중치를 찾아 붙이고 개수를 출력
```

## 학습 배선

```python
from gr00t.data.dataset.quant_gate_labels import GateLabelLookup, patch_dataset_gate_labels

# 학습 범위를 명시적으로 지정한다 — 체크포인트 설정에 맡기면 안 된다(아래 참고)
model.backbone.set_trainable_parameters(tune_llm=False, tune_visual=False,
                                        tune_top_llm_layers=4)
model.backbone.to(torch.bfloat16)          # 동결 뒤 dtype 되돌리기 (아래 참고)

# 액션 탭은 건드리지 않는다 — 16 은 검증된 설정이다
model.attach_quant_gate(gate_layer=14, loss_weight=1.0)

lookup = GateLabelLookup("labels/v6b_phase5_1call_full.parquet")
patch_dataset_gate_labels(dataset, lookup)
```

라벨은 `get_datapoint` 에서 실린다. 콜레이터가 샘플 딕셔너리의 모든 키를 스택하므로
**셔딩된 데이터셋을 다시 굽지 않아도** 모델 `forward` 까지 전달된다.

교사 라벨이 없는 스텝은 `gate_valid=0` 으로 손실에서 **제외**한다. 0.5 를 타겟으로 주면
모델이 "모르는 구간은 애매하다"를 학습해버린다.

손실은 `loss = action_loss + λ · gate_loss` 이고, `action_loss_only` 와 `gate_loss` 가
출력에 함께 담기므로 둘을 따로 볼 수 있다.

### 체크포인트 설정을 믿으면 안 된다

배포된 `nvidia/GR00T-N1.7-3B` 의 config 는 `tune_llm=True`, `tune_visual=True` 다.
그냥 로드하면 **1.5B 백본 전체가, 비전 타워까지 학습가능**이 된다.

| | `tune_llm` | `tune_visual` | `tune_top_llm_layers` |
|---|---|---|---|
| n1.7 코드 기본값 | False | False | 0 |
| **배포 체크포인트 config** | **True** | **True** | 0 |

부작용이 하나 더 있다. 로드 시 학습가능한 파라미터는 `trainable_params_fp32` 로 fp32 캐스팅되는데,
그 상태에서는 FlashAttention 이 `only support fp16 and bf16` 로 죽는다. 그래서 동결을 적용한 뒤
`backbone.to(torch.bfloat16)` 로 되돌려야 한다.

### 권장 구성

```
액션 탭 16   (기본값, 건드리지 않는다 — 검증된 설정이다)
게이트 탭 14
tune_top_llm_layers = 4  → 레이어 12~15 학습

  레이어 0~11  동결
  레이어 12·13 게이트 그래디언트가 형성 (액션도 통과하므로 공유)
  레이어 14·15 액션 전용
  비전 타워    동결
```

게이트를 액션보다 **위**에 두는 것도 코드는 지원한다(`action_layer` 인자).
다만 트렁크 깊이는 액션 정보를 담지 않는다 — 액션은 별도 DiT 헤드가 만든다.
게이트가 액션을 고려하게 하려면 **1스텝 디노이징 결과를 헤드 입력으로** 넣어야 하고,
그 자리는 `QuantGateHead(act_dim=...)` 로 열려 있다.

### 탭 레이어 제약 — 여기서 두 번 틀렸다

`hidden_states[k]` 는 **k 번째 레이어의 출력**이다(`[0]` 은 임베딩). 그래서 게이트 손실은
레이어 `0..k-1` 로만 흐른다. 액션 손실은 최상위를 읽으므로 학습 가능한 전 구간으로 흐른다.

**게이트 탭이 동결 구간에 있으면 게이트는 백본을 전혀 바꾸지 못한다.**
그런데 학습은 정상으로 보이고 손실도 내려간다 — 게이트 헤드만 학습되기 때문이다.
레이어 특화라는 이 접근의 핵심만 조용히 사라지고, 폐루프에서 "효과 없음"이라는
틀린 결론이 나온다. 실제로 두 번 당했다: 탭 10(동결 구간)에서 한 번,
탭 12(오프바이원으로 여전히 레이어 11 이하)에서 또 한 번.

```
필요 조건:  gate_layer - 1 >= (첫 학습가능 레이어)
16층 · tune_top_llm_layers=4 (레이어 12~15 학습) 이면 gate_layer >= 13
권장 조합:  gate_layer=14  →  레이어 12·13 은 게이트가 형성, 14·15 는 액션 전용
```

`attach_quant_gate` 가 이 조건을 검사하고 어긋나면 즉시 에러를 낸다.
통과하면 어느 레이어가 게이트에 형성되고 어느 레이어가 액션 전용인지 출력한다.

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
