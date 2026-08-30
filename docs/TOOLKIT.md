# `qgate` — 노트북 없이 결과 읽기

결과를 **만드는** 쪽은 아무것도 바뀌지 않았다. `vlm_gate/scripts/` 의 스크립트 135개와
`vlm_gate/run_scripts/` 의 잡 스크립트 89개는 그대로 제출되고 그대로 돈다.
없었던 것은 그 결과를 **읽는** 길이다 — 224개 중 어느 파일인지 기억하지 않고,
어느 디렉터리에 있든 상관없이.

`bin/qgate` 가 그 읽기 도구다. 어느 작업 디렉터리에서도 돌고, 텍스트 명령은 환경
활성화가 필요 없으며, 요청하면 JSON 을 낸다. ssh 한 줄이면 충분하다.

    ssh <host> ~/quantization_agent_workspace/bin/qgate results robocasa

읽기만 한다. 잡이 돌고 있는 벤치마크에 대고 실행해도 안전하고, 아직 쓰이는 중인
런은 그냥 에피소드 수가 적게 나온다.

## 명령

| 명령 | 답하는 질문 |
|---|---|
| `qgate paths` | 이 머신에서 각 경로가 어디로 잡혔나 |
| `qgate jobs` | 무엇이 큐에 있고 무엇이 돌고 있나 |
| `qgate results <bench>` | 그 벤치마크의 모든 평가 런, 성공률 순 |
| `qgate run <bench> <run>` | 한 런을 태스크별로 |
| `qgate compare <bench> <a> <b>` | 두 런의 태스크별 차이 |
| `qgate tradeoff <bench> --fast … --slow …` | 속도/성공 거래를 실제로 이긴 런은 무엇인가 |
| `qgate actions <bench> <dataset>` | 액션 벡터에 뭐가 들었나, 병합이 어떻게 되나 |
| `qgate ckpt` | 디스크의 체크포인트와 크기 |

어디에나 `--json` 을 붙일 수 있다.

## 설명이 필요한 두 측정

**게이트의 순위를 매기는 건 `tradeoff` 다.** 게이트는 로봇을 빠르게 만들면서 조금
나쁘게 만든다. 그래서 성공률만으로는 순위가 거꾸로 나온다 — 아무것도 압축하지 않는
게이트가 1등이 되고, 그건 쓸모가 없다. 무압축과 무지성 압축은 (스텝, 성공률) 평면의
두 점이고, 그 사이 직선이 **공짜로 얻을 수 있는 거래**다. `excess` 는 런이 그 선보다
얼마나 위에 있는지다. RoboCasa 의 앵커는 214스텝 0.598 과 327스텝 0.657,
기울기는 스텝당 0.00052 다.

    $ qgate tradeoff robocasa --fast baseline_compress_K2 \
        --slow baseline_full_v2_with_action_steps --runs phase5_dinov3s_tau0p5 …
    restricted to the 23 tasks every run finished (50+ episodes)

    run                          steps  success  on line   excess   saved
    phase5_dinov3s_tau0p5          276    0.642    0.629  +0.0128    16%
    phase5_softA_module_tau0p5     267    0.635    0.624  +0.0105    19%
    phase5_module_tau0p5           275    0.638    0.628  +0.0100    17%
    phase5_softB_module_tau0p5     252    0.627    0.617  +0.0099    24%
    excluded as incomplete: TurnOnSinkFaucet

성공률만 보면 softB 가 꼴찌다. excess 로 보면 넷이 사실상 붙어 있고
(+0.0099~+0.0128, 1,150 에피소드 성공률의 표준오차가 약 0.014), softB 는 같은 값을
**에피소드의 24%를 지우고** 얻는다 — 나머지는 16~19% 다. 갈리는 건 품질이 아니라
같은 품질에서의 속도다.

`--episodes` 기본값 50 이 하는 일이 여기서 결정적이다. 이걸 끄고(`--episodes 0`)
각 런의 전체 집계를 쓰면 `TurnOnSinkFaucet` 이 섞여 들어오는데, 두 런에서 그 태스크는
50개가 아니라 2개와 5개만 끝나 있다. 예전에 softB 가 +0.0111 로 확실히 앞선다고
표에 적었던 것도 같은 종류의 실수였다 — 23태스크 런 값을 24태스크 앵커에 대고 쟀다.
양쪽을 같은 태스크 집합에 올리면 그 격차는 사라진다.

스텝 수는 **성공한 에피소드만의 평균**이다. 실패한 에피소드는 환경 제한 시간까지
돌기 때문에, 실패를 포함하면 정책이 아니라 타임아웃을 평균내게 된다. 강하게 압축된
런을 읽을 때 하나 유념할 것: 살아남은 에피소드는 다른, 대체로 더 쉬운 부분집합이라
스텝 평균이 엄밀하게 비교되지 않는다. dexjoco 에서 K3 이 K2 보다 스텝이 **많게**
나오는 것(326 대 318)이 순전히 그 선택 효과다.

**`compare` 는 두 런이 모두 끝낸 태스크만 본다.** 기본값 `--episodes 50` 은 양쪽 모두
50 에피소드 이상인 태스크만 센다. 이건 미관 문제가 아니다. 50개 중 5개만 끝난 태스크의
성공률은 0.2 단위로 양자화돼 있고, 그걸 끝난 태스크와 비교하면 없는 차이를 만들어낸다.
예전에 RoboCasa 수도꼭지 태스크를 0.60 → 0.20 퇴행으로 읽었던 것이 정확히 이 경우였다 —
5개 중 1개를 50개 중 30개와 비교한 것이었다. 제외된 태스크는 항상 같이 출력된다.

## `actions` — 임베디먼트가 연산을 결정한다

K스텝 압축이 인접 스텝을 **더할지** 중간 것을 **건너뛸지**는 선택이 아니라 데이터의
성질이고, `qgate actions` 가 그걸 판정하는 측정이다.

    $ qgate actions dexjoco assets/datasets/dexjoco_lerobot/.../click_mouse
    single step over +-1.0: 0.9995   summed over k=2: 1.0000
    Single steps already sit outside +-1.0, so these are absolute targets …

RoboCasa 와 LIBERO 는 엔드이펙터 델타다. 단일 스텝은 컨트롤러의 ±1 을 절대 벗어나지
않는데 더한 쌍은 벗어난다(LIBERO 병합의 7.9%). 그 초과분은 로봇이 끝내 가지 않는
변위다. dexjoco 와 allex 는 절대 관절 목표라 범위 검사 자체가 무의미하고, 압축은
중간 목표를 버리는 방식이어야 한다.

`--sweep` 은 그 벤치마크의 결정론적 기술자 모듈을 샘플 청크에 돌려 각 위험 플래그가
얼마나 자주 켜지는지 보여준다. 새 기술자를 믿기 전에 돌려볼 값어치가 있는 보정
단계다. 한 번도 안 켜지는 플래그는 정보가 없고, 임계값을 다시 잡아도 살아나지 않는다.
LIBERO 의 `reversal` 이 여기서 걸렸다 — 궤적이 매우 매끄러워(인접 스텝 코사인 중앙값
0.998) 방향 반전이 스텝의 0.02% 에서만 일어난다 — 그래서 데이터 자체의 5퍼센타일에서
자른 연속값 `turn` 으로 교체했다.

dexjoco 에 같은 스윕을 돌리면 이걸 왜 벤치마크 단위가 아니라 **태스크 단위로** 봐야
하는지가 드러난다. `reversal` 은 여섯 태스크 중 다섯에서 죽은 것처럼 보이지만
(청크의 0.0~0.2%), `hammer_nail` 에서는 10% 에서 켜진다. 망치질이 곧 반전이기
때문이다. 고장난 플래그가 아니라 태스크 특화 검출기이고, `hammer_nail` 은 K2 압축이
가장 크게 망가뜨리는 태스크 축에 든다. 벤치마크 전체 평균을 냈으면 양쪽 다 가려졌다.

## 다른 곳에서 돌리기

`qgate paths` 가 무엇이 어디로 잡혔는지, 없는 게 무엇인지 찍어준다. 워크스페이스
루트는 위로 올라가며 `vlm_gate/` 를 찾아 잡으므로 다른 위치에 클론해도 수정이 필요
없다. 코드와 결과가 다른 마운트에 있을 때는 개별 루트를 덮어쓰면 된다.

    QGATE_OUTPUT=/mnt/scratch/output qgate results libero

`QGATE_WS`, `QGATE_VLM_GATE`, `QGATE_OUTPUT`, `QGATE_ANALYSIS`, `QGATE_DOCS`,
`QGATE_ASSETS`, `QGATE_CKPT_ROOT`, `QGATE_PYTHON` 을 모두 받는다. 실제 환경이 필요한
건 `qgate actions` 하나뿐이고(parquet 을 pandas 로 읽는다), `bin/qgate` 가 통상 위치의
`quant_gate` 환경을 알아서 집는다.
