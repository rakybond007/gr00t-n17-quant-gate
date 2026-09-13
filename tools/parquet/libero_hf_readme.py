"""배포 parquet 에서 HF 데이터셋 README 를 만든다.

숫자를 손으로 적지 않는 이유가 있다. robocasa 판 README 의 표는 손으로 옮긴
것이고, 라벨을 다시 만들면 표만 옛 값으로 남는다. 여기서는 **표를 전부 parquet
에서 계산**하므로 README 와 파일이 어긋날 수 없다.

    python libero_hf_readme.py <parquet> [출력 README.md]
"""
import os
import sys

import pandas as pd

Q = list("ABCDE")
TAUS = (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60)


def main():
    src = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        os.path.dirname(src) or ".", "README.md")
    d = pd.read_parquet(src)
    import importlib
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    C = importlib.import_module("libero_v3c_checks")

    n, ne, nt = len(d), d.episode_index.nunique(), d.task.nunique()
    bnd = float((d.contact_bnd == 1).mean())
    ladder = "\n".join(
        f"| {t:.2f} | {(d.conf_vlm>=t).mean():.0%} | {(d.conf_both>=t).mean():.0%} |"
        for t in TAUS)
    # 권장 tau: conf_both 압축 비율이 50% 에 가장 가까운 값
    tau = min(TAUS, key=lambda t: abs((d.conf_both >= t).mean() - 0.50))
    b = d.contact_bnd == 1
    gap = d.conf_vlm[b].mean() - d.conf_vlm[~b].mean()
    v, qq = d.conf_vlm >= tau, d.contact_bnd == 0
    ch = v.mean() * qq.mean() + (1 - v.mean()) * (1 - qq.mean())
    wl = " · ".join(f"{q} {C.WEIGHT[q]:+.3f}".replace("+", "−" if C.SIGN[q] < 0 else "+")
                    for q in Q)
    suite = "\n".join(
        f"| `{s}` | {g.episode_index.nunique()} | {(g.conf_both>=tau).mean():.0%} |"
        for s, g in d.groupby("suite"))

    txt = f"""---
license: apache-2.0
task_categories:
- robotics
tags:
- action-chunk-compression
- vla
- libero
---

# 압축 신뢰도 라벨 (conf labels) — LIBERO v3c

로봇 정책이 내는 액션 청크를 **이 시점에 압축해도 되는지** 적은 라벨입니다.
`LIBERO` 벤치마크, stride 1(전 프레임)로 **{n:,} 행 · {ne:,} 에피 · {nt} 태스크** 입니다.

`prehj/robocasa-conf-labels-v7` 과 **같은 열 이름·같은 뜻**이라 두 벤치마크를 같은
코드로 읽을 수 있습니다. 압축 여부는 쓰는 쪽이 역치로 정합니다.

```
압축한다  <=>  conf >= tau
```

## 게이트 세 갈래가 한 파일에 있습니다

| 열 | 게이트 | tau={tau:.2f} 에서 압축 |
|---|---|---|
| `conf_vlm` | VLM 만 | {(d.conf_vlm>=tau).mean():.1%} |
| `conf_contact` | 접촉만 (전이면 0, 아니면 1) | {(d.conf_contact>=tau).mean():.1%} |
| `conf_both` | VLM + 접촉 (전이를 0으로 덮음) | {(d.conf_both>=tau).mean():.1%} |

`conf_contact` 는 0/1 뿐이라 어떤 tau 에서도 같은 게이트입니다.

**robocasa 판과 다른 점이 하나 있습니다.** robocasa 에서는 VLM 신뢰도가 접촉 경계를
전혀 막지 못했습니다(경계 0.3565 대 비경계 0.3583, p=1.00). LIBERO v3c 는 **부분적으로
막습니다** — 접촉 전이에서 `conf_vlm` {d.conf_vlm[b].mean():.4f}, 그 밖
{d.conf_vlm[~b].mean():.4f} 로 {gap:+.4f} 차이가 있고, tau={tau:.2f} 에서 VLM 과 접촉의
결정이 {(v==qq).mean():.1%} 일치합니다(우연수준 {ch:.1%}, 초과 {(v==qq).mean()-ch:+.1%}).
문항 B 가 그리퍼가 닫히는 순간을 감점으로 잡기 때문입니다. 그래도 완전히 겹치지는
않으므로 **`conf_both` 가 기본 권장**입니다.

## 역치 고르는 법

`conf` 는 확률이 아니라 순서 값입니다. 원하는 압축 비율에서 자르십시오.

| tau | `conf_vlm` | `conf_both` |
|---|---|---|
{ladder}

수트별로는 `conf_both` tau={tau:.2f} 에서 이렇게 갈립니다.

| 수트 | 에피 | 압축 |
|---|---|---|
{suite}

## 열

| 열 | 뜻 |
|---|---|
| `conf_vlm` · `conf_contact` · `conf_both` | 위 세 게이트 |
| `A`~`E` | VLM 이 문항 다섯에 매긴 1~5 등급 (정수) |
| `eA`~`eE` | 같은 등급의 **분포 기댓값** Sum (i+1)*p(i). conf 는 이 값으로 계산됩니다 |
| `contact_bnd` | 접촉 전이 프레임이면 1 ({bnd:.1%}) |
| `contact_level` | 1 = 접촉 전이 · 2/3/4 = 이동량 하위/중간/상위 |
| `contact_valid` | 이동량 기준이 유효한 시점이면 1 (에피 꼬리에서 0) |
| `suite` · `task` · `instruction` | LIBERO 수트, 태스크 이름, 지시문 |

**`eA`~`eE` 가 들어 있는 이유.** conf 는 가중 다섯 개로 정해지고 가중은 쓰임에 따라
달라집니다. 정수 등급만 남기면 나중에 가중을 바꿀 때 분포가 이미 버려져 정확히
재계산할 수 없습니다. `eA`~`eE` 가 있으면 **어떤 가중으로도 conf 를 정확히 다시 낼 수
있습니다.** 이 라벨의 이전 판(YES/NO 확률로 만든 것)은 이 값이 없어서 역치가
0.45~0.50 사이에서 96.7% → 42.7% 로 튀었습니다.

```python
W = {{{", ".join(f'"{q}": {C.WEIGHT[q]:.4f}' for q in Q)}}}
SIGN = {{{", ".join(f'"{q}": {C.SIGN[q]:+d}' for q in Q)}}}
g = {{q: (d["e"+q] - 1) / 4 for q in "ABCDE"}}
conf = ((1 + sum(SIGN[q]*W[q]*g[q] for q in "ABCDE")) / 2).clip(0, 1)
conf_both = conf.where(d.contact_bnd == 0, 0.0)
```

## 문항 다섯

가중 {wl} (감점 A·B / 가점 C·D·E).

| | 뜻 | 부호 |
|---|---|---|
| A | 좁고 정해진 자리에 반듯이 놓는가 | 감점 |
| B | 집을 것이 얹혀 있거나 사이에 끼어 있는가 | 감점 |
| C | 받아 주는 통에 넣는가 | 가점 |
| D | 물체보다 훨씬 큰 면에 놓는가 | 가점 |
| E | 붙들 필요 없는 일인가 | 가점 |

문항 문구와 도출 과정은 `prompts/` 에 함께 넣었습니다.

## 접촉 라벨의 출처

`contact_*` 열은 `TTaekwan/libero_contact` 의 `flevel_v2` 를 프레임 격자에 맞춰 붙인
것입니다. MuJoCo 물리에서 읽은 오라클 라벨이라 **학습 target 으로만** 쓰고 배포 시점
입력으로는 읽지 않습니다.
"""
    open(out, "w").write(txt)
    print(f"README -> {out} ({len(txt):,}자) · 권장 tau {tau:.2f}")


if __name__ == "__main__":
    main()
