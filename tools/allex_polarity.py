"""allex 라벨의 부호·채점표. robocasa `phase_scorecard.py` 의 allex 판.

정답지는 사용자가 준 naive 균등 배속 eval 이다 -- 셀 여섯, 셀마다 30 에피.
1x 기준선이 없으므로 유지율이 아니라 성공 수다. Pass 두 칸이 두 배속 모두
천장(30/30, 29/30)이라 실제로 갈리는 것은 Rotate Box 와 Bring PolyBag 둘뿐이고,
그래서 **상관 하나로 판정하지 않는다** -- 순서와 문항별 진단을 같이 낸다.

conf 는 저장된 값을 쓰지 않고 **gp(또는 eg)에서 현재 가중으로 다시 낸다.**
그래야 가중이 다른 실행끼리 비교된다(ablation 실행은 가중이 변 합 1 로 다시
정규화되어 저장돼 있다).

    python allex_polarity.py <records.jsonl> [더 많은 파일...]
"""
import collections
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import allex_v3_checks as C                                        # noqa: E402

BASE = os.path.dirname(HERE)
GT_PATH = f"{BASE}/analysis/allex_retention/naive_speedup_eval.json"
CELLS = ["Bring PolyBag", "Bring Box", "Rotate PolyBag", "Rotate Box",
         "Pass PolyBag", "Pass Box"]


def conf_of(vals, use=None):
    """등급(기댓값이면 더 좋다) -> conf. 가중은 현재 CHECKS 모듈이 정한다."""
    Q = [q for q in C.ACTIVE if use is None or q in use]
    g = {q: (float(p) - 1.0) / (C.NGRADE - 1)
         for q, p in zip(C.ACTIVE, vals) if q in Q}
    g = {q: w for q, w in g.items() if w > 0}

    def side(sg):
        w = {q: v for q, v in g.items() if C.SIGN[q] == sg}
        if not w:
            return 0.0
        return sum(C.WEIGHT[q] * v for q, v in w.items()) / sum(C.WEIGHT[q] for q in w)
    return float(min(1.0, max(0.0, (1.0 + side(+1) - side(-1)) / 2.0)))


def grades_of(r):
    """기댓값 등급을 쓴다. gp 가 있으면 거기서, 없으면 eg, 없으면 정수."""
    gp = r.get("gp")
    if gp and len(gp) == len(C.ACTIVE):
        return [sum((i + 1) * float(p) for i, p in enumerate(row)) for row in gp]
    eg = r.get("eg")
    if eg and len(eg) == len(C.ACTIVE):
        return list(eg)
    return [r.get(q) for q in C.ACTIVE]


def main():
    files = sys.argv[1:]
    if not files:
        raise SystemExit(__doc__)
    gt = json.load(open(GT_PATH))
    rs = []
    for f in files:
        for line in open(f):
            try:
                rs.append(json.loads(line))
            except Exception:
                continue
    rs = [r for r in rs if r.get("cell") in CELLS and all(
        r.get(q) is not None for q in C.ACTIVE)]
    if not rs:
        raise SystemExit("표준 여섯 칸 이름을 쓰는 행이 없다")
    src = " + ".join(os.path.basename(os.path.dirname(f)) or f for f in files)
    print(f"{src}: {len(rs):,}행 · 문항 {list(C.ACTIVE)}")
    ngp = sum(1 for r in rs if r.get("gp"))
    print(f"등급 분포(gp) 있는 행 {ngp:,} = {ngp/len(rs):.0%}"
          + ("  (eg 로 대체)" if not ngp else ""))

    G = np.array([grades_of(r) for r in rs], float)

    # --- 죽은 문항 (R7) ---
    print("\n문항별  평균  4등급이상  1등급   의도")
    for i, q in enumerate(C.ACTIVE):
        iv = np.array([r[q] for r in rs], float)
        flag = "  <- 죽었다" if (iv >= 4).mean() < 0.01 else ""
        print(f"  {q:6s} {G[:,i].mean():5.2f}  {(iv>=4).mean():8.1%} {(iv==1).mean():7.1%}"
              f"   {'감점' if C.SIGN[q]<0 else '가점'}{flag}")

    # --- 에피소드마다 conf, 대표 칸으로 묶는다 ---
    # 프레임을 그냥 평균하면 긴 칸이 무게를 더 갖는다. eval 단위는 에피소드다.
    def cell_means(use=None):
        byep = collections.defaultdict(list)
        for r, g in zip(rs, G):
            byep[(r.get("ep"), r.get("task"))].append((r["cell"], conf_of(g, use)))
        cc = collections.defaultdict(list)
        for _, v in byep.items():
            dom = collections.Counter(c for c, _ in v).most_common(1)[0][0]
            cc[dom].append(float(np.mean([x for _, x in v])))
        return cc

    cc = cell_means()
    have = [c for c in CELLS if cc.get(c)]
    print(f"\n{'칸':16s} {'에피':>4s} {'conf':>7s}   {'2.0x':>6s} {'2.5x':>6s}")
    for c in have:
        print(f"  {c:16s} {len(cc[c]):4d} {np.mean(cc[c]):7.4f}   "
              f"{gt['success']['2.0'][c]:3d}/30 {gt['success']['2.5'][c]:3d}/30")
    if len(have) < 4:
        raise SystemExit(f"\n칸이 {len(have)}개뿐이라 상관을 내지 않는다")

    m = np.array([np.mean(cc[c]) for c in have])
    print("\nconf 대 성공 (칸 %d개)" % len(have))
    for k in ("2.0", "2.5"):
        y = np.array([gt["success"][k][c] for c in have], float)
        rho, p = spearmanr(m, y)
        verdict = "부호 맞다" if rho > 0 else "**뒤집혔다**"
        print(f"  {k}x: 순위상관 {rho:+.3f} (p={p:.3f})  {verdict}")
    y25 = np.array([gt["success"]["2.5"][c] for c in have], float)
    print("\n  conf 낮은->높은: " + " < ".join(np.array(have)[np.argsort(m)]))
    print("  성공 낮은->높은: " + " < ".join(np.array(have)[np.argsort(y25)]))

    # --- 문항 하나를 빼면 나아지나 ---
    print("\n문항 하나 빼기 (2.5x 순위상관)")
    full = set(C.ACTIVE)
    base = spearmanr(m, y25)[0]
    print(f"  (전부)     {base:+.3f}")
    for q in C.ACTIVE:
        cc2 = cell_means(full - {q})
        m2 = np.array([np.mean(cc2[c]) for c in have])
        print(f"  {q:6s} 빼기 {spearmanr(m2, y25)[0]:+.3f}")
    print("\n천장에 붙은 칸이 넷이라 상관 하나로 판정하지 않는다. 순서와 "
          "문항별 진단을 같이 읽을 것.")


if __name__ == "__main__":
    main()
