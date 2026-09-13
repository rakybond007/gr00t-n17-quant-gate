"""phase9 v7 라벨 jsonl -> 배포용 parquet. 기존 라벨 데이터셋과 같은 스키마로.

기존 배포 파일(`robocasa_contact_ratio_instr.parquet`)의 열을 그대로 따른다:

    episode_index int32 · frame_index int32 · A..E int8 · fixed int8
    ratio float32 · task object · instruction object · conf float32

여기에 **기댓값 등급 eA..eE(float32) 다섯 열을 더한다.** 이유가 있다. conf 는
가중 다섯 개로 정해지는데(conf = (1 + Sw*가점 - Sw*감점)/2, g=(등급-1)/4), 가중은
쓰임에 따라 달라진다. 정수 등급만 남기면 나중에 가중을 바꿀 때 등급 분포가 이미
버려져 있어 정확히 재계산할 수 없다 -- 뽑힌 등급의 확률이 평균 0.513, 60% 가 0.5
미만이라 정수와 기댓값의 차이가 등급 하나당 평균 0.55 칸이다. eA..eE 를 남기면
**어떤 가중으로도 conf 를 정확히 다시 낼 수 있다.** 원본 분포(gp, 행당 25개 실수)는
크기 때문에 싣지 않는다.

    python phase9_v7_to_parquet.py <라벨디렉터리> [출력.parquet] [--stride-expand]

`--stride-expand` 를 주면 stride 4 로 라벨된 행을 전 프레임으로 선형보간해 펼친다.
학습 파이프라인이 stride 를 안 쓰는 경우를 위한 것이다. 실측 복원 성능(에피 800개
22.7만 프레임): conf 상관 0.966 · 이진 재현율 95.1%(1 비율 50%) / 91.4%(25%) /
82.1%(10%). 펼치지 않은 원본이 기본이다 -- 나중에 stride 8 로 걸러 쓰려면
frame_index % 8 == 0 으로 고르면 된다(8 이 4 의 배수).
"""
import argparse
import glob
import importlib
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
BASE = os.path.dirname(HERE)
DS = ("/sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/kimtaey/"
      "robocasa_mg_gr00t_300")
# 접촉 경계 플래그는 외부 접촉 데이터셋을 병합해 만든 것이라 액션에서 다시 낼 수
# 없다. 기존 배포 파일에서 (ep, f) 로 가져온다 -- stride 4 위치는 전부 거기 있다.
PREV = f"{BASE}/output/_gate_distill/robocasa_contact_ratio_instr.parquet"
Q = list("ABCDE")


def expected(gp, ngrade=5):
    """등급 분포 -> 기댓값. sum (i+1) * p_i."""
    if not gp or len(gp) != len(Q):
        return None
    out = []
    for row in gp:
        if not row or len(row) != ngrade:
            return None
        out.append(sum((i + 1) * float(p) for i, p in enumerate(row)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("indir")
    ap.add_argument("out", nargs="?", default=None)
    ap.add_argument("--checks", default="phase9_checks_v7")
    ap.add_argument("--stride-expand", action="store_true")
    ap.add_argument("--allow-missing-gp", action="store_true",
                    help="등급 분포가 없는 행을 정수 등급으로 처리한다. 기본은 거부 "
                         "-- gp 0%% 로 만들어진 라벨을 모르고 배포한 적이 있다")
    a = ap.parse_args()
    C = importlib.import_module(a.checks)
    out = a.out or f"{BASE}/output/_gate_distill/robocasa_phase9_{a.checks}.parquet"

    rows = []
    files = sorted(glob.glob(f"{a.indir}/labels_*.jsonl"))
    if not files:
        raise SystemExit(f"라벨 파일이 없다: {a.indir}/labels_*.jsonl")
    for f in files:
        for line in open(f):
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    print(f"라벨 파일 {len(files)}개 · {len(rows):,}행")
    d = pd.DataFrame(rows).drop_duplicates(subset=["ep", "f"], keep="last")
    d = d.rename(columns={"ep": "episode_index", "f": "frame_index"})
    print(f"중복 제거 후 {len(d):,}행 · 에피 {d.episode_index.nunique()}")

    ngp = int(d.get("gp", pd.Series([None] * len(d))).notna().sum())
    print(f"등급 분포(gp) 있는 행 {ngp:,} = {ngp/max(len(d),1):.1%}")
    if not ngp and not a.allow_missing_gp:
        raise SystemExit(
            "등급 분포가 한 행도 없다. 정수 등급으로 떨어지면 conf 가 계단이 되고 "
            "역치가 작동하지 않는다(tau=0.50/0.517/0.55 가 같은 설정이 된다). "
            "라벨러를 고쳐 다시 만들거나, 일부러 그럴 때만 --allow-missing-gp.")

    # 기댓값 등급. 없으면 정수 등급으로 떨어진다.
    eg = [expected(g) if isinstance(g, list) else None for g in d.get("gp", [None] * len(d))]
    for i, q in enumerate(Q):
        d["e" + q] = [(r[i] if r else float(d[q].iloc[j])) for j, r in enumerate(eg)]

    # conf 는 기댓값 등급으로. 가중은 CHECKS 모듈이 정한다.
    g = {q: (d["e" + q].astype(float) - 1.0) / (C.NGRADE - 1) for q in Q}
    risk = sum(C.WEIGHT[q] * g[q] for q in Q if C.SIGN[q] < 0)
    safe = sum(C.WEIGHT[q] * g[q] for q in Q if C.SIGN[q] > 0)
    d["conf"] = ((1.0 + safe - risk) / 2.0).clip(0.0, 1.0)

    # task / instruction / fixed 를 붙인다
    instr, task = {}, {}
    for line in open(f"{DS}/meta/episodes.jsonl"):
        e = json.loads(line)
        c = [t for t in e.get("tasks", []) if isinstance(t, str) and len(t.split()) > 1]
        instr[e["episode_index"]] = c[0] if c else ""
    prev = pd.read_parquet(PREV, columns=["episode_index", "frame_index", "task", "fixed"])
    d = d.merge(prev, on=["episode_index", "frame_index"], how="left")
    miss = int(d.task.isna().sum())
    if miss:
        print(f"  [!] 기존 파일에 없는 (ep,f) {miss:,}행 -- task/fixed 가 빈다")
    d["instruction"] = d.episode_index.map(instr).fillna("")
    d["fixed"] = d.fixed.fillna(0)
    d["ratio"] = np.float32(np.nan)     # 배속은 이 판에서 쓰지 않는다. 열은 남긴다

    if a.stride_expand:
        d = expand(d)

    cols = (["episode_index", "frame_index"] + Q + ["e" + q for q in Q]
            + ["fixed", "ratio", "task", "instruction", "conf"])
    d = d[cols].sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
    for c in ["episode_index", "frame_index"]:
        d[c] = d[c].astype("int32")
    for c in Q + ["fixed"]:
        d[c] = d[c].fillna(0).astype("int8")
    for c in ["e" + q for q in Q] + ["ratio", "conf"]:
        d[c] = d[c].astype("float32")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    d.to_parquet(out, compression="zstd", index=False)
    print(f"\n{len(d):,}행 · {len(cols)}열 -> {out}"
          f"  ({os.path.getsize(out)/1e6:.1f} MB)")
    print("  " + " · ".join(f"{c}:{d[c].dtype}" for c in d.columns))
    print(f"  conf 평균 {d.conf.mean():.4f} · 분위 "
          + "/".join(f"{d.conf.quantile(q):.3f}" for q in (.25, .5, .75)))


def expand(d):
    """stride 로 라벨된 행을 전 프레임으로 선형보간해 펼친다."""
    tot = {}
    for line in open(f"{DS}/meta/episodes.jsonl"):
        e = json.loads(line)
        tot[e["episode_index"]] = max(0, e["length"] - 4)
    out = []
    num = ["conf"] + ["e" + q for q in Q]
    for ep, g in d.groupby("episode_index"):
        n = tot.get(ep, 0)
        if n <= 0:
            continue
        g = g.sort_values("frame_index")
        f = g.frame_index.to_numpy()
        full = np.arange(n)
        r = pd.DataFrame({"episode_index": ep, "frame_index": full})
        for c in num:
            r[c] = np.interp(full, f, g[c].to_numpy(float))
        # 정수 등급과 이산 열은 가장 가까운 라벨 프레임에서 가져온다
        j = np.abs(full[:, None] - f[None, :]).argmin(1)
        for c in Q + ["fixed", "task", "instruction", "ratio"]:
            r[c] = g[c].to_numpy()[j]
        out.append(r)
    e = pd.concat(out, ignore_index=True)
    print(f"  펼침: {len(d):,} -> {len(e):,}행 (선형보간)")
    return e


if __name__ == "__main__":
    main()
