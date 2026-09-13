"""libero v3c 라벨 jsonl -> 배포용 parquet. 게이트 세 갈래를 한 파일에서 다 낸다.

`phase9_v7_to_parquet.py`(robocasa) 와 같은 자리이고 같은 규율을 따른다. 다른 점은
접촉 라벨을 **외부 데이터셋에서 가져오지 않고 사이드카에서 직접 붙인다**는 것이다 --
`TTaekwan/libero_contact` 의 `flevel_v2/sidecar/episode_{i:06d}.parquet` 가 LeRobot
프레임 격자에 그대로 맞춰져 있어 stride 1 라벨과 1:1 로 붙는다. robocasa 판은
접촉이 창 단위로만 있어 VLM 라벨과 격자가 어긋났는데 여기는 어긋나지 않는다.

**한 파일에서 세 가지를 다 알 수 있게 만든다** (기존 배포 방식과 같다):

    접촉만       contact_quant == 1
    VLM 만       conf >= tau
    VLM + 접촉   conf >= tau  AND  contact_quant == 1

`conf` 는 기댓값 등급으로 낸다. 정수 등급만 남기면 conf 가 계단이 되어 역치가 안
먹고(tau 0.50/0.517/0.55 가 같은 설정이 된다), 나중에 가중을 바꿀 때 등급 분포가
이미 버려져 있어 정확히 재계산할 수 없다. 그래서 **eA..eE 를 같이 싣는다** -- 이
다섯 열이 있으면 어떤 가중으로도 conf 를 정확히 다시 낼 수 있다. 원본 분포(gp,
행당 25개 실수)는 크기 때문에 싣지 않는다.

    python libero_v3c_to_parquet.py <라벨디렉터리 또는 글로브> [출력.parquet]

라벨은 f in [0, len-4) 에만 있다(descriptors 가 f+4 를 본다). 에피 꼬리 4프레임은
행이 없다 -- 붙이는 쪽은 (episode_index, frame_index) 로 맞추면 된다.
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
      "libero_gr00t_delta")
CONTACT = f"{BASE}/output/_gate_distill/libero_contact/flevel_v2"
Q = list("ABCDE")


def expected(gp, ngrade):
    """등급 분포 -> 기댓값. sum (i+1) * p_i."""
    if not isinstance(gp, list) or len(gp) != len(Q):
        return None
    out = []
    for row in gp:
        if not row or len(row) != ngrade:
            return None
        out.append(sum((i + 1) * float(p) for i, p in enumerate(row)))
    return out


def load_contact(eps):
    """사이드카에서 (ep, f) -> quant/level/valid. 에피마다 길이를 확인한다."""
    L = {json.loads(l)["episode_index"]: json.loads(l)["length"]
         for l in open(f"{DS}/meta/episodes.jsonl")}
    rec, miss = [], 0
    for ep in sorted(eps):
        p = f"{CONTACT}/sidecar/episode_{ep:06d}.parquet"
        if not os.path.exists(p):
            miss += 1
            continue
        d = pd.read_parquet(p)
        fl = np.asarray(d["flevel"].to_list(), dtype=np.float32)
        # **길이를 확인한다.** 사이드카가 다른 판의 LeRobot 에서 만들어졌으면
        # 프레임이 한 칸씩 밀려 붙고, 그건 조용히 잘못된 라벨이 된다.
        if len(d) != L.get(ep, -1):
            raise SystemExit(
                f"ep{ep}: 사이드카 {len(d)}행 != LeRobot {L.get(ep)}행. "
                "접촉 데이터셋이 kimtaey/libero_gr00t_delta 와 같은 판이 아니다.")
        rec.append(pd.DataFrame({
            "episode_index": ep, "frame_index": np.arange(len(d), dtype=np.int32),
            "contact_quant": d["quant"].to_numpy(np.int8),
            "contact_level": fl[:, 3].astype(np.int8),
            "contact_valid": fl[:, 4].astype(np.int8)}))
    if miss:
        print(f"  [!] 사이드카 없는 에피 {miss}개")
    return pd.concat(rec, ignore_index=True) if rec else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("indir", help="라벨 디렉터리 또는 글로브(libero_v3c_s1_s16_*.jsonl)")
    ap.add_argument("out", nargs="?", default=None)
    ap.add_argument("--checks", default="libero_v3c_checks")
    ap.add_argument("--allow-missing-gp", action="store_true",
                    help="등급 분포가 없는 행을 정수 등급으로 처리한다. 기본은 거부 "
                         "-- gp 0%% 로 만들어진 라벨을 모르고 배포한 적이 있다")
    ap.add_argument("--allow-partial", action="store_true",
                    help="26.7만 행이 안 차도 진행한다(스모크용)")
    a = ap.parse_args()
    C = importlib.import_module(a.checks)
    out = a.out or f"{BASE}/output/_gate_distill/libero_{a.checks}.parquet"

    if os.path.isfile(a.indir):
        files = [a.indir]                      # 파일 하나를 그대로 넘길 수 있다
    else:
        pat = a.indir if "*" in a.indir else f"{a.indir}/libero_v3c*_s*_*.jsonl"
        files = sorted(f for f in glob.glob(pat) if not f.endswith("_meta.json"))
    if not files:
        raise SystemExit(f"라벨 파일이 없다: {pat}")
    rows = []
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
    if len(d) < 260000 and not a.allow_partial:
        raise SystemExit(
            f"{len(d):,}행뿐이다. stride 1 전량이면 266,693 근처여야 한다 -- "
            "라벨링이 안 끝났거나 매니페스트가 stride 4 판이었다. "
            "스모크로 보려면 --allow-partial.")

    # 진단 실행(`libero_gp_check.py`)은 A~E 대신 `picks` 리스트로 적는다. 전량
    # 라벨러(`libero_label_chunks.py`)는 A~E 로 적는다. 둘 다 받는다 -- 형식이
    # 다르다는 이유로 진단 라벨에서 parquet 을 못 만드는 것은 불편하기만 하다.
    if "A" not in d.columns and "picks" in d.columns:
        pk = d["picks"].to_list()
        for i, q in enumerate(Q):
            d[q] = [(r[i] if isinstance(r, (list, tuple)) and len(r) == len(Q)
                     else np.nan) for r in pk]
        print("  picks -> A~E 로 펼쳤다 (진단 형식)")

    # **등급이 정말 등급인지 확인한다.** libero 에는 A~E 가 YES 확률(0~1 실수)인
    # 옛 판이 있다(`libero_dense_*`, ans="NO,NO,NO,NO,NO" -- 강제된 슬롯 로짓을
    # 답으로 삼는 금지된 경로다, CLAUDE.md 되돌리지 말 것 1). 그 파일을 그대로
    # 넣으면 int8 로 깎여 A~E 가 전부 0 이 되고 conf 는 뜻 없는 값이 되는데
    # 행 수와 열은 정상이라 산출물을 세는 것으로는 구별되지 않는다.
    for q in Q:
        v = pd.to_numeric(d[q], errors="coerce")
        frac = float(((v == v.round()) & v.between(1, C.NGRADE)).mean())
        if frac < 0.99:
            raise SystemExit(
                f"{q} 열이 1~{C.NGRADE} 정수가 아니다 (정수 비율 {frac:.1%}, "
                f"범위 {v.min():.3g}~{v.max():.3g}). YES/NO 확률로 만들어진 옛 "
                f"라벨(libero_dense)을 넣은 것 같다 -- 등급 라벨이 필요하다.")

    ngp = int(d.get("gp", pd.Series([None] * len(d))).notna().sum())
    print(f"등급 분포(gp) 있는 행 {ngp:,} = {ngp/max(len(d),1):.1%}")
    if not ngp and not a.allow_missing_gp:
        raise SystemExit(
            "등급 분포가 한 행도 없다. 정수 등급으로 떨어지면 conf 가 계단이 되고 "
            "역치가 작동하지 않는다. 라벨러를 고쳐 다시 만들거나 --allow-missing-gp.")

    eg = [expected(g, C.NGRADE) for g in d.get("gp", [None] * len(d))]
    for i, q in enumerate(Q):
        d["e" + q] = [(r[i] if r else float(d[q].iloc[j])) for j, r in enumerate(eg)]

    # conf 는 기댓값 등급으로. 가중은 CHECKS 모듈이 정한다(R5: 라벨링 뒤에 정한다).
    g = {q: (d["e" + q].astype(float) - 1.0) / (C.NGRADE - 1) for q in Q}
    risk = sum(C.WEIGHT[q] * g[q] for q in Q if C.SIGN[q] < 0)
    safe = sum(C.WEIGHT[q] * g[q] for q in Q if C.SIGN[q] > 0)
    d["conf_vlm"] = ((1.0 + safe - risk) / 2.0).clip(0.0, 1.0)

    # 접촉 라벨
    ct = load_contact(set(d.episode_index.unique()))
    if ct is None:
        raise SystemExit(f"접촉 사이드카를 못 읽었다: {CONTACT}/sidecar")
    d = d.merge(ct, on=["episode_index", "frame_index"], how="left")
    nm = int(d.contact_quant.isna().sum())
    print(f"접촉 붙은 행 {len(d)-nm:,}/{len(d):,}" + (f" · 빈 행 {nm:,}" if nm else ""))
    for c in ["contact_quant", "contact_level", "contact_valid"]:
        d[c] = d[c].fillna(-1)

    # **신뢰도 열을 세 개 둔다.** robocasa 판(`prehj/robocasa-conf-labels-v7`)과
    # 같은 이름·같은 뜻이라 쓰는 쪽이 두 벤치마크를 같은 코드로 읽는다.
    #   conf_vlm      VLM 만
    #   conf_contact  접촉만 (전이면 0, 아니면 1) -- 0/1 이라 tau 와 무관하다
    #   conf_both     둘 다 (접촉 전이를 0 으로 덮는다)  <- 기본 권장
    d["contact_bnd"] = (d.contact_quant == 0).astype("int8")
    d["conf_contact"] = d.contact_quant.clip(lower=0).astype("float32")
    d["conf_both"] = d.conf_vlm.where(d.contact_quant == 1, 0.0)

    # suite / task / instruction
    em = {e["episode_index"]: e for e in
          json.load(open(f"{CONTACT}/_episode_map.json"))["episodes"]}
    instr = {}
    for line in open(f"{DS}/meta/episodes.jsonl"):
        e = json.loads(line)
        c = [t for t in e.get("tasks", []) if isinstance(t, str) and len(t.split()) > 1]
        instr[e["episode_index"]] = c[0] if c else ""
    d["suite"] = d.episode_index.map(lambda x: em.get(x, {}).get("suite", ""))
    d["task"] = d.episode_index.map(lambda x: em.get(x, {}).get("task", ""))
    d["instruction"] = d.episode_index.map(instr).fillna("")

    cols = (["episode_index", "frame_index"] + Q + ["e" + q for q in Q]
            + ["conf_vlm", "conf_contact", "conf_both", "contact_bnd",
               "contact_level", "contact_valid", "suite", "task", "instruction"])
    d = d[cols].sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
    for c in ["episode_index", "frame_index"]:
        d[c] = d[c].astype("int32")
    for c in Q + ["contact_bnd", "contact_level", "contact_valid"]:
        d[c] = d[c].fillna(0).astype("int8")
    for c in ["e" + q for q in Q] + ["conf_vlm", "conf_contact", "conf_both"]:
        d[c] = d[c].astype("float32")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    d.to_parquet(out, compression="zstd", index=False)

    print(f"\n{len(d):,}행 · {len(cols)}열 -> {out}"
          f"  ({os.path.getsize(out)/1e6:.1f} MB)")
    print(f"  conf_vlm 평균 {d.conf_vlm.mean():.4f} · 분위 "
          + "/".join(f"{d.conf_vlm.quantile(q):.3f}" for q in (.25, .5, .75)))
    print(f"  접촉 통과율 {(d.contact_bnd==0).mean():.1%} "
          f"(접촉 전이 {(d.contact_bnd==1).mean():.1%})")
    # 세 갈래가 정말 서로 다른 결정을 내는지 보여 준다. 같은 결정이면 열 하나가
    # 남는 것이고, 그건 만든 쪽이 알아야 한다.
    print("\n  역치별 압축 비율 (conf_vlm / conf_both):")
    for tau in (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60):
        print(f"    tau {tau:.2f}: {(d.conf_vlm>=tau).mean():6.1%} / "
              f"{(d.conf_both>=tau).mean():6.1%}")
    # **우연수준을 같이 낸다.** 일치율만 보면 해석할 수 없다 -- 두 게이트의 통과율이
    # 각각 정해져 있으면 무관해도 일치율이 저절로 46% 쯤 나온다. 초과분이 0 이면
    # 두 신호가 독립이라는 뜻이고, 그러면 합치는 것이 실제로 정보를 더한다.
    v, qq = d.conf_vlm >= 0.50, d.contact_bnd == 0
    ch = v.mean() * qq.mean() + (1 - v.mean()) * (1 - qq.mean())
    print(f"  tau 0.50 에서 VLM 과 접촉이 같은 결정: {(v==qq).mean():.1%} "
          f"(우연수준 {ch:.1%}, 초과 {(v==qq).mean()-ch:+.1%})")
    b = d.contact_level == 1
    if b.any():
        print(f"  접촉 전이(level 1) conf_vlm {d.conf_vlm[b].mean():.4f} · 그 밖 "
              f"{d.conf_vlm[~b].mean():.4f} (차 {d.conf_vlm[b].mean()-d.conf_vlm[~b].mean():+.4f}; "
              f"음수여야 -- 경계에서 VLM 도 압축을 꺼려야 한다)")


if __name__ == "__main__":
    main()
