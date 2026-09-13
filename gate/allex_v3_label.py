"""Label allex with the v3 checks -- one stage, graded, answered in text.

v2 ran two stages and read teacher-forced YES/NO slots. Both are gone. There is
one question set now (allex_v3_checks), the model WRITES its five digits, and
what comes out is already a ratio -- ceiling_from_checks returns K, so no
confidence, no tau ladder, no rank normalisation.

Nothing is written to the parquet. This writes records.jsonl only, for the
renderer to read.

The output goes to output/allex_v3checks, NOT output/allex_v3: "v3" was already
taken by the second allex capture, whose labels live there under a different
schema and a different episode numbering. Writing here first resumed off that
file -- 586 chunks of episodes 0-3 were skipped as "already done" when their
(ep, f) meant something else entirely.

    python allex_v3_label.py <port> [episodes,comma,separated]
"""
import hashlib
import json
import os
import sys

import av
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from allex_v2_common import TASKS, descriptors  # noqa: E402
from allex_v3_checks import (ACTIVE, ASK, GUIDANCE, NGRADE, SIGN, WEIGHT,  # noqa: E402
                             confidence, expected_grades,
                             facts_v3, ratio_for, snap)
from vlm_gate import VLMGate  # noqa: E402

DS = os.environ.get(
    "ALLEX_DS",
    "/rlwrld2/home/david/action_quantization/v1/subtask_labeled_data_update_eef_256x256_hojin")
OUTDIR = os.path.expanduser(os.environ.get(
    "ALLEX_OUT", "~/quantization_agent_workspace/vlm_gate/output/allex_v3checks"))
os.makedirs(OUTDIR, exist_ok=True)
CHUNK = 16
BATCH = int(os.environ.get("ALLEX_BATCH", 8))

FULL = os.environ.get("ALLEX_FULL", "") == "1"
STRIDE = int(os.environ.get("ALLEX_STRIDE", 1))
SHARD = int(os.environ.get("ALLEX_SHARD", 0))
NSH = int(os.environ.get("ALLEX_NSHARDS", 1))

PORT = sys.argv[1]
# 표본 파일(allex_v3_sample.py 가 만든 D.json/H.json)이면 그 (ep, f) 만 돈다.
# 층 이름도 같이 읽어 기록에 넣는다 -- 검증이 쓰고, 모델에는 안 간다.
SAMPLE = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2].endswith(".json") else ""
if SAMPLE:
    _wl = json.load(open(SAMPLE))
    CELL = {(int(e), int(f)): lab for lab, v in _wl["strata"].items() for e, f in v}
    WANT = {}
    for (e, f) in CELL:
        WANT.setdefault(e, []).append(f)
    EPS = sorted(WANT)
elif FULL:
    CELL, WANT = {}, {}
    EPS = []
    for _l in open(f"{DS}/meta/episodes.jsonl"):
        _e = json.loads(_l)["episode_index"]
        if _e % NSH == SHARD:
            EPS.append(_e)
    EPS.sort()
else:
    CELL, WANT = {}, {}
    EPS = [int(e) for e in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0]
# ---- 데이터셋 전체를 도는 길 -------------------------------------------
# 개발집합은 표본 파일로 돌지만 배달은 전체를 돈다. 이 길에서만 필요한 것 셋:
# 에피소드가 1000 개를 넘으면 chunk-001 이 있고, 칸은 주기에서 뽑아야 하고,
# stride 로 건너뛰며 본다.
try:
    CHUNKS_SIZE = int(json.load(open(f"{DS}/meta/info.json"))["chunks_size"])
except Exception:
    CHUNKS_SIZE = 1000
SEGS = {}


def cdir(ep):
    return f"chunk-{ep // CHUNKS_SIZE:03d}"


if FULL:
    # 주기가 물체를 말해 준다: 가져오기 -> (뒤집기) -> 넘기기. 그 주기에 낀
    # 회전이 그 주기의 물체다. 회전이 없는 주기는 물체를 모르므로 뺀다 --
    # 칸을 모르면 상한을 못 준다.
    _OBJ = {"Rotate Box": "Box", "Rotate PolyBag": "PolyBag"}
    _by = {}
    for _l in open(f"{DS}/meta/subtasks.jsonl"):
        _r = json.loads(_l)
        _by.setdefault(_r["episode_index"], []).append(_r)
    _cell_of_seg, _unknown = {}, 0
    for _ep, _v in _by.items():
        _v.sort(key=lambda z: z["start_frame"])
        for _x in _v:
            if _x["label"] in _OBJ:
                _cell_of_seg[_x["id"]] = f"Rotate {_OBJ[_x['label']]}"
        _i = 0
        while _i < len(_v):
            if _v[_i]["label"] != "Bring Object":
                _i += 1
                continue
            _j, _mid = _i + 1, []
            while _j < len(_v) and _v[_j]["label"] not in ("Bring Object", "Pass Object"):
                _mid.append(_v[_j]["label"])
                _j += 1
            if _j < len(_v) and _v[_j]["label"] == "Pass Object":
                _o = {_OBJ[_m] for _m in _mid if _m in _OBJ}
                if len(_o) == 1:
                    _oo = _o.pop()
                    _cell_of_seg[_v[_i]["id"]] = f"Bring {_oo}"
                    _cell_of_seg[_v[_j]["id"]] = f"Pass {_oo}"
                else:
                    _unknown += 2
                _i = _j + 1
            else:
                _unknown += 1
                _i = _j
        # 칸을 못 정한 구간은 서브태스크 이름을 그대로 칸으로 쓴다. 그 이름의
        # 띠는 TASK_RANGE 에 있고, Bring 은 박스·봉투 중 낮은 쪽으로 잡혀 있다.
        SEGS[_ep] = [(_x["start_frame"], _x["end_frame"],
                      _cell_of_seg.get(_x["id"]) or _x["label"], _x["label"])
                     for _x in _v]
    print(f"  구간 {sum(len(v) for v in SEGS.values())}개, 칸을 못 정한 구간 {_unknown}개",
          flush=True)


def seg_at(ep, f):
    """청크 한가운데가 든 구간의 (칸, 서브태스크 이름, 구간 번호)."""
    mid = f + CHUNK // 2
    for k, (a, b, cell, lab) in enumerate(SEGS.get(ep, ())):
        if a <= mid < b:
            return cell, lab, k
    return None, None, None


OUT = (f"{OUTDIR}/records_s{NSH}_{SHARD}.jsonl" if FULL and NSH > 1
       else f"{OUTDIR}/records.jsonl")

# **이 실행이 무슨 문항·부호·가중을 썼는지 남긴다.** 남기지 않으면 나중에
# records.jsonl 만 보고는 알 수 없다. 실제로 그 상황이 생겨서, 저장된 conf 에
# 최소제곱을 걸어 가중을 되찾아야 했다 -- eg 를 저장해 둔 덕에 가능했지만
# 21개 실행을 일일이 역산하는 비용을 치렀다. ablation(LOOSE 빼기, SHOVE 를
# 감점으로 돌리기 등)을 돌리면 가중이 변 합 1 로 다시 정규화되므로, 어떤
# 조합이었는지는 파일에 적혀 있어야 한다.
json.dump({"active": list(ACTIVE), "ngrade": NGRADE,
           "sign": {q: SIGN[q] for q in ACTIVE},
           "weight": {q: WEIGHT[q] for q in ACTIVE},
           "guidance_sha1": hashlib.sha1(GUIDANCE.encode()).hexdigest()[:12],
           "ask_sha1": hashlib.sha1(ASK.encode()).hexdigest()[:12]},
          open(OUT.replace(".jsonl", "_meta.json"), "w"),
          ensure_ascii=False, indent=1)

gate = VLMGate(f"http://127.0.0.1:{PORT}", timeout=300)

done = set()
if os.path.exists(OUT):
    for line in open(OUT):
        try:
            r = json.loads(line)
            done.add((r["ep"], r["f"]))
        except Exception:
            pass


def grab(ep, frames, side):
    """Decode the requested frame indices from one ego camera."""
    want, got = set(frames), {}
    path = (f"{DS}/videos/{cdir(ep)}/observation.images.camera_ego_{side}/"
            f"episode_{ep:06d}.mp4")
    with av.open(path) as c:
        for i, fr in enumerate(c.decode(video=0)):
            if i in want:
                got[i] = Image.fromarray(fr.to_ndarray(format="rgb24"))
                if len(got) == len(want):
                    break
    return got


out = open(OUT, "a")
ntot = nempty = 0
for ep in EPS:
    d = pd.read_parquet(f"{DS}/data/{cdir(ep)}/episode_{ep:06d}.parquet")
    A = np.stack(d["action"].values)
    WR = np.stack(d["action.right_wrist_wrt_base"].values)
    WL = np.stack(d["action.left_wrist_wrt_base"].values)
    ti = d["task_index"].values
    if SAMPLE:
        starts = sorted(f for f in WANT.get(ep, []) if (ep, f) not in done)
    elif FULL:
        # stride 를 구간 **안에서** 센다. 에피소드 전체에 대고 세면 짧은 구간이
        # 통째로 빠진다. 이러면 구간마다 첫 청크는 반드시 라벨된다.
        starts, cnt = [], {}
        for f in range(0, len(A) - CHUNK, CHUNK):
            cell, lab, si = seg_at(ep, f)
            if cell is None:
                continue                      # 칸을 모르면 상한을 못 준다
            c = cnt.get(si, 0)
            cnt[si] = c + 1
            if c % STRIDE == 0 and (ep, f) not in done:
                starts.append(f)
                CELL[(ep, f)] = cell
    else:
        starts = [f for f in range(0, len(A) - CHUNK, CHUNK) if (ep, f) not in done]
    if not starts:
        print(f"ep{ep}: 이미 함", flush=True)
        continue
    L, R = grab(ep, starts, "left"), grab(ep, starts, "right")
    for b0 in range(0, len(starts), BATCH):
        grp = starts[b0:b0 + BATCH]
        payload, meta = [], []
        for f in grp:
            x = descriptors(A, WR, WL, f, CHUNK)
            seg = ti[f:f + CHUNK]
            task = (seg_at(ep, f)[1] if FULL else None) or \
                TASKS[int(np.bincount(seg, minlength=len(TASKS)).argmax())]
            # The subtask name is recorded but NOT sent: see facts_v3.
            # 지시문은 넣어봤다가 뺐다. 넣으면 모델이 화면 대신 지시문을 보고 답해
            # 칸 안에서 상수가 된다 -- Rotate Box 77청크가 전부 한 값이 됐고
            # 정답지 rho 가 +0.525 에서 +0.491 로, 분산이 0.112 에서 0.049 로
            # 내려갔다. SHOVE 가 1등급에 붙박인 것도 지시문으로는 안 풀렸다.
            payload.append(([L[f], R[f]], facts_v3(x)))
            meta.append((f, task, x))
        try:
            rs = gate.judge_batch(payload, GUIDANCE, question=ASK, n_ask=len(ACTIVE), n_grade=NGRADE)
        except Exception as e:
            print(f"  ep{ep} f{grp[0]}+: {type(e).__name__}: {e}", flush=True)
            continue
        for (f, task, x), r in zip(meta, rs):
            # A judge that has died answers every call identically, so a file of
            # the right length with no content is indistinguishable from success
            # unless the empties are counted.
            if r.get("error") or not r.get("text", "").strip():
                nempty += 1
                if nempty >= 5 and ntot < 5:
                    raise SystemExit(f"판정기가 답을 안 함: {r.get('error', 'empty')}")
                continue
            nempty = 0
            picks = r.get("picks") or [None] * len(ACTIVE)
            gp = r.get("grade_probs")
            K = ratio_for(picks, CELL.get((ep, f)), gp)
            rec = {"ep": ep, "f": f, "task": task,
                   "cell": CELL.get((ep, f)), "conf": round(confidence(picks, gp), 3), "eg": [round(v,2) for v in (expected_grades(gp) or [])],
                   **{q: picks[i] for i, q in enumerate(ACTIVE)},
                   "K": round(K, 3), "K_snap": snap(K),
                   "text": r.get("text", "").replace("\n", " | "),
                   **{k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                      for k, v in x.items()}}
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            ntot += 1
        out.flush()
    print(f"ep{ep}: {ntot} chunks", flush=True)
out.close()

if FULL and NSH > 1:
    print(f"끝: {ntot} chunks -> {OUT}  (샤드라 띠에 펴는 것은 합친 뒤에 한다)")
    raise SystemExit(0)

# 확신을 칸의 띠에 편다. 정답지를 쓰지 않고 띠 폭과 그 칸의 자기 분포만 쓴다.
import collections as _c
from allex_v3_checks import TASK_RANGE, DEFAULT_RANGE, band_place
rows = [json.loads(l) for l in open(OUT)]
byc = _c.defaultdict(list)
for i, r in enumerate(rows):
    byc[r.get("cell")].append(i)
for cell, idx in byc.items():
    lo, hi = TASK_RANGE.get(cell, DEFAULT_RANGE)
    z = band_place([rows[i]["conf"] for i in idx], lo, hi)
    for i, zz in zip(idx, z):        # band_place 는 이미 [lo, hi] 안의 값을 준다
        rows[i]["K_spread"] = round(float(zz), 3)
        rows[i]["conf_spread"] = round((float(zz) - lo) / (hi - lo), 3)
with open(OUT, "w") as fh:
    for r in rows:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"끝: {ntot} chunks -> {OUT}")
