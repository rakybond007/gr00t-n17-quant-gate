"""LIBERO 청크 라벨러 — cosmos_1call_v6.py 의 LIBERO 판.

로보카사와 같은 1콜 설계: 2층(계산)은 libero_descriptors 가 사실로 진술하고,
3층(VLM)은 정지화면으로만 알 수 있는 다섯 문항을 한 번의 이미지 prefill 로 답한다.

로보카사와 다른 점.
  * 뷰가 둘(front + left_wrist)이다. 타일을 3등분하지 않고 2등분한다.
  * 액션이 7차원(0:3 위치 델타 · 3:6 회전 델타 · 6 그리퍼 ±1)이다 —
    libero_descriptors 가 이미 그 규격으로 계산한다.
  * 프레임 스트라이드 4 (10fps · 평균 162 스텝). 로보카사 stride 8/20fps 와
    같은 0.4 초 간격이다.

재개(resume) 규율. 백그라운드 파티션은 선점당한다. 시작할 때
  1) 기존 출력에서 파싱되는 줄만 남기고 다시 쓴다 (선점으로 끊긴 마지막 줄 제거),
  2) (ep,f) 를 done 으로 올려 다시 내보내지 않는다.
그래서 재큐된 샤드가 행을 중복시키지 않는다 — qgate labels 가 잡는 실패 모드다.
판정 실패(judge 오류)는 행을 쓰지 않고 건너뛴다. 0 으로 채워 쓰면 다음 실행이
그 청크를 완료로 보고 영영 다시 묻지 않는다.

사용법:  python libero_label_chunks.py <port> <shard> <nshards>
"""
import json, os, sys, numpy as np, pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vlm_gate import VLMGate
from libero_descriptors import descriptors, facts_text, computed_risk

BASE = "/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate"
DS = "/sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/kimtaey/libero_gr00t_delta"
# 타일 디렉터리도 env 로 고른다. 부호 검증에는 두 풀이 다 들어간
# libero_pools 를 쓴다 (gen_libero_tiles_pools.py).
TIL = os.environ.get("TILES", f"{BASE}/output/_gate_distill/libero_full/tiles")
MAN = os.environ.get("MANIFEST", f"{BASE}/output/_gate_distill/libero_tiles_manifest.txt")
TAG = os.environ.get("TAG", "libero_v2")
NVIEW = 2
NQ, SLOTS = 5, "ABCDE"
NGRADE = 5          # 1~5 등급. phase9·allex 와 같은 경로다.

PORT, SHARD, NSH = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
LIMIT = int(os.environ.get("LIMIT", "0"))          # >0 이면 스모크용으로 이만큼만
OUT = f"{BASE}/output/_gate_distill/{TAG}_s{NSH}_{SHARD}.jsonl"

# 프롬프트 판은 env 로 고른다. v1 은 YES/NO 5문항(등급 없음)이고 검증 루프를
# 거친 적이 없다. v2 는 측정된 손상에서 뽑은 5문항 + robocasa 와 같은 5등급이다
# (`prompts/libero_v2.txt`, 도출은 `scripts/derive_libero_questions.py`).
PV = os.environ.get("PROMPT_VER", "v2")
G = open(f"{BASE}/analysis/_evolver/_libero/libero_guidance_{PV}.txt").read().strip()
ASK = open(f"{BASE}/analysis/_evolver/_libero/libero_questions_{PV}.txt").read().strip()

info = json.load(open(f"{DS}/meta/info.json"))
instr = {}
for l in open(f"{DS}/meta/episodes.jsonl"):
    d = json.loads(l)
    c = [t for t in d.get("tasks", []) if isinstance(t, str) and len(t.split()) > 1]
    instr[d["episode_index"]] = c[0] if c else ""

# --- 재개: 온전한 줄만 남기고 다시 쓴 뒤 done 집합을 만든다 ---
done = set()
if os.path.exists(OUT):
    keep = []
    for l in open(OUT):
        try:
            r = json.loads(l)
        except Exception:
            continue                      # 선점으로 잘린 줄 — 버린다
        if (r.get("ep"), r.get("f")) in done:
            continue                      # 과거 실행이 남긴 중복 — 한 번만 남긴다
        done.add((r["ep"], r["f"])); keep.append(json.dumps(r))
    tmp = OUT + ".tmp"
    with open(tmp, "w") as f:
        f.write("".join(s + "\n" for s in keep))
    os.replace(tmp, OUT)
    print(f"shard{SHARD}: resume, {len(done)} chunks already done", flush=True)

gate = VLMGate(f"http://127.0.0.1:{PORT}", timeout=180)

acts = {}
def A(ep):
    if ep not in acts:
        ch = ep // info["chunks_size"]
        try:
            acts[ep] = np.stack(pd.read_parquet(
                f"{DS}/data/chunk-{ch:03d}/episode_{ep:06d}.parquet")["action"].values)
        except Exception:
            acts[ep] = None
        if len(acts) > 40:
            for k in list(acts)[:20]:
                acts.pop(k, None)
    return acts[ep]

out = open(OUT, "a")
n = skipped = 0

# **배치는 라벨의 일부다.** phase9 에서 재어 보니 단건과 배치8 이 32칸 중 2칸
# 달랐다 -- 배치 폭이 다르면 다른 커널이 잡히고 bfloat16 끝자리가 움직인다.
# 각각은 재현되므로 값을 meta 에 적어 두고 재실행 때 같은 값을 쓴다.
BATCH = int(os.environ.get("LIBERO_BATCH", "8"))
json.dump({"batch": BATCH, "shard": SHARD, "nshard": NSH, "prompt": PV,
           "tag": TAG, "tiles": TIL, "manifest": MAN},
          open(OUT.replace(".jsonl", "_meta.json"), "w"), ensure_ascii=False, indent=1)


def prep(nm):
    """타일 하나를 판정 입력으로. 못 읽으면 None."""
    # 프레임은 마침표까지 읽는다. [:3] 으로 자르면 f1000.png 가 100 이 되어
    # 다른 프레임의 행을 덮어쓴다 (이 데이터셋은 최대 505 라 무사하지만 함정이다).
    ep = int(nm[2:6]); f = int(nm.split("_f")[1].split(".")[0])
    if ep % NSH != SHARD or (ep, f) in done:
        return None
    a = A(ep)
    if a is None or f >= len(a) - 4:
        return None
    x = descriptors(a, f)
    try:
        im = np.array(Image.open(f"{TIL}/{nm}").convert("RGB"))
    except Exception:
        return None
    h, w, _ = im.shape
    views = [Image.fromarray(im[:, k * w // NVIEW:(k + 1) * w // NVIEW])
             for k in range(NVIEW)]
    return ep, f, x, (views, f"{instr.get(ep, '')}\n{facts_text(x)}")


names = sorted(open(MAN).read().split())
buf = []
for nm in names + [None]:                      # None 이 마지막 배치를 흘려보낸다
    if nm is not None:
        g = prep(nm)
        if g is None:
            continue
        buf.append(g)
        if len(buf) < BATCH:
            continue
    if not buf:
        continue
    # **n_grade 와 mode="text" 를 반드시 같이 넘긴다.** 둘 중 하나라도 빠지면
    # 판정기가 강제된 YES/NO 슬롯의 로짓을 읽는 경로로 간다 -- 모델이 하지 않은
    # 답을 짓는 것이라 저장소가 금지한 것이다(CLAUDE.md 되돌리지 말 것 1).
    try:
        rs = gate.judge_batch([g[3] for g in buf], G, question=ASK,
                              n_ask=NQ, n_grade=NGRADE, mode="text")
    except Exception as e:
        print(f"shard{SHARD}: batch {type(e).__name__}: {e}", flush=True)
        buf = []
        continue
    for (ep, f, x, _), r in zip(buf, rs):
        # 죽은 판정기는 모든 호출에 같은 답을 준다. 길이만 맞는 파일이 나오면
        # 개수를 세는 것으로는 성공과 구별되지 않는다 -- 본문이 비면 실패로 센다.
        c = r.get("picks")
        if r.get("error") or not str(r.get("text", "")).strip() \
                or not c or len(c) != NQ or any(v is None for v in c):
            skipped += 1
            if skipped % 200 == 1:
                print(f"shard{SHARD}: judge miss ep{ep} f{f}: "
                      f"err={r.get('error','')!r} picks={r.get('picks')!r} "
                      f"text={str(r.get('text',''))[:60]!r}", flush=True)
            continue
        # **등급 분포를 같이 적는다.** 모델이 텍스트로 답한 뒤 그 답이 얼마나
        # 확실했는지를 덧붙이는 것이다 -- 강제된 슬롯의 로짓을 답으로 삼는
        # 옛 방식과 다르다(CLAUDE.md 되돌리지 말 것 1 은 그쪽을 금지한 것이다).
        # P(3)=0.9 와 P(2)=.3/P(3)=.35/P(4)=.3 은 전혀 다른 상태인데 정수로만
        # 받으면 둘이 같아진다. allex 는 이미 이 값으로 신뢰도를 낸다.
        gp = r.get("grade_probs")
        rec = {"ep": ep, "f": f, **{k: int(v) for k, v in zip(SLOTS, c)},
               **computed_risk(x), "speed_mean": x["speed_mean"],
               "ans": r.get("text", "")}
        if gp and len(gp) == NQ:
            rec["gp"] = [[round(float(v), 4) for v in row] for row in gp]
        out.write(json.dumps(rec) + "\n")
        n += 1
    out.flush()                                 # 선점에 대비해 배치마다 flush
    buf = []
    if n % 400 < BATCH:
        print(f"shard{SHARD}: {n} (miss {skipped})", flush=True)
    if LIMIT and n >= LIMIT:
        break
out.close()
print(f"shard{SHARD} done: {n} rows, {skipped} miss", flush=True)
print(f"shard{SHARD} 완료 {n} (skipped {skipped}) -> {OUT}")
