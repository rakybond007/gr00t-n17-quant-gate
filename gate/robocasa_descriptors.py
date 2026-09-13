"""로보카사 2층 — 계획된 청크에서 기술자를 정확히 계산한다.

기존 액션 문항 E/F/G/H는 전부 액션 숫자만으로 정확히 계산되는 값이었다.
VLM에게 물었을 때의 정확도는 E=0.834, F=0.520, G=0.929, H=0.616 —
특히 F는 사실상 무신호였다. 계산으로 내리면 정확하고, 콜도 하나 줄어든다.

액션 12차원: 0-4 미사용, 5-7 EE delta xyz, 8-10 회전, 11 그리퍼(0/1).
"""
import json
import os

import numpy as np

CLIP = 1.0   # 컨트롤러 액션 한계

# 창 평균속도를 전체 분포의 백분위로 옮기는 표 (build_speed_percentiles.py).
# 101개 경계값이라 searchsorted 한 번이면 된다. 표가 없으면 속도 줄은 예전처럼
# 절대 경계로 떨어진다 -- 없다고 죽지는 않되, 그 경우는 눈에 보이게 둔다.
_PCT_PATH = (f"{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}"
             f"/configs/robocasa_speed_percentiles.json")
try:
    _PCT = np.asarray(json.load(open(_PCT_PATH))["percentiles"], dtype=float)
except Exception:                                    # noqa: BLE001
    _PCT = None
# SPEED_PCT=0 으로 예전 절대 경계 문구로 되돌린다. 둘을 같은 장면에서 견주려고
# 둔 스위치이고, 기본은 분위다.
if os.environ.get("SPEED_PCT", "1") == "0":
    _PCT = None


def speed_percentile(v):
    """speed_mean -> 0..100. 표가 없으면 None."""
    if _PCT is None:
        return None
    return float(np.clip(np.searchsorted(_PCT, v, side="right") - 1, 0, 100))

# 사실 문장의 지평을 창 밖으로 늘린다. 접근 장면이 전체의 61% 인데 창 안에 그리퍼
# 사건이 없어서 "stays open throughout" 한 문장만 받는다 -- 위험한 순간이 1초 뒤인지
# 5초 뒤인지 구별할 근거가 없다. 등급표는 이미 그 근접성을 묻는다("4 = 한 뼘 앞",
# "3 = 향해 가는 중이나 아직 멀다")는데 정보가 없던 자리다. 방향 수정이 통했던 것과
# 같은 형태의 공백이지만, 그쪽은 측정된 뭉갬이었고 이쪽은 가설이므로 기본은 끈다.
GRIP_HORIZON = int(os.environ.get("GRIP_HORIZON", "0"))


def descriptors(a, f, n=16):
    """a:(T,12) 계획 청크, f: 시작 프레임. -> dict"""
    w = a[f:f+n]
    g = w[:, -1]
    d = w[:, 5:8]
    mag = np.linalg.norm(d, axis=1)
    # 그리퍼 상태 전이 (파지·해제 순간)
    grip_change = float(np.abs(np.diff(g, prepend=g[0])).max() > 0.5)
    # **어느 방향으로, 창 안 언제.** 지금까지 사실 문장은 "opens or closes" 한 줄로
    # 합쳐 줬고, 그래서 모델의 답이 닫힘과 열림에서 사실상 같았다(conf 0.309 대
    # 0.318, 차이 -0.009). 파지를 시작하는 순간과 놓고 손을 떼는 순간은 압축
    # 위험이 다를 텐데 같은 문장을 받았다. 시점도 안 줘서 창 앞쪽에서 일어나는
    # 전이와 끝에서 일어나는 전이를 구별할 수 없었다. 둘 다 공짜로 계산된다.
    _dg = np.diff(g)
    _cl = np.nonzero(_dg > 0.5)[0]
    _op = np.nonzero(_dg < -0.5)[0]
    grip_close = float(len(_cl) > 0)
    grip_open = float(len(_op) > 0)
    _idx = list(_cl) + list(_op)
    grip_at = int(min(_idx)) if _idx else -1
    # 실제 방향 반전: 연속 두 스텝 모두 유의미하게 크고 90도 넘게 꺾임
    rev = 0.0
    if len(d) > 1:
        v1, v2 = d[:-1], d[1:]
        m1, m2 = np.linalg.norm(v1,axis=1), np.linalg.norm(v2,axis=1)
        cos = np.sum(v1*v2,axis=1)/np.maximum(m1*m2, 1e-9)
        rev = float(np.any((m1>0.10) & (m2>0.10) & (cos<0)))
    # 닫은 채 저속 — 정밀 배치 구간의 서명
    closed_slow = float(((g>0.5) & (mag<0.12)).mean() > 0.5)
    # 감속하며 끝남
    decel = float(len(mag)>=8 and mag[-4:].mean()<0.15 and mag[:4].mean()>mag[-4:].mean())
    # K2 병합이 컨트롤러 한계를 넘는가 (실현 가능성)
    merged = w[0:-1:2, 5:11] + w[1::2, 5:11] if len(w)>=2 else np.zeros((1,6))
    clip_excess = float(np.mean(np.abs(merged) > CLIP))
    # 창 밖 다음 파지까지 몇 스텝. 창 안에 파지가 있으면 0 으로 두어 문장이 겹치지
    # 않게 한다. 없으면 -1.
    next_grasp = -1
    if GRIP_HORIZON > 0 and not grip_close:
        _fut = np.diff((np.asarray(a[:, -1]) > 0.5).astype(int))
        _nx = [i for i in np.nonzero(_fut > 0.5)[0] if i >= f + n]
        if _nx and (_nx[0] - (f + n)) <= GRIP_HORIZON:
            next_grasp = int(_nx[0] - (f + n))
    return {"grip_change":grip_change, "grip_close":grip_close,
            "grip_open":grip_open, "grip_at":grip_at,
            "next_grasp":next_grasp,
            "reversal":rev, "closed_slow":closed_slow,
            "decel":decel, "clip_excess":clip_excess,
            "speed_mean":float(mag.mean()), "speed_max":float(mag.max()),
            "speed_pct":speed_percentile(float(mag.mean())),
            "gripper_closed":float((g>0.5).mean())}

def facts_text(x):
    """계산값을 사실 문장으로. 임계 판정까지 끝내고 결론만 준다.

    문장 다섯 줄이 조건과 무관하게 항상 나오고, 각 줄은 정해진 문구 중 하나를
    고를 뿐이다. 조건부로 줄이 붙었다 빠졌다 하면 프롬프트 길이가 프레임마다
    달라지고, 그러면 배치가 패딩을 필요로 한다 -- 그 패딩이 mm_token_type_ids
    를 어긋나게 해서 배치 8 에서 32 개 중 17 개가 빈 답으로 돌아왔다. 숫자는
    이미 %.2f 고정 폭이므로, 줄 수만 고정하면 길이가 완전히 같아진다.

    컨트롤러 클리핑 초과율을 알려주던 꼬리는 뺐다. 클리핑은 평가에서 풀어야 할
    하네스 제약이지 이 순간의 성질이 아니다.
    """
    parts = []
    _at = x.get("grip_at", -1)
    _when = f" {_at} of 16 steps in" if _at >= 0 else ""
    if x.get("grip_close") and x.get("grip_open"):
        parts.append(f"the gripper closes and then opens again{_when}")
    elif x.get("grip_close"):
        parts.append(f"the gripper CLOSES on something{_when} -- the grasp is made "
                     f"inside this window")
    elif x.get("grip_open"):
        parts.append(f"the gripper OPENS to let go{_when} -- the release happens "
                     f"inside this window")
    elif x["grip_change"]:
        parts.append("the gripper opens or closes during this window")
    else:
        parts.append("the gripper stays closed throughout" if x["gripper_closed"] > 0.5
                     else "the gripper stays open throughout")
    parts.append("the end-effector reverses direction sharply" if x["reversal"]
                 else "the end-effector keeps a consistent direction")
    # 절대 경계(0.12 / 0.50)는 하위 1.6% 와 46.6% 를 자른다 -- "barely moving" 은
    # 사실상 안 나오고 과반이 "moving fast" 가 된다. 전체 분포의 백분위로 주면
    # 다섯 구간이 각각 20% 씩 쓰인다.
    pc = speed_percentile(x["speed_mean"])
    if pc is None:
        sp = ("barely moving" if x["speed_mean"] < 0.12 else
              "moving at a normal pace" if x["speed_mean"] < 0.50 else "moving fast")
        parts.append(f"it is {sp} (mean step {x['speed_mean']:.2f}, "
                     f"peak {x['speed_max']:.2f})")
    else:
        sp = ("among the slowest motions in this dataset" if pc < 20 else
              "on the slow side for this dataset" if pc < 40 else
              "middling in speed for this dataset" if pc < 60 else
              "on the fast side for this dataset" if pc < 80 else
              "among the fastest motions in this dataset")
        parts.append(f"it is {sp} -- faster than {pc:.0f}% of all moments here "
                     f"(mean step {x['speed_mean']:.2f}, peak {x['speed_max']:.2f})")
    parts.append("it is holding something while creeping along" if x["closed_slow"]
                 else "it is not creeping along with something held")
    _ng = x.get("next_grasp", -1)
    if _ng >= 0:
        # 줄 수를 늘리지 않는다 -- 마지막 줄에 덧붙인다. 줄 수가 프레임마다
        # 달라지면 배치가 패딩을 필요로 한다.
        parts.append(("it is decelerating to a near stop" if x["decel"]
                      else "it is not decelerating to a stop")
                     + f", and the next grasp comes {_ng} steps after this window")
    else:
        parts.append("it is decelerating to a near stop" if x["decel"]
                     else "it is not decelerating to a stop")
    return ("MEASURED FROM THE PLANNED MOTION over the next ~1 second (these are computed "
            "facts, not estimates): " + "; ".join(parts) + ".")


# 계산에서 바로 나오는 위험 플래그 — VLM에게 묻지 않는다
def computed_risk(x):
    return {"grip_transition": x["grip_change"],
            "reversal": x["reversal"],
            "precise_hold": x["closed_slow"],
            "infeasible_merge": float(x["clip_excess"] > 0.20)}
