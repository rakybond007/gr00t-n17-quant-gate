"""로보카사 2층 — 계획된 청크에서 기술자를 정확히 계산한다.

기존 액션 문항 E/F/G/H는 전부 액션 숫자만으로 정확히 계산되는 값이었다.
VLM에게 물었을 때의 정확도는 E=0.834, F=0.520, G=0.929, H=0.616 —
특히 F는 사실상 무신호였다. 계산으로 내리면 정확하고, 콜도 하나 줄어든다.

액션 12차원: 0-4 미사용, 5-7 EE delta xyz, 8-10 회전, 11 그리퍼(0/1).
"""
import numpy as np

CLIP = 1.0   # 컨트롤러 액션 한계

def descriptors(a, f, n=16):
    """a:(T,12) 계획 청크, f: 시작 프레임. -> dict"""
    w = a[f:f+n]
    g = w[:, -1]
    d = w[:, 5:8]
    mag = np.linalg.norm(d, axis=1)
    # 그리퍼 상태 전이 (파지·해제 순간)
    grip_change = float(np.abs(np.diff(g, prepend=g[0])).max() > 0.5)
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
    return {"grip_change":grip_change, "reversal":rev, "closed_slow":closed_slow,
            "decel":decel, "clip_excess":clip_excess,
            "speed_mean":float(mag.mean()), "speed_max":float(mag.max()),
            "gripper_closed":float((g>0.5).mean())}

def facts_text(x):
    """계산값을 사실 문장으로. 임계 판정까지 끝내고 결론만 준다."""
    parts=[]
    parts.append("the gripper opens or closes during this window" if x["grip_change"]
                 else ("the gripper stays closed throughout" if x["gripper_closed"]>0.5
                       else "the gripper stays open throughout"))
    parts.append("the end-effector reverses direction sharply" if x["reversal"]
                 else "the end-effector keeps a consistent direction")
    sp = ("barely moving" if x["speed_mean"]<0.12 else
          "moving at a normal pace" if x["speed_mean"]<0.50 else "moving fast")
    parts.append(f"it is {sp} (mean step {x['speed_mean']:.2f}, peak {x['speed_max']:.2f})")
    if x["closed_slow"]: parts.append("it is holding something while creeping along")
    if x["decel"]:       parts.append("it is decelerating to a near stop")
    tail = (" Merging pairs of steps would exceed the controller limit on "
            f"{x['clip_excess']:.0%} of the merged commands." if x["clip_excess"]>0.05 else "")
    return ("MEASURED FROM THE PLANNED MOTION over the next ~1 second (these are computed "
            "facts, not estimates): " + "; ".join(parts) + "." + tail)

# 계산에서 바로 나오는 위험 플래그 — VLM에게 묻지 않는다
def computed_risk(x):
    return {"grip_transition": x["grip_change"],
            "reversal": x["reversal"],
            "precise_hold": x["closed_slow"],
            "infeasible_merge": float(x["clip_excess"] > 0.20)}
