"""LIBERO 결정론적 위험 기술자.

액션 7차원: 0:3 EE 위치 델타 · 3:6 EE 회전 델타 · 6 그리퍼(이진 ±1).
RoboCasa 와 같은 델타 임베디먼트라 K2 압축은 인접 두 스텝을 **더한다**.

임계값은 이 데이터셋에서 실측한 값이다 (30 에피소드 8,294 스텝):
  위치 델타 크기  p50 0.518 · p95 0.958 · p99 1.155
  단일 스텝이 ±1 을 넘는 비율 0.0000 · K2 병합이 넘는 비율 0.0790
컨트롤러가 각 차원을 ±1 로 자르므로, 병합 초과는 그대로 잃는 변위가 된다.
"""
import numpy as np

CLIP = 1.0          # 컨트롤러 차원별 한계
SLOW_POS = 0.20     # p25 근처 — "기어가는" 구간
REV_MIN = 0.30      # 이보다 작은 움직임의 방향 변화는 잡음
TURN_COS = 0.94     # 약 20도. 이 데이터의 인접 쌍 코사인 하위 5% 지점.
                    # 방향 "반전"(cos<0)은 LIBERO 에서 0.02% 뿐이라 신호가 안 된다 —
                    # 궤적이 매우 매끄럽다(코사인 중앙값 0.998). 대신 꺾임 정도를 쓴다.


def descriptors(a, f=0, n=16, k=2):
    """a:(T,7) 계획 청크, f: 시작 프레임 -> dict"""
    w = np.asarray(a[f:f + n], dtype=np.float64)
    if w.shape[0] < 2:
        w = np.repeat(w, 2, axis=0)
    pos, rot, g = w[:, 0:3], w[:, 3:6], w[:, 6]
    speed = np.linalg.norm(pos, axis=1)
    npair = max(w.shape[0] // k, 1)
    p0, p1 = w[0:k * npair:k], w[1:k * npair:k]

    grip_change = float(np.abs(np.diff(g, prepend=g[0])).max() > 0.5)
    grip_pairs = float(np.mean(np.abs(p1[:, 6] - p0[:, 6]) > 0.5))

    v0, v1 = p0[:, 0:3], p1[:, 0:3]
    m0, m1 = np.linalg.norm(v0, axis=1), np.linalg.norm(v1, axis=1)
    cos = np.sum(v0 * v1, axis=1) / np.maximum(m0 * m1, 1e-9)
    big = (m0 > REV_MIN) & (m1 > REV_MIN)
    # 합치면 모퉁이가 잘려나가는 쌍의 비율. 각이 클수록 잘리는 변위가 크다.
    turn = float(np.mean(big & (cos < TURN_COS)))

    closed_slow = float(np.mean((g > 0) & (speed < SLOW_POS)))
    merged = np.abs(p0[:, 0:6] + p1[:, 0:6])
    clip_excess = float(np.mean(merged > CLIP))

    return {
        "speed_mean": float(speed.mean()), "speed_max": float(speed.max()),
        "rot_speed_mean": float(np.linalg.norm(rot, axis=1).mean()),
        "gripper_closed": float((g > 0).mean()),
        "grip_change": grip_change, "grip_pairs": grip_pairs,
        "turn": turn, "closed_slow": closed_slow,
        "clip_excess": clip_excess,
    }


def facts_text(x):
    """계산값을 사실 문장으로. 임계 판정까지 끝내고 결론만 준다."""
    parts = []
    parts.append("the gripper opens or closes during this window" if x["grip_change"]
                 else ("the gripper stays closed throughout" if x["gripper_closed"] > 0.5
                       else "the gripper stays open throughout"))
    parts.append(f"the path bends part-way through on {x['turn']:.0%} of the merge pairs"
                 if x["turn"] > 0.05 else "the end-effector keeps a consistent direction")
    sp = ("barely moving" if x["speed_mean"] < SLOW_POS else
          "moving at a normal pace" if x["speed_mean"] < 0.7 else "moving fast")
    parts.append(f"it is {sp} (mean step {x['speed_mean']:.2f}, peak {x['speed_max']:.2f})")
    if x["closed_slow"] > 0.5:
        parts.append("it is holding something while creeping along")
    tail = (" Merging pairs of steps would exceed the controller limit on "
            f"{x['clip_excess']:.0%} of the merged commands." if x["clip_excess"] > 0.05 else "")
    return ("MEASURED FROM THE PLANNED MOTION over the next ~1.6 seconds (these are computed "
            "facts, not estimates): " + "; ".join(parts) + "." + tail)


def computed_risk(x):
    """계산에서 바로 나오는 위험 — VLM 에게 묻지 않는다. 심각도 비례 연속값."""
    return {"grip_transition": 0.9 * (x["grip_change"] > 0),   # 사건형: 한 번이면 충분
            "turn": x["turn"],                                 # 누적형 — 꺾인 쌍의 비율
            "precise_hold": x["closed_slow"],                  # 누적형
            "infeasible_merge": x["clip_excess"]}              # 누적형
