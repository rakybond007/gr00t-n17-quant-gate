"""계산 위험 플래그의 연속값 버전.

기존 computed_risk 는 네 값을 전부 0/1 로 내보냈다. noisy-OR 특성상 하나만 1이면
risk=1 이 되어 점수가 0 으로 포화되고, 그 구간(전체의 29.5%)에서는 VLM 판단이
통째로 버려지며 순위도 무의미해진다. 실측: p_raw==0 과 '플래그 하나라도 1' 이 1.0000 일치.

네 값을 공통 단위 — **K2 병합 연산 중 실제로 해로운 것의 비율** — 로 바꾼다.
재라벨링이 필요 없다. 액션 숫자만으로 계산되므로 VLM 재호출 0회.
"""
import numpy as np

CLIP = 1.0

def soft_risk(a, f, n=16):
    """a:(T,12) 계획 청크, f: 시작 프레임. -> {플래그: 0~1 심각도}"""
    w = a[f:f+n]
    g = w[:, -1]
    d = w[:, 5:8]
    mag = np.linalg.norm(d, axis=1)
    n_pair = max(len(w)//2, 1)

    # K2 는 인접 두 스텝을 합친다. 각 병합쌍이 해로운지 쌍 단위로 센다.
    p0, p1 = w[0:2*n_pair:2], w[1:2*n_pair:2]

    # ① 그리퍼 상태가 바뀌는 지점을 가로지르는 병합쌍의 비율
    grip = float(np.mean(np.abs(p1[:, -1] - p0[:, -1]) > 0.5))

    # ② 두 스텝이 서로 반대라 합치면 상쇄되는 병합쌍의 비율
    v0, v1 = p0[:, 5:8], p1[:, 5:8]
    m0, m1 = np.linalg.norm(v0, axis=1), np.linalg.norm(v1, axis=1)
    cos = np.sum(v0*v1, axis=1) / np.maximum(m0*m1, 1e-9)
    rev = float(np.mean((m0 > 0.10) & (m1 > 0.10) & (cos < 0)))

    # ③ 잡은 채 기어가는 스텝의 비율 (정밀 배치 구간)
    hold = float(np.mean((g > 0.5) & (mag < 0.12)))

    # ④ 합친 명령이 컨트롤러 한계를 넘는 비율 — 원래부터 연속값이었는데 >0.20 에서 잘렸다
    merged = p0[:, 5:11] + p1[:, 5:11]
    merge = float(np.mean(np.abs(merged) > CLIP))

    return {"grip_transition": grip, "reversal": rev,
            "precise_hold": hold, "infeasible_merge": merge}
