"""LIBERO 지시문 -> 서브액션 분해. 규칙 기반이라 재현된다.

**왜 규칙인가.** 서브액션 분해는 파이프라인에서 LLM 이 하는 단계다. 40개를 손으로
적으면 재현도 검증도 안 된다. LIBERO 지시문은 정형이라 규칙으로 뽑히고, 규칙이
틀리는 지시문은 눈에 띈다.

어휘는 배속 한계가 갈릴 만한 것으로만 나눈다.

    grasp_open       열린 자리에 놓인 것을 집음
    grasp_stacked    다른 물체 위에 얹힌 것을 집음 (밑을 건드리면 무너진다)
    grasp_between    두 물체 사이에 낀 것을 집음
    grasp_confined   서랍·용기 안에서 집음
    carry            쥐고 옮김
    place_catcher    받아 주는 곳(바구니·그릇)에 넣음
    place_precise    좁고 정해진 자리(접시·랙·스토브·캐비닛 위)에 놓음
    place_open       넓은 면에 놓음
    grasp_handle     손잡이를 쥠
    pull_swing       쥔 채 서랍·문을 엶
    push_close       문·서랍을 밀어 닫음
    push_object      물체를 밀어 옮김
    turn_knob        꼭지를 돌림
"""
import re

DEST = [
    (r"in the basket|in the bowl", "place_catcher"),
    (r"on the plate|on the left plate|on the right|on the rack|on the stove|"
     r"on top of the cabinet|on it\b|in the back compartment|in the microwave|"
     r"in the bottom drawer|to the right of", "place_precise"),
]
SRC = [
    (r"\bon the (ramekin|cookie box|wooden cabinet|stove)\b", "grasp_stacked"),
    (r"\bbetween the\b", "grasp_between"),
    (r"\bin the top drawer\b|\bfrom the drawer\b", "grasp_confined"),
]


def decompose(instr):
    s = instr.lower()
    out = []
    n_obj = 2 if (s.startswith("put both") or " and put " in s) else 1
    if re.search(r"\bopen the .*drawer\b", s) and "put" not in s:
        return ["grasp_handle", "pull_swing"]
    if re.search(r"^open the ", s):
        out += ["grasp_handle", "pull_swing"]
    if re.search(r"^push the ", s):
        return ["push_object"]
    if re.search(r"^turn on the stove$", s):
        return ["turn_knob"]
    if re.search(r"turn on the stove and", s):
        out += ["turn_knob"]
    # 집기-옮기기-놓기
    if re.search(r"pick up|put the|put both|place it|put it", s):
        src = next((v for p, v in SRC if re.search(p, s)), "grasp_open")
        dst = next((v for p, v in DEST if re.search(p, s)), "place_open")
        out += ([src, "carry", dst] * n_obj)
    if re.search(r"and close it", s):
        out += ["push_close"]
    return out or ["grasp_open", "carry", "place_open"]
