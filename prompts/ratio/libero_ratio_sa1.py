"""LIBERO 배속 문항 sa1 -- 서브액션을 속성으로 판정해 배속을 낸다.

**conf 문항과 다른 물건이다.** v3c 는 압축 여부를 0~1 신뢰도로 내는 문항이고,
이것은 배속을 내는 문항이다. 부호·가중으로 합산하지 않고 답에서 서브액션을 읽어
한계표로 배속을 정한다.

한계표는 LIBERO 사다리(1.0/1.667/2.0)에서 읽는다. 사다리 끝인 2.0 에서도 82% 가
멀쩡하므로 **천장이 사다리 밖**이고, 그래서 한계를 촘촘히 나눌 수 없다. 대신
**깨지는 소수만 낮게 잡고 나머지는 관대하게** 둔다 -- 실측이 그 구조다.

    깨지는 7/40 태스크의 공통점 (유지율 0.793 대 0.954, p=0.0032)
      집을 것 옆·밑에 높이 있는 이웃 (라미킨·쿠키상자·서랍 안)  4개
      한 지시 안에 단계가 둘 이상 (and close it / put both)     3개
    멀쩡한 쪽은 이웃이 납작하다 (접시·스토브·조리대 위: 0.86~0.90)

    hemmed_tall   1.25   높은 이웃 사이·위에서 집기
    multi_stage   1.67   한 지시에 단계 둘
    base          2.00   그 밖 (사다리 끝, 관대하게)
"""
NGRADE = 5
# 배속 문항이라 부호·가중이 없다. 답 -> 서브액션 -> 한계.
LIMITS = {"hemmed_tall": 1.25, "multi_stage": 1.67, "base": 2.00}
NAME = {"A": "HEMMED_TALL", "B": "FLAT_NEIGHBOUR", "C": "STAGE_LEFT"}
SIGN = {"A": -1, "B": +1, "C": -1}          # 채점기 호환용
WEIGHT = {"A": 0.70, "C": 0.30, "B": 1.00}

GUIDANCE = (
    "You are judging one moment of a robot arm on a table top, to decide how coarsely "
    "the next second of its motion could be run. The measurements below are computed "
    "from the planned motion and are facts -- they tell you whether the gripper closes "
    "or opens here and how far a merged step would jump. Do not answer about them.\n\n"
    "Answer only about what the picture shows. Most of this robot's work tolerates a "
    "long step: crossing open table, carrying something with room around it, dropping "
    "into a basket. Two situations do not.\n\n"
    "The first is reaching for something with a **tall neighbour** -- a bowl, a cup, a "
    "box standing beside it or underneath it. A wider swing strikes the neighbour or "
    "topples what is stacked. A flat plate or the bare table beside the object is not "
    "this: height is what matters, not proximity.\n\n"
    "The second is a job that still has **another stage to go** after this one -- a "
    "drawer to shut, a second object to fetch. The arm has to arrive somewhere it can "
    "continue from, not merely finish this motion."
)

ASK = (
    "Answer each check on its own line as \"A) 3\", in order, nothing else -- one digit "
    "from 1 to 5 per check:\n"
    "  5 = clearly true of this moment   4 = mostly   3 = partly   2 = barely\n"
    "  1 = not true at all\n"
    "A) Is the object the hand is going for HEMMED IN BY SOMETHING WITH HEIGHT -- a\n"
    "   bowl, cup or box standing next to it, or one it is stacked on top of -- that a\n"
    "   wider swing would strike or topple?\n"
    "B) Is the space around the object FLAT AND CLEAR -- bare table, a plate lying\n"
    "   flat, nothing standing up within reach of a wider swing?\n"
    "C) After this motion, does the job still have ANOTHER STAGE to go -- a container\n"
    "   left open that must be shut, or a second object still to be moved?\n"
    "Answer:"
)


def subaction(grades):
    """등급 -> 서브액션. A 가 높으면 끼임, C 가 높으면 다단계, 아니면 base."""
    a, b, c = (grades["A"], grades["B"], grades["C"])
    if a >= 3.5 and a > b:
        return "hemmed_tall"
    if c >= 3.5:
        return "multi_stage"
    return "base"


def ratio(grades):
    return LIMITS[subaction(grades)]
