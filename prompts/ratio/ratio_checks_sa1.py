"""배속 문항 sa1 -- 서브액션을 속성으로 판정한다.

역산된 한계표(analysis/subaction/robocasa_limits.json)를 쓰려면 지금 순간이 어느
서브액션인지 알아야 한다. **10지선다로 묻지 않는다** -- 앞서 4지선다 국면 분류를
시켰더니 32% 로 한 답에 붕괴했다(파지·해제 예측 0%).

대신 서브액션을 가르는 **속성**을 등급으로 묻는다. 속성은 화면에 보이는 것만
고른다. 그리퍼 개폐·속도·변위는 액션에서 나오므로 계산 사실로 넘긴다.

    A HOLDING     지금 무언가를 쥐고 있는가        -> carry 계열인가
    B TIGHT_SPOT  놓을 자리가 좁고 정확한가         -> place_precise (1.25)
    C CATCHER     받아 주는 곳에 넣는가            -> place_catcher (2.37)
    D FIXTURE     붙박이를 밀거나 돌리는가          -> contact_* (3.2)
    E CONFINED    닿아야 할 것이 좁은 데 있는가      -> grasp_confined (2.61)

부호는 한계표의 순서에서 나온다: 좁은 자리·붙들고 옮김이 낮고, 붙박이 조작과
열린 곳에서 집기가 높다.
"""
NGRADE = 5
SIGN = {"A": -1, "B": -1, "C": +1, "D": +1, "E": -1}
WEIGHT = {"A": 0.35, "B": 0.45, "E": 0.20, "C": 0.40, "D": 0.60}
NAME = {"A": "HOLDING", "B": "TIGHT_SPOT", "C": "CATCHER",
        "D": "FIXTURE", "E": "CONFINED"}

GUIDANCE = (
    "You are judging one moment of a robot arm working in a kitchen. The measurements "
    "below are computed from the planned motion and are facts -- they already tell you "
    "whether the gripper closes or opens in this stretch, how fast the arm moves, and "
    "how far a merged step would jump. Do not re-estimate them and do not answer about "
    "them.\n\n"
    "Answer only about what the picture shows: **what the hand is dealing with.** "
    "The stages of a kitchen manipulation differ in how much accuracy they demand, and "
    "which stage this is can be read off the scene:\n"
    "  - shoving a drawer shut, pressing a button, turning a knob: the fixture moves as "
    "one piece and a coarse contact does it\n"
    "  - taking hold of something that stands clear on a surface: forgiving\n"
    "  - reaching into a microwave, a sink, a cabinet, under a machine: the way in is "
    "narrow\n"
    "  - carrying something already held: the object must not be lost\n"
    "  - dropping into a basin, bin or cabinet that catches: forgiving at the far end\n"
    "  - setting down on a plate, a rack, a burner, under a dispenser: the least "
    "forgiving moment there is"
)

ASK = (
    "Answer each check on its own line as \"A) 3\", in order, nothing else -- one digit "
    "from 1 to 5 per check:\n"
    "  5 = clearly true of this moment   4 = mostly   3 = partly   2 = barely\n"
    "  1 = not true at all\n"
    "A) Is the hand HOLDING something right now and moving it, rather than empty or\n"
    "   only touching a fixture?\n"
    "B) Is something about to be SET DOWN ON A SMALL EXACT SPOT -- a plate, a rack, a\n"
    "   burner, under a machine -- where it has to sit squarely?\n"
    "C) Is something about to go INTO SOMETHING THAT CATCHES IT -- a sink basin, a bin,\n"
    "   an open cabinet or microwave -- where landing off-centre changes nothing?\n"
    "D) Is the hand working a FIXTURE that moves as one piece -- a drawer face, a door,\n"
    "   a button, a knob -- rather than handling a loose object?\n"
    "E) Is the thing the hand must reach INSIDE A CONFINED SPACE -- in a microwave, a\n"
    "   sink, a cabinet, a pan, under a dispenser -- so the way in is narrow?\n"
    "Answer:"
)
