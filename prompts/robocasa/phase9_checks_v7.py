"""phase9 문항 v3 = v2 + **C 한 문항 확장.** 세 번째 수정이 값을 하는지 재는 판.

v2 에서 밀기·돌리기 부류(유지율 1.032, 가장 강함)는 감점이 0 이 되지만 가점도 거의
못 받는다 -- C 가 "힌지·레일 위의 문·서랍" 만 덮으므로 버튼·손잡이·수도꼭지는
어디에도 걸리지 않아 conf 가 0.5 근처에 앉는다. 순서는 맞아도 역치에 너무 가깝다.
그래서 C 를 "기구가 잡아 주는 것" 으로 넓혀 손잡이·스위치·버튼·레버를 포함시킨다.
문항 수와 부호는 그대로이고 바뀐 것은 C 의 문구 한 줄이다.

이 확장이 필요한지는 재서 정한다 -- v2 와 같은 720 장면에서 나란히 돌린다.


문항 수 5, 감점 2 / 가점 3, 등급표, conf 공식 모두 v1 그대로다. 바뀐 것은 A 와 B 가
무엇을 가리키느냐이고, 그 근거는 24 태스크 x 50 에피 폐루프 실측이다.

v1 이 왜 거꾸로였나. A(감점 가중 0.667, 다섯 중 최대)가 "버튼 누르기·스위치·다이얼
돌리기" 를 깎았는데, 그 부류가 압축에 **가장 강한** 부류다.

    고정된 것에 닫음(노브·버튼·밀어닫기) 11개  K2 유지율 1.032  K3 0.933
    열린 조리대에서 집음    4개             0.969      0.828
    좁은 속에서 꺼냄        5개             0.766      0.509
    손잡이 파지/기계에 끼움  4개             0.730      0.495
    구속 단계 vs 유지율  K2 -0.681 (p=0.0003) · K3 -0.737 (p<0.0001)

즉 압축이 깨뜨리는 것은 "한 번의 정확한 접촉" 이 아니라 **그리퍼가 무엇에 닫히는가**
다. 액션 채널만 보면 이것이 안 보인다 -- 노브를 돌릴 때도 버튼을 누를 때도 그리퍼는
닫는다(CoffeePressButton 닫힘 1.000, TurnOnStove 1.000). 그래서 계산 특징으로는
유지율을 못 맞춘다(닫힘 횟수 -0.245 p=0.25). 가르는 것은 닫히는 **대상**이 기구에
고정돼 있는지다. 고정된 것에 닫는 것은 가장 강하고, 열린 조리대에서
집는 것은 거의 그만큼 강하며, 좁은 속에서 꺼내거나 손잡이를 잡는 것이 약하다.

그래서 A 는 "닫아야 하는가", B 는 "그 닫는 자리가 좁은가" 를 묻는다. B 가 v1 에서
사실상 무작위였던(엔트로피 0.896, 뽑힌 등급 확률 0.352) 이유도 이것이다 -- B 는
**도착지**가 좁은지 물었고, 실제로 가르는 것은 **파지하는 자리**였다. 동작의 반대쪽
끝을 보고 있었으니 일관되게 볼 것이 없었다.

C·D·E 의 문구는 손대지 않았다. 가중만 다시 잡았다(아래 근거).
"""
NGRADE = 5

# 가중은 실측된 분리 폭에서 읽었다. 두 쪽 합은 v1 처럼 각각 1 이다.
#
# A = 0.6 -- "닫아야 하는가" 는 단계 0(1.032) 과 1~3(평균 0.79) 을 가른다. 폭 0.24.
# B = 0.4 -- "그 자리가 좁은가" 는 단계 1(0.969) 과 2~3(평균 0.75) 을 가른다. 폭 0.22
#            로 A 에 거의 맞먹고, K3 에서는 오히려 더 크다(0.828 대 0.502 = 0.33 對
#            A 의 0.32). v1 에서 B 를 0.333 으로 두고도 아무 일이 없었던 것은 무게가
#            아니라 겨눈 곳이 틀렸기 때문이므로, 다시 겨눈 B 에는 무게를 준다.
#
# C = 0.50 -- 기구가 경로를 정한다는 것은 실측된 안정 쪽 논거이고, 다섯 문항 중 모델이
#            유일하게 확신하는 문항이다(뽑힌 등급 확률 0.773, 엔트로피 0.407,
#            0.5 미만 10%). 가점 쪽 최대.
# D = 0.15 -- "넓은 면에 내려놓는다" 에 가점을 주는데, 그 넓은 면은 조리대이고
#            *ToCounter 가 오히려 약한 쪽이다(0.766). D 가 가장 잘 맞는 태스크들에서
#            증거가 반대를 가리키므로 최소로 둔다. 문구는 그대로 두었다 -- 지우는
#            것보다 무게를 내리는 쪽이 변경이 작다.
# E = 0.35 -- "열린 공간을 건너 옮긴다" 는 PnP 두 부류에 모두 걸려 중립 증거이고,
#            모델도 잘 못 답한다(확률 0.407). 가운데.
SIGN = {"A": -1, "B": -1, "C": +1, "D": +1, "E": +1}
WEIGHT = {"A": 0.60, "B": 0.40,
          "C": 0.50, "D": 0.15, "E": 0.35}

GUIDANCE = (
    "You are judging one second of a kitchen robot's motion, to decide how much of it "
    "could be thinned out -- how many of its commanded poses could be dropped, letting "
    "the arm travel further between the ones that remain, without changing the outcome.\n\n"
    "What thinning takes away is the arm's chance to correct itself on the way. That "
    "matters most where the gripper has to close on one exact spot, and it matters most "
    "of all when that spot is hemmed in. It matters least where the arm only pushes or "
    "turns something that is already held in place by its own hinge, rail or mount.\n\n"
    "Judge the moment in front of you, not the task as a whole. One task passes through "
    "both kinds from one second to the next."
)

SCALE = (
    "  5 = it is happening right now -- the picture shows the contact, the grip or\n"
    "      the position the check describes\n"
    "  4 = not yet, but the hand is right up against it, a centimetre or a single\n"
    "      motion away\n"
    "  3 = the hand is heading for it and still some way off\n"
    "  2 = the thing the check is about is there in the picture, but the arm is busy\n"
    "      with something else\n"
    "  1 = there is nothing in this picture the check could be about\n"
)

_AXES = (
    "A) Does this motion need the gripper to CLOSE on one exact spot -- a handle, a rim,\n"
    "   the body of an object it must pick up -- rather than only pushing or turning\n"
    "   something that stays where it is?\n"
    "B) Is that closing spot hemmed in -- reached into a cabinet, a microwave, an oven or\n"
    "   a sink, or lined up with a slot or a handle -- so the gripper has little room\n"
    "   around it as it closes?\n"
    "C) Is the thing being acted on held in place by the furniture or appliance itself\n"
    "   -- a door or drawer on its hinge or rail, a knob, switch, button or lever on its\n"
    "   mount -- so the mechanism, not the arm, decides where it ends up?\n"
    "D) Is the hand letting go, or about to -- the object already resting where it is\n"
    "   meant to end up, so that what remains is only to open and withdraw?\n"
    "E) Is the gripper carrying something through open space -- already holding it,\n"
    "   still well away from wherever it is going?\n")

ASK = ("The measurements above are stated as fact -- do not re-estimate or repeat "
       "them. Answer each check from what the cameras show about the MOMENT in "
       "front of you, read together with those measurements.\n"
       "Answer each check on its own line as \"A) 3\", in order, nothing else "
       "-- one digit from 1 to 5 per check, rating how far that check describes this "
       "moment:\n" + SCALE + "A grade refers only to the check on that line.\n"
       + _AXES + "Answer:")
