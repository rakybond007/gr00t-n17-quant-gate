"""libero v2 문항의 부호·가중치. `prompts/libero_v2.txt` 와 짝이다.

robocasa 의 `phase9_checks.py` 와 같은 자리다. **벤치마크마다 다르므로 이 값을
쓰는 쪽에 박아 두지 않는다** -- 한 번 박아 두면 robocasa 라벨을 libero 가중치로
계산하는 일이 조용히 일어난다.

부호는 어느 풀에서 뽑혔는지가 정하고, 가중치는 그 문항이 덮는 태스크 수를 각
변의 합이 1 이 되게 정규화한 것이다(`PROMPT_METHOD.md` 4·5단계). 도출 과정은
`analysis/libero_prompt_derivation.md`, 재현은 `derive_libero_questions.py`.

    A SMALL_TARGET  감점  좁고 정해진 자리에 반듯이 놓는가 -- 접시, 버너, 걸이
    B CROWDED_PICK  감점  집을 것이 얹혀 있거나 사이에 끼어 있는가
    C CATCHER       가점  받아 주는 통에 넣는가
    D WIDE_SURFACE  가점  물체보다 훨씬 큰 면에 놓는가
    E NO_HOLD       가점  붙들 필요 없는 일인가

검증(571장, 위험 9 · 안정 12 태스크): 형식 100%, 부호 다섯 다 통과
(A +0.16 · B +0.09 · C −1.11 · D −0.19 · E −0.32), 문항 상관 최대 0.19.
"""
NGRADE = 5

SIGN = {"A": -1, "B": -1, "C": +1, "D": +1, "E": +1}

# 덮는 태스크 수: 감점 7·4 (합 11) · 가점 8·4·3 (합 15)
WEIGHT = {"A": 7 / 11, "B": 4 / 11,
          "C": 8 / 15, "D": 4 / 15, "E": 3 / 15}

NAME = {"A": "SMALL_TARGET", "B": "CROWDED_PICK", "C": "CATCHER",
        "D": "WIDE_SURFACE", "E": "NO_HOLD"}
