"""RoboCasa Kitchen 모달리티 설정 (n1.7 --modality-config-path 용).

임포트되면 스스로 등록한다 — launch_finetune 이 모듈을 임포트만 하기 때문이다.

키 이름은 데이터셋 meta/modality.json 을 그대로 따르고, 액션 12차원 레이아웃도
그와 일치한다:  0:4 base_motion · 4:5 control_mode · 5:8 EE 위치 · 8:11 EE 회전 · 11:12 그리퍼

action_configs 는 일부러 비워 둔다. 비우면 전 키가 ABSOLUTE / NON_EEF / DEFAULT 로
채워지는데, 이는 액션 값을 변환 없이 그대로 쓴다는 뜻이다. robocasa 액션은 이미
컨트롤러가 직접 소비하는 델타 명령이라 이게 맞고, N1.5 도 같은 방식이었다.
RELATIVE 로 두면 현재 상태 기준 델타로 다시 계산하려 들어 이중 변환이 된다.
"""
from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ModalityConfig

robocasa_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["left_view", "right_view", "wrist_view"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "end_effector_position_relative",
            "end_effector_rotation_relative",
            "gripper_qpos",
            "base_position",
            "base_rotation",
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(16)),          # 청크 지평 16 — 라벨과 같은 단위
        modality_keys=[
            "end_effector_position",
            "end_effector_rotation",
            "gripper_close",
            "base_motion",
            "control_mode",
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.action.task_description"],
    ),
}

register_modality_config(robocasa_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
