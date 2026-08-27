"""학습 배치에 quantization confidence 라벨을 실어 보낸다.

콜레이터가 샘플 딕셔너리의 모든 키를 스택하므로, get_datapoint 가 gate_label 을
넣어주면 모델 forward 까지 그대로 전달된다. 셔딩된 데이터셋을 다시 굽지 않아도 된다.

라벨은 VLM 교사가 만든 parquet (episode_index, frame_index, p_yes) 이다.
"""
import os
import numpy as np
import torch


class GateLabelLookup:
    """(에피소드, 프레임) -> p_yes. 라벨이 없는 스텝은 마스크로 걸러진다."""

    def __init__(self, parquet_path: str):
        import pandas as pd
        df = pd.read_parquet(parquet_path, columns=["episode_index", "frame_index", "p_yes"])
        self.tbl = {(int(e), int(f)): float(y)
                    for e, f, y in zip(df.episode_index, df.frame_index, df.p_yes)}
        self.n = len(self.tbl)
        print(f"[gate] 라벨 {self.n}개 로드: {os.path.basename(parquet_path)}", flush=True)

    def get(self, episode_index: int, step_index: int):
        return self.tbl.get((int(episode_index), int(step_index)))


def patch_dataset_gate_labels(dataset, lookup: GateLabelLookup):
    """get_datapoint 를 감싸 gate_label / gate_valid 를 샘플에 추가한다."""
    orig = dataset.get_datapoint

    def wrapped(episode_data, step_index):
        out = orig(episode_data, step_index)
        ep = None
        try:
            if "episode_index" in episode_data:
                ep = int(np.asarray(episode_data["episode_index"])[0])
        except Exception:
            ep = None
        y = lookup.get(ep, step_index) if ep is not None else None
        # 라벨이 없는 스텝은 0.5 로 채우고 gate_valid=0 으로 손실에서 제외한다
        out["gate_label"] = torch.tensor(0.5 if y is None else y, dtype=torch.float32)
        out["gate_valid"] = torch.tensor(0.0 if y is None else 1.0, dtype=torch.float32)
        return out

    dataset.get_datapoint = wrapped
    dataset._gate_lookup = lookup
    return dataset
