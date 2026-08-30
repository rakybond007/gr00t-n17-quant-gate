"""학습 배치에 quantization confidence 라벨을 실어 보낸다.

라벨은 VLM 교사가 만든 parquet (episode_index, frame_index, p_yes) 이고,
`(에피소드, 스텝)` 으로 조인한다. 라벨이 없는 스텝은 gate_valid=0 으로 손실에서
빠진다 — VLA 데이터로더는 모든 시점을 돌지만 라벨은 일정 간격으로만 있기 때문이다.

에피소드 번호를 어디서 얻는가
-----------------------------
`get_datapoint(episode_data, step_index)` 이 받는 DataFrame 에는 episode_index 가
**없다**. LeRobotEpisodeLoader._load_parquet_data 가 `pd.DataFrame()` 을 새로 만들어
모달리티 컬럼만 채우기 때문이다. 원본 parquet 에는 있지만 여기까지 오지 않는다.

그래서 이전 판은 항상 ep=None -> gate_valid=0 -> gate_loss 가 매 스텝 정확히 0 이
되었고, 게이트 헤드는 1만 스텝을 돌고도 초기화 상태 그대로였다(체크포인트 6000 과
10000 사이 파라미터 변화량 0.000e+00, 같은 구간 백본은 3~7e-3 이동).

이번 판은 에피소드 번호를 shard 순회에서 직접 가져온다. get_shard 는 ep_idx 를
알고 있으므로, 그 지점에서 라벨을 붙이면 추측이 필요 없다.
"""
import os

import numpy as np
import torch


class GateLabelLookup:
    """(에피소드, 프레임) -> p_yes."""

    def __init__(self, parquet_path: str):
        import pandas as pd
        df = pd.read_parquet(parquet_path, columns=["episode_index", "frame_index", "p_yes"])
        self.tbl = {(int(e), int(f)): float(y)
                    for e, f, y in zip(df.episode_index, df.frame_index, df.p_yes)}
        self.n = len(self.tbl)
        self.episodes = {k[0] for k in self.tbl}
        print(f"[gate] 라벨 {self.n}개 로드: {os.path.basename(parquet_path)} "
              f"({len(self.episodes)} 에피소드)", flush=True)

    def get(self, episode_index, step_index):
        if episode_index is None:
            return None
        return self.tbl.get((int(episode_index), int(step_index)))


def _attach(sample, y):
    # 라벨이 없는 스텝은 0.5 로 채우되 gate_valid=0 이라 손실에서 빠진다.
    sample["gate_label"] = torch.tensor(0.5 if y is None else y, dtype=torch.float32)
    sample["gate_valid"] = torch.tensor(0.0 if y is None else 1.0, dtype=torch.float32)
    return sample


def patch_dataset_gate_labels(dataset, lookup: GateLabelLookup):
    """샤드 순회를 감싸 gate_label / gate_valid 를 각 샘플에 붙인다.

    get_datapoint 가 아니라 get_shard 를 감싸는 이유는 위 주석에 있다 — 에피소드
    번호는 여기에만 있다.
    """
    orig_shard = getattr(dataset, "get_shard", None)
    orig_dp = getattr(dataset, "get_datapoint", None)
    if orig_shard is None or orig_dp is None:
        raise AttributeError(
            f"{type(dataset).__name__} 에 get_shard/get_datapoint 가 없다 — "
            "데이터셋 구조가 바뀌었으면 이 래퍼도 고쳐야 한다")

    stats = {"seen": 0, "valid": 0}

    def wrapped_shard(idx):
        out = []
        for ep_idx, step_indices in dataset.sharded_episodes[idx]:
            episode_data = dataset.episode_loader[ep_idx]
            for step_index in step_indices:
                y = lookup.get(ep_idx, step_index)
                stats["seen"] += 1
                stats["valid"] += (y is not None)
                out.append(_attach(orig_dp(episode_data, step_index), y))
        return out

    dataset.get_shard = wrapped_shard
    dataset._gate_lookup = lookup
    dataset._gate_stats = stats
    return dataset


def verify_gate_labels(dataset, lookup: GateLabelLookup, n_shards: int = 2):
    """붙었는지 실제로 한 샤드를 뽑아 확인한다.

    이전 판이 조용히 실패한 것은 배선을 검증한 적이 없기 때문이다. 스모크는
    게이트 헤드에 직접 손실을 걸어 backward 했을 뿐, 라벨이 forward 까지 오는지는
    보지 않았다. 그래서 여기서는 실제 샤드를 뽑아, 키가 있는지와 유효 라벨이
    하나라도 있는지를 본다. 없으면 학습을 시작하지 않는다.
    """
    total, valid = 0, 0
    for i in range(min(n_shards, getattr(dataset, "num_shards", 1) or 1)):
        for s in dataset.get_shard(i):
            if "gate_label" not in s or "gate_valid" not in s:
                raise RuntimeError(
                    "gate_label 이 샘플에 없다 — 래퍼가 실제 순회 경로를 못 잡았다")
            total += 1
            valid += float(s["gate_valid"]) > 0.5
    if total == 0:
        raise RuntimeError("샤드에서 샘플을 하나도 못 얻었다")
    if valid == 0:
        raise RuntimeError(
            f"{total}개 샘플 전부 gate_valid=0 이다. 라벨이 하나도 조인되지 않았으므로 "
            "gate_loss 가 항상 0 이 되고 게이트는 학습되지 않는다. "
            "라벨 parquet 의 episode_index 가 이 데이터셋의 에피소드 번호와 같은지 확인할 것.")
    print(f"[gate] 배선 확인 — 샘플 {total}개 중 {valid}개에 라벨 "
          f"({valid / total:.1%})", flush=True)
    return valid / total
