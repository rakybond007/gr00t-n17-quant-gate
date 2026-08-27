"""n1.7 백본에 quantization confidence 게이트를 붙이는 탭.

(원본: quantization_agent_workspace/vlm_gate/scripts/n17_gate_tap.py)

액션 헤드는 select_layer(최상위) 출력을 쓰고, 게이트는 그보다 아래 레이어를 쓴다.
상위 레이어를 unfreeze 한 채 공동 파인튜닝하면 두 손실이 서로 다른 깊이에 걸려
레이어가 각자 특화된다. 백본 forward 는 정책이 어차피 하므로 추가 비전 연산은 0.

출력 타겟은 다른 모듈과 동일하게 청크당 스칼라 P(quantize) 하나다.
"""
import torch
import torch.nn as nn


def patch_backbone_gate_tap(bb, gate_layer: int):
    """forward 를 감싸 게이트용 중간 레이어 출력을 함께 남긴다.

    image_mask 는 input_ids 로 정해지므로 레이어와 무관하다 — 같은 마스크를
    중간 레이어에도 그대로 쓸 수 있다.
    """
    orig = bb.forward

    def wrapped(vl_input):
        bb.set_frozen_modules_to_eval_mode()
        keys = ["input_ids", "attention_mask", "pixel_values", "image_grid_thw"]
        vi = {k: vl_input[k] for k in keys}
        out = bb.model(**vi, output_hidden_states=True)
        hs = out.hidden_states
        top = hs[-1]
        bb._gate_hidden = hs[gate_layer]                     # (B, T, D)
        image_mask = vi["input_ids"] == bb.model.config.image_token_id
        attention_mask = vi["attention_mask"] == 1
        bb._gate_image_mask = image_mask
        from transformers.feature_extraction_utils import BatchFeature
        return BatchFeature(data={
            "backbone_features": top,
            "backbone_attention_mask": attention_mask,
            "image_mask": image_mask,
        })

    bb.forward = wrapped
    bb._gate_tap_orig = orig
    return orig


class QuantGateHead(nn.Module):
    """이미지 토큰에만 attention 풀링 → P(quantize).

    masked-mean 은 텍스트 토큰까지 섞어 공간·객체 정보를 잃는다(B 변형이 그랬다).
    여기서는 image_mask 로 이미지 토큰만 남기고 학습되는 질의 하나로 골라낸다.
    """

    def __init__(self, dim, nheads=8, act_dim=0):
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.att = nn.MultiheadAttention(dim, nheads, batch_first=True)
        self.ln = nn.LayerNorm(dim)
        self.act_enc = (nn.Sequential(nn.Linear(act_dim, dim), nn.ReLU(),
                                      nn.Linear(dim, dim)) if act_dim else None)
        din = dim * (2 if act_dim else 1)
        self.head = nn.Sequential(nn.Linear(din, 256), nn.ReLU(),
                                  nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, hidden, image_mask, action=None):
        """hidden:(B,T,D)  image_mask:(B,T) bool  action:(B,A) 또는 None -> (B,1) 로짓"""
        h = self.ln(hidden.float())
        B = h.shape[0]
        # 이미지 토큰이 없는 샘플이 생기지 않도록 안전장치
        km = ~image_mask                       # True = 무시
        km = torch.where(km.all(dim=1, keepdim=True), torch.zeros_like(km), km)
        o, _ = self.att(self.q.expand(B, -1, -1).float(), h, h, key_padding_mask=km)
        f = o[:, 0]
        if self.act_enc is not None and action is not None:
            f = torch.cat([f, self.act_enc(action.float())], dim=1)
        return self.head(f)
