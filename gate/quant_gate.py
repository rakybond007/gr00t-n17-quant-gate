"""n1.7 백본에 quantization confidence 게이트를 붙이는 탭.

(원본: quantization_agent_workspace/vlm_gate/scripts/n17_gate_tap.py)

액션 헤드는 select_layer(최상위) 출력을 쓰고, 게이트는 그보다 아래 레이어를 쓴다.
상위 레이어를 unfreeze 한 채 공동 파인튜닝하면 두 손실이 서로 다른 깊이에 걸려
레이어가 각자 특화된다. 백본 forward 는 정책이 어차피 하므로 추가 비전 연산은 0.

출력 타겟은 다른 모듈과 동일하게 청크당 스칼라 P(quantize) 하나다.
"""
import torch
import torch.nn as nn


def patch_backbone_gate_tap(bb, gate_layer: int, action_layer: int | None = None):
    """forward 를 감싸 두 탭을 각각 지정한다.

    액션 헤드는 hidden_states[action_layer], 게이트는 hidden_states[gate_layer] 를 쓴다.
    둘 사이의 레이어는 위쪽 탭에만 그래디언트가 흐르므로 그쪽 목적에 특화된다.

    hidden_states[k] 는 k 번째 레이어의 출력이다([0] 은 임베딩). 즉 탭 k 는
    레이어 0..k-1 로만 그래디언트를 보낸다.

    image_mask 는 input_ids 로 정해지므로 레이어와 무관하다 — 중간 레이어에도 그대로 쓴다.
    """
    from transformers.feature_extraction_utils import BatchFeature

    n_layers = len(bb.model.language_model.layers)
    action_layer = n_layers if action_layer is None else action_layer
    for name, k in (("action_layer", action_layer), ("gate_layer", gate_layer)):
        if not (1 <= k <= n_layers):
            raise ValueError(
                f"{name}={k} 가 범위를 벗어났다. 이 백본에는 레이어가 {n_layers} 개뿐이다"
                f"(hidden_states 인덱스 1..{n_layers}). 더 위를 쓰려면 백본을 더 큰 "
                f"select_layer 로 다시 로드해 위쪽 레이어를 남겨야 한다 — "
                f"GR00T 는 select_layer 위쪽을 삭제한다."
            )

    orig = bb.forward

    def wrapped(vl_input):
        bb.set_frozen_modules_to_eval_mode()
        keys = ["input_ids", "attention_mask", "pixel_values", "image_grid_thw"]
        vi = {k: vl_input[k] for k in keys}
        out = bb.model(**vi, output_hidden_states=True)
        hs = out.hidden_states
        image_mask = vi["input_ids"] == bb.model.config.image_token_id
        bb._gate_hidden = hs[gate_layer]
        bb._gate_image_mask = image_mask
        return BatchFeature(data={
            "backbone_features": hs[action_layer],
            "backbone_attention_mask": vi["attention_mask"] == 1,
            "image_mask": image_mask,
        })

    bb.forward = wrapped
    bb._gate_tap_orig = orig
    bb._gate_layer = gate_layer
    bb._action_layer = action_layer
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


def attach_from_checkpoint(model, ckpt_dir, gate_layer=None, action_layer=None):
    """체크포인트에 quant_gate.* 가중치가 있으면 게이트를 붙이고 로드한다.

    추론 경로(run_gr00t_server 등)는 attach_quant_gate 를 부르지 않으므로, 학습된
    게이트 가중치가 파일에 있어도 모듈이 없어 그냥 버려진다. 조용히 게이트 없는
    정책이 되므로 알아채기 어렵다 — 여기서 명시적으로 붙이고 몇 개를 로드했는지 찍는다.

    gate_layer 는 학습 때와 같아야 한다. 기본 14 (환경변수 GATE_LAYER 로 덮어쓸 수 있다).
    """
    import glob
    import json
    import os

    import torch
    from safetensors import safe_open

    idx_path = os.path.join(ckpt_dir, "model.safetensors.index.json")
    if os.path.exists(idx_path):
        wmap = json.load(open(idx_path))["weight_map"]
        keys = [k for k in wmap if k.startswith("quant_gate.")]
        files = {wmap[k] for k in keys}
    else:
        keys, files = [], set(os.path.basename(f) for f in glob.glob(f"{ckpt_dir}/*.safetensors"))
        for f in list(files):
            with safe_open(os.path.join(ckpt_dir, f), framework="pt") as h:
                keys += [k for k in h.keys() if k.startswith("quant_gate.")]
    if not keys:
        print("[gate] 체크포인트에 게이트 가중치 없음 — 게이트 없이 서빙", flush=True)
        return None

    gl = int(os.environ.get("GATE_LAYER", gate_layer if gate_layer is not None else 14))
    al = action_layer if action_layer is not None else None
    model.backbone.set_trainable_parameters(tune_llm=False, tune_visual=False,
                                            tune_top_llm_layers=4)
    gate = model.attach_quant_gate(gate_layer=gl, action_layer=al)
    sd = {}
    for f in files:
        with safe_open(os.path.join(ckpt_dir, f), framework="pt") as h:
            for k in h.keys():
                if k.startswith("quant_gate."):
                    sd[k[len("quant_gate."):]] = h.get_tensor(k)
    missing, unexpected = gate.load_state_dict(sd, strict=False)
    gate.to(next(model.parameters()).device).eval()
    print(f"[gate] 가중치 {len(sd)}개 로드 (layer {gl}) · missing {len(missing)} · "
          f"unexpected {len(unexpected)}", flush=True)
    return gate
