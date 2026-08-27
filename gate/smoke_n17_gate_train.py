"""n1.7 공동 파인튜닝 스모크 — 게이트 손실이 실제로 계산되고 역전파되는지.

확인 항목
  ① attach_quant_gate 가 실제 모델에서 동작하는가 (config 경로·차원)
  ② 중간 레이어 탭과 image_mask 가 맞물리는가
  ③ gate_loss 가 나오고 backward 가 되는가
  ④ 그래디언트가 의도한 곳에만 흐르는가 (상위 4층 + 게이트 헤드)
"""
import os, sys, torch
from PIL import Image
import numpy as np

BB = "nvidia/Cosmos-Reason2-2B"
CK = "nvidia/GR00T-N1.7-3B"
GATE_LAYER = int(os.environ.get("GATE_LAYER", "14"))
_AL = os.environ.get("ACTION_LAYER", "")
ACTION_LAYER = int(_AL) if _AL else None

from transformers import AutoProcessor
from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7

print("① 모델 로드")
m = Gr00tN1d7.from_pretrained(CK, torch_dtype=torch.bfloat16).cuda()
TOP = int(os.environ.get("TUNE_TOP", "4"))
m.backbone.set_trainable_parameters(False, False, TOP)   # 상위 TOP 층 unfreeze
# 로드 시점엔 백본 전체가 학습가능이라 trainable_params_fp32 로 fp32 캐스팅된다.
# 그 상태로는 FlashAttention 이 fp32 를 받아 죽는다. 동결을 적용한 뒤 bf16 으로 되돌린다.
m.backbone.to(torch.bfloat16)
n32 = sum(1 for _, q in m.backbone.named_parameters() if q.dtype == torch.float32)
print(f"   fp32 로 남은 백본 텐서 {n32}개 (0 이어야 함)")
gate = m.attach_quant_gate(gate_layer=GATE_LAYER, action_layer=ACTION_LAYER, loss_weight=1.0)
print(f"   게이트 부착 OK — layer {GATE_LAYER}, 헤드 파라미터 "
      f"{sum(p.numel() for p in gate.parameters())/1e6:.2f}M")

print("② 백본 입력 구성")
proc = AutoProcessor.from_pretrained(BB)
imgs = [Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)) for _ in range(3)]
msgs = [{"role": "user", "content": [{"type": "image"}] * 3 +
         [{"type": "text", "text": "pick up the mug from the counter"}]}]
text = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
enc = proc(text=[text], images=imgs, return_tensors="pt")
enc = {k: v.cuda() for k, v in enc.items() if k in
       ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")}
enc["pixel_values"] = enc["pixel_values"].to(torch.bfloat16)
print("   " + "  ".join(f"{k}{tuple(v.shape)}" for k, v in enc.items()))

print("③ 백본 forward + 탭 확인")
out = m.backbone(enc)
h = m.backbone._gate_hidden
im = m.backbone._gate_image_mask
print(f"   최상위 {tuple(out['backbone_features'].shape)}  "
      f"게이트층[{GATE_LAYER}] {tuple(h.shape)}  이미지토큰 {int(im.sum())}개")
assert h.shape[:2] == im.shape, "탭과 마스크 모양 불일치"

print("④ 게이트 손실 + 역전파")
logit = gate(h, im)
y = torch.tensor([[0.7]], device=logit.device)
loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, y)
loss.backward()
print(f"   logit {float(logit):.4f}  loss {float(loss):.4f}")

gh = sum(1 for p in gate.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
top, low, vis = 0, 0, 0
for n, p in m.backbone.named_parameters():
    if p.grad is None or p.grad.abs().sum() == 0:
        continue
    if ".layers." in n:
        li = int(n.split(".layers.")[1].split(".")[0])
        (top if li >= 12 else low).__class__      # noqa
        if li >= 12: top += 1
        else: low += 1
    elif "visual" in n:
        vis += 1
print(f"   그래디언트 있는 텐서 — 게이트헤드 {gh} / 상위층(≥12) {top} / "
      f"하위층(<12) {low} / 비전타워 {vis}")
n_layers = len(m.backbone.model.language_model.layers)
first_trainable = n_layers - TOP
print(f"   unfreeze 범위: 레이어 {first_trainable}~{n_layers-1}, 게이트 탭 {GATE_LAYER}")
if GATE_LAYER < first_trainable:
    print("   [경고] 게이트 탭이 동결 구간이다 — 게이트 손실이 백본을 전혀 못 바꾼다.")
else:
    print("   기대: 게이트헤드 >0, 그리고 게이트 탭 이하 학습가능 층에 그래디언트 >0")
