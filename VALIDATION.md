# AdaptViG port — validation record

## What ran (CPU, this branch)

Primary gate from the brief — numerical parity vs the reference implementation, no weight
download. `tests/test_adaptvig.py` vendors the reference `models/adaptvig.py`
(mmunir127/AdaptViG, Apache-2.0) as the oracle, copies its `state_dict` into the timm model
through `checkpoint_filter_fn` (exercising the `{'state_dict', 'state_dict_ema'}` envelope
unwrap and the `dist_head` → `head_dist` rename, `strict=True` load), and forwards the same
fixed input (`torch.manual_seed(0); x = torch.randn(2, 3, 224, 224)`) with both models in
`eval()` (heads averaged, DeiT-distilled semantics):

| variant | params  | max-abs logit diff (bar: < 1e-4) |
|---------|---------|---------------------------------|
| adaptvig_s | 8.69M | **0.0** (bit-exact) |
| adaptvig_m | 18.06M | **0.0** (bit-exact) |
| adaptvig_b | 26.99M | **0.0** (bit-exact) |

Also ran (same file, plus ad-hoc checks):

- `features_only=True`: 4 maps at strides 4/8/16/32 with channels [32, 64, 128, 256]
  (`adaptvig_s`, 224² input → 56/28/14/7 px). Non-FX path (`feature_cls='getter'` +
  `forward_intermediates`); FX tracing is never used, so the `H.bit_length()` loop in
  `AdaptConv` is fine.
- `reset_classifier(10)`: both `head` and `head_dist` rewired, forward returns (2, 10).
- Envelope preference: `state_dict` wins when both keys exist; `state_dict_ema` is the
  fallback.
- `distilled=False`: `head_dist` keys dropped by the filter, single-head forward OK.
- `num_classes=0`: headless forward returns (1, 768) embedding; `create_optimizer_v2(...,
  layer_decay=0.65)` builds against `group_matcher` (68 param groups).
- `distilled_training`: default train-mode forward returns averaged logits;
  `set_distilled_training(True)` returns the `(logits, logits_dist)` tuple. Gradients reach
  every `AdaptConv.temperature` through the exponential decay gate.

Command: `python -m pytest tests/test_adaptvig.py -v` → 5 passed in ~4 s.

## Weight source

`pretrained_cfg` points at the original author release on the Hugging Face Hub —
`hf_hub_id='SLDGroup/AdaptViG'` with per-variant `hf_hub_filename`
(`AdaptViG_{S,M,B}_model_best.pth`). No timm re-host exists, so `url` is empty. The filter
prefers the top-level `state_dict` key over `state_dict_ema` because the reference
`main.py` tracks its reported best top-1 (`max_accuracy`) on the non-EMA model and saves
`model_best.pth` when that model improves; both keys are stored in the file.

## Deferred (per brief — not blocking)

1. Bit-exact parity against the real downloaded ImageNet-1k checkpoints + a top-1 spot-check.
   Plan: on a machine with network access, `timm.create_model('adaptvig_s', pretrained=True)`
   (and m/b), run `tests/test_models.py`-style forward on a handful of ImageNet val images,
   and compare logits against the reference repo's `main.py --eval` on the same weights; a
   top-1 within ±0.1 of the reported number confirms the envelope-key choice (`state_dict`
   vs `state_dict_ema`) and the head averaging. If the EMA weights match the reported top-1
   instead, flip the preference order in `checkpoint_filter_fn` (one-line change).
2. GPU throughput/latency profiling (paper reports ~8.6M params / 82.0 top-1 for -S) and the
   detection / semantic-segmentation configs (mmcv-based, out of scope for this port).
