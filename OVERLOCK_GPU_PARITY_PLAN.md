# OverLoCK (timm port) — GPU weight-parity plan & implementation self-review

This branch adds `timm/models/overlock.py`: an OverLoCK backbone port (arXiv:2502.20087, CVPR 2025
Oral) registered as `overlock_{xt,t,s,b}`. It is a **draft for human review**; the gate that must
clear before any upstream PR is the GPU weight-parity run described below, which is **deferred and
NOT verified in CI** (or anywhere in this branch).

## 1. GPU weight-parity plan (the REAL gate — human / Colab / `@remyx-ai validate` step)

Goal: prove the pure-PyTorch port loads the released checkpoints cleanly and reproduces the
reference's ImageNet-1k accuracy (or its logits) on GPU.

For each of `overlock_xt`, `overlock_t`, `overlock_s`, `overlock_b`:

1. **Clean `load_state_dict`.** Build via `timm.create_model('overlock_<v>', pretrained=True)`
   (weights: `https://github.com/LMMMEng/OverLoCK/releases/download/v1/overlock_<v>_in1k_224.pth`)
   and assert the load result has no missing/unexpected keys after `checkpoint_filter_fn`
   (which unwraps `state_dict`, strips `module.`/`backbone.` prefixes, and drops `aux_head.*`).
   Any remaining key mismatch is a port bug — stop and fix before proceeding.
2. **Logit parity vs the natten reference (strongest check).** On a GPU with `natten` installed,
   run the reference `LMMMEng/OverLoCK` `models/overlock.py` model with the same checkpoint on a
   fixed batch (e.g. 64 ImageNet-val images, plus a `torch.randn` batch at 224²) and compare
   logits with the timm port: expect `max |Δlogit|` at float-eps scale (the value-aggregation
   path is the identical function; only kernel scheduling differs). If a systematic gap appears,
   suspect the kernel-position ↔ attention-weight layout correspondence in
   `DynamicConvBlock.neighborhood_av` / `apply_rpb` and the border handling
   (`F.pad(..., 'replicate')` of the aggregated valid-region map).
3. **Top-1 sanity.** Validate on ImageNet-1k val (timm `validate.py`) and compare to the
   reference repo's reported numbers within tolerance (XT 82.7 / T 84.2 / S ~85.0 / B 85.1 top-1
   at 224²; see the
   [reference repo results table](https://github.com/LMMMEng/OverLoCK) and
   [arXiv:2502.20087](https://arxiv.org/abs/2502.20087)). A drop of more than ~0.1-0.2 top-1
   indicates a port defect, not noise.
4. **Feature-map parity (detection surface).** For each of the four `features_only` stage outputs,
   compare against the reference mmdet backbone stage outputs (the reference exposes the same
   four reductions via `out_indices` in its mmdet configs).

Scripted sketches for steps 1-2 are ~30 lines each; run them on any CUDA machine or Colab.

## 2. What was implemented (and where)

- **Call site**: new module `timm/models/overlock.py` + the single registration-surface line
  `from .overlock import *` in `timm/models/__init__.py`; tests in `tests/test_models.py`
  (added `overlock` to `FEAT_INTER_FILTERS`, `overlock*` to `EXCLUDE_JIT_FILTERS`, and three
  dedicated smoke tests at the bottom of the file). **No `.github/`, `.ai/`, or workflow/CI files
  were created, modified, or deleted.**
- **Port-with-attribution** from the Apache-2.0 reference https://github.com/LMMMEng/OverLoCK
  (`models/overlock.py` + the `F.unfold` fallback from `models/contmix.py`). GRN originates from
  ConvNeXt-V2, DilatedReparamBlock from UniRepLKNet, as credited in the module docstring.
- **natten-free**: `DynamicConvBlock.neighborhood_av` is the reference's own pure-PyTorch
  `F.unfold` + replicate-pad fallback, always on — no `natten` import, no try/except. This is
  weight-faithful because `na2d_av` is parameter-free neighborhood-attention value aggregation:
  it introduces no trained weights, so replacing it cannot invalidate checkpoint parameters.
- **iGEMM-free**: `DilatedReparamBlock` always uses `nn.Conv2d` (the reference's fallback when
  the UniRepLKNet CUDA op is absent). No new dependencies of any kind; einops was also removed
  (all `rearrange`/`einsum` rewritten as `view`/`permute`/`reshape`/`torch.einsum`).
- **timm idiom**: `build_model_with_cfg` + `generate_default_cfgs` + `@register_model`;
  `pretrained_cfg` `url`s point at the released GitHub-release checkpoints (no `hf_hub_id` — the
  weights are not on the HF hub); `feature_info` + native `forward_intermediates` with
  `feature_cfg=dict(out_indices=(0,1,2,3), feature_cls='getter')` (the two-stream x/ctx dataflow
  cannot be flattened into a sequential extractor); `forward_head`/`reset_classifier`/
  `group_matcher`/`no_weight_decay`/`set_grad_checkpointing`; `checkpoint_filter_fn`.
- Reference `forward_pre_features` / `forward_base_features` / `forward_sub_features` staging is
  preserved; timm's `forward_features` returns the main-branch tensor and classification is split
  into `forward_head`.

## 3. Honesty — what is and is NOT verified here

**Verified in this branch (CPU, no weights):**
- The model constructs and runs for all four variants; logits are finite with the right shape;
  param counts track the published ones (xt 16.97M / t 34.48M / s 57.8M / b 97.43M).
- The four `features_only` stage outputs have the documented channels/reductions
  (4/8/16/32), the main branch only (context branch is auxiliary and not exposed).
- `forward_intermediates` final features are exactly equal to `forward_features` (bitwise, CPU).
- The `F.unfold` neighborhood aggregation was validated against an explicit-gather
  implementation: exact match over the valid interior, with the reference's replicate-padded
  border semantics reproduced verbatim.
- The deployment reparameterization `reparam()` folds all dilated branches and changes eval
  outputs only at float-rounding scale (rel. diff ~1e-6).
- timm's standard model test battery passes for all four variants (forward, backward,
  default-cfg API, features, intermediates, fx-forward/backward) plus the new smoke tests.
- 210 tests for neighboring model families still pass (no regressions from the test-file edits).

**NOT verified here (explicitly deferred to the GPU/human step in §1):**
- Anything about the released checkpoints: key compatibility, load cleanliness, logit parity,
  and top-1 accuracy. **No accuracy claim is made or implied by this branch.** The checkpoints
  were never downloaded here (no network access); key-name parity is derived from the reference
  source only.
- GPU behavior, throughput, and training stability.

**Known deviations from the reference (all deliberate, documented in the module docstring):**
- The training-only deep-supervision `aux_head` (reference `use_ds=True`) is not built, and its
  checkpoint keys are dropped by `checkpoint_filter_fn`. timm models must return a single logits
  tensor; an unused head would also silently miss gradients under timm's test suite. This does
  not affect inference parity (the aux head only contributes a training loss).
- `forward_features` returns the main feature tensor instead of the reference's
  `(x, ctx_cls)` tuple (timm contract); the tuple-returning staging methods are preserved.
- `use_gemm` / `use_sync_bn` switches and the mmengine-style SyncBatchNorm conversion at init
  are dropped (pure-torch `nn.Conv2d` / `nn.BatchNorm2d` only, per the no-iGEMM constraint).
- The reference's `*_reparam` model variants and their `*_reparam.pth` checkpoints are not
  registered; the deployment path itself is available via `OverLoCK(deploy=True)` / `.reparam()`.
- `LayerNorm2d`/`DropPath` come from `timm.layers` (mathematically identical, same parameter
  names for LayerNorm2d); GRN and LayerScale are vendored locally to preserve the reference's
  `gamma`/`beta` and `weight`/`bias` checkpoint key names and exact arithmetic.
- `min(h, w) < kernel_size` inputs use the reference's interpolate-up path (needed for the test
  suite's 96px feature-extraction inputs).

**Intentionally out of scope (not stubs — deliberate non-goals for this bounded contribution):**
- README results-table entries (would require the verified accuracy numbers from §1).
- Detection/segmentation config files or HF-hub weight uploads (upstream follow-ups after
  parity clears).
- The `*_reparam` pretrained variants (see above).

**TorchScript note**: `overlock*` is excluded from the jit test battery because
`DilatedReparamBlock` addresses its dilated branches via dynamically formatted attribute names
(`dil_conv_k5_1`, ...) that must be preserved for checkpoint-key fidelity; TorchScript supports
neither dynamic `__getattr__` nor this reshape flow. FX tracing is supported, with
`DynamicConvBlock` registered as a no-trace leaf (the established timm pattern for
shape-dependent control flow).
