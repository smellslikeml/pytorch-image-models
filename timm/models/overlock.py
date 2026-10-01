"""OverLoCK: An Overview-first-Look-Closely-next ConvNet with Context-Mixing Dynamic Kernels

Paper: `OverLoCK: An Overview-first-Look-Closely-next ConvNet with Context-Mixing Dynamic Kernels`
- https://arxiv.org/abs/2502.20087 (CVPR 2025 Oral)

Reference implementation: https://github.com/LMMMEng/OverLoCK (Apache-2.0). This is a port of the
classification backbone from the reference `models/overlock.py`, with the dynamic-kernel value
aggregation taken from the pure-PyTorch `F.unfold` fallback in the reference `models/contmix.py`.

This port is intentionally dependency-free beyond torch/timm:
* **natten-free** - the reference calls `natten.functional.na2d_av` for neighborhood-attention value
  aggregation. `na2d_av` is parameter-free (no weights), and the reference ships an `F.unfold` +
  `F.pad(..., 'replicate')` fallback computing the identical function. This port uses only that
  pure-PyTorch path (no natten import, no try/except), so released checkpoints transfer faithfully.
* **iGEMM-free** - the reference `DilatedReparamBlock` optionally uses the UniRepLKNet iGEMM CUDA
  depthwise-conv op and falls back to `nn.Conv2d` when absent. Only the `nn.Conv2d` path is kept.
* **einops-free** - the reference einops `rearrange` / `einsum` calls are rewritten as
  `view` / `permute` / `reshape` / `torch.einsum`.

Sub-module provenance (as in the reference): GRN originates from ConvNeXt-V2
(https://arxiv.org/abs/2301.00808), DilatedReparamBlock from UniRepLKNet
(https://github.com/AILab-CVC/UniRepLKNet), the relative-position-bias layout follows natten.

Deviations from the reference, required by the timm model API:
* `forward_features()` returns the final (main-branch) feature tensor, and classification is split
  into `forward_head()`; the reference's `forward_pre_features` / `forward_base_features` /
  `forward_sub_features` staging is preserved verbatim.
* The training-only deep-supervision `aux_head` (reference `use_ds=True`) is not built; timm models
  return a single logits tensor and unused head parameters would silently miss gradients.
  `checkpoint_filter_fn` drops `aux_head.*` keys, which do not affect inference.
* The reference `use_gemm` / `use_sync_bn` switches and the mmengine-style SyncBatchNorm
  conversion at init are dropped (see iGEMM-free note above); `deploy` reparameterization and
  `OverLoCK.reparam()` (structure-equivalent deployment folding of the dilated branches) are kept.

Weight-parity of this port against the released checkpoints is deferred to a GPU validation run
(described in OVERLOCK_GPU_PARITY_PLAN.md); it is not verified here.

Modifications and timm integration Copyright 2026, originally from LMMMEng/OverLoCK (Apache-2.0)
"""
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import (
    DropPath,
    LayerNorm2d,
    SelectAdaptivePool2d,
    calculate_drop_path_rates,
    get_device_dtype,
)

from ._builder import build_model_with_cfg
from ._features import feature_take_indices
from ._features_fx import register_notrace_module
from ._registry import generate_default_cfgs, register_model

__all__ = ['OverLoCK']


def fuse_bn(conv: nn.Module, bn: nn.BatchNorm2d) -> Tuple[torch.Tensor, torch.Tensor]:
    conv_bias = 0 if conv.bias is None else conv.bias
    std = (bn.running_var + bn.eps).sqrt()
    return (
        conv.weight * (bn.weight / std).reshape(-1, 1, 1, 1),
        bn.bias + (conv_bias - bn.running_mean) * bn.weight / std,
    )


def convert_dilated_to_nondilated(kernel: torch.Tensor, dilate_rate: int) -> torch.Tensor:
    identity_kernel = torch.ones((1, 1, 1, 1), device=kernel.device, dtype=kernel.dtype)
    if kernel.size(1) == 1:
        #   This is a DW kernel
        return F.conv_transpose2d(kernel, identity_kernel, stride=dilate_rate)
    #   This is a dense or group-wise (but not DW) kernel
    slices = []
    for i in range(kernel.size(1)):
        dilated = F.conv_transpose2d(kernel[:, i:i + 1, :, :], identity_kernel, stride=dilate_rate)
        slices.append(dilated)
    return torch.cat(slices, dim=1)


def merge_dilated_into_large_kernel(
        large_kernel: torch.Tensor,
        dilated_kernel: torch.Tensor,
        dilate_r: int,
) -> torch.Tensor:
    large_k = large_kernel.size(2)
    dilated_k = dilated_kernel.size(2)
    equivalent_kernel_size = dilate_r * (dilated_k - 1) + 1
    equivalent_kernel = convert_dilated_to_nondilated(dilated_kernel, dilate_r)
    rows_to_pad = large_k // 2 - equivalent_kernel_size // 2
    return large_kernel + F.pad(equivalent_kernel, [rows_to_pad] * 4)


class SEModule(nn.Module):
    def __init__(self, dim: int, red: int = 8, inner_act: Type[nn.Module] = nn.GELU,
                 out_act: Type[nn.Module] = nn.Sigmoid):
        super().__init__()
        inner_dim = max(16, dim // red)
        self.proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, inner_dim, kernel_size=1),
            inner_act(),
            nn.Conv2d(inner_dim, dim, kernel_size=1),
            out_act(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.proj(x)


class LayerScale(nn.Module):
    """Depthwise-conv layer scale, with `weight` of shape (dim, 1, 1, 1) and per-channel `bias`."""

    def __init__(self, dim: int, init_value: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, 1, 1, 1) * init_value, requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(dim), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, weight=self.weight, bias=self.bias, groups=x.shape[1])


class GRN(nn.Module):
    """GRN (Global Response Normalization) with reference parameter names (`gamma` / `beta`).

    Originally proposed in ConvNeXt V2 (https://arxiv.org/abs/2301.00808). Equivalent to timm's
    `timm.layers.GlobalResponseNorm`, kept in reference form so checkpoint keys (`gamma`, `beta`)
    and the exact reference arithmetic transfer unchanged. Inputs are (N, C, H, W).
    """

    def __init__(self, dim: int, use_bias: bool = True):
        super().__init__()
        self.use_bias = use_bias
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        if self.use_bias:
            self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = torch.norm(x, p=2, dim=(-1, -2), keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)
        if self.use_bias:
            return (self.gamma * nx + 1) * x + self.beta
        return (self.gamma * nx + 1) * x


class DilatedReparamBlock(nn.Module):
    """Dilated Reparam Block proposed in UniRepLKNet (https://github.com/AILab-CVC/UniRepLKNet).

    The optional UniRepLKNet iGEMM CUDA large-kernel implementation is deliberately NOT used;
    `lk_origin` is always an `nn.Conv2d` (the reference's fallback when iGEMM is unavailable).
    Inputs are (N, C, H, W).
    """

    def __init__(self, channels: int, kernel_size: int, deploy: bool = False):
        super().__init__()
        self.lk_origin = nn.Conv2d(
            channels, channels, kernel_size, stride=1,
            padding=kernel_size // 2, dilation=1, groups=channels, bias=deploy,
        )

        #   Default settings. We did not tune them carefully. Different settings may work better.
        if kernel_size == 19:
            self.kernel_sizes = [5, 7, 9, 9, 3, 3, 3]
            self.dilates = [1, 1, 1, 2, 4, 5, 7]
        elif kernel_size == 17:
            self.kernel_sizes = [5, 7, 9, 3, 3, 3]
            self.dilates = [1, 1, 2, 4, 5, 7]
        elif kernel_size == 15:
            self.kernel_sizes = [5, 7, 7, 3, 3, 3]
            self.dilates = [1, 1, 2, 3, 5, 7]
        elif kernel_size == 13:
            self.kernel_sizes = [5, 7, 7, 3, 3, 3]
            self.dilates = [1, 1, 2, 3, 4, 5]
        elif kernel_size == 11:
            self.kernel_sizes = [5, 7, 5, 3, 3, 3]
            self.dilates = [1, 1, 2, 3, 4, 5]
        elif kernel_size == 9:
            self.kernel_sizes = [5, 7, 5, 3, 3]
            self.dilates = [1, 1, 2, 3, 4]
        elif kernel_size == 7:
            self.kernel_sizes = [5, 3, 3, 3]
            self.dilates = [1, 1, 2, 3]
        elif kernel_size == 5:
            self.kernel_sizes = [3, 3]
            self.dilates = [1, 2]
        else:
            raise ValueError('Dilated Reparam Block requires kernel_size >= 5')

        if not deploy:
            self.origin_bn = nn.BatchNorm2d(channels)
            for k, r in zip(self.kernel_sizes, self.dilates):
                self.__setattr__(
                    'dil_conv_k{}_{}'.format(k, r),
                    nn.Conv2d(
                        in_channels=channels, out_channels=channels, kernel_size=k, stride=1,
                        padding=(r * (k - 1) + 1) // 2, dilation=r, groups=channels, bias=False),
                )
                self.__setattr__('dil_bn_k{}_{}'.format(k, r), nn.BatchNorm2d(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not hasattr(self, 'origin_bn'):  # deploy mode
            return self.lk_origin(x)
        out = self.origin_bn(self.lk_origin(x))
        for k, r in zip(self.kernel_sizes, self.dilates):
            conv = self.__getattr__('dil_conv_k{}_{}'.format(k, r))
            bn = self.__getattr__('dil_bn_k{}_{}'.format(k, r))
            out = out + bn(conv(x))
        return out

    def merge_dilated_branches(self) -> None:
        if hasattr(self, 'origin_bn'):
            origin_k, origin_b = fuse_bn(self.lk_origin, self.origin_bn)
            for k, r in zip(self.kernel_sizes, self.dilates):
                conv = self.__getattr__('dil_conv_k{}_{}'.format(k, r))
                bn = self.__getattr__('dil_bn_k{}_{}'.format(k, r))
                branch_k, branch_b = fuse_bn(conv, bn)
                origin_k = merge_dilated_into_large_kernel(origin_k, branch_k, r)
                origin_b += branch_b
            merged_conv = nn.Conv2d(
                origin_k.size(0), origin_k.size(0), origin_k.size(2), stride=1,
                padding=origin_k.size(2) // 2, dilation=1, groups=origin_k.size(0), bias=True,
            )
            merged_conv.weight.data = origin_k
            merged_conv.bias.data = origin_b
            self.lk_origin = merged_conv
            self.__delattr__('origin_bn')
            for k, r in zip(self.kernel_sizes, self.dilates):
                self.__delattr__('dil_conv_k{}_{}'.format(k, r))
                self.__delattr__('dil_bn_k{}_{}'.format(k, r))


class CTXDownsample(nn.Module):
    def __init__(self, dim: int, h_dim: int):
        super().__init__()
        self.x_proj = nn.Sequential(
            nn.Conv2d(dim, h_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(h_dim),
        )
        self.h_proj = nn.Sequential(
            nn.Conv2d(h_dim // 4, h_dim // 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(h_dim // 4),
        )

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.x_proj(x)
        ctx = self.h_proj(ctx)
        return x, ctx


class ResDWConv(nn.Conv2d):
    """Depthwise conv with residual connection."""

    def __init__(self, dim: int, kernel_size: int = 3):
        super().__init__(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # F.conv2d instead of super().forward() (padding_mode is 'zeros', so this is identical)
        return x + F.conv2d(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class RepConvBlock(nn.Module):
    """Stage 1-4 block: dilated-reparam large-kernel conv + SE + ConvMLP, with residual scaling."""

    def __init__(
            self,
            dim: int = 64,
            kernel_size: int = 7,
            mlp_ratio: float = 4.,
            ls_init_value: Optional[float] = None,
            res_scale: bool = False,
            drop_path: float = 0.,
            norm_layer: Type[nn.Module] = LayerNorm2d,
            deploy: bool = False,
            use_checkpoint: bool = False,
    ):
        super().__init__()
        self.res_scale = res_scale
        self.use_checkpoint = use_checkpoint

        mlp_dim = int(dim * mlp_ratio)

        self.dwconv = ResDWConv(dim, kernel_size=3)

        self.proj = nn.Sequential(
            norm_layer(dim),
            DilatedReparamBlock(dim, kernel_size=kernel_size, deploy=deploy),
            nn.BatchNorm2d(dim),
            SEModule(dim),
            nn.Conv2d(dim, mlp_dim, kernel_size=1),
            nn.GELU(),
            ResDWConv(mlp_dim, kernel_size=3),
            GRN(mlp_dim),
            nn.Conv2d(mlp_dim, dim, kernel_size=1),
            DropPath(drop_path) if drop_path > 0. else nn.Identity(),
        )

        self.ls = LayerScale(dim, init_value=ls_init_value) if ls_init_value is not None else nn.Identity()

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dwconv(x)

        if self.res_scale:
            x = self.ls(x) + self.proj(x)
        else:
            drop_path = self.proj[-1]
            x = x + drop_path(self.ls(self.proj[:-1](x)))

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and x.requires_grad and not torch.jit.is_scripting():
            x = torch.utils.checkpoint.checkpoint(self.forward_features, x, use_reentrant=False)
        else:
            x = self.forward_features(x)
        return x


@register_notrace_module
class DynamicConvBlock(nn.Module):
    """ContMix context-mixing dynamic kernel block (reference `DynamicConvBlock`), natten-free.

    Dynamic per-position kernels are generated by correlating local queries with a context summary
    of the top-down `ctx` branch, then used to aggregate values over two neighborhood sizes
    (`smk_size`, `kernel_size`). The neighborhood aggregation is the pure-PyTorch `F.unfold` path
    from the reference `models/contmix.py` fallback - numerically identical to
    `natten.functional.na2d_av`, which is parameter-free, so weights transfer unchanged.
    """

    def __init__(
            self,
            dim: int = 64,
            ctx_dim: int = 32,
            kernel_size: int = 7,
            smk_size: int = 5,
            num_heads: int = 2,
            mlp_ratio: float = 4.,
            ls_init_value: Optional[float] = None,
            res_scale: bool = False,
            drop_path: float = 0.,
            norm_layer: Type[nn.Module] = LayerNorm2d,
            is_first: bool = False,
            is_last: bool = False,
            deploy: bool = False,
            use_checkpoint: bool = False,
            **kwargs: Any,
    ):
        super().__init__()

        ctx_dim = ctx_dim // 4
        out_dim = dim + ctx_dim
        mlp_dim = int(dim * mlp_ratio)
        self.kernel_size = kernel_size
        self.res_scale = res_scale
        self.smk_size = smk_size
        self.num_heads = num_heads * 2
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5
        self.is_first = is_first
        self.is_last = is_last
        self.use_checkpoint = use_checkpoint

        if not is_first:
            self.x_scale = LayerScale(ctx_dim, init_value=1)
            self.h_scale = LayerScale(ctx_dim, init_value=1)

        self.dwconv1 = ResDWConv(out_dim, kernel_size=3)
        self.norm1 = norm_layer(out_dim)

        self.fusion = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, groups=out_dim),
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
            nn.Conv2d(out_dim, dim, kernel_size=1),
            GRN(dim),
        )

        self.weight_query = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim // 2),
        )

        self.weight_key = nn.Sequential(
            nn.AdaptiveAvgPool2d(7),
            nn.Conv2d(ctx_dim, dim // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim // 2),
        )

        self.weight_proj = nn.Conv2d(49, kernel_size ** 2 + smk_size ** 2, kernel_size=1)

        self.dyconv_proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
        )

        self.lepe = nn.Sequential(
            DilatedReparamBlock(dim, kernel_size=kernel_size, deploy=deploy),
            nn.BatchNorm2d(dim),
        )

        self.se_layer = SEModule(dim)

        self.gate = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(),
        )

        self.proj = nn.Sequential(
            nn.BatchNorm2d(dim),
            nn.Conv2d(dim, out_dim, kernel_size=1),
        )

        self.dwconv2 = ResDWConv(out_dim, kernel_size=3)
        self.norm2 = norm_layer(out_dim)

        self.mlp = nn.Sequential(
            nn.Conv2d(out_dim, mlp_dim, kernel_size=1),
            nn.GELU(),
            ResDWConv(mlp_dim, kernel_size=3),
            GRN(mlp_dim),
            nn.Conv2d(mlp_dim, out_dim, kernel_size=1),
        )

        self.ls1 = LayerScale(out_dim, init_value=ls_init_value) if ls_init_value is not None else nn.Identity()
        self.ls2 = LayerScale(out_dim, init_value=ls_init_value) if ls_init_value is not None else nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.get_rpb()

    def get_rpb(self) -> None:
        self.rpb_size1 = 2 * self.smk_size - 1
        self.rpb1 = nn.Parameter(torch.empty(self.num_heads, self.rpb_size1, self.rpb_size1))
        self.rpb_size2 = 2 * self.kernel_size - 1
        self.rpb2 = nn.Parameter(torch.empty(self.num_heads, self.rpb_size2, self.rpb_size2))
        nn.init.zeros_(self.rpb1)
        nn.init.zeros_(self.rpb2)

    @torch.no_grad()
    def generate_idx(self, kernel_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rpb_size = 2 * kernel_size - 1
        idx_h = torch.arange(0, kernel_size)
        idx_w = torch.arange(0, kernel_size)
        idx_k = ((idx_h.unsqueeze(-1) * rpb_size) + idx_w).view(-1)
        return idx_h, idx_w, idx_k

    def apply_rpb(
            self,
            attn: torch.Tensor,
            rpb: torch.Tensor,
            height: int,
            width: int,
            kernel_size: int,
            idx_h: torch.Tensor,
            idx_w: torch.Tensor,
            idx_k: torch.Tensor,
    ) -> torch.Tensor:
        """Relative position bias, layout borrowed from natten (https://tinyurl.com/mrbub4t3)."""
        num_repeat_h = torch.ones(kernel_size, dtype=torch.long)
        num_repeat_w = torch.ones(kernel_size, dtype=torch.long)
        num_repeat_h[kernel_size // 2] = height - (kernel_size - 1)
        num_repeat_w[kernel_size // 2] = width - (kernel_size - 1)
        bias_hw = (
            idx_h.repeat_interleave(num_repeat_h).unsqueeze(-1) * (2 * kernel_size - 1)
        ) + idx_w.repeat_interleave(num_repeat_w)
        bias_idx = bias_hw.unsqueeze(-1) + idx_k
        bias_idx = bias_idx.reshape(-1, int(kernel_size ** 2))
        bias_idx = torch.flip(bias_idx, [0])
        rpb = torch.flatten(rpb, 1, 2).index_select(1, bias_idx.reshape(-1))
        rpb = rpb.reshape(
            self.num_heads, height, width, int(kernel_size ** 2)
        ).unsqueeze(0)
        return attn + rpb

    @staticmethod
    def neighborhood_av(attn: torch.Tensor, value: torch.Tensor, kernel_size: int) -> torch.Tensor:
        """Pure-PyTorch `na2d_av` equivalent (reference `models/contmix.py` fallback).

        Args:
            attn: (num_batch, num_heads, height, width, kernel_size ** 2) neighborhood weights.
            value: (num_batch, num_heads, height, width, head_dim) values to aggregate.
            kernel_size: neighborhood size.

        Returns:
            (num_batch, num_heads, height, width, head_dim) aggregated values.

        `F.unfold` produces the (channel-major, kernel-position-minor) valid-region patches; the
        replicate pad of the aggregated map restores the input resolution at the borders. This
        matches natten's value aggregation exactly while introducing no parameters.
        """
        num_batch, num_heads, height, width, _ = attn.shape
        head_dim = value.shape[-1]
        pad = kernel_size // 2

        # einops 'b g h w c -> b (g c) h w'
        v = value.permute(0, 1, 4, 2, 3).reshape(num_batch, num_heads * head_dim, height, width)
        v = F.unfold(v, kernel_size=kernel_size).reshape(
            num_batch, -1, height - 2 * pad, width - 2 * pad)
        v = F.pad(v, (pad, pad, pad, pad), mode='replicate')
        # einops 'b (g c k) h w -> b g c h w k'
        v = v.view(num_batch, num_heads, head_dim, kernel_size ** 2, height, width)
        v = v.permute(0, 1, 2, 4, 5, 3)
        return torch.einsum('bghwk,bgchwk->bghwc', attn, v)

    def _forward_inner(
            self, x: torch.Tensor, h_x: torch.Tensor, h_r: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        b, c, h, w = x.shape
        input_resolution = (h, w)
        c_h = h_x.shape[1]

        if not self.is_first:
            h_x = self.x_scale(h_x) + self.h_scale(h_r)

        x_f = torch.cat([x, h_x], dim=1)
        x_f = self.dwconv1(x_f)
        identity = x_f
        x_f = self.norm1(x_f)
        x = self.fusion(x_f)
        gate = self.gate(x)
        lepe = self.lepe(x)

        is_pad = False
        if min(h, w) < self.kernel_size:
            is_pad = True
            if h < w:
                size = (self.kernel_size, int(self.kernel_size / h * w))
            else:
                size = (int(self.kernel_size / w * h), self.kernel_size)
            x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
            x_f = F.interpolate(x_f, size=size, mode='bilinear', align_corners=False)
            h, w = size

        query, key = torch.split(x_f, split_size_or_sections=[c, c_h], dim=1)
        query = self.weight_query(query) * self.scale
        key = self.weight_key(key)

        # einops 'b (g c) h w -> b g c (h w)' - note the pooled key keeps its own 7x7 positions
        query = query.reshape(b, self.num_heads, -1, h * w)
        key = key.reshape(b, self.num_heads, -1, key.shape[-2] * key.shape[-1])
        weight = torch.einsum('bgcn,bgcl->bgnl', query, key)
        # einops 'b g n l -> b l g n', then 'b l g (h w) -> b g h w l'
        weight = weight.permute(0, 3, 1, 2).contiguous()
        weight = self.weight_proj(weight)
        weight = weight.reshape(b, weight.shape[1], self.num_heads, h, w).permute(0, 2, 3, 4, 1)

        attn1, attn2 = torch.split(
            weight, split_size_or_sections=[self.smk_size ** 2, self.kernel_size ** 2], dim=-1)
        attn1 = self.apply_rpb(attn1, self.rpb1, h, w, self.smk_size, *self._rpb_idx(self.smk_size))
        attn2 = self.apply_rpb(attn2, self.rpb2, h, w, self.kernel_size, *self._rpb_idx(self.kernel_size))
        attn1 = torch.softmax(attn1, dim=-1)
        attn2 = torch.softmax(attn2, dim=-1)

        # einops 'b (m g c) h w -> m b g h w c'
        value = x.reshape(b, 2, self.num_heads, -1, h, w).permute(1, 0, 2, 4, 5, 3)

        x1 = self.neighborhood_av(attn1, value[0], self.smk_size)
        x2 = self.neighborhood_av(attn2, value[1], self.kernel_size)

        x = torch.cat([x1, x2], dim=1)
        # einops 'b g h w c -> b (g c) h w'
        x = x.permute(0, 1, 4, 2, 3).reshape(b, -1, h, w)

        if is_pad:
            x = F.adaptive_avg_pool2d(x, input_resolution)

        x = self.dyconv_proj(x)

        x = x + lepe
        x = self.se_layer(x)

        x = gate * x
        x = self.proj(x)

        if self.res_scale:
            x = self.ls1(identity) + self.drop_path(x)
        else:
            x = identity + self.drop_path(self.ls1(x))

        x = self.dwconv2(x)

        if self.res_scale:
            x = self.ls2(x) + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.ls2(self.mlp(self.norm2(x))))

        if self.is_last:
            return x, None
        l_x, h_x = torch.split(x, split_size_or_sections=[c, c_h], dim=1)
        return l_x, h_x

    def _rpb_idx(self, kernel_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reference `generate_idx`, with indices moved to the parameter's device."""
        idx_h, idx_w, idx_k = self.generate_idx(kernel_size)
        return idx_h.to(self.rpb1.device), idx_w.to(self.rpb1.device), idx_k.to(self.rpb1.device)

    def forward(
            self, x: torch.Tensor, h_x: torch.Tensor, h_r: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.use_checkpoint and x.requires_grad and not torch.jit.is_scripting():
            out = torch.utils.checkpoint.checkpoint(self._forward_inner, x, h_x, h_r, use_reentrant=False)
        else:
            out = self._forward_inner(x, h_x, h_r)
        return out


def _stem(in_chans: int, embed_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_chans, embed_dim // 2, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(embed_dim // 2),
        nn.GELU(),
        nn.Conv2d(embed_dim // 2, embed_dim // 2, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(embed_dim // 2),
        nn.GELU(),
        nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(embed_dim),
        nn.GELU(),
        nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(embed_dim),
    )


def _downsample(in_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(out_dim),
    )


class OverLoCK(nn.Module):
    """OverLoCK backbone with classification head.

    Hierarchical ConvNet with a top-down overview (context) branch that guides context-mixing
    dynamic kernels in the look-closely (main) branch. Stages 1-4 use `RepConvBlock`; the stage-3
    and stage-4 features are then refined by `DynamicConvBlock`s conditioned on an upsampled
    projection of the stage-4 context features.

    Args:
        depths: Block counts for the four overview stages.
        sub_depths: DynamicConvBlock counts refining stages 3 and 4.
        in_chans: Number of input image channels.
        embed_dim: Channel width of each of the four stages.
        kernel_size: Large-kernel size per stage.
        mlp_ratio: MLP expansion ratio of the overview stage blocks.
        sub_mlp_ratio: MLP expansion ratio of the `DynamicConvBlock`s.
        sub_num_heads: Dynamic-kernel heads of the `DynamicConvBlock`s.
        ls_init_value: LayerScale init value per stage (None disables).
        res_scale: Reference residual layout (scale the identity path instead of the branch).
        smk_size: Secondary (small) dynamic kernel size.
        deploy: Build `DilatedReparamBlock` in fused deployment mode (use with reparameterized
            weights + `reparam()`).
        projection: Hidden width of the classification head.
        num_classes: Classifier output classes, 0 to remove the classifier.
        global_pool: Global pooling type ('avg' or '' to disable).
        drop_rate: Classifier dropout rate.
        drop_path_rate: Stochastic depth rate (linear ramp over all blocks).
        norm_layer: Normalization layer for the block pre-norms.
        use_checkpoint: Per-stage count of leading blocks to run under gradient checkpointing.
    """

    def __init__(
            self,
            depths: List[int] = [2, 2, 2, 2],
            sub_depths: List[int] = [4, 2],
            in_chans: int = 3,
            embed_dim: List[int] = [96, 192, 384, 768],
            kernel_size: List[int] = [7, 7, 7, 7],
            mlp_ratio: List[float] = [4., 4., 4., 4.],
            sub_mlp_ratio: List[float] = [4., 4.],
            sub_num_heads: List[int] = [4, 8],
            ls_init_value: List[Optional[float]] = [None, None, 1., 1.],
            res_scale: bool = True,
            smk_size: int = 5,
            deploy: bool = False,
            projection: int = 1024,
            num_classes: int = 1000,
            global_pool: str = 'avg',
            drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            norm_layer: Type[nn.Module] = LayerNorm2d,
            use_checkpoint: List[int] = [0, 0, 0, 0],
    ):
        super().__init__()

        fusion_dim = embed_dim[-1] + embed_dim[-1] // 4
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.num_features = fusion_dim
        self.head_hidden_size = projection
        self.embed_dim = embed_dim
        self.grad_checkpointing = False

        self.patch_embed1 = _stem(in_chans, embed_dim[0])
        self.patch_embed2 = _downsample(embed_dim[0], embed_dim[1])
        self.patch_embed3 = _downsample(embed_dim[1], embed_dim[2])
        self.patch_embed4 = _downsample(embed_dim[2], embed_dim[3])
        self.high_level_proj = nn.Conv2d(embed_dim[-1], embed_dim[-1] // 4, kernel_size=1)
        self.patch_embedx = CTXDownsample(embed_dim[2], embed_dim[3])

        dpr = calculate_drop_path_rates(drop_path_rate, sum(depths) + sum(sub_depths))

        self.blocks1 = nn.ModuleList()
        self.blocks2 = nn.ModuleList()
        self.blocks3 = nn.ModuleList()
        self.blocks4 = nn.ModuleList()
        self.sub_blocks3 = nn.ModuleList()
        self.sub_blocks4 = nn.ModuleList()

        for i in range(depths[0]):
            self.blocks1.append(
                RepConvBlock(
                    dim=embed_dim[0],
                    kernel_size=kernel_size[0],
                    mlp_ratio=mlp_ratio[0],
                    ls_init_value=ls_init_value[0],
                    res_scale=res_scale,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    deploy=deploy,
                    use_checkpoint=(i < use_checkpoint[0]),
                )
            )

        for i in range(depths[1]):
            self.blocks2.append(
                RepConvBlock(
                    dim=embed_dim[1],
                    kernel_size=kernel_size[1],
                    mlp_ratio=mlp_ratio[1],
                    ls_init_value=ls_init_value[1],
                    res_scale=res_scale,
                    drop_path=dpr[i + depths[0]],
                    norm_layer=norm_layer,
                    deploy=deploy,
                    use_checkpoint=(i < use_checkpoint[1]),
                )
            )

        for i in range(depths[2]):
            self.blocks3.append(
                RepConvBlock(
                    dim=embed_dim[2],
                    kernel_size=kernel_size[2],
                    mlp_ratio=mlp_ratio[2],
                    ls_init_value=ls_init_value[2],
                    res_scale=res_scale,
                    drop_path=dpr[i + sum(depths[:2])],
                    norm_layer=norm_layer,
                    deploy=deploy,
                    use_checkpoint=(i < use_checkpoint[2]),
                )
            )

        for i in range(depths[3]):
            self.blocks4.append(
                RepConvBlock(
                    dim=embed_dim[3],
                    kernel_size=kernel_size[3],
                    mlp_ratio=mlp_ratio[3],
                    ls_init_value=ls_init_value[3],
                    res_scale=res_scale,
                    drop_path=dpr[i + sum(depths[:3])],
                    norm_layer=norm_layer,
                    deploy=deploy,
                    use_checkpoint=(i < use_checkpoint[3]),
                )
            )

        for i in range(sub_depths[0]):
            self.sub_blocks3.append(
                DynamicConvBlock(
                    dim=embed_dim[2],
                    ctx_dim=embed_dim[-1],
                    kernel_size=kernel_size[2],
                    num_heads=sub_num_heads[0],
                    mlp_ratio=sub_mlp_ratio[0],
                    ls_init_value=ls_init_value[2],
                    res_scale=res_scale,
                    drop_path=dpr[i + sum(depths)],
                    norm_layer=norm_layer,
                    smk_size=smk_size,
                    deploy=deploy,
                    is_first=(i == 0),
                    use_checkpoint=(i < use_checkpoint[2]),
                )
            )

        for i in range(sub_depths[1]):
            self.sub_blocks4.append(
                DynamicConvBlock(
                    dim=embed_dim[3],
                    ctx_dim=embed_dim[-1],
                    kernel_size=kernel_size[-1],
                    num_heads=sub_num_heads[1],
                    mlp_ratio=sub_mlp_ratio[1],
                    ls_init_value=ls_init_value[3],
                    res_scale=res_scale,
                    drop_path=dpr[i + sum(depths) + sub_depths[0]],
                    norm_layer=norm_layer,
                    smk_size=smk_size,
                    is_first=False,
                    is_last=(i == sub_depths[1] - 1),
                    deploy=deploy,
                    use_checkpoint=(i < use_checkpoint[3]),
                )
            )

        # Main Cls Head - the AdaptiveAvgPool2d entry is kept (index 3) so that the classifier
        # conv stays at `head.4` as in the released checkpoints; pooling in `forward_head` is
        # routed through `self.global_pool` to support timm's disable-pooling API.
        self.head = nn.Sequential(
            nn.Conv2d(fusion_dim, projection, kernel_size=1, bias=False),
            nn.BatchNorm2d(projection),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(projection, num_classes, kernel_size=1) if num_classes > 0 else nn.Identity(),
        )
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.flatten = nn.Flatten(1) if global_pool else nn.Identity()  # don't flatten if pooling disabled

        # feature pyramid of the main branch (context branch is auxiliary and not exposed)
        self.feature_info = [
            dict(num_chs=embed_dim[0], reduction=4, module='blocks1'),
            dict(num_chs=embed_dim[1], reduction=8, module='blocks2'),
            dict(num_chs=embed_dim[2], reduction=16, module='sub_blocks3'),
            dict(num_chs=fusion_dim, reduction=32, module='sub_blocks4'),
        ]

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
            nn.init.trunc_normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d, nn.BatchNorm1d)):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self) -> Set[str]:
        # dynamic-kernel relative position biases are bias-like, keep them out of weight decay
        return {name for name, _ in self.named_parameters() if name.endswith(('rpb1', 'rpb2'))}

    @torch.jit.ignore
    def group_matcher(self, coarse: bool = False) -> Dict[str, Any]:
        # NOTE stage numbers in module names are 1-based (`blocks1`, `sub_blocks3`, ...) and the
        # look-closely refinement of a stage shares the overview stage's group
        matcher = dict(
            stem=r'^patch_embed\d',
            blocks=[
                (
                    r'^(?:blocks|sub_blocks)(\d+)' if coarse else r'^(?:blocks|sub_blocks)(\d+)\.(\d+)',
                    None,
                ),
            ],
        )
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True):
        self.grad_checkpointing = enable
        depths = (len(self.blocks1), len(self.blocks2), len(self.blocks3), len(self.blocks4))
        use_checkpoint = depths if enable else (0, 0, 0, 0)
        for stage_idx, stage in enumerate((self.blocks1, self.blocks2, self.blocks3, self.blocks4)):
            for i, blk in enumerate(stage):
                blk.use_checkpoint = i < use_checkpoint[stage_idx]
        for stage_idx, stage in enumerate((self.sub_blocks3, self.sub_blocks4)):
            for i, blk in enumerate(stage):
                blk.use_checkpoint = i < use_checkpoint[stage_idx + 2]

    @torch.jit.ignore
    def get_classifier(self) -> nn.Module:
        return self.head[4]

    def reset_classifier(self, num_classes: int, global_pool: Optional[str] = None) -> None:
        dd = get_device_dtype(self)
        was_training = self.training
        self.num_classes = num_classes
        if global_pool is not None:
            self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
            self.flatten = nn.Flatten(1) if global_pool else nn.Identity()  # don't flatten if pool off
            self.global_pool.train(was_training)
        self.head[4] = nn.Conv2d(
            self.head_hidden_size, num_classes, kernel_size=1, **dd,
        ) if num_classes > 0 else nn.Identity()
        self.head[4].train(was_training)

    def reparam(self) -> None:
        """Fold the dilated branches of all `DilatedReparamBlock`s into their large kernel."""
        for m in self.modules():
            if isinstance(m, DilatedReparamBlock):
                m.merge_dilated_branches()

    def forward_pre_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed1(x)
        for blk in self.blocks1:
            x = blk(x)

        x = self.patch_embed2(x)
        for blk in self.blocks2:
            x = blk(x)

        return x

    def forward_base_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.patch_embed3(x)
        for blk in self.blocks3:
            x = blk(x)

        ctx = self.patch_embed4(x)
        for blk in self.blocks4:
            ctx = blk(ctx)

        return x, ctx

    def forward_sub_features(
            self, x: torch.Tensor, ctx: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ctx_cls = ctx
        ctx_ori = self.high_level_proj(ctx)
        ctx_up = F.interpolate(ctx_ori, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)

        for idx, blk in enumerate(self.sub_blocks3):
            if idx == 0:
                ctx = ctx_up
            x, ctx = blk(x, ctx, ctx_up)

        x, ctx = self.patch_embedx(x, ctx)
        for idx, blk in enumerate(self.sub_blocks4):
            x, ctx = blk(x, ctx, ctx_ori)

        return x, ctx_cls

    def _forward_pyramid(
            self,
            x: torch.Tensor,
            take_indices: Optional[Set[int]] = None,
            max_index: int = 3,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Shared pipeline of `forward_features` and `forward_intermediates`.

        The stage-3 / stage-4 pyramid entries are the look-closely refined features (after
        `sub_blocks3` / `sub_blocks4`); the overview/context side-branch is not exposed.
        """
        take_indices = take_indices or set()
        intermediates = []

        x = self.patch_embed1(x)
        for blk in self.blocks1:
            x = blk(x)
        if 0 in take_indices:
            intermediates.append(x)
        if max_index == 0:
            return x, intermediates

        x = self.patch_embed2(x)
        for blk in self.blocks2:
            x = blk(x)
        if 1 in take_indices:
            intermediates.append(x)
        if max_index == 1:
            return x, intermediates

        x = self.patch_embed3(x)
        for blk in self.blocks3:
            x = blk(x)

        ctx = self.patch_embed4(x)
        for blk in self.blocks4:
            ctx = blk(ctx)

        ctx_ori = self.high_level_proj(ctx)
        ctx_up = F.interpolate(ctx_ori, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)

        ctx = ctx_up
        for blk in self.sub_blocks3:
            x, ctx = blk(x, ctx, ctx_up)
        if 2 in take_indices:
            intermediates.append(x)
        if max_index == 2:
            return x, intermediates

        x, ctx = self.patch_embedx(x, ctx)
        for blk in self.sub_blocks4:
            x, ctx = blk(x, ctx, ctx_ori)
        if 3 in take_indices:
            intermediates.append(x)
        return x, intermediates

    def forward_intermediates(
            self,
            x: torch.Tensor,
            indices: Optional[Union[int, List[int]]] = None,
            norm: bool = False,
            stop_early: bool = False,
            output_fmt: str = 'NCHW',
            intermediates_only: bool = False,
    ) -> Union[List[torch.Tensor], Tuple[torch.Tensor, List[torch.Tensor]]]:
        """Forward features that returns intermediates.

        Args:
            x: Input image tensor.
            indices: Take last n blocks if int, all if None, select matching indices if sequence.
            norm: Apply norm layer to compatible intermediates (no-op, OverLoCK has no final norm).
            stop_early: Stop iterating over blocks when last desired intermediate hit.
            output_fmt: Shape of intermediate feature outputs.
            intermediates_only: Only return intermediate features.

        Returns:
            List of intermediate features or tuple of (final features, intermediates).
        """
        assert output_fmt in ('NCHW',), 'Output shape must be NCHW.'
        take_indices, max_index = feature_take_indices(len(self.feature_info), indices)
        take_indices = set(take_indices)

        x, intermediates = self._forward_pyramid(
            x, take_indices=take_indices, max_index=max_index if stop_early else 3)

        if intermediates_only:
            return intermediates

        return x, intermediates

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_pre_features(x)
        x, ctx = self.forward_base_features(x)
        x, _ = self.forward_sub_features(x, ctx)
        return x

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        x = self.head[0](x)
        x = self.head[1](x)
        x = self.head[2](x)
        x = self.global_pool(x)
        x = F.dropout(x, self.drop_rate, self.training) if self.drop_rate > 0. else x
        if not pre_logits:
            x = self.head[4](x)
        x = self.flatten(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def checkpoint_filter_fn(state_dict: Dict[str, torch.Tensor], model: nn.Module) -> Dict[str, torch.Tensor]:
    """Prepare released OverLoCK checkpoints for this port.

    The released classification checkpoints match the reference module naming used here
    (`patch_embed*`, `blocks*`, `sub_blocks*`, `head`). This filter unwraps trainer containers,
    strips DDP / detection-framework prefixes, and drops the training-only deep-supervision
    `aux_head` entries (not built in timm, does not affect inference).
    """
    state_dict = state_dict.get('state_dict', state_dict)
    if all(k.startswith('module.') for k in state_dict):
        state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
    if all(k.startswith('backbone.') for k in state_dict):
        state_dict = {k[len('backbone.'):]: v for k, v in state_dict.items()}
    return {k: v for k, v in state_dict.items() if not k.startswith('aux_head.')}


def _cfg(url: str = '', **kwargs: Any) -> Dict[str, Any]:
    return {
        'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed1.0', 'classifier': 'head.4',
        'origin_url': 'https://github.com/LMMMEng/OverLoCK', 'license': 'apache-2.0',
        **kwargs,
    }


default_cfgs = generate_default_cfgs({
    'overlock_xt.in1k': _cfg(
        url='https://github.com/LMMMEng/OverLoCK/releases/download/v1/overlock_xt_in1k_224.pth',
        crop_pct=0.925,
    ),
    'overlock_t.in1k': _cfg(
        url='https://github.com/LMMMEng/OverLoCK/releases/download/v1/overlock_t_in1k_224.pth',
        crop_pct=0.95,
    ),
    'overlock_s.in1k': _cfg(
        url='https://github.com/LMMMEng/OverLoCK/releases/download/v1/overlock_s_in1k_224.pth',
        crop_pct=0.95,
    ),
    'overlock_b.in1k': _cfg(
        url='https://github.com/LMMMEng/OverLoCK/releases/download/v1/overlock_b_in1k_224.pth',
        crop_pct=0.975,
    ),
})


def _create_overlock(variant: str, pretrained: bool = False, **kwargs: Any) -> OverLoCK:
    # feature_cls='getter': the top-down context dataflow (x / ctx streams) cannot be flattened
    # into a sequential feature extractor, so features_only goes through forward_intermediates()
    model = build_model_with_cfg(
        OverLoCK, variant, pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        feature_cfg=dict(out_indices=(0, 1, 2, 3), feature_cls='getter'),
        **kwargs,
    )
    return model


@register_model
def overlock_xt(pretrained: bool = False, **kwargs: Any) -> OverLoCK:
    """Instantiate OverLoCK-Xt model variant."""
    model_args = dict(
        depths=[2, 2, 3, 2],
        sub_depths=[6, 2],
        embed_dim=[56, 112, 256, 336],
        kernel_size=[17, 15, 13, 7],
        mlp_ratio=[4., 4., 4., 4.],
        sub_num_heads=[4, 6],
        sub_mlp_ratio=[3., 3.],
    )
    return _create_overlock('overlock_xt', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def overlock_t(pretrained: bool = False, **kwargs: Any) -> OverLoCK:
    """Instantiate OverLoCK-T model variant."""
    model_args = dict(
        depths=[4, 4, 6, 2],
        sub_depths=[12, 2],
        embed_dim=[64, 128, 256, 512],
        kernel_size=[17, 15, 13, 7],
        mlp_ratio=[4., 4., 4., 4.],
        sub_num_heads=[4, 8],
        sub_mlp_ratio=[3., 3.],
    )
    return _create_overlock('overlock_t', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def overlock_s(pretrained: bool = False, **kwargs: Any) -> OverLoCK:
    """Instantiate OverLoCK-S model variant."""
    model_args = dict(
        depths=[6, 6, 8, 3],
        sub_depths=[16, 3],
        embed_dim=[64, 128, 320, 512],
        kernel_size=[17, 15, 13, 7],
        mlp_ratio=[4., 4., 4., 4.],
        sub_num_heads=[8, 16],
        sub_mlp_ratio=[3., 3.],
    )
    return _create_overlock('overlock_s', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def overlock_b(pretrained: bool = False, **kwargs: Any) -> OverLoCK:
    """Instantiate OverLoCK-B model variant."""
    model_args = dict(
        depths=[8, 8, 10, 4],
        sub_depths=[20, 4],
        embed_dim=[80, 160, 384, 576],
        kernel_size=[17, 15, 13, 7],
        mlp_ratio=[4., 4., 4., 4.],
        sub_num_heads=[6, 9],
        sub_mlp_ratio=[3., 3.],
    )
    return _create_overlock('overlock_b', pretrained=pretrained, **dict(model_args, **kwargs))
