"""AdaptViG - Adaptive Vision GNN with Exponential Decay Gating

Papers:
* `AdaptViG: Adaptive Vision GNN with Exponential Decay Gating` - https://arxiv.org/abs/2511.09942 (WACV 2026)

Code: https://github.com/mmunir127/AdaptViG

AdaptViG mixes depthwise-separable local blocks with graph blocks on a 4-stage hierarchical
pyramid (strides 4/8/16/32). The first three stages use `AdaptConv`, a static axial graph
convolution: the neighbourhood is fixed (always-on local shifts plus long-range connections
at dyadic offsets) and the only data-dependent term is an exponential decay gate
`exp(-|x - x_j|_1 / temperature)` on the long-range edges. The forward is plain PyTorch
(conv / cat / BN / LN / exp / bmm / max) with no top-k, gather, or scatter, so it runs
everywhere including CPU. The final stage uses dense `GlobalAttentionGraphConv` attention.

The classification model is trained with hard distillation (RegNetY-160 teacher) and carries
a dual head. It follows timm's DeiT-distilled convention: the distillation head is `head_dist`,
eval forwards return the averaged logits, and `set_distilled_training(True)` returns the
`(logits, logits_dist)` tuple while training.

Ported from the official implementation (Apache-2.0 license, Copyright (c) 2025 Mustafa Munir,
Md Mostafijur Rahman, Radu Marculescu). Module and parameter names match the reference so
upstream checkpoints load with a near-identity `checkpoint_filter_fn` (only `dist_head` is
renamed to `head_dist` and the training-checkpoint envelope is unwrapped).
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import DropPath, calculate_drop_path_rates, get_device_dtype
from ._builder import build_model_with_cfg
from ._features import feature_take_indices
from ._manipulate import checkpoint_seq
from ._registry import generate_default_cfgs, register_model

__all__ = ['AdaptViG']


class Stem(nn.Module):

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(input_dim, output_dim // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(output_dim // 2),
            nn.GELU(),
            nn.Conv2d(output_dim // 2, output_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(output_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stem(x)


class DepthWiseSeparable(nn.Module):

    def __init__(self, in_dim: int, kernel: int = 3, expansion: int = 4) -> None:
        super().__init__()
        hidden_dim = in_dim * expansion
        self.pw1 = nn.Conv2d(in_dim, hidden_dim, 1)
        self.norm1 = nn.BatchNorm2d(hidden_dim)
        self.act1 = nn.GELU()
        self.dw = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel, padding=kernel // 2, groups=hidden_dim)
        self.norm2 = nn.BatchNorm2d(hidden_dim)
        self.act2 = nn.GELU()
        self.pw2 = nn.Conv2d(hidden_dim, in_dim, 1)
        self.norm3 = nn.BatchNorm2d(in_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.norm1(self.pw1(x)))
        x = self.act2(self.norm2(self.dw(x)))
        return self.norm3(self.pw2(x))


class InvertedResidual(nn.Module):

    def __init__(
            self,
            dim: int,
            kernel: int = 3,
            expansion: int = 4,
            drop_path: float = 0.,
            use_layer_scale: bool = True,
            layer_scale_init_value: float = 1e-5,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.dws = DepthWiseSeparable(in_dim=dim, kernel=kernel, expansion=expansion)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_2 = nn.Parameter(
                torch.full((dim,), layer_scale_init_value, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.dws(x))
        else:
            x = x + self.drop_path(self.dws(x))
        return x


class AdaptConv(nn.Module):
    """Static axial graph convolution with exponential decay gating.

    Local neighbours (offset <= K) are always connected. Long-range neighbours at dyadic
    offsets (2, 4, ..., < H) are gated by `exp(-|x - x_j|_1 / temperature)`, so edge influence
    decays with feature distance while the temperature is learned. The element-wise maximum
    over candidate edges forms the graph aggregate `x_j`, which is concatenated to the input
    and fused with a 1x1 convolution.
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            K: int = 2,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.K = K
        # learnable temperature for the exponential decay gate
        self.temperature = nn.Parameter(torch.tensor(1.0, device=device, dtype=dtype))
        self.nn = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, H, W = x.shape
        x_j = x - x  # zero init on the input device/dtype

        # 1. guaranteed local connections (always on, ungated)
        x_c_local = torch.cat([x[:, :, -self.K:, :], x[:, :, :-self.K, :]], dim=2)
        x_j = torch.max(x_j, x_c_local - x)
        x_r_local = torch.cat([x[:, :, :, -self.K:], x[:, :, :, :-self.K]], dim=3)
        x_j = torch.max(x_j, x_r_local - x)

        # 2. gated long-range connections with exponential decay
        temperature = torch.abs(self.temperature) + 1e-6
        for i in range(1, H.bit_length()):
            dist_exp = 2 ** i
            neighbor = torch.cat([x[:, :, -dist_exp:, :], x[:, :, :-dist_exp, :]], dim=2)
            dist = torch.norm(x - neighbor, p=1, dim=1, keepdim=True)
            gate = torch.exp(-dist / temperature)
            x_j = torch.max(x_j, (neighbor - x) * gate)

        for i in range(1, W.bit_length()):
            dist_exp = 2 ** i
            neighbor = torch.cat([x[:, :, :, -dist_exp:], x[:, :, :, :-dist_exp]], dim=3)
            dist = torch.norm(x - neighbor, p=1, dim=1, keepdim=True)
            gate = torch.exp(-dist / temperature)
            x_j = torch.max(x_j, (neighbor - x) * gate)

        x = torch.cat([x, x_j], dim=1)
        return self.nn(x)


class GlobalAttentionGraphConv(nn.Module):
    """Dense global attention graph convolution, used for the final stage."""

    def __init__(self, in_channels: int, out_channels: int, K: int = 2) -> None:
        super().__init__()
        self.K = K
        self.q_conv = nn.Conv2d(in_channels, in_channels // 4, 1, bias=False)
        self.k_conv = nn.Conv2d(in_channels, in_channels // 4, 1, bias=False)
        self.v_conv = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        # layer norm over the channel dim stabilizes the dot product
        self.q_norm = nn.LayerNorm(in_channels // 4)
        self.k_norm = nn.LayerNorm(in_channels // 4)
        self.nn = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_j = x - x
        # project to q, k, v as (B, N, C) token sequences
        q = self.q_norm(self.q_conv(x).view(B, -1, H * W).permute(0, 2, 1))
        k = self.k_norm(self.k_conv(x).view(B, -1, H * W).permute(0, 2, 1))
        v = self.v_conv(x).view(B, -1, H * W).permute(0, 2, 1)
        attn = torch.bmm(q, k.transpose(-2, -1)) / (C // 4) ** 0.5
        attn = attn.softmax(dim=-1)
        x_attn = torch.bmm(attn, v).permute(0, 2, 1).view(B, C, H, W)
        x_j = torch.max(x_j, x_attn - x)
        x = torch.cat([x, x_j], dim=1)
        return self.nn(x)


class ConditionalPositionEncoding(nn.Module):

    def __init__(self, in_channels: int, kernel_size: int = 7) -> None:
        super().__init__()
        self.pe = nn.Conv2d(
            in_channels, in_channels, kernel_size, padding=kernel_size // 2, bias=True, groups=in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe(x) + x


class AdaptiveGrapher(nn.Module):

    def __init__(
            self,
            in_channels: int,
            K: int,
            graph_conv_type: str = 'gating',
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.cpe = ConditionalPositionEncoding(in_channels, kernel_size=7)
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
        )
        if graph_conv_type == 'gating':
            self.graph_conv = AdaptConv(in_channels, in_channels, K=K, device=device, dtype=dtype)
        else:  # 'global'
            self.graph_conv = GlobalAttentionGraphConv(in_channels, in_channels, K=K)
        self.fc2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cpe(x)
        x = self.fc1(x)
        x = self.graph_conv(x)
        x = self.fc2(x)
        return x


class AdaptiveGraphConvBlock(nn.Module):

    def __init__(
            self,
            in_dim: int,
            drop_path: float = 0.,
            K: int = 2,
            graph_conv_type: str = 'gating',
            use_layer_scale: bool = True,
            layer_scale_init_value: float = 1e-5,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.mixer = AdaptiveGrapher(in_dim, K, graph_conv_type, device=device, dtype=dtype)
        self.ffn = nn.Sequential(
            nn.Conv2d(in_dim, in_dim * 4, 1),
            nn.BatchNorm2d(in_dim * 4),
            nn.GELU(),
            nn.Conv2d(in_dim * 4, in_dim, 1),
            nn.BatchNorm2d(in_dim),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                torch.full((in_dim,), layer_scale_init_value, device=device, dtype=dtype))
            self.layer_scale_2 = nn.Parameter(
                torch.full((in_dim,), layer_scale_init_value, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.mixer(x))
            x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.ffn(x))
        else:
            x = x + self.drop_path(self.mixer(x))
            x = x + self.drop_path(self.ffn(x))
        return x


class Downsample(nn.Module):

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class AdaptViG(nn.Module):
    """AdaptViG - Adaptive Vision GNN with Exponential Decay Gating.

    A 4-stage hierarchical vision GNN (strides 4/8/16/32) mixing depthwise-separable local
    blocks with `AdaptConv` graph blocks, and dense global attention in the final stage.
    """

    def __init__(
            self,
            blocks: List[List[int]],
            channels: List[int],
            kernels: int = 3,
            emb_dims: int = 768,
            K: Sequence[int] = (8, 4, 2, 1),
            in_chans: int = 3,
            num_classes: int = 1000,
            drop_rate: float = 0.,
            drop_path_rate: float = 0.1,
            distilled: bool = True,
            layer_scale_init_value: float = 1e-5,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.distilled = distilled
        self.distilled_training = False  # set True to train with a (logits, logits_dist) tuple output
        self.grad_checkpointing = False

        dpr = calculate_drop_path_rates(drop_path_rate, sum(sum(stage) for stage in blocks))
        dpr_idx = 0
        self.stem = Stem(input_dim=in_chans, output_dim=channels[0])
        self.backbone = nn.ModuleList()
        self.feature_info = []
        num_stages = len(blocks)
        for i in range(num_stages):
            self.feature_info.append(dict(num_chs=channels[i], reduction=2 ** (i + 2), module=f'backbone.{i}'))
            stage = []
            local_blocks, global_blocks = blocks[i]
            graph_conv_type = 'global' if i == num_stages - 1 else 'gating'
            if i > 0:
                stage.append(Downsample(channels[i - 1], channels[i]))
            for _ in range(local_blocks):
                stage.append(InvertedResidual(
                    dim=channels[i], kernel=kernels, drop_path=dpr[dpr_idx],
                    layer_scale_init_value=layer_scale_init_value, device=device, dtype=dtype))
                dpr_idx += 1
            for _ in range(global_blocks):
                stage.append(AdaptiveGraphConvBlock(
                    channels[i], drop_path=dpr[dpr_idx], K=K[i], graph_conv_type=graph_conv_type,
                    layer_scale_init_value=layer_scale_init_value, device=device, dtype=dtype))
                dpr_idx += 1
            self.backbone.append(nn.Sequential(*stage))

        self.num_features = channels[-1]
        self.head_hidden_size = emb_dims
        self.prediction = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels[-1], emb_dims, 1, bias=True),
            nn.BatchNorm2d(emb_dims),
            nn.GELU(),
            nn.Dropout(drop_rate),
        )
        self.head = (
            nn.Conv2d(emb_dims, num_classes, 1, bias=True, device=device, dtype=dtype)
            if num_classes > 0 else nn.Identity())
        self.head_dist = (
            nn.Conv2d(emb_dims, num_classes, 1, bias=True, device=device, dtype=dtype)
            if distilled and num_classes > 0 else nn.Identity())

        self.init_weights()

    def init_weights(self) -> None:
        """Kaiming-normal init of all conv weights with zero biases, as in the reference."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @torch.jit.ignore
    def set_distilled_training(self, enable: bool = True) -> None:
        self.distilled_training = enable

    @torch.jit.ignore
    def group_matcher(self, coarse: bool = False) -> Dict[str, Any]:
        if coarse:
            return dict(stem=r'^stem', blocks=r'^backbone')
        return dict(
            stem=r'^stem',
            blocks=[
                (r'^backbone\.(\d+)\.(\d+)', None),
                (r'^prediction', (99999,)),  # pool + projection sit after the last stage
            ],
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True) -> None:
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self) -> Union[nn.Module, Tuple[nn.Module, nn.Module]]:
        return (self.head, self.head_dist) if self.distilled else self.head

    def reset_classifier(self, num_classes: int, global_pool: Optional[str] = None) -> None:
        dd = get_device_dtype(self)
        self.num_classes = num_classes
        self.head = (
            nn.Conv2d(self.head_hidden_size, num_classes, 1, bias=True, **dd)
            if num_classes > 0 else nn.Identity())
        self.head_dist = (
            nn.Conv2d(self.head_hidden_size, num_classes, 1, bias=True, **dd)
            if self.distilled and num_classes > 0 else nn.Identity())
        self.head.train(self.training)
        self.head_dist.train(self.training)

    def forward_intermediates(
            self,
            x: torch.Tensor,
            indices: Optional[Union[int, Sequence[int]]] = None,
            norm: bool = False,
            stop_early: bool = False,
            output_fmt: str = 'NCHW',
            intermediates_only: bool = False,
    ) -> Union[List[torch.Tensor], Tuple[torch.Tensor, List[torch.Tensor]]]:
        """Forward features that returns intermediates.

        Args:
            x: Input image tensor
            indices: Take last n blocks if int, all if None, select matching indices if sequence
            norm: Unused for this model (no final norm layer)
            stop_early: Stop iterating over blocks when last desired intermediate hit
            output_fmt: Shape of intermediate feature outputs
            intermediates_only: Only return intermediate features
        Returns:

        """
        assert output_fmt in ('NCHW',), 'Output shape must be NCHW.'
        intermediates = []
        take_indices, max_index = feature_take_indices(len(self.backbone), indices)

        x = self.stem(x)
        if torch.jit.is_scripting() or not stop_early:  # can't slice stages in torchscript
            stages = self.backbone
        else:
            stages = self.backbone[:max_index + 1]

        for feat_idx, stage in enumerate(stages):
            x = stage(x)
            if feat_idx in take_indices:
                intermediates.append(x)

        if intermediates_only:
            return intermediates

        return x, intermediates

    def prune_intermediate_layers(
            self,
            indices: Union[int, Sequence[int]] = 1,
            prune_norm: bool = False,
            prune_head: bool = True,
    ) -> Sequence[int]:
        """Prune layers not required for specified intermediates."""
        take_indices, max_index = feature_take_indices(len(self.backbone), indices)
        prune_end = max_index < len(self.backbone) - 1
        self.backbone = self.backbone[:max_index + 1]  # truncate stages w/ stem as idx 0
        if prune_end:
            # features now end at the last kept tap, the head is sized for the final features so it goes too
            self.num_features = self.feature_info[max_index]['num_chs']
        if prune_head or prune_end:
            self.reset_classifier(0, '')
        return take_indices

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for stage in self.backbone:
            if self.grad_checkpointing and self.training:
                x = checkpoint_seq(stage, x)
            else:
                x = stage(x)
        return x

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        x = self.prediction(x)
        if pre_logits:
            return torch.flatten(x, 1)
        if self.distilled:
            x, x_dist = self.head(x), self.head_dist(x)
            x, x_dist = torch.flatten(x, 1), torch.flatten(x_dist, 1)
            if self.distilled_training and self.training and not torch.jit.is_scripting():
                # return logits for both heads while distillation training
                return x, x_dist
            return (x + x_dist) / 2
        return torch.flatten(self.head(x), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        return self.forward_head(x)


def checkpoint_filter_fn(
        state_dict: Dict[str, torch.Tensor],
        model: nn.Module,
) -> Dict[str, torch.Tensor]:
    """Map upstream AdaptViG checkpoints onto this module's names.

    Module names match the reference implementation. The released checkpoints nest the
    weights under `state_dict` (final model) with `state_dict_ema` alongside; the reference
    training script tracks its reported best top-1 on the non-EMA model, so `state_dict` is
    preferred with `state_dict_ema` as fallback. The reference distillation head is named
    `dist_head` and is renamed to timm's `head_dist`; it is dropped for non-distilled models.
    """
    if 'stem.stem.0.weight' not in state_dict:
        for envelope_key in ('state_dict', 'state_dict_ema'):
            if isinstance(state_dict.get(envelope_key), dict):
                state_dict = state_dict[envelope_key]
                break
    state_dict = {k.replace('dist_head.', 'head_dist.'): v for k, v in state_dict.items()}
    if isinstance(model, AdaptViG) and not model.distilled:
        # distilled tensors have no target module, loading would report them unexpected
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith('head_dist.')}
    return state_dict


def _cfg(hf_hub_filename: str, **kwargs: Any) -> Dict[str, Any]:
    return {
        'url': '',
        'hf_hub_id': 'SLDGroup/AdaptViG',
        'hf_hub_filename': hf_hub_filename,
        'num_classes': 1000,
        'input_size': (3, 224, 224),
        'pool_size': None,
        'crop_pct': 0.9,
        'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN,
        'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'stem.stem.0',
        # both heads are num_classes-sized -- list them so timm resets each on a num_classes change
        'classifier': ('head', 'head_dist'),
        'paper_ids': 'arXiv:2511.09942',
        'origin_url': 'https://github.com/mmunir127/AdaptViG',
        'license': 'apache-2.0',
        **kwargs,
    }


default_cfgs = generate_default_cfgs({
    'adaptvig_s': _cfg('AdaptViG_S_model_best.pth'),
    'adaptvig_m': _cfg('AdaptViG_M_model_best.pth'),
    'adaptvig_b': _cfg('AdaptViG_B_model_best.pth'),
})


def _create_adaptvig(variant: str, pretrained: bool = False, **kwargs: Any) -> AdaptViG:
    feature_cfg = dict(out_indices=(0, 1, 2, 3), feature_cls='getter')
    return build_model_with_cfg(
        AdaptViG,
        variant,
        pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        feature_cfg=feature_cfg,
        **kwargs,
    )


@register_model
def adaptvig_s(pretrained: bool = False, **kwargs: Any) -> AdaptViG:
    """AdaptViG-S: 8.7M params, distilled dual head."""
    model_args = dict(
        blocks=[[3, 3], [3, 3], [9, 3], [3, 3]],
        channels=[32, 64, 128, 256],
        kernels=3,
        emb_dims=768,
        K=[8, 4, 2, 1],
        drop_path_rate=0.1,
    )
    return _create_adaptvig('adaptvig_s', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def adaptvig_m(pretrained: bool = False, **kwargs: Any) -> AdaptViG:
    """AdaptViG-M: 18.1M params, distilled dual head."""
    model_args = dict(
        blocks=[[4, 4], [4, 4], [12, 4], [4, 4]],
        channels=[48, 96, 192, 320],
        kernels=3,
        emb_dims=768,
        K=[8, 4, 2, 1],
        drop_path_rate=0.1,
    )
    return _create_adaptvig('adaptvig_m', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def adaptvig_b(pretrained: bool = False, **kwargs: Any) -> AdaptViG:
    """AdaptViG-B: 27.0M params, distilled dual head."""
    model_args = dict(
        blocks=[[5, 5], [5, 5], [15, 5], [5, 5]],
        channels=[48, 96, 192, 384],
        kernels=3,
        emb_dims=768,
        K=[8, 4, 2, 1],
        drop_path_rate=0.1,
    )
    return _create_adaptvig('adaptvig_b', pretrained=pretrained, **dict(model_args, **kwargs))
