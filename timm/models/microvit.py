"""MicroViTv2

Papers:
* `MicroViTv2: Beyond the FLOPS for Edge Energy-Friendly Vision Transformers` - https://arxiv.org/abs/2605.10148

Code: https://github.com/novendrastywn/MicroViT

Ported from the reference implementation which builds on SHViT (MIT license, Copyright (c) 2024 ysj9909).
Only the block types used by the released MicroViTv2 weights are ported: the depthwise RepConv mixer ('c')
and SDTA, the single depth-wise transposed attention of the paper. The MDTA / ESHA / DSHA / SSHA block
variants of the reference are not used by any released checkpoint and are omitted here (this also drops the
einops dependency of the reference, it is only needed by MDTA). The distillation head and the optional
final-feature projection of the reference are likewise unused by the released checkpoints and not ported.
"""
import math
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import GroupNorm1, SelectAdaptivePool2d, trunc_normal_, get_device_dtype
from ._builder import build_model_with_cfg
from ._features import feature_take_indices
from ._manipulate import checkpoint_seq
from ._registry import register_model, generate_default_cfgs

__all__ = ['MicroViTv2']


class Residual(nn.Module):
    def __init__(self, m: nn.Module):
        super().__init__()
        self.m = m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.m(x)

    @torch.no_grad()
    def fuse(self) -> nn.Module:
        if isinstance(self.m, (Conv2dNorm, RepConv)):
            m = self.m.fuse()
            assert m.groups == m.in_channels and m.stride == (1, 1)
            identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
            identity = F.pad(identity, [1, 1, 1, 1])
            m.weight += identity.to(m.weight.device)
            return m
        else:
            return self


class RepConv(nn.Module):
    """Re-parameterizable convolution, parallel full-kernel and half-kernel branches with a shared batch-norm."""
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 1,
            stride: int = 1,
            padding: int = 0,
            groups: int = 1,
            device=None,
            dtype=None,
    ):
        dd = {'device': device, 'dtype': dtype}
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, groups=groups, **dd)
        self.repconv = nn.Conv2d(
            in_channels, out_channels, kernel_size // 2, stride, padding // 2, groups=groups, **dd)
        self.bn = nn.BatchNorm2d(out_channels, **dd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x) + self.repconv(x)
        return self.bn(x)

    @torch.no_grad()
    def fuse(self) -> nn.Conv2d:
        conv = self.conv
        repconv = self.repconv
        # pad the half-kernel branch up to the full kernel size and sum the two branches
        kw, kh = (conv.weight.shape[2] - repconv.weight.shape[2]) // 2, \
            (conv.weight.shape[3] - repconv.weight.shape[3]) // 2
        weight = conv.weight + F.pad(repconv.weight, [kh, kh, kw, kw])
        bias = conv.bias + repconv.bias

        # fold the shared batch-norm
        bn = self.bn
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        weight = weight * w[:, None, None, None]
        bias = bn.bias + (bias - bn.running_mean) * bn.weight / (bn.running_var + bn.eps) ** 0.5

        m = nn.Conv2d(
            conv.in_channels, conv.out_channels, conv.kernel_size,
            stride=conv.stride, padding=conv.padding, groups=conv.groups,
            device=conv.weight.device, dtype=conv.weight.dtype)
        m.weight.data.copy_(weight)
        m.bias.data.copy_(bias)
        return m


class Conv2dNorm(nn.Sequential):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 1,
            stride: int = 1,
            padding: int = 0,
            bn_weight_init: int = 1,
            device=None,
            dtype=None,
            **kwargs,
    ):
        dd = {'device': device, 'dtype': dtype}
        super().__init__()
        self.add_module('c', nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=False, **dd, **kwargs))
        self.add_module('bn', nn.BatchNorm2d(out_channels, **dd))
        nn.init.constant_(self.bn.weight, bn_weight_init)
        nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self) -> nn.Conv2d:
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / (bn.running_var + bn.eps) ** 0.5
        m = nn.Conv2d(
            in_channels=w.size(1) * self.c.groups,
            out_channels=w.size(0),
            kernel_size=w.shape[2:],
            stride=self.c.stride,
            padding=self.c.padding,
            dilation=self.c.dilation,
            groups=self.c.groups,
            device=c.weight.device,
            dtype=c.weight.dtype,
        )
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class NormLinear(nn.Sequential):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool = True,
            std: float = 0.02,
            device=None,
            dtype=None,
    ):
        dd = {'device': device, 'dtype': dtype}
        super().__init__()
        self.add_module('bn', nn.BatchNorm1d(in_features, **dd))
        self.add_module('l', nn.Linear(in_features, out_features, bias=bias, **dd))
        trunc_normal_(self.l.weight, std=std)
        if bias:
            nn.init.constant_(self.l.bias, 0)

    @torch.no_grad()
    def fuse(self) -> nn.Linear:
        bn, l = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        b = bn.bias - self.bn.running_mean * self.bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = l.weight * w[None, :]
        if l.bias is None:
            b = b @ self.l.weight.T
        else:
            b = (l.weight @ b[:, None]).view(-1) + self.l.bias
        m = nn.Linear(w.size(1), w.size(0), device=l.weight.device, dtype=l.weight.dtype)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class FFN(nn.Module):
    def __init__(
            self,
            dim: int,
            embed_dim: int,
            act_layer: Type[nn.Module] = nn.ReLU,
            device=None,
            dtype=None,
    ):
        dd = {'device': device, 'dtype': dtype}
        super().__init__()
        self.pw1 = Conv2dNorm(dim, embed_dim, **dd)
        self.act = act_layer()
        self.pw2 = Conv2dNorm(embed_dim, dim, **dd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pw1(x)
        x = self.act(x)
        x = self.pw2(x)
        return x


class SDTA(nn.Module):
    """Single depth-wise transposed attention, the attention op of the MicroViTv2 paper."""
    def __init__(
            self,
            dim: int,
            pdim: int,
            qk_dim: int = 16,
            device=None,
            dtype=None,
    ):
        dd = {'device': device, 'dtype': dtype}
        super().__init__()
        self.scale = qk_dim ** -0.5
        self.qk_dim = qk_dim
        self.dim = dim
        self.pdim = pdim
        self.split_index = (qk_dim, qk_dim, pdim, dim - pdim)

        self.pre_norm = GroupNorm1(dim, **dd)
        self.in_proj = nn.Sequential(
            RepConv(dim, dim, 3, 1, 1, groups=dim, **dd),
            Conv2dNorm(dim, qk_dim * 2 + dim, **dd),
        )
        self.out_proj = nn.Sequential(
            nn.GELU(),
            Conv2dNorm(dim, dim, **dd),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre_norm(x)
        q, k, v, u = self.in_proj(x).split(self.split_index, dim=1)
        q, k, v = q.flatten(2), k.flatten(2), v.flatten(2)

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)

        B, _, H, W = u.shape
        attn = (v @ attn.transpose(-2, -1)).reshape(B, self.pdim, H, W)
        x = self.out_proj(torch.cat((attn, u), dim=1))
        return x


class BasicBlock(nn.Module):
    def __init__(
            self,
            dim: int,
            qk_dim: int,
            pdim: int,
            type: str,
            act_layer: Type[nn.Module] = nn.SiLU,
            device=None,
            dtype=None,
    ):
        dd = {'device': device, 'dtype': dtype}
        super().__init__()
        if type == 'sdta':
            self.mixer = Residual(SDTA(dim, pdim, qk_dim, **dd))
        else:
            self.mixer = Residual(RepConv(dim, dim, 3, 1, 1, groups=dim, **dd))
        self.ffn = Residual(FFN(dim, int(dim * 2), act_layer, **dd))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mixer(x)
        x = self.ffn(x)
        return x


class StageBlock(nn.Module):
    def __init__(
            self,
            prev_dim: int,
            dim: int,
            qk_dim: int,
            pdim: int,
            type: str,
            depth: int,
            act_layer: Type[nn.Module] = nn.SiLU,
            device=None,
            dtype=None,
    ):
        dd = {'device': device, 'dtype': dtype}
        super().__init__()
        self.grad_checkpointing = False
        self.downsample = nn.Sequential(
            Residual(RepConv(prev_dim, prev_dim, 3, 1, 1, groups=prev_dim, **dd)),
            Residual(FFN(prev_dim, int(prev_dim * 2), act_layer, **dd)),
            RepConv(prev_dim, dim, 3, 2, 1, **dd),
            Residual(RepConv(dim, dim, 3, 1, 1, groups=dim, **dd)),
            Residual(FFN(dim, int(dim * 2), act_layer, **dd)),
        ) if prev_dim != dim else nn.Identity()

        self.blocks = nn.Sequential(*[
            BasicBlock(dim, qk_dim, pdim, type, act_layer, **dd) for _ in range(depth)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        return x


class MicroViTv2(nn.Module):
    def __init__(
            self,
            in_chans: int = 3,
            num_classes: int = 1000,
            global_pool: str = 'avg',
            embed_dim: Tuple[int, int, int] = (128, 224, 320),
            partial_dim: Tuple[int, int, int] = (0, 0, 80),
            qk_dim: int = 16,
            depth: Tuple[int, int, int] = (2, 6, 5),
            types: Tuple[str, str, str] = ('c', 'c', 'sdta'),
            patch_size: int = 16,
            drop_rate: float = 0.,
            act_layer: Type[nn.Module] = nn.SiLU,
            device=None,
            dtype=None,
    ):
        super().__init__()
        dd = {'device': device, 'dtype': dtype}
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.drop_rate = drop_rate
        self.feature_info = []

        # Patch embedding, a chain of stride-2 RepConv with the channel count doubling at every step
        stem_chs = embed_dim[0]
        stem_blocks = math.ceil(patch_size ** 0.5)
        stem_dims = [in_chans] + [stem_chs // 2 ** i for i in range(stem_blocks - 1, -1, -1)]
        self.stem = nn.Sequential(*[
            nn.Sequential(
                RepConv(stem_dims[i], stem_dims[i + 1], 3, 2, 1, **dd),
                act_layer(),
            ) if i < stem_blocks - 1 else RepConv(stem_dims[i], stem_dims[i + 1], 3, 2, 1, **dd)
            for i in range(stem_blocks)
        ])

        # Build MicroViTv2 blocks
        stages = []
        prev_chs = stem_chs
        for i in range(len(embed_dim)):
            stages.append(StageBlock(
                prev_dim=prev_chs,
                dim=embed_dim[i],
                qk_dim=qk_dim,
                pdim=partial_dim[i],
                type=types[i],
                depth=depth[i],
                act_layer=act_layer,
                **dd,
            ))
            prev_chs = embed_dim[i]
            self.feature_info.append(dict(num_chs=prev_chs, reduction=2 ** (i + 4), module=f'stages.{i}'))
        self.stages = nn.Sequential(*stages)

        # Classifier head
        self.num_features = self.head_hidden_size = embed_dim[-1]
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        assert self.global_pool.feat_mult() == 1, 'Pooling that changes the pooled width is not supported.'
        self.flatten = nn.Flatten(1) if global_pool else nn.Identity()  # don't flatten if pooling disabled
        self.head = NormLinear(self.head_hidden_size, num_classes, **dd) if num_classes > 0 else nn.Identity()

    @torch.jit.ignore
    def no_weight_decay(self) -> Set:
        return set()

    @torch.jit.ignore
    def group_matcher(self, coarse: bool = False) -> Dict[str, Any]:
        matcher = dict(
            stem=r'^stem',  # stem and embed
            blocks=r'^stages\.(\d+)' if coarse else [
                (r'^stages\.(\d+).downsample', (0,)),
                (r'^stages\.(\d+)\.blocks\.(\d+)', None),
            ]
        )
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        for s in self.stages:
            s.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self) -> nn.Module:
        return self.head.l

    def reset_classifier(self, num_classes: int, global_pool: Optional[str] = None):
        dd = get_device_dtype(self)
        self.num_classes = num_classes
        if global_pool is None:
            global_pool = self.global_pool.pool_type
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        assert self.global_pool.feat_mult() == 1, 'Pooling that changes the pooled width is not supported.'
        self.global_pool.train(self.training)
        self.flatten = nn.Flatten(1) if global_pool else nn.Identity()  # don't flatten if pooling disabled
        self.flatten.train(self.training)
        self.head = NormLinear(self.head_hidden_size, num_classes, **dd) if num_classes > 0 else nn.Identity()
        self.head.train(self.training)

    def forward_intermediates(
            self,
            x: torch.Tensor,
            indices: Optional[Union[int, List[int]]] = None,
            norm: bool = False,
            stop_early: bool = False,
            output_fmt: str = 'NCHW',
            intermediates_only: bool = False,
    ) -> Union[List[torch.Tensor], Tuple[torch.Tensor, List[torch.Tensor]]]:
        """ Forward features that returns intermediates.

        Args:
            x: Input image tensor
            indices: Take last n blocks if int, all if None, select matching indices if sequence
            norm: Apply norm layer to compatible intermediates
            stop_early: Stop iterating over blocks when last desired intermediate hit
            output_fmt: Shape of intermediate feature outputs
            intermediates_only: Only return intermediate features
        Returns:

        """
        assert output_fmt in ('NCHW',), 'Output shape must be NCHW.'
        intermediates = []
        take_indices, max_index = feature_take_indices(len(self.stages), indices)

        # forward pass
        x = self.stem(x)
        if torch.jit.is_scripting() or not stop_early:  # can't slice blocks in torchscript
            stages = self.stages
        else:
            stages = self.stages[:max_index + 1]

        for feat_idx, stage in enumerate(stages):
            x = stage(x)
            if feat_idx in take_indices:
                intermediates.append(x)

        if intermediates_only:
            return intermediates

        return x, intermediates

    def prune_intermediate_layers(
            self,
            indices: Union[int, List[int]] = 1,
            prune_norm: bool = False,
            prune_head: bool = True,
    ):
        """ Prune layers not required for specified intermediates.
        """
        take_indices, max_index = feature_take_indices(len(self.stages), indices)
        prune_end = max_index < len(self.stages) - 1
        self.stages = self.stages[:max_index + 1]  # truncate blocks w/ stem as idx 0
        if prune_end:
            # features now end at the last kept tap, the head is sized for the final features so it goes too
            self.num_features = self.head_hidden_size = self.feature_info[max_index]['num_chs']
        if prune_head or prune_end:
            self.reset_classifier(0, '')
        return take_indices

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stages(x)
        return x

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        x = self.global_pool(x)
        x = self.flatten(x)
        if self.drop_rate > 0.:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        return x if pre_logits else self.head(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x

    @torch.no_grad()
    def fuse(self):
        def fuse_children(net):
            for child_name, child in net.named_children():
                if hasattr(child, 'fuse'):
                    fused = child.fuse()
                    setattr(net, child_name, fused)
                    fuse_children(fused)
                else:
                    fuse_children(child)

        fuse_children(self)
        return self


def checkpoint_filter_fn(state_dict: Dict[str, torch.Tensor], model: nn.Module) -> Dict[str, torch.Tensor]:
    """ Remap reference MicroViTv2 checkpoints to the timm module layout.

    The reference packs the whole network into one flat nn.Sequential: stem, stage blocks, and the
    pre-patch / merge / post-patch modules between stages are flat entries stages.0 .. stages.9. timm
    nests them as stem + stages.{i}.downsample + stages.{i}.blocks, mirroring the SHViT layout.
    """
    state_dict = state_dict.get('model', state_dict)
    if not any(k.startswith('stages.0.stem.') for k in state_dict):
        # already in timm layout (e.g. the re-uploaded timm/ checkpoints)
        return state_dict

    replace_rules = [
        (re.compile(r'^stages\.0\.stem\.'), 'stem.'),
        # stage 1 blocks
        (re.compile(r'^stages\.1\.'), 'stages.0.'),
        # stage 2 downsample: pre-patch pair, RepConv merge, post-patch pair
        (re.compile(r'^stages\.2\.0\.'), 'stages.1.downsample.0.'),
        (re.compile(r'^stages\.2\.1\.'), 'stages.1.downsample.1.'),
        (re.compile(r'^stages\.3\.'), 'stages.1.downsample.2.'),
        (re.compile(r'^stages\.4\.0\.'), 'stages.1.downsample.3.'),
        (re.compile(r'^stages\.4\.1\.'), 'stages.1.downsample.4.'),
        # stage 2 blocks
        (re.compile(r'^stages\.5\.'), 'stages.1.'),
        # stage 3 downsample
        (re.compile(r'^stages\.6\.0\.'), 'stages.2.downsample.0.'),
        (re.compile(r'^stages\.6\.1\.'), 'stages.2.downsample.1.'),
        (re.compile(r'^stages\.7\.'), 'stages.2.downsample.2.'),
        (re.compile(r'^stages\.8\.0\.'), 'stages.2.downsample.3.'),
        (re.compile(r'^stages\.8\.1\.'), 'stages.2.downsample.4.'),
        # stage 3 blocks
        (re.compile(r'^stages\.9\.'), 'stages.2.'),
        # reference Block packs mixer + ffn in an unnamed nn.Sequential
        (re.compile(r'\.block\.0\.'), '.mixer.'),
        (re.compile(r'\.block\.1\.'), '.ffn.'),
        (re.compile(r'^head\.classifier\.'), 'head.'),
    ]

    out_dict = {}
    for k, v in state_dict.items():
        for pattern, replacement in replace_rules:
            k = pattern.sub(replacement, k)
        out_dict[k] = v

    return out_dict


def _cfg(url: str = '', **kwargs: Any) -> Dict[str, Any]:
    return {
        'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (4, 4),
        'crop_pct': 0.875, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'stem.0.0.conv', 'classifier': 'head.l',
        'license': 'mit',
        'paper_ids': 'arXiv:2605.10148',
        'paper_name': 'MicroViTv2: Beyond the FLOPS for Edge Energy-Friendly Vision Transformers',
        'origin_url': 'https://github.com/novendrastywn/MicroViT',
        **kwargs
    }


default_cfgs = generate_default_cfgs({
    'microvitv2_s1.in1k': _cfg(hf_hub_id='timm/'),
    'microvitv2_s2.in1k': _cfg(hf_hub_id='timm/'),
    'microvitv2_s3.in1k': _cfg(hf_hub_id='timm/'),
})


def _create_microvit(variant: str, pretrained: bool = False, **kwargs: Any) -> MicroViTv2:
    model = build_model_with_cfg(
        MicroViTv2, variant, pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        feature_cfg=dict(out_indices=(0, 1, 2), flatten_sequential=True),
        **kwargs,
    )
    return model


@register_model
def microvitv2_s1(pretrained: bool = False, **kwargs: Any) -> MicroViTv2:
    model_args = dict(
        embed_dim=(128, 224, 320), depth=(2, 6, 5), partial_dim=(0, 0, 80), types=('c', 'c', 'sdta'))
    return _create_microvit('microvitv2_s1', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def microvitv2_s2(pretrained: bool = False, **kwargs: Any) -> MicroViTv2:
    model_args = dict(
        embed_dim=(128, 308, 448), depth=(2, 7, 5), partial_dim=(0, 0, 112), types=('c', 'c', 'sdta'))
    return _create_microvit('microvitv2_s2', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def microvitv2_s3(pretrained: bool = False, **kwargs: Any) -> MicroViTv2:
    model_args = dict(
        embed_dim=(192, 384, 448), depth=(3, 7, 6), partial_dim=(0, 0, 112), types=('c', 'c', 'sdta'))
    return _create_microvit('microvitv2_s3', pretrained=pretrained, **dict(model_args, **kwargs))
