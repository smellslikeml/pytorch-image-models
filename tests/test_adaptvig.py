"""AdaptViG model tests.

Parity is verified against a vendored copy of the reference implementation
(https://github.com/mmunir127/AdaptViG, models/adaptvig.py, Apache-2.0). The oracle classes
below are intentionally verbatim copies of the reference math (with the module-level cuda
device global removed) -- do not "fix" them to match this port; their job is to be the
reference. Weights flow reference -> timm through `checkpoint_filter_fn`, so the key map and
the `dist_head` -> `head_dist` rename are exercised on every load.
"""

from typing import List

import pytest
import torch
import torch.nn as nn

import timm
from timm.layers import DropPath
from timm.models.adaptvig import checkpoint_filter_fn


class RefStem(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(RefStem, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(input_dim, output_dim // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(output_dim // 2),
            nn.GELU(),
            nn.Conv2d(output_dim // 2, output_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(output_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.stem(x)


class RefDepthWiseSeparable(nn.Module):
    def __init__(self, in_dim, kernel, expansion=4):
        super().__init__()
        self.pw1 = nn.Conv2d(in_dim, in_dim * 4, 1)
        self.norm1 = nn.BatchNorm2d(in_dim * 4)
        self.act1 = nn.GELU()
        self.dw = nn.Conv2d(in_dim * 4, in_dim * 4, kernel_size=kernel, stride=1, padding=1, groups=in_dim * 4)
        self.norm2 = nn.BatchNorm2d(in_dim * 4)
        self.act2 = nn.GELU()
        self.pw2 = nn.Conv2d(in_dim * 4, in_dim, 1)
        self.norm3 = nn.BatchNorm2d(in_dim)

    def forward(self, x):
        x = self.pw1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.dw(x)
        x = self.norm2(x)
        x = self.act2(x)
        x = self.pw2(x)
        x = self.norm3(x)
        return x


class RefInvertedResidual(nn.Module):
    def __init__(self, dim, kernel, expansion_ratio=4., drop=0., drop_path=0., use_layer_scale=True,
                 layer_scale_init_value=1e-5):
        super().__init__()
        self.dws = RefDepthWiseSeparable(in_dim=dim, kernel=kernel, expansion=expansion_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.dws(x))
        else:
            x = x + self.drop_path(self.dws(x))
        return x


class RefAdaptConv(nn.Module):
    def __init__(self, in_channels, out_channels, K=2, **kwargs):
        super().__init__()
        self.K = K
        # Learnable temperature parameter for the exponential decay
        self.temperature = nn.Parameter(torch.tensor(1.0))
        self.nn = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x_j = x - x  # Initialize with zeros

        # 1. Guaranteed Local Connections (Always On)
        x_c_local = torch.cat([x[:, :, -self.K:, :], x[:, :, :-self.K, :]], dim=2)
        x_j = torch.max(x_j, x_c_local - x)
        x_r_local = torch.cat([x[:, :, :, -self.K:], x[:, :, :, :-self.K]], dim=3)
        x_j = torch.max(x_j, x_r_local - x)

        # 2. Gated Long-Range Connections using Exponential Decay
        Hbit, Wbit = H.bit_length(), W.bit_length()
        for i in range(1, Hbit):
            dist_exp = 2 ** i
            neighbor = torch.cat([x[:, :, -dist_exp:, :], x[:, :, :-dist_exp, :]], dim=2)
            dist = torch.norm(x - neighbor, p=1, dim=1, keepdim=True)
            # Use abs() + eps on temperature to ensure it's a valid, non-zero divisor
            gate = torch.exp(-dist / (torch.abs(self.temperature) + 1e-6))
            x_j = torch.max(x_j, (neighbor - x) * gate)

        for i in range(1, Wbit):
            dist_exp = 2 ** i
            neighbor = torch.cat([x[:, :, :, -dist_exp:], x[:, :, :, :-dist_exp]], dim=3)
            dist = torch.norm(x - neighbor, p=1, dim=1, keepdim=True)
            gate = torch.exp(-dist / (torch.abs(self.temperature) + 1e-6))
            x_j = torch.max(x_j, (neighbor - x) * gate)

        x = torch.cat([x, x_j], dim=1)
        return self.nn(x)


class RefGlobalAttentionGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, K=2, **kwargs):
        super().__init__()
        self.K = K
        self.q_conv = nn.Conv2d(in_channels, in_channels // 4, 1, bias=False)
        self.k_conv = nn.Conv2d(in_channels, in_channels // 4, 1, bias=False)
        self.v_conv = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.q_norm = nn.LayerNorm(in_channels // 4)
        self.k_norm = nn.LayerNorm(in_channels // 4)
        self.softmax = nn.Softmax(dim=-1)
        self.nn = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x_j = x - x
        q = self.q_conv(x).view(B, -1, H * W).permute(0, 2, 1)
        k = self.k_conv(x).view(B, -1, H * W).permute(0, 2, 1)
        v = self.v_conv(x).view(B, -1, H * W).permute(0, 2, 1)
        q = self.q_norm(q)
        k = self.k_norm(k)
        attn = self.softmax(torch.bmm(q, k.transpose(-2, -1)) / (C // 4) ** 0.5)
        x_attn = torch.bmm(attn, v).permute(0, 2, 1).view(B, C, H, W)
        x_j = torch.max(x_j, x_attn - x)
        x = torch.cat([x, x_j], dim=1)
        return self.nn(x)


class RefConditionalPositionEncoding(nn.Module):
    def __init__(self, in_channels, kernel_size):
        super().__init__()
        self.pe = nn.Conv2d(
            in_channels=in_channels, out_channels=in_channels,
            kernel_size=kernel_size, stride=1, padding=kernel_size // 2,
            bias=True, groups=in_channels
        )

    def forward(self, x):
        x = self.pe(x) + x
        return x


class RefAdaptiveGrapher(nn.Module):
    def __init__(self, in_channels, K, graph_conv_type='gating'):
        super(RefAdaptiveGrapher, self).__init__()
        self.cpe = RefConditionalPositionEncoding(in_channels, kernel_size=7)
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        if graph_conv_type == 'gating':
            self.graph_conv = RefAdaptConv(in_channels, in_channels, K=K)
        else:  # 'global'
            self.graph_conv = RefGlobalAttentionGraphConv(in_channels, in_channels, K=K)
        self.fc2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )

    def forward(self, x):
        x = self.cpe(x)
        x = self.fc1(x)
        x = self.graph_conv(x)
        x = self.fc2(x)
        return x


class RefAdaptiveGraphConvBlock(nn.Module):
    def __init__(self, in_dim, drop_path=0., K=2, graph_conv_type='gating', use_layer_scale=True,
                 layer_scale_init_value=1e-5):
        super().__init__()
        self.mixer = RefAdaptiveGrapher(in_dim, K, graph_conv_type)
        self.ffn = nn.Sequential(
            nn.Conv2d(in_dim, in_dim * 4, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_dim * 4),
            nn.GELU(),
            nn.Conv2d(in_dim * 4, in_dim, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_dim),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones(in_dim), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones(in_dim), requires_grad=True)

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.mixer(x))
            x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.ffn(x))
        else:
            x = x + self.drop_path(self.mixer(x))
            x = x + self.drop_path(self.ffn(x))
        return x


class RefDownsample(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        return self.conv(x)


class RefAdaptViG(nn.Module):
    """Reference AdaptViG (dual head named `dist_head`, as in the released checkpoints)."""

    def __init__(self, blocks, channels, kernels, dropout=0., drop_path=0., emb_dims=512,
                 K=2, distillation=True, num_classes=1000):
        super(RefAdaptViG, self).__init__()
        self.distillation = distillation
        n_blocks = sum([sum(x) for x in blocks])
        dpr = [x.item() for x in torch.linspace(0, drop_path, n_blocks)]
        dpr_idx = 0
        self.stem = RefStem(input_dim=3, output_dim=channels[0])
        self.backbone = nn.ModuleList()
        num_stages = len(blocks)
        for i in range(num_stages):
            stage = []
            local_stages, global_stages = blocks[i]
            graph_conv_type = 'global' if i == num_stages - 1 else 'gating'
            if i > 0:
                stage.append(RefDownsample(channels[i - 1], channels[i]))
            for _ in range(local_stages):
                stage.append(RefInvertedResidual(dim=channels[i], kernel=kernels, expansion_ratio=4,
                                                 drop_path=dpr[dpr_idx]))
                dpr_idx += 1
            for _ in range(global_stages):
                stage.append(RefAdaptiveGraphConvBlock(channels[i], drop_path=dpr[dpr_idx], K=K[i],
                                                       graph_conv_type=graph_conv_type))
                dpr_idx += 1
            self.backbone.append(nn.Sequential(*stage))

        self.prediction = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels[-1], emb_dims, 1, bias=True),
            nn.BatchNorm2d(emb_dims),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.head = nn.Conv2d(emb_dims, num_classes, 1, bias=True)
        self.dist_head = nn.Identity()
        if self.distillation:
            self.dist_head = nn.Conv2d(emb_dims, num_classes, 1, bias=True)

        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, inputs):
        x = self.stem(inputs)
        for stage_block in self.backbone:
            x = stage_block(x)
        x = self.prediction(x)
        if self.distillation:
            x = self.head(x).squeeze(-1).squeeze(-1), self.dist_head(x).squeeze(-1).squeeze(-1)
            if not self.training:
                x = (x[0] + x[1]) / 2
        else:
            x = self.head(x).squeeze(-1).squeeze(-1)
        return x


REF_VARIANTS = {
    'adaptvig_s': dict(
        blocks=[[3, 3], [3, 3], [9, 3], [3, 3]], channels=[32, 64, 128, 256],
        kernels=3, dropout=0., drop_path=0.1, emb_dims=768, K=[8, 4, 2, 1],
        distillation=True, num_classes=1000),
    'adaptvig_m': dict(
        blocks=[[4, 4], [4, 4], [12, 4], [4, 4]], channels=[48, 96, 192, 320],
        kernels=3, dropout=0., drop_path=0.1, emb_dims=768, K=[8, 4, 2, 1],
        distillation=True, num_classes=1000),
    'adaptvig_b': dict(
        blocks=[[5, 5], [5, 5], [15, 5], [5, 5]], channels=[48, 96, 192, 384],
        kernels=3, dropout=0., drop_path=0.1, emb_dims=768, K=[8, 4, 2, 1],
        distillation=True, num_classes=1000),
}


@pytest.mark.parametrize('variant', ['adaptvig_s', 'adaptvig_m', 'adaptvig_b'])
def test_parity_vs_reference(variant: str):
    """timm port matches the reference impl bit-for-bit after a filtered state_dict copy."""
    torch.manual_seed(0)
    ref = RefAdaptViG(**REF_VARIANTS[variant])

    model = timm.create_model(variant, pretrained=False)

    # exercise the full checkpoint path: released files are {'state_dict': ..., 'state_dict_ema': ...}
    checkpoint = {'state_dict': ref.state_dict(), 'state_dict_ema': ref.state_dict()}
    model.load_state_dict(checkpoint_filter_fn(checkpoint, model), strict=True)

    ref.eval()
    model.eval()
    torch.manual_seed(0)
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        ref_out = ref(x)
        timm_out = model(x)

    assert isinstance(ref_out, torch.Tensor), 'reference eval output is the averaged dual-head logits'
    max_abs_diff = (ref_out - timm_out).abs().max().item()
    assert max_abs_diff < 1e-4, f'{variant} max-abs logit diff {max_abs_diff:.3e} exceeds 1e-4'


def test_features_only():
    """features_only=True returns the 4 pyramid taps at strides 4/8/16/32."""
    model = timm.create_model('adaptvig_s', pretrained=False, features_only=True)
    model.eval()
    assert model.feature_info.channels() == [32, 64, 128, 256]
    assert model.feature_info.reduction() == [4, 8, 16, 32]
    with torch.no_grad():
        feats: List[torch.Tensor] = model(torch.randn(2, 3, 224, 224))
    expected = [(2, 32, 56, 56), (2, 64, 28, 28), (2, 128, 14, 14), (2, 256, 7, 7)]
    assert [tuple(f.shape) for f in feats] == expected


def test_reset_classifier():
    """reset_classifier rewires both heads and the model still forwards."""
    model = timm.create_model('adaptvig_s', pretrained=False)
    model.reset_classifier(10)
    head, head_dist = model.get_classifier()
    assert head.out_channels == 10
    assert head_dist.out_channels == 10
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 10)
