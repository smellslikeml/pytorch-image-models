""" Convert MicroViTv2 weights from the reference release to timm

Checkpoints: https://github.com/novendrastywn/MicroViT

The reference packs the whole network into one flat nn.Sequential (stem, stage blocks, and the
pre-patch / merge / post-patch modules between stages as flat entries stages.0 .. stages.9). The remap
to the timm layout (stem + stages.{i}.downsample + stages.{i}.blocks) lives in
`checkpoint_filter_fn` in timm/models/microvit.py, so `pretrained=True` works on either layout and this
script is only needed to produce re-uploaded timm-layout checkpoints.

Usage:
    python convert/convert_microvitv2.py microvitv2_s1.pth --model microvitv2_s1 --output microvitv2_s1.in1k.pth
"""
import argparse
from typing import Any, Dict

import torch

import timm
from timm.models.microvit import checkpoint_filter_fn

parser = argparse.ArgumentParser(description='Convert reference MicroViTv2 weights to the timm layout')
parser.add_argument('checkpoint', metavar='CKPT', help='path of the reference checkpoint')
parser.add_argument('--model', default='microvitv2_s1', help='name of the target timm model')
parser.add_argument('--output', default='./converted.pth', help='output path of the converted checkpoint')


def convert_state_dict(state_dict: Dict[str, Any], model) -> Dict[str, torch.Tensor]:
    state_dict = state_dict.get('model', state_dict)
    return checkpoint_filter_fn(state_dict, model)


def main():
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']

    model = timm.create_model(args.model, pretrained=False)
    state_dict = convert_state_dict(checkpoint, model)
    model.load_state_dict(state_dict)  # strict, fails loudly on a layout mismatch
    torch.save(state_dict, args.output)
    print(f'Converted {len(state_dict)} tensors from {args.checkpoint} to {args.output}')


if __name__ == '__main__':
    main()
