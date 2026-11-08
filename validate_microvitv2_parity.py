"""Validate MicroViTv2 logit parity against the reference implementation.

Runs OUTSIDE CI (downstream of the port): it needs a checkout of the reference repo
(https://github.com/novendrastywn/MicroViT) with its einops dependency installed, and the released
checkpoint files. It loads the same weights into the reference MicroViTv2 and the timm port (through
`checkpoint_filter_fn`) and compares logits on random input noise.

Usage:
    python validate_microvitv2_parity.py --ref-repo /path/to/MicroViT \
        --checkpoint microvitv2_s1.pth --model microvitv2_s1
"""
import argparse
import importlib
import sys

import torch

import timm
from timm.models.microvit import checkpoint_filter_fn

# variant kwargs of the reference model/build.py entries
REF_VARIANTS = {
    'microvitv2_s1': dict(dims=(128, 224, 320), depths=(2, 6, 5)),
    'microvitv2_s2': dict(dims=(128, 308, 448), depths=(2, 7, 5)),
    'microvitv2_s3': dict(dims=(192, 384, 448), depths=(3, 7, 6)),
}


def main():
    parser = argparse.ArgumentParser(description='Validate MicroViTv2 port parity against the reference')
    parser.add_argument('--ref-repo', required=True, help='path of the MicroViT reference repo checkout')
    parser.add_argument('--checkpoint', required=True, help='path of the released reference checkpoint')
    parser.add_argument('--model', default='microvitv2_s1', choices=list(REF_VARIANTS),
        help='name of the variant to validate')
    parser.add_argument('--num-classes', type=int, default=1000)
    args = parser.parse_args()

    # import model/microvitv2.py from the reference checkout (not a package, plain module import)
    sys.path.insert(0, args.ref_repo + '/model')
    reference = importlib.import_module('microvitv2')

    torch.manual_seed(0)
    x = torch.randn(2, 3, 224, 224)

    variant = REF_VARIANTS[args.model]
    ref_model = reference.MicroViTv2(
        num_classes=args.num_classes, dims=variant['dims'], depths=variant['depths'],
        type=['c', 'c', 'sdta'], attn_cr=[0, 0, 0.25], patch_size=16,
        act_layer=torch.nn.SiLU, final_feature=None, distillation=False,
    ).eval()

    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    state_dict = checkpoint.get('model', checkpoint) if isinstance(checkpoint, dict) else checkpoint
    ref_model.load_state_dict(state_dict)

    timm_model = timm.create_model(args.model, num_classes=args.num_classes).eval()
    timm_model.load_state_dict(checkpoint_filter_fn(state_dict, timm_model))

    with torch.no_grad():
        ref_logits = ref_model(x)
        timm_logits = timm_model(x)
    max_diff = (ref_logits - timm_logits).abs().max().item()
    print(f'{args.model}: max|logit diff| = {max_diff:.3e}')
    if not torch.allclose(ref_logits, timm_logits, atol=1e-4, rtol=1e-4):
        raise SystemExit(f'FAIL: logit parity broken for {args.model}')
    print(f'PASS: {args.model} matches the reference')


if __name__ == '__main__':
    main()
