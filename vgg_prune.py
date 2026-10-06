"""
HRel reproduction on chenyaofo's pretrained vgg16_bn + CIFAR-10

NOTE:
all 13 conv layers are pruned (unlike hrank, which skips the last one)
final widths are the paper's HRel-1 (84.7% flops pruned, 93.54% reported)
"""

import os

import torch
import hrel_core as core

ARCH = "cifar10_vgg16_bn"

VGG16_CFG = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M"]
VGG16_CFG += [512, 512, 512, "M", 512, 512, 512, "M"]

# paper's HRel-1 final widths per conv layer
TARGETS = [21, 48, 64, 64, 95, 107, 107, 175, 71, 71, 44, 44, 56]

# each round a layer loses (this % of its remaining filters) + 1, from the author's code
PRUNE_PCT = [0.02] * 2 + [0.04] * 2 + [0.05] * 3 + [0.1] * 6

SIGMA_BATCHES = 20
RELEVANCE_BATCHES = int(os.environ.get("RELEVANCE_BATCHES", 20))  # paper: a full epoch
FINETUNE_EPOCHS = int(os.environ.get("FINETUNE_EPOCHS", 4))  # paper: 90 per round


def conv_bn_relu_indices(cfg):
    """
    Positions of the Conv2d/BatchNorm2d/ReLU triplets inside chenyaofo's
    `features` Sequential, derived from the VGG-D cfg list
    """
    conv_idx, bn_idx, relu_idx = [], [], []
    idx = 0

    for v in cfg:
        if v == "M":
            idx += 1
        else:
            conv_idx.append(idx)
            bn_idx.append(idx + 1)
            relu_idx.append(idx + 2)
            idx += 3

    return conv_idx, bn_idx, relu_idx


def main():
    train_loader, test_loader = core.get_loaders()

    print(f"== loading pretrained {ARCH} ==")
    model = torch.hub.load(
        "chenyaofo/pytorch-cifar-models",
        ARCH,
        pretrained=True,
        trust_repo=True,  # pyright: ignore[reportArgumentType]
    )
    model = model.to(core.DEVICE)

    baseline_acc = core.evaluate(model, test_loader)
    print(f"pretrained baseline accuracy: {baseline_acc:.2f}%")

    conv_idx, bn_idx, relu_idx = conv_bn_relu_indices(VGG16_CFG)
    features = model.features
    convs = [features[i] for i in conv_idx]
    bns = [features[i] for i in bn_idx]
    registrars = [
        lambda tap, r=features[i]: r.register_forward_hook(tap.hook) for i in relu_idx
    ]

    print(f"\n== calibrating sigma for all {len(convs)} conv layers ==")
    sigmas = core.calibrate_sigmas(model, train_loader, SIGMA_BATCHES, registrars)
    print(f"    sigmas: {[round(float(s), 1) for s in sigmas]}")

    per_round = [lambda alive, p=p: int(alive * p) + 1 for p in PRUNE_PCT]
    core.iterative_prune(
        model,
        convs,
        bns,
        registrars,
        sigmas,
        per_round,
        TARGETS,
        train_loader,
        test_loader,
        FINETUNE_EPOCHS,
        RELEVANCE_BATCHES,
    )

    final_acc = core.evaluate(model, test_loader)
    print(f"\nbaseline: {baseline_acc:.2f}%  ->  final (HRel-1 widths): {final_acc:.2f}%")


if __name__ == "__main__":
    main()
