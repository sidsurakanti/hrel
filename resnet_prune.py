"""
HRel reproduction on chenyaofo's pretrained resnet56 + CIFAR-10

NOTE:
54 convs are pruned: the stem conv is left alone (as in the author's code)
each stage prunes a fixed number of filters per round and stops at a fixed width
final widths are the paper's HRel-1 (62.06% flops pruned, 93.19% reported)
downsample shortcuts are left dense (same as the hrank repro)

chenyaofo's BasicBlock shares one `relu`, so conv1's activation is only visible
as conv2's input (pre-hook) and conv2's activation is the block output
"""

import os

import torch
import hrel_core as core

ARCH = "cifar10_resnet56"

# per stage (16 / 32 / 64 wide): final width, and filters lost per layer per round
TARGETS = [10, 20, 38]
STEP = [1, 2, 4]
BLOCKS_PER_STAGE = 9

SIGMA_BATCHES = 20
RELEVANCE_BATCHES = int(os.environ.get("RELEVANCE_BATCHES", 20))  # paper: a full epoch
FINETUNE_EPOCHS = int(os.environ.get("FINETUNE_EPOCHS", 4))  # paper: 100 per round


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

    blocks = [
        block for layer in (model.layer1, model.layer2, model.layer3) for block in layer
    ]

    # flatten layers: conv1, conv2 of every block (stem excluded)
    convs, bns, registrars = [], [], []
    for block in blocks:
        convs += [block.conv1, block.conv2]
        bns += [block.bn1, block.bn2]
        registrars.append(
            lambda tap, c2=block.conv2: c2.register_forward_pre_hook(tap.pre_hook)
        )
        registrars.append(lambda tap, b=block: b.register_forward_hook(tap.hook))

    # stage of layer i: 2 convs per block, 9 blocks per stage
    stage = [i // (2 * BLOCKS_PER_STAGE) for i in range(len(convs))]
    targets = [TARGETS[s] for s in stage]
    per_round = [lambda alive, k=STEP[s]: k for s in stage]

    print(f"\n== calibrating sigma for all {len(convs)} conv layers ==")
    sigmas = core.calibrate_sigmas(model, train_loader, SIGMA_BATCHES, registrars)

    core.iterative_prune(
        model,
        convs,
        bns,
        registrars,
        sigmas,
        per_round,
        targets,
        train_loader,
        test_loader,
        FINETUNE_EPOCHS,
        RELEVANCE_BATCHES,
    )

    final_acc = core.evaluate(model, test_loader)
    print(f"\nbaseline: {baseline_acc:.2f}%  ->  final (HRel-1 widths): {final_acc:.2f}%")


if __name__ == "__main__":
    main()
