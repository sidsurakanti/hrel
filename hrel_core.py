"""
Shared HRel primitives
======================

- data loading (CIFAR-10)
- relevance scoring (see relevance.py) hooked into a model
- masking
- the iterative prune -> finetune loop

* architecture-specific code (which convs to target, how to reach their activations)
  lives in vgg_prune.py / resnet_prune.py

--------------
tldr pseudocode

    STEP 1: CALIBRATE SIGMA (once, pretrained model)
    -----
    for each conv layer:
        hook its post-relu output, run a few batches
        => sigma[layer] = best-aligned-with-labels kernel width (EMA over batches)

    STEP 2: ITERATIVE PRUNE
    ------
    masks = all ones
    while any layer is above its target width:
        relevance = MI(filter activation, labels) averaged over a few batches
                    (recomputed every round, on the current masked model)

        for each layer above its target:
            mask the lowest-relevance remaining filters (+ their BN)

        finetune the whole network for a few epochs,
            reapplying every mask after each optimizer step

    NOTE: this is all soft pruning (same as hrank)
    NOTE: unlike hrank, scores are recomputed every round, and all layers are pruned together
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import relevance as R

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 10


def get_loaders(batch_size=100):  # HRel uses 100 for both mi estimation and training
    mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    train_tf = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    test_tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=train_tf
    )
    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=test_tf
    )

    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,  # mi needs full batches of 100
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )
    test_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=256,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    return train_loader, test_loader


def evaluate(model, loader):
    model.eval()
    correct = total = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    return 100.0 * correct / total


def apply_mask(conv, bn, mask):
    mask = mask.to(conv.weight.device)
    with torch.no_grad():
        conv.weight.data *= mask.view(-1, 1, 1, 1)
        bn.weight.data *= mask
        bn.bias.data *= mask


def train_epochs(model, loader, epochs, masks=(), lr=0.01):
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    # decay at the same fractions of the run as the paper's 40/70 of 90 epochs
    milestones = [max(1, round(epochs * 40 / 90)), max(1, round(epochs * 70 / 90))]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        for x, y in loader:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            for conv, bn, mask in masks:
                apply_mask(conv, bn, mask)  # keep pruned filters at zero
        scheduler.step()
        print(f"    epoch {epoch + 1}/{epochs} done")


class Tap:
    """base: forwards a hooked activation (and the batch's label kernel) to `process`"""

    k_y: torch.Tensor

    def process(self, acts):
        raise NotImplementedError

    def hook(self, _module, _inp, output):  # for forward hook
        self.process(output)

    def pre_hook(self, _module, inp):  # for pre-forward hooks
        self.process(inp[0])


class SigmaTap(Tap):
    """EMA (0.9 / 0.1, like the author's code) of the best sigma over batches"""

    def __init__(self):
        self.sigma: torch.Tensor | None = None

    def process(self, acts):
        new = R.calibrate_sigma(acts, self.k_y)
        self.sigma = new if self.sigma is None else 0.9 * self.sigma + 0.1 * new


class RelevanceTap(Tap):
    """running average of per-filter relevance across batches"""

    def __init__(self, sigma):
        self.sigma = sigma
        self.total: torch.Tensor | None = None
        self.batches = 0

    def process(self, acts):
        rel = R.relevance(acts, self.k_y, self.sigma)
        self.total = rel if self.total is None else self.total + rel
        self.batches += 1

    def result(self):
        assert self.total is not None, "no batches were run through the hook yet"
        return (self.total / self.batches).cpu()


def run_taps(model, loader, num_batches, registrars, taps):
    """
    `registrars` is one `f(tap) -> handle` callable per target layer
    (a plain forward hook for most layers, a pre-hook where the activation
    is only visible as the next conv's input)
    """
    handles = [register(tap) for register, tap in zip(registrars, taps)]

    model.eval()
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if i >= num_batches:
                break
            k_y = R.label_kernel(y.to(DEVICE), NUM_CLASSES)
            for tap in taps:
                tap.k_y = k_y
            model(x.to(DEVICE))
    for h in handles:
        h.remove()


def calibrate_sigmas(model, loader, num_batches, registrars):
    taps = [SigmaTap() for _ in registrars]
    run_taps(model, loader, num_batches, registrars, taps)
    return [tap.sigma for tap in taps]


def compute_relevances(model, loader, num_batches, registrars, sigmas):
    taps = [RelevanceTap(sigma) for sigma in sigmas]
    run_taps(model, loader, num_batches, registrars, taps)
    return [tap.result() for tap in taps]


def prune_lowest(mask, relevance, n):
    """zero the `n` lowest-relevance filters among those still alive"""
    alive = mask.nonzero().squeeze(1)
    worst = alive[relevance[alive].argsort()[:n]]
    mask[worst] = 0.0


def iterative_prune(
    model,
    convs,
    bns,
    registrars,
    sigmas,
    per_round,
    targets,
    train_loader,
    test_loader,
    finetune_epochs,
    relevance_batches,
):
    """
    the outer loop for hrel

    `per_round[i](alive)` = how many filters layer i loses this round,
    `targets[i]` = final width of layer i
    """
    masks = [torch.ones(c.out_channels) for c in convs]
    triples = list(zip(convs, bns, masks))

    round_idx = 0
    while any(m.sum() > t for m, t in zip(masks, targets)):
        round_idx += 1
        print(f"\n== ROUND {round_idx}: scoring filters ==")
        rel = compute_relevances(
            model, train_loader, relevance_batches, registrars, sigmas
        )

        # prune every layer that is still above its target
        for conv, bn, mask, r, step, target in zip(
            convs, bns, masks, rel, per_round, targets
        ):
            alive = int(mask.sum())
            if alive > target:
                prune_lowest(mask, r, min(step(alive), alive - target))
                apply_mask(conv, bn, mask)

        widths = [int(m.sum()) for m in masks]
        print(f"    widths: {widths}")

        print(f"== ROUND {round_idx}: FINETUNING ({finetune_epochs} epochs) ==")
        train_epochs(model, train_loader, finetune_epochs, masks=triples)
        print(f"    test acc: {evaluate(model, test_loader):.2f}%")
