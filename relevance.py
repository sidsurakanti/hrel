"""
HRel relevance scoring
======================

relevance of a filter = mutual information between its activation map and the labels

--------------
tldr pseudocode

    for one batch of s samples (s=100):
        K_y = same-class indicator kernel on one-hot labels (sigma = 0.1)

        for each filter:
            a    = its activation map, flattened to (s, h*w)
            K_a  = gaussian kernel on pairwise distances of a (per-layer sigma)
            I    = H(K_a) + H(K_y) - H(K_a * K_y)       # * is hadamard

    where H(K) = -sum(l * log2(l)) over eigenvalues l of K / trace(K)

    the sigma for each layer is picked once: the sigma that best aligns the
    layer's full output kernel with K_y (see `calibrate_sigma`)

mirrors the author's code (VGG/vgg_network.py), batched over filters
"""

import torch

LABEL_SIGMA = 0.1


def gaussian_kernel(x, sigma):
    # x: (..., s, d) -> (..., s, s), exp(-dist^2 / sigma^2)
    d = torch.cdist(x, x)
    return torch.exp(-(d**2) / sigma**2)


def entropy(k):
    # k: (..., s, s) kernel with unit diagonal -> normalize by trace, then -sum(l log2 l)
    k = k / k.diagonal(dim1=-2, dim2=-1).sum(-1)[..., None, None]
    try:
        eig = torch.linalg.eigvalsh(k)
    except torch.linalg.LinAlgError:
        # gpu solver can fail on near-rank-1 kernels (dead filters), cpu float64 is robust
        eig = torch.linalg.eigvalsh(k.double().cpu()).to(k)
    eig = eig.clamp(min=0)
    return -torch.xlogy(eig, eig).sum(-1) / torch.log(torch.tensor(2.0))


def label_kernel(y, num_classes):
    onehot = torch.nn.functional.one_hot(y, num_classes).float()
    return gaussian_kernel(onehot, LABEL_SIGMA)


def relevance(acts, k_y, sigma):
    """
    acts: (s, channels, h, w) activations of one layer, k_y: (s, s) label kernel
    returns (channels,) MI of each channel's map with the labels
    """
    s, c = acts.shape[:2]
    a = acts.detach().float().permute(1, 0, 2, 3).reshape(c, s, -1)

    k_a = gaussian_kernel(a, sigma)
    h_a = entropy(k_a)
    h_y = entropy(k_y)
    h_joint = entropy(k_a * k_y)

    return h_a + h_y - h_joint


def calibrate_sigma(acts, k_y, num_grid=100):
    """
    pick the sigma whose full-layer kernel best aligns with the label kernel
    (the author's rule: maximize 2*log2(tr(K_l K_y) / (|K_l| |K_y|)) over a grid)
    """
    x = acts.detach().float().flatten(1)
    d = torch.cdist(x, x)
    grid = torch.linspace(0.3, 10 * d.mean().item(), num_grid, device=x.device)

    k_l = torch.exp(-(d**2) / grid[:, None, None] ** 2) / x.shape[0]  # (grid, s, s)
    align = (k_l * k_y).sum((1, 2)) / (k_l.flatten(1).norm(dim=1) * k_y.norm())

    return grid[(2 * torch.log2(align)).argmax()]
