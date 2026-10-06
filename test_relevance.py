"""
sanity checks for the relevance scorer

filters are given the same marginal statistics so only label dependence differs
(mi at a fixed sigma also tracks activation scale, a property of the method)
"""

import torch
import relevance as R

torch.manual_seed(0)
s, classes = 100, 10
y = torch.randint(0, classes, (s,))
k_y = R.label_kernel(y, classes)

# exact identity: a class-coded map with tiny sigma gives k_x == k_y, so mi == H(Y)
coded = torch.nn.functional.one_hot(y, classes).float()[:, None, :, None]
assert torch.isclose(R.relevance(coded, k_y, 0.1)[0], R.entropy(k_y), atol=1e-3)

# class prototypes: filter 0 follows the labels, filter 1 follows shuffled labels
protos = torch.randn(classes, 64)
noise = lambda: 0.3 * torch.randn(s, 64)
acts = torch.stack(
    [
        protos[y] + noise(),
        protos[y[torch.randperm(s)]] + noise(),
        torch.ones(s, 64),
    ],
    dim=1,
).reshape(s, 3, 8, 8)

sigma = R.calibrate_sigma(acts[:, :2], k_y)
rel = R.relevance(acts, k_y, sigma)
print("sigma", round(float(sigma), 2))
print("relevance [label-linked, shuffled, constant]:", [round(v, 3) for v in rel.tolist()])

assert rel[0] > rel[1], "label-linked filter should outscore the shuffled one"
assert rel[2].abs() < 1e-3, "constant map has zero mi"
print("ok")
