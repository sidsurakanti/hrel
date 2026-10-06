"""FLOPs / params of every config we ran (and the paper's), vs the unpruned nets"""

from flops import vgg_cost, resnet_cost

# hrank: surviving = n - int(rate * n); schedules from hrank/vgg_prune.py, resnet_prune.py
HRANK_VGG_RATE = ([0.95] + [0.5] * 6 + [0.9] * 4 + [0.8] * 2)[:12] + [0.0]
HRANK_RESNET_RATE = [0.1] + [0.60] * 35 + [0.0] * 2 + [0.6] * 6 + [0.4] * 3 + [0.1, 0.4] * 4
VGG_FULL = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
RESNET_FULL = [16] * 18 + [32] * 18 + [64] * 18

# hrel: paper's HRel-1 final widths
HREL_VGG = [21, 48, 64, 64, 95, 107, 107, 175, 71, 71, 44, 44, 56]
HREL_RESNET = [10] * 18 + [20] * 18 + [38] * 18


def keep(full, rates):
    return [n - int(r * n) for n, r in zip(full, rates)]


rows = []

base = vgg_cost(VGG_FULL)
rows.append(("VGG-16-bn", "unpruned", base, base))
rows.append(("VGG-16-bn", "hrank", vgg_cost(keep(VGG_FULL, HRANK_VGG_RATE)), base))
rows.append(("VGG-16-bn", "hrel (HRel-1)", vgg_cost(HREL_VGG), base))

# hrank's README counts all 13 rates and keeps conv inputs dense (reproduces its 108.61M)
readme_rate = [0.95] + [0.5] * 6 + [0.9] * 4 + [0.8] * 2
readme = vgg_cost(keep(VGG_FULL, readme_rate), dense_input=True)
rows.append(("VGG-16-bn (hrank's own count)", "hrank README", readme, vgg_cost(VGG_FULL, dense_input=True)))

for dense in (False, True):
    tag = "dense stream" if dense else "paper convention"
    base = resnet_cost(16, RESNET_FULL, dense)
    stem, *rest = keep([16] + RESNET_FULL, HRANK_RESNET_RATE)
    rows.append((f"ResNet-56 ({tag})", "unpruned", base, base))
    rows.append((f"ResNet-56 ({tag})", "hrank", resnet_cost(stem, rest, dense), base))
    rows.append((f"ResNet-56 ({tag})", "hrel (HRel-1)", resnet_cost(16, HREL_RESNET, dense), base))

print(f"{'net':32} {'method':15} {'FLOPs':>9} {'pruned':>7} {'params':>8} {'pruned':>7}")
for net, method, (m, p), (bm, bp) in rows:
    print(f"{net:32} {method:15} {m / 1e6:8.2f}M {100 * (1 - m / bm):6.1f}% {p / 1e6:7.2f}M {100 * (1 - p / bp):6.1f}%")
