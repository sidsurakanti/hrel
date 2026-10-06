# Results log

Pretrained checkpoints from [chenyaofo/pytorch-cifar-models](https://github.com/chenyaofo/pytorch-cifar-models),
pruned to the paper's HRel-1 widths.

## VGG-16-bn

| stage | accuracy |
|---|---|
| pretrained baseline | 94.15% |
| pruned, 0 epochs/round (no finetune) | 10.00% |
| pruned, 4 epochs/round, 35 rounds | 92.20% |
| pruned, 15 epochs/round | 93.00% |
| pruned, 90 epochs/round (paper budget) | **93.40%** |
| HRel-1 (paper, 90 epochs/round) | 93.54% |

Accuracy drifts from 92.8% (early rounds) to 92.2% at the end.

## ResNet-56

| stage | accuracy |
|---|---|
| pretrained baseline | 94.38% |
| pruned, 0 epochs/round (no finetune) | 16.82% |
| pruned, 4 epochs/round, 7 rounds | 91.97% |
| pruned, 15 epochs/round | 92.80% |
| pruned, 100 epochs/round (paper budget) | **93.20%** |
| HRel-1 (paper, 100 epochs/round) | 93.19% |

Per round: 93.29, 93.00, 92.88, 92.61, 92.25, 91.83, 91.97.

## Compression (structural count, `compare.py`)

| net | method | FLOPs | cut | params | cut |
|---|---|---|---|---|---|
| VGG-16-bn | unpruned | 313.73M | | 15.24M | |
| VGG-16-bn | hrank | 43.49M | 86.1% | 1.61M | 89.4% |
| VGG-16-bn | hrel | 48.12M | 84.7% | 1.01M | 93.4% |
| ResNet-56 | unpruned | 125.49M | | 0.85M | |
| ResNet-56 | hrank | 33.30M | 73.5% | 0.30M | 64.5% |
| ResNet-56 | hrel | 48.29M | 61.5% | 0.31M | 63.7% |

## hrank vs hrel, same convention

| net | method | accuracy | FLOPs cut | epochs total |
|---|---|---|---|---|
| VGG-16-bn | hrank (15/layer) | 91.25% | 86.1% | 180 |
| VGG-16-bn | hrel (4/round) | **92.20%** | 84.7% | 140 |
| ResNet-56 | hrank (15/layer) | **93.11%** | 73.5% | 825 |
| ResNet-56 | hrel (4/round) | 91.97% | 61.5% | 28 |

- VGG: similar compression and similar total epochs, hrel is about 1pp better
- ResNet: not a fair comparison yet, hrank gets 30x the epochs and prunes more

## FLOPs conventions

- **Structural (used above, and by the HRel paper).** A conv's input width is the previous conv's surviving width
  - reproduces the HRel paper's VGG numbers: 313.73M baseline, 84.7% cut for HRel-1 (paper: 47.98M, 84.70%)
  - ResNet HRel-1 gives 61.5% vs the paper's 62.06%
- **hrank's own VGG count keeps conv inputs dense**, which understates savings
  - it reproduces the README's 108.61M (we get 109.35M)
  - the same schedule is an 86% cut structurally, not 65%
- **ResNet "dense stream" count** (residual stream stays full width) is the more conservative view of what soft masking saves
  - hrank 61.4%, hrel 50.4%
- hrank's README ResNet figure (62.72M) is not reproduced by any variant tried, and the README says it was corrected after the paper
- Params count conv + fc weights only, and our VGG has a 3-layer head (paper: 2), so baseline is 15.24M not 14.98M
