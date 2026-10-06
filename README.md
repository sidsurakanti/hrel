# HRel reproduction

Reproducing [HRel: Filter Pruning based on High Relevance between Activation Maps and Class Labels](https://arxiv.org/abs/2202.10716) (Neural Networks 2022) on CIFAR-10

Sibling of [`../hrank`](../hrank), same checkpoints and same soft-masking approach

## Results

| network | ours | paper | FLOPs cut | retrain budget (ours vs paper) |
|---|---|---|---|---|
| VGG-16-bn | 92.20% | 93.54% | 84.7% | 4 vs 90 epochs/round |
| ResNet-56 | 91.97% | 93.19% | 61.5% | 4 vs 100 epochs/round |

Full tables, no-finetune controls, FLOPs conventions and the hrank comparison: `RESULTS.md`.

## The method

- Score each filter by the mutual information between its activation map and the class labels ("relevance")
- Low-relevance filters get pruned
- Pruning is iterative: every round prunes a few filters from every layer, then finetunes
- Relevance is recomputed every round on the current model (hrank scores once)

See `hrel_core.py`'s module docstring.

## Files

| file | what it is |
|---|---|
| `relevance.py` | the scorer: matrix-entropy MI per filter, plus per-layer sigma calibration |
| `hrel_core.py` | shared engine: data, hooks, masking, iterative prune->finetune loop |
| `vgg_prune.py` | reproduction on pretrained `vgg16_bn` |
| `resnet_prune.py` | reproduction on pretrained `resnet56` |
| `test_relevance.py` | sanity checks for the scorer |
| `flops.py`, `compare.py` | FLOPs / params from layer widths, for hrel and hrank |
| `RESULTS.md` | every run, its accuracy, and any gap to the paper |

## Known deviations from the paper

- **Retrain budget.** 4 epochs per round, not 90-100 (budget is the main gap, see `RESULTS.md`)
- **Relevance batches.** 20 batches per round, the paper averages a full epoch
- **Sigma.** Calibrated once on the pretrained model with the author's alignment rule, they calibrate during training from scratch
- **Pretrained checkpoints** are chenyaofo's, not the authors' own
- **resnet56 shortcut type.** Learned (option-B) downsample, left dense, as in `../hrank`
- **Final widths** are exact (the paper's HRel-1); we stop at the target instead of overshooting it
- **Estimator.** Follows the author's code (shannon entropy of the normalized kernel), not the alpha=2 in the paper text

## How to run

```bash
uv run python test_relevance.py   # sanity check: the scorer works
uv run python vgg_prune.py        # ~25 min on an RTX 4060
uv run python resnet_prune.py     # ~10 min
uv run python compare.py          # FLOPs / params table
```

Env vars: `FINETUNE_EPOCHS` (default 4, 0 = no-finetune control), `RELEVANCE_BATCHES` (default 20)
