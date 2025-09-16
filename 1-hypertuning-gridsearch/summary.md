# Summary week 1

Find the [notebook](./notebook.ipynb) and the [instructions](./instructions.md)

[Go back to Homepage](../README.md)

## Hypothesis

I expected that:
Deeper models (3 layers) would outperform shallower ones (2 layers), but only if the hidden layers were wide enough.
More units would increase validation accuracy, up to the point where overfitting would appear.
Adam with a moderate learning rate (1e-3) would train faster and reach higher accuracy than SGD.
Batch size would mainly affect stability: smaller batches more noisy but slightly better generalization; larger batches smoother but less accurate.

## Findings
Optimizer: Adam consistently performed better than SGD, confirming part of the hypothesis.
Learning rate: 1e-3 was most reliable; 1e-2 was unstable, 1e-4 too slow.
Depth vs Units:
Depth=2 worked fine with small hidden sizes (64).
Depth=3 only helped when units ≥256; otherwise it overfitted or converged slower.
Batch size: Smaller batches gave slightly better validation accuracy, but training curves were noisier.
Overall, the best configuration was: Depth=3, Units=256, Batch=32, Adam, LR=1e-3, 10 epochs.

## Visualization
The heatmap below shows the relationship between Units1 (y-axis) and Depth (x-axis), with color indicating best validation accuracy across other hyperparameters:
