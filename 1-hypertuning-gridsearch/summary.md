# Grid Search Report

Find the [notebook](./notebook.ipynb) and the [instructions](./instructions.md).

[Go back to Homepage](../README.md)

## Hypothesis
Validation accuracy should improve when the first hidden layer is large enough (≥128 units) to extract features, while keeping the second layer slightly narrower (64–256 units) encourages generalisation. Very small layers may underfit; equal widths could converge slower.

## Experiment design
Dataset: Fashion-MNIST via mads_datasets, batch size 64, default preprocessor.
Model: Two-hidden-layer MLP (units1, units2) with ReLU activations.
Training: 3 epochs with Adam (lr=1e-3) and cross-entropy loss.
Grid: units1, units2 ∈ {64, 128, 256}.
Logging: TensorBoard scalars and TOML configs per run.

## Findings
Best run: units1=128, units2=256, reaching validation accuracy ≈0.872 with loss ≈0.355.
First-layer width had the strongest effect: moving from 64 → 128 improved accuracy across the board.
Extremely wide second layers (256 with units1=256) didn’t help and sometimes hurt performance.
Training converged within 3 epochs for all configs; larger first layers stabilised loss earlier.

## Visualisations
Line plots of training loss, validation loss, and validation accuracy per epoch show faster and smoother convergence for wider first layers.

## Reflection
The hypothesis mostly held: the first hidden layer’s capacity is critical, while a moderately sized second layer avoids overfitting. Future work could test longer training, dropout, or larger grids to confirm whether the 128→256 configuration is consistently best.
