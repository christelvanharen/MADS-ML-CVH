# Summary week 2

Find the [notebook](./notebook.ipynb) and the [instructions](./instructions.md).

[Go back to Homepage](../README.md)

## Hypothesis
I expected validation accuracy to improve when the first hidden layer offered enough capacity to extract features (≥128 units) while the second layer stayed slightly narrower (64–256 units) to encourage generalisation. Very small layers would likely underfit, while identical widths might converge more slowly.

## Experiment design
Dataset: Fashion-MNIST via mads_datasets, batch size 64, default preprocessor.
Model: Two-hidden-layer MLP (units1, units2) with ReLU activations.
Training: 3 epochs with Adam (lr=1e-3) and cross-entropy loss.
Grid: units1, units2 ∈ {64, 128, 256}.
Logging: TensorBoard scalars and TOML configs per run.

## Findings
Best run: units1=128, units2=256 reached validation accuracy ≈0.872 with validation loss ≈0.355.
Increasing the first layer from 64 → 128 units improved accuracy across all second-layer sizes.
Very wide second layers (e.g. 256 when units1=256) showed diminishing returns and sometimes reduced accuracy.
All setups converged quickly within 3 epochs, with larger first layers stabilising loss earlier.