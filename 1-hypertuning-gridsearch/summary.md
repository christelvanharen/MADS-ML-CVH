# Grid Search Report

Find the [notebook](./notebook.ipynb) and the [instructions](./instructions.md).

[Go back to Homepage](../README.md)

## Hypothesis
I expected that giving the model enough capacity in the first hidden layer (around 128 units) would help it learn useful features, while keeping the second layer a bit narrower (roughly 64–256 units) would encourage better generalisation. Very small layers would likely underfit, and making both layers equally wide might slow training.

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
