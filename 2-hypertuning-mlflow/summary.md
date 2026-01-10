# Summary week 2

Find the [notebook](./notebook.ipynb) and the [instructions](./instructions.md).

[Go back to Homepage](../README.md)

## Hypothesis
I expected convolutional feature extractors with batch normalisation and dropout to outperform the earlier MLP baseline. Wider early conv blocks should capture richer texture patterns, while slightly stronger dropout would mitigate overfitting caused by the extra capacity.

## Experiment design
Dataset: Fashion-MNIST via `mads_datasets`, batch size 64, default preprocessor.
Architecture: ConvClassifier with two conv blocks (Conv2d → BatchNorm2d → ReLU → MaxPool2d → Dropout) followed by a linear head with BatchNorm1d and dropout.
Hyperparameter grid: conv_channels ∈ {(32, 64), (64, 128)}, linear_units ∈ {(128,), (256,)}, dropout ∈ {0.25, 0.35}.
Training: 3 epochs, Adam (lr=1e-3, weight_decay=1e-5), ReduceLROnPlateau scheduler, early-stop monitor disabled.
Logging: TensorBoard scalars, TOML configs, and MLflow (complete with signatures/input examples).

## Findings
Best run (conv 64-128, linear 256, dropout 0.35) reached validation accuracy 0.9048 — a clear gain over the ≈0.87 MLP baseline.
Increasing conv channel widths consistently helped; the deeper head benefited from pairing the larger conv stack with 256 hidden units.
Dropout 0.35 gave the most stable validation curve for the high-capacity models, whereas 0.25 sometimes overfit after epoch two.
Adding batch normalisation and logging the model signature in MLflow made comparison and deployment readiness smoother.
