# Summary week 3

Find the [notebook](./notebook.ipynb) and the [instructions](./instructions.md).

[Go back to Homepage](../README.md)

## Hypothesis
The baseline `BaseRNN` struggled to exploit the variable-length gesture sequences (≈53% accuracy after three epochs). I expected bidirectional recurrent layers with sequence-aware pooling and targeted regularisation to break past 90%. Adding learnable Conv1D front-ends could sculpt local motion patterns before the recurrent stack and further stabilise training.

## Experimental setup
- **Data**: SmartWatch gestures dataset via `mads_datasets`, batch size 32, padded to the longest sequence per batch.
- **Training loop**: `Trainer` with 80-epoch budget, cosine-style ReduceLROnPlateau scheduler (factor 0.5, patience 4), early stopping with patience 8, metrics = accuracy.
- **Optimisation**: Adam (`lr=1e-3`, weight decay default), gradient on CPU/MPS as available.
- **Logging**: TensorBoard + TOML configs in `gestures/`, MLflow tracking in `mlflow.db` (models stored under `mlruns`).
- **Evaluation**: Mean-pooled logits across valid timestamps; validation accuracy computed with a mask to ignore zero-padding.

## Architectures explored
| Run name | Architecture | Key hyperparameters | Best val acc. | Final val acc. |
| --- | --- | --- | --- | --- |
| `bigru_mean_pool` | 2-layer bidirectional GRU | hidden=128, dropout=0.3, mean pooling | 0.9938 | 0.9938 |
| `bilstm_dropout` | 2-layer bidirectional LSTM | hidden=160, dropout=0.35, mean pooling | 0.9938 | 0.9938 |
| `conv_bigru` | Conv1D (32→64, k=5, dropout 0.2) + 2-layer BiGRU | hidden=128, dropout=0.3 | 0.9938 | 0.9922 |

Notes:
- All recurrent stacks see true sequence lengths inferred from non-zero timesteps, ensuring pooling ignores padded frames.
- Conv front-end widens receptive field but slightly over-regularised with dropout 0.2, explaining the marginally lower final accuracy.

## What worked
1. **Bidirectionality + mean pooling**: Averaging the hidden trajectory was consistently ~4–5% better than naïvely reading the last step because zero padding otherwise skews logits.
2. **Moderate dropout (0.3–0.35)**: Keeps recurrent stacks from overfitting; validation accuracy plateaus near 0.99 within 20–25 epochs.
3. **Learning-rate scheduling**: ReduceLROnPlateau halved the learning rate twice; accuracy jumps from ~0.93 to ~0.99 immediately after the first decay.
4. **Conv1D front-end**: Adds slight robustness to jitter by aggregating short windows before the GRU, and still reaches >0.99 accuracy.

## What did not help
- **Shallow RNN without pooling**: The starter `BaseRNN` hit only ~0.53 accuracy and oscillated heavily because the last step often corresponded to padded zeros.
- **Excessive convolutional dropout**: Raising Conv1D dropout above 0.2 slowed convergence and produced noisier loss curves without improving generalisation.