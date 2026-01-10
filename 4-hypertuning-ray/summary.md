Hypertuning run — brief report
=============================

Date: 2026-01-05

Objective: execute the hypertuning script and collect a short journal + final report.

Files:
- Hypertuner: [4-hypertuning-ray/hypertune.py](4-hypertuning-ray/hypertune.py#L1-L200)

What I did:
- Inspected `hypertune.py` and attempted to run it from the workspace venv.
- Recorded progress and errors using `jrnl` (journal created in the working folder).

Result:
- The script failed immediately with: ModuleNotFoundError: No module named 'ray'.

Result and smoke test:
- After installing dependencies the smoke test (2 samples, 2 epochs) completed successfully. Two trials ran and terminated. Best trial had `test_loss=2.07293`, `Accuracy=0.2375` with params `hidden_size=65`, `dropout=0.05098`, `num_layers=4`.

Immediate next steps to run full hypertuning:

- Revert `NUM_SAMPLES` and `MAX_EPOCHS` in `4-hypertuning-ray/hypertune.py` back to the intended values (50 and 10) before running the full experiment.
- Optionally set up TensorBoard to monitor live metrics:

```bash
tensorboard --logdir logs/ray
```

- Run the full hypertune:

```bash
/Users/christelvanharen/MADS/Semester3/MADS-ML-CVH/.venv/bin/python 4-hypertuning-ray/hypertune.py
```

Notes:
- The script expects a gestures dataset at `data/raw/gestures/gestures-dataset` and will create the directory if missing. Full execution also requires `mltrainer` and `mads_datasets` to be available and may require additional heavy dependencies (e.g., PyTorch). If not present, install or adjust the environment accordingly.

Journal:
- I recorded the run attempts using `jrnl` during this session.

If you want, I can now install the missing dependencies and re-run a short test (small `NUM_SAMPLES`/`MAX_EPOCHS`) and append the results to the journal and this summary. Proceed? 

Final run (full hypertuning)
---------------------------

- Date: 2026-01-05
- Settings: `NUM_SAMPLES=50`, `MAX_EPOCHS=10` (default script values)
- Location of logs: `logs/ray/train_2026-01-05_11-51-20`
- Best trial: test_loss=0.08206186844035983, Accuracy=0.982812
- Best hyperparameters: `hidden_size=115`, `dropout≈0.009256`, `num_layers=3`

Notes:
- All 50 trials completed (many terminated early by AsyncHyperBand). The best models reached ~98% accuracy on the test set for this configuration. Full per-trial results and checkpoints are in the `logs/ray` run folder.
- To inspect metrics live or afterwards, use TensorBoard:

```bash
tensorboard --logdir logs/ray
```

If you'd like, I can extract the top-k trials into a CSV, save the best checkpoint into `models/`, or produce visualizations (heatmaps, parameter-effect plots) and add them to this folder. Which would you prefer next?

