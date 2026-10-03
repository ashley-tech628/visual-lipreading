# Experiment protocol and research results

## Primary research record

The Bachelor thesis is the primary record for the project's research results. The author confirms that its metrics were verified and that the recovered source was an early version. The repository's main results table follows the thesis, rather than attempting to infer its outcomes from the recovered prototype.

| Source | Quantity | Value | Interpretation |
|---|---|---:|---|
| Table 4 | LSTM comparison recognition | 0.354 | Initial model comparison; split not named in the table |
| Table 4 | DenseNet comparison recognition | 0.298 | Same table's comparison definition |
| §4.2.1 / Figure 18 | LSTM training recognition | 45.04075% | Training partition |
| §4.2.1 / Figure 18 | LSTM test recognition | 33.7778996% | Test partition |
| §4.2.1 / Figure 18 | LSTM validation recognition | 21.45462% | Validation partition |
| §4.3.3 / Figure 31 | Optimized-model validation recognition | 46.73446% | Validation recognition around which the curve fluctuates |
| §4.3.3 / Figure 31 | Optimized-model test trajectory | around 20% | Qualitative description, not a precise aggregate test score |

The README rounds these values for readability. The source definitions should be preserved when discussing the work in interviews. In particular, 46.73% is a validation result. Subtracting Table 4's DenseNet value from it would mix definitions; the repository does not publish that subtraction as a matched test-set improvement.

## Research design

The thesis compares model behavior on LRW-1000, investigates mouth representation and temporal processing, studies recurrent-node and data-loader settings, then explores 3D DenseNet and feature fusion with LipNet. The defense deck discusses an approximately 8:1:1 partition and a short-sequence training stage followed by longer sequences. The maintained CLI makes actual partitions and sequence limits auditable; it does not silently infer partition membership from that ratio.

Worker count affects loading throughput, while recurrent width affects the model. They should be studied independently. A worker-count sweep needs synchronized timing, fixed model settings and the same data. A recurrent-width comparison needs fixed data, training budget and repeated seeds.

## Executable experiment records

The repository includes a controlled CPU validation protocol using the available pre-cropped frame subset. It contains six training labels and accepted partitions of 617 / 194 / 90 segments. Its partition sizes differ from an 8:1:1 full benchmark and are reported as supplied. These run records support engineering claims: training executes, checkpoints reload, labels remain aligned, padding is masked and prediction works.

The registered compact comparison varies temporal aggregation and augmentation with two seeds. An additional compact fusion comparison exercises both dense and LipNet branches. Checkpoint and serving-model selection use validation balanced accuracy only. Confidence calibration fits validation logits only. Test outcomes are reported after these choices.

Source-clip IDs do not overlap across partitions. Multiple segments from one clip are grouped during accuracy bootstrap. The provided annotations do not include speaker IDs, so this is not evidence of speaker-independent evaluation. Source index SHA-256 hashes and rejected-row counts are kept in `results/validation.json`.

The working data remain local. Dataset arrays, raw frames and checkpoint binaries are excluded from the ordinary Git repository. `results` holds aggregate metrics and run metadata. Model artifacts can be attached separately to a release after reviewing their provenance and size.

## Controlled deeper experiments

| Question | Fixed factors | Variable | Evidence to report |
|---|---|---|---|
| Does fusion improve recognition? | Data, optimizer, budget, seeds | Dense-only / LipNet-only / fusion | Accuracy, macro F1, class recall, parameters |
| Does recurrence capture ordering? | Encoder and data | Mean / BiGRU / BiLSTM | Recognition, reversed-order sensitivity |
| Does attention help aggregation? | Encoder, recurrence, budget | Last state / attention | Metrics, run variance, temporal weight distribution |
| What is the spatial trade-off? | Data and frame count | 48 / 80 / 112 pixels | Recognition, latency, GPU memory |
| What is the temporal trade-off? | Data and spatial size | 12 / 30 / 60 frames | Recognition, runtime, truncated-duration statistics |
| Are probabilities meaningful? | Selected checkpoint | Temperature scaling | NLL, ECE, Brier, reliability plot |
| How robust is the model? | Selected checkpoint | Brightness, missing frames, ROI shifts | Metrics relative to clean input |

Only completed runs belong in a results table. Larger-resolution, audio-visual and sentence-level experiments require additional resources and evaluation and are not assigned fabricated scores.
