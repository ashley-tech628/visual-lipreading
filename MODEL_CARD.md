# Model card

## Intended use

Research and educational demonstrations of closed-vocabulary visual word recognition. Inputs are short mouth-centered video segments. Outputs are word-class logits/probabilities. Audio is not used. The thesis is the primary research record; executable run records describe the actual maintained configuration.

## Models

- Thesis-scale dense 3D / LipNet feature fusion with two-layer bidirectional recurrence.
- Compact dense encoder with mean, recurrent or attention aggregation.
- Configurable dense-only, LipNet-only, 2D dense and fusion ablations.

## Data and evaluation

The local engineering subset contains six pinyin labels. Accepted train/validation/test counts are 617/194/90. Clip overlap is checked and absent. Speaker identity and demographic coverage are not supplied. Training and validation class distributions differ substantially, making raw validation accuracy unsuitable as the sole checkpoint criterion.

The thesis reports approximately 46.73% optimized-model validation recognition. This is not the compact bundle's test accuracy. Full benchmark performance is not inferred from a six-class engineering run.

## Limitations

Homophones and visually similar articulations can be ambiguous. Unsupported words are forced into known classes. A high softmax probability does not establish correctness or in-distribution input. Manual ROI, codec support and segment timing affect the video demo. The system is not validated for clinical use, identity verification, covert surveillance, or reliable accessibility assistance.

Temperature scaling uses the validation partition; class-prior shift can degrade calibration on test data. Temporal attention is an aggregation weight, not causal explanation. Source-clip bootstrap intervals do not remove selection bias or establish speaker independence.

## Artifacts and provenance

Serving bundles include model and preprocessing metadata and use weight-only loading. Accept bundles only from trusted sources. Raw video, images and database credentials are not distributed. The thesis public copy omits administrative personal identifiers. No third-party detector checkpoint is bundled.
