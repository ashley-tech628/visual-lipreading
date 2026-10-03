# Engineering validation

Recorded: 2026-10-03T03:16:15.583381+00:00

Real local LRW-style mouth-image subset: 617 training, 194 validation and 90 test segments across six classes. Original source clips are disjoint across partitions; speaker independence cannot be established from the supplied indices. Two training segments were excluded for missing or insufficient readable frames.

Each variant was trained for 12 epochs, seeds 17 and 29, 48 x 48 grayscale frames and up to 12 timesteps. Class-weighted cross entropy, AdamW and cosine decay were used. Validation balanced accuracy selected checkpoints and the serving bundle; validation-only temperature scaling calibrated logits. Test data only reports outcomes. These compact configurations are not a remeasurement of the thesis benchmark.

| Variant | Seed | Test accuracy | Balanced accuracy | Macro F1 | Parameters | Forward p50 ms |
|---|---:|---:|---:|---:|---:|---:|
| dense3d_mean | 17 | 12.22% | 20.12% | 0.087 | 2,902 | 1.45 |
| dense3d_mean | 29 | 15.56% | 16.30% | 0.102 | 2,902 | 0.98 |
| dense3d_bigru | 17 | 35.56% | 28.02% | 0.261 | 19,798 | 1.79 |
| dense3d_bigru | 29 | 33.33% | 34.15% | 0.265 | 19,798 | 1.90 |
| dense3d_bigru_attention | 17 | 37.78% | 33.44% | 0.299 | 20,999 | 2.01 |
| dense3d_bigru_attention | 29 | 30.00% | 35.93% | 0.251 | 20,999 | 2.06 |
| attention_augmentation | 17 | 35.56% | 38.54% | 0.278 | 20,999 | 1.95 |
| attention_augmentation | 29 | 34.44% | 33.47% | 0.285 | 20,999 | 2.07 |
| dense3d_lipnet_fusion | 17 | 33.33% | 25.54% | 0.237 | 33,967 | 5.75 |
| dense3d_lipnet_fusion | 29 | 37.78% | 32.78% | 0.295 | 33,967 | 5.13 |

Training-majority baseline test accuracy: 8.89%. The prediction is the most common TRAINING label, not a majority chosen using test labels.

Selected bundle: `dense3d_lipnet_fusion`, seed 29. Reload prediction differences were zero for all completed runs. Latency uses a batch of one on CPU and excludes video decoding, ROI selection and preprocessing.

## Perturbation probes

| Condition | Test accuracy | Macro F1 |
|---|---:|---:|
| clean | 37.78% | 0.295 |
| darken | 37.78% | 0.256 |
| brighten | 30.00% | 0.227 |
| frame_reverse | 34.44% | 0.267 |
| half_frames_zeroed | 40.00% | 0.318 |

The perturbation probes are post-selection analyses, not training augmentation claims. Reversing frames tests temporal sensitivity; zeroing half the sampled frames simulates missing observations. Darkening/brightening operate in the normalized image domain. They do not cover all domain shifts or establish deployment reliability.

## Uncertainty and calibration

- dense3d_mean, seed 17: source-clip bootstrap accuracy 95% interval [4.6%, 21.1%], validation-fitted temperature 8.680, test NLL 1.793, test ECE 0.047.
- dense3d_mean, seed 29: source-clip bootstrap accuracy 95% interval [7.8%, 23.7%], validation-fitted temperature 1.344, test NLL 1.788, test ECE 0.045.
- dense3d_bigru, seed 17: source-clip bootstrap accuracy 95% interval [24.2%, 47.1%], validation-fitted temperature 0.783, test NLL 1.637, test ECE 0.082.
- dense3d_bigru, seed 29: source-clip bootstrap accuracy 95% interval [23.0%, 44.7%], validation-fitted temperature 0.784, test NLL 1.732, test ECE 0.169.
- dense3d_bigru_attention, seed 17: source-clip bootstrap accuracy 95% interval [26.0%, 49.2%], validation-fitted temperature 0.682, test NLL 1.687, test ECE 0.116.
- dense3d_bigru_attention, seed 29: source-clip bootstrap accuracy 95% interval [20.1%, 40.0%], validation-fitted temperature 0.704, test NLL 1.721, test ECE 0.126.
- attention_augmentation, seed 17: source-clip bootstrap accuracy 95% interval [24.0%, 46.7%], validation-fitted temperature 0.776, test NLL 1.668, test ECE 0.079.
- attention_augmentation, seed 29: source-clip bootstrap accuracy 95% interval [24.4%, 44.6%], validation-fitted temperature 0.829, test NLL 1.629, test ECE 0.089.
- dense3d_lipnet_fusion, seed 17: source-clip bootstrap accuracy 95% interval [22.5%, 45.3%], validation-fitted temperature 0.812, test NLL 1.653, test ECE 0.150.
- dense3d_lipnet_fusion, seed 29: source-clip bootstrap accuracy 95% interval [26.8%, 49.7%], validation-fitted temperature 0.807, test NLL 1.620, test ECE 0.085.
