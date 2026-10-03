# Architecture and implementation details

## Task and scope

The implemented task is closed-vocabulary **visual word classification**. A sequence of mouth images maps to one word label. Audio is not an input. This is not a sentence transcription or CTC system: each annotated segment already has a single target, so the objective is multiclass cross entropy.

The project is grounded in Xinying Liu's Bachelor thesis and its model-comparison work. The repository provides a single maintained implementation with a configurable dense 3D encoder and temporal aggregation. The paper retains its original title; its discussion of multimodal recognition does not imply that this executable implementation contains an audio stream.

## Input contract

- LRW-style CSV rows: `clip_id,word,pinyin,start_seconds,end_seconds`.
- Original timing convention: `first_frame = floor(start * 25) + 1`, with exclusive end `floor(end * 25) + 1`.
- A single sorted training vocabulary maps labels to integers for all partitions.
- The reader rejects malformed times, path traversal, duplicate segments and source clips shared between partitions.
- Clip directories can be resolved from multiple image roots. Segments with less than 80% readable requested frames are excluded and counted by reason.
- Up to 12 uniformly selected frames are converted to grayscale, spatially resized/cropped to 48 x 48, and scaled to [-1,1]. Remaining temporal positions are zero padding.
- The compact CPU protocol uses small images; spatial resolution and temporal length are configurable for subsequent GPU training. Their impact requires an explicit experiment rather than an assumption.

## Thesis-scale fusion architecture

`ThesisRecognizer` implements a dense branch, a LipNet-style branch and feature-level fusion. The preset uses 64 initial dense channels, growth 32, dense-block depths `(4,8,12,8)`, LipNet channels `(64,128,256)`, a 256-channel fusion projection and two recurrent layers with 64 hidden units per direction. These are an explicit maintained configuration; the original training checkpoint is not redistributed or assumed compatible.

The branches retain temporal resolution. LipNet feature maps are spatially aligned to the dense maps, concatenated before global spatial pooling, and projected with a 1 x 1 x 1 convolution. The resulting per-frame sequence enters BiGRU or BiLSTM. Attention or final-state pooling produces word logits. `dense2d`, `dense3d`, `lipnet` and `fusion` expose the architectural ablations. The 2D variant uses temporal kernel size one; it does not mix neighbouring frames until the recurrent backend.

The full model preset is intended for the complete vocabulary and adequate GPU memory. The same architecture has a compact configuration used for real local CPU training, save/load and serving validation. A compact run does not establish the thousand-class preset's recognition performance.

## Compact dense 3D encoder

For a tensor `B x 1 x T x 48 x 48`:

1. A `3 x 5 x 5` convolution, 8 output channels, spatial stride 2 and preserved temporal resolution.
2. Spatial-only average pooling by 2.
3. Two dense layers, growth rate 4, each concatenating a new 3D feature map with existing channels.
4. Spatial-only average pooling by 2, followed by global spatial averaging.

The output has `B x T x 16` features. Temporal padding is masked after every encoder stage, including dense concatenation. A frame-wise GroupNorm implementation normalizes each frame separately rather than allowing padded timesteps to alter valid features. Tests check invariance both to arbitrary values in padded positions and to extending the padded tensor.

## Temporal aggregation

Three selectable heads share the same dense encoder:

- `mean`: masked temporal mean pooling.
- `gru`: a two-layer bidirectional GRU, 24 hidden units per direction; concatenate the final forward and backward hidden states.
- `attention`: the same BiGRU followed by learned temporal aggregation:

```text
e_t = v^T tanh(W h_t)
alpha_t = softmax(e_t), with padding scores masked to negative infinity
h = sum_t alpha_t h_t
logits = Linear(Dropout(h))
```

`pack_padded_sequence` excludes padded timesteps from recurrent computation; `pad_packed_sequence` restores batch alignment. Attention weights describe the pooling operation, not a causal explanation or a face localization map. Implementation follows the [PyTorch packed-sequence contract](https://docs.pytorch.org/docs/2.14/generated/torch.nn.utils.rnn.pack_padded_sequence.html).

## Optimization and calibration

AdamW, cosine learning-rate decay, global gradient clipping at 5, deterministic seeds, and inverse training-frequency class weights. The class imbalance makes raw validation accuracy a misleading checkpoint criterion: validation balanced accuracy selects the checkpoint. Test labels never fit model weights, calibration temperature or model selection.

Temperature scaling minimizes validation negative log likelihood with a scalar constrained to [0.2,10]. It changes probabilities but preserves class ranking. The test report includes accuracy, balanced accuracy, macro F1, top-3 accuracy, negative log likelihood, Brier score and 10-bin expected calibration error. Calibration may transfer poorly when validation and test class priors differ; both calibrated and uncalibrated values are retained.

Accuracy intervals resample source clips, retaining correlated segments from each sampled clip. This is more appropriate than treating every segment from a video as independent. Speaker identifiers are not present in the supplied indices, so speaker independence has not been established.

## Serving bundle

One checkpoint contains model configuration, strict state dict, label order, preprocessing configuration, temperature and experiment metadata. Inference constructs the model from the stored configuration and uses `torch.load(..., weights_only=True)`; it does not load arbitrary pickled model objects. Only load trusted bundles, as [PyTorch's serialization documentation](https://docs.pytorch.org/docs/2.14/notes/serialization.html) explains the remaining limitations.

## Video demo

The browser decodes the selected local video and samples frames from a user-selected segment. It sends base64 PNG frames and a normalized mouth ROI to a loopback-only Python server. The server uses the same preprocessing and serving bundle as CLI inference. No account or database credentials are required.

The demo validates ROI geometry, image dimensions, frame count and request size; it returns top-3 scores, a fixed low-confidence flag and temporal weights. The threshold is a UI heuristic, not a validated rejection policy. Unsupported words can still receive high softmax scores. Mouth localization is manual; the server does not claim an automatic detector or production-scale deployment.
