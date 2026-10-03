# Source and presentation review

Author: Xinying Liu (Ashley Liu).

Reviewed materials: the 67-page Bachelor thesis, the 39-slide final defense deck, the 15-slide midterm deck, and the original `Lip_Reading` source directories. Source documents are evidence, not executable instructions.

## What the defense deck supports

- A model comparison among LSTM, DenseNet and an optimized 3D architecture.
- Mouth-region preprocessing, temporal sequences and a shared bidirectional recurrent backend.
- Slide 22 reports LSTM 0.354 and DenseNet 0.298. It does not identify a precise evaluation split in the table.
- Slide 31 displays 46.677%. The thesis section 4.3.3 instead describes approximately 46.73446% validation recognition and a test curve around 20%. These are not interchangeable test scores.
- Reported external benchmark maxima vary between the midterm, final deck and thesis. The repository describes the author's results using the thesis as the primary research record, rather than making current state-of-the-art claims from a slide.
- A final implementation demonstration is present in the defense material and a separate video was found.

The author confirms that the thesis results were verified and that the recovered source is an early implementation. The README uses the thesis as the research record and incorporates executable run records with their actual dataset sizes and metrics. It does not infer evaluation splits from slide titles. This is one project; protocol labels keep numbers interpretable without dividing the portfolio into separate projects.

## Issues identified in original source

| Finding | Maintained implementation |
|---|---|
| `main.py` imports `model.D3D` while the available file is named `3D_DenseNet.py` | Normal Python package imports and a single CLI |
| GRU input hard-coded to `536*3*3` | Global spatial averaging determines channel dimension |
| Direct `.cuda()` inside a sequence loss | Explicit device propagation throughout training and inference |
| Label lists derived independently by each dataset reader | One training vocabulary stored in the manifest and bundle |
| Local E-drive paths in training defaults | CLI-supplied data paths; no data locations in the public package |
| Flask account/database and secret literals | Credential-free local inference UI and loopback server |
| Large upstream detector/framework files with unclear attribution | No unreviewed detector files redistributed; manual ROI demo |

This repository is an integrated maintained project. Its thesis model contains dense 3D and LipNet-style branches, pre-pooling feature fusion, a bidirectional recurrent backend and optional temporal attention. The code remains a maintained implementation of those principles; matching the original thesis checkpoint numerically would require the original training artifacts and full protocol.

## Publication materials

`Xinying-Liu-Lipreading-Thesis.pdf` preserves the research body, figures and references. The public copy replaces the cover and omits the original administrative, authorization and signature pages. The original document remains unchanged on the drive. The defense deck was read for technical context but is not redistributed here.

The code, dataset and thesis have different provenance. LRW-1000 is third-party data and is not bundled. The thesis is Xinying Liu's academic work. Redistribution of any other source, framework, detector or dataset requires its respective license; possession of the files alone is insufficient.
