# GNN Image Classification Benchmark

Benchmarking **Graph Neural Network** architectures on a 3-class image classification task — **Car / Human / Noise** — by converting every image into a pixel-grid graph.

> [!IMPORTANT]
> **The dataset is confidential and is not included in this repository.**
> Raw data, rendered sample visualizations and trained weights must never be pushed. See [DATA.md](DATA.md).

---

## Overview

Each input image is resized to **80 × 50** and turned into a graph:

- **Nodes** – one per pixel (4,000 nodes), node features = RGB values
- **Edges** – grid neighbourhood connectivity (4- or 8-neighbourhood, undirected)
- **Edge features** *(GINE / GINE-TW only)* – color difference and relative spatial offset between neighbouring pixels

A GNN encodes the graph, a global pooling layer (mean / max) produces a graph embedding, and an MLP head predicts the class.

```
image (80×50×3) ──► pixel-grid graph ──► GNN layers ──► global pooling ──► MLP ──► {Car, Human, Noise}
```

## Results

Held-out test set: **910 samples** (stratified 70 / 15 / 15 split, `random_state=42`).

| Model | Folder | Test accuracy | Macro F1 |
| --- | --- | :---: | :---: |
| **GINE-TW** (trainable edge weighting) | [`models/GINE_TW`](models/GINE_TW) | **0.95** | **0.96** |
| GIN | [`models/GIN`](models/GIN) | 0.94 | 0.94 |
| GINE | [`models/GINE`](models/GINE) | 0.93 | 0.94 |
| GAT | [`models/GAT`](models/GAT) | 0.92 | 0.93 |
| GraphSAGE | [`models/GraphSAGE`](models/GraphSAGE) | 0.92 | 0.93 |
| ChebNet | [`models/ChebNet`](models/ChebNet) | 0.91 | 0.92 |
| Hybrid CNN + GNN | [`models/Hybrid_CNN_GNN`](models/Hybrid_CNN_GNN) | 0.91 | 0.92 |
| GCN | [`models/GCN`](models/GCN) | 0.90 | 0.91 |
| ChebNetII | [`models/ChebNet_II`](models/ChebNet_II) | 0.85 | 0.86 |

Full classification reports and confusion matrices are in the `*.txt` file inside each model folder. Where a folder has several runs, the best one is shown.

## Repository structure

```
.
├── models/                     # one self-contained folder per architecture
│   ├── ChebNet/
│   │   ├── model.py            # architecture
│   │   ├── train.py            # training + evaluation, writes <name>.pth / <name>.txt
│   │   ├── data_processing.py  # image → graph dataset (if model-specific)
│   │   ├── saliency.py         # grad × activation saliency per layer
│   │   ├── visualize_layers.py # node-activation graphs per layer
│   │   └── *.txt               # test-set classification report
│   ├── ChebNet_II/
│   ├── GAT/
│   ├── GCN/
│   ├── GIN/
│   ├── GINE/
│   ├── GINE_TW/
│   ├── GraphSAGE/
│   └── Hybrid_CNN_GNN/
├── utils/
│   ├── directions.py           # shared config: dataset path, image size, hyper-parameters
│   ├── data_processing.py      # shared Image2Graph dataset + grid-graph builder
│   ├── train.py                # generic GNN trainer
│   ├── count_parameters.py     # model parameter summary
│   └── image_graph_visualization.py
├── scripts/
│   └── image_to_graph_demo.py  # visualize how an arbitrary image becomes a graph
├── data/                       # (git-ignored) place the confidential dataset here
├── checkpoints/                # (git-ignored) trained weights
├── DATA.md                     # data confidentiality policy
├── requirements.txt
└── README.md
```

## Getting started

### 1. Install

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Install a PyTorch build that matches your CUDA version first if you want GPU training — see [pytorch.org](https://pytorch.org/get-started/locally/) and the [PyG install guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

### 2. Point to the data (authorized users only)

```bash
export GNN_DATA_ROOT=/secure/path/to/dataset   # folder containing Car/, Human/, Noice/
```

If the variable is not set, the code looks in `./data/`.

### 3. Train a model

Run from the repository root:

```bash
python models/GCN/train.py
```

The best checkpoint (`GCN.pth`) and the test report (`GCN.txt`) are written into the model's folder.

### 4. Inspect what the model learned

```bash
export GNN_SAMPLE_IMAGE=/secure/path/to/sample.bmp
python models/GCN/saliency.py        # figures saved to models/GCN/outputs/
```

> [!WARNING]
> Saliency and layer figures contain the real input sample. They are written to `outputs/`, which is git-ignored — keep it that way.

### Print a model's parameter count

```bash
python models/GIN/model_GIN.py
```

## Configuration

Shared settings live in [`utils/directions.py`](utils/directions.py):

| Setting | Default |
| --- | --- |
| `IMAGE_H × IMAGE_W` | 80 × 50 |
| `CLASSES` | `Car`, `Human`, `Noice` |
| `BATCH_SIZE` | 32 |
| `LR` | 1e-3 |
| `EPOCHS` | 50 |
| `HIDDEN_DIM` | 64 |

`models/GINE` and `models/GINE_TW` keep their own `directions.py` with model-specific values (e.g. GINE-TW uses batch 64 / 100 epochs).

## Data confidentiality

The data behind this project is **confidential and cannot be published**. This repository contains only code, documentation and aggregate metrics. See [DATA.md](DATA.md) for details and a pre-push checklist.
