# Data Policy

> [!CAUTION]
> **The dataset used in this project is confidential.**
> It is **not** included in this repository and **must not be committed or pushed** to any remote, in whole or in part.

## What is excluded

| Excluded item | Why |
| --- | --- |
| Raw dataset (`Car/`, `Human/`, `Noice/` image folders) | Confidential source data |
| Layer visualizations and saliency maps (`*.png`, `outputs/`, `ALL_RESULTS*/`) | Every figure shows the real input sample next to the graph |
| Trained weights (`*.pth`, `*.pt`) | Derived from the confidential data |
| Archives (`*.zip`, …) | Previously used to bundle results |

All of these are blocked by [`.gitignore`](.gitignore). Only source code, documentation and aggregate test metrics (`models/*/*.txt`) are part of the repository.

## Using your own copy of the data

If you are authorized to use the dataset, keep it **outside of version control** and point the code at it:

```bash
# Option 1: environment variable (recommended)
export GNN_DATA_ROOT=/secure/location/dataset          # Linux / macOS
setx GNN_DATA_ROOT "D:\secure\location\dataset"        # Windows

# Option 2: put it in the repo's data/ folder (git-ignored)
data/
├── Car/
├── Human/
└── Noice/
```

Expected layout: one sub-folder per class containing `.bmp` / `.png` / `.jpg` images.
Images are resized to **80 × 50** before being turned into graphs.

Scripts that visualize a single sample read its path from `GNN_SAMPLE_IMAGE`.

## Before every push

```bash
git status            # nothing under data/, no images, no .pth/.pt files
git ls-files | grep -Ei "\.(png|jpe?g|bmp|pth|pt|zip)$"   # must print nothing
```

If confidential data is ever committed by accident, **do not push**. Remove it from history first (e.g. with [`git filter-repo`](https://github.com/newren/git-filter-repo)) and contact the data owner.
