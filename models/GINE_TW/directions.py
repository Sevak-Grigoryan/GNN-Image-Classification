import os
import torch
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# The dataset is confidential and is NOT part of this repository (see DATA.md).
# Point GNN_DATA_ROOT at your local copy, or place it under <repo>/data/.
DATASET_ROOT = os.environ.get("GNN_DATA_ROOT", str(PROJECT_ROOT / "data"))
CARS_DATA   = os.path.join(DATASET_ROOT, "Car")
HUMANS_DATA = os.path.join(DATASET_ROOT, "Human")
NOISE_DATA  = os.path.join(DATASET_ROOT, "Noice")
CLASSES = ["Car", "Human", "Noice"]
NUM_CLASSES = len(CLASSES)
IMAGE_H = 80
IMAGE_W = 50
GRID_H = IMAGE_H
GRID_W = IMAGE_W
GRID_SIZE = GRID_H * GRID_W
BATCH_SIZE = 64#32 
LR = 1e-3
EPOCHS = 100
IN_CHANNELS = 3
HIDDEN_DIM = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"