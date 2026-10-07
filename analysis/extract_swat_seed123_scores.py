from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

import numpy as np
import torch
from torch.utils.data import DataLoader

import train_swat_static_temporal_graph as static_exp
import train_swat_dynamic_graph as dynamic_exp
import train_swat_random_static_temporal_graph as random_exp


SEED = 123
BATCH_SIZE = 256
GRAPH_K = 5

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CHECKPOINT_DIR = PROJECT_ROOT / "results" / "checkpoints"
OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "swat_seed123_score_analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
