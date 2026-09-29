import numpy as np
from scipy.stats import wasserstein_distance
import os
from datetime import datetime

# ============================================================
# Evaluation metrics
# ============================================================

def tv_distance(samples: np.ndarray, true_probs: np.ndarray, S: int) -> float:
    counts = np.bincount(samples.flatten().clip(0, S - 1).astype(int), minlength=S)
    emp    = counts / counts.sum()
    tru    = true_probs[:S].flatten()
    tru    = tru / tru.sum()
    return float(0.5 * np.abs(emp - tru).sum())

def w1_distance(samples: np.ndarray, data: np.ndarray) -> float:
    return float(wasserstein_distance(samples.flatten().astype(float),
                                      data.flatten().astype(float)))

# ============================================================
# Utilities for saving
# ============================================================

def create_run_dir(base_dir):
    # Create base directory if it doesn't exist
    os.makedirs(base_dir, exist_ok=True)
    # Create a unique folder name using timestamp
    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

