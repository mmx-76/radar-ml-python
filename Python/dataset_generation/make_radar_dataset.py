#!/usr/bin/env python3
"""
Generate synthetic radar-like 1D range profile data for ML training.

Classes:
    0 = no target        (noise only)
    1 = near target      (strong peak in early bins)
    2 = mid target       (strong peak in middle bins)
    3 = far target       (strong peak in late bins)

Each sample is an 8-bin vector representing amplitude vs range bin.
Outputs are saved under data/training_vectors/.
"""

import numpy as np
from pathlib import Path
from typing import Tuple


# -----------------------------
# Configuration
# -----------------------------

NUM_BINS = 8                     # length of each radar profile
N_SAMPLES_PER_CLASS = 500        # total samples per class (train+val+test)
RANDOM_SEED = 42                 # for reproducibility

# Train/val/test split ratios (must sum to 1.0)
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15


# -----------------------------
# Profile generation helpers
# -----------------------------

def generate_no_target(n_samples: int, num_bins: int) -> np.ndarray:
    """
    Class 0: no target - low amplitude noise across all bins.
    """
    # Small random noise around 0.05
    noise = np.random.normal(loc=0.05, scale=0.02, size=(n_samples, num_bins))
    profiles = np.clip(noise, 0.0, 0.3)
    return profiles


def generate_target_class(
    n_samples: int,
    num_bins: int,
    bin_range: Tuple[int, int],
    peak_amp_range: Tuple[float, float] = (0.7, 1.0),
    sigma: float = 0.7,
    noise_std: float = 0.03,
) -> np.ndarray:
    """
    Generate profiles with a Gaussian-like peak in a specified bin range.

    Args:
        n_samples: number of samples to generate
        num_bins: length of each profile
        bin_range: (min_bin, max_bin) inclusive range for the peak centre
        peak_amp_range: (min_amp, max_amp) range for random peak amplitude
        sigma: Gaussian width
        noise_std: standard deviation of additive noise
    """
    profiles = np.zeros((n_samples, num_bins), dtype=np.float32)

    min_bin, max_bin = bin_range
    assert 0 <= min_bin <= max_bin < num_bins, "Invalid bin_range"

    for i in range(n_samples):
        # Choose a random peak position within the specified range
        peak_bin = np.random.randint(min_bin, max_bin + 1)
        peak_amp = np.random.uniform(*peak_amp_range)

        x = np.arange(num_bins, dtype=np.float32)
        # Gaussian bump
        base = peak_amp * np.exp(-0.5 * ((x - peak_bin) / sigma) ** 2)

        # Additive Gaussian noise
        noise = np.random.normal(loc=0.0, scale=noise_std, size=num_bins)

        profile = base + noise
        # Ensure amplitudes are in [0, 1]
        profile = np.clip(profile, 0.0, 1.0)

        profiles[i, :] = profile

    return profiles


def make_dataset_for_all_classes(
    n_per_class: int,
    num_bins: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate X, y for all 4 classes.

    Returns:
        X: shape (N, num_bins)
        y: shape (N,)
    """
    # Class 0: no target
    X0 = generate_no_target(n_per_class, num_bins)
    y0 = np.zeros(n_per_class, dtype=np.int64)

    # Class 1: near target (bins 0–2)
    X1 = generate_target_class(n_per_class, num_bins, bin_range=(0, 2))
    y1 = np.ones(n_per_class, dtype=np.int64) * 1

    # Class 2: mid target (bins 3–4)
    X2 = generate_target_class(n_per_class, num_bins, bin_range=(3, 4))
    y2 = np.ones(n_per_class, dtype=np.int64) * 2

    # Class 3: far target (bins 5–7)
    X3 = generate_target_class(n_per_class, num_bins, bin_range=(5, 7))
    y3 = np.ones(n_per_class, dtype=np.int64) * 3

    # Stack everything
    X = np.vstack([X0, X1, X2, X3])
    y = np.concatenate([y0, y1, y2, y3])

    return X, y


# -----------------------------
# Splitting & saving
# -----------------------------

def split_dataset(
    X: np.ndarray, y: np.ndarray,
    train_ratio: float, val_ratio: float, test_ratio: float
):
    """
    Split into train / val / test sets.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

    n_samples = X.shape[0]
    indices = np.random.permutation(n_samples)

    n_train = int(train_ratio * n_samples)
    n_val = int(val_ratio * n_samples)
    # test = rest

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def save_numpy_and_csv(
    out_dir: Path,
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
):
    """
    Save arrays as .npy and .csv for inspection.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # NPY saves (good for Python loading)
    np.save(out_dir / "X_train.npy", X_train)
    np.save(out_dir / "y_train.npy", y_train)
    np.save(out_dir / "X_val.npy", X_val)
    np.save(out_dir / "y_val.npy", y_val)
    np.save(out_dir / "X_test.npy", X_test)
    np.save(out_dir / "y_test.npy", y_test)

    # CSV saves (nice for quick viewing / Mahara screenshots)
    np.savetxt(out_dir / "X_train.csv", X_train, delimiter=",")
    np.savetxt(out_dir / "y_train.csv", y_train, fmt="%d", delimiter=",")
    np.savetxt(out_dir / "X_val.csv", X_val, delimiter=",")
    np.savetxt(out_dir / "y_val.csv", y_val, fmt="%d", delimiter=",")
    np.savetxt(out_dir / "X_test.csv", X_test, delimiter=",")
    np.savetxt(out_dir / "y_test.csv", y_test, fmt="%d", delimiter=",")


def main():
    np.random.seed(RANDOM_SEED)

    print("Generating synthetic radar dataset...")
    X, y = make_dataset_for_all_classes(
        n_per_class=N_SAMPLES_PER_CLASS,
        num_bins=NUM_BINS,
    )

    print(f"Total samples: {X.shape[0]} | Num bins: {X.shape[1]}")
    unique, counts = np.unique(y, return_counts=True)
    print("Class distribution:", dict(zip(unique, counts)))

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_dataset(
        X, y,
        TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
    )

    out_dir = Path(__file__).resolve().parents[2] / "data" / "training_vectors"
    print(f"Saving dataset to: {out_dir}")
    save_numpy_and_csv(
        out_dir,
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
    )

    print("Done.")


if __name__ == "__main__":
    main()
