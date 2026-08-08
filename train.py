"""
train.py
--------
Orchestrates the end-to-end training workflow:
  1. Generate (or load) synthetic data.
  2. Physics-informed feature engineering.
  3. Time-based train/val/test split (NEVER shuffle time series data).
  4. Fit both the Independent and Coupled architectures.
  5. Persist fitted models + the engineered test set for evaluate.py.

Time-based split rationale: shuffling would let the model "see the future"
via leaked temporal correlation (e.g. rolling means computed from
neighboring rows). We hold out the trailing `test_fraction` of the series
as the test set, and carve a validation slice from the end of the
remaining training data for early stopping.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd

from config import SimConfig
from data_generation import generate_dataset, save_dataset
from feature_engineering import build_feature_matrix
from models import IndependentArchitecture, CoupledArchitecture, TARGET_COLS

ARTIFACT_DIR = Path("artifacts")


def time_based_split(df: pd.DataFrame, test_fraction: float, val_fraction_of_train: float = 0.15):
    n = len(df)
    test_start = int(n * (1 - test_fraction))
    train_val = df.iloc[:test_start]
    test = df.iloc[test_start:]

    val_start = int(len(train_val) * (1 - val_fraction_of_train))
    train = train_val.iloc[:val_start]
    val = train_val.iloc[val_start:]
    return train, val, test


def run_training(cfg: SimConfig = None, save_artifacts: bool = True):
    cfg = cfg or SimConfig()
    ARTIFACT_DIR.mkdir(exist_ok=True)

    print(f"[1/4] Generating synthetic dataset ({cfg.periods_days} days, {cfg.freq_minutes}-min resolution)...")
    raw_df = generate_dataset(cfg)
    save_dataset(raw_df, out_csv=str(ARTIFACT_DIR / "synthetic_data.csv"))

    print("[2/4] Physics-informed feature engineering...")
    feat_df = build_feature_matrix(raw_df, cfg)
    # Drop the warm-up rows where lag/rolling features are still NaN
    feat_df = feat_df.dropna(subset=[c for c in feat_df.columns if "lag" in c or "rollmean" in c])

    print("[3/4] Time-based train/val/test split...")
    train_df, val_df, test_df = time_based_split(feat_df, cfg.forecast.test_fraction)
    print(f"    train={len(train_df)} rows, val={len(val_df)} rows, test={len(test_df)} rows")

    print("[4/4] Training Independent and Coupled architectures...")
    independent = IndependentArchitecture(cfg=cfg).fit(train_df, val_df)
    coupled = CoupledArchitecture(cfg=cfg).fit(train_df, val_df)

    if save_artifacts:
        with open(ARTIFACT_DIR / "independent_model.pkl", "wb") as f:
            pickle.dump(independent, f)
        with open(ARTIFACT_DIR / "coupled_model.pkl", "wb") as f:
            pickle.dump(coupled, f)
        test_df.to_csv(ARTIFACT_DIR / "test_set.csv")
        with open(ARTIFACT_DIR / "config.pkl", "wb") as f:
            pickle.dump(cfg, f)
        print(f"Artifacts saved to {ARTIFACT_DIR.resolve()}")

    return independent, coupled, train_df, val_df, test_df


if __name__ == "__main__":
    # Smaller default run for a quick sanity check; the full example script
    # (run_example.py) uses a larger/more realistic configuration.
    cfg = SimConfig(periods_days=90)
    run_training(cfg)
