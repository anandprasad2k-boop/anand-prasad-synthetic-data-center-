"""
feature_engineering.py
-----------------------
Turns the raw synthetic data center time series into a feature matrix for
tree-based forecasting models.

PHYSICS-INFORMED DESIGN
========================
Rather than relying on the trees to rediscover the electrical/thermal
coupling purely from raw lags, we hand-engineer features that encode the
known physics from the project brief:

  * Lagged & rolling-window stats of IT load and weather        -> captures
    autocorrelation and recent trend without the model needing huge trees.
  * An approximate PUE / coefficient-of-performance feature computed the
    SAME way the generator computes true PUE (economizer/chiller curve as a
    function of outdoor temp + humidity) -> gives the model direct access to
    the thermodynamic driver of cooling load instead of making it infer a
    nonlinear temperature->efficiency relationship from scratch.
  * Ramp-rate features (recent MW/min change) -> encodes grid/chiller
    response-limit information, useful for anticipating how fast load CAN
    move next.
  * Calendar features (hour, day-of-week, holiday flag) -> captures
    workload scheduling patterns (business hours, weekday/weekend effects).
  * A "thermal lag" feature: IT load shifted by the known/assumed thermal
    lag, i.e., what cooling *should* be reacting to right now.

All of this is computed with only backward-looking (causal) windows, so
nothing here leaks future information -- important for it to be usable in
a real forecasting setting.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

from config import SimConfig


# A small fixed set of US-style holidays (month, day) used only to derive a
# binary "is_holiday" calendar feature -- deliberately simple since the
# focus of this project is the physics coupling, not calendar exactness.
_HOLIDAYS_MD: List[Tuple[int, int]] = [
    (1, 1), (7, 4), (11, 11), (12, 25), (12, 31),
]


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.index
    df["hour_of_day"] = idx.hour + idx.minute / 60.0
    df["day_of_week"] = idx.dayofweek
    df["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    df["month"] = idx.month
    df["is_holiday"] = [(m, d) in _HOLIDAYS_MD for m, d in zip(idx.month, idx.day)]
    df["is_holiday"] = df["is_holiday"].astype(int)
    # Cyclical encodings so the model sees hour 23 and hour 0 as adjacent
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    return df


def add_lag_rolling_features(df: pd.DataFrame, cfg: SimConfig, cols: List[str]) -> pd.DataFrame:
    """Backward-looking lag and rolling-window (mean, std, rate-of-change)
    features for the given source columns (e.g. it_load_mw, outdoor_temp_c).
    """
    fc = cfg.forecast
    for col in cols:
        for lag_min in fc.lags_minutes:
            steps = max(1, int(round(lag_min / cfg.freq_minutes)))
            df[f"{col}_lag_{lag_min}m"] = df[col].shift(steps)

        for win_min in fc.rolling_windows_minutes:
            steps = max(1, int(round(win_min / cfg.freq_minutes)))
            roll = df[col].shift(1).rolling(steps, min_periods=max(1, steps // 3))
            df[f"{col}_rollmean_{win_min}m"] = roll.mean()
            df[f"{col}_rollstd_{win_min}m"] = roll.std()

        # Rate of change over the shortest lag window -> approximates the
        # instantaneous ramp rate at time t.
        shortest = max(1, int(round(fc.lags_minutes[0] / cfg.freq_minutes)))
        df[f"{col}_roc_{fc.lags_minutes[0]}m"] = df[col].diff(shortest) / (shortest * cfg.freq_minutes)
    return df


def add_pue_proxy_feature(df: pd.DataFrame, cfg: SimConfig) -> pd.DataFrame:
    """Feature-engineering-side PUE proxy: computed the same functional
    form as the generator's true PUE curve (economizer/chiller changeover
    with a humidity penalty), but using only information a forecaster would
    actually have available (current & recent outdoor conditions) -- this
    is the model's estimate of thermodynamic efficiency, not a leaked copy
    of the ground truth `pue` column (which is dropped before modeling).
    """
    t = cfg.thermal
    span = max(t.chiller_saturation_c - t.economizer_changeover_c, 1e-6)
    frac = np.clip((df["outdoor_temp_c"] - t.economizer_changeover_c) / span, 0.0, 1.0)
    pue_proxy = t.pue_min + frac * (t.pue_max - t.pue_min)
    pue_proxy += t.humidity_penalty_per_pct * np.clip(df["humidity_pct"] - 40, 0, None)
    df["pue_proxy"] = pue_proxy

    # A simple inverse "coefficient of performance" proxy (higher = more
    # efficient cooling), useful as an alternative nonlinear encoding.
    df["cop_proxy"] = 1.0 / np.clip(pue_proxy - 1.0, 1e-3, None)
    return df


def add_ramp_rate_features(df: pd.DataFrame, cfg: SimConfig) -> pd.DataFrame:
    """Encode the physical ramp-rate LIMITS (not just observed rate of
    change) as static reference features -- e.g. how much headroom is left
    before the IT load hits its ramp ceiling, given how fast it's currently
    moving. This helps the model understand that IT/cooling load literally
    cannot jump arbitrarily far in one step.
    """
    df["it_ramp_limit_mw_per_step"] = cfg.cluster.it_ramp_limit_mw_per_min * cfg.freq_minutes
    df["cooling_ramp_limit_mw_per_step"] = cfg.thermal.cooling_ramp_limit_mw_per_min * cfg.freq_minutes

    it_roc_col = f"it_load_mw_roc_{cfg.forecast.lags_minutes[0]}m"
    if it_roc_col in df.columns:
        df["it_ramp_headroom_frac"] = 1 - (
            df[it_roc_col].abs() / df["it_ramp_limit_mw_per_step"].replace(0, np.nan)
        ).clip(0, 1)
    return df


def add_thermal_lag_feature(df: pd.DataFrame, cfg: SimConfig) -> pd.DataFrame:
    """The IT load value shifted by the assumed thermal-lag delay -- i.e.
    "what the cooling system should currently be responding to" given the
    known chiller/thermal-mass response delay. This directly hands the
    model the physically-relevant predictor for cooling_load(t) instead of
    making it search for the right lag amongst dozens of generic lags.
    """
    lag_steps = max(0, int(round(cfg.thermal.thermal_lag_minutes / cfg.freq_minutes)))
    df["it_load_thermal_lagged"] = df["it_load_mw"].shift(lag_steps)
    return df


def add_coupling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-series features that explicitly encode the electrical/thermal
    coupling relationship: an implied PUE from recent observed data, and
    the recent cooling-to-IT ratio.
    """
    # Backward-looking implied PUE from realized recent load (avoids
    # dividing by ~0 IT load with a floor).
    it_safe = df["it_load_mw"].shift(1).clip(lower=1e-3)
    cooling_prev = df["cooling_load_mw"].shift(1)
    df["implied_pue_recent"] = 1 + (cooling_prev / it_safe)
    df["cooling_to_it_ratio_recent"] = cooling_prev / it_safe
    return df


def build_feature_matrix(df: pd.DataFrame, cfg: SimConfig) -> pd.DataFrame:
    """Full physics-informed feature engineering pipeline. Operates on the
    "measured" (sensor-realistic) columns where available, so the resulting
    model reflects what would actually be observable in production, and
    returns a feature-augmented DataFrame (targets included, still to be
    split off downstream in train.py).
    """
    out = df.copy()

    # Prefer the noisy/measured sensor columns as the "observed" series a
    # real deployment would see; fall back to the clean columns if the
    # measured variant isn't present. Forward-fill short sensor dropouts.
    for base in ["it_load_mw", "cooling_load_mw", "total_load_mw"]:
        meas = f"{base}_measured"
        if meas in out.columns:
            out[base] = out[meas].ffill().bfill()

    out = add_calendar_features(out)
    out = add_pue_proxy_feature(out, cfg)
    out = add_thermal_lag_feature(out, cfg)
    out = add_coupling_features(out)
    out = add_lag_rolling_features(out, cfg, cols=["it_load_mw", "cooling_load_mw", "outdoor_temp_c", "humidity_pct"])
    out = add_ramp_rate_features(out, cfg)

    return out


if __name__ == "__main__":
    from config import SimConfig
    from data_generation import generate_dataset

    cfg = SimConfig(periods_days=30)
    raw = generate_dataset(cfg)
    feats = build_feature_matrix(raw, cfg)
    print(feats.shape)
    print([c for c in feats.columns if "lag" in c or "roll" in c][:10])
