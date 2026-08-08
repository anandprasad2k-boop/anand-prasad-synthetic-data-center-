"""
data_generation.py
-------------------
Generates a synthetic, physically-plausible time series dataset for an AI
data center campus: IT (electrical/compute) load, outdoor weather, cooling
load, and total facility load.

PHYSICS-INFORMED DESIGN
========================
The core idea of this whole project is that cooling load is NOT an
independent series -- it is a (lagged, rate-limited, noisy) function of IT
load and outdoor weather, mediated by PUE:

    cooling_load(t) ~= IT_load(t - lag) * (PUE(t) - 1)
    total_load(t)    = IT_load(t) + cooling_load(t) + overhead_losses(t)

PUE itself is modeled as a function of outdoor temperature/humidity via a
simple economizer/chiller curve:
  - Below `economizer_changeover_c`: mostly free (air/water-side economizer)
    cooling -> PUE close to pue_min.
  - Above `chiller_saturation_c`: mechanical chillers run near their least
    efficient point -> PUE close to pue_max.
  - Between the two: linear interpolation, plus a small humidity penalty
    (higher humidity worsens evaporative/economizer effectiveness).

We generate IT load FIRST (driven by job-scheduling patterns), then weather
(independent driver), then derive cooling load from both, then total load
from IT + cooling + overhead. Anomalies are injected as a final pass so they
can affect any of the three series in physically sensible ways (e.g. a
curtailment event reduces IT load AND consequently cooling load).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import SimConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ramp_limit(series: np.ndarray, max_step_per_sample: float) -> np.ndarray:
    """Enforce a maximum per-sample change (ramp-rate limit) on a series by
    clipping the increment at each step. This mimics physical equipment
    (chillers, transformers, grid interconnects) that cannot instantaneously
    jump to a new setpoint.
    """
    out = series.copy()
    for i in range(1, len(out)):
        delta = out[i] - out[i - 1]
        if delta > max_step_per_sample:
            out[i] = out[i - 1] + max_step_per_sample
        elif delta < -max_step_per_sample:
            out[i] = out[i - 1] - max_step_per_sample
    return out


def _apply_lag(series: np.ndarray, lag_steps: int) -> np.ndarray:
    """Shift a series forward in time by `lag_steps` samples (i.e. the
    output at time t reflects the input at time t - lag_steps), padding the
    start with the first value. Represents thermal response delay.
    """
    if lag_steps <= 0:
        return series.copy()
    out = np.empty_like(series)
    out[:lag_steps] = series[0]
    out[lag_steps:] = series[:-lag_steps]
    return out


# ---------------------------------------------------------------------------
# Weather
# ---------------------------------------------------------------------------

def generate_weather(cfg: SimConfig, timestamps: pd.DatetimeIndex, rng: np.random.Generator) -> pd.DataFrame:
    """Seasonal + diurnal synthetic weather (outdoor dry-bulb temperature and
    relative humidity), plus an approximate wet-bulb temperature.
    """
    c = cfg.climate
    day_of_year = timestamps.dayofyear.values.astype(float)
    hour_of_day = timestamps.hour.values.astype(float) + timestamps.minute.values.astype(float) / 60.0

    # Seasonal component: cosine peaking at `peak_day_of_year`
    seasonal = c.annual_amplitude_c * np.cos(2 * np.pi * (day_of_year - c.peak_day_of_year) / 365.25)
    # Diurnal component: coolest ~05:00, warmest ~15:00
    diurnal = c.diurnal_amplitude_c * np.sin(2 * np.pi * (hour_of_day - 9) / 24)
    temp = c.annual_mean_temp_c + seasonal + diurnal + rng.normal(0, c.weather_noise_std_c, size=len(timestamps))

    # Humidity: anti-correlated with temperature swings + its own seasonal cycle, clipped to [5, 95]
    hum_seasonal = c.annual_amplitude_humidity_pct * np.cos(2 * np.pi * (day_of_year - (c.peak_day_of_year + 150)) / 365.25)
    humidity = c.annual_mean_humidity_pct + hum_seasonal + rng.normal(0, c.humidity_noise_std_pct, size=len(timestamps))
    humidity = np.clip(humidity, 5, 95)

    # Approximate wet-bulb temperature (Stull 2011 approximation)
    wet_bulb = (
        temp * np.arctan(0.151977 * np.sqrt(humidity + 8.313659))
        + np.arctan(temp + humidity)
        - np.arctan(humidity - 1.676331)
        + 0.00391838 * humidity ** 1.5 * np.arctan(0.023101 * humidity)
        - 4.686035
    )

    return pd.DataFrame(
        {
            "outdoor_temp_c": temp,
            "humidity_pct": humidity,
            "wet_bulb_c": wet_bulb,
        },
        index=timestamps,
    )


# ---------------------------------------------------------------------------
# IT (electrical/compute) load
# ---------------------------------------------------------------------------

def generate_it_load(cfg: SimConfig, timestamps: pd.DatetimeIndex, rng: np.random.Generator) -> np.ndarray:
    """GPU cluster electrical load driven by job scheduling: diurnal cycle,
    weekday/weekend effect, random big-job (training run) spikes, plus a
    ramp-rate limit so load transitions look like real equipment behavior
    rather than teleporting between values.
    """
    cl = cfg.cluster
    n = len(timestamps)
    hour = timestamps.hour.values + timestamps.minute.values / 60.0
    weekday = timestamps.dayofweek.values  # 0=Mon .. 6=Sun

    nameplate = cl.nameplate_it_mw
    baseline = cl.baseline_load_frac * nameplate
    daytime_ceiling = cl.typical_daytime_frac * nameplate

    # Diurnal shape: inference traffic + interactive workloads peak midday,
    # trough overnight. Weekend traffic is somewhat lower (fewer batch jobs
    # scheduled by human teams, though inference traffic is less affected).
    diurnal_shape = 0.5 * (1 + np.sin(2 * np.pi * (hour - 8) / 24 - np.pi / 2))
    weekend_mask = (weekday >= 5).astype(float)
    weekday_factor = 1.0 - 0.15 * weekend_mask

    base_load = baseline + (daytime_ceiling - baseline) * diurnal_shape * weekday_factor
    base_load += rng.normal(0, 0.02 * nameplate, size=n)  # small scheduling noise

    # Big training-job events: Poisson-arrival plateaus that push load
    # towards nameplate capacity for a period of hours.
    load = base_load.copy()
    steps_per_hour = 60 / cfg.freq_minutes
    n_weeks = cfg.periods_days / 7.0
    n_events = rng.poisson(cl.big_job_events_per_week * n_weeks)
    for _ in range(n_events):
        start_idx = rng.integers(0, max(1, n - 1))
        duration_hours = max(0.5, rng.normal(cl.big_job_duration_hours_mean, cl.big_job_duration_hours_std))
        duration_steps = int(duration_hours * steps_per_hour)
        end_idx = min(n, start_idx + duration_steps)
        target = rng.uniform(0.85, 1.0) * nameplate
        # Smooth plateau: blend up then down using a half-cosine window
        length = end_idx - start_idx
        if length <= 0:
            continue
        window = np.hanning(length) if length > 1 else np.array([1.0])
        load[start_idx:end_idx] = np.maximum(
            load[start_idx:end_idx], base_load[start_idx:end_idx] + window * (target - base_load[start_idx:end_idx])
        )

    load = np.clip(load, 0.05 * nameplate, nameplate)

    # Enforce physical ramp-rate limit (MW/min -> MW/sample)
    max_step = cl.it_ramp_limit_mw_per_min * cfg.freq_minutes
    load = _ramp_limit(load, max_step)

    return load


# ---------------------------------------------------------------------------
# PUE / cooling
# ---------------------------------------------------------------------------

def compute_pue(cfg: SimConfig, outdoor_temp_c: np.ndarray, humidity_pct: np.ndarray) -> np.ndarray:
    """Physics-informed PUE curve: free cooling below the economizer
    changeover temperature, linear rise to mechanical-chiller saturation,
    plus a small humidity penalty (reduces economizer/evaporative
    effectiveness at high humidity).
    """
    t = cfg.thermal
    span = max(t.chiller_saturation_c - t.economizer_changeover_c, 1e-6)
    frac = np.clip((outdoor_temp_c - t.economizer_changeover_c) / span, 0.0, 1.0)
    pue = t.pue_min + frac * (t.pue_max - t.pue_min)
    pue += t.humidity_penalty_per_pct * np.clip(humidity_pct - 40, 0, None)
    return np.clip(pue, t.pue_min, t.pue_max + 0.3)  # allow slight overshoot for degradation events later


def generate_cooling_load(cfg: SimConfig, it_load_mw: np.ndarray, pue: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Derive cooling load from IT load and PUE, with thermal lag, a ramp
    rate limit representing chiller/CRAH response time, and sensor-level
    noise. cooling_load ~= IT_load(t - lag) * (PUE(t) - 1).
    """
    t = cfg.thermal
    lag_steps = max(0, int(round(t.thermal_lag_minutes / cfg.freq_minutes)))
    lagged_it = _apply_lag(it_load_mw, lag_steps)

    raw_cooling = lagged_it * (pue - 1.0)
    raw_cooling = np.clip(raw_cooling, 0, None)

    max_step = t.cooling_ramp_limit_mw_per_min * cfg.freq_minutes
    cooling = _ramp_limit(raw_cooling, max_step)

    cooling += rng.normal(0, t.cooling_noise_std_mw, size=len(cooling))
    cooling = np.clip(cooling, 0, None)
    return cooling


def compute_total_load(cfg: SimConfig, it_load_mw: np.ndarray, cooling_load_mw: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Total facility load = IT + cooling + overhead losses (UPS,
    switchgear, lighting, misc), modeled as a fraction of (IT + cooling)
    plus small noise.
    """
    t = cfg.thermal
    subtotal = it_load_mw + cooling_load_mw
    overhead = t.overhead_loss_frac * subtotal + rng.normal(0, 0.05 * t.overhead_loss_frac * subtotal.mean(), size=len(subtotal))
    overhead = np.clip(overhead, 0, None)
    return subtotal + overhead


# ---------------------------------------------------------------------------
# Anomaly injection
# ---------------------------------------------------------------------------

def inject_anomalies(cfg: SimConfig, df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Inject realistic anomalies AFTER the base physics-consistent series
    is built, then re-derive dependent quantities so the anomalies remain
    physically coherent (e.g. a curtailment event that cuts IT load also
    reduces the cooling load that IT load would have driven).

    Anomaly types:
      1. Grid curtailment events -- utility/ISO forces IT load reduction.
      2. Cooling system degradation -- effective PUE penalty for a period
         (fouled heat exchanger, refrigerant leak, etc.).
      3. Sudden GPU cluster scale-up -- step change in baseline load.
      4. Sensor noise / dropout -- corrupts or nulls individual readings
         (does not affect the physical truth, only what's "measured").
    """
    a = cfg.anomaly
    n = len(df)
    steps_per_hour = 60 / cfg.freq_minutes
    steps_per_day = cfg.steps_per_day

    it = df["it_load_mw"].to_numpy().copy()
    pue = df["pue"].to_numpy().copy()

    df["anomaly_curtailment"] = 0
    df["anomaly_degradation"] = 0
    df["anomaly_scaleup"] = 0

    # --- 1. Grid curtailment events ---
    n_months = cfg.periods_days / 30.44
    n_curtailments = rng.poisson(a.curtailment_events_per_month * n_months)
    for _ in range(n_curtailments):
        start_idx = rng.integers(0, max(1, n - 1))
        duration_h = max(0.5, rng.normal(a.curtailment_duration_hours_mean, 1.0))
        duration_steps = int(duration_h * steps_per_hour)
        end_idx = min(n, start_idx + duration_steps)
        it[start_idx:end_idx] *= (1 - a.curtailment_severity_frac)
        df.iloc[start_idx:end_idx, df.columns.get_loc("anomaly_curtailment")] = 1

    # --- 2. Cooling system degradation (raises effective PUE for a window) ---
    n_quarters = cfg.periods_days / 91.3
    n_degradations = rng.poisson(a.degradation_events_per_quarter * n_quarters)
    for _ in range(n_degradations):
        start_idx = rng.integers(0, max(1, n - 1))
        duration_d = max(0.5, rng.normal(a.degradation_duration_days_mean, 2.0))
        duration_steps = int(duration_d * steps_per_day)
        end_idx = min(n, start_idx + duration_steps)
        pue[start_idx:end_idx] += a.degradation_pue_penalty
        df.iloc[start_idx:end_idx, df.columns.get_loc("anomaly_degradation")] = 1

    # --- 3. Sudden GPU cluster scale-up (permanent step change from that point on) ---
    n_scaleups = rng.poisson(a.scaleup_events_per_year * cfg.periods_days / 365.25)
    scaleup_points = sorted(rng.integers(0, max(1, n - 1), size=n_scaleups))
    cumulative_step = np.zeros(n)
    step_val = 0.0
    prev_idx = 0
    for idx in scaleup_points:
        step_val += a.scaleup_step_frac * cfg.cluster.nameplate_it_mw
        cumulative_step[idx:] = step_val
        df.iloc[idx:min(idx + steps_per_hour.__int__() * 6, n), df.columns.get_loc("anomaly_scaleup")] = 1
    it = it + cumulative_step
    it = np.clip(it, 0, cfg.cluster.nameplate_it_mw * 1.05)

    # Re-derive cooling & total from the anomaly-adjusted IT load and PUE so
    # the three series stay physically consistent with each other.
    cooling = generate_cooling_load(cfg, it, pue, rng)
    total = compute_total_load(cfg, it, cooling, rng)

    df["it_load_mw"] = it
    df["pue"] = pue
    df["cooling_load_mw"] = cooling
    df["total_load_mw"] = total

    # --- 4. Sensor noise / dropout (measurement-layer only) ---
    for col in ["it_load_mw", "cooling_load_mw", "total_load_mw"]:
        meas_col = f"{col}_measured"
        measured = df[col].to_numpy().copy()
        noise_mask = rng.random(n) < 0.02  # light measurement noise everywhere
        measured[noise_mask] += rng.normal(0, a.sensor_noise_std_mw, size=noise_mask.sum())
        dropout_mask = rng.random(n) < a.sensor_dropout_prob
        measured[dropout_mask] = np.nan
        df[meas_col] = measured

    return df


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------

def generate_dataset(cfg: SimConfig) -> pd.DataFrame:
    """Run the full synthetic-data pipeline end to end and return a clean,
    timestamp-indexed DataFrame with electrical, cooling, and total load
    (in MW), the weather drivers, PUE, and anomaly flags.
    """
    rng = np.random.default_rng(cfg.random_seed)

    timestamps = pd.date_range(
        start=cfg.start_date, periods=cfg.n_periods, freq=f"{cfg.freq_minutes}min"
    )

    weather = generate_weather(cfg, timestamps, rng)
    it_load = generate_it_load(cfg, timestamps, rng)
    pue = compute_pue(cfg, weather["outdoor_temp_c"].to_numpy(), weather["humidity_pct"].to_numpy())
    cooling_load = generate_cooling_load(cfg, it_load, pue, rng)
    total_load = compute_total_load(cfg, it_load, cooling_load, rng)

    df = weather.copy()
    df["it_load_mw"] = it_load
    df["pue"] = pue
    df["cooling_load_mw"] = cooling_load
    df["total_load_mw"] = total_load

    df = inject_anomalies(cfg, df, rng)

    df.index.name = "timestamp"
    return df


def save_dataset(df: pd.DataFrame, out_csv: str = None, out_parquet: str = None) -> None:
    if out_csv:
        df.to_csv(out_csv)
    if out_parquet:
        try:
            df.to_parquet(out_parquet)
        except ImportError:
            pass  # parquet engine (pyarrow/fastparquet) not installed; CSV still saved


if __name__ == "__main__":
    cfg = SimConfig(periods_days=30)  # quick smoke test
    df = generate_dataset(cfg)
    print(df.describe())
    print(df.head())
