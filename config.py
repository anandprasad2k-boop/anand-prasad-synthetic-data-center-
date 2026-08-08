"""
config.py
---------
Central configuration for the AI data center synthetic-data generator and
forecasting pipeline. Everything that a user might want to tweak (cluster
size, climate, PUE assumptions, anomaly rates, forecasting horizons, model
hyperparameters) lives here as a single dataclass so the rest of the code
never hard-codes assumptions.

Design note: we keep this as ONE dataclass (with nested dataclasses) rather
than scattering magic numbers through data_generation.py / models.py. That
makes it trivial to run sensitivity studies (e.g. "what if PUE is worse in
this climate?") by instantiating a new Config with different fields.
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class ClusterConfig:
    # Nameplate IT (compute) capacity of the campus, in MW. This is the
    # ceiling that GPU load can ramp towards during large training jobs.
    nameplate_it_mw: float = 120.0
    # Baseline idle/always-on IT load (management plane, storage, idle GPUs
    # still drawing standby power), as a fraction of nameplate.
    baseline_load_frac: float = 0.35
    # Typical fraction of nameplate reached during "normal" daytime
    # inference/training mix (before big batch jobs).
    typical_daytime_frac: float = 0.65
    # How often (expected events per week) a large full-cluster training
    # ramp-up event occurs, driving load towards nameplate.
    big_job_events_per_week: float = 2.0
    # How long a big job event lasts, in hours (mean, std)
    big_job_duration_hours_mean: float = 6.0
    big_job_duration_hours_std: float = 2.0
    # Max allowed ramp rate of IT load, in MW per minute. GPU clusters can
    # ramp fast, but we still cap it to keep the series physically sane.
    it_ramp_limit_mw_per_min: float = 2.5


@dataclass
class ClimateConfig:
    # Simple sinusoidal seasonal + diurnal weather model. Values roughly
    # correspond to a hot-summer / mild-winter climate (e.g. Phoenix-like)
    # by default, but every parameter is overridable for other climates.
    annual_mean_temp_c: float = 22.0
    annual_amplitude_c: float = 14.0          # summer/winter swing
    diurnal_amplitude_c: float = 8.0          # day/night swing
    annual_mean_humidity_pct: float = 35.0
    annual_amplitude_humidity_pct: float = 15.0
    weather_noise_std_c: float = 1.2
    humidity_noise_std_pct: float = 4.0
    # Day-of-year (0-365) at which annual temperature peaks (~ late July)
    peak_day_of_year: int = 205


@dataclass
class ThermalConfig:
    # PUE (Power Usage Effectiveness) model: PUE = pue_min at very low
    # outdoor temps (full free/economizer cooling) and rises towards
    # pue_max as outdoor temp increases past the economizer changeover
    # point (mechanical chillers take over, less efficient).
    pue_min: float = 1.08
    pue_max: float = 1.55
    economizer_changeover_c: float = 15.0     # below this: mostly free cooling
    chiller_saturation_c: float = 35.0        # above this: PUE ~ pue_max
    humidity_penalty_per_pct: float = 0.003   # wet-bulb / humidity effect on PUE
    # Thermal lag: cooling system doesn't respond instantly to IT load
    # changes (thermal mass of the room, chiller plant response time).
    thermal_lag_minutes: float = 12.0
    # Max ramp rate of the cooling plant, in MW per minute (chillers/CRAHs
    # cannot instantly match a GPU load spike).
    cooling_ramp_limit_mw_per_min: float = 1.5
    cooling_noise_std_mw: float = 0.8
    # Fixed overhead losses (UPS, switchgear, lighting, misc) as a fraction
    # of (IT + cooling) load, representing non-IT non-cooling facility draw.
    overhead_loss_frac: float = 0.03


@dataclass
class AnomalyConfig:
    # Grid curtailment: utility/ISO forces a load reduction event.
    curtailment_events_per_month: float = 1.0
    curtailment_duration_hours_mean: float = 3.0
    curtailment_severity_frac: float = 0.4     # fraction of IT load shed

    # Cooling system degradation: gradual efficiency loss (e.g. fouled
    # heat exchanger) that raises effective PUE for a period.
    degradation_events_per_quarter: float = 1.0
    degradation_duration_days_mean: float = 5.0
    degradation_pue_penalty: float = 0.15

    # Sudden GPU cluster scale-up (new hardware brought online), a
    # step-change increase in nameplate-relative baseline load.
    scaleup_events_per_year: float = 2.0
    scaleup_step_frac: float = 0.08            # fractional increase in baseline

    # Sensor noise / dropout (missing / corrupted readings)
    sensor_dropout_prob: float = 0.001          # per-timestep probability
    sensor_noise_std_mw: float = 0.3


@dataclass
class ForecastConfig:
    # Multi-step-ahead forecast horizons, expressed in NUMBER OF STEPS at
    # the chosen data resolution (see SimConfig.freq_minutes). E.g. with
    # 15-min data, horizons [4, 24, 96] = 1hr, 6hr, 24hr ahead.
    horizon_steps: List[int] = field(default_factory=lambda: [4, 24, 96])
    horizon_labels: List[str] = field(default_factory=lambda: ["1hr", "6hr", "24hr"])
    quantiles: List[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])
    test_fraction: float = 0.2      # trailing fraction of the series held out
    lags_minutes: List[int] = field(default_factory=lambda: [15, 60, 360, 1440])
    rolling_windows_minutes: List[int] = field(default_factory=lambda: [60, 360, 1440])


@dataclass
class SimConfig:
    start_date: str = "2023-01-01"
    periods_days: int = 365          # length of synthetic series, in days
    freq_minutes: int = 15           # sampling resolution
    random_seed: int = 42

    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    climate: ClimateConfig = field(default_factory=ClimateConfig)
    thermal: ThermalConfig = field(default_factory=ThermalConfig)
    anomaly: AnomalyConfig = field(default_factory=AnomalyConfig)
    forecast: ForecastConfig = field(default_factory=ForecastConfig)

    @property
    def steps_per_day(self) -> int:
        return int(24 * 60 / self.freq_minutes)

    @property
    def n_periods(self) -> int:
        return int(self.periods_days * self.steps_per_day)
