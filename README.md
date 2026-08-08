# AI Data Center Load Forecasting: Physics-Informed, Coupled Electrical/Cooling Models

Synthetic-data research prototype exploring whether treating electrical and
cooling load as **physically coupled** (rather than independent series)
improves forecasting for AI data center grid-congestion and
cooling-efficiency decisions.

## Research Question

> Can a model that treats electrical and thermal load as physically coupled
> forecast demand well enough to help reduce grid congestion and cooling
> inefficiency at AI data centers?

## Project Layout

```
config.py               SimConfig dataclass -- every tunable assumption lives here
data_generation.py       Synthetic IT load, weather, PUE, cooling, total load + anomalies
feature_engineering.py   Physics-informed features (lags, PUE proxy, ramp rates, thermal lag)
models.py                IndependentArchitecture and CoupledArchitecture (LightGBM/XGBoost)
train.py                 Time-based split + training orchestration
evaluate.py               Metrics, physical-consistency check, curtailment/waste analysis, SHAP, plots
run_example.py            End-to-end example: run this to reproduce everything
requirements.txt
artifacts/                Generated on run: models, data, metrics, plots/
```

## Quick Start

```bash
pip install -r requirements.txt
python run_example.py
```

This generates ~1 year of 15-minute synthetic data for a 120MW AI campus,
trains both architectures across 1hr/6hr/24hr horizons (with q10/q50/q90
quantile forecasts), evaluates them, and writes everything to `artifacts/`.
Takes roughly 3-4 minutes on a modern CPU.

To explore other scenarios, edit the `SimConfig(...)` call in
`run_example.py` (or import `config.SimConfig` yourself) -- cluster size,
climate (temperature/humidity profile), PUE min/max and economizer
changeover point, anomaly rates, and forecast horizons are all
parameterized.

## Physics Encoded in the Simulator

- `cooling_load(t) ≈ IT_load(t - thermal_lag) * (PUE(t) - 1)`
- `PUE(t)` follows an economizer/chiller curve: near `pue_min` below the
  economizer changeover temperature (free cooling), rising linearly to
  `pue_max` at the chiller-saturation temperature, plus a small humidity
  penalty.
- `total_load(t) = IT_load(t) + cooling_load(t) + overhead_losses(t)`
- Both IT load and cooling load are subject to **ramp-rate limits**
  (MW/min) representing real equipment response constraints.
- Anomalies (grid curtailment, cooling-system degradation, GPU cluster
  scale-ups, sensor noise/dropout) are injected *after* the base physics is
  built and dependent series are re-derived, so the three targets stay
  mutually consistent even during anomalous periods.

## Modeling Approach

Two tree-based (LightGBM, with XGBoost/sklearn fallback) architectures are
trained and compared, each with **direct multi-horizon** forecasting (a
separate model per horizon rather than recursive forecasting, to avoid
compounding error) and **quantile regression** (q10/q50/q90) for
prediction intervals:

- **Independent**: one model per (target, horizon) using only the shared
  physics-informed feature set (lags, rolling stats, PUE proxy, calendar,
  ramp-rate features, thermal-lag feature). Targets never see each other.
- **Coupled**: electrical (IT load) model trained first; its prediction is
  fed as an *input feature* to the cooling model; both predictions feed the
  total-load model. To avoid the cooling/total models training on
  artificially perfect in-sample electrical predictions (which would not
  reflect real deployment accuracy), the electrical prediction used as a
  training-time feature is generated via **time-series-safe out-of-fold
  (OOF) prediction** (expanding-window folds), not in-sample fitted values.

## Key Findings (this synthetic-data run)

See `artifacts/metrics.csv` for exact numbers; on the ~1 year test run captured
here:

- **Cooling load**: the Coupled model modestly *outperforms* the Independent
  model (e.g. ~3% lower MAE at 1hr horizon), supporting the physics-informed
  hypothesis for the target where the coupling matters most directly.
- **Electrical (IT) load**: identical by construction (Stage 1 of the
  Coupled architecture *is* the Independent electrical model).
- **Total load**: the Coupled model is slightly *worse* than the
  Independent model. This is a genuine and important finding, not a bug:
  chaining electrical -> cooling -> total predictions **cascades
  forecasting error** through three stages, and even with OOF-safe training
  features, the total-load model inherits noise from both upstream
  predictions rather than fitting directly on the (very strong)
  autocorrelation of total load itself. Independent, direct modeling of
  total load turns out to be more robust here. This nuances the "physics
  coupling always helps" intuition: it helps most for the specific
  physically-derived quantity (cooling), but naive error cascading can hurt
  a quantity (total load) that is already easy to forecast directly.
- **Physical consistency**: both architectures stay within a realistic
  implied-PUE envelope on 100% of test predictions (see
  `physical_consistency.csv`) -- neither model predicts physically
  impossible electrical/cooling combinations, though this check does not by
  itself distinguish which architecture is "more correct," only which is
  implausible.
- **Illustrative curtailment/waste analysis** (`curtailment_waste_analysis.csv`):
  at the 1hr horizon the model catches roughly half of synthetic
  near-nameplate "congestion risk" events before they occur; at 6/24hr
  horizons lead-time capture drops sharply, which is expected since a
  training-job spike is essentially a scheduling decision, not a
  slowly-evolving physical process -- there's no strong long-horizon
  precursor signal in this simulator. The q90 cooling forecast implies a
  large potential reduction vs. a naive "always provision to worst-case
  PUE" static margin. **This is an illustrative demonstration on synthetic
  data, not a real-world savings estimate.**

## Deliberate Design Choices / Limitations

- Direct (not recursive) multi-horizon forecasting to avoid compounding
  error, at the cost of training more models.
- OOF coupling features avoid the most obvious form of train/inference
  leakage in the Coupled architecture; residual train-serving skew (OOF
  models are fit on less data than the final full-data model used at
  inference) still exists and likely contributes to the total-load result
  above -- a natural next step would be nested/repeated CV or a
  meta-learner correction.
- The simulator is a simplified physical model (single-zone PUE curve, no
  explicit chiller plant model, no real weather data) -- appropriate for a
  research prototype demonstrating the *methodology*, not for operational
  deployment.
- `pue`, the noise-free sensor columns, and anomaly flags are excluded from
  the model feature set to prevent label leakage; the `pue_proxy` feature
  the model actually uses is computed from the same functional form but
  only from causal (current/lagged) weather observations.
