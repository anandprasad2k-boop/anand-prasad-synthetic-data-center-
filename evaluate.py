"""
evaluate.py
-----------
Evaluation and analysis tied back to the project's research question: does
treating electrical & cooling load as physically coupled forecast better,
and in a way that's usable for grid-congestion / cooling-efficiency
decisions?

Provides:
  1. Standard metrics (MAE, RMSE, MAPE) per target x horizon x architecture.
  2. A physical-consistency check: does the predicted cooling load stay
     within a realistic PUE-implied range given the predicted IT load, or
     does the model predict physically impossible combinations?
  3. An illustrative curtailment / overprovisioning analysis: how forecast
     accuracy converts into potential lead time before a congestion event,
     and how tight prediction intervals could reduce cooling
     overprovisioning margins.
  4. Feature importance / SHAP analysis highlighting the physical drivers.
  5. Plots: true vs predicted, residuals, feature importance, and
     electrical-vs-cooling coupling.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import SimConfig
from models import IndependentArchitecture, CoupledArchitecture, TARGET_COLS, make_horizon_target

warnings.filterwarnings("ignore")

PLOT_DIR = Path("artifacts/plots")


# ---------------------------------------------------------------------------
# 1. Standard metrics
# ---------------------------------------------------------------------------

def _mape(y_true, y_pred, eps=1e-3):
    denom = np.clip(np.abs(y_true), eps, None)
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100


def compute_metrics(test_df: pd.DataFrame, preds: pd.DataFrame, cfg: SimConfig, arch_name: str) -> pd.DataFrame:
    rows = []
    for target in TARGET_COLS:
        for h_label in cfg.forecast.horizon_labels:
            h_steps = cfg.forecast.horizon_steps[cfg.forecast.horizon_labels.index(h_label)]
            y_true = make_horizon_target(test_df, target, h_steps)
            y_pred = preds[f"pred_{target}_{h_label}"]
            mask = y_true.notna() & y_pred.notna()
            yt, yp = y_true[mask].to_numpy(), y_pred[mask].to_numpy()
            if len(yt) == 0:
                continue
            mae = np.mean(np.abs(yt - yp))
            rmse = np.sqrt(np.mean((yt - yp) ** 2))
            mape = _mape(yt, yp)
            rows.append({
                "architecture": arch_name, "target": target, "horizon": h_label,
                "MAE_MW": mae, "RMSE_MW": rmse, "MAPE_pct": mape, "n": len(yt),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Physical consistency check
# ---------------------------------------------------------------------------

def physical_consistency_check(test_df: pd.DataFrame, preds: pd.DataFrame, cfg: SimConfig, h_label: str) -> Dict:
    """Checks whether predicted_cooling / predicted_IT implies a PUE within
    a realistic envelope [pue_min - slack, pue_max + slack]. Predictions
    that imply an impossible PUE (e.g. negative, or far above the
    mechanical-chiller ceiling) indicate the model has learned a physically
    inconsistent electrical/cooling relationship.
    """
    t = cfg.thermal
    slack = 0.15
    lo, hi = t.pue_min - slack, t.pue_max + slack

    it_pred = preds[f"pred_it_load_mw_{h_label}"].clip(lower=1e-3)
    cooling_pred = preds[f"pred_cooling_load_mw_{h_label}"]
    implied_pue = 1 + cooling_pred / it_pred

    violations = (implied_pue < lo) | (implied_pue > hi)
    return {
        "h_label": h_label,
        "pct_violating": 100 * violations.mean(),
        "implied_pue_mean": implied_pue.mean(),
        "implied_pue_std": implied_pue.std(),
        "valid_range": (lo, hi),
    }


# ---------------------------------------------------------------------------
# 3. Illustrative curtailment / overprovisioning analysis
# ---------------------------------------------------------------------------

def curtailment_and_waste_analysis(test_df: pd.DataFrame, preds: pd.DataFrame, cfg: SimConfig, h_label: str) -> Dict:
    """Illustrative, synthetic-data-only analysis connecting forecast
    quality to two operational outcomes named in the research question:

    (a) Grid congestion / curtailment risk: treat a near-nameplate IT load
        surge as a "would trigger curtailment risk" event. Compare how
        often the model's forecast crosses a warning threshold BEFORE the
        true load does (lead time) vs. how often it's caught by surprise.

    (b) Cooling overprovisioning: using the model's own upper-quantile
        (q0.9) cooling forecast as the safety margin operators would
        provision to, compare that margin against a naive "always
        provision to nameplate-implied max PUE" baseline to illustrate
        potential energy/capacity savings from tighter, uncertainty-aware
        forecasts.

    This is explicitly a simplified, illustrative demonstration on
    synthetic data -- not a claim about real-world savings.
    """
    h_steps = cfg.forecast.horizon_steps[cfg.forecast.horizon_labels.index(h_label)]
    nameplate = cfg.cluster.nameplate_it_mw
    warn_threshold = 0.9 * nameplate

    y_true_it = make_horizon_target(test_df, "it_load_mw", h_steps)
    y_pred_it = preds[f"pred_it_load_mw_{h_label}"]
    mask = y_true_it.notna() & y_pred_it.notna()
    yt, yp = y_true_it[mask], y_pred_it[mask]

    true_events = yt >= warn_threshold
    predicted_flag = yp >= warn_threshold
    caught = (true_events & predicted_flag).sum()
    missed = (true_events & ~predicted_flag).sum()
    total_events = true_events.sum()
    lead_time_capture_rate = 100 * caught / total_events if total_events > 0 else np.nan

    # Cooling overprovisioning comparison
    cooling_q90 = preds.loc[mask.index[mask], f"pred_cooling_load_mw_{h_label}_q0.9"] if f"pred_cooling_load_mw_{h_label}_q0.9" in preds.columns else None
    naive_provision = nameplate * (cfg.thermal.pue_max - 1.0)  # worst-case static provisioning
    if cooling_q90 is not None:
        model_provision_mean = cooling_q90.mean()
        provisioning_saving_pct = 100 * (1 - model_provision_mean / naive_provision)
    else:
        model_provision_mean, provisioning_saving_pct = np.nan, np.nan

    return {
        "h_label": h_label,
        "n_true_congestion_events": int(total_events),
        "pct_events_forecast_caught": lead_time_capture_rate,
        "n_missed_events": int(missed),
        "naive_static_cooling_provision_mw": naive_provision,
        "model_q90_cooling_provision_mw": model_provision_mean,
        "illustrative_provisioning_saving_pct": provisioning_saving_pct,
    }


# ---------------------------------------------------------------------------
# 4. Feature importance / SHAP
# ---------------------------------------------------------------------------

def _get_model_and_inputs(arch, target: str, h_label: str, X_sample: pd.DataFrame):
    """Uniform accessor across IndependentArchitecture and
    CoupledArchitecture that returns (model, X) with X containing exactly
    the columns the model was trained on (including any coupling-prediction
    columns for the Coupled architecture, which don't exist in the raw
    dataframe and must be reconstructed by chaining through upstream
    models).
    """
    if hasattr(arch, "get_model_and_inputs"):  # CoupledArchitecture
        return arch.get_model_and_inputs(X_sample, target, h_label)
    key = f"{target}|h={h_label}"
    return arch.models[key], X_sample[arch.feature_cols]


def feature_importance_report(arch, target: str, h_label: str, X_sample: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    model, X = _get_model_and_inputs(arch, target, h_label, X_sample)
    if hasattr(model, "feature_importances_"):
        df = pd.DataFrame({"feature": X.columns, "importance": model.feature_importances_})
        return df.sort_values("importance", ascending=False).head(top_n)
    return pd.DataFrame()


def shap_summary(arch, target: str, h_label: str, X_sample: pd.DataFrame, max_display: int = 15):
    try:
        import shap
    except ImportError:
        return None, None
    model, X = _get_model_and_inputs(arch, target, h_label, X_sample)
    X = X.dropna()
    if len(X) == 0:
        return None, None
    X = X.sample(min(300, len(X)), random_state=0)
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
    except Exception:
        return None, None
    return shap_values, X


# ---------------------------------------------------------------------------
# 5. Plots
# ---------------------------------------------------------------------------

def plot_true_vs_pred(test_df, preds, cfg, h_label, arch_name, out_dir: Path):
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    h_steps = cfg.forecast.horizon_steps[cfg.forecast.horizon_labels.index(h_label)]
    for ax, target in zip(axes, TARGET_COLS):
        y_true = make_horizon_target(test_df, target, h_steps)
        y_pred = preds[f"pred_{target}_{h_label}"]
        n_show = min(500, len(y_true))
        ax.plot(y_true.index[:n_show], y_true.iloc[:n_show], label="True", color="black", linewidth=1)
        ax.plot(y_pred.index[:n_show], y_pred.iloc[:n_show], label="Predicted", color="tab:red", alpha=0.8, linewidth=1)
        if f"pred_{target}_{h_label}_q0.1" in preds.columns:
            lo = preds[f"pred_{target}_{h_label}_q0.1"].iloc[:n_show]
            hi = preds[f"pred_{target}_{h_label}_q0.9"].iloc[:n_show]
            ax.fill_between(y_pred.index[:n_show], lo, hi, color="tab:red", alpha=0.15, label="q10-q90 interval")
        ax.set_ylabel(f"{target}\n(MW)")
        ax.legend(loc="upper right", fontsize=8)
    axes[0].set_title(f"{arch_name}: True vs Predicted ({h_label} ahead)")
    axes[-1].set_xlabel("Time")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"true_vs_pred_{arch_name}_{h_label}.png", dpi=120)
    plt.close(fig)


def plot_residuals(test_df, preds, cfg, h_label, arch_name, out_dir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    h_steps = cfg.forecast.horizon_steps[cfg.forecast.horizon_labels.index(h_label)]
    for ax, target in zip(axes, TARGET_COLS):
        y_true = make_horizon_target(test_df, target, h_steps)
        y_pred = preds[f"pred_{target}_{h_label}"]
        mask = y_true.notna() & y_pred.notna()
        resid = (y_true[mask] - y_pred[mask])
        ax.hist(resid, bins=40, color="tab:blue", alpha=0.7)
        ax.axvline(0, color="black", linewidth=1)
        ax.set_title(f"{target} residuals")
        ax.set_xlabel("True - Predicted (MW)")
    fig.suptitle(f"{arch_name} residual distributions ({h_label} ahead)")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"residuals_{arch_name}_{h_label}.png", dpi=120)
    plt.close(fig)


def plot_feature_importance(imp_df: pd.DataFrame, title: str, out_path: Path):
    if imp_df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    imp_df = imp_df.sort_values("importance")
    ax.barh(imp_df["feature"], imp_df["importance"], color="tab:green")
    ax.set_title(title)
    ax.set_xlabel("Importance (gain)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_electrical_cooling_coupling(test_df: pd.DataFrame, out_dir: Path):
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(
        test_df["it_load_mw"], test_df["cooling_load_mw"],
        c=test_df["outdoor_temp_c"], cmap="coolwarm", s=6, alpha=0.5,
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Outdoor Temp (C)")
    ax.set_xlabel("IT Load (MW)")
    ax.set_ylabel("Cooling Load (MW)")
    ax.set_title("Electrical vs Cooling Load Coupling\n(color = outdoor temperature)")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "electrical_cooling_coupling.png", dpi=120)
    plt.close(fig)


def plot_metric_comparison(metrics_df: pd.DataFrame, out_dir: Path):
    """Bar chart comparing Independent vs Coupled MAE per target/horizon --
    the key plot for answering the project's research question."""
    fig, axes = plt.subplots(1, len(TARGET_COLS), figsize=(15, 4), sharey=False)
    for ax, target in zip(axes, TARGET_COLS):
        sub = metrics_df[metrics_df["target"] == target]
        pivot = sub.pivot(index="horizon", columns="architecture", values="MAE_MW")
        pivot.plot(kind="bar", ax=ax)
        ax.set_title(target)
        ax.set_ylabel("MAE (MW)")
        ax.legend(fontsize=8)
    fig.suptitle("Independent vs Coupled Architecture: MAE by target & horizon")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "architecture_comparison_mae.png", dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_evaluation(independent: IndependentArchitecture, coupled: CoupledArchitecture,
                    test_df: pd.DataFrame, cfg: SimConfig, out_dir: Path = PLOT_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)

    preds_ind = independent.predict(test_df)
    preds_coup = coupled.predict(test_df)

    metrics_ind = compute_metrics(test_df, preds_ind, cfg, "Independent")
    metrics_coup = compute_metrics(test_df, preds_coup, cfg, "Coupled")
    metrics_all = pd.concat([metrics_ind, metrics_coup], ignore_index=True)
    metrics_all.to_csv(out_dir.parent / "metrics.csv", index=False)
    print("\n=== Forecast accuracy (MAE / RMSE / MAPE) ===")
    print(metrics_all.to_string(index=False))

    print("\n=== Physical consistency check (implied PUE from predictions) ===")
    consistency_rows = []
    for h_label in cfg.forecast.horizon_labels:
        for name, preds in [("Independent", preds_ind), ("Coupled", preds_coup)]:
            res = physical_consistency_check(test_df, preds, cfg, h_label)
            res["architecture"] = name
            consistency_rows.append(res)
    consistency_df = pd.DataFrame(consistency_rows)
    print(consistency_df.to_string(index=False))
    consistency_df.to_csv(out_dir.parent / "physical_consistency.csv", index=False)

    print("\n=== Illustrative curtailment / overprovisioning analysis (Coupled model) ===")
    waste_rows = [curtailment_and_waste_analysis(test_df, preds_coup, cfg, h) for h in cfg.forecast.horizon_labels]
    waste_df = pd.DataFrame(waste_rows)
    print(waste_df.to_string(index=False))
    waste_df.to_csv(out_dir.parent / "curtailment_waste_analysis.csv", index=False)

    print("\n=== Generating plots ===")
    for h_label in cfg.forecast.horizon_labels:
        plot_true_vs_pred(test_df, preds_ind, cfg, h_label, "Independent", out_dir)
        plot_true_vs_pred(test_df, preds_coup, cfg, h_label, "Coupled", out_dir)
        plot_residuals(test_df, preds_coup, cfg, h_label, "Coupled", out_dir)

    plot_electrical_cooling_coupling(test_df, out_dir)
    plot_metric_comparison(metrics_all, out_dir)

    # Feature importance for the shortest horizon of the coupled cooling model
    h0 = cfg.forecast.horizon_labels[0]
    imp_cooling = feature_importance_report(coupled, "cooling_load_mw", h0, test_df)
    plot_feature_importance(imp_cooling, f"Coupled cooling model feature importance ({h0})",
                             out_dir / f"feature_importance_coupled_cooling_{h0}.png")
    imp_it = feature_importance_report(coupled, "it_load_mw", h0, test_df)
    plot_feature_importance(imp_it, f"Coupled electrical model feature importance ({h0})",
                             out_dir / f"feature_importance_coupled_it_{h0}.png")

    print(f"\nAll plots saved to {out_dir.resolve()}")
    return metrics_all, consistency_df, waste_df, preds_ind, preds_coup


if __name__ == "__main__":
    import pickle
    with open("artifacts/independent_model.pkl", "rb") as f:
        independent = pickle.load(f)
    with open("artifacts/coupled_model.pkl", "rb") as f:
        coupled = pickle.load(f)
    with open("artifacts/config.pkl", "rb") as f:
        cfg = pickle.load(f)
    test_df = pd.read_csv("artifacts/test_set.csv", index_col=0, parse_dates=True)
    run_evaluation(independent, coupled, test_df, cfg)
