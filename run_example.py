"""
run_example.py
---------------
End-to-end example: generate synthetic data -> feature engineering ->
train Independent & Coupled architectures -> evaluate -> plots.

Run with:  python run_example.py

Adjust `SimConfig` below (cluster size, climate, PUE assumptions, dataset
length) to explore different scenarios -- everything is parameterized in
config.py.
"""
import pickle
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from config import SimConfig
from train import run_training
from evaluate import run_evaluation, shap_summary
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    t0 = time.time()

    # ---- 1. Configure the scenario ----
    # Default: ~1 year of 15-minute data for a 120MW AI data center campus
    # in a moderate/hot climate (Phoenix-like defaults). Modify freely.
    cfg = SimConfig(
        start_date="2023-01-01",
        periods_days=365,
        freq_minutes=15,
        random_seed=42,
    )

    # ---- 2. Generate data, engineer features, train both architectures ----
    independent, coupled, train_df, val_df, test_df = run_training(cfg, save_artifacts=True)

    # ---- 3. Evaluate: metrics, physical consistency, curtailment/waste, plots ----
    metrics_all, consistency_df, waste_df, preds_ind, preds_coup = run_evaluation(
        independent, coupled, test_df, cfg
    )

    # ---- 4. SHAP analysis (physical drivers) for the shortest-horizon cooling model ----
    h0 = cfg.forecast.horizon_labels[0]
    shap_values, X_sample = shap_summary(coupled, "cooling_load_mw", h0, test_df)
    if shap_values is not None:
        import shap
        plt.figure(figsize=(8, 6))
        shap.summary_plot(shap_values, X_sample, show=False, max_display=15)
        plt.title(f"SHAP: Coupled cooling model drivers ({h0} ahead)")
        plt.tight_layout()
        out_path = Path("artifacts/plots/shap_summary_coupled_cooling.png")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=120)
        plt.close()
        print(f"SHAP summary plot saved to {out_path}")
    else:
        print("SHAP not available or failed -- skipping SHAP plot (feature_importances_ plots still generated).")

    elapsed = time.time() - t0
    print(f"\n=== Done in {elapsed / 60:.1f} minutes ===")
    print("Artifacts directory: artifacts/  (models, synthetic_data.csv, metrics.csv, plots/)")

    # ---- 5. Summarize the research-question answer in plain terms ----
    print("\n=== Research question summary ===")
    for target in ["it_load_mw", "cooling_load_mw", "total_load_mw"]:
        sub = metrics_all[metrics_all["target"] == target]
        ind_mae = sub[sub.architecture == "Independent"]["MAE_MW"].mean()
        coup_mae = sub[sub.architecture == "Coupled"]["MAE_MW"].mean()
        better = "Coupled" if coup_mae < ind_mae else "Independent"
        delta_pct = 100 * (ind_mae - coup_mae) / ind_mae if ind_mae else float("nan")
        print(f"  {target}: Independent MAE={ind_mae:.2f} MW, Coupled MAE={coup_mae:.2f} MW "
              f"-> {better} architecture is better on average ({delta_pct:+.1f}% MAE change)")


if __name__ == "__main__":
    main()
