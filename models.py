"""
models.py
---------
Tree-based, multi-horizon, multi-output forecasters for electrical /
cooling / total facility load, in two architectures:

  A) IndependentArchitecture
     One model per (target, horizon[, quantile]) trained on the same
     physics-informed feature set. Targets don't see each other's
     predictions at all -- this is the "naive" baseline architecture.

  B) CoupledArchitecture
     Trains the electrical model first. Its (in-sample, causal) prediction
     is then fed as an INPUT FEATURE to the cooling model, and both
     electrical + cooling predictions feed the total model. This enforces
     the physical dependency electrical -> cooling -> total instead of
     treating the three series as unrelated regression problems.

Both architectures support:
  * Direct multi-step forecasting (a separate model per horizon, rather
    than recursive multi-step, to avoid compounding error -- default and
    recommended here).
  * LightGBM quantile regression for prediction intervals (falls back to
    XGBoost/sklearn GradientBoosting point forecasts if LightGBM/XGBoost
    aren't available, though both are expected to be installed).

Why direct-multi-horizon instead of recursive: recursive forecasting
(predict t+1, feed it back in to predict t+2, ...) compounds errors and
is more fragile when the model already has strong lag/rolling features.
Direct multi-horizon trains one model per horizon on shifted targets,
which is simpler to reason about and works well with tree ensembles.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False

try:
    import xgboost as xgb
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

from config import SimConfig

TARGET_COLS = ["it_load_mw", "cooling_load_mw", "total_load_mw"]

# Columns that must NEVER be used as model inputs: ground-truth targets at
# time t (not lagged), the true simulator PUE (would leak the answer), and
# bookkeeping / raw sensor columns.
_LEAKY_OR_NONFEATURE_COLS = set(TARGET_COLS) | {
    "pue", "it_load_mw_measured", "cooling_load_mw_measured", "total_load_mw_measured",
    "anomaly_curtailment", "anomaly_degradation", "anomaly_scaleup",
    "wet_bulb_c",  # keep weather forecasts realistic: only temp/humidity assumed "known"/forecasted
}


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in _LEAKY_OR_NONFEATURE_COLS]


def make_horizon_target(df: pd.DataFrame, target_col: str, horizon_steps: int) -> pd.Series:
    """Direct multi-step target: the value of `target_col` `horizon_steps`
    samples in the future, aligned to the current row's feature vector.
    """
    return df[target_col].shift(-horizon_steps)


# ---------------------------------------------------------------------------
# Single-model training helper (LightGBM primary, XGBoost fallback)
# ---------------------------------------------------------------------------

def _fit_point_model(X_train, y_train, X_val=None, y_val=None, params: Optional[dict] = None):
    params = params or {}
    if _HAS_LGB:
        model = lgb.LGBMRegressor(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=-1,
            **params,
        )
        eval_set = [(X_val, y_val)] if X_val is not None else None
        callbacks = [lgb.early_stopping(30, verbose=False)] if eval_set else None
        model.fit(X_train, y_train, eval_set=eval_set, callbacks=callbacks)
        return model
    elif _HAS_XGB:
        model = xgb.XGBRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            **params,
        )
        model.fit(X_train, y_train)
        return model
    else:
        from sklearn.ensemble import GradientBoostingRegressor
        model = GradientBoostingRegressor(random_state=42, **params)
        model.fit(X_train, y_train)
        return model


def _fit_quantile_model(X_train, y_train, quantile: float, X_val=None, y_val=None):
    if _HAS_LGB:
        model = lgb.LGBMRegressor(
            objective="quantile", alpha=quantile,
            n_estimators=400, learning_rate=0.05, num_leaves=31,
            min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbosity=-1,
        )
        eval_set = [(X_val, y_val)] if X_val is not None else None
        callbacks = [lgb.early_stopping(30, verbose=False)] if eval_set else None
        model.fit(X_train, y_train, eval_set=eval_set, callbacks=callbacks)
        return model
    else:
        from sklearn.ensemble import GradientBoostingRegressor
        model = GradientBoostingRegressor(loss="quantile", alpha=quantile, random_state=42)
        model.fit(X_train, y_train)
        return model


def _predict(model, X):
    return np.asarray(model.predict(X))


# ---------------------------------------------------------------------------
# Independent architecture
# ---------------------------------------------------------------------------

@dataclass
class IndependentArchitecture:
    """One model per (target, horizon[, quantile]) -- targets never see
    each other's predictions.
    """
    cfg: SimConfig
    models: Dict[str, object] = field(default_factory=dict)   # key -> fitted model
    feature_cols: List[str] = field(default_factory=list)

    def _key(self, target: str, horizon_label: str) -> str:
        return f"{target}|h={horizon_label}"

    def fit(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None):
        self.feature_cols = get_feature_columns(train_df)
        fc = self.cfg.forecast

        for target in TARGET_COLS:
            for h_steps, h_label in zip(fc.horizon_steps, fc.horizon_labels):
                y_train = make_horizon_target(train_df, target, h_steps)
                valid = y_train.notna()
                X_tr = train_df.loc[valid, self.feature_cols]
                y_tr = y_train.loc[valid]

                X_val = y_val = None
                if val_df is not None:
                    y_val_full = make_horizon_target(val_df, target, h_steps)
                    v_valid = y_val_full.notna()
                    X_val = val_df.loc[v_valid, self.feature_cols]
                    y_val = y_val_full.loc[v_valid]

                # Point (median-ish) model
                key = self._key(target, h_label)
                self.models[key] = _fit_point_model(X_tr, y_tr, X_val, y_val)

                # Quantile models for prediction intervals
                for q in fc.quantiles:
                    qkey = f"{key}|q={q}"
                    self.models[qkey] = _fit_quantile_model(X_tr, y_tr, q, X_val, y_val)
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        fc = self.cfg.forecast
        preds = {}
        for target in TARGET_COLS:
            for h_label in fc.horizon_labels:
                key = self._key(target, h_label)
                preds[f"pred_{target}_{h_label}"] = _predict(self.models[key], df[self.feature_cols])
                for q in fc.quantiles:
                    qkey = f"{key}|q={q}"
                    preds[f"pred_{target}_{h_label}_q{q}"] = _predict(self.models[qkey], df[self.feature_cols])
        return pd.DataFrame(preds, index=df.index)


# ---------------------------------------------------------------------------
# Coupled architecture
# ---------------------------------------------------------------------------

@dataclass
class CoupledArchitecture:
    """Sequential architecture enforcing the physical dependency:
    electrical -> cooling -> total. The cooling model receives the
    electrical model's (causal, held-out-fold) prediction as an extra
    input feature; the total model receives both.

    To avoid the cooling/total models training on artificially perfect
    "predictions" (which would be the case if we just used in-sample
    fitted values from the electrical model), we generate the electrical
    model's predictions on the TRAINING set via K-fold cross-validation
    (out-of-fold predictions) before using them as a feature -- otherwise
    the coupled model would learn to lean on a feature that's far more
    accurate at train time than it will be at inference time.
    """
    cfg: SimConfig
    models: Dict[str, object] = field(default_factory=dict)
    feature_cols: List[str] = field(default_factory=list)
    n_folds: int = 6

    def _oof_predict(self, df: pd.DataFrame, target: str, h_steps: int, feature_cols: List[str]) -> np.ndarray:
        """Time-series-safe out-of-fold predictions for `target` at
        horizon `h_steps`, used as a coupling feature during training.
        Uses expanding-window folds (never trains on future data).
        """
        n = len(df)
        y_full = make_horizon_target(df, target, h_steps)
        oof = np.full(n, np.nan)
        fold_bounds = np.linspace(0, n, self.n_folds + 1).astype(int)
        for i in range(1, self.n_folds):
            train_end = fold_bounds[i]
            val_start, val_end = fold_bounds[i], fold_bounds[i + 1] if i + 1 < len(fold_bounds) else n
            y_tr = y_full.iloc[:train_end]
            valid = y_tr.notna()
            X_tr = df.iloc[:train_end].loc[valid, feature_cols]
            y_tr = y_tr.loc[valid]
            if len(X_tr) < 50:
                continue
            model = _fit_point_model(X_tr, y_tr)
            X_val = df.iloc[val_start:val_end][feature_cols]
            oof[val_start:val_end] = _predict(model, X_val)
        return oof

    def fit(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None):
        fc = self.cfg.forecast
        base_feature_cols = get_feature_columns(train_df)
        self.feature_cols = base_feature_cols

        for h_steps, h_label in zip(fc.horizon_steps, fc.horizon_labels):
            # --- Stage 1: electrical (IT load) model, standard features only ---
            y_it_train = make_horizon_target(train_df, "it_load_mw", h_steps)
            valid = y_it_train.notna()
            X_it_tr = train_df.loc[valid, base_feature_cols]
            y_it_tr = y_it_train.loc[valid]

            X_val = y_val = None
            if val_df is not None:
                y_it_val_full = make_horizon_target(val_df, "it_load_mw", h_steps)
                v_valid = y_it_val_full.notna()
                X_val = val_df.loc[v_valid, base_feature_cols]
                y_val = y_it_val_full.loc[v_valid]

            it_model = _fit_point_model(X_it_tr, y_it_tr, X_val, y_val)
            self.models[f"it_load_mw|h={h_label}"] = it_model
            for q in fc.quantiles:
                self.models[f"it_load_mw|h={h_label}|q={q}"] = _fit_quantile_model(X_it_tr, y_it_tr, q, X_val, y_val)

            # Out-of-fold electrical predictions -> coupling feature for cooling model
            it_pred_oof = self._oof_predict(train_df, "it_load_mw", h_steps, base_feature_cols)
            train_aug = train_df.copy()
            train_aug[f"coupled_it_pred_{h_label}"] = it_pred_oof
            cooling_feature_cols = base_feature_cols + [f"coupled_it_pred_{h_label}"]

            # --- Stage 2: cooling model, sees electrical prediction as input ---
            y_cool_train = make_horizon_target(train_aug, "cooling_load_mw", h_steps)
            valid_c = y_cool_train.notna() & train_aug[f"coupled_it_pred_{h_label}"].notna()
            X_cool_tr = train_aug.loc[valid_c, cooling_feature_cols]
            y_cool_tr = y_cool_train.loc[valid_c]

            X_cool_val = y_cool_val = None
            if val_df is not None:
                val_aug = val_df.copy()
                val_aug[f"coupled_it_pred_{h_label}"] = _predict(it_model, val_df[base_feature_cols])
                y_cool_val_full = make_horizon_target(val_aug, "cooling_load_mw", h_steps)
                v_valid_c = y_cool_val_full.notna()
                X_cool_val = val_aug.loc[v_valid_c, cooling_feature_cols]
                y_cool_val = y_cool_val_full.loc[v_valid_c]

            cool_model = _fit_point_model(X_cool_tr, y_cool_tr, X_cool_val, y_cool_val)
            self.models[f"cooling_load_mw|h={h_label}"] = cool_model
            for q in fc.quantiles:
                self.models[f"cooling_load_mw|h={h_label}|q={q}"] = _fit_quantile_model(X_cool_tr, y_cool_tr, q, X_cool_val, y_cool_val)
            self.models[f"cooling_feature_cols|h={h_label}"] = cooling_feature_cols

            # --- Stage 3: total model, sees electrical + cooling predictions ---
            cool_pred_oof = np.full(len(train_aug), np.nan)
            # reuse expanding-fold approach for cooling OOF predictions too
            n = len(train_aug)
            fold_bounds = np.linspace(0, n, self.n_folds + 1).astype(int)
            for i in range(1, self.n_folds):
                train_end = fold_bounds[i]
                val_start, val_end = fold_bounds[i], fold_bounds[i + 1] if i + 1 < len(fold_bounds) else n
                y_tr_fold = y_cool_train.iloc[:train_end]
                valid_fold = y_tr_fold.notna() & train_aug[f"coupled_it_pred_{h_label}"].iloc[:train_end].notna()
                X_tr_fold = train_aug.iloc[:train_end].loc[valid_fold, cooling_feature_cols]
                y_tr_fold = y_tr_fold.loc[valid_fold]
                if len(X_tr_fold) < 50:
                    continue
                fold_model = _fit_point_model(X_tr_fold, y_tr_fold)
                X_val_fold = train_aug.iloc[val_start:val_end][cooling_feature_cols]
                cool_pred_oof[val_start:val_end] = _predict(fold_model, X_val_fold)

            train_aug[f"coupled_cooling_pred_{h_label}"] = cool_pred_oof
            total_feature_cols = cooling_feature_cols + [f"coupled_cooling_pred_{h_label}"]

            y_total_train = make_horizon_target(train_aug, "total_load_mw", h_steps)
            valid_t = (
                y_total_train.notna()
                & train_aug[f"coupled_it_pred_{h_label}"].notna()
                & train_aug[f"coupled_cooling_pred_{h_label}"].notna()
            )
            X_total_tr = train_aug.loc[valid_t, total_feature_cols]
            y_total_tr = y_total_train.loc[valid_t]

            X_total_val = y_total_val = None
            if val_df is not None:
                val_aug[f"coupled_cooling_pred_{h_label}"] = _predict(cool_model, val_aug[cooling_feature_cols])
                y_total_val_full = make_horizon_target(val_aug, "total_load_mw", h_steps)
                v_valid_t = y_total_val_full.notna()
                X_total_val = val_aug.loc[v_valid_t, total_feature_cols]
                y_total_val = y_total_val_full.loc[v_valid_t]

            total_model = _fit_point_model(X_total_tr, y_total_tr, X_total_val, y_total_val)
            self.models[f"total_load_mw|h={h_label}"] = total_model
            for q in fc.quantiles:
                self.models[f"total_load_mw|h={h_label}|q={q}"] = _fit_quantile_model(X_total_tr, y_total_tr, q, X_total_val, y_total_val)
            self.models[f"total_feature_cols|h={h_label}"] = total_feature_cols

        return self

    def get_model_and_inputs(self, df: pd.DataFrame, target: str, h_label: str):
        """Return (fitted_model, X) for `target` at horizon `h_label`,
        building whatever coupling-feature columns that model expects
        (electrical prediction for the cooling model; electrical + cooling
        predictions for the total model). Used by evaluate.py so SHAP /
        feature-importance analysis sees exactly the columns the model was
        trained on -- avoids the classic "N features on model vs M on
        explainer" mismatch when coupling features aren't present in the
        raw dataframe.
        """
        it_model = self.models[f"it_load_mw|h={h_label}"]
        if target == "it_load_mw":
            return it_model, df[self.feature_cols]

        it_pred = _predict(it_model, df[self.feature_cols])
        cooling_feature_cols = self.models[f"cooling_feature_cols|h={h_label}"]
        df_c = df.copy()
        df_c[f"coupled_it_pred_{h_label}"] = it_pred
        cool_model = self.models[f"cooling_load_mw|h={h_label}"]
        if target == "cooling_load_mw":
            return cool_model, df_c[cooling_feature_cols]

        cool_pred = _predict(cool_model, df_c[cooling_feature_cols])
        total_feature_cols = self.models[f"total_feature_cols|h={h_label}"]
        df_t = df_c.copy()
        df_t[f"coupled_cooling_pred_{h_label}"] = cool_pred
        total_model = self.models[f"total_load_mw|h={h_label}"]
        return total_model, df_t[total_feature_cols]

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        fc = self.cfg.forecast
        preds = {}
        for h_label in fc.horizon_labels:
            it_model = self.models[f"it_load_mw|h={h_label}"]
            it_pred = _predict(it_model, df[self.feature_cols])
            preds[f"pred_it_load_mw_{h_label}"] = it_pred
            for q in fc.quantiles:
                preds[f"pred_it_load_mw_{h_label}_q{q}"] = _predict(self.models[f"it_load_mw|h={h_label}|q={q}"], df[self.feature_cols])

            cooling_feature_cols = self.models[f"cooling_feature_cols|h={h_label}"]
            df_c = df.copy()
            df_c[f"coupled_it_pred_{h_label}"] = it_pred
            cool_model = self.models[f"cooling_load_mw|h={h_label}"]
            cool_pred = _predict(cool_model, df_c[cooling_feature_cols])
            preds[f"pred_cooling_load_mw_{h_label}"] = cool_pred
            for q in fc.quantiles:
                preds[f"pred_cooling_load_mw_{h_label}_q{q}"] = _predict(self.models[f"cooling_load_mw|h={h_label}|q={q}"], df_c[cooling_feature_cols])

            total_feature_cols = self.models[f"total_feature_cols|h={h_label}"]
            df_t = df_c.copy()
            df_t[f"coupled_cooling_pred_{h_label}"] = cool_pred
            total_model = self.models[f"total_load_mw|h={h_label}"]
            total_pred = _predict(total_model, df_t[total_feature_cols])
            preds[f"pred_total_load_mw_{h_label}"] = total_pred
            for q in fc.quantiles:
                preds[f"pred_total_load_mw_{h_label}_q{q}"] = _predict(self.models[f"total_load_mw|h={h_label}|q={q}"], df_t[total_feature_cols])

        return pd.DataFrame(preds, index=df.index)
