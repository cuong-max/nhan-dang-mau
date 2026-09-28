# -*- coding: utf-8 -*-
"""
Regression pipeline with preprocessing and PCA dimensionality reduction
=======================================================================
Assignment: Build a prediction program on a dataset with a continuous target,
compare a baseline against several regression models, and study the effect of PCA.

Dataset: Iris (UCI ML Repository, ID 53). 150 samples, 4 features.
Iris is originally a CLASSIFICATION problem (target = flower species). Because the
assignment requires a CONTINUOUS target (regression), we reframe it as REGRESSION:
    - Target (y):    petal width (cm)  -- a continuous variable
    - Features (X):  sepal length, sepal width, petal length (+ polynomial features)

Model selection criteria: MAE / RMSE, R^2, and stability across Cross-Validation.

Run:  ./venv/Scripts/python.exe src/pipeline_regression.py
Outputs: outputs/figures/*.png, outputs/results_cv.csv, outputs/results_test.csv,
         outputs/summary.json
"""
import os
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, RepeatedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
)

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({"figure.dpi": 120, "savefig.bbox": "tight", "font.size": 10})


# ----------------------------------------------------------------------------
# 1. LOAD DATA & REFRAME AS A REGRESSION PROBLEM
# ----------------------------------------------------------------------------
def load_data():
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    # Column names: 'sepal length (cm)', 'sepal width (cm)', 'petal length (cm)',
    #               'petal width (cm)', 'target'
    target_col = "petal width (cm)"
    feature_cols = [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
    ]
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    return df, X, y, feature_cols, target_col


# ----------------------------------------------------------------------------
# 2. BUILD PIPELINES (with / without PCA)
# ----------------------------------------------------------------------------
def get_models():
    """Return a dict {model_name: estimator}."""
    return {
        "Linear": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, random_state=RANDOM_STATE),
        "Lasso": Lasso(alpha=0.01, random_state=RANDOM_STATE, max_iter=10000),
        "KNN": KNeighborsRegressor(n_neighbors=5),
        "DecisionTree": DecisionTreeRegressor(random_state=RANDOM_STATE, max_depth=5),
        "RandomForest": RandomForestRegressor(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
        "SVR_RBF": SVR(kernel="rbf", C=10, gamma="scale"),
    }


def build_pipeline(model, use_pca=False, n_components=0.95, poly_degree=2):
    """
    Pipeline: PolynomialFeatures -> StandardScaler -> [PCA] -> model
    - PolynomialFeatures creates a higher-dimensional (collinear) space so PCA is meaningful.
    - StandardScaler standardizes features (required before PCA and for distance-based models).
    - PCA (optional) reduces dimensionality, keeping 95% of the variance by default.
    """
    steps = [
        ("poly", PolynomialFeatures(degree=poly_degree, include_bias=False)),
        ("scaler", StandardScaler()),
    ]
    if use_pca:
        steps.append(("pca", PCA(n_components=n_components, random_state=RANDOM_STATE)))
    steps.append(("model", model))
    return Pipeline(steps)


# ----------------------------------------------------------------------------
# 3. CROSS-VALIDATION EVALUATION (stability)
# ----------------------------------------------------------------------------
def evaluate_cv(X, y):
    cv = RepeatedKFold(n_splits=5, n_repeats=4, random_state=RANDOM_STATE)
    scoring = {
        "MAE": "neg_mean_absolute_error",
        "RMSE": "neg_root_mean_squared_error",
        "R2": "r2",
    }
    rows = []
    # Baseline first
    configs = [("Baseline (mean)", DummyRegressor(strategy="mean"), None)]
    for name, model in get_models().items():
        configs.append((name, model, False))  # without PCA
        configs.append((name, model, True))  # with PCA

    for name, model, use_pca in configs:
        if name == "Baseline (mean)":
            pipe = model  # DummyRegressor needs no preprocessing
            variant = "Baseline"
        else:
            pipe = build_pipeline(model, use_pca=use_pca)
            variant = "PCA" if use_pca else "No-PCA"

        res = cross_validate(
            pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=False
        )
        rows.append(
            {
                "Model": name,
                "Variant": variant,
                "MAE_mean": -res["test_MAE"].mean(),
                "MAE_std": res["test_MAE"].std(),
                "RMSE_mean": -res["test_RMSE"].mean(),
                "RMSE_std": res["test_RMSE"].std(),
                "R2_mean": res["test_R2"].mean(),
                "R2_std": res["test_R2"].std(),
            }
        )
    cv_df = pd.DataFrame(rows)
    return cv_df


# ----------------------------------------------------------------------------
# 4. HOLD-OUT TEST EVALUATION
# ----------------------------------------------------------------------------
def evaluate_holdout(X, y):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE
    )
    rows = []
    predictions = {}

    # Baseline
    base = DummyRegressor(strategy="mean").fit(X_tr, y_tr)
    yp = base.predict(X_te)
    rows.append(_metric_row("Baseline (mean)", "Baseline", y_te, yp))

    for name, model in get_models().items():
        for use_pca in (False, True):
            pipe = build_pipeline(model, use_pca=use_pca).fit(X_tr, y_tr)
            yp = pipe.predict(X_te)
            variant = "PCA" if use_pca else "No-PCA"
            rows.append(_metric_row(name, variant, y_te, yp))
            predictions[f"{name}|{variant}"] = (np.asarray(y_te), yp)

    test_df = pd.DataFrame(rows)
    return test_df, predictions, (X_tr, X_te, y_tr, y_te)


def _metric_row(name, variant, y_true, y_pred):
    return {
        "Model": name,
        "Variant": variant,
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": root_mean_squared_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


# ----------------------------------------------------------------------------
# 5. PLOTS
# ----------------------------------------------------------------------------
def plot_correlation(df, path):
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(corr.columns, fontsize=8)
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            ax.text(
                j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8
            )
    ax.set_title("Correlation matrix (Iris dataset)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(path)
    plt.close(fig)


def plot_target_distribution(y, target_col, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(y, bins=20, color="#4C72B0", edgecolor="white")
    ax.set_xlabel(target_col)
    ax.set_ylabel("Frequency")
    ax.set_title(f"Continuous target distribution: {target_col}")
    fig.savefig(path)
    plt.close(fig)


def plot_pca_variance(X, path):
    pipe = Pipeline(
        [
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("scaler", StandardScaler()),
            ("pca", PCA(random_state=RANDOM_STATE)),
        ]
    )
    pipe.fit(X)
    pca = pipe.named_steps["pca"]
    ratio = pca.explained_variance_ratio_
    cum = np.cumsum(ratio)
    fig, ax = plt.subplots(figsize=(7, 4))
    comps = np.arange(1, len(ratio) + 1)
    ax.bar(comps, ratio, alpha=0.6, label="Per-PC variance")
    ax.plot(comps, cum, "o-", color="crimson", label="Cumulative variance")
    ax.axhline(0.95, ls="--", color="gray", label="95% threshold")
    ax.set_xlabel("Number of principal components (PC)")
    ax.set_ylabel("Explained variance ratio")
    ax.set_title("PCA - explained variance (after PolynomialFeatures degree=2)")
    ax.legend(fontsize=8)
    ax.set_xticks(comps)
    fig.savefig(path)
    plt.close(fig)
    n95 = int(np.argmax(cum >= 0.95) + 1)
    return ratio, cum, n95


def plot_model_comparison(test_df, path):
    """Compare test-set RMSE across models (No-PCA vs PCA)."""
    df = test_df[test_df["Model"] != "Baseline (mean)"].copy()
    models = df["Model"].unique()
    x = np.arange(len(models))
    w = 0.38
    no_pca = [df[(df.Model == m) & (df.Variant == "No-PCA")]["RMSE"].values[0] for m in models]
    pca = [df[(df.Model == m) & (df.Variant == "PCA")]["RMSE"].values[0] for m in models]
    baseline = test_df[test_df["Model"] == "Baseline (mean)"]["RMSE"].values[0]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - w / 2, no_pca, w, label="Without PCA", color="#4C72B0")
    ax.bar(x + w / 2, pca, w, label="With PCA", color="#DD8452")
    ax.axhline(baseline, ls="--", color="red", label=f"Baseline RMSE={baseline:.3f}")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylabel("RMSE (test) - lower is better")
    ax.set_title("Test-set RMSE comparison: with / without PCA")
    ax.legend(fontsize=8)
    fig.savefig(path)
    plt.close(fig)


def plot_cv_stability(cv_df, path):
    """Plot CV R2 mean +/- std (stability)."""
    df = cv_df[cv_df["Model"] != "Baseline (mean)"].copy()
    df = df[df["Variant"] == "No-PCA"].sort_values("R2_mean", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(
        df["Model"],
        df["R2_mean"],
        xerr=df["R2_std"],
        color="#55A868",
        capsize=3,
    )
    ax.set_xlabel("Mean CV R2 (error bar = standard deviation)")
    ax.set_title("Model stability across Cross-Validation (without PCA)")
    fig.savefig(path)
    plt.close(fig)


def plot_pred_vs_actual(predictions, best_key, path):
    y_true, y_pred = predictions[best_key]
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(y_true, y_pred, alpha=0.7, color="#4C72B0", edgecolor="white")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", label="y = x (ideal)")
    ax.set_xlabel("Actual value")
    ax.set_ylabel("Predicted value")
    ax.set_title(f"Predicted vs Actual - {best_key}")
    ax.legend(fontsize=8)
    fig.savefig(path)
    plt.close(fig)


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("REGRESSION PIPELINE WITH PREPROCESSING + PCA  |  Dataset: Iris (as regression)")
    print("=" * 70)

    df, X, y, feature_cols, target_col = load_data()
    print(f"Samples: {len(df)} | Features: {feature_cols} | Target: {target_col}")
    print(f"Target range: [{y.min():.2f}, {y.max():.2f}]  mean={y.mean():.3f}\n")

    # --- Exploratory plots ---
    plot_correlation(df.rename(columns=lambda c: c.replace(" (cm)", "")),
                     os.path.join(FIG, "01_correlation.png"))
    plot_target_distribution(y, target_col, os.path.join(FIG, "02_target_dist.png"))
    ratio, cum, n95 = plot_pca_variance(X, os.path.join(FIG, "03_pca_variance.png"))
    print(f"PCA: {n95} PC(s) needed to keep >=95% variance "
          f"(original space after poly = {len(ratio)} dims)\n")

    # --- Cross-validation ---
    print("Running Cross-Validation (5-fold x 4 repeats)...")
    cv_df = evaluate_cv(X, y)
    cv_df_sorted = cv_df.sort_values(["RMSE_mean"]).reset_index(drop=True)
    cv_df_sorted.to_csv(os.path.join(OUT, "results_cv.csv"), index=False)
    print("CV results (top by RMSE):")
    print(cv_df_sorted.head(8).round(4).to_string(index=False))
    print()

    # --- Hold-out test ---
    print("Evaluating on independent test set (25%)...")
    test_df, predictions, _ = evaluate_holdout(X, y)
    test_df_sorted = test_df.sort_values("RMSE").reset_index(drop=True)
    test_df_sorted.to_csv(os.path.join(OUT, "results_test.csv"), index=False)
    print("Test results (top by RMSE):")
    print(test_df_sorted.head(8).round(4).to_string(index=False))
    print()

    # --- Comparison plots ---
    plot_model_comparison(test_df, os.path.join(FIG, "04_model_comparison.png"))
    plot_cv_stability(cv_df, os.path.join(FIG, "05_cv_stability.png"))

    # Best model by CV (excluding baseline)
    best = cv_df[cv_df["Model"] != "Baseline (mean)"].sort_values("RMSE_mean").iloc[0]
    best_key = f"{best['Model']}|{best['Variant']}"
    if best_key not in predictions:  # fallback
        best_key = list(predictions.keys())[0]
    plot_pred_vs_actual(predictions, best_key, os.path.join(FIG, "06_pred_vs_actual.png"))

    # --- Summarize PCA impact ---
    pca_impact = []
    for m in get_models().keys():
        r_no = test_df[(test_df.Model == m) & (test_df.Variant == "No-PCA")]["RMSE"].values[0]
        r_pca = test_df[(test_df.Model == m) & (test_df.Variant == "PCA")]["RMSE"].values[0]
        pca_impact.append({"Model": m, "RMSE_NoPCA": r_no, "RMSE_PCA": r_pca,
                           "Delta": r_pca - r_no})
    pca_impact_df = pd.DataFrame(pca_impact)

    # --- Save summary JSON for report generation ---
    baseline_test = test_df[test_df["Model"] == "Baseline (mean)"].iloc[0]
    summary = {
        "n_samples": int(len(df)),
        "features": feature_cols,
        "target": target_col,
        "target_min": float(y.min()),
        "target_max": float(y.max()),
        "target_mean": float(y.mean()),
        "target_std": float(y.std()),
        "poly_dim": int(len(ratio)),
        "pca_n95": int(n95),
        "pca_ratio": [float(v) for v in ratio],
        "pca_cum": [float(v) for v in cum],
        "baseline_test": {
            "MAE": float(baseline_test["MAE"]),
            "RMSE": float(baseline_test["RMSE"]),
            "R2": float(baseline_test["R2"]),
        },
        "best_cv": {
            "Model": best["Model"],
            "Variant": best["Variant"],
            "RMSE_mean": float(best["RMSE_mean"]),
            "RMSE_std": float(best["RMSE_std"]),
            "MAE_mean": float(best["MAE_mean"]),
            "R2_mean": float(best["R2_mean"]),
            "R2_std": float(best["R2_std"]),
        },
        "best_test_key": best_key,
        "pca_impact": pca_impact_df.to_dict(orient="records"),
    }
    with open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=" * 70)
    print(f"Best model (by CV RMSE): {best['Model']} [{best['Variant']}]")
    print(f"  RMSE={best['RMSE_mean']:.4f} +/- {best['RMSE_std']:.4f} | "
          f"MAE={best['MAE_mean']:.4f} | R2={best['R2_mean']:.4f}")
    print(f"Baseline test: RMSE={baseline_test['RMSE']:.4f} | R2={baseline_test['R2']:.4f}")
    print("=" * 70)
    print(f"\nSaved: outputs/results_cv.csv, results_test.csv, summary.json")
    print(f"       outputs/figures/*.png (6 figures)")


if __name__ == "__main__":
    main()
