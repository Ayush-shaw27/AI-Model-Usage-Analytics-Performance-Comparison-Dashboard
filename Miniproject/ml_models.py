"""
ml_models.py  —  AI Model Analytics & Decision Intelligence Platform
Machine Learning & Analytical Layer

FIXES applied
─────────────
1. detect_outliers: Z-score NaN when std=0 → safe_zscore() returns 0 instead of NaN.
2. compute_composite_score: MinMaxScaler on single-row / identical rows → returns 50.0
   (neutral midpoint) instead of NaN/zero.
3. perform_clustering: guard against n_clusters > rows.
4. train_cost_model: explicit minimum-row guard + clear None return.
5. provider_summary: handles missing composite_score column gracefully.
6. recommend_models: returns empty DataFrame cleanly, never crashes.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score
from scipy import stats


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _safe_zscore(series: pd.Series) -> pd.Series:
    """
    Z-score that returns 0.0 for every element when std == 0
    (avoids NaN when all values are identical, e.g. after aggressive filtering).
    """
    arr = series.fillna(0).values.astype(float)
    std = arr.std()
    if std == 0 or np.isnan(std):
        return pd.Series(np.zeros(len(arr)), index=series.index)
    return pd.Series(np.abs((arr - arr.mean()) / std), index=series.index)


def _safe_minmax(series: pd.Series) -> np.ndarray:
    """
    Min-max normalise to [0, 1].
    Returns 0.5 for all elements when min == max (no variation).
    """
    arr = series.fillna(series.median()).values.astype(float)
    lo, hi = arr.min(), arr.max()
    if hi == lo:
        return np.full(len(arr), 0.5)
    return (arr - lo) / (hi - lo)


# ─────────────────────────────────────────────────────────────────────────────
# 1. COMPOSITE SCORE  (single source of truth — NOT in data.py)
# ─────────────────────────────────────────────────────────────────────────────

def compute_composite_score(
    df: pd.DataFrame,
    acc_w:   float = 0.40,
    cost_w:  float = 0.25,
    lat_w:   float = 0.20,
    speed_w: float = 0.15,
) -> pd.DataFrame:
    """
    Weighted composite score 0–100.  Higher = better.
    Uses _safe_minmax so single-row / identical-value DataFrames never crash.
    Weights must sum to 1.0.
    """
    if abs(acc_w + cost_w + lat_w + speed_w - 1.0) > 1e-6:
        raise ValueError("Weights must sum to 1.0")

    df = df.copy()

    norm_acc  = _safe_minmax(df["accuracy"])
    norm_cost = 1 - _safe_minmax(df["cost_usd_1m"])   # lower cost → higher score
    norm_lat  = 1 - _safe_minmax(df["latency_s"])      # lower latency → higher score
    norm_spd  = _safe_minmax(df["speed_tok_s"])

    df["composite_score"] = (
        acc_w  * norm_acc  +
        cost_w * norm_cost +
        lat_w  * norm_lat  +
        speed_w * norm_spd
    ) * 100

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. CLUSTERING
# ─────────────────────────────────────────────────────────────────────────────

def perform_clustering(df: pd.DataFrame, n_clusters: int = 3):
    """
    KMeans on [accuracy, cost_usd_1m, latency_s, speed_tok_s].

    Returns (df_with_cluster_cols, kmeans_model, scaler).
    Cluster labels: High-Performance | Budget | Balanced.

    Safe when n_clusters > len(df) — clamps automatically.
    """
    df = df.copy()
    FEATURES = ["accuracy", "cost_usd_1m", "latency_s", "speed_tok_s"]

    feat_df = df[FEATURES].copy()
    # Fill NaN per column with its median
    for col in FEATURES:
        feat_df[col] = feat_df[col].fillna(feat_df[col].median())

    n = max(1, min(n_clusters, len(df)))   # clamp: 1 ≤ n ≤ len(df)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(feat_df)

    kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(scaled)

    # ── Semantic labels ───────────────────────────────────────────────────────
    stats_df = df.groupby("cluster")[["accuracy", "cost_usd_1m"]].mean()
    acc_med  = stats_df["accuracy"].median()
    cost_med = stats_df["cost_usd_1m"].median()

    labels: dict = {}
    for cid, row in stats_df.iterrows():
        if row["accuracy"] >= acc_med:
            labels[cid] = "High-Performance"
        elif row["cost_usd_1m"] <= cost_med:
            labels[cid] = "Budget"
        else:
            labels[cid] = "Balanced"

    df["cluster_label"] = df["cluster"].map(labels)

    return df, kmeans, scaler


# ─────────────────────────────────────────────────────────────────────────────
# 3. REGRESSION — Predict Cost
# ─────────────────────────────────────────────────────────────────────────────

def train_cost_model(df: pd.DataFrame):
    """
    LinearRegression: predict cost_usd_1m from accuracy, speed_tok_s, latency_s.

    Returns (model, r2_score, feature_names).
    Returns (None, None, feature_names) if fewer than 5 usable rows.
    """
    FEATURES = ["accuracy", "speed_tok_s", "latency_s"]
    reg_df = df[FEATURES + ["cost_usd_1m"]].dropna()

    if len(reg_df) < 5:
        return None, None, FEATURES

    X = reg_df[FEATURES].values
    y = reg_df["cost_usd_1m"].values

    model = LinearRegression()
    model.fit(X, y)
    r2 = r2_score(y, model.predict(X))

    return model, round(float(r2), 4), FEATURES


def predict_cost(model, accuracy: float, speed: float, latency: float) -> float:
    """
    Predict cost using a trained LinearRegression model.
    Returns np.nan if model is None.
    """
    if model is None:
        return np.nan
    pred = model.predict([[accuracy, speed, latency]])[0]
    return max(0.0, round(float(pred), 6))


# ─────────────────────────────────────────────────────────────────────────────
# 4. OUTLIER DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag outliers using Z-score (|z| > 2.5) on cost & latency.

    FIX: uses _safe_zscore() so std=0 (all-same values after filtering)
    returns 0 instead of NaN — no crash.

    Adds columns: cost_outlier, latency_outlier, is_outlier.
    """
    df = df.copy()

    df["cost_z"]    = _safe_zscore(df["cost_usd_1m"])
    df["latency_z"] = _safe_zscore(df["latency_s"])

    df["cost_outlier"]    = df["cost_z"]    > 2.5
    df["latency_outlier"] = df["latency_z"] > 2.5
    df["is_outlier"]      = df["cost_outlier"] | df["latency_outlier"]

    return df.drop(columns=["cost_z", "latency_z"])


# ─────────────────────────────────────────────────────────────────────────────
# 5. PROVIDER ANALYTICS
# ─────────────────────────────────────────────────────────────────────────────

def provider_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate stats per provider.
    Handles missing composite_score gracefully (uses 0 if absent).
    """
    if df.empty:
        return pd.DataFrame(columns=[
            "provider", "model_count", "avg_accuracy",
            "avg_cost", "avg_latency", "avg_speed", "avg_composite",
        ])

    # Ensure composite_score exists
    if "composite_score" not in df.columns:
        df = compute_composite_score(df)

    grp = df.groupby("provider", dropna=False).agg(
        model_count   = ("model_name",       "count"),
        avg_accuracy  = ("accuracy",          "mean"),
        avg_cost      = ("cost_usd_1m",       "mean"),
        avg_latency   = ("latency_s",          "mean"),
        avg_speed     = ("speed_tok_s",        "mean"),
        avg_composite = ("composite_score",    "mean"),
    ).reset_index()

    for col, decimals in [
        ("avg_accuracy", 2), ("avg_cost", 4),
        ("avg_latency", 3),  ("avg_speed", 1), ("avg_composite", 2),
    ]:
        grp[col] = grp[col].round(decimals)

    return grp.sort_values("avg_composite", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 6. RECOMMENDATION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def recommend_models(
    df: pd.DataFrame,
    budget: float,
    min_accuracy: float,
    top_n: int = 5,
) -> pd.DataFrame:
    """
    Filter by budget (cost_usd_1m ≤ budget) and accuracy (≥ min_accuracy),
    then rank by composite_score.

    Returns top_n rows, or an empty DataFrame with the expected columns
    if nothing matches — never crashes.
    """
    EXPECTED_COLS = ["model_name", "provider", "accuracy",
                     "cost_usd_1m", "latency_s", "speed_tok_s",
                     "composite_score", "cluster_label"]

    if df.empty:
        return pd.DataFrame(columns=EXPECTED_COLS)

    # Ensure composite_score present
    if "composite_score" not in df.columns:
        df = compute_composite_score(df)

    mask = (
        (df["cost_usd_1m"] <= budget) &
        (df["accuracy"]    >= min_accuracy)
    )
    filtered = df[mask].copy()

    if filtered.empty:
        return pd.DataFrame(columns=EXPECTED_COLS)

    return (
        filtered
        .sort_values("composite_score", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )