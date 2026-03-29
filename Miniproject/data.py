"""
data.py  —  AI Model Analytics & Decision Intelligence Platform
Data Pipeline Layer

FIXES applied
─────────────
1. Column-name mapping instead of brittle positional iloc indices.
2. composite_score removed — ml_models.py is the single source of truth.
3. Cache validation checks columns AND row count, not just file existence.
4. Graceful fallback when ai_models_performance.csv is missing → try ai_models_perform.csv.
5. inf / NaN purge before saving.
"""

import os
import re
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
_HERE       = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(_HERE, "data")
OUTPUT_CSV  = os.path.join(_HERE, "final_dataset.csv")

# Required columns that must exist in the cached CSV to be considered valid
_REQUIRED_COLS = {"model_name", "provider", "cost_usd_1m", "speed_tok_s",
                  "latency_s", "accuracy", "cost_efficiency", "speed_efficiency"}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _clean_dollar(val) -> float:
    """'$4.81' → 4.81  |  handles NaN / non-numeric gracefully."""
    if pd.isna(val):
        return np.nan
    s = str(val).replace("$", "").replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return np.nan


def _clean_int_prefix(val):
    """'41\\nE' → 41.0  |  strips trailing garbage after digits."""
    if pd.isna(val):
        return np.nan
    m = re.match(r"^(\d+(?:\.\d+)?)", str(val).strip())
    return float(m.group(1)) if m else np.nan


def _normalize_model_name(name: str) -> str:
    """Lowercase, strip HTML tags and excess whitespace for fuzzy merge key."""
    if not isinstance(name, str):
        return ""
    name = re.sub(r"<[^>]+>", "", name.lower().strip())
    return re.sub(r"\s+", " ", name)


def _cache_valid() -> bool:
    """Return True only if OUTPUT_CSV exists, is non-empty, and has all required cols."""
    if not os.path.exists(OUTPUT_CSV):
        return False
    try:
        header = pd.read_csv(OUTPUT_CSV, nrows=0)
        if not _REQUIRED_COLS.issubset(set(header.columns)):
            return False
        # Check it actually has rows
        test = pd.read_csv(OUTPUT_CSV, nrows=5)
        return len(test) > 0
    except Exception:
        return False


# ── Loaders ──────────────────────────────────────────────────────────────────

def _load_performance() -> pd.DataFrame:
    """
    Load ai_models_performance.csv (or ai_models_perform.csv as fallback).
    Uses COLUMN NAMES not positional indices — robust to column reordering.

    Expected columns (case-insensitive strip match):
        Model | Creator | Intelligence Index |
        Price (Blended USD/1M Tokens) | Speed(median token/s) |
        Latency (First Answer Chunk /s)
    """
    # Locate file
    candidates = [
        os.path.join(DATA_DIR, "ai_models_performance.csv"),
        os.path.join(DATA_DIR, "ai_models_perform.csv"),
        # also check alongside this file (when run from project root)
        os.path.join(_HERE, "ai_models_performance.csv"),
        os.path.join(_HERE, "ai_models_perform.csv"),
    ]
    path = next((p for p in candidates if os.path.exists(p)), None)
    if path is None:
        raise FileNotFoundError(
            "Could not find ai_models_performance.csv or ai_models_perform.csv. "
            f"Searched: {candidates}"
        )

    raw = pd.read_csv(path)
    raw.columns = raw.columns.str.strip()

    # ── Column name mapping (flexible) ──────────────────────────────────────
    COL_MAP = {
        "model_name":      ["Model", "model", "model_name", "Name"],
        "provider":        ["Creator", "creator", "Provider", "provider", "Company"],
        "intelligence_idx":["Intelligence Index", "intelligence_index",
                            "Intelligence_Index", "IQ", "Score"],
        "price_raw":       ["Price (Blended USD/1M Tokens)", "Price", "price",
                            "Cost (USD/1M)", "cost"],
        "speed_tok_s":     ["Speed(median token/s)", "Speed (tok/s)", "speed",
                            "Speed_tok_s", "Tokens/s"],
        "latency_s":       ["Latency (First Answer Chunk /s)", "Latency",
                            "latency", "Latency_s", "First Token Latency (s)"],
    }

    def pick_col(candidates):
        for c in candidates:
            if c in raw.columns:
                return raw[c]
        return pd.Series([np.nan] * len(raw))

    out = pd.DataFrame()
    out["model_name"]      = pick_col(COL_MAP["model_name"]).astype(str)
    out["provider"]        = pick_col(COL_MAP["provider"]).astype(str).str.strip()
    out["intelligence_idx"]= pick_col(COL_MAP["intelligence_idx"]).apply(_clean_int_prefix)
    out["cost_usd_1m"]     = pick_col(COL_MAP["price_raw"]).apply(_clean_dollar)
    out["speed_tok_s"]     = pd.to_numeric(pick_col(COL_MAP["speed_tok_s"]), errors="coerce")
    out["latency_s"]       = pd.to_numeric(pick_col(COL_MAP["latency_s"]),   errors="coerce")

    return out


def _load_llm_leaderboard() -> pd.DataFrame:
    """
    Load open_llm_leaderboard_train.csv.
    Uses COLUMN NAMES for robustness.

    Expected: fullname  |  Average ⬆️
    """
    candidates = [
        os.path.join(DATA_DIR, "open_llm_leaderboard_train.csv"),
        os.path.join(_HERE,    "open_llm_leaderboard_train.csv"),
    ]
    path = next((p for p in candidates if os.path.exists(p)), None)
    if path is None:
        return pd.DataFrame(columns=["model_name_llm", "avg_score"])

    raw = pd.read_csv(path, low_memory=False)
    raw.columns = raw.columns.str.strip()

    NAME_COLS  = ["fullname", "Full Name", "model_name", "Model", "name"]
    SCORE_COLS = ["Average ⬆️", "Average", "avg_score", "Score", "average"]

    def pick(candidates):
        for c in candidates:
            if c in raw.columns:
                return raw[c]
        return pd.Series([np.nan] * len(raw))

    out = pd.DataFrame()
    out["model_name_llm"] = pick(NAME_COLS).astype(str)
    out["avg_score"]      = pd.to_numeric(pick(SCORE_COLS), errors="coerce")

    # Strip HTML tags that appear in fullname
    out["model_name_llm"] = out["model_name_llm"].apply(
        lambda x: re.sub(r"<[^>]+>", "", x).strip()
    )

    return out.drop_duplicates(subset=["model_name_llm"])


# ── Public API ────────────────────────────────────────────────────────────────

def preprocess(force_rebuild: bool = False) -> pd.DataFrame:
    """
    Build (or load from cache) the cleaned, merged AI-model dataset.

    Steps
    ─────
    1. Load ai_models_performance.csv  → performance metrics
    2. Load open_llm_leaderboard_train.csv → benchmark accuracy scores
    3. Fuzzy-merge on normalised model name
    4. Clean & feature-engineer
    5. Save to final_dataset.csv

    NOTE: composite_score is NOT computed here.
          Call ml_models.compute_composite_score(df) after loading.
    """
    # ── Try cache ─────────────────────────────────────────────────────────────
    if not force_rebuild and _cache_valid():
        try:
            df = pd.read_csv(OUTPUT_CSV)
            # Validate required columns present
            if _REQUIRED_COLS.issubset(set(df.columns)) and len(df) > 0:
                return df
        except Exception:
            pass  # Fall through to rebuild

    # ── Load raw sources ──────────────────────────────────────────────────────
    df_perf = _load_performance()
    df_llm  = _load_llm_leaderboard()

    # ── Fuzzy merge on normalised name ────────────────────────────────────────
    df_perf["_key"] = df_perf["model_name"].apply(_normalize_model_name)
    df_llm["_key"]  = df_llm["model_name_llm"].apply(_normalize_model_name)

    df = df_perf.merge(df_llm[["_key", "avg_score"]], on="_key", how="left")
    df = df.drop(columns=["_key"], errors="ignore")

    # ── Drop rows missing core numeric columns ────────────────────────────────
    df = df.dropna(subset=["cost_usd_1m", "speed_tok_s", "latency_s"])

    # ── Accuracy: prefer leaderboard avg_score, fall back to intelligence_idx ─
    df["accuracy"] = df["avg_score"].combine_first(df["intelligence_idx"])
    # Fill remaining NaN with median
    acc_median = df["accuracy"].median()
    df["accuracy"] = df["accuracy"].fillna(acc_median if pd.notna(acc_median) else 0)

    # ── Derived features ──────────────────────────────────────────────────────
    df["cost_efficiency"]  = df["accuracy"] / (df["cost_usd_1m"]  + 1e-9)
    df["speed_efficiency"] = df["speed_tok_s"] / (df["latency_s"] + 1e-9)

    # ── Final purge of inf / NaN ──────────────────────────────────────────────
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=list(_REQUIRED_COLS))
    df = df.reset_index(drop=True)

    if df.empty:
        raise ValueError(
            "Dataset is empty after cleaning. "
            "Check that your CSV files are correctly formatted."
        )

    # ── Persist ───────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(OUTPUT_CSV) or ".", exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    return df


# ── Quick self-test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = preprocess(force_rebuild=True)
    print(f"Rows: {len(df)}  Cols: {list(df.columns)}")
    print(df.head(3).to_string())