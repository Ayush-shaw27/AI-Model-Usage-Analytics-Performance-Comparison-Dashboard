from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "ai_model_data.csv"

MODEL_ORDER = [
    "GPT-4",
    "Claude-3-Opus",
    "Gemini-Ultra",
    "Claude-3-Sonnet",
    "Gemini-Pro",
    "GPT-3.5-Turbo",
    "Llama-3-70B",
    "Mistral-7B",
]

MODEL_COLORS = {
    "GPT-4": "#4A90D9",
    "Claude-3-Opus": "#E8854A",
    "Gemini-Ultra": "#9B6DD4",
    "Claude-3-Sonnet": "#F0B860",
    "Gemini-Pro": "#7B52C8",
    "GPT-3.5-Turbo": "#48B8D8",
    "Llama-3-70B": "#5CB85C",
    "Mistral-7B": "#999999",
}

USE_CASES = [
    "Chatbot",
    "Code Generation",
    "Content Creation",
    "Data Analysis",
    "Translation",
    "Summarization",
    "Question Answering",
]

REGIONS = [
    "North America",
    "Europe",
    "Asia-Pacific",
    "Latin America",
    "Middle East",
]

MODEL_PROFILES = {
    "GPT-4":            {"accuracy": (92.88, 1.35), "cost": (0.0455, 0.0045), "latency": (2914, 160), "throughput": (358, 35),  "uptime": (99.13, 0.04)},
    "Claude-3-Opus":    {"accuracy": (92.15, 1.30), "cost": (0.0399, 0.0042), "latency": (2791, 150), "throughput": (376, 32),  "uptime": (99.13, 0.04)},
    "Gemini-Ultra":     {"accuracy": (90.37, 1.45), "cost": (0.0370, 0.0040), "latency": (2517, 140), "throughput": (398, 34),  "uptime": (99.15, 0.04)},
    "Claude-3-Sonnet":  {"accuracy": (87.63, 1.65), "cost": (0.0180, 0.0018), "latency": (1539, 95),  "throughput": (673, 45),  "uptime": (99.14, 0.05)},
    "Gemini-Pro":       {"accuracy": (83.66, 1.90), "cost": (0.0123, 0.0013), "latency": (1275, 90),  "throughput": (818, 55),  "uptime": (99.16, 0.05)},
    "GPT-3.5-Turbo":    {"accuracy": (82.96, 2.00), "cost": (0.0080, 0.0008), "latency": (1181, 85),  "throughput": (875, 60),  "uptime": (99.13, 0.05)},
    "Llama-3-70B":      {"accuracy": (81.12, 2.20), "cost": (0.0041, 0.0005), "latency": (912, 70),    "throughput": (1134, 70), "uptime": (99.08, 0.06)},
    "Mistral-7B":       {"accuracy": (74.92, 2.60), "cost": (0.0020, 0.0003), "latency": (688, 60),    "throughput": (1476, 85), "uptime": (99.04, 0.06)},
}

MODEL_PROB = np.array([0.18, 0.16, 0.12, 0.12, 0.12, 0.12, 0.10, 0.08])
MODEL_PROB = MODEL_PROB / MODEL_PROB.sum()

USE_CASE_PROB = np.array([0.19, 0.14, 0.14, 0.16, 0.13, 0.12, 0.12])
USE_CASE_PROB = USE_CASE_PROB / USE_CASE_PROB.sum()

REGION_PROB = np.array([0.34, 0.23, 0.24, 0.11, 0.08])
REGION_PROB = REGION_PROB / REGION_PROB.sum()


def _clip(value: float, lower: float, upper: float) -> float:
    return float(np.clip(value, lower, upper))


def _coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def generate_sample_data(n_rows: int = 600, seed: int = 42) -> pd.DataFrame:
    """Generate a realistic AI model usage dataset if one is not already present."""
    rng = np.random.default_rng(seed)
    rows = []

    use_case_accuracy_boost = {
        "Chatbot": 0.0,
        "Code Generation": 2.0,
        "Content Creation": 1.0,
        "Data Analysis": 1.2,
        "Translation": 0.2,
        "Summarization": 0.0,
        "Question Answering": 1.0,
    }

    region_latency_penalty = {
        "North America": 0,
        "Europe": 12,
        "Asia-Pacific": 28,
        "Latin America": 42,
        "Middle East": 35,
    }

    for _ in range(n_rows):
        model = rng.choice(MODEL_ORDER, p=MODEL_PROB)
        use_case = rng.choice(USE_CASES, p=USE_CASE_PROB)
        region = rng.choice(REGIONS, p=REGION_PROB)
        profile = MODEL_PROFILES[model]

        users = int(np.clip(rng.lognormal(mean=6.8, sigma=1.2), 80, 60000))
        requests_per_day = int(np.clip(users * rng.uniform(4, 18) + rng.normal(0, 150), 200, 500000))

        accuracy = rng.normal(*profile["accuracy"]) + use_case_accuracy_boost[use_case]
        cost = rng.normal(*profile["cost"])
        latency = rng.normal(*profile["latency"]) + region_latency_penalty[region]
        throughput = rng.normal(*profile["throughput"])
        uptime = rng.normal(*profile["uptime"])

        rows.append(
            {
                "Model": model,
                "Use_Case": use_case,
                "Region": region,
                "Users": users,
                "Requests_per_day": requests_per_day,
                "Accuracy_pct": round(_clip(accuracy, 60, 99.5), 2),
                "Cost_per_1K": round(_clip(cost, 0.0005, 0.12), 5),
                "Latency_ms": round(_clip(latency, 300, 8000), 1),
                "Uptime_pct": round(_clip(uptime, 94.0, 100.0), 2),
                "Throughput_rps": round(_clip(throughput, 50, 2000), 1),
            }
        )

    df = pd.DataFrame(rows)
    df["Requests_per_day"] = df["Requests_per_day"].astype(int)
    df["Users"] = df["Users"].astype(int)
    df.to_csv(DATA_PATH, index=False)
    return df


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = {}
    for col in df.columns:
        key = (
            str(col)
            .strip()
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
            .replace("/", "_")
        )
        normalized[col] = key

    df = df.rename(columns=normalized)

    aliases = {
        "model": "Model",
        "ai_model": "Model",
        "use_case": "Use_Case",
        "usecase": "Use_Case",
        "region": "Region",
        "accuracy_pct": "Accuracy_pct",
        "accuracy": "Accuracy_pct",
        "cost_per_1k": "Cost_per_1K",
        "cost_per_1k_tokens": "Cost_per_1K",
        "cost": "Cost_per_1K",
        "latency_ms": "Latency_ms",
        "latency": "Latency_ms",
        "uptime_pct": "Uptime_pct",
        "uptime": "Uptime_pct",
        "throughput_rps": "Throughput_rps",
        "throughput": "Throughput_rps",
        "requests_per_day": "Requests_per_day",
        "requests": "Requests_per_day",
        "users": "Users",
    }

    rename_map = {}
    for col in df.columns:
        rename_map[col] = aliases.get(col, col)

    df = df.rename(columns=rename_map)
    return df


def load_data() -> pd.DataFrame:
    """Load dataset and make sure the expected dashboard columns exist."""
    if not DATA_PATH.exists():
        df = generate_sample_data()
    else:
        df = pd.read_csv(DATA_PATH)

    df = _normalize_columns(df)

    required = [
        "Model",
        "Use_Case",
        "Region",
        "Accuracy_pct",
        "Cost_per_1K",
        "Latency_ms",
        "Requests_per_day",
        "Uptime_pct",
        "Throughput_rps",
        "Users",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns in ai_model_data.csv: "
            + ", ".join(missing)
        )

    df = df[required].copy()

    numeric_cols = [
        "Accuracy_pct",
        "Cost_per_1K",
        "Latency_ms",
        "Requests_per_day",
        "Uptime_pct",
        "Throughput_rps",
        "Users",
    ]
    df = _coerce_numeric(df, numeric_cols)
    df = df.dropna(subset=required).drop_duplicates().reset_index(drop=True)

    df["Requests_per_day"] = df["Requests_per_day"].astype(int)
    df["Users"] = df["Users"].astype(int)

    return df


def _minmax_high(series: pd.Series) -> pd.Series:
    denom = series.max() - series.min()
    if denom == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - series.min()) / (denom + 1e-9) * 100


def _minmax_low(series: pd.Series) -> pd.Series:
    denom = series.max() - series.min()
    if denom == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series.max() - series) / (denom + 1e-9) * 100


def aggregate_model_data(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics by model and compute the dashboard value score."""
    agg = (
        df.groupby("Model", as_index=False)
        .agg(
            Accuracy=("Accuracy_pct", "mean"),
            Cost=("Cost_per_1K", "mean"),
            Latency=("Latency_ms", "mean"),
            Uptime=("Uptime_pct", "mean"),
            Throughput=("Throughput_rps", "mean"),
            Requests=("Requests_per_day", "sum"),
            Users=("Users", "sum"),
        )
    )

    agg["Score"] = (
        _minmax_high(agg["Accuracy"]) * 0.35
        + _minmax_low(agg["Cost"]) * 0.25
        + _minmax_low(agg["Latency"]) * 0.20
        + _minmax_high(agg["Uptime"]) * 0.10
        + _minmax_high(agg["Throughput"]) * 0.10
    ).round(1)

    return agg.round({"Accuracy": 1, "Cost": 4, "Latency": 0, "Uptime": 2, "Throughput": 0})


def get_use_case_data(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["Use_Case", "Model"], as_index=False)["Requests_per_day"]
        .mean()
    )


def get_region_data(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["Region", "Model"], as_index=False)["Requests_per_day"]
        .mean()
    )


def get_winners_table(use_case_df: pd.DataFrame) -> pd.DataFrame:
    if use_case_df.empty:
        return pd.DataFrame(columns=["Use Case", "Best Model", "Avg Daily Requests"])
    winners = use_case_df.loc[
        use_case_df.groupby("Use_Case")["Requests_per_day"].idxmax()
    ].copy()
    winners = winners.rename(
        columns={
            "Use_Case": "Use Case",
            "Model": "Best Model",
            "Requests_per_day": "Avg Daily Requests",
        }
    )
    winners["Avg Daily Requests"] = winners["Avg Daily Requests"].round(0).astype(int)
    return winners[["Use Case", "Best Model", "Avg Daily Requests"]]


def get_dashboard_data() -> dict:
    df = load_data()
    agg = aggregate_model_data(df)
    use_case = get_use_case_data(df)
    region = get_region_data(df)
    return {
        "raw": df,
        "models": agg,
        "use_cases": use_case,
        "regions": region,
        "winners": get_winners_table(use_case),
    }


def model_picker_score(
    df: pd.DataFrame,
    weight_accuracy: int,
    weight_cost: int,
    weight_speed: int,
    weight_reliability: int,
    weight_throughput: int,
) -> pd.DataFrame:
    total = weight_accuracy + weight_cost + weight_speed + weight_reliability + weight_throughput
    if total <= 0:
        df = df.copy()
        df["custom_score"] = 0.0
        return df

    scored = df.copy()
    scored["custom_score"] = (
        _minmax_high(scored["Accuracy"]) * (weight_accuracy / total)
        + _minmax_low(scored["Cost"]) * (weight_cost / total)
        + _minmax_low(scored["Latency"]) * (weight_speed / total)
        + _minmax_high(scored["Uptime"]) * (weight_reliability / total)
        + _minmax_high(scored["Throughput"]) * (weight_throughput / total)
    ) * 100
    return scored.round({"custom_score": 1})


if __name__ == "__main__":
    df = generate_sample_data()
    print(f"Generated {len(df)} rows at {DATA_PATH}")