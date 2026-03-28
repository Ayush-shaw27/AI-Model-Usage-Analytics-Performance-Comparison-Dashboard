"""
generate_dataset.py
Generates a realistic synthetic dataset for AI Model Usage Analytics.
Run this once to produce ai_model_data.csv in the same directory.
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 600

# ── Model characteristics ─────────────────────────────────────────────────────
MODEL_PARAMS = {
    "GPT-4":            {"accuracy": (93, 2.5), "cost": (0.0450, 0.006), "latency": (2900, 380)},
    "GPT-3.5-Turbo":    {"accuracy": (83, 3.5), "cost": (0.0080, 0.002), "latency": (1150, 190)},
    "Claude-3-Opus":    {"accuracy": (92, 2.5), "cost": (0.0400, 0.006), "latency": (2700, 350)},
    "Claude-3-Sonnet":  {"accuracy": (87, 3.0), "cost": (0.0180, 0.003), "latency": (1500, 230)},
    "Gemini-Ultra":     {"accuracy": (91, 2.8), "cost": (0.0380, 0.005), "latency": (2500, 310)},
    "Gemini-Pro":       {"accuracy": (84, 3.5), "cost": (0.0120, 0.002), "latency": (1300, 200)},
    "Llama-3-70B":      {"accuracy": (81, 4.5), "cost": (0.0040, 0.001), "latency": (920,  150)},
    "Mistral-7B":       {"accuracy": (76, 5.0), "cost": (0.0020, 0.0005),"latency": (680,  110)},
}

REGIONS = ["North America", "Europe", "Asia-Pacific", "South America", "Middle East"]
REGION_WEIGHTS = [0.35, 0.25, 0.25, 0.10, 0.05]

USE_CASES = ["Chatbot", "Code Generation", "Summarization", "Translation",
             "Question Answering", "Content Creation", "Data Analysis"]

rows = []
for _ in range(N):
    model  = np.random.choice(list(MODEL_PARAMS.keys()))
    region = np.random.choice(REGIONS, p=REGION_WEIGHTS)
    mp     = MODEL_PARAMS[model]

    # Users — log-normal so we get a realistic heavy tail
    users = int(np.clip(np.random.lognormal(mean=6.8, sigma=1.3), 80, 60_000))

    # Requests linearly correlated with users + noise
    requests_per_day = int(np.clip(
        users * np.random.uniform(4, 18) + np.random.normal(0, 150),
        200, 500_000
    ))

    accuracy   = round(np.clip(np.random.normal(*mp["accuracy"]), 60, 99.5), 2)
    cost       = round(np.clip(np.random.normal(*mp["cost"]),  0.0005, 0.12), 5)
    latency    = round(np.clip(np.random.normal(*mp["latency"]), 300, 8000),  1)
    uptime     = round(np.clip(np.random.normal(99.15, 0.55), 94.0, 100.0),  2)
    throughput = round(np.clip(np.random.normal(1000 / latency * 1000, 50), 50, 2000), 1)
    use_case   = np.random.choice(USE_CASES)

    rows.append({
        "Model":            model,
        "Region":           region,
        "Use_Case":         use_case,
        "Users":            users,
        "Requests_per_day": requests_per_day,
        "Accuracy_pct":     accuracy,
        "Cost_per_1K":      cost,
        "Latency_ms":       latency,
        "Uptime_pct":       uptime,
        "Throughput_rps":   throughput,
    })

df = pd.DataFrame(rows)

# Derived features
df["Efficiency_Score"] = (df["Accuracy_pct"] / df["Cost_per_1K"]).round(2)
df["Value_Index"]      = ((df["Accuracy_pct"] * 0.5) +
                          ((1 / df["Cost_per_1K"]) * 0.3) +
                          ((1 / df["Latency_ms"]) * 1000 * 0.2)).round(4)

df.to_csv("ai_model_data.csv", index=False)
print(f"✅  Dataset saved: {len(df)} rows × {len(df.columns)} columns")
print(df.head())
