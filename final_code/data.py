# data.py
import pandas as pd
import numpy as np
import os

print("🔄 Starting data processing...")

# ---------------- SAFE LOAD FUNCTION ----------------
def safe_load(file_path):
    try:
        if file_path.endswith(".csv"):
            return pd.read_csv(file_path)
        elif file_path.endswith(".xlsx"):
            return pd.read_excel(file_path)
        else:
            raise Exception(f"Unsupported file: {file_path}")
    except Exception as e:
        print(f"❌ Error loading {file_path}: {e}")
        return pd.DataFrame()

# ---------------- CLEAN COLUMN NAMES ----------------
def clean_columns(df):
    df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")
    return df

# ---------------- FIND COLUMN ----------------
def find_column(df, possible_names):
    for col in df.columns:
        for name in possible_names:
            if name in col:
                return col
    return None

# ---------------- PROCESS DATASET ----------------
def process_model_dataset(df):
    df = clean_columns(df)

    model_col = find_column(df, ["model", "name", "id"])
    acc_col = find_column(df, ["accuracy", "score"])
    lat_col = find_column(df, ["latency", "time"])
    cost_col = find_column(df, ["cost", "price"])

    selected = {}

    if model_col:
        selected["model"] = df[model_col]
    if acc_col:
        selected["accuracy"] = pd.to_numeric(df[acc_col], errors='coerce')
    if lat_col:
        selected["latency"] = pd.to_numeric(df[lat_col], errors='coerce')
    if cost_col:
        selected["cost"] = pd.to_numeric(df[cost_col], errors='coerce')

    return pd.DataFrame(selected)

# ---------------- LOAD ALL FILES ----------------
files = [
    "ai_model_data.csv",
    "ai_models_performance.csv",
    "models.xlsx",
    "open_llm_leaderboard_train.csv"
]

dfs = []

for file in files:
    if os.path.exists(file):
        print(f"📂 Loading {file}")
        df = safe_load(file)
        if not df.empty:
            dfs.append((file, df))
    else:
        print(f"⚠️ File not found: {file}")

# Check if we have any files loaded
if not dfs:
    print("❌ No data files found! Creating sample data for testing...")
    
    # Create sample data for testing
    sample_data = pd.DataFrame({
        "model": ["gpt-4", "claude-3", "gemini-pro", "llama-3", "mistral-large", 
                  "gpt-3.5-turbo", "claude-2", "palm-2", "falcon-40b", "bert-large"],
        "accuracy": [92.5, 90.3, 88.7, 86.2, 84.9, 82.1, 85.4, 83.6, 81.2, 79.8],
        "latency": [280, 265, 240, 210, 190, 150, 255, 230, 320, 180],
        "cost": [0.030, 0.025, 0.020, 0.015, 0.012, 0.008, 0.022, 0.018, 0.025, 0.010],
        "leaderboard_score": [91.8, 89.5, 87.9, 85.3, 83.7, 81.2, 84.8, 82.9, 80.5, 78.9]
    })
    
    dfs = [("sample_data", sample_data)]
    print("✅ Created sample data for testing")

# ---------------- PROCESS FIRST 3 DATASETS ----------------
processed_dfs = []

for name, df in dfs[:3]:
    processed = process_model_dataset(df)
    if not processed.empty:
        processed_dfs.append(processed)

# ---------------- HANDLE LEADERBOARD DATA ----------------
if len(dfs) >= 4:
    df4 = dfs[3][1]  # fourth dataset
else:
    df4 = dfs[0][1]  # use first dataset if no leaderboard

df4 = clean_columns(df4)

print("📊 Leaderboard columns:", df4.columns.tolist())

model_col = find_column(df4, ["model", "name", "id"])
score_col = find_column(df4, ["score", "accuracy", "average", "leaderboard_score"])

if not model_col or not score_col:
    print("⚠️ Could not detect leaderboard columns automatically")
    print("Using fallback numeric column")

    numeric_cols = df4.select_dtypes(include=np.number).columns
    if len(numeric_cols) > 0:
        score_col = numeric_cols[0]
        # Also try to find a model column
        for col in df4.columns:
            if 'model' in col.lower() or 'name' in col.lower():
                model_col = col
                break
        if not model_col:
            model_col = df4.columns[0]  # use first column as model
    else:
        raise Exception("❌ No usable columns in leaderboard dataset")

# Create a copy to avoid warnings
df4_processed = df4[[model_col, score_col]].copy()
df4_processed.rename(columns={
    model_col: "model",
    score_col: "leaderboard_score"
}, inplace=True)

# Convert leaderboard_score to numeric
df4_processed["leaderboard_score"] = pd.to_numeric(df4_processed["leaderboard_score"], errors='coerce')

# ---------------- COMBINE DATA ----------------
if processed_dfs:
    df_combined = pd.concat(processed_dfs, ignore_index=True)
else:
    # If no processed data, use the first dataframe
    df_combined = dfs[0][1].copy()
    if "model" in df_combined.columns:
        df_combined = df_combined[["model"]]
    else:
        df_combined = pd.DataFrame({"model": df_combined.iloc[:, 0]})

# Normalize model names
df_combined["model"] = df_combined["model"].astype(str).str.lower().str.strip()
df4_processed["model"] = df4_processed["model"].astype(str).str.lower().str.strip()

# Merge
final_df = pd.merge(df_combined, df4_processed, on="model", how="left")

# ---------------- CLEAN DATA ----------------
final_df.drop_duplicates(inplace=True)

# Convert all numeric columns to proper numeric types
numeric_cols = ["accuracy", "latency", "cost", "leaderboard_score"]
for col in numeric_cols:
    if col in final_df.columns:
        final_df[col] = pd.to_numeric(final_df[col], errors='coerce')

# Fill missing values with mean for numeric columns only
for col in numeric_cols:
    if col in final_df.columns:
        # Check if column has any numeric values before calculating mean
        if final_df[col].notna().any():
            mean_val = final_df[col].mean()
            final_df[col].fillna(mean_val, inplace=True)
        else:
            # If all values are NaN, fill with 0
            final_df[col].fillna(0, inplace=True)

# ---------------- FEATURE ENGINEERING ----------------
if "accuracy" in final_df.columns and "cost" in final_df.columns:
    final_df["efficiency"] = final_df["accuracy"] / (final_df["cost"] + 1e-6)

if "latency" in final_df.columns:
    final_df["latency_score"] = 1 / (final_df["latency"] + 1e-6)

# ---------------- REMOVE BAD ROWS ----------------
final_df = final_df.replace([np.inf, -np.inf], np.nan)
final_df.dropna(inplace=True)

# ---------------- SAVE ----------------
os.makedirs("data", exist_ok=True)
final_path = "data/final_dataset.csv"

final_df.to_csv(final_path, index=False)

print("✅ FINAL DATASET CREATED:", final_path)
print("📊 Shape:", final_df.shape)
print("\n📋 First 5 rows:")
print(final_df.head())
print("\n📊 Data types:")
print(final_df.dtypes)
print("\n✅ Data processing completed successfully!")