import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, mean_squared_error

# Vercel specific configuration
st.set_page_config(
    layout="wide", 
    page_title="AI Model Analytics Dashboard", 
    page_icon="🤖",
    initial_sidebar_state="expanded"
)

# ---------------- CHECK DATA AVAILABILITY ----------------
def check_data_files():
    """Check if data files exist, if not create sample data"""
    required_files = [
        "ai_model_data.csv",
        "ai_models_performance.csv", 
        "models.xlsx",
        "open_llm_leaderboard_train.csv"
    ]
    
    existing_files = [f for f in required_files if os.path.exists(f)]
    
    if len(existing_files) >= 2:
        return True
    return False

# ---------------- CREATE SAMPLE DATA FOR DEPLOYMENT ----------------
def create_sample_data():
    """Create comprehensive sample data for demo"""
    st.info("📊 Using sample data for demonstration (data files not found)")
    
    data = {
        "model": [
            "Claude 3.5 Opus", "GPT-4o", "Gemini Ultra 2.0", "Claude 3.5 Sonnet",
            "GPT-4 Turbo", "DeepSeek-V3", "LLaMA-4", "Mistral Large 2",
            "Gemini Pro 2.0", "Qwen 2.5 Max", "Claude 3 Haiku", "GPT-4o Mini",
            "LLaMA-3.3 70B", "Mistral Medium", "Phi-3 Mini", "Gemma 2 9B",
            "Falcon 180B", "Cohere Command R+", "Yi 34B", "OpenELM 3B"
        ],
        "accuracy": [97.2, 96.5, 96.8, 95.8, 95.2, 95.5, 94.2, 94.8, 
                     94.5, 94.3, 92.5, 91.8, 90.5, 89.5, 81.5, 80.2, 
                     88.5, 89.2, 79.5, 78.5],
        "latency": [180, 160, 190, 140, 150, 140, 130, 125, 
                    110, 115, 95, 85, 120, 110, 40, 48,
                    180, 130, 52, 35],
        "cost": [0.030, 0.025, 0.028, 0.015, 0.020, 0.009, 0.008, 0.010,
                 0.012, 0.007, 0.005, 0.003, 0.006, 0.005, 0.0005, 0.0006,
                 0.012, 0.007, 0.0009, 0.0004],
        "leaderboard_score": [96.8, 95.9, 96.2, 95.2, 94.8, 94.9, 93.5, 94.2,
                              93.8, 93.6, 91.8, 91.2, 89.8, 88.9, 80.2, 79.5,
                              87.9, 88.5, 78.8, 77.5],
        "provider": ["Anthropic", "OpenAI", "Google", "Anthropic", "OpenAI", 
                     "DeepSeek", "Meta", "Mistral AI", "Google", "Alibaba",
                     "Anthropic", "OpenAI", "Meta", "Mistral AI", "Microsoft",
                     "Google", "TII", "Cohere", "01.AI", "Apple"]
    }
    
    df = pd.DataFrame(data)
    df["efficiency"] = df["accuracy"] / df["cost"]
    
    # Add tiers
    conditions = [
        df['accuracy'] >= 95,
        df['accuracy'] >= 90,
        df['accuracy'] >= 85,
        df['accuracy'] >= 80
    ]
    choices = ['🚀 Elite', '⭐ Premium', '💪 Standard', '🎯 Budget']
    df['tier'] = np.select(conditions, choices, default='📱 Entry')
    
    return df

# ---------------- LOAD DATA ----------------
@st.cache_data(ttl=3600)
def load_data():
    try:
        # Try to load from data/final_dataset.csv first
        if os.path.exists("data/final_dataset.csv"):
            df = pd.read_csv("data/final_dataset.csv")
            if not df.empty:
                return df
        
        # If not found, check if source files exist and process them
        if check_data_files():
            # Import data processing function (if available)
            try:
                from data import process_all_data
                df = process_all_data()
                return df
            except:
                pass
        
        # Fallback to sample data
        return create_sample_data()
        
    except Exception as e:
        st.warning(f"Error loading data: {e}")
        return create_sample_data()

# Load data
df = load_data()

# ---------------- UI STYLE ----------------
st.markdown("""
<style>
.main-title {
    font-size: 42px;
    font-weight: bold;
    background: linear-gradient(135deg, #00C6FF, #0072FF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 20px;
}
.card {
    background: linear-gradient(135deg, #1f1c2c, #2a2a3a);
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    color: white;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
}
.metric-title {
    font-size: 14px;
    color: #aaa;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.metric-value {
    font-size: 32px;
    font-weight: bold;
    margin-top: 10px;
}
.stButton > button {
    background: linear-gradient(135deg, #00C6FF, #0072FF);
    color: white;
    border: none;
    padding: 10px 24px;
    border-radius: 8px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ Control Panel")
st.sidebar.markdown("---")

# Model selection
st.sidebar.subheader("📊 Model Selection")
all_models = df["model"].unique()
selected_models = st.sidebar.multiselect(
    "Select Models to Display",
    all_models,
    default=all_models[:min(10, len(all_models))],
    help="Choose which models to analyze"
)

if selected_models:
    df_filtered = df[df["model"].isin(selected_models)].copy()
else:
    df_filtered = df.copy()

st.sidebar.markdown("---")
st.sidebar.caption("🚀 Deployed on Vercel")

# ---------------- TITLE ----------------
st.markdown('<p class="main-title">🤖 AI Model Analytics Dashboard</p>', unsafe_allow_html=True)

# ---------------- KPI CARDS ----------------
col1, col2, col3, col4 = st.columns(4)

def card(title, value, color="#00C6FF"):
    return f"""
    <div class="card">
        <div class="metric-title">{title}</div>
        <div class="metric-value" style="color: {color};">{value}</div>
    </div>
    """

with col1:
    st.markdown(card("Total Models", len(df_filtered)), unsafe_allow_html=True)
with col2:
    st.markdown(card("Avg Accuracy", f"{df_filtered['accuracy'].mean():.1f}%", "#00FF88"), unsafe_allow_html=True)
with col3:
    best_model = df_filtered.loc[df_filtered["accuracy"].idxmax(), "model"]
    st.markdown(card("Best Model", best_model, "#FFD700"), unsafe_allow_html=True)
with col4:
    efficient_model = df_filtered.loc[df_filtered["efficiency"].idxmax(), "model"]
    st.markdown(card("Most Efficient", efficient_model, "#FF6B6B"), unsafe_allow_html=True)

st.markdown("---")

# ---------------- TABS ----------------
tab1, tab2, tab3, tab4 = st.tabs(["📊 Performance", "🧠 ML Model", "🔍 Clustering", "💡 Insights"])

# ---------------- TAB 1: PERFORMANCE ----------------
with tab1:
    st.subheader("📊 Model Performance Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_scatter = px.scatter(
            df_filtered,
            x="cost",
            y="accuracy",
            color="efficiency",
            size="efficiency",
            hover_name="model",
            template="plotly_dark",
            title="Accuracy vs Cost (by Efficiency)",
            color_continuous_scale="Viridis"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with col2:
        top_models = df_filtered.nlargest(10, "accuracy")[["model", "accuracy", "cost"]]
        fig_bar = px.bar(
            top_models,
            x="model",
            y="accuracy",
            color="cost",
            template="plotly_dark",
            title="Top 10 Models by Accuracy",
            color_continuous_scale="Plasma"
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Performance table
    st.subheader("Performance Metrics")
    performance_df = df_filtered[["model", "accuracy", "latency", "cost", "efficiency"]].round(2)
    performance_df = performance_df.sort_values("accuracy", ascending=False)
    st.dataframe(performance_df, use_container_width=True)

# ---------------- TAB 2: ML MODEL ----------------
with tab2:
    st.subheader("🧠 Random Forest Regression Model")
    
    if len(df_filtered) > 5:
        X = df_filtered[["cost", "latency"]]
        if "leaderboard_score" in df_filtered.columns:
            X["leaderboard_score"] = df_filtered["leaderboard_score"]
        
        y = df_filtered["accuracy"]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("R² Score", f"{r2_score(y_test, y_pred):.3f}")
        with col2:
            st.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
        
        # Feature importance
        importance_df = pd.DataFrame({
            "Feature": X.columns,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=True)
        
        fig_importance = px.bar(
            importance_df,
            x="Importance",
            y="Feature",
            orientation='h',
            template="plotly_dark",
            title="Feature Importance"
        )
        st.plotly_chart(fig_importance, use_container_width=True)
        
        # Actual vs Predicted
        comparison_df = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred})
        fig_comparison = px.scatter(comparison_df, x="Actual", y="Predicted", template="plotly_dark")
        fig_comparison.add_trace(go.Scatter(
            x=[comparison_df["Actual"].min(), comparison_df["Actual"].max()],
            y=[comparison_df["Actual"].min(), comparison_df["Actual"].max()],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash')
        ))
        st.plotly_chart(fig_comparison, use_container_width=True)
    else:
        st.warning("Not enough data for ML model")

# ---------------- TAB 3: CLUSTERING ----------------
with tab3:
    st.subheader("🔍 Model Clustering Analysis")
    
    if len(df_filtered) >= 4:
        cluster_data = df_filtered[["accuracy", "latency", "cost"]].copy()
        n_clusters = min(3, len(cluster_data))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(cluster_data)
        
        df_clustered = df_filtered.copy()
        df_clustered["Cluster"] = clusters.astype(str)
        
        fig_cluster = px.scatter_3d(
            df_clustered,
            x="accuracy",
            y="latency",
            z="cost",
            color="Cluster",
            hover_name="model",
            template="plotly_dark",
            title=f"Model Clusters ({n_clusters} Clusters)"
        )
        st.plotly_chart(fig_cluster, use_container_width=True)
        
        # Cluster summary
        cluster_summary = df_clustered.groupby("Cluster").agg({
            "model": "count",
            "accuracy": "mean",
            "latency": "mean",
            "cost": "mean"
        }).round(2)
        cluster_summary.columns = ["Count", "Avg Accuracy", "Avg Latency", "Avg Cost"]
        st.dataframe(cluster_summary, use_container_width=True)
    else:
        st.warning("Not enough data for clustering")

# ---------------- TAB 4: INSIGHTS ----------------
with tab4:
    st.subheader("💡 Key Insights & Recommendations")
    
    # Top performers
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🏆 Top Performers")
        top_accuracy = df_filtered.nlargest(5, "accuracy")[["model", "accuracy", "cost"]]
        st.dataframe(top_accuracy, use_container_width=True)
    
    with col2:
        st.markdown("### 💰 Most Affordable")
        top_affordable = df_filtered.nsmallest(5, "cost")[["model", "cost", "accuracy"]]
        st.dataframe(top_affordable, use_container_width=True)
    
    # Recommendations
    st.markdown("### 🤖 Smart Recommendation")
    
    choice = st.selectbox(
        "Select Your Priority",
        ["🎯 High Accuracy", "💰 Low Cost", "⚖️ Balanced (Best Efficiency)"]
    )
    
    if choice == "🎯 High Accuracy":
        recommended = df_filtered.loc[df_filtered["accuracy"].idxmax()]
    elif choice == "💰 Low Cost":
        recommended = df_filtered.loc[df_filtered["cost"].idxmin()]
    else:
        recommended = df_filtered.loc[df_filtered["efficiency"].idxmax()]
    
    st.success(f"""
    ### ✅ Recommended Model: **{recommended['model']}**
    
    **Metrics:**
    - 🎯 Accuracy: {recommended['accuracy']:.1f}%
    - ⏱️ Latency: {recommended['latency']:.0f}ms
    - 💰 Cost: ${recommended['cost']:.4f}
    - ⚡ Efficiency: {recommended['efficiency']:.1f}
    """)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; padding: 20px;'>"
    "🚀 AI Model Analytics Dashboard | Powered by Streamlit & Deployed on Vercel"
    "</div>",
    unsafe_allow_html=True
)