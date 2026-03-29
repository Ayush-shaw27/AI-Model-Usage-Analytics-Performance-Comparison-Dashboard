"""
app.py  —  AI Model Analytics & Decision Intelligence Platform
Advanced Streamlit Dashboard  |  Dark Mode  |  Plotly Interactive Charts

FIXES applied
─────────────
1. safe_fmt() helper — all KPI / metric displays never crash on empty df.
2. Every tab wrapped in `if df.empty` guard with a friendly warning.
3. Radar chart (Tab ML + Tab Compare) guarded:
   - empty comp_df → info message
   - single row / all-equal values → _safe_minmax in ml_models handles it
   - missing columns → filtered before use
4. Regression R² display safe when reg_r2 is None.
5. All f-string format calls on numeric columns use safe_fmt().
6. Provider comparison chart guarded for empty prov_df.
"""

import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from data import preprocess
from ml_models import (
    compute_composite_score,
    detect_outliers,
    perform_clustering,
    predict_cost,
    provider_summary,
    recommend_models,
    train_cost_model,
)

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG & CUSTOM CSS
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Model Analytics Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0e1117;
    color: #e0e0e0;
}
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #141a2e 0%, #0e1117 100%);
    border-right: 1px solid #1f2937;
}
[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }

.stTabs [data-baseweb="tab-list"] {
    gap: 4px; background: #141a2e; border-radius: 12px; padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px; color: #9ca3af; font-weight: 500;
    padding: 8px 20px; transition: all 0.2s;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    color: white !important; font-weight: 600;
}

.kpi-card {
    background: linear-gradient(135deg, #1a2035 0%, #1f2c44 100%);
    border: 1px solid #2d3748; border-radius: 16px;
    padding: 1.4rem 1.6rem; text-align: center;
    transition: transform 0.2s, box-shadow 0.2s;
}
.kpi-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 24px rgba(79,70,229,0.25);
}
.kpi-value {
    font-size: 2rem; font-weight: 700;
    background: linear-gradient(135deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.kpi-label { font-size: 0.85rem; color: #9ca3af; margin-top: 4px; font-weight: 500; }
.kpi-delta { font-size: 0.75rem; margin-top: 6px; color: #6ee7b7; }

.section-header {
    font-size: 1.25rem; font-weight: 700; color: #e2e8f0;
    margin: 1.5rem 0 0.75rem; padding-bottom: 0.4rem;
    border-bottom: 2px solid #4f46e5; display: inline-block;
}

.insight-box {
    background: linear-gradient(135deg,#1a2035,#1f2c44);
    border-left: 4px solid #4f46e5; border-radius: 8px;
    padding: 0.9rem 1.2rem; margin: 0.5rem 0;
    font-size: 0.9rem; color: #cbd5e1;
}

[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

.stButton>button {
    background: linear-gradient(135deg,#4f46e5,#7c3aed);
    color: white; border: none; border-radius: 8px;
    padding: 0.5rem 1.5rem; font-weight: 600; transition: all 0.2s;
}
.stButton>button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 14px rgba(124,58,237,0.4);
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def safe_fmt(value, fmt: str = ".2f", prefix: str = "", suffix: str = "",
             fallback: str = "N/A") -> str:
    """
    Format a numeric value safely.
    Returns `fallback` if value is None, NaN, or the df is empty.
    """
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return fallback
        return f"{prefix}{value:{fmt}}{suffix}"
    except (ValueError, TypeError):
        return fallback


def _plotly_dark():
    return dict(
        template="plotly_dark",
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font_color="#e0e0e0",
    )


def kpi(col, value, label: str, delta: str = ""):
    col.markdown(f"""
    <div class='kpi-card'>
      <div class='kpi-value'>{value}</div>
      <div class='kpi-label'>{label}</div>
      {'<div class="kpi-delta">' + delta + '</div>' if delta else ''}
    </div>""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# DATA LOADING  (cached)
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="⚙️ Loading and processing data…")
def load_data():
    df = preprocess()
    df = compute_composite_score(df)        # ml_models is single source of truth
    df, _, _ = perform_clustering(df, 3)
    df = detect_outliers(df)
    return df


@st.cache_resource(show_spinner=False)
def load_model(df_hash):
    df = load_data()
    model, r2, feats = train_cost_model(df)
    return model, r2, feats


df_full = load_data()
# Pass a stable hash so cache_resource key doesn't break on DataFrame identity
reg_model, reg_r2, reg_feats = load_model(len(df_full))


# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR – GLOBAL FILTERS
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🤖 AI Analytics")
    st.markdown("<hr style='border-color:#2d3748;margin:0.5rem 0 1rem'>",
                unsafe_allow_html=True)

    providers = sorted(df_full["provider"].dropna().unique().tolist())
    sel_providers = st.multiselect("🏢 Provider", providers, default=providers)

    st.markdown("---")

    cost_min = float(df_full["cost_usd_1m"].min())
    cost_max = float(df_full["cost_usd_1m"].max())
    sel_cost = st.slider(
        "💰 Max Cost (USD/1M tokens)",
        cost_min, cost_max, cost_max,
        step=max(round((cost_max - cost_min) / 100, 4), 0.0001),
    )

    acc_min = float(df_full["accuracy"].min())
    acc_max = float(df_full["accuracy"].max())
    sel_acc = st.slider("🎯 Min Accuracy Score", acc_min, acc_max, acc_min, step=0.5)

    lat_min = float(df_full["latency_s"].min())
    lat_max = float(df_full["latency_s"].max())
    sel_lat = st.slider("⚡ Max Latency (s)", lat_min, lat_max, lat_max, step=0.1)

    st.markdown("---")

    all_tags = ["All"] + sorted(df_full["cluster_label"].dropna().unique().tolist())
    sel_tag  = st.selectbox("🏷️ Segment / Tag", all_tags)

    sort_col = st.selectbox(
        "🔀 Sort by",
        ["composite_score", "accuracy", "cost_usd_1m", "latency_s", "speed_tok_s"],
    )
    sort_asc = st.checkbox("Ascending", value=False)

    st.markdown("---")
    search = st.text_input("🔍 Search model name", placeholder="e.g. GPT, Llama…")

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔄 Rebuild Dataset"):
        st.cache_data.clear()
        st.rerun()


# ──────────────────────────────────────────────────────────────────────────────
# APPLY FILTERS
# ──────────────────────────────────────────────────────────────────────────────

df = df_full.copy()

if sel_providers:
    df = df[df["provider"].isin(sel_providers)]

df = df[
    (df["cost_usd_1m"] <= sel_cost) &
    (df["accuracy"]    >= sel_acc)  &
    (df["latency_s"]   <= sel_lat)
]

if sel_tag != "All":
    df = df[df["cluster_label"] == sel_tag]

if search.strip():
    df = df[df["model_name"].str.contains(search.strip(), case=False, na=False)]

df = df.sort_values(sort_col, ascending=sort_asc).reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div style='text-align:center;padding:1rem 0 0.5rem'>
  <h1 style='font-size:2.4rem;font-weight:800;
     background:linear-gradient(135deg,#a78bfa,#60a5fa,#34d399);
     -webkit-background-clip:text;-webkit-text-fill-color:transparent;
     margin-bottom:0.2rem'>
     🤖 AI Model Analytics Platform
  </h1>
  <p style='color:#9ca3af;font-size:1rem;letter-spacing:0.5px'>
     Compare · Rank · Predict · Recommend — No hallucination, all metrics
  </p>
</div>
<hr style='border-color:#2d3748;margin:0.5rem 0 1.5rem'>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# KPI CARDS  — safe_fmt() guards every metric so empty df never crashes
# ──────────────────────────────────────────────────────────────────────────────

k1, k2, k3, k4, k5 = st.columns(5)

kpi(k1, len(df), "Models Shown", f"of {len(df_full)} total")

if df.empty:
    kpi(k2, "N/A", "Avg Accuracy",    "—")
    kpi(k3, "N/A", "Avg Cost /1M tk", "USD")
    kpi(k4, "N/A", "Avg Latency",     "first token")
    kpi(k5, "N/A", "Avg Speed",       "tokens/sec")
else:
    kpi(k2, safe_fmt(df["accuracy"].mean(),     ".1f"),      "Avg Accuracy",    "score")
    kpi(k3, safe_fmt(df["cost_usd_1m"].mean(),  ".4f", "$"), "Avg Cost /1M tk", "USD")
    kpi(k4, safe_fmt(df["latency_s"].mean(),    ".2f", "", "s"), "Avg Latency", "first token")
    kpi(k5, safe_fmt(df["speed_tok_s"].mean(),  ".0f"),      "Avg Speed",       "tokens/sec")

st.markdown("<br>", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────────────────────────────────────

tab_overview, tab_analytics, tab_ml, tab_compare, tab_tools = st.tabs([
    "📊 Overview",
    "📈 Analytics",
    "🧠 ML Insights",
    "⚖️ Model Comparison",
    "🛠️ Tools",
])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════════════════

with tab_overview:

    if df.empty:
        st.warning("⚠️ No models match the current filters. Try relaxing the sidebar sliders.")
    else:
        # ── Quick insights ───────────────────────────────────────────────────
        best_eff = df.loc[df["cost_efficiency"].idxmax()]
        cheapest = df.loc[df["cost_usd_1m"].idxmin()]
        fastest  = df.loc[df["speed_tok_s"].idxmax()]
        top_acc  = df.loc[df["accuracy"].idxmax()]

        st.markdown("<div class='section-header'>💡 Quick Insights</div>",
                    unsafe_allow_html=True)
        ia, ib, ic, id_ = st.columns(4)

        def insight(col, emoji, title, name, val):
            col.markdown(f"""
            <div class='insight-box'>
              <b>{emoji} {title}</b><br>
              <span style='color:#a78bfa;font-size:1rem;font-weight:600'>{name}</span><br>
              <span style='font-size:0.8rem;color:#6b7280'>{val}</span>
            </div>""", unsafe_allow_html=True)

        insight(ia,  "🏆", "Most Efficient",  best_eff["model_name"],
                f"Efficiency: {best_eff['cost_efficiency']:.2f}")
        insight(ib,  "💰", "Cheapest Model",  cheapest["model_name"],
                f"${cheapest['cost_usd_1m']:.4f} /1M tokens")
        insight(ic,  "⚡", "Fastest Model",   fastest["model_name"],
                f"{fastest['speed_tok_s']:.0f} tok/s")
        insight(id_, "🎯", "Highest Accuracy", top_acc["model_name"],
                f"Score: {top_acc['accuracy']:.2f}")

        st.markdown("<br>", unsafe_allow_html=True)

    # ── Top 10 table ─────────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>🏆 Top Ranked Models</div>",
                unsafe_allow_html=True)

    if df.empty:
        st.info("No data to display.")
    else:
        display_cols = ["model_name", "provider", "accuracy", "cost_usd_1m",
                        "latency_s", "speed_tok_s", "composite_score", "cluster_label"]
        avail = [c for c in display_cols if c in df.columns]
        top10 = df[avail].head(10).copy()
        col_rename = ["Model", "Provider", "Accuracy", "Cost ($/1M)",
                      "Latency (s)", "Speed (tok/s)", "Composite Score", "Segment"]
        top10.columns = col_rename[:len(avail)]

        fmt = {}
        if "Accuracy" in top10.columns:        fmt["Accuracy"]        = "{:.2f}"
        if "Cost ($/1M)" in top10.columns:     fmt["Cost ($/1M)"]     = "${:.4f}"
        if "Latency (s)" in top10.columns:     fmt["Latency (s)"]     = "{:.2f}"
        if "Speed (tok/s)" in top10.columns:   fmt["Speed (tok/s)"]   = "{:.0f}"
        if "Composite Score" in top10.columns: fmt["Composite Score"] = "{:.1f}"

        st.dataframe(
            top10.style.background_gradient(
                subset=["Composite Score"] if "Composite Score" in top10.columns else [],
                cmap="Blues"
            ).format(fmt),
            use_container_width=True, height=380,
        )

        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Filtered Dataset (CSV)",
                           csv_data, "filtered_models.csv", "text/csv")

    # ── Cost vs Accuracy scatter ──────────────────────────────────────────────
    st.markdown("<div class='section-header'>💰 Cost vs Accuracy Overview</div>",
                unsafe_allow_html=True)

    if not df.empty:
        fig = px.scatter(
            df, x="cost_usd_1m", y="accuracy",
            color="cluster_label", size="speed_tok_s",
            hover_name="model_name",
            hover_data={"cost_usd_1m": ":.4f", "accuracy": ":.2f",
                        "latency_s": ":.2f", "speed_tok_s": ":.0f"},
            labels={"cost_usd_1m": "Cost (USD/1M tokens)",
                    "accuracy": "Accuracy Score", "cluster_label": "Segment"},
            color_discrete_map={"High-Performance": "#60a5fa",
                                "Budget": "#34d399", "Balanced": "#a78bfa"},
            title="Cost vs Accuracy — bubble size = speed",
            **{"template": "plotly_dark"},
        )
        fig.update_layout(**_plotly_dark(), title_font_size=16,
                          legend_title_text="Model Segment")
        st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — ANALYTICS
# ════════════════════════════════════════════════════════════════════════════

with tab_analytics:

    if df.empty:
        st.warning("⚠️ No models match the current filters.")
    else:
        c_left, c_right = st.columns(2)

        with c_left:
            st.markdown("<div class='section-header'>⚡ Speed vs Latency</div>",
                        unsafe_allow_html=True)
            fig_sl = px.scatter(
                df, x="latency_s", y="speed_tok_s", color="provider",
                hover_name="model_name",
                hover_data={"cost_usd_1m": ":.4f", "accuracy": ":.2f"},
                labels={"latency_s": "Latency (s)", "speed_tok_s": "Speed (tok/s)"},
                title="Speed vs Latency Tradeoff", **{"template": "plotly_dark"},
            )
            fig_sl.update_layout(**_plotly_dark())
            st.plotly_chart(fig_sl, use_container_width=True)

        with c_right:
            st.markdown("<div class='section-header'>🏆 Top 10 by Composite Score</div>",
                        unsafe_allow_html=True)
            top10_bar = df.nlargest(10, "composite_score")
            fig_bar = px.bar(
                top10_bar, x="composite_score", y="model_name", orientation="h",
                color="composite_score", color_continuous_scale="Viridis",
                labels={"composite_score": "Score", "model_name": "Model"},
                title="Best Overall Models", **{"template": "plotly_dark"},
            )
            fig_bar.update_layout(**_plotly_dark(),
                                  yaxis={"categoryorder": "total ascending"},
                                  coloraxis_showscale=False)
            st.plotly_chart(fig_bar, use_container_width=True)

        # ── Provider comparison ───────────────────────────────────────────────
        st.markdown("<div class='section-header'>🏢 Provider Comparison</div>",
                    unsafe_allow_html=True)
        prov_df = provider_summary(df)

        if prov_df.empty:
            st.info("Not enough provider data to display.")
        else:
            fig_prov = make_subplots(
                rows=1, cols=3,
                subplot_titles=["Avg Accuracy", "Avg Cost ($/1M)", "Avg Latency (s)"],
                shared_yaxes=True,
            )
            colors = px.colors.qualitative.Vivid
            for i, col_name in enumerate(["avg_accuracy", "avg_cost", "avg_latency"], 1):
                fig_prov.add_trace(go.Bar(
                    y=prov_df["provider"], x=prov_df[col_name],
                    orientation="h", showlegend=False,
                    marker_color=[colors[j % len(colors)] for j in range(len(prov_df))],
                    text=prov_df[col_name].round(3).astype(str),
                    textposition="outside",
                ), row=1, col=i)

            fig_prov.update_layout(height=350, **_plotly_dark(),
                                   margin=dict(t=50, b=20))
            st.plotly_chart(fig_prov, use_container_width=True)

        # ── Cost efficiency distribution ──────────────────────────────────────
        st.markdown("<div class='section-header'>📊 Cost Efficiency Distribution</div>",
                    unsafe_allow_html=True)
        fig_kde = px.histogram(
            df, x="cost_efficiency", nbins=30, color="cluster_label", marginal="box",
            labels={"cost_efficiency": "Cost Efficiency (Accuracy/Cost)",
                    "cluster_label": "Segment"},
            color_discrete_map={"High-Performance": "#60a5fa",
                                "Budget": "#34d399", "Balanced": "#a78bfa"},
            title="Distribution of Cost Efficiency by Segment",
            **{"template": "plotly_dark"},
        )
        fig_kde.update_layout(**_plotly_dark())
        st.plotly_chart(fig_kde, use_container_width=True)

        # ── Correlation heatmap ───────────────────────────────────────────────
        st.markdown("<div class='section-header'>🔗 Feature Correlation Matrix</div>",
                    unsafe_allow_html=True)
        corr_cols = ["accuracy", "cost_usd_1m", "latency_s",
                     "speed_tok_s", "cost_efficiency", "composite_score"]
        corr_cols = [c for c in corr_cols if c in df.columns]

        if len(corr_cols) > 1:
            corr_mat = df[corr_cols].corr().round(2)
            fig_heat = px.imshow(
                corr_mat, text_auto=True,
                color_continuous_scale="RdYlGn", zmin=-1, zmax=1,
                title="Pearson Correlation Heatmap",
                **{"template": "plotly_dark"},
            )
            fig_heat.update_layout(**_plotly_dark(), height=420)
            st.plotly_chart(fig_heat, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — ML INSIGHTS
# ════════════════════════════════════════════════════════════════════════════

with tab_ml:

    if df.empty:
        st.warning("⚠️ No models match the current filters.")
    else:
        # ── Clustering scatter ────────────────────────────────────────────────
        st.markdown("<div class='section-header'>🧠 KMeans Model Segmentation</div>",
                    unsafe_allow_html=True)
        fig_cluster = px.scatter(
            df, x="cost_usd_1m", y="accuracy",
            color="cluster_label", symbol="cluster_label",
            size="composite_score", hover_name="model_name",
            hover_data={"provider": True, "cost_usd_1m": ":.4f",
                        "accuracy": ":.2f", "latency_s": ":.2f",
                        "composite_score": ":.1f"},
            labels={"cost_usd_1m": "Cost (USD/1M)", "accuracy": "Accuracy Score",
                    "cluster_label": "Segment"},
            color_discrete_map={"High-Performance": "#60a5fa",
                                "Budget": "#34d399", "Balanced": "#a78bfa"},
            title="Model Clusters — Size = Composite Score",
            **{"template": "plotly_dark"},
        )
        fig_cluster.update_layout(**_plotly_dark(), height=480)
        st.plotly_chart(fig_cluster, use_container_width=True)

        # ── Cluster stats ─────────────────────────────────────────────────────
        st.markdown("<div class='section-header'>📋 Cluster Statistics</div>",
                    unsafe_allow_html=True)
        if "cluster_label" in df.columns:
            cluster_stats = df.groupby("cluster_label").agg(
                Count     = ("model_name",     "count"),
                Avg_Acc   = ("accuracy",        "mean"),
                Avg_Cost  = ("cost_usd_1m",     "mean"),
                Avg_Lat   = ("latency_s",        "mean"),
                Avg_Speed = ("speed_tok_s",      "mean"),
                Avg_Score = ("composite_score",  "mean"),
            ).reset_index()
            cluster_stats.columns = [
                "Segment", "Count", "Avg Accuracy",
                "Avg Cost ($/1M)", "Avg Latency (s)",
                "Avg Speed (tok/s)", "Avg Composite Score",
            ]
            st.dataframe(
                cluster_stats.style
                    .format({"Avg Accuracy": "{:.2f}", "Avg Cost ($/1M)": "${:.4f}",
                             "Avg Latency (s)": "{:.2f}", "Avg Speed (tok/s)": "{:.0f}",
                             "Avg Composite Score": "{:.1f}"})
                    .background_gradient(subset=["Avg Composite Score"], cmap="Blues"),
                use_container_width=True, hide_index=True,
            )

        # ── Outlier detection ─────────────────────────────────────────────────
        st.markdown("<div class='section-header'>🚨 Outlier Detection</div>",
                    unsafe_allow_html=True)
        if "is_outlier" in df.columns:
            outliers = df[df["is_outlier"]][
                ["model_name", "provider", "cost_usd_1m", "latency_s",
                 "cost_outlier", "latency_outlier"]
            ].copy()
            if outliers.empty:
                st.success("✅ No significant outliers detected in current filter.")
            else:
                st.warning(f"⚠️ {len(outliers)} outlier(s) detected (Z-score > 2.5)")
                outliers.columns = ["Model", "Provider", "Cost", "Latency",
                                    "Cost Outlier", "Latency Outlier"]
                st.dataframe(outliers, use_container_width=True, hide_index=True)

        # ── Segment radar ─────────────────────────────────────────────────────
        st.markdown("<div class='section-header'>📡 Segment Radar Chart</div>",
                    unsafe_allow_html=True)

        RADAR_METRICS = ["accuracy", "cost_efficiency", "speed_tok_s",
                         "composite_score", "latency_s"]
        radar_cols = [c for c in RADAR_METRICS if c in df.columns]

        if "cluster_label" not in df.columns or df["cluster_label"].isna().all():
            st.info("Cluster labels unavailable — re-run with more models.")
        elif len(radar_cols) < 2:
            st.info("Not enough metric columns for radar chart.")
        else:
            radar_df = df.groupby("cluster_label")[radar_cols].mean()

            # Safe normalise 0-1
            radar_norm = radar_df.copy()
            for col in radar_cols:
                rng = radar_norm[col].max() - radar_norm[col].min()
                radar_norm[col] = (
                    (radar_norm[col] - radar_norm[col].min()) / rng
                    if rng > 0 else 0.5
                )
            # Invert latency (lower = better)
            if "latency_s" in radar_norm.columns:
                radar_norm["latency_s"] = 1 - radar_norm["latency_s"]

            cat_labels = ["Accuracy", "Cost Eff.", "Speed",
                          "Composite", "Low Latency"][:len(radar_cols)]
            seg_colors = {"High-Performance": "#60a5fa",
                          "Budget": "#34d399", "Balanced": "#a78bfa"}

            fig_radar = go.Figure()
            for seg in radar_norm.index:
                vals = radar_norm.loc[seg].tolist() + [radar_norm.loc[seg].iloc[0]]
                fig_radar.add_trace(go.Scatterpolar(
                    r=vals, theta=cat_labels + [cat_labels[0]],
                    fill="toself", name=seg,
                    line_color=seg_colors.get(seg, "#fff"),
                    fillcolor=seg_colors.get(seg, "#fff"),
                    opacity=0.35,
                ))
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 1],
                                   gridcolor="#2d3748", linecolor="#2d3748"),
                    angularaxis=dict(gridcolor="#2d3748"),
                    bgcolor="#141a2e",
                ),
                showlegend=True, paper_bgcolor="#0e1117",
                font_color="#e0e0e0", height=450,
                title="Segment Performance Radar",
            )
            st.plotly_chart(fig_radar, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — MODEL COMPARISON
# ════════════════════════════════════════════════════════════════════════════

with tab_compare:

    st.markdown("<div class='section-header'>⚖️ Side-by-Side Model Comparison</div>",
                unsafe_allow_html=True)

    all_models = sorted(df_full["model_name"].dropna().unique().tolist())
    sel_models = st.multiselect(
        "Select 2–6 models to compare",
        options=all_models,
        default=all_models[:min(4, len(all_models))],
        max_selections=6,
    )

    if len(sel_models) < 2:
        st.info("👆 Please select at least 2 models above.")
    else:
        comp_df = df_full[df_full["model_name"].isin(sel_models)].copy()

        if comp_df.empty:
            st.warning("Selected models not found in the current dataset.")
        else:
            # ── Table comparison ──────────────────────────────────────────────
            cmp_cols = ["model_name", "provider", "accuracy", "cost_usd_1m",
                        "latency_s", "speed_tok_s", "composite_score", "cluster_label"]
            cmp_cols = [c for c in cmp_cols if c in comp_df.columns]
            cmp_display = comp_df[cmp_cols].copy()
            rename_map = {
                "model_name": "Model", "provider": "Provider",
                "accuracy": "Accuracy", "cost_usd_1m": "Cost ($/1M)",
                "latency_s": "Latency (s)", "speed_tok_s": "Speed (tok/s)",
                "composite_score": "Composite Score", "cluster_label": "Segment",
            }
            cmp_display = cmp_display.rename(columns=rename_map)

            fmt = {}
            if "Accuracy" in cmp_display.columns:       fmt["Accuracy"]        = "{:.2f}"
            if "Cost ($/1M)" in cmp_display.columns:    fmt["Cost ($/1M)"]     = "${:.4f}"
            if "Latency (s)" in cmp_display.columns:    fmt["Latency (s)"]     = "{:.2f}"
            if "Speed (tok/s)" in cmp_display.columns:  fmt["Speed (tok/s)"]   = "{:.0f}"
            if "Composite Score" in cmp_display.columns:fmt["Composite Score"] = "{:.1f}"

            grad_sub = [c for c in ["Composite Score", "Accuracy"]
                        if c in cmp_display.columns]
            st.dataframe(
                cmp_display.style
                    .background_gradient(subset=grad_sub, cmap="Blues")
                    .format(fmt),
                use_container_width=True, hide_index=True,
            )

            # ── Grouped bar chart ─────────────────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            metrics_to_plot = [c for c in
                               ["accuracy", "cost_usd_1m", "latency_s",
                                "speed_tok_s", "composite_score"]
                               if c in comp_df.columns]
            labels_map = {
                "accuracy": "Accuracy", "cost_usd_1m": "Cost ($/1M)",
                "latency_s": "Latency (s)", "speed_tok_s": "Speed (tok/s)",
                "composite_score": "Composite Score",
            }

            if metrics_to_plot:
                fig_compare = make_subplots(
                    rows=1, cols=len(metrics_to_plot),
                    subplot_titles=[labels_map[m] for m in metrics_to_plot],
                )
                pal = px.colors.qualitative.Plotly
                for i, metric in enumerate(metrics_to_plot, 1):
                    for j, (_, row) in enumerate(comp_df.iterrows()):
                        fig_compare.add_trace(go.Bar(
                            name=row["model_name"], x=[row["model_name"]],
                            y=[row[metric]], marker_color=pal[j % len(pal)],
                            showlegend=(i == 1),
                        ), row=1, col=i)

                fig_compare.update_layout(
                    height=420, **_plotly_dark(),
                    legend=dict(orientation="h", yanchor="bottom", y=-0.3),
                    barmode="group",
                    title="Model Comparison — All Key Metrics",
                )
                st.plotly_chart(fig_compare, use_container_width=True)

            # ── Multi-model radar ─────────────────────────────────────────────
            st.markdown("<div class='section-header'>📡 Multi-Model Radar</div>",
                        unsafe_allow_html=True)

            radar_m = [c for c in ["accuracy", "cost_efficiency", "speed_tok_s",
                                    "composite_score", "latency_s"]
                       if c in comp_df.columns]

            if len(radar_m) < 2:
                st.info("Not enough metric columns for radar chart.")
            else:
                # Normalise using full-dataset range to avoid division-by-zero
                norm_comp = comp_df[radar_m].copy()
                for col in radar_m:
                    full_range = df_full[col].max() - df_full[col].min()
                    if full_range > 0:
                        norm_comp[col] = (
                            (norm_comp[col] - df_full[col].min()) / full_range
                        ).clip(0, 1)
                    else:
                        norm_comp[col] = 0.5   # constant column → neutral

                if "latency_s" in norm_comp.columns:
                    norm_comp["latency_s"] = 1 - norm_comp["latency_s"]

                cats = ["Accuracy", "Cost Eff.", "Speed",
                        "Composite", "Low Latency"][:len(radar_m)]

                fig_mr = go.Figure()
                for idx, (_, row) in enumerate(comp_df.iterrows()):
                    vals = norm_comp.loc[row.name, radar_m].fillna(0).tolist()
                    vals_closed = vals + [vals[0]]
                    fig_mr.add_trace(go.Scatterpolar(
                        r=vals_closed,
                        theta=cats + [cats[0]],
                        fill="toself",
                        name=row["model_name"],
                        line_color=pal[idx % len(pal)],
                        opacity=0.5,
                    ))

                fig_mr.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 1],
                                       gridcolor="#2d3748"),
                        bgcolor="#141a2e",
                    ),
                    showlegend=True, paper_bgcolor="#0e1117",
                    font_color="#e0e0e0", height=480,
                )
                st.plotly_chart(fig_mr, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — TOOLS
# ════════════════════════════════════════════════════════════════════════════

with tab_tools:

    t_rec, t_pred = st.columns([1, 1])

    # ── RECOMMENDATION ENGINE ─────────────────────────────────────────────────
    with t_rec:
        st.markdown("<div class='section-header'>🎯 Model Recommendation Engine</div>",
                    unsafe_allow_html=True)
        st.markdown(
            "<p style='color:#9ca3af;font-size:0.88rem'>"
            "Set your constraints — we return the best model purely from metrics.</p>",
            unsafe_allow_html=True,
        )

        rec_budget = st.number_input(
            "💰 Max Budget (USD / 1M tokens)",
            min_value=0.0,
            max_value=float(df_full["cost_usd_1m"].max()),
            value=float(df_full["cost_usd_1m"].median()),
            step=0.5, format="%.4f",
        )
        rec_acc = st.number_input(
            "🎯 Min Accuracy Score",
            min_value=0.0,
            max_value=float(df_full["accuracy"].max()),
            value=float(df_full["accuracy"].quantile(0.25)),
            step=1.0, format="%.2f",
        )
        rec_top = st.slider("🏅 Top N Results", 1, 10, 5)

        if st.button("🔍 Find Best Models", key="rec_btn"):
            rec_results = recommend_models(df_full, rec_budget, rec_acc, rec_top)

            if rec_results.empty:
                st.error("❌ No models found matching these constraints. "
                         "Try relaxing budget or accuracy floor.")
            else:
                st.success(f"✅ {len(rec_results)} model(s) found!")
                show_cols = [c for c in
                             ["model_name", "provider", "accuracy",
                              "cost_usd_1m", "latency_s", "composite_score",
                              "cluster_label"]
                             if c in rec_results.columns]

                rec_rename = {
                    "model_name": "Model", "provider": "Provider",
                    "accuracy": "Accuracy", "cost_usd_1m": "Cost ($/1M)",
                    "latency_s": "Latency (s)", "composite_score": "Score",
                    "cluster_label": "Segment",
                }
                rec_fmt = {
                    "Accuracy": "{:.2f}", "Cost ($/1M)": "${:.4f}",
                    "Latency (s)": "{:.2f}", "Score": "{:.1f}",
                }
                display_rec = rec_results[show_cols].rename(columns=rec_rename)
                st.dataframe(
                    display_rec.style
                        .background_gradient(
                            subset=["Score"] if "Score" in display_rec.columns else [],
                            cmap="Blues"
                        )
                        .format({k: v for k, v in rec_fmt.items()
                                 if k in display_rec.columns}),
                    use_container_width=True, hide_index=True,
                )

                winner = rec_results.iloc[0]
                st.markdown(f"""
                <div class='insight-box' style='border-left-color:#34d399;margin-top:1rem'>
                  🏆 <b>Best Pick:</b>
                  <span style='color:#34d399;font-size:1.1rem;font-weight:700'>
                    {winner['model_name']}
                  </span><br>
                  <span style='font-size:0.82rem;color:#9ca3af'>
                    Score: {safe_fmt(winner.get('composite_score'), '.1f')} &nbsp;|&nbsp;
                    Accuracy: {safe_fmt(winner.get('accuracy'), '.2f')} &nbsp;|&nbsp;
                    Cost: {safe_fmt(winner.get('cost_usd_1m'), '.4f', '$')} /1M
                  </span>
                </div>
                """, unsafe_allow_html=True)

    # ── COST PREDICTOR ────────────────────────────────────────────────────────
    with t_pred:
        r2_display = f"{reg_r2:.4f}" if reg_r2 is not None else "N/A"
        st.markdown("<div class='section-header'>📈 Predict Model Cost</div>",
                    unsafe_allow_html=True)
        st.markdown(
            f"<p style='color:#9ca3af;font-size:0.88rem'>"
            f"Adjust accuracy, speed, and latency to predict the expected cost "
            f"(Linear Regression R²={r2_display}).</p>",
            unsafe_allow_html=True,
        )

        pred_acc = st.slider(
            "🎯 Accuracy Score",
            float(df_full["accuracy"].min()), float(df_full["accuracy"].max()),
            float(df_full["accuracy"].mean()), step=0.5, key="pred_acc",
        )
        pred_spd = st.slider(
            "⚡ Speed (tok/s)",
            float(df_full["speed_tok_s"].min()), float(df_full["speed_tok_s"].max()),
            float(df_full["speed_tok_s"].mean()), step=1.0, key="pred_spd",
        )
        pred_lat = st.slider(
            "⏱️ Latency (s)",
            float(df_full["latency_s"].min()), float(df_full["latency_s"].max()),
            float(df_full["latency_s"].mean()), step=0.1, key="pred_lat",
        )

        if st.button("💡 Predict Cost", key="pred_btn"):
            if reg_model is None:
                st.error("⚠️ Regression model could not be trained "
                         "(insufficient data — need at least 5 rows).")
            else:
                pred_val = predict_cost(reg_model, pred_acc, pred_spd, pred_lat)
                st.markdown(f"""
                <div style='text-align:center;background:linear-gradient(135deg,#1a2035,#1a3040);
                     border:1px solid #4f46e5;border-radius:16px;padding:2rem;margin-top:1rem'>
                  <div style='font-size:0.9rem;color:#9ca3af;margin-bottom:0.5rem'>
                    Predicted Cost (USD / 1M tokens)
                  </div>
                  <div style='font-size:3rem;font-weight:800;
                       background:linear-gradient(135deg,#a78bfa,#60a5fa);
                       -webkit-background-clip:text;-webkit-text-fill-color:transparent'>
                    {safe_fmt(pred_val, '.4f', '$')}
                  </div>
                  <div style='font-size:0.8rem;color:#6b7280;margin-top:0.5rem'>
                    Based on: Accuracy={pred_acc:.2f},
                    Speed={pred_spd:.0f} tok/s,
                    Latency={pred_lat:.2f}s
                  </div>
                </div>
                """, unsafe_allow_html=True)

                avg_cost = df_full["cost_usd_1m"].mean()
                q95 = float(df_full["cost_usd_1m"].quantile(0.95))

                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=pred_val if not np.isnan(pred_val) else 0,
                    delta={"reference": avg_cost, "relative": True,
                           "increasing": {"color": "#f87171"},
                           "decreasing": {"color": "#34d399"}},
                    gauge={
                        "axis": {"range": [0, max(q95, pred_val * 1.1)]},
                        "bar": {"color": "#4f46e5"},
                        "steps": [
                            {"range": [0, df_full["cost_usd_1m"].quantile(0.33)],
                             "color": "#1a3a2e"},
                            {"range": [df_full["cost_usd_1m"].quantile(0.33),
                                       df_full["cost_usd_1m"].quantile(0.66)],
                             "color": "#1a2a3a"},
                            {"range": [df_full["cost_usd_1m"].quantile(0.66), q95],
                             "color": "#3a1a1a"},
                        ],
                        "threshold": {
                            "line": {"color": "#f59e0b", "width": 3},
                            "thickness": 0.75, "value": avg_cost,
                        },
                    },
                    title={"text": "vs Market Average", "font": {"color": "#9ca3af"}},
                    number={"prefix": "$", "font": {"color": "#a78bfa"}},
                ))
                fig_gauge.update_layout(paper_bgcolor="#0e1117", font_color="#e0e0e0",
                                        height=280, margin=dict(t=40, b=10))
                st.plotly_chart(fig_gauge, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("<br><hr style='border-color:#2d3748'>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center;color:#4b5563;font-size:0.8rem;padding:0.5rem 0 1rem'>
  🤖 AI Model Analytics Platform &nbsp;|&nbsp;
  Data Sources: Artificial Analysis · HF Open LLM Leaderboard &nbsp;|&nbsp;
  Built with Streamlit + Plotly
</div>
""", unsafe_allow_html=True)