from __future__ import annotations

import warnings
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data import (
    MODEL_COLORS,
    get_dashboard_data,
    model_picker_score,
)

warnings.filterwarnings("ignore")

st.set_page_config(
    layout="wide",
    page_title="AI Model Analytics Dashboard",
    page_icon="🤖",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", size=12, color="#cccccc"),
    margin=dict(l=10, r=20, t=45, b=10),
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 34px;
        font-weight: 700;
        background: linear-gradient(90deg, #00C6FF, #0072FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
    }
    .sub-title {
        color: #888;
        font-size: 13px;
        margin-bottom: 20px;
    }
    .kpi-box {
        background: #1a1a2e;
        border: 1px solid #2a2a4a;
        border-radius: 12px;
        padding: 16px 18px;
        text-align: center;
    }
    .kpi-label {
        font-size: 10px;
        color: #777;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .kpi-value {
        font-size: 22px;
        font-weight: 700;
        margin-top: 5px;
    }
    .kpi-sub {
        font-size: 10px;
        color: #555;
        margin-top: 3px;
    }
    .insight-box {
        background: #0d1117;
        border-left: 3px solid #00C6FF;
        border-radius: 0 8px 8px 0;
        padding: 10px 14px;
        font-size: 13px;
        color: #bbb;
        margin: 6px 0;
        line-height: 1.5;
    }
    .rec-card {
        background: #101826;
        border: 1px solid #2a2f3a;
        border-radius: 16px;
        padding: 20px;
    }
    .rec-card.top {
        border: 2px solid #00C6FF;
        box-shadow: 0 0 0 1px rgba(0,198,255,0.08);
    }
    .badge {
        display: inline-flex;
        align-items: center;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 600;
    }
    .badge-blue { background: rgba(74,144,217,.16); color: #7ec3ff; }
    .badge-green { background: rgba(92,184,92,.16); color: #8dd98d; }
    .badge-amber { background: rgba(240,184,96,.16); color: #ffd38a; }
    .score-bar-wrap { display:flex; align-items:center; gap:8px; }
    .score-bar { height: 6px; border-radius: 999px; background: rgba(255,255,255,.12); flex: 1; overflow: hidden; }
    .score-fill { height: 100%; border-radius: 999px; }
    .score-num { font-size: 12px; font-weight: 600; min-width: 32px; text-align: right; }
    .rec-grid { display:grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap:10px; margin-top: 10px; }
    .rec-item { padding: 10px 12px; background: #141c29; border-radius: 12px; }
    .rec-item-label { font-size: 11px; color: #8892a0; margin-bottom: 4px; }
    .rec-item-val { font-size: 14px; font-weight: 600; color: #f5f7fb; }
    .select-row { display:flex; gap:10px; flex-wrap:wrap; margin: 8px 0 16px; }
    .sel-btn {
        padding: 7px 14px;
        font-size: 12px;
        font-weight: 600;
        border: 1px solid rgba(255,255,255,.14);
        border-radius: 12px;
        background: transparent;
        color: #bcc5d1;
        cursor: pointer;
    }
    .sel-btn:hover { color: #fff; border-color: rgba(255,255,255,.28); }
    .sel-btn.active { background: rgba(0,198,255,.12); color: #7fdfff; border-color: rgba(0,198,255,.3); }
    .what-means {
        background: #141c29;
        border-radius: 12px;
        padding: 16px;
        font-size: 12px;
        color: #bac4d1;
        line-height: 1.7;
        margin-top: 1rem;
    }
    .what-means strong { color: #ffffff; font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)


def hex_to_rgba(hex_color: str, alpha: float = 0.22) -> str:
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def best_of(df, col: str, ascending: bool = False):
    return df.loc[df[col].idxmin() if ascending else df[col].idxmax()]


@st.cache_data
def load_data():
    return get_dashboard_data()


data = load_data()
df_raw = data["raw"].copy()
df_agg = data["models"].copy()
df_use = data["use_cases"].copy()
df_region = data["regions"].copy()
df_winners = data["winners"].copy()

st.sidebar.title("🎛️ Filters")
st.sidebar.markdown("---")

all_models = sorted(df_agg["Model"].tolist())
selected_models = st.sidebar.multiselect("Select AI Models", all_models, default=all_models)

all_use_cases = sorted(df_raw["Use_Case"].unique().tolist())
selected_use_cases = st.sidebar.multiselect("Select Use Cases", all_use_cases, default=all_use_cases)

all_regions = sorted(df_raw["Region"].unique().tolist())
selected_regions = st.sidebar.multiselect("Select Regions", all_regions, default=all_regions)

selected_models = selected_models or all_models
selected_use_cases = selected_use_cases or all_use_cases
selected_regions = selected_regions or all_regions

st.sidebar.markdown("---")
st.sidebar.caption("📊 Data Overview")
st.sidebar.caption(f"• {len(df_raw)} deployments")
st.sidebar.caption(f"• {len(all_models)} AI models")
st.sidebar.caption(f"• {len(all_regions)} regions")
st.sidebar.caption(f"• {len(all_use_cases)} use cases")

df_filtered = df_raw[
    df_raw["Model"].isin(selected_models)
    & df_raw["Use_Case"].isin(selected_use_cases)
    & df_raw["Region"].isin(selected_regions)
].copy()

df_agg_filtered = df_agg[df_agg["Model"].isin(selected_models)].copy()

st.markdown(
    '<p class="main-title">🤖 AI Model Usage Analytics & Performance Dashboard</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="sub-title">Compare GPT-4, Claude, Gemini, Llama & Mistral — based on real-world deployments across the globe</p>',
    unsafe_allow_html=True,
)

if not df_agg_filtered.empty:
    c1, c2, c3, c4 = st.columns(4)

    best_accuracy = best_of(df_agg_filtered, "Accuracy")
    best_value = best_of(df_agg_filtered, "Score")
    cheapest = best_of(df_agg_filtered, "Cost", ascending=True)
    fastest = best_of(df_agg_filtered, "Latency", ascending=True)

    with c1:
        st.markdown(
            f"""
            <div class="kpi-box">
                <div class="kpi-label">🏆 Highest Accuracy</div>
                <div class="kpi-value" style="color:#FFD700">{best_accuracy['Model']}</div>
                <div class="kpi-sub">{best_accuracy['Accuracy']:.1f}% accuracy</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="kpi-box">
                <div class="kpi-label">⭐ Best Value Model</div>
                <div class="kpi-value" style="color:#00FF88">{best_value['Model']}</div>
                <div class="kpi-sub">Score {best_value['Score']:.0f} / 100</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="kpi-box">
                <div class="kpi-label">💰 Cheapest Model</div>
                <div class="kpi-value" style="color:#00C6FF">{cheapest['Model']}</div>
                <div class="kpi-sub">${cheapest['Cost']:.4f} per 1K tokens</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"""
            <div class="kpi-box">
                <div class="kpi-label">⚡ Fastest Model</div>
                <div class="kpi-value" style="color:#FF6B6B">{fastest['Model']}</div>
                <div class="kpi-sub">{fastest['Latency']:.0f} ms latency</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("---")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📊 Model Comparison",
        "💰 Cost & Speed",
        "🌍 Usage by Region",
        "🎯 Use Case Analysis",
        "🏅 Model Picker",
    ]
)

with tab1:
    st.subheader("📊 Head-to-Head AI Model Comparison")

    if df_agg_filtered.empty:
        st.warning("No data available for selected filters")
    else:
        sorted_agg = df_agg_filtered.sort_values("Score", ascending=False).reset_index(drop=True)
        show_df = sorted_agg[["Model", "Accuracy", "Cost", "Latency", "Uptime", "Throughput", "Requests", "Users", "Score"]].copy()
        show_df["Accuracy"] = show_df["Accuracy"].map(lambda x: f"{x:.1f}%")
        show_df["Cost"] = show_df["Cost"].map(lambda x: f"${x:.4f}")
        show_df["Latency"] = show_df["Latency"].map(lambda x: f"{int(round(x))} ms")
        show_df["Uptime"] = show_df["Uptime"].map(lambda x: f"{x:.2f}%")
        show_df["Throughput"] = show_df["Throughput"].map(lambda x: f"{int(round(x))} rps")
        show_df["Requests"] = show_df["Requests"].map(lambda x: f"{int(x):,}")
        show_df["Users"] = show_df["Users"].map(lambda x: f"{int(x):,}")
        show_df["Score"] = show_df["Score"].map(lambda x: f"{x:.1f}")

        st.dataframe(show_df, use_container_width=True, hide_index=True)

        c1, c2 = st.columns(2)

        with c1:
            acc_df = df_agg_filtered.sort_values("Accuracy", ascending=True)
            fig_acc = px.bar(
                acc_df,
                x="Accuracy",
                y="Model",
                orientation="h",
                color="Model",
                color_discrete_map=MODEL_COLORS,
                title="Model Accuracy Comparison",
                text=acc_df["Accuracy"].map(lambda x: f"{x:.1f}%"),
            )
            fig_acc.update_traces(textposition="outside", cliponaxis=False)
            fig_acc.update_layout(**PLOTLY_LAYOUT, showlegend=False, height=500, xaxis_title="Accuracy (%)", yaxis_title="")
            st.plotly_chart(fig_acc, use_container_width=True)

        with c2:
            score_df = df_agg_filtered.sort_values("Score", ascending=True)
            fig_score = px.bar(
                score_df,
                x="Score",
                y="Model",
                orientation="h",
                color="Model",
                color_discrete_map=MODEL_COLORS,
                title="Value Score (Higher is Better)",
                text=score_df["Score"].map(lambda x: f"{x:.1f}"),
            )
            fig_score.update_traces(textposition="outside", cliponaxis=False)
            fig_score.update_layout(**PLOTLY_LAYOUT, showlegend=False, height=500, xaxis_title="Score (0–100)", yaxis_title="")
            st.plotly_chart(fig_score, use_container_width=True)

        st.subheader("Multi-Metric Radar Chart")
        radar_data = df_agg_filtered.copy()
        radar_data["Accuracy_norm"] = (radar_data["Accuracy"] - radar_data["Accuracy"].min()) / (radar_data["Accuracy"].max() - radar_data["Accuracy"].min() + 1e-9) * 100
        radar_data["Cost_norm"] = (radar_data["Cost"].max() - radar_data["Cost"]) / (radar_data["Cost"].max() - radar_data["Cost"].min() + 1e-9) * 100
        radar_data["Latency_norm"] = (radar_data["Latency"].max() - radar_data["Latency"]) / (radar_data["Latency"].max() - radar_data["Latency"].min() + 1e-9) * 100
        radar_data["Uptime_norm"] = (radar_data["Uptime"] - radar_data["Uptime"].min()) / (radar_data["Uptime"].max() - radar_data["Uptime"].min() + 1e-9) * 100
        radar_data["Throughput_norm"] = (radar_data["Throughput"] - radar_data["Throughput"].min()) / (radar_data["Throughput"].max() - radar_data["Throughput"].min() + 1e-9) * 100

        categories = ["Accuracy", "Cost Efficiency", "Speed", "Uptime", "Throughput"]
        fig_radar = go.Figure()
        for _, row in radar_data.iterrows():
            values = [
                row["Accuracy_norm"],
                row["Cost_norm"],
                row["Latency_norm"],
                row["Uptime_norm"],
                row["Throughput_norm"],
            ]
            values.append(values[0])
            color = MODEL_COLORS.get(row["Model"], "#888888")
            fig_radar.add_trace(
                go.Scatterpolar(
                    r=values,
                    theta=categories + [categories[0]],
                    name=row["Model"],
                    fill="toself",
                    line=dict(color=color, width=2),
                    fillcolor=hex_to_rgba(color, 0.18),
                )
            )
        fig_radar.update_layout(
            **PLOTLY_LAYOUT,
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            height=520,
            showlegend=True,
            legend=dict(orientation="h"),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

with tab2:
    st.subheader("💰 Cost & Speed Analysis")

    if df_agg_filtered.empty:
        st.warning("No data available for selected filters")
    else:
        c1, c2 = st.columns(2)

        with c1:
            cost_df = df_agg_filtered.sort_values("Cost", ascending=True)
            fig_cost = px.bar(
                cost_df,
                x="Model",
                y="Cost",
                color="Model",
                color_discrete_map=MODEL_COLORS,
                title="Cost per 1,000 Tokens (USD)",
                text=cost_df["Cost"].map(lambda x: f"${x:.4f}"),
            )
            fig_cost.update_traces(textposition="outside", cliponaxis=False)
            fig_cost.update_layout(**PLOTLY_LAYOUT, showlegend=False, xaxis_tickangle=-45, height=460)
            st.plotly_chart(fig_cost, use_container_width=True)

        with c2:
            latency_df = df_agg_filtered.sort_values("Latency", ascending=True)
            fig_latency = px.bar(
                latency_df,
                x="Model",
                y="Latency",
                color="Model",
                color_discrete_map=MODEL_COLORS,
                title="Average Latency (ms)",
                text=latency_df["Latency"].map(lambda x: f"{int(round(x))} ms"),
            )
            fig_latency.update_traces(textposition="outside", cliponaxis=False)
            fig_latency.update_layout(**PLOTLY_LAYOUT, showlegend=False, xaxis_tickangle=-45, height=460)
            st.plotly_chart(fig_latency, use_container_width=True)

        fig_scatter = px.scatter(
            df_agg_filtered,
            x="Cost",
            y="Accuracy",
            size="Requests",
            color="Model",
            color_discrete_map=MODEL_COLORS,
            hover_name="Model",
            title="Cost vs Accuracy Trade-off (Bubble size = Total Requests)",
            labels={"Cost": "Cost per 1K tokens ($)", "Accuracy": "Accuracy (%)"},
        )
        fig_scatter.update_layout(**PLOTLY_LAYOUT, height=520)
        st.plotly_chart(fig_scatter, use_container_width=True)

        st.markdown("### 💡 Key Insights")
        expensive_ratio = df_agg_filtered["Cost"].max() / max(df_agg_filtered["Cost"].min(), 1e-9)
        latency_ratio = df_agg_filtered["Latency"].max() / max(df_agg_filtered["Latency"].min(), 1e-9)

        st.markdown(
            f"""
            <div class="insight-box">
            💰 <b>{best_of(df_agg_filtered, 'Cost', ascending=True)['Model']}</b> is the most cost-effective at 
            <b>${df_agg_filtered['Cost'].min():.4f}/1K tokens</b> — {expensive_ratio:.0f}× cheaper than the most expensive option.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="insight-box">
            ⚡ <b>{best_of(df_agg_filtered, 'Latency', ascending=True)['Model']}</b> is the fastest at 
            <b>{df_agg_filtered['Latency'].min():.0f} ms</b> — {latency_ratio:.1f}× faster than the slowest model.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="insight-box">
            🎯 <b>{best_of(df_agg_filtered, 'Accuracy')['Model']}</b> leads in accuracy at 
            <b>{df_agg_filtered['Accuracy'].max():.1f}%</b> — ideal for critical tasks.
            </div>
            """,
            unsafe_allow_html=True,
        )

with tab3:
    st.subheader("🌍 AI Model Usage by Region")

    if df_filtered.empty:
        st.warning("No data available for selected filters")
    else:
        region_usage = df_filtered.groupby(["Region", "Model"], as_index=False)["Requests_per_day"].mean()

        fig_region = px.bar(
            region_usage,
            x="Region",
            y="Requests_per_day",
            color="Model",
            color_discrete_map=MODEL_COLORS,
            title="Average Daily Requests by Region",
            labels={"Requests_per_day": "Avg Requests/Day", "Region": ""},
            barmode="group",
        )
        fig_region.update_layout(**PLOTLY_LAYOUT, height=460)
        st.plotly_chart(fig_region, use_container_width=True)

        heatmap_data = region_usage.pivot(index="Model", columns="Region", values="Requests_per_day").fillna(0)
        fig_heatmap = px.imshow(
            heatmap_data,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="Blues",
            title="Request Volume Heatmap (Model × Region)",
            labels=dict(x="Region", y="Model", color="Avg Requests/Day"),
        )
        fig_heatmap.update_layout(**PLOTLY_LAYOUT, height=460)
        st.plotly_chart(fig_heatmap, use_container_width=True)

        c1, c2 = st.columns(2)

        with c1:
            region_total = df_filtered.groupby("Region", as_index=False)["Requests_per_day"].sum()
            fig_pie = px.pie(
                region_total,
                values="Requests_per_day",
                names="Region",
                title="Request Distribution by Region",
            )
            fig_pie.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig_pie, use_container_width=True)

        with c2:
            top_by_region = region_usage.loc[region_usage.groupby("Region")["Requests_per_day"].idxmax()].copy()
            top_by_region = top_by_region.rename(
                columns={"Region": "Region", "Model": "Top Model", "Requests_per_day": "Avg Daily Requests"}
            )
            top_by_region["Avg Daily Requests"] = top_by_region["Avg Daily Requests"].round(0).astype(int)
            st.dataframe(top_by_region, use_container_width=True, hide_index=True)

with tab4:
    st.subheader("🎯 Use Case Analysis")

    if df_filtered.empty:
        st.warning("No data available for selected filters")
    else:
        use_case_usage = df_filtered.groupby(["Use_Case", "Model"], as_index=False)["Requests_per_day"].mean()

        available_use_cases = sorted(use_case_usage["Use_Case"].unique().tolist())
        if not available_use_cases:
            st.warning("No use case data available")
        else:
            selected_use_case = st.selectbox("Select Use Case for Detailed Analysis", available_use_cases)
            uc_filtered = use_case_usage[use_case_usage["Use_Case"] == selected_use_case].sort_values("Requests_per_day", ascending=True)

            fig_uc = px.bar(
                uc_filtered,
                x="Requests_per_day",
                y="Model",
                orientation="h",
                color="Model",
                color_discrete_map=MODEL_COLORS,
                title=f"Model Popularity for {selected_use_case}",
                labels={"Requests_per_day": "Avg Daily Requests", "Model": ""},
                text=uc_filtered["Requests_per_day"].round(0).astype(int),
            )
            fig_uc.update_traces(textposition="outside", cliponaxis=False)
            fig_uc.update_layout(**PLOTLY_LAYOUT, showlegend=False, height=500)
            st.plotly_chart(fig_uc, use_container_width=True)

            if not uc_filtered.empty:
                winner = uc_filtered.iloc[-1]
                winner_details = df_agg_filtered[df_agg_filtered["Model"] == winner["Model"]].iloc[0]

                st.markdown(
                    f"""
                    <div style="background: linear-gradient(135deg, #1a2a3a, #0a1a2a); padding: 20px; border-radius: 12px; margin: 20px 0; border: 1px solid rgba(255,255,255,.08);">
                        <h3>🏆 Best Model for {selected_use_case}</h3>
                        <h2 style="color: #FFD700; margin: 10px 0;">{winner['Model']}</h2>
                        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-top: 15px;">
                            <div><div style="font-size: 12px; color: #888;">Accuracy</div><div style="font-size: 24px; font-weight: bold;">{winner_details['Accuracy']:.1f}%</div></div>
                            <div><div style="font-size: 12px; color: #888;">Cost/1K</div><div style="font-size: 24px; font-weight: bold;">${winner_details['Cost']:.4f}</div></div>
                            <div><div style="font-size: 12px; color: #888;">Latency</div><div style="font-size: 24px; font-weight: bold;">{winner_details['Latency']:.0f} ms</div></div>
                            <div><div style="font-size: 12px; color: #888;">Value Score</div><div style="font-size: 24px; font-weight: bold;">{winner_details['Score']:.1f}</div></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.subheader("Model Popularity Across All Use Cases")
            fig_all_uc = px.bar(
                use_case_usage,
                x="Use_Case",
                y="Requests_per_day",
                color="Model",
                color_discrete_map=MODEL_COLORS,
                title="Average Daily Requests by Use Case",
                labels={"Requests_per_day": "Avg Requests/Day", "Use_Case": ""},
                barmode="group",
            )
            fig_all_uc.update_layout(**PLOTLY_LAYOUT, height=500, xaxis_tickangle=-45)
            st.plotly_chart(fig_all_uc, use_container_width=True)

            st.dataframe(data["winners"], use_container_width=True, hide_index=True)

with tab5:
    st.subheader("🏅 Find Your Ideal AI Model")
    st.markdown("Adjust the sliders to find the model that best matches your priorities")

    if df_agg_filtered.empty:
        st.warning("No data available for selected filters")
    else:
        c1, c2 = st.columns([1, 2])

        with c1:
            st.markdown("#### Set Your Priorities")
            weight_accuracy = st.slider("🎯 Accuracy", 0, 10, 7)
            weight_cost = st.slider("💰 Low Cost", 0, 10, 5)
            weight_speed = st.slider("⚡ Speed", 0, 10, 4)
            weight_reliability = st.slider("🔒 Reliability (Uptime)", 0, 10, 3)
            weight_throughput = st.slider("📈 Throughput", 0, 10, 2)

        total_weight = weight_accuracy + weight_cost + weight_speed + weight_reliability + weight_throughput

        if total_weight == 0:
            st.warning("Set at least one priority weight")
        else:
            df_scored = model_picker_score(
                df_agg_filtered,
                weight_accuracy,
                weight_cost,
                weight_speed,
                weight_reliability,
                weight_throughput,
            ).sort_values("custom_score", ascending=False).reset_index(drop=True)

            top_model = df_scored.iloc[0]

            with c2:
                st.markdown(
                    f"""
                    <div style="background: linear-gradient(135deg, #1a2f3f, #0a1a2a); border: 2px solid #00C6FF; border-radius: 16px; padding: 25px; margin-bottom: 20px;">
                        <div style="font-size: 12px; color: #00C6FF; letter-spacing: 2px; margin-bottom: 10px;">🎯 RECOMMENDED FOR YOU</div>
                        <div style="font-size: 32px; font-weight: bold; color: #FFD700; margin-bottom: 20px;">{top_model['Model']}</div>
                        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px;">
                            <div><div style="font-size: 11px; color: #888;">Accuracy</div><div style="font-size: 20px; font-weight: bold;">{top_model['Accuracy']:.1f}%</div></div>
                            <div><div style="font-size: 11px; color: #888;">Cost/1K</div><div style="font-size: 20px; font-weight: bold;">${top_model['Cost']:.4f}</div></div>
                            <div><div style="font-size: 11px; color: #888;">Latency</div><div style="font-size: 20px; font-weight: bold;">{top_model['Latency']:.0f} ms</div></div>
                        </div>
                        <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid #2a2a4a;">
                            <div style="font-size: 14px; color: #00FF88;">Match Score: {top_model['custom_score']:.1f} / 100</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("#### Full Model Ranking")
            ranking_df = df_scored[["Model", "Accuracy", "Cost", "Latency", "Score", "custom_score"]].copy()
            ranking_df["Accuracy"] = ranking_df["Accuracy"].map(lambda x: f"{x:.1f}%")
            ranking_df["Cost"] = ranking_df["Cost"].map(lambda x: f"${x:.4f}")
            ranking_df["Latency"] = ranking_df["Latency"].map(lambda x: f"{int(round(x))} ms")
            ranking_df["Score"] = ranking_df["Score"].map(lambda x: f"{x:.1f}")
            ranking_df["custom_score"] = ranking_df["custom_score"].map(lambda x: f"{x:.1f}")
            ranking_df.index = range(1, len(ranking_df) + 1)

            st.dataframe(ranking_df, use_container_width=True, hide_index=False)

            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=float(top_model["custom_score"]),
                    title={"text": f"{top_model['Model']} - Match Score"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#00C6FF"},
                        "steps": [
                            {"range": [0, 33], "color": "#331a1a"},
                            {"range": [33, 66], "color": "#1a331a"},
                            {"range": [66, 100], "color": "#1a2a33"},
                        ],
                        "threshold": {
                            "line": {"color": "#FFD700", "width": 4},
                            "thickness": 0.75,
                            "value": float(top_model["custom_score"]),
                        },
                    },
                )
            )
            fig_gauge.update_layout(**PLOTLY_LAYOUT, height=320)
            st.plotly_chart(fig_gauge, use_container_width=True)

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; padding: 20px;'>🤖 AI Model Analytics Dashboard | Powered by Real-World Deployment Data</div>",
    unsafe_allow_html=True,
)