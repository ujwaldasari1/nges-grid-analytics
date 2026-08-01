# NGES Grid Analytics - interactive dashboard
# Explores demand drivers, infrastructure risk, and demand response ROI for a
# simulated regional utility (NorthGrid Energy Services).
# Run locally:  streamlit run app.py
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------- palette (consistent with the written reports) ----------
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
RED, SERIOUS, MUTED = "#d03b3b", "#ec835a", "#898781"
SEQ = ["#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"]

st.set_page_config(page_title="NGES Grid Analytics", page_icon="⚡", layout="wide")

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

@st.cache_data
def load():
    d1 = pd.read_csv(os.path.join(DATA, "NGES Energy Demand Forecasting Dataset.csv"), encoding="utf-8-sig")
    d2 = pd.read_csv(os.path.join(DATA, "NGES Infrastructure Risk & Planning Dataset.csv"), encoding="utf-8-sig")
    d3 = pd.read_csv(os.path.join(DATA, "NGES Demand Response Effectiveness Dataset.csv"), encoding="utf-8-sig")
    d1["Timestamp"] = pd.to_datetime(d1["Timestamp"])
    d2["RiskTier"] = pd.cut(d2.RiskScore, [0, 40, 55, 70, 100],
                            labels=["Low", "Moderate", "Elevated", "Critical"])
    d3["CostPerMW"] = d3.ProgramCost / d3.DemandReductionMW
    d3["Quartile"] = pd.qcut(d3.CostPerMW, 4, labels=["Q1 (best)", "Q2", "Q3", "Q4 (worst)"])
    return d1, d2, d3

demand, infra, dr = load()

def style(fig, showlegend=False):
    fig.update_layout(
        template="plotly_white", showlegend=showlegend,
        margin=dict(l=10, r=10, t=48, b=10), height=380,
        font=dict(family="Segoe UI, sans-serif", color="#52514e"),
        title_font=dict(size=16, color="#0b0b0b"),
        plot_bgcolor="#fcfcfb", paper_bgcolor="#fcfcfb")
    fig.update_xaxes(gridcolor="#e1e0d9", zerolinecolor="#c3c2b7")
    fig.update_yaxes(gridcolor="#e1e0d9", zerolinecolor="#c3c2b7")
    return fig

# ---------- header + KPI row ----------
st.title("⚡ NGES Grid Analytics")
st.caption("Why peak demand events keep rising at a regional utility - and the $45.8M/yr answer. "
           "Analysis by Ujwal Dasari. Simulated utility data (25,000 hourly records, 500 substations, 8,000 demand response events).")

k = st.columns(5)
k[0].metric("Peak demand events (hours)", "2,500", "top 10% of all hours")
k[1].metric("Forecast error (MAPE)", "3.3%", "-17% with proposed model", delta_color="inverse")
k[2].metric("Critical substations", "31 / 500", "10 North, 8 West", delta_color="off")
k[3].metric("DR savings opportunity", "$98.5M", "same MW, better execution")
k[4].metric("Expected annual value", "$45.8M/yr", "payback < 1 year")

tab1, tab2, tab3, tab4 = st.tabs(["🔌 Demand & Forecasting", "🏗️ Infrastructure Risk",
                                  "🔋 Demand Response", "💰 Recommendations & ROI"])

# ---------- TAB 1: demand ----------
with tab1:
    c1, c2 = st.columns(2)
    with c1:
        s = demand.dropna(subset=["Temperature"]).sample(3000, random_state=7)
        fig = px.scatter(s, x="Temperature", y="ActualDemandMW", opacity=0.35,
                         title="Temperature vs demand: correlation 0.002 - weather is not the driver",
                         labels={"Temperature": "Temperature (F)", "ActualDemandMW": "Hourly demand (MW)"})
        fig.update_traces(marker=dict(color=AQUA, size=5))
        st.plotly_chart(style(fig), use_container_width=True)
    with c2:
        s2 = demand.sample(3000, random_state=7).assign(
            LoadSum=lambda d: d.ResidentialLoadMW + d.CommercialLoadMW + d.IndustrialLoadMW)
        fig = px.scatter(s2, x="LoadSum", y="ActualDemandMW", opacity=0.35,
                         title="Customer segment loads vs demand: correlation 0.984 - this is the driver",
                         labels={"LoadSum": "Residential + Commercial + Industrial load (MW)",
                                 "ActualDemandMW": "Hourly demand (MW)"})
        fig.update_traces(marker=dict(color=BLUE, size=5))
        st.plotly_chart(style(fig), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        thresh = demand.loc[demand.PeakDemandEvent == 1, "ActualDemandMW"].min()
        fig = px.histogram(demand, x="ActualDemandMW", nbins=60,
                           title="Peak events are simply the top decile of demand",
                           labels={"ActualDemandMW": "Hourly demand (MW)"})
        fig.update_traces(marker_color=BLUE)
        fig.add_vline(x=thresh, line_color=RED, line_width=2,
                      annotation_text=f"Peak threshold {thresh:,.0f} MW", annotation_font_color=RED)
        st.plotly_chart(style(fig), use_container_width=True)
    with c4:
        comp = pd.DataFrame({
            "Model": ["Current NGES forecast", "Weather-only model", "Segment-load linear model", "Random forest"],
            "MAE (MW)": [24.0, 113.5, 19.9, 24.4]})
        fig = px.bar(comp, x="Model", y="MAE (MW)", text="MAE (MW)",
                     title="A simple segment-load model beats the current forecast by 17%")
        fig.update_traces(marker_color=[MUTED, SERIOUS, BLUE, SEQ[1]],
                          texttemplate="%{text:.1f}", textposition="outside")
        st.plotly_chart(style(fig), use_container_width=True)
    st.info("**Finding:** demand volatility is a customer-load story, not a weather story. The winning "
            "forecast model is simple and fully explainable - and because peak events are just the top 10% "
            "of demand hours, an accurate forecast doubles as a peak-event early-warning system "
            "(88.6% precision / 86% recall on held-out 2025 data).")

# ---------- TAB 2: infrastructure ----------
with tab2:
    regions = st.multiselect("Filter regions", sorted(infra.Region.unique()), default=[])
    inf = infra if not regions else infra[infra.Region.isin(regions)]
    c1, c2 = st.columns(2)
    with c1:
        fig = px.scatter(inf, x="CurrentUtilizationPct", y="RiskScore",
                         color=(inf.RiskTier == "Critical").map({True: "Critical (risk ≥ 70)", False: "Other"}),
                         color_discrete_map={"Critical (risk ≥ 70)": RED, "Other": BLUE},
                         hover_data=["SubstationID", "Region", "InfrastructureAgeYears", "OutagesLast5Years"],
                         title="Utilization vs risk - critical tier in red",
                         labels={"CurrentUtilizationPct": "Current utilization (%)", "RiskScore": "Composite risk score", "color": ""})
        fig.add_vline(x=100, line_dash="dash", line_color=MUTED)
        st.plotly_chart(style(fig, showlegend=True), use_container_width=True)
    with c2:
        bands = pd.cut(inf.CurrentUtilizationPct, [0, 70, 85, 90, 100, 999],
                       labels=["under 70%", "70-85%", "85-90%", "90-100%", "over 100%"]).value_counts().sort_index()
        fig = px.bar(x=bands.index.astype(str), y=bands.values, text=bands.values,
                     title="Substations by utilization band",
                     labels={"x": "Utilization band", "y": "Substations"})
        fig.update_traces(marker_color=[SEQ[0], SEQ[1], SEQ[2], SERIOUS, RED], textposition="outside")
        st.plotly_chart(style(fig), use_container_width=True)
    st.markdown("**Top capital priorities** (risk score blended with projected load growth)")
    prio = (inf[inf.RiskScore >= 70]
            .assign(PriorityScore=lambda d: (0.7 * d.RiskScore + 3 * d.ProjectedLoadGrowthPct).round(1))
            .sort_values("PriorityScore", ascending=False)
            [["SubstationID", "Region", "InfrastructureAgeYears", "CurrentUtilizationPct",
              "OutagesLast5Years", "ProjectedLoadGrowthPct", "RiskScore", "PriorityScore"]]
            .head(15))
    st.dataframe(prio, use_container_width=True, hide_index=True)
    st.info("**Finding:** risk is concentrated, not spread - 31 of 500 substations are critical "
            "(load growth r = 0.55 and age r = 0.52 are the top drivers) and 35 already run past 100% of rated capacity. "
            "A two-speed plan funds itself: upgrade the critical 31 in ranked order, monitor the rest, "
            "and defer $100-150M of lower-priority spend.")

# ---------- TAB 3: demand response ----------
with tab3:
    c1, c2 = st.columns(2)
    with c1:
        q = dr.groupby("Quartile", observed=True).CostPerMW.mean().round(0)
        fig = px.bar(x=q.index.astype(str), y=q.values, text=[f"${v:,.0f}" for v in q.values],
                     title="Cost per MW by execution quartile - the worst is 11x the best",
                     labels={"x": "Cost-effectiveness quartile", "y": "Avg cost per MW reduced ($)"})
        fig.update_traces(marker_color=[BLUE, SEQ[1], SERIOUS, RED], textposition="outside")
        st.plotly_chart(style(fig), use_container_width=True)
    with c2:
        med = dr.CostPerMW.median()
        worst = dr[dr.Quartile == "Q4 (worst)"]
        vals = [dr.ProgramCost.sum() / 1e6,
                (dr.ProgramCost.sum() - worst.ProgramCost.sum() + worst.DemandReductionMW.sum() * med) / 1e6]
        fig = px.bar(x=["Actual program spend", "Same MW at median cost-effectiveness"], y=vals,
                     text=[f"${v:,.0f}M" for v in vals],
                     title="Rebalancing the worst quartile frees $98.5M - zero MW lost",
                     labels={"x": "", "y": "Program cost ($M)"})
        fig.update_traces(marker_color=[MUTED, BLUE], textposition="outside")
        st.plotly_chart(style(fig), use_container_width=True)
    ct = st.columns(3)
    for i, (t, g) in enumerate(dr.groupby("CustomerType")):
        ct[i].metric(f"{t} median cost/MW", f"${g.CostPerMW.median():,.0f}",
                     f"{(g.DemandReductionMW / g.PeakDemandBeforeMW).mean() * 100:.1f}% avg peak reduction",
                     delta_color="off")
    st.info("**Finding:** every customer type performs the same (median ≈ $330/MW, ~11% peak reduction), "
            "so the 640x spread between the cheapest and most expensive events is an execution problem - "
            "event timing, sizing, and incentives - not a recruitment problem. That makes it fixable "
            "from inside the existing budget.")

# ---------- TAB 4: recommendations ----------
with tab4:
    st.subheader("Three moves, in priority order")
    r1, r2, r3 = st.columns(3)
    with r1:
        st.markdown(f"##### :orange[1. Fix demand response execution]")
        st.markdown("Standardize event sizing, timing, and incentives against the program's own median "
                    "cost-effectiveness. Retire or redesign chronically inefficient events.\n\n"
                    "**Worth $22-40M/yr** (expected $34.8M)")
    with r2:
        st.markdown(f"##### :blue[2. Target capital at the critical 31]")
        st.markdown("Upgrade the 31 critical substations over 3-5 years in ranked order; cover the "
                    "elevated tier with monitoring and DR instead of immediate replacement.\n\n"
                    "**Worth $6-15M/yr** (expected $10M)")
    with r3:
        st.markdown(f"##### :green[3. Upgrade forecasting & peak warning]")
        st.markdown("Replace the forecast with the segment-load model fed by AMI data; trigger DR "
                    "automatically when forecasts approach the peak threshold.\n\n"
                    "**Worth $0.5-2M/yr** - and it powers moves 1 and 2")
    scen = pd.DataFrame({
        "Scenario": ["Conservative", "Expected", "Optimistic"],
        "DR optimization": [22.1, 34.8, 40.0],
        "Capital targeting": [6.0, 10.0, 15.0],
        "Forecast upgrade": [0.5, 1.0, 2.0]})
    fig = go.Figure()
    for col, color in [("DR optimization", BLUE), ("Capital targeting", ORANGE), ("Forecast upgrade", AQUA)]:
        fig.add_bar(x=scen.Scenario, y=scen[col], name=col, marker_color=color)
    totals = scen[["DR optimization", "Capital targeting", "Forecast upgrade"]].sum(axis=1)
    fig.add_trace(go.Scatter(x=scen.Scenario, y=totals + 2.5, mode="text",
                             text=[f"${t:,.1f}M/yr" for t in totals],
                             textfont=dict(color="#0b0b0b", size=14), showlegend=False))
    fig.update_layout(barmode="stack", title="Estimated annual value by scenario")
    st.plotly_chart(style(fig, showlegend=True), use_container_width=True)
    st.success("**The bottom line:** ~$8M one-time plus ~$5M/yr to run, against $45.8M/yr of expected "
               "value - payback in under a year. Over a 3-year planning window the expected case creates "
               "~$137M of value, more than neutralizing the client's stated $95-120M exposure.")
    st.caption("Assumptions: 2.83-year data window annualization; 7-8% cost of capital on deferrals; "
               "$100-200/MWh emergency power premium; DR savings priced at the program's own median of $330/MW. "
               "Full methodology in the analysis/ folder of the repo.")
