# Builds all chart PNGs (figures/) and Power BI dashboard extracts (dashboard/)
# from the three NGES datasets in data/. Usage: python scripts/build_charts_and_extracts.py
import pandas as pd, numpy as np, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "data")       # input CSVs
CHARTS = os.path.join(ROOT, "figures")  # chart PNG output
DASH = os.path.join(ROOT, "dashboard")  # Power BI extract output
for p in [CHARTS, DASH]:
    os.makedirs(p, exist_ok=True)

# ---- palette (validated default, light mode) ----
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
CRIT, SERIOUS = "#d03b3b", "#ec835a"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"]
SURF, INK, SEC, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "font.family": "Segoe UI", "figure.facecolor": SURF, "axes.facecolor": SURF,
    "axes.edgecolor": AXIS, "axes.labelcolor": SEC, "xtick.color": MUTED,
    "ytick.color": MUTED, "text.color": INK, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "font.size": 11, "figure.dpi": 200})

def style(ax, ygrid_only=True):
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    if ygrid_only:
        ax.grid(axis="y"); ax.grid(axis="x", visible=False)

def save(fig, name):
    fig.savefig(os.path.join(CHARTS, name), bbox_inches="tight", facecolor=SURF)
    plt.close(fig); print("chart:", name)

# ---- load data ----
d1 = pd.read_csv(os.path.join(BASE, "NGES Energy Demand Forecasting Dataset.csv"), encoding="utf-8-sig")
d2 = pd.read_csv(os.path.join(BASE, "NGES Infrastructure Risk & Planning Dataset.csv"), encoding="utf-8-sig")
d3 = pd.read_csv(os.path.join(BASE, "NGES Demand Response Effectiveness Dataset.csv"), encoding="utf-8-sig")
d1["Timestamp"] = pd.to_datetime(d1["Timestamp"])

# 1. Peak demand events by year (from engagement packet)
fig, ax = plt.subplots(figsize=(6.4, 3.6))
yrs, ev = ["2022", "2023", "2024", "2025"], [14, 21, 29, 34]
bars = ax.bar(yrs, ev, width=0.55, color=BLUE, zorder=3)
for b, v in zip(bars, ev):
    ax.text(b.get_x() + b.get_width()/2, v + 0.7, str(v), ha="center", fontsize=12, color=INK, fontweight="bold")
ax.set_ylabel("Peak demand events"); ax.set_ylim(0, 40); style(ax)
ax.set_title("Peak demand events have grown 2.4x in four years", loc="left", fontsize=13, fontweight="bold", color=INK, pad=12)
save(fig, "01_peak_events_by_year.png")

# 2. Demand distribution with peak threshold
thresh = d1[d1.PeakDemandEvent == 1].ActualDemandMW.min()
fig, ax = plt.subplots(figsize=(6.4, 3.6))
ax.hist(d1.ActualDemandMW, bins=60, color=BLUE, zorder=3)
ax.axvline(thresh, color=CRIT, lw=2)
ax.text(thresh + 12, ax.get_ylim()[1]*0.9, f"Peak-event threshold\n{thresh:,.0f} MW (top 10% of hours)", color=CRIT, fontsize=10, va="top")
ax.set_xlabel("Hourly system demand (MW)"); ax.set_ylabel("Hours"); style(ax)
ax.set_title("Peak events are the top decile of hourly demand", loc="left", fontsize=13, fontweight="bold", color=INK, pad=12)
save(fig, "02_demand_distribution.png")

# 3. What drives demand: RF feature importance (from model run)
fig, ax = plt.subplots(figsize=(6.8, 3.6))
labels = ["Residential load", "Commercial load", "Industrial load", "All weather variables", "Calendar (hour/month/day)", "Customer count"]
vals = [51.7, 27.6, 18.6, 0.9, 0.7, 0.3]
colors = [BLUE, BLUE, BLUE, MUTED, MUTED, MUTED]
y = np.arange(len(labels))[::-1]
ax.barh(y, vals, height=0.55, color=colors, zorder=3)
for yi, v in zip(y, vals):
    ax.text(v + 0.8, yi, f"{v:.1f}%", va="center", fontsize=10, color=INK)
ax.set_yticks(y, labels); ax.set_xlabel("Share of model feature importance"); ax.set_xlim(0, 60)
style(ax); ax.grid(axis="x"); ax.grid(axis="y", visible=False)
ax.set_title("Customer segment loads explain demand; weather does not", loc="left", fontsize=13, fontweight="bold", color=INK, pad=12)
save(fig, "03_demand_drivers.png")

# 4. Temperature vs demand scatter (the null result, shown honestly)
s = d1.dropna(subset=["Temperature"]).sample(3000, random_state=7)
fig, ax = plt.subplots(figsize=(6.4, 3.6))
ax.scatter(s.Temperature, s.ActualDemandMW, s=6, color=AQUA, alpha=0.35, zorder=3, edgecolors="none")
z = np.polyfit(s.Temperature, s.ActualDemandMW, 1)
xs = np.linspace(s.Temperature.min(), s.Temperature.max(), 50)
ax.plot(xs, np.polyval(z, xs), color=INK, lw=2)
ax.set_xlabel("Temperature (F)"); ax.set_ylabel("Hourly demand (MW)"); style(ax)
ax.grid(axis="both")
ax.set_title("Temperature vs demand: correlation 0.002 - no relationship", loc="left", fontsize=13, fontweight="bold", color=INK, pad=12)
save(fig, "04_temp_vs_demand.png")

# 5. Model comparison (MAE, lower is better)
fig, ax = plt.subplots(figsize=(6.4, 3.6))
names = ["Current NGES\nforecast", "Weather-only\nmodel (baseline)", "Segment-load\nlinear model", "Segment-load\nrandom forest"]
maes = [24.0, 113.5, 19.9, 24.4]
cols = [MUTED, SERIOUS, BLUE, SEQ[2]]
bars = ax.bar(names, maes, width=0.55, color=cols, zorder=3)
for b, v in zip(bars, maes):
    ax.text(b.get_x() + b.get_width()/2, v + 2.2, f"{v:.1f}", ha="center", fontsize=11, color=INK, fontweight="bold")
ax.set_ylabel("Mean absolute error (MW), 2025 hold-out"); ax.set_ylim(0, 130); style(ax)
ax.set_title("A simple segment-load model beats the current forecast by 17%", loc="left", fontsize=13, fontweight="bold", color=INK, pad=12)
save(fig, "05_model_comparison.png")

# 6. Infrastructure: age vs risk, critical tier highlighted
fig, ax = plt.subplots(figsize=(6.4, 3.6))
lo = d2[d2.RiskScore < 70]; hi = d2[d2.RiskScore >= 70]
ax.scatter(lo.InfrastructureAgeYears, lo.RiskScore, s=14, color=BLUE, alpha=0.45, edgecolors="none", zorder=3, label="Substations")
ax.scatter(hi.InfrastructureAgeYears, hi.RiskScore, s=26, color=CRIT, zorder=4, label="Critical tier (risk >= 70): 31 assets")
ax.axhline(70, color=CRIT, lw=1, ls="--", alpha=0.6)
ax.set_xlabel("Infrastructure age (years)"); ax.set_ylabel("Composite risk score"); style(ax); ax.grid(axis="both")
ax.legend(frameon=False, loc="lower right", fontsize=9)
ax.set_title("Age is the strongest risk driver (r = 0.52); 31 assets are critical", loc="left", fontsize=13, fontweight="bold", color=INK, pad=12)
save(fig, "06_infra_risk.png")

# 7. Utilization bands
bands = pd.cut(d2.CurrentUtilizationPct, [0, 70, 85, 90, 100, 999],
               labels=["under 70%", "70-85%", "85-90%", "90-100%", "over 100%"])
cnt = bands.value_counts().reindex(["under 70%", "70-85%", "85-90%", "90-100%", "over 100%"])
fig, ax = plt.subplots(figsize=(6.4, 3.6))
cols = [SEQ[1], SEQ[2], SEQ[3], SERIOUS, CRIT]
bars = ax.bar(cnt.index.astype(str), cnt.values, width=0.55, color=cols, zorder=3)
for b, v in zip(bars, cnt.values):
    ax.text(b.get_x() + b.get_width()/2, v + 4, str(v), ha="center", fontsize=11, color=INK, fontweight="bold")
ax.set_ylabel("Substations"); style(ax)
ax.set_title("162 of 500 substations run above 85% of rated capacity", loc="left", fontsize=13, fontweight="bold", color=INK, pad=12)
save(fig, "07_utilization_bands.png")

# 8. DR cost-effectiveness by quartile
d3["cpm"] = d3.ProgramCost / d3.DemandReductionMW
d3["quartile"] = pd.qcut(d3.cpm, 4, labels=["Q1 (best)", "Q2", "Q3", "Q4 (worst)"])
q = d3.groupby("quartile", observed=True).cpm.mean()
fig, ax = plt.subplots(figsize=(6.4, 3.6))
cols = [BLUE, SEQ[2], SERIOUS, CRIT]
bars = ax.bar(q.index.astype(str), q.values, width=0.55, color=cols, zorder=3)
for b, v in zip(bars, q.values):
    ax.text(b.get_x() + b.get_width()/2, v + 30, f"${v:,.0f}", ha="center", fontsize=11, color=INK, fontweight="bold")
ax.set_ylabel("Avg cost per MW reduced ($)"); style(ax)
ax.set_title("The worst DR quartile costs 11x more per MW than the best", loc="left", fontsize=13, fontweight="bold", color=INK, pad=12)
save(fig, "08_dr_cost_quartiles.png")

# 9. DR savings opportunity
fig, ax = plt.subplots(figsize=(6.4, 3.6))
names = ["Actual program spend\n(8,000 events)", "Same MW at median\ncost-effectiveness"]
vals = [420.2, 321.7]
bars = ax.bar(names, vals, width=0.45, color=[MUTED, BLUE], zorder=3)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width()/2, v + 8, f"${v:,.0f}M", ha="center", fontsize=12, color=INK, fontweight="bold")
ax.set_ylabel("Program cost ($M)"); ax.set_ylim(0, 480); style(ax)
ax.set_title("Rebalancing the worst quartile frees \\$98.5M with zero MW lost", loc="left", fontsize=13, fontweight="bold", color=INK, pad=12)
save(fig, "09_dr_savings.png")

# 10. ROI scenarios (annual value)
fig, ax = plt.subplots(figsize=(6.8, 3.6))
scen = ["Conservative", "Expected", "Optimistic"]
dr = [22.1, 34.8, 40.0]; infra = [6.0, 10.0, 15.0]; fc = [0.5, 1.0, 2.0]
x = np.arange(3)
b1 = ax.bar(x, dr, 0.5, color=BLUE, zorder=3, label="DR optimization")
b2 = ax.bar(x, infra, 0.5, bottom=dr, color=ORANGE, zorder=3, label="Risk-based capital targeting")
b3 = ax.bar(x, fc, 0.5, bottom=np.array(dr)+np.array(infra), color=AQUA, zorder=3, label="Forecast upgrade")
tot = np.array(dr) + np.array(infra) + np.array(fc)
for xi, t in zip(x, tot):
    ax.text(xi, t + 1.2, f"${t:,.1f}M/yr", ha="center", fontsize=11, color=INK, fontweight="bold")
ax.set_xticks(x, scen); ax.set_ylabel("Estimated annual value ($M)"); ax.set_ylim(0, 64); style(ax)
ax.legend(frameon=False, fontsize=9, loc="upper left")
ax.set_title("Expected value \\$45.8M per year against ~\\$5M annual run cost", loc="left", fontsize=13, fontweight="bold", color=INK, pad=12)
save(fig, "10_roi_scenarios.png")

# ---------- Power BI extracts ----------
h = d1.copy()
h["Date"] = h.Timestamp.dt.date; h["Hour"] = h.Timestamp.dt.hour
h["Month"] = h.Timestamp.dt.to_period("M").astype(str)
h["Season"] = h.Timestamp.dt.month.map({12:"Winter",1:"Winter",2:"Winter",3:"Spring",4:"Spring",5:"Spring",6:"Summer",7:"Summer",8:"Summer",9:"Fall",10:"Fall",11:"Fall"})
h["Weekend"] = (h.Timestamp.dt.dayofweek >= 5).astype(int)
h["AbsErrorMW"] = (h.ActualDemandMW - h.ForecastedDemandMW).abs()
h["TempBand"] = pd.cut(h.Temperature, [-100, 32, 60, 80, 95, 200], labels=["Below 32F", "32-60F", "60-80F", "80-95F", "Above 95F"])
h.to_csv(os.path.join(DASH, "demand_hourly_enriched.csv"), index=False)

m = h.groupby(["Month", "Region"], as_index=False).agg(
    AvgDemandMW=("ActualDemandMW", "mean"), MaxDemandMW=("ActualDemandMW", "max"),
    PeakEvents=("PeakDemandEvent", "sum"), DREvents=("DemandResponseEvent", "sum"),
    MAE=("AbsErrorMW", "mean"), Hours=("ActualDemandMW", "size"))
m.to_csv(os.path.join(DASH, "demand_by_region_month.csv"), index=False)

i = d2.copy()
i["RiskTier"] = pd.cut(i.RiskScore, [0, 40, 55, 70, 100.01], right=False,
                       labels=["Low", "Moderate", "Elevated", "Critical"])
i["UtilizationBand"] = pd.cut(i.CurrentUtilizationPct, [0, 70, 85, 90, 100, 999], labels=["Under 70%", "70-85%", "85-90%", "90-100%", "Over 100%"])
i["PriorityRank"] = i.RiskScore.rank(ascending=False).astype(int)
i.to_csv(os.path.join(DASH, "infrastructure_risk_scored.csv"), index=False)

e = d3.copy()
e["CostPerMW"] = (e.ProgramCost / e.DemandReductionMW).round(2)
e["EfficiencyQuartile"] = pd.qcut(e.CostPerMW, 4, labels=["Q1 best", "Q2", "Q3", "Q4 worst"])
e["ReductionPct"] = (e.DemandReductionMW / e.PeakDemandBeforeMW * 100).round(2)
e.to_csv(os.path.join(DASH, "dr_events_scored.csv"), index=False)

kpi = pd.DataFrame([{
    "PeakDemandEvents": int(d1.PeakDemandEvent.sum()),
    "ForecastMAPE_Pct": 3.32, "ProposedModelMAPE_Pct": 2.75,
    "AvgRiskScore": round(d2.RiskScore.mean(), 1), "CriticalSubstations": int((d2.RiskScore >= 70).sum()),
    "SubstationsOver90Util": int((d2.CurrentUtilizationPct > 90).sum()),
    "TotalDRReductionMW": round(d3.DemandReductionMW.sum()), "TotalDRSpendM": round(d3.ProgramCost.sum()/1e6, 1),
    "MedianCostPerMW": round(d3.ProgramCost.div(d3.DemandReductionMW).median(), 0),
    "DRSavingsOpportunityM": 98.5, "ExpectedAnnualValueM": 45.8}])
kpi.to_csv(os.path.join(DASH, "kpi_summary.csv"), index=False)
print("dashboard extracts written to", DASH)
