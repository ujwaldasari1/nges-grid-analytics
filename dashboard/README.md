# Dashboard extracts

Aggregated, BI-ready CSVs produced by `scripts/build_charts_and_extracts.py`:

- `demand_hourly_enriched.csv` - all 25,000 hourly records + Season/Weekend/TempBand/AbsError columns
- `demand_by_region_month.csv` - monthly rollup per region (avg/max demand, peak events, MAE)
- `infrastructure_risk_scored.csv` - 500 substations + RiskTier, UtilizationBand, PriorityRank
- `dr_events_scored.csv` - 8,000 DR events + CostPerMW, EfficiencyQuartile, ReductionPct
- `kpi_summary.csv` - one-row headline KPIs

The Streamlit app (`app.py`) computes from the raw data directly; these extracts exist for
anyone who wants to build a Power BI / Tableau version.
