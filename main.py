"""
main.py — SA Business Intelligence Suite
=========================================
Author  : Dumisani Abrahm Baloyi
GitHub  : https://github.com/Dvmisani

Runs the full analytics pipeline:
  1. Generate / load transaction data
  2. RFM Customer Segmentation
  3. Revenue Forecasting
  4. Anomaly Detection
  5. Regional Heatmap
  6. Category Performance

All charts are saved to /output/
"""

import os, sys
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from generate_data    import generate_transactions
from rfm_analysis     import compute_rfm, plot_rfm
from forecasting      import build_forecast, plot_forecast
from visuals          import (plot_anomaly, plot_regional_heatmap,
                               plot_category_performance)

OUTPUT_DIR = "output"
DATA_PATH  = "data/transactions.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("data",     exist_ok=True)

# ── Banner ─────────────────────────────────────────────────────────────────
print("\n" + "="*62)
print("   SA BUSINESS INTELLIGENCE SUITE")
print("   Author: Dumisani Abrahm Baloyi  |  github.com/Dvmisani")
print("="*62)

# ── Step 1: Data ───────────────────────────────────────────────────────────
if not os.path.exists(DATA_PATH):
    print("\n[1/5] Generating synthetic transaction data...")
    df = generate_transactions()
    df.to_csv(DATA_PATH, index=False)
else:
    print("\n[1/5] Loading transaction data...")
    df = pd.read_csv(DATA_PATH)

print(f"      {len(df):,} transactions | "
      f"{df['customer_id'].nunique()} customers | "
      f"{df['province'].nunique()} provinces")

# ── Step 2: RFM ────────────────────────────────────────────────────────────
print("\n[2/5] Running RFM customer segmentation...")
rfm = compute_rfm(df)
seg_counts = rfm["Segment"].value_counts()
for seg, cnt in seg_counts.items():
    pct = cnt / len(rfm) * 100
    print(f"      {seg:<22} {cnt:>3} customers  ({pct:.0f}%)")
path = plot_rfm(rfm, OUTPUT_DIR)
print(f"      Chart saved → {path}")

# ── Step 3: Forecast ───────────────────────────────────────────────────────
print("\n[3/5] Building revenue forecast...")
monthly, forecast_df = build_forecast(df)
avg_actual   = monthly["revenue"].mean()
avg_forecast = forecast_df["yhat"].mean()
growth       = (avg_forecast - avg_actual) / avg_actual * 100
print(f"      Avg actual monthly revenue  : R{avg_actual:>10,.0f}")
print(f"      Avg forecast (next 6 months): R{avg_forecast:>10,.0f}  ({growth:+.1f}%)")
path = plot_forecast(monthly, forecast_df, OUTPUT_DIR)
print(f"      Chart saved → {path}")

# ── Step 4: Anomaly Detection ──────────────────────────────────────────────
print("\n[4/5] Detecting revenue anomalies...")
path = plot_anomaly(df, OUTPUT_DIR)
print(f"      Chart saved → {path}")

# ── Step 5: Regional Heatmap ───────────────────────────────────────────────
print("\n[5/5] Generating regional & category visuals...")
path = plot_regional_heatmap(df, OUTPUT_DIR)
print(f"      Chart saved → {path}")
path = plot_category_performance(df, OUTPUT_DIR)
print(f"      Chart saved → {path}")

# ── Summary ────────────────────────────────────────────────────────────────
total_rev  = df["amount"].sum()
total_tx   = len(df)
top_prov   = df.groupby("province")["amount"].sum().idxmax()
top_cat    = df.groupby("category")["amount"].sum().idxmax()
n_anomaly  = int(df["is_anomaly"].sum()) if "is_anomaly" in df.columns else "N/A"

print("\n" + "="*62)
print("   EXECUTIVE SUMMARY (2023–2024)")
print("="*62)
print(f"   Total Revenue       : R{total_rev:>12,.0f}")
print(f"   Total Transactions  : {total_tx:>13,}")
print(f"   Top Province        : {top_prov}")
print(f"   Top Category        : {top_cat}")
print(f"   Anomalies Detected  : {n_anomaly}")
print(f"   Champions (RFM)     : {seg_counts.get('Champion',0)} customers")
print(f"   At-Risk Customers   : {seg_counts.get('At Risk',0)} customers")
print("\n   All charts saved to /output/")
print("="*62 + "\n")
