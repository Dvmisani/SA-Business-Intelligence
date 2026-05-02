"""
generate_data.py
Generates realistic synthetic South African retail transaction data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

PROVINCES   = ["Gauteng", "Western Cape", "KwaZulu-Natal", "Eastern Cape", "Limpopo"]
PROVINCE_W  = [0.40, 0.22, 0.18, 0.12, 0.08]          # population-weighted
CATEGORIES  = ["Electronics", "Clothing", "Groceries", "Furniture",
               "Stationery", "Tools", "Health & Beauty", "Sports"]
CAT_AVG     = [3200, 850, 420, 5800, 210, 1100, 380, 950]   # avg spend per category

def seasonal_factor(date):
    """Higher spend in Nov/Dec, lower in Jan/Feb."""
    m = date.month
    return {1:0.72, 2:0.68, 3:0.80, 4:0.82, 5:0.88,
            6:0.85, 7:0.90, 8:0.93, 9:0.95,
            10:1.05, 11:1.25, 12:1.50}.get(m, 1.0)

def generate_transactions(n_customers=400, start="2023-01-01", end="2024-12-31"):
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt   = datetime.strptime(end,   "%Y-%m-%d")
    total_days = (end_dt - start_dt).days

    # Assign each customer a home province and spending tier
    customer_province = np.random.choice(PROVINCES, n_customers, p=PROVINCE_W)
    customer_tier     = np.random.choice(["high","mid","low"], n_customers, p=[0.15,0.55,0.30])

    rows = []
    for cid in range(1, n_customers + 1):
        tier = customer_tier[cid - 1]
        n_tx = {"high": np.random.randint(18, 40),
                "mid":  np.random.randint(6,  18),
                "low":  np.random.randint(1,   6)}[tier]

        for _ in range(n_tx):
            day   = np.random.randint(0, total_days)
            date  = start_dt + timedelta(days=int(day))
            cat_i = np.random.randint(0, len(CATEGORIES))
            base  = CAT_AVG[cat_i]
            sf    = seasonal_factor(date)
            tier_mult = {"high": 1.8, "mid": 1.0, "low": 0.5}[tier]
            amount = max(50, np.random.normal(base * sf * tier_mult, base * 0.25))

            rows.append({
                "transaction_id": f"TXN{len(rows)+1:05d}",
                "customer_id":    f"CUST{cid:04d}",
                "date":           date.strftime("%Y-%m-%d"),
                "amount":         round(amount, 2),
                "category":       CATEGORIES[cat_i],
                "province":       customer_province[cid - 1],
                "tier":           tier,
            })

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

    # Plant 12 anomalies (unusually large transactions)
    anomaly_idx = np.random.choice(df.index, 12, replace=False)
    df.loc[anomaly_idx, "amount"] = df.loc[anomaly_idx, "amount"] * np.random.uniform(4, 8, 12)
    df.loc[anomaly_idx, "is_anomaly"] = True
    df["is_anomaly"] = df["is_anomaly"].fillna(False)

    return df

if __name__ == "__main__":
    df = generate_transactions()
    df.to_csv("data/transactions.csv", index=False)
    print(f"Generated {len(df):,} transactions for {df['customer_id'].nunique()} customers.")
    print(df.head())
