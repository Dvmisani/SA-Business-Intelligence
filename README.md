# 🧠 SA Business Intelligence Suite

A senior-level, end-to-end business intelligence pipeline built in Python — covering customer segmentation, revenue forecasting, anomaly detection, and regional performance analysis for a South African retail business.

---

## 📌 Project Overview

This project demonstrates a full analytics workflow that a Data Analyst would run on real business transaction data. It processes **4,756 transactions** across **400 customers** and **5 South African provinces** to produce actionable business intelligence.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3 | Core language |
| pandas | Data wrangling & aggregation |
| NumPy | Statistical computation |
| matplotlib | Custom chart rendering |
| seaborn | Heatmaps & statistical plots |
| scikit-learn | Scoring & segmentation logic |

---

## 📊 Analyses Performed

### 1. 🧩 RFM Customer Segmentation
Customers are scored on **Recency**, **Frequency**, and **Monetary** value (each scored 1–5), then segmented into strategic groups:

| Segment | Description |
|---|---|
| Champion | Bought recently, buys often, spends the most |
| Loyal | Buys regularly with good spend |
| At Risk | Previously high-value, now inactive |
| Lost | No recent activity, low engagement |
| Potential Loyalist | Recent buyers with growth potential |

> 26% of customers were classified as **Champions**, while 15% were identified as **Lost** — actionable targets for re-engagement campaigns.

---

### 2. 📈 Revenue Forecasting
Time-series decomposition separates **trend** from **seasonality** to project revenue 6 months forward with a **90% confidence interval**.

> Forecast shows a **+4.6% growth** in average monthly revenue for the next 6 months, driven by consistent seasonal patterns peaking in November–December.

---

### 3. 🚨 Anomaly Detection
Rolling **Z-score analysis** (14-day window) automatically flags transactions or daily revenue spikes more than **2.5 standard deviations** from the rolling mean.

> **12 anomalies** detected across the 2-year period — useful for fraud detection or unusual promotional spend identification.

---

### 4. 🗺️ Regional Heatmap
Monthly revenue broken down by **South African province** — visualised as a colour-coded heatmap showing which regions drive the most revenue and when.

> **Gauteng** consistently leads in revenue (40% of total), reflecting its population and economic weight.

---

### 5. 📦 Category Performance
Compares all 8 product categories on **total revenue**, **average order value**, and **customer reach** — combining a ranked bar chart with a bubble scatter plot.

> **Furniture** generated the highest total revenue despite fewer orders, while **Groceries** had the broadest customer reach.

---

## 📁 Project Structure

```
SA-Business-Intelligence/
│
├── main.py                  # Entry point — runs full pipeline
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── generate_data.py     # Synthetic data generator
│   ├── rfm_analysis.py      # RFM scoring & segmentation
│   ├── forecasting.py       # Trend decomposition & forecast
│   └── visuals.py           # Anomaly, heatmap & category charts
│
├── data/
│   └── transactions.csv     # Generated transaction dataset
│
└── output/                  # All charts saved here (auto-created)
    ├── 1_rfm_segmentation.png
    ├── 2_revenue_forecast.png
    ├── 3_anomaly_detection.png
    ├── 4_regional_heatmap.png
    └── 5_category_performance.png
```

---

## 🚀 How to Run

```bash
# 1. Clone the repo
git clone https://github.com/Dvmisani/SA-Business-Intelligence.git
cd SA-Business-Intelligence

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the full pipeline
python main.py
```

Output charts will appear in the `/output/` folder.

---

## 📋 Sample Output

```
==============================================================
   EXECUTIVE SUMMARY (2023–2024)
==============================================================
   Total Revenue       : R   9,286,622
   Total Transactions  :         4,756
   Top Province        : Gauteng
   Top Category        : Furniture
   Anomalies Detected  : 12
   Champions (RFM)     : 105 customers
   At-Risk Customers   : 6 customers
==============================================================
```

---

## 👤 Author

**Dumisani Abrahm Baloyi**
- 📧 dvmisani@gmail.com
- 🐙 [github.com/Dvmisani](https://github.com/Dvmisani)
- 📍 Pretoria, Gauteng, South Africa

---

## 📄 License

MIT License — free to use and adapt.
