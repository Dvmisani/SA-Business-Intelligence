# SA Business Intelligence Suite

A Python project I built to practice end-to-end data analytics - from raw transactions all the way to customer segmentation, revenue forecasting, and anomaly detection. The data is synthetic but modelled on realistic South African retail patterns.

---

## What it does

The pipeline takes a transaction dataset (4 700+ records across 400 customers and 5 provinces) and runs four analyses:

**RFM Customer Segmentation** - scores every customer on how recently they bought, how often, and how much they spend. From there it groups them into segments like Champions, Loyal, At Risk, and Lost. This kind of analysis is useful when you want to know who to focus retention efforts on.

**Revenue Forecasting** - decomposes monthly revenue into trend and seasonal components, then projects 6 months forward with a 90% confidence interval. The model picked up the November/December spike which is very common in SA retail.

**Anomaly Detection** - uses a 14-day rolling Z-score to flag days where revenue spiked unexpectedly (more than 2.5 standard deviations from the rolling mean). Picked up 12 anomalies in the 2023–2024 period.

**Regional & Category Analysis** - heatmap of monthly revenue by province, plus a breakdown of which product categories drive the most value vs which reach the most customers.

---

## Running it

```bash
git clone https://github.com/Dvmisani/SA-Business-Intelligence.git
cd SA-Business-Intelligence
pip install -r requirements.txt
python main.py
```

Charts get saved to the `/output/` folder automatically.

---

## Stack

Python, pandas, NumPy, matplotlib, seaborn, scikit-learn

---

## Project layout

```
SA-Business-Intelligence/
├── main.py
├── src/
│   ├── generate_data.py
│   ├── rfm_analysis.py
│   ├── forecasting.py
│   └── visuals.py
├── data/
│   └── transactions.csv
└── output/
```

---

Built by **Dumisani Abrahm Baloyi** — dvmisani@gmail.com
