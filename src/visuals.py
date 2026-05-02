"""
visuals.py
Anomaly detection using rolling Z-scores, regional heatmap,
category performance matrix, and KPI summary card.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
import seaborn as sns


BG    = "#0F1923"
NAVY  = "#1B2A4A"
GOLD  = "#B8962E"
TEAL  = "#2E86AB"
RED   = "#C0392B"
GREEN = "#1E8449"


# ── Chart 3: Anomaly Detection ─────────────────────────────────────────────

def plot_anomaly(df: pd.DataFrame, output_dir: str = "output"):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    daily = df.groupby("date")["amount"].sum().reset_index()
    daily.columns = ["date", "revenue"]
    daily = daily.sort_values("date")

    # Rolling mean and std (14-day window)
    daily["rolling_mean"] = daily["revenue"].rolling(14, min_periods=5).mean()
    daily["rolling_std"]  = daily["revenue"].rolling(14, min_periods=5).std()
    daily["z_score"]      = ((daily["revenue"] - daily["rolling_mean"])
                              / daily["rolling_std"].replace(0, np.nan))
    daily["anomaly"]      = daily["z_score"].abs() > 2.5

    fig, axes = plt.subplots(2, 1, figsize=(14, 9),
                             gridspec_kw={"height_ratios": [2, 1]})
    fig.patch.set_facecolor(BG)

    # Top: revenue line with anomalies
    ax = axes[0]
    ax.set_facecolor(BG)
    ax.plot(daily["date"], daily["revenue"], color=TEAL, linewidth=1.4,
            alpha=0.9, label="Daily Revenue", zorder=3)
    ax.fill_between(daily["date"],
                    daily["rolling_mean"] - 2.5 * daily["rolling_std"],
                    daily["rolling_mean"] + 2.5 * daily["rolling_std"],
                    alpha=0.12, color=GOLD, label="Normal Range (±2.5σ)")
    ax.plot(daily["date"], daily["rolling_mean"], color=GOLD,
            linewidth=1.5, linestyle="--", label="14-Day Rolling Mean")
    anomalies = daily[daily["anomaly"]]
    ax.scatter(anomalies["date"], anomalies["revenue"],
               color=RED, zorder=5, s=70, marker="^",
               edgecolors="white", linewidth=0.5, label=f"Anomaly ({len(anomalies)} detected)")

    ax.set_title("Daily Revenue — Anomaly Detection (Rolling Z-Score)",
                 color="white", fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel("Revenue (ZAR)", color="#AAAAAA", fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"R{x/1000:.0f}k"))
    ax.tick_params(colors="#AAAAAA")
    for s in ax.spines.values(): s.set_color("#333333")
    ax.grid(True, color="#1E2D3D", linewidth=0.5, axis="y")
    legend = ax.legend(fontsize=9, facecolor=NAVY, labelcolor="white", edgecolor="#333333")

    # Bottom: Z-score bars
    ax2 = axes[1]
    ax2.set_facecolor(BG)
    zcolors = [RED if a else TEAL for a in daily["anomaly"]]
    ax2.bar(daily["date"], daily["z_score"].fillna(0), color=zcolors,
            width=1.0, alpha=0.85)
    ax2.axhline( 2.5, color=RED,   linewidth=1.0, linestyle="--", alpha=0.7)
    ax2.axhline(-2.5, color=RED,   linewidth=1.0, linestyle="--", alpha=0.7)
    ax2.axhline( 0,   color="#555", linewidth=0.8)
    ax2.set_title("Z-Score per Day  (|z| > 2.5 flagged as anomaly)",
                  color="white", fontsize=11, fontweight="bold", pad=8)
    ax2.set_ylabel("Z-Score", color="#AAAAAA", fontsize=9)
    ax2.tick_params(colors="#AAAAAA")
    for s in ax2.spines.values(): s.set_color("#333333")
    ax2.grid(True, color="#1E2D3D", linewidth=0.5, axis="y")

    plt.suptitle("Transaction Anomaly Detection", color="white",
                 fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = f"{output_dir}/3_anomaly_detection.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    return path


# ── Chart 4: Regional Heatmap ─────────────────────────────────────────────

def plot_regional_heatmap(df: pd.DataFrame, output_dir: str = "output"):
    df = df.copy()
    df["date"]  = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.strftime("%b %Y")
    df["month_dt"] = df["date"].dt.to_period("M")

    pivot = df.groupby(["province", "month_dt"])["amount"].sum().unstack("month_dt")
    pivot.columns = [str(c) for c in pivot.columns]

    # Sort columns chronologically
    sorted_cols = sorted(pivot.columns)
    pivot = pivot[sorted_cols]
    col_labels = [pd.Period(c).strftime("%b %y") for c in sorted_cols]

    fig, ax = plt.subplots(figsize=(16, 5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    cmap = sns.color_palette("YlOrRd", as_cmap=True)
    sns.heatmap(pivot, ax=ax, cmap=cmap, linewidths=0.5, linecolor="#0F1923",
                annot=True, fmt=".0f",
                annot_kws={"size": 7, "color": "#111111"},
                cbar_kws={"shrink": 0.6},
                xticklabels=col_labels)

    ax.set_title("Monthly Revenue by Province (ZAR)", color="white",
                 fontsize=14, fontweight="bold", pad=14)
    ax.set_xlabel("Month", color="#AAAAAA", fontsize=10)
    ax.set_ylabel("Province", color="#AAAAAA", fontsize=10)
    ax.tick_params(colors="#AAAAAA", labelsize=9)
    plt.xticks(rotation=45, ha="right")

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(colors="#AAAAAA", labelsize=8)
    cbar.ax.yaxis.label.set_color("#AAAAAA")

    plt.suptitle("Regional Sales Performance Heatmap", color="white",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = f"{output_dir}/4_regional_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    return path


# ── Chart 5: Category Performance ─────────────────────────────────────────

def plot_category_performance(df: pd.DataFrame, output_dir: str = "output"):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    cat = df.groupby("category").agg(
        total_revenue = ("amount",         "sum"),
        avg_order     = ("amount",         "mean"),
        n_orders      = ("transaction_id", "count"),
        n_customers   = ("customer_id",    "nunique"),
    ).reset_index().sort_values("total_revenue", ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor(BG)

    # Left: horizontal bar — total revenue
    ax = axes[0]
    ax.set_facecolor(BG)
    bar_colors = [GOLD if i == len(cat)-1 else TEAL for i in range(len(cat))]
    bars = ax.barh(cat["category"], cat["total_revenue"],
                   color=bar_colors, edgecolor=BG, linewidth=0.4)
    for bar, val in zip(bars, cat["total_revenue"]):
        ax.text(bar.get_width() + cat["total_revenue"].max()*0.01,
                bar.get_y() + bar.get_height()/2,
                f"R{val/1000:.0f}k", va="center", ha="left",
                fontsize=8, color="white")
    ax.set_title("Total Revenue by Category", color="white",
                 fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Revenue (ZAR)", color="#AAAAAA", fontsize=10)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"R{x/1000:.0f}k"))
    ax.tick_params(colors="#AAAAAA")
    for s in ax.spines.values(): s.set_color("#333333")
    ax.grid(True, color="#1E2D3D", linewidth=0.5, axis="x")

    # Right: scatter — avg order value vs num customers
    ax2 = axes[1]
    ax2.set_facecolor(BG)
    scatter = ax2.scatter(cat["n_customers"], cat["avg_order"],
                          s=cat["total_revenue"] / 800,
                          c=cat["total_revenue"], cmap="YlOrRd",
                          edgecolors="white", linewidth=0.5, zorder=3, alpha=0.9)
    for _, row in cat.iterrows():
        ax2.annotate(row["category"],
                     (row["n_customers"], row["avg_order"]),
                     textcoords="offset points", xytext=(8, 4),
                     fontsize=8, color="white")
    ax2.set_title("Avg Order Value vs Customer Reach", color="white",
                  fontsize=13, fontweight="bold", pad=10)
    ax2.set_xlabel("Unique Customers", color="#AAAAAA", fontsize=10)
    ax2.set_ylabel("Avg Order Value (ZAR)", color="#AAAAAA", fontsize=10)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"R{x:,.0f}"))
    ax2.tick_params(colors="#AAAAAA")
    for s in ax2.spines.values(): s.set_color("#333333")
    ax2.grid(True, color="#1E2D3D", linewidth=0.5)
    cbar = fig.colorbar(scatter, ax=ax2, shrink=0.6)
    cbar.ax.tick_params(colors="#AAAAAA", labelsize=7)
    cbar.set_label("Total Revenue", color="#AAAAAA", fontsize=8)
    ax2.text(0.02, 0.95, "Bubble size = Total Revenue",
             transform=ax2.transAxes, color="#888888", fontsize=8, style="italic")

    plt.suptitle("Category Performance Analysis", color="white",
                 fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = f"{output_dir}/5_category_performance.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    return path
