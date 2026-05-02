"""
forecasting.py
Time-series revenue forecasting using trend + seasonality decomposition.
Projects 6 months forward with confidence intervals.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def build_forecast(df: pd.DataFrame, periods: int = 6):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    monthly = df.groupby(df["date"].dt.to_period("M"))["amount"].sum().reset_index()
    monthly.columns = ["period", "revenue"]
    monthly["ds"] = monthly["period"].dt.to_timestamp()
    monthly["t"]  = np.arange(len(monthly))

    # Fit linear trend
    coeffs    = np.polyfit(monthly["t"], monthly["revenue"], 1)
    trend_fn  = np.poly1d(coeffs)
    monthly["trend"] = trend_fn(monthly["t"])

    # Seasonal indices (ratio-to-moving-average)
    monthly["ratio"] = monthly["revenue"] / monthly["trend"]
    monthly["month"] = monthly["ds"].dt.month
    seasonal_idx = monthly.groupby("month")["ratio"].mean()

    # Residuals for std dev
    monthly["seasonal"] = monthly["month"].map(seasonal_idx)
    monthly["residual"] = monthly["revenue"] - (monthly["trend"] * monthly["seasonal"])
    resid_std = monthly["residual"].std()

    # Forecast
    future_t      = np.arange(len(monthly), len(monthly) + periods)
    future_months = pd.date_range(monthly["ds"].iloc[-1] + pd.offsets.MonthBegin(1),
                                  periods=periods, freq="MS")
    future_trend  = trend_fn(future_t)
    future_seas   = [seasonal_idx.get(m.month, 1.0) for m in future_months]
    future_yhat   = future_trend * np.array(future_seas)

    z = 1.645  # 90% CI
    future_upper = future_yhat + z * resid_std
    future_lower = np.maximum(future_yhat - z * resid_std, 0)

    forecast_df = pd.DataFrame({
        "ds":    future_months,
        "yhat":  future_yhat,
        "upper": future_upper,
        "lower": future_lower,
    })
    return monthly, forecast_df


def plot_forecast(monthly: pd.DataFrame, forecast_df: pd.DataFrame,
                  output_dir: str = "output"):
    NAVY   = "#1B2A4A"
    GOLD   = "#B8962E"
    BG     = "#0F1923"
    TEAL   = "#2E86AB"

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios":[2,1]})
    fig.patch.set_facecolor(BG)

    # ── Top: Actual + Forecast ──────────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor(BG)

    ax.plot(monthly["ds"], monthly["revenue"], color=TEAL, linewidth=2.5,
            marker="o", markersize=5, label="Actual Revenue", zorder=4)
    ax.plot(monthly["ds"], monthly["trend"], color=GOLD, linewidth=1.5,
            linestyle="--", label="Trend Line", zorder=3)
    ax.plot(forecast_df["ds"], forecast_df["yhat"], color="#E07B39", linewidth=2.5,
            marker="D", markersize=5, linestyle="--", label="Forecast", zorder=4)
    ax.fill_between(forecast_df["ds"], forecast_df["lower"], forecast_df["upper"],
                    alpha=0.20, color="#E07B39", label="90% Confidence Interval")

    # Divider line
    ax.axvline(monthly["ds"].iloc[-1], color="#555555", linewidth=1.2,
               linestyle=":", zorder=2)
    ax.text(monthly["ds"].iloc[-1], ax.get_ylim()[1] * 0.95,
            " Forecast →", color="#888888", fontsize=9, va="top")

    ax.set_title("Monthly Revenue — Actual vs Forecast (6-Month Projection)",
                 color="white", fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel("Revenue (ZAR)", color="#AAAAAA", fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"R{x/1000:.0f}k"))
    ax.tick_params(colors="#AAAAAA")
    for spine in ax.spines.values(): spine.set_color("#333333")
    ax.grid(True, color="#1E2D3D", linewidth=0.6, axis="y")
    legend = ax.legend(fontsize=9, facecolor="#1B2A4A", labelcolor="white",
                       edgecolor="#333333")

    # ── Bottom: Month-over-Month Growth % ──────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor(BG)

    monthly["growth"] = monthly["revenue"].pct_change() * 100
    colors = ["#1E8449" if g >= 0 else "#C0392B" for g in monthly["growth"].fillna(0)]
    ax2.bar(monthly["ds"], monthly["growth"].fillna(0), color=colors,
            width=20, edgecolor=BG, linewidth=0.5)
    ax2.axhline(0, color="#555555", linewidth=0.8)
    ax2.set_title("Month-over-Month Revenue Growth (%)", color="white",
                  fontsize=12, fontweight="bold", pad=8)
    ax2.set_ylabel("Growth %", color="#AAAAAA", fontsize=10)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax2.tick_params(colors="#AAAAAA")
    for spine in ax2.spines.values(): spine.set_color("#333333")
    ax2.grid(True, color="#1E2D3D", linewidth=0.6, axis="y")

    plt.suptitle("Revenue Forecasting & Growth Analysis", color="white",
                 fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = f"{output_dir}/2_revenue_forecast.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    return path
