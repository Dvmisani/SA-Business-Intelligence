"""
rfm_analysis.py
Segments customers using Recency, Frequency, and Monetary (RFM) analysis.
Each dimension is scored 1-5. Combined score maps to a business segment.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

SEGMENT_MAP = {
    r"[4-5][4-5]": "Champion",
    r"[3-5][3-5]": "Loyal",
    r"[4-5][1-2]": "Recent Customer",
    r"[2-3][3-4]": "Potential Loyalist",
    r"[1-2][4-5]": "At Risk",
    r"[1-2][2-3]": "Needs Attention",
    r"[1-2][1-2]": "Lost",
}

SEGMENT_COLORS = {
    "Champion":          "#1B2A4A",
    "Loyal":             "#B8962E",
    "Recent Customer":   "#2E86AB",
    "Potential Loyalist":"#6B8E23",
    "At Risk":           "#E07B39",
    "Needs Attention":   "#9B59B6",
    "Lost":              "#C0392B",
}

def compute_rfm(df: pd.DataFrame, snapshot_date: str = "2025-01-01") -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    snap = pd.to_datetime(snapshot_date)

    rfm = df.groupby("customer_id").agg(
        recency   = ("date",   lambda x: (snap - x.max()).days),
        frequency = ("transaction_id", "count"),
        monetary  = ("amount", "sum"),
    ).reset_index()

    # Score 1-5 (5 = best)
    rfm["R"] = pd.qcut(rfm["recency"],   5, labels=[5,4,3,2,1], duplicates="drop").astype(int)
    rfm["F"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1,2,3,4,5]).astype(int)
    rfm["M"] = pd.qcut(rfm["monetary"],  5, labels=[1,2,3,4,5], duplicates="drop").astype(int)
    rfm["RFM_Score"] = rfm["R"].astype(str) + rfm["F"].astype(str)

    def assign_segment(score):
        import re
        for pattern, seg in SEGMENT_MAP.items():
            if re.match(pattern, score):
                return seg
        return "Other"

    rfm["Segment"] = rfm["RFM_Score"].apply(assign_segment)
    return rfm


def plot_rfm(rfm: pd.DataFrame, output_dir: str = "output"):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor("#0F1923")

    # ── Left: Bubble chart ──────────────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor("#0F1923")

    for seg, group in rfm.groupby("Segment"):
        ax.scatter(
            group["R"], group["F"],
            s=group["monetary"] / 120,
            color=SEGMENT_COLORS.get(seg, "#888888"),
            alpha=0.75, edgecolors="white", linewidth=0.3,
            label=seg, zorder=3
        )

    ax.set_title("RFM Customer Segmentation", color="white", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Recency Score  (5 = most recent)", color="#AAAAAA", fontsize=10)
    ax.set_ylabel("Frequency Score  (5 = most frequent)", color="#AAAAAA", fontsize=10)
    ax.tick_params(colors="#AAAAAA")
    for spine in ax.spines.values(): spine.set_color("#333333")
    ax.grid(True, color="#1E2D3D", linewidth=0.7)
    legend = ax.legend(fontsize=8, facecolor="#1B2A4A", labelcolor="white",
                       edgecolor="#333333", markerscale=0.8, title="Segment",
                       title_fontsize=9)
    legend.get_title().set_color("white")
    ax.text(0.98, 0.02, "Bubble size = Total Spend",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, color="#888888", style="italic")

    # ── Right: Segment donut ────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor("#0F1923")

    counts = rfm["Segment"].value_counts()
    colors = [SEGMENT_COLORS.get(s, "#888888") for s in counts.index]
    wedges, texts, autotexts = ax2.pie(
        counts, labels=None, autopct="%1.0f%%",
        colors=colors, startangle=140,
        wedgeprops={"edgecolor": "#0F1923", "linewidth": 2},
        pctdistance=0.78
    )
    for at in autotexts:
        at.set_color("white"); at.set_fontsize(9); at.set_fontweight("bold")

    # Draw inner circle for donut
    centre = plt.Circle((0, 0), 0.50, color="#0F1923")
    ax2.add_patch(centre)
    ax2.text(0, 0.08, str(len(rfm)), ha="center", va="center",
             fontsize=28, fontweight="bold", color="white")
    ax2.text(0, -0.18, "Customers", ha="center", va="center",
             fontsize=11, color="#AAAAAA")

    patches = [mpatches.Patch(color=SEGMENT_COLORS.get(s,"#888"), label=f"{s}  ({c})")
               for s, c in counts.items()]
    ax2.legend(handles=patches, loc="lower center", bbox_to_anchor=(0.5, -0.18),
               ncol=2, fontsize=8, facecolor="#1B2A4A", labelcolor="white",
               edgecolor="#333333")
    ax2.set_title("Customer Segment Distribution", color="white", fontsize=14,
                  fontweight="bold", pad=12)

    plt.suptitle("RFM Analysis — Customer Intelligence", color="white",
                 fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = f"{output_dir}/1_rfm_segmentation.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0F1923")
    plt.close()
    return path
