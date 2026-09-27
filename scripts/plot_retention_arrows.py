#!/usr/bin/env python3
"""Figure 1: per-label test recall of the Unanimous-trained and Unanimous-IPW-trained models relative
to the Consensus-trained model, against label retention, for Qwen3.5-4B and 9B at epoch 3.

Labels whose recall drops by at least LABEL_LOSS pp under Unanimous training are named. Reads the
ipw_label_analysis.py output (label-analysis.csv) and writes PDF (TrueType fonts, as AAAI requires)
and PNG. Usage: plot_retention_arrows.py LABEL_DIR_4B LABEL_DIR_9B OUTPUT_STEM, e.g.

    python scripts/plot_retention_arrows.py outputs/qwen35-4b/ipw-label-analysis \
      outputs/qwen35-9b/ipw-label-analysis outputs/figures/fig1-retention-recall
"""

import csv
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

INK, INK_2, AXIS, GRID, LEADER = "#0b0b0b", "#52514e", "#c3c2b7", "#e1e0d9", "#898781"
UNANIMOUS, IPW = "#2a78d6", "#eb6834"          # categorical slots 1-2 (validated on white)
LABEL_LOSS = 10.0                              # name labels whose Unanimous recall drops >= 10 pp
NAMES = {
    "suicideideation_active_ongoing": "Active suicidal ideation (ongoing)",
    "suicideideation_active_past": "Active suicidal ideation (past)",
    "suicideideation_passive_ongoing": "Passive suicidal ideation (ongoing)",
    "suicideideation_passive_past": "Passive suicidal ideation (past)",
    "selfharm_ongoing": "Self-harm (ongoing)", "selfharm_past": "Self-harm (past)",
    "domesticviolence_ongoing": "Domestic violence (ongoing)", "domesticviolence_past": "Domestic violence (past)",
    "rape_ongoing": "Rape (ongoing)", "rape_past": "Rape (past)",
    "sexualharassment_ongoing": "Sexual harassment\n(ongoing)", "sexualharassment_past": "Sexual harassment (past)",
    "childabuse_endangerment_ongoing": "Child abuse (ongoing)", "childabuse_endangerment_past": "Child abuse (past)",
}
# Text anchor (retention, pp), horizontal alignment, and the text edge the leader line leaves from
# (0 = left, 1 = right) for each named label; placed in empty regions so that leader lines never cross.
TEXT_AT = {
    "4B": {"suicideideation_passive_past": (0.47, -33, "left", 0), "childabuse_endangerment_ongoing": (0.47, -39, "left", 0),
           "selfharm_past": (0.565, -16, "right", 1), "sexualharassment_ongoing": (0.617, -23, "left", 0)},
    "9B": {"childabuse_endangerment_ongoing": (0.47, -37, "left", 0), "suicideideation_passive_past": (0.47, -29, "left", 0),
           "suicideideation_active_past": (0.47, -21, "left", 0)},
}

plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans", "font.size": 6.5,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK_2, "xtick.color": INK_2,
    "ytick.color": INK_2, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
    "ytick.major.size": 2.5, "legend.frameon": False,
})


def load(directory):
    return [r for r in csv.DictReader((Path(directory) / "label-analysis.csv").open()) if r["label"] != "no_crisis"]


def panel(ax, rows, size):
    ax.axhline(0, color=INK_2, linewidth=0.8, zorder=1)
    ax.grid(axis="y", color=GRID, linewidth=0.5, zorder=0)
    for r in rows:
        x = float(r["retention"])
        c = float(r["Consensus recall"]) * 100
        u = float(r["Unanimous recall"]) * 100 - c
        i = float(r["Unanimous-IPW recall"]) * 100 - c
        ax.scatter([x], [u], s=16, color=UNANIMOUS, edgecolors="white", linewidths=0.5, zorder=4)
        if abs(i - u) > 1e-9:
            ax.annotate("", xy=(x, i), xytext=(x, u), zorder=3,
                        arrowprops=dict(arrowstyle="-|>,head_length=0.35,head_width=0.18", color=IPW,
                                        linewidth=1.0, shrinkA=2.0, shrinkB=0))
        else:                                                # IPW recall equals Unanimous recall
            ax.plot([x - 0.008, x + 0.008], [i, i], color=IPW, linewidth=1.1, zorder=5, solid_capstyle="butt")
        if u <= -LABEL_LOSS:
            tx, ty, align, edge = TEXT_AT[size][r["label"]]
            ax.annotate(NAMES[r["label"]], xy=(x, u), xytext=(tx, ty), fontsize=5.5, color=INK,
                        ha=align, va="center", zorder=6,
                        arrowprops=dict(arrowstyle="-", color=LEADER, linewidth=0.5, shrinkA=1.5, shrinkB=2.5,
                                        relpos=(edge, 0.5)))
    ax.text(0.01, 0.97, f"Qwen3.5-{size}", transform=ax.transAxes, ha="left", va="top",
            fontsize=7, color=INK, fontweight="bold")
    ax.set_xlim(0.36, 0.72)
    ax.set_xticks([0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7])
    ax.set_ylim(-42, 24)
    ax.set_yticks([-40, -30, -20, -10, 0, 10, 20])
    ax.spines[["top", "right"]].set_visible(False)


def main():
    rows4, rows9 = load(sys.argv[1]), load(sys.argv[2])
    stem = Path(sys.argv[3])
    fig, axes = plt.subplots(2, 1, figsize=(3.33, 3.2), sharex=True)
    panel(axes[0], rows4, "4B")
    panel(axes[1], rows9, "9B")
    axes[1].set_xlabel("Label retention under unanimous filtering\n(share of Consensus training posts kept in Unanimous)")
    fig.supylabel("Per-label recall difference from Consensus SFT (pp)", fontsize=6.5, color=INK_2, x=0.02)
    handles = [Line2D([], [], marker="o", linestyle="none", markersize=4, color=UNANIMOUS,
                      markeredgecolor="white", label="Unanimous SFT"),
               Line2D([], [], marker="^", linestyle="-", markersize=3.8, color=IPW, linewidth=1.0,
                      label="Unanimous-IPW SFT"),
               Line2D([], [], color=INK_2, linewidth=0.8, label="Consensus SFT (= 0)")]
    fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=5.8, handlelength=1.4,
               columnspacing=0.9, bbox_to_anchor=(0.53, 1.0))
    fig.subplots_adjust(left=0.13, right=0.985, top=0.935, bottom=0.14, hspace=0.1)
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=400)
    print("wrote", stem.with_suffix(".pdf"), stem.with_suffix(".png"))


if __name__ == "__main__":
    main()
