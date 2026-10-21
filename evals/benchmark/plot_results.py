"""Plot benchmark results as a grouped bar chart.

Reads evals/benchmark/results.json (produced by run_benchmark.py) and writes
evals/benchmark/results.png.

Run:
    python evals/benchmark/plot_results.py
"""
from __future__ import annotations

import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
RESULTS = HERE / "results.json"
OUTPUT = HERE / "results.png"


def main() -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend — safe in CI/scripts
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not installed. Run: pip install matplotlib", file=sys.stderr)
        sys.exit(1)

    if not RESULTS.exists():
        print(f"results.json not found at {RESULTS}. Run run_benchmark.py first.", file=sys.stderr)
        sys.exit(1)

    data = json.loads(RESULTS.read_text())
    ds = data["deepscout"]
    bl = data["baseline"]

    metrics = [
        ("Accuracy", "accuracy"),
        ("Faithfulness", "mean_faithfulness"),
        ("Hallucination\nRate", "hallucination_rate"),
    ]

    labels = [m[0] for m in metrics]
    ds_vals = [ds[m[1]] for m in metrics]
    bl_vals = [bl[m[1]] for m in metrics]

    x = np.arange(len(labels))
    bar_width = 0.32
    gap = 0.04

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#F8F9FA")

    bars_ds = ax.bar(
        x - (bar_width / 2 + gap / 2),
        ds_vals,
        width=bar_width,
        label="DeepScout",
        color="#2563EB",
        edgecolor="white",
        linewidth=0.8,
    )
    bars_bl = ax.bar(
        x + (bar_width / 2 + gap / 2),
        bl_vals,
        width=bar_width,
        label="Baseline",
        color="#F59E0B",
        edgecolor="white",
        linewidth=0.8,
    )

    # Value labels on bars
    for bar in list(bars_ds) + list(bars_bl):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.012,
            f"{h:.3f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#374151",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10.5)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Score (0 – 1)", fontsize=10)
    ax.set_title(
        "DeepScout vs Baseline — Benchmark Comparison",
        fontsize=13,
        fontweight="bold",
        pad=14,
        color="#111827",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#D1D5DB")
    ax.spines["bottom"].set_color("#D1D5DB")
    ax.yaxis.set_tick_params(color="#9CA3AF")
    ax.xaxis.set_tick_params(color="#9CA3AF")
    ax.tick_params(axis="y", labelcolor="#6B7280", labelsize=9)

    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, color="#E5E7EB", zorder=0)
    ax.set_axisbelow(True)

    n = ds.get("n", "?")
    subtitle = f"n = {n} questions   |   mock mode (substitute real provider for live scores)"
    fig.text(0.5, 0.01, subtitle, ha="center", fontsize=8, color="#9CA3AF")

    legend = ax.legend(
        frameon=True,
        framealpha=0.9,
        edgecolor="#E5E7EB",
        fontsize=9.5,
        loc="upper right",
    )
    legend.get_frame().set_linewidth(0.8)

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(OUTPUT, dpi=150, bbox_inches="tight")
    print(f"Chart saved to {OUTPUT}")


if __name__ == "__main__":
    main()
