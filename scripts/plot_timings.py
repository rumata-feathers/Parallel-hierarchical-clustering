#!/usr/bin/env python3
"""
Reads results/timings.csv and produces two charts:
  1. Wall time: serial vs parallel per dataset/linkage
  2. Speedup: serial_time / parallel_time

Usage:
    python3 scripts/plot_timings.py [results_dir]
"""
import sys
import os

import pandas as pd
import matplotlib.pyplot as plt


def compute_speedup(df: pd.DataFrame) -> pd.DataFrame:
    serial   = df[df["mode"] == "serial"].set_index(["dataset", "linkage"])["wall_ms"]
    parallel = df[df["mode"] == "parallel"].copy()
    parallel["speedup"] = parallel.apply(
        lambda r: serial.get((r["dataset"], r["linkage"]), float("nan")) / r["wall_ms"],
        axis=1,
    )
    return parallel


if __name__ == "__main__":
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results"
    timing_path = os.path.join(results_dir, "timings.csv")
    plots_dir   = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    if not os.path.exists(timing_path):
        print(f"No timings file at {timing_path} — run the pipeline first.")
        sys.exit(1)

    df = pd.read_csv(timing_path)
    df["label"] = df["dataset"] + " / " + df["linkage"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- Wall time bar chart ---
    pivot = df.pivot_table(index="label", columns="mode", values="wall_ms", aggfunc="mean")
    pivot.plot(kind="bar", ax=axes[0], colormap="Set2")
    axes[0].set_title("Wall time: serial vs parallel (ms)")
    axes[0].set_ylabel("ms")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].legend(title="mode")

    # --- Speedup bar chart ---
    speedup_df = compute_speedup(df)
    if not speedup_df.empty:
        speedup_df.groupby("label")["speedup"].mean().plot(
            kind="bar", ax=axes[1], color="steelblue"
        )
        axes[1].axhline(1.0, color="red", linestyle="--", linewidth=1, label="no speedup")
        axes[1].set_title("Speedup  (serial ms / parallel ms)")
        axes[1].set_ylabel("speedup")
        axes[1].set_xlabel("")
        axes[1].tick_params(axis="x", rotation=45)
        axes[1].legend()

    fig.tight_layout()
    out_path = os.path.join(plots_dir, "timings.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
