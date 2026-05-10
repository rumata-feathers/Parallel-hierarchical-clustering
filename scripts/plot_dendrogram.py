#!/usr/bin/env python3
"""
Reads *_linkage.csv files from the results directory and plots dendrograms.
Optionally colors leaves by true label if a matching *_labels.csv exists.

Usage:
    python3 scripts/plot_dendrogram.py [results_dir]
"""
import sys
import os
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram


def plot_one(csv_path: str, plots_dir: str) -> None:
    df = pd.read_csv(csv_path)
    Z = df[["cluster_i", "cluster_j", "distance", "count"]].values.astype(float)

    # Try to load matching labels file for leaf coloring
    labels_path = csv_path.replace("_linkage.csv", "").rsplit("_", 2)[0] + "_labels.csv"
    leaf_labels = None
    if os.path.exists(labels_path):
        leaf_labels = pd.read_csv(labels_path)["label"].astype(str).tolist()

    name = os.path.basename(csv_path).replace("_linkage.csv", "")
    fig, ax = plt.subplots(figsize=(14, 5))

    dendrogram(
        Z,
        ax=ax,
        labels=leaf_labels,
        leaf_rotation=90,
        leaf_font_size=6,
        no_labels=(leaf_labels is None),
    )
    ax.set_title(name)
    ax.set_xlabel("Sample" if leaf_labels is None else "True label")
    ax.set_ylabel("Distance")

    out_path = os.path.join(plots_dir, name + "_dendrogram.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path}")


if __name__ == "__main__":
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results"
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(results_dir, "*_linkage.csv")))
    if not files:
        print(f"No linkage CSV files found in {results_dir}/")
        sys.exit(1)

    for csv_path in files:
        try:
            plot_one(csv_path, plots_dir)
        except Exception as e:
            print(f"  error on {csv_path}: {e}")
