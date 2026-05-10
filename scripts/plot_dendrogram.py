#!/usr/bin/env python3
"""
For each *_serial_linkage.csv in results_dir, produces a two-panel figure:
  Left  — scatter of raw points colored by true label (2-D datasets only)
  Right — dendrogram where each branch is colored by its true-label group
           (pure-label subtrees keep their color; mixed subtrees turn gray)

Usage:
    python3 scripts/plot_dendrogram.py [results_dir] [data_dir]
"""
import sys
import os
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.cluster.hierarchy import dendrogram


def load_dataset(stem: str, data_dir: str):
    path = os.path.join(data_dir, stem + ".csv")
    if not os.path.exists(path):
        return None, None
    df = pd.read_csv(path)
    return df.iloc[:, :-1], df.iloc[:, -1].astype(str)


def build_palette(labels: pd.Series) -> dict:
    unique  = sorted(labels.unique())
    palette = plt.get_cmap("tab10", max(len(unique), 1))
    return {lbl: palette(i) for i, lbl in enumerate(unique)}


def scatter_panel(ax, features: pd.DataFrame, labels: pd.Series,
                  color_map: dict, title: str) -> None:
    if features.shape[1] < 2:
        ax.text(0.5, 0.5, "Not 2-D — cannot scatter",
                ha="center", va="center", transform=ax.transAxes)
        return

    ax.scatter(features.iloc[:, 0], features.iloc[:, 1],
               c=labels.map(color_map), s=8, linewidths=0)

    handles = [plt.Line2D([0], [0], marker="o", color="w",
                           markerfacecolor=color_map[lbl], markersize=6, label=lbl)
               for lbl in sorted(color_map)]
    ax.legend(handles=handles, title="true label",
              fontsize=6, title_fontsize=6, loc="best")
    ax.set_title(f"{title}  —  true labels")
    ax.set_xlabel(features.columns[0])
    ax.set_ylabel(features.columns[1])
    ax.set_aspect("equal", adjustable="datalim")


def build_link_color_func(Z, labels: pd.Series, color_map: dict):
    """
    Returns a link_color_func for scipy's dendrogram.

    Each internal node n+k is assigned the true-label set of all leaves beneath it.
    If that set is a single label  → use the matching color from color_map.
    If it contains multiple labels → gray (the subtree mixes true clusters).
    """
    n = len(labels)

    # node_labels[i] = frozenset of true labels under node i
    node_labels = {i: frozenset([labels.iloc[i]]) for i in range(n)}
    for k, (ci, cj, _, _) in enumerate(Z):
        node_labels[n + k] = node_labels[int(ci)] | node_labels[int(cj)]

    def link_color_func(node_id):
        lbl_set = node_labels.get(node_id, frozenset())
        if len(lbl_set) == 1:
            return mcolors.to_hex(color_map[next(iter(lbl_set))])
        return "#bbbbbb"

    return link_color_func


def dendrogram_panel(ax, Z, labels: pd.Series, color_map: dict, title: str) -> None:
    link_color_func = build_link_color_func(Z, labels, color_map)

    dend = dendrogram(
        Z, ax=ax,
        no_labels=False,
        leaf_rotation=90,
        leaf_font_size=4,
        labels=labels.tolist(),
        link_color_func=link_color_func,
    )

    # Color leaf tick labels to match scatter
    for tick, idx in zip(ax.get_xticklabels(), dend["leaves"]):
        tick.set_color(color_map[labels.iloc[idx]])

    ax.set_title(f"{title}  —  dendrogram (branch color = true label)")
    ax.set_xlabel("samples")
    ax.set_ylabel("distance")


def plot_one(csv_path: str, plots_dir: str, data_dir: str) -> None:
    df = pd.read_csv(csv_path)
    Z  = df[["cluster_i", "cluster_j", "distance", "count"]].values.astype(float)

    base  = os.path.basename(csv_path).replace("_linkage.csv", "")
    stem  = base.rsplit("_", 2)[0]

    features, labels = load_dataset(stem, data_dir)
    has_data = features is not None and features.shape[1] >= 2

    if has_data:
        color_map = build_palette(labels)
        fig, (ax_scatter, ax_dend) = plt.subplots(1, 2, figsize=(18, 5))
        scatter_panel(ax_scatter, features, labels, color_map, stem)
        dendrogram_panel(ax_dend, Z, labels, color_map, base)
    else:
        fig, ax_dend = plt.subplots(figsize=(14, 5))
        dendrogram(Z, ax=ax_dend, no_labels=True)
        ax_dend.set_title(f"{base}  —  dendrogram")
        ax_dend.set_ylabel("distance")

    out_path = os.path.join(plots_dir, base + "_dendrogram.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path}")


if __name__ == "__main__":
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results"
    data_dir    = sys.argv[2] if len(sys.argv) > 2 else \
                  "data/Clustering-Datasets/02. Synthetic"
    plots_dir   = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(results_dir, "*_serial_linkage.csv")))
    if not files:
        print(f"No serial linkage CSV files found in {results_dir}/")
        sys.exit(1)

    for csv_path in files:
        try:
            plot_one(csv_path, plots_dir, data_dir)
        except Exception as e:
            print(f"  error on {os.path.basename(csv_path)}: {e}")
