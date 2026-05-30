#!/usr/bin/env python3
"""
Reads results/timings.csv and produces four figures.
Works with any subset of modes (serial, parallel, cuda, …).
All comparisons are relative to 'serial' as the baseline.

  Figure 1 — Runtime scatter (log-log), one panel per non-serial mode
      x = serial ms, y = mode ms.  Point size ∝ log(n).  Below y=x = faster.

  Figure 2 — Speedup bars (sorted descending)
      Bar labels include n_points.  Red dashed line at 1× (break-even).

  Figure 3 — Speedup heatmap per non-serial mode
      Rows = datasets (with n=X), columns = linkages, colour = speedup.

  Figure 4 — Scaling: wall time and speedup vs n_points
      Shows how runtime and speedup grow with dataset size.
      Skipped gracefully if n_points column is absent (old CSV format).

Usage:
    python3 scripts/plot_timings.py [results_dir]
"""
import sys
import os

import numpy as np


def _bar(step: int, total: int, width: int = 40) -> str:
    filled = width * step // total if total else 0
    pct    = 100 * step // total   if total else 0
    return f"  [{'█' * filled}{'░' * (width - filled)}] {step}/{total}  ({pct}%)"
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

BASELINE       = "serial"
LINKAGE_ORDER  = ["single", "complete", "average", "ward", "centroid", "median"]
MODE_COLORS    = {"serial": "dimgray", "parallel": "steelblue", "cuda": "darkorange"}
MODE_COLOR_DEF = "mediumpurple"


def load_timings(path: str) -> pd.DataFrame:
    # on_bad_lines='skip' handles rows whose column count doesn't match the
    # header — this can happen when old 5-column timings (without n_points)
    # are merged with new 6-column timings in the same file.
    return pd.read_csv(path, on_bad_lines="skip")


def has_n_points(df: pd.DataFrame) -> bool:
    return "n_points" in df.columns


def non_serial_modes(df: pd.DataFrame) -> list[str]:
    present = set(df["mode"].unique()) - {BASELINE}
    order   = ["parallel", "cuda"]
    return [m for m in order if m in present] + sorted(present - set(order))


def make_speedup_table(df: pd.DataFrame, compare_mode: str) -> pd.DataFrame:
    """
    Returns DataFrame: [dataset, linkage, n_points?, serial_ms, <mode>_ms, speedup].
    Duplicate (dataset, linkage) rows (re-runs) are averaged.
    Rows where compare_mode exited near-instantly (unsupported linkage) are dropped.
    """
    base = (df[df["mode"] == BASELINE]
            .groupby(["dataset", "linkage"])["wall_ms"].mean()
            .rename("serial_ms"))
    comp = (df[df["mode"] == compare_mode]
            .groupby(["dataset", "linkage"])["wall_ms"].mean()
            .rename(f"{compare_mode}_ms"))
    merged = pd.concat([base, comp], axis=1).dropna()
    merged = merged[merged[f"{compare_mode}_ms"] > 0.5]
    merged["speedup"] = merged["serial_ms"] / merged[f"{compare_mode}_ms"]
    merged = merged.reset_index()

    if has_n_points(df):
        n_pts = df.groupby("dataset")["n_points"].first()
        merged = merged.join(n_pts, on="dataset")

    return merged


def linkage_palette() -> dict:
    cmap = plt.get_cmap("tab10")
    return {l: cmap(i) for i, l in enumerate(LINKAGE_ORDER)}


def _point_sizes(n_pts_series: pd.Series, lo: int = 30, hi: int = 200) -> np.ndarray:
    """Scale point sizes proportionally to log(n) in the range [lo, hi]."""
    log_n = np.log1p(n_pts_series.values.astype(float))
    rng   = log_n.max() - log_n.min()
    if rng == 0:
        return np.full(len(log_n), (lo + hi) / 2)
    return lo + (hi - lo) * (log_n - log_n.min()) / rng


# ---------------------------------------------------------------
# Figure 1: log-log scatter, one subplot per non-serial mode
# ---------------------------------------------------------------
def plot_scatter(df: pd.DataFrame, modes: list[str], plots_dir: str) -> None:
    n = len(modes)
    if n == 0:
        return

    use_n = has_n_points(df)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6.5), squeeze=False)
    palette   = linkage_palette()

    for col, mode in enumerate(modes):
        ax  = axes[0][col]
        tbl = make_speedup_table(df, mode)
        if tbl.empty:
            ax.text(0.5, 0.5, f"No data for '{mode}'",
                    ha="center", va="center", transform=ax.transAxes)
            continue

        for linkage, grp in tbl.groupby("linkage"):
            sizes = _point_sizes(grp["n_points"]) if use_n else 60
            ax.scatter(grp["serial_ms"], grp[f"{mode}_ms"],
                       color=palette.get(linkage, "gray"), label=linkage,
                       s=sizes, alpha=0.85, edgecolors="white", linewidths=0.4)

        lo = min(tbl["serial_ms"].min(), tbl[f"{mode}_ms"].min()) * 0.8
        hi = max(tbl["serial_ms"].max(), tbl[f"{mode}_ms"].max()) * 1.3
        xs = np.array([lo, hi])
        ax.plot(xs, xs,     color="black",    lw=1.3, ls="--", label="1× (no speedup)")
        ax.plot(xs, xs / 2, color="seagreen", lw=1.0, ls=":",  label="2× speedup")
        ax.plot(xs, xs / 4, color="seagreen", lw=0.8, ls="-.", label="4× speedup")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Serial wall time (ms)", fontsize=10)
        ax.set_ylabel(f"{mode.capitalize()} wall time (ms)", fontsize=10)
        size_note = "  (point size ∝ n)" if use_n else ""
        ax.set_title(f"Serial  vs  {mode}  (log–log){size_note}\nbelow dashed line = {mode} wins",
                     fontsize=10)
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.legend(title="linkage", fontsize=7, title_fontsize=7, loc="upper left")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, which="both", ls=":", alpha=0.4)

    fig.suptitle("Runtime comparison  (log–log scatter)", fontsize=12, y=1.01)
    fig.tight_layout()
    out = os.path.join(plots_dir, "timings_scatter.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out}")


# ---------------------------------------------------------------
# Figure 2: grouped speedup bars sorted descending, labels include n
# ---------------------------------------------------------------
def plot_speedup_bars(df: pd.DataFrame, modes: list[str], plots_dir: str) -> None:
    tables = {m: make_speedup_table(df, m) for m in modes}
    tables = {m: t for m, t in tables.items() if not t.empty}
    if not tables:
        return

    use_n      = has_n_points(df)
    first_mode = list(tables.keys())[0]
    base_tbl   = tables[first_mode].set_index(["dataset", "linkage"])
    index_sorted = base_tbl["speedup"].sort_values(ascending=False).index

    n_bars   = len(tables)
    n_groups = len(index_sorted)
    width    = 0.8 / n_bars
    xs       = np.arange(n_groups)

    # Build n_points lookup
    if use_n:
        n_pts_map = tables[first_mode].set_index("dataset")["n_points"].to_dict()
    else:
        n_pts_map = {}

    fig, ax = plt.subplots(figsize=(max(10, n_groups * 0.45), 5))

    for i, (mode, tbl) in enumerate(tables.items()):
        speedup_map = dict(zip(zip(tbl["dataset"], tbl["linkage"]), tbl["speedup"]))
        heights     = [speedup_map.get(idx, float("nan")) for idx in index_sorted]
        color       = MODE_COLORS.get(mode, MODE_COLOR_DEF)
        ax.bar(xs + i * width, heights, width=width, label=mode,
               color=color, edgecolor="white", linewidth=0.4, alpha=0.85)

        if n_groups <= 40:
            for x, h in zip(xs + i * width, heights):
                if not np.isnan(h):
                    ax.text(x, h + 0.03, f"{h:.2f}×",
                            ha="center", va="bottom", fontsize=5.5)

    ax.axhline(1.0, color="red", lw=1.3, ls="--", label="break-even (1×)")

    if use_n:
        bar_labels = [f"{ds}\n{lk}\nn={int(n_pts_map.get(ds, 0))}"
                      for ds, lk in index_sorted]
    else:
        bar_labels = [f"{ds}\n{lk}" for ds, lk in index_sorted]

    ax.set_xticks(xs + (n_bars - 1) * width / 2)
    ax.set_xticklabels(bar_labels, rotation=45, ha="right", fontsize=6.5)
    ax.set_ylabel("Speedup  (serial / mode)", fontsize=10)
    ax.set_title("Speedup over serial — sorted by highest speedup", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(axis="y", ls=":", alpha=0.4)
    fig.tight_layout()

    out = os.path.join(plots_dir, "timings_speedup.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out}")


# ---------------------------------------------------------------
# Figure 3: speedup heatmap with n_points in row labels
# ---------------------------------------------------------------
def plot_speedup_heatmap(df: pd.DataFrame, modes: list[str], plots_dir: str) -> None:
    n = len(modes)
    if n == 0:
        return

    use_n = has_n_points(df)
    fig, axes = plt.subplots(1, n, figsize=(8 * n, 4), squeeze=False)

    for col, mode in enumerate(modes):
        ax  = axes[0][col]
        tbl = make_speedup_table(df, mode)
        if tbl.empty:
            ax.text(0.5, 0.5, f"No data for '{mode}'",
                    ha="center", va="center", transform=ax.transAxes)
            continue

        pivot = tbl.pivot_table(index="dataset", columns="linkage",
                                values="speedup", aggfunc="mean")
        cols  = [c for c in LINKAGE_ORDER if c in pivot.columns]
        pivot = pivot[cols]
        vals  = pivot.values.astype(float)
        n_rows = len(pivot)
        fig.set_figheight(max(4, n_rows * 0.38))

        vmax = max(2.0, float(np.nanmax(vals)))
        im = ax.imshow(vals, aspect="auto", cmap="RdYlGn", vmin=0.5, vmax=vmax)

        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(cols, fontsize=9)
        ax.set_yticks(range(n_rows))

        if use_n:
            n_pts_map = tbl.set_index("dataset")["n_points"].to_dict()
            row_labels = [f"{ds}  (n={int(n_pts_map.get(ds, 0))})"
                          for ds in pivot.index]
        else:
            row_labels = list(pivot.index)
        ax.set_yticklabels(row_labels, fontsize=7)

        ax.set_title(f"Speedup heatmap: serial / {mode}\n"
                     f"(green > 1 = {mode} wins)", fontsize=10)
        fig.colorbar(im, ax=ax, label="speedup", shrink=0.6)

        for i in range(n_rows):
            for j in range(len(cols)):
                v = vals[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}",
                            ha="center", va="center", fontsize=6)

    fig.suptitle("Speedup heatmaps", fontsize=12, y=1.01)
    fig.tight_layout()
    out = os.path.join(plots_dir, "timings_heatmap.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out}")


# ---------------------------------------------------------------
# Figure 4: scaling — wall time and speedup vs n_points
# ---------------------------------------------------------------
def plot_scaling(df: pd.DataFrame, modes: list[str], plots_dir: str) -> None:
    if not has_n_points(df):
        print("  n_points column not in CSV — skipping scaling plot "
              "(delete results/timings.csv and re-run to regenerate)")
        return

    all_modes = [BASELINE] + modes

    # Average wall_ms over linkages for each (dataset, mode, n_points)
    agg = (df[df["mode"].isin(all_modes)]
           .groupby(["dataset", "mode", "n_points"])["wall_ms"]
           .mean()
           .reset_index()
           .sort_values("n_points"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ---- Left: wall time vs n ----
    for mode in all_modes:
        sub   = agg[agg["mode"] == mode]
        color = MODE_COLORS.get(mode, MODE_COLOR_DEF)
        ax1.plot(sub["n_points"], sub["wall_ms"], "o-",
                 color=color, label=mode, lw=1.8, ms=5, alpha=0.9)

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("n  (number of points)", fontsize=11)
    ax1.set_ylabel("Wall time (ms, avg over linkages)", fontsize=11)
    ax1.set_title("Runtime scaling  (log–log)", fontsize=11)
    ax1.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax1.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax1.legend(fontsize=9)
    ax1.grid(True, which="both", ls=":", alpha=0.4)

    # ---- Right: speedup vs n ----
    serial_agg = agg[agg["mode"] == BASELINE].set_index(["dataset", "n_points"])["wall_ms"]

    for mode in modes:
        mode_agg = agg[agg["mode"] == mode].set_index(["dataset", "n_points"])["wall_ms"]
        common   = serial_agg.index.intersection(mode_agg.index)
        if common.empty:
            continue
        speedup = (serial_agg.loc[common] / mode_agg.loc[common]).reset_index()
        speedup.columns = ["dataset", "n_points", "speedup"]
        speedup = speedup.sort_values("n_points")
        color   = MODE_COLORS.get(mode, MODE_COLOR_DEF)
        ax2.plot(speedup["n_points"], speedup["speedup"], "o-",
                 color=color, label=mode, lw=1.8, ms=5, alpha=0.9)

    ax2.axhline(1.0, color="red", lw=1.2, ls="--", label="break-even")
    ax2.set_xscale("log")
    ax2.set_xlabel("n  (number of points)", fontsize=11)
    ax2.set_ylabel("Speedup  (serial / mode, avg over linkages)", fontsize=11)
    ax2.set_title("Speedup vs n", fontsize=11)
    ax2.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax2.legend(fontsize=9)
    ax2.grid(True, which="both", ls=":", alpha=0.4)

    fig.tight_layout()
    out = os.path.join(plots_dir, "timings_scaling.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out}")

# Figure 5: grouped bar chart — wall time per linkage (avg over datasets)
# One group per linkage, one bar per mode (serial / parallel / cuda).
def plot_grouped_bar(df: pd.DataFrame, modes: list[str], plots_dir: str) -> None:
    all_modes = [BASELINE] + modes
 
    avg = (df[df["mode"].isin(all_modes)]
           .groupby(["linkage", "mode"])["wall_ms"]
           .mean()
           .reset_index())
    if avg.empty:
        return
 
    linkages = [l for l in LINKAGE_ORDER if l in avg["linkage"].unique()]
    x = np.arange(len(linkages))
    n_modes = len(all_modes)
    w = 0.8 / n_modes
 
    fig, ax = plt.subplots(figsize=(max(10, len(linkages) * 1.8), 5.5))
 
    for i, mode in enumerate(all_modes):
        vals = []
        for l in linkages:
            cell = avg.loc[(avg.linkage == l) & (avg["mode"] == mode), "wall_ms"]
            vals.append(cell.values[0] if len(cell) else np.nan)
        color = MODE_COLORS.get(mode, MODE_COLOR_DEF)
        bars = ax.bar(x + i * w, vals, w, label=mode,
                      color=color, alpha=0.87, edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() * 1.02,
                        f"{v:.0f}", ha="center", va="bottom", fontsize=6.5)
 
    ax.set_yscale("log")
    ax.set_xticks(x + (n_modes - 1) * w / 2)
    ax.set_xticklabels(linkages, fontsize=10)
    ax.set_ylabel("Wall time (ms, avg over datasets, log scale)", fontsize=10)
    ax.set_title("Wall time by linkage:  serial vs parallel vs cuda", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(axis="y", which="both", ls=":", alpha=0.4)
    fig.tight_layout()
 
    out = os.path.join(plots_dir, "timings_grouped_bar.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out}")
# Figure 6: thread scaling — speedup vs thread count
# Reads results/thread_scaling.csv (separate experiment, not timings.csv).
GRAPH_LINKAGES = ["single", "complete", "average"]
GEOM_LINKAGES  = ["ward", "centroid", "median"]
 
 
def plot_thread_scaling(results_dir: str, plots_dir: str) -> None:
    path = os.path.join(results_dir, "thread_scaling.csv")
    if not os.path.exists(path):
        print(f"  no {path} — skipping thread-scaling plot "
              f"(run scripts/thread_sweep.sh to generate)")
        return
 
    df = pd.read_csv(path, on_bad_lines="skip")
    if df.empty or "threads" not in df.columns:
        print("  thread_scaling.csv empty or malformed — skipping")
        return
 
    # average over any duplicate runs
    agg = (df.groupby(["linkage", "threads"])["wall_ms"]
           .mean().reset_index())
 
    threads = sorted(agg["threads"].unique())
    dataset = df["dataset"].iloc[0] if "dataset" in df.columns else "dataset"
    n_pts   = int(df["n_points"].iloc[0]) if "n_points" in df.columns else None
 
    palette = linkage_palette()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
 
    def draw(ax, linkages, title):
        for lk in linkages:
            sub = agg[agg["linkage"] == lk].sort_values("threads")
            if sub.empty:
                continue
            t1 = sub.loc[sub["threads"] == sub["threads"].min(), "wall_ms"].values[0]
            speedup = t1 / sub["wall_ms"].values
            ax.plot(sub["threads"].values, speedup, "o-",
                    color=palette.get(lk, "gray"), label=lk, lw=2, ms=7)
        ax.plot(threads, threads, "k--", lw=1.2, alpha=0.4, label="ideal (linear)")
        ax.axhline(1.0, color="red", lw=1, ls=":", alpha=0.5)
        ax.set_xscale("log", base=2)
        ax.set_xticks(threads)
        ax.set_xticklabels([str(int(t)) for t in threads])
        ax.set_xlabel("Number of threads", fontsize=11)
        ax.set_ylabel("Speedup over 1 thread", fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, which="both", ls=":", alpha=0.4)
 
    draw(ax1, GRAPH_LINKAGES, "Graph linkages (single / complete / average)")
    draw(ax2, GEOM_LINKAGES,  "Geometric linkages (ward / centroid / median)")
 
    suffix = f"  (n={n_pts})" if n_pts else ""
    fig.suptitle(f"Thread scaling — {dataset}{suffix}", fontsize=13)
    fig.tight_layout()
 
    out = os.path.join(plots_dir, "timings_thread_scaling.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out}")
# ---------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------
if __name__ == "__main__":
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results"
    timing_path = os.path.join(results_dir, "timings.csv")
    plots_dir   = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    if not os.path.exists(timing_path):
        print(f"No timings file at {timing_path} — run the pipeline first.")
        sys.exit(1)

    df    = load_timings(timing_path)
    modes = non_serial_modes(df)

    if not modes:
        print("Only serial timings found — nothing to compare.")
        sys.exit(0)
    charts = [
        ("grouped bar (wall time)",   lambda: plot_grouped_bar(df, modes, plots_dir)),
        ("speedup heatmap",           lambda: plot_speedup_heatmap(df, modes, plots_dir)),
        ("scaling (wall time vs n)",  lambda: plot_scaling(df, modes, plots_dir)),
        ("scatter (log-log)",         lambda: plot_scatter(df, modes, plots_dir)),
        ("speedup bars",              lambda: plot_speedup_bars(df, modes, plots_dir)),
        ("thread scaling",            lambda: plot_thread_scaling(results_dir, plots_dir)),
    ] 

    total = len(charts)
    print(f"Modes detected: {[BASELINE] + modes}")
    print(f"Plotting {total} timing charts...\n")

    for step, (label, fn) in enumerate(charts, 1):
        print(f"[{step}/{total}] {label}")
        fn()
        print(_bar(step, total))
