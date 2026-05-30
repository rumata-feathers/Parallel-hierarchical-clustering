#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/thread_scaling.csv")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

for linkage, group in df.groupby("linkage"):
    group_sorted = group.sort_values("threads")
    ax1.plot(group_sorted["threads"], group_sorted["wall_ms"], marker="o", label=linkage)
ax1.set_xlabel("Threads")
ax1.set_ylabel("Wall time (ms)")
ax1.set_title("Wall Time vs Thread Count")
ax1.set_xticks([1, 2, 4, 8, 16])
ax1.legend()
ax1.grid(True)


for linkage, group in df.groupby("linkage"):
    group_sorted = group.sort_values("threads")
    baseline = group_sorted[group_sorted["threads"] == 1]["wall_ms"].values[0]
    speedup = baseline / group_sorted["wall_ms"]
    ax2.plot(group_sorted["threads"], speedup, marker="o", label=linkage)
ax2.plot([1, 16], [1, 16], "k--", alpha=0.3, label="ideal")
ax2.set_xlabel("Threads")
ax2.set_ylabel("Speedup")
ax2.set_title("Speedup vs Thread Count")
ax2.set_xticks([1, 2, 4, 8, 16])
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig("results/plots/thread_scaling.png", dpi=150)
print("saved: results/plots/thread_scaling.png")