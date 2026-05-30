#!/usr/bin/env bash
set -euo pipefail
# thread scaling
DATASET="data/Clustering-Datasets/02. Synthetic/3MC.csv"
OUT="results/thread_scaling.csv"
LINKAGES=(single complete average ward centroid median)
THREAD_COUNTS=(1 2 4 8 16)
echo "dataset,linkage,mode,threads,n_points,wall_ms" > "$OUT"
for t in "${THREAD_COUNTS[@]}"; do
  for lk in "${LINKAGES[@]}"; do
    echo "  threads=$t  linkage=$lk"
    ./build/hac --dataset "$DATASET" --linkage "$lk" \
      --mode parallel --threads "$t" \
      --out-dir results/parallel >/dev/null
    tail -n 1 results/parallel/timings.csv >> "$OUT"
  done
done
echo "Done -> $OUT"