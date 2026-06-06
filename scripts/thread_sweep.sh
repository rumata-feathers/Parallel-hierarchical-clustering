#!/usr/bin/env bash
set -euo pipefail
# thread scaling
DATASET="data/Clustering-Datasets/02. Synthetic/3MC.csv"
OUT="results/thread_scaling.csv"
LINKAGES=(single complete average ward centroid median)
THREAD_COUNTS=(1 2 4 8 16)
RUNS=3
echo "dataset,linkage,mode,threads,n_points,wall_ms" > "$OUT"
for t in "${THREAD_COUNTS[@]}"; do
  for lk in "${LINKAGES[@]}"; do
    echo "  threads=$t  linkage=$lk  (${RUNS} runs)"
    
    wall_vals=()
    for (( r = 0; r < RUNS; r++ )); do
    ./build/hac --dataset "$DATASET" --linkage "$lk" \
      --mode parallel --threads "$t" \
      --out-dir results/parallel >/dev/null
    val=$(tail -n 1 results/parallel/timings.csv | awk -F',' '{print $NF}')
    wall_vals+=("$val")
  done

    # average via python one-liner
    avg=$(python3 -c "
vals = [$(IFS=,; echo "${wall_vals[*]}")]
print(f'{sum(vals)/len(vals):.3f}')
")

    # grab a representative row (last run) and replace wall_ms with the average
    last_row=$(tail -n 1 results/parallel/timings.csv)
    prefix=$(echo "$last_row" | awk -F',' 'BEGIN{OFS=","}{NF--; print}')
    echo "${prefix},${avg}" >> "$OUT"
  done
done

echo "Done -> $OUT  (each value is mean of $RUNS runs)"