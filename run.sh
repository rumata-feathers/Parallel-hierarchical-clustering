#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="data/Clustering-Datasets/02. Synthetic"
RESULTS_DIR="results"
THREADS=4
LINKAGES=(single complete average ward centroid median)


# ---------------------------------------------------------------
# 1. Build
# ---------------------------------------------------------------

echo "Building..."
mkdir -p build && cd build && cmake .. && make -j$THREADS && cd ..


# ---------------------------------------------------------------
# 2. Execute
# ---------------------------------------------------------------
echo "Running hac..."
mkdir -p "$RESULTS_DIR"

shopt -s nullglob
csv_files=("$DATA_DIR"/*.csv)
if [ ${#csv_files[@]} -eq 0 ]; then
    echo "ERROR: No CSV files found in $DATA_DIR/"
    echo "       Run: git submodule update --init"
    exit 1
fi

for dataset in "${csv_files[@]}"; do
    for linkage in "${LINKAGES[@]}"; do
        echo "  $(basename "$dataset")  [$linkage]  serial..."
        ./build/hac \
            --dataset  "$dataset" \
            --linkage  "$linkage" \
            --mode     serial \
            --out-dir  "$RESULTS_DIR" \
            --save-labels

        echo "  $(basename "$dataset")  [$linkage]  parallel (${THREADS} threads)..."
        ./build/hac \
            --dataset  "$dataset" \
            --linkage  "$linkage" \
            --mode     parallel \
            --threads  "$THREADS" \
            --out-dir  "$RESULTS_DIR"
    done
done

# ---------------------------------------------------------------
# 3. Plot
# ---------------------------------------------------------------
echo "==> Installing Python dependencies..."
pip install -q -r scripts/requirements.txt

echo "==> Plotting dendrograms..."
python3 scripts/plot_dendrogram.py "$RESULTS_DIR"

echo "==> Plotting timings..."
python3 scripts/plot_timings.py "$RESULTS_DIR"

echo ""
echo "==> All done. Results: $RESULTS_DIR/  |  Plots: $RESULTS_DIR/plots/"

# crafted with the help from Claude