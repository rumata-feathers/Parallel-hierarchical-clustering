#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="data/Clustering-Datasets/02. Synthetic"

RESULTS_DIR="results"
THREADS=4
LINKAGES=(single complete average ward centroid median)
CUDA=false   # set to true if built with -DENABLE_CUDA=ON

# ---------------------------------------------------------------
# 0. Setup Python environment
# ---------------------------------------------------------------
echo "Setting up Python environment..."
ENV_DIR="env"
if [ ! -d "$ENV_DIR" ]; then
  python3 -m venv "$ENV_DIR"
  echo "Python virtual environment created at $ENV_DIR"
fi

# Activate the environment
source "$ENV_DIR/bin/activate"
echo "Python environment activated."

# ---------------------------------------------------------------
# 1. Build
# ---------------------------------------------------------------

echo "Building..."
# mkdir -p build && cd build && cmake .. && make -j$THREADS && cd ..
cmake -S . -B build && cmake --build build -j$THREADS

# ---------------------------------------------------------------
# 2. Execute
# ---------------------------------------------------------------
echo "Running hac..."
mkdir -p "$RESULTS_DIR"

shopt -s nullglob
csv_files=("$DATA_DIR"/*.csv)
echo "Datasets found: ${#csv_files[@]}"

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

        if [ "$CUDA" = true ]; then
            echo "  $(basename "$dataset")  [$linkage]  cuda..."
            ./build/hac \
                --dataset  "$dataset" \
                --linkage  "$linkage" \
                --mode     cuda \
                --out-dir  "$RESULTS_DIR"
        fi
    done
done

# ---------------------------------------------------------------
# 3. Plot
# ---------------------------------------------------------------
echo "==> Installing Python dependencies..."
pip install -r scripts/requirements.txt

echo "==> Plotting dendrograms..."
python3 scripts/plot_dendrogram.py "$RESULTS_DIR"

echo "==> Plotting timings..."
python3 scripts/plot_timings.py "$RESULTS_DIR"

echo ""
echo "==> All done. Results: $RESULTS_DIR/  |  Plots: $RESULTS_DIR/plots/"

# crafted with the help from Claude