#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------
# Usage: ./run.sh [--small | --full]
#   --small  quick smoke-test on synthetic datasets ≤ 500 rows
#   --full   full benchmark across all synthetic datasets (default)
#
# Output layout:
#   results/
#     serial/          linkage matrices from serial runs
#     parallel/        linkage matrices from parallel runs
#     cuda/            linkage matrices from CUDA runs (if enabled)
#     labels/          true-label CSVs (one per dataset)
#     timings.csv      merged timings from all modes (for plotting)
#     plots/           dendrograms + timing charts
# ---------------------------------------------------------------

SYNTH_DIR="data/Clustering-Datasets/02. Synthetic"
RESULTS_DIR="results"
THREADS=4
LINKAGES=(single complete average ward centroid median)
CUDA=false   # set to true if built with -DENABLE_CUDA=ON

SERIAL_DIR="$RESULTS_DIR/serial"
PARALLEL_DIR="$RESULTS_DIR/parallel"
CUDA_DIR="$RESULTS_DIR/cuda"
LABELS_DIR="$RESULTS_DIR/labels"

# ---------------------------------------------------------------
# 0. Parse mode flag
# ---------------------------------------------------------------
MODE="${1:---full}"

case "$MODE" in
  --small)
    SMALL_SETS=(
      "insect.csv"
      "gaussians1.csv"
      "hepta.csv"
      "zelnik6.csv"
      "flame.csv"
      "spherical_5_2.csv"
      "pearl.csv"
      "zelnik3.csv"
      "zelnik1.csv"
      "blobs.csv"
      "pathbased.csv"
      "spherical_6_2.csv"
      "zelnik2.csv"
      "3-spiral.csv"
      "jain.csv"
      "compound.csv"
      "3MC.csv"
    )
    csv_files=()
    for name in "${SMALL_SETS[@]}"; do
      f="$SYNTH_DIR/$name"
      [ -f "$f" ] && csv_files+=("$f")
    done
    echo "Mode: --small  (${#csv_files[@]} datasets, ≤500 rows each)"
    ;;
  --full)
    shopt -s nullglob
    csv_files=("$SYNTH_DIR"/*.csv)
    echo "Mode: --full  (${#csv_files[@]} datasets)"
    ;;
  *)
    echo "Usage: $0 [--small | --full]"
    exit 1
    ;;
esac

if [ ${#csv_files[@]} -eq 0 ]; then
  echo "ERROR: No CSV files found. Run: git submodule update --init"
  exit 1
fi

# ---------------------------------------------------------------
# 1. Setup Python environment
# ---------------------------------------------------------------
echo "Setting up Python environment..."
ENV_DIR="env"
if [ ! -d "$ENV_DIR" ]; then
  python3 -m venv "$ENV_DIR"
  echo "Python virtual environment created at $ENV_DIR"
fi
source "$ENV_DIR/bin/activate"

# ---------------------------------------------------------------
# 2. Build
# ---------------------------------------------------------------
echo "Building..."
cmake -S . -B build && cmake --build build -j$THREADS

# ---------------------------------------------------------------
# 2.5. Correctness tests
# ---------------------------------------------------------------
echo "Running correctness tests..."
./build/test_hac

# ---------------------------------------------------------------
# 3. Execute (results split by mode)
# ---------------------------------------------------------------
echo "Running hac..."
mkdir -p "$SERIAL_DIR" "$PARALLEL_DIR" "$LABELS_DIR"

# Remove stale timings so old-format rows (5 cols, no n_points) can't mix
# with new-format rows (6 cols) and break the CSV parser.
rm -f "$SERIAL_DIR/timings.csv" "$PARALLEL_DIR/timings.csv" \
      "$CUDA_DIR/timings.csv"  "$RESULTS_DIR/timings.csv"

for dataset in "${csv_files[@]}"; do
  for linkage in "${LINKAGES[@]}"; do
    echo "  $(basename "$dataset")  [$linkage]  serial..."
    ./build/hac \
      --dataset  "$dataset" \
      --linkage  "$linkage" \
      --mode     serial \
      --out-dir  "$SERIAL_DIR" \
      --save-labels

    echo "  $(basename "$dataset")  [$linkage]  parallel (${THREADS} threads)..."
    ./build/hac \
      --dataset  "$dataset" \
      --linkage  "$linkage" \
      --mode     parallel \
      --threads  "$THREADS" \
      --out-dir  "$PARALLEL_DIR"

    if [ "$CUDA" = true ]; then
      mkdir -p "$CUDA_DIR"
      echo "  $(basename "$dataset")  [$linkage]  cuda..."
      ./build/hac \
        --dataset  "$dataset" \
        --linkage  "$linkage" \
        --mode     cuda \
        --out-dir  "$CUDA_DIR"
    fi
  done
done

# ---------------------------------------------------------------
# 4. Reorganise: move labels, merge timings
# ---------------------------------------------------------------

# Labels were written alongside serial matrices — move them to labels/
mv "$SERIAL_DIR"/*_labels.csv "$LABELS_DIR"/ 2>/dev/null || true

# Merge per-mode timings.csv into a single top-level file for plotting
{
  [ -f "$SERIAL_DIR/timings.csv" ]   && cat "$SERIAL_DIR/timings.csv"
  [ -f "$PARALLEL_DIR/timings.csv" ] && tail -n +2 "$PARALLEL_DIR/timings.csv"
  [ -f "$CUDA_DIR/timings.csv" ]     && tail -n +2 "$CUDA_DIR/timings.csv"
} > "$RESULTS_DIR/timings.csv"

# ---------------------------------------------------------------
# 5. Plot
# ---------------------------------------------------------------
echo "==> Installing Python dependencies..."
pip install -r scripts/requirements.txt

echo "==> Plotting dendrograms..."
python3 scripts/plot_dendrogram.py "$RESULTS_DIR"

echo "==> Plotting timings..."
python3 scripts/plot_timings.py "$RESULTS_DIR"

echo ""
echo "==> All done."
echo "    Linkage matrices : $RESULTS_DIR/serial/  $RESULTS_DIR/parallel/"
echo "    Labels           : $RESULTS_DIR/labels/"
echo "    Timings          : $RESULTS_DIR/timings.csv"
echo "    Plots            : $RESULTS_DIR/plots/"

# crafted with the help from Claude
