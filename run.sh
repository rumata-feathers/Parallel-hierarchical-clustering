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
if command -v nvcc &> /dev/null; then CUDA=true; else CUDA=false; fi
RUNS=3      # number of repetitions per configuration
SERIAL_DIR="$RESULTS_DIR/serial"
PARALLEL_DIR="$RESULTS_DIR/parallel"
CUDA_DIR="$RESULTS_DIR/cuda"
LABELS_DIR="$RESULTS_DIR/labels"

# ---------------------------------------------------------------
# Progress bar helper
# Usage: draw_bar <current> <total> <label>
# Draws in-place using \r; call with a trailing \n after the loop.
# ---------------------------------------------------------------
draw_bar() {
  local cur=$1 tot=$2 label=$3
  local width=40
  local filled=$(( width * cur / tot ))
  local pct=$(( 100 * cur / tot ))
  local bar='' i
  for ((i = 0; i < width; i++)); do
    [[ $i -lt $filled ]] && bar+='█' || bar+='░'
  done
  printf '\r  [%s] %3d%%  %d/%d  %-45s' "$bar" "$pct" "$cur" "$tot" "$label"
}
# ---------------------------------------------------------------
# Average wall_ms from RUNS timing rows produced by repeated runs.
#
# The binary appends one CSV row per invocation to the mode timings file.
# We run RUNS times, collect the last RUNS rows, average wall_ms,
# then write a single averaged row to the output file.
#
# Usage: run_averaged <mode> <dataset> <linkage> <out_dir> [extra args...]
#
# The function writes one averaged CSV row to <out_dir>/timings.csv.
# ---------------------------------------------------------------
run_averaged() {
  local mode=$1
  local dataset=$2
  local linkage=$3
  local out_dir=$4
  shift 4
  local extra_args=("$@")

  # temporary file to collect raw rows from repeated runs
  local tmp
  tmp=$(mktemp)

  for (( r = 0; r < RUNS; r++ )); do
    ./build/hac \
      --dataset  "$dataset" \
      --linkage  "$linkage" \
      --mode     "$mode" \
      --out-dir  "$out_dir" \
      "${extra_args[@]}" > /dev/null
    # the binary appends a row to $out_dir/timings.csv; grab the last line
    tail -n 1 "$out_dir/timings.csv" >> "$tmp"
  done

  # parse the repeated rows and average wall_ms (column index depends on header)
  # header: dataset,linkage,mode,threads,n_points,wall_ms
  # we keep all fields from run 1 and replace wall_ms with the average
  python3 - "$tmp" <<'PYEOF'
import sys, csv, statistics
rows = []
with open(sys.argv[1]) as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(line.split(','))
if not rows:
    sys.exit(0)
# wall_ms is the last column
wall_vals = [float(r[-1]) for r in rows]
avg_wall = statistics.mean(wall_vals)
# use first row as template, replace wall_ms with average
out = rows[0][:-1] + [f"{avg_wall:.3f}"]
print(','.join(out))
PYEOF

  rm -f "$tmp"
}

# ---------------------------------------------------------------
# 0. Parse mode flag
# ---------------------------------------------------------------
MODE="${1:---full}"

case "$MODE" in
  --small)
    SMALL_SETS=(
      "zelnik6.csv"
      "zelnik1.csv"
      "3MC.csv"
      "target.csv"
      "2spiral.csv"
      "smile2.csv"
      "diamond9.csv"
      "complex9.csv"
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
# mkdir -p build && cd build && cmake .. && make -j$THREADS && cd ..
if ($CUDA); then
    echo "  with CUDA support"
    cmake -S . -B build -DENABLE_CUDA=ON && cmake --build build -j$THREADS
else
    echo "  without CUDA support"
    cmake -S . -B build && cmake --build build -j$THREADS
fi

# ---------------------------------------------------------------
# 2.5. Correctness tests
# ---------------------------------------------------------------
echo "Running correctness tests..."
./build/test_hac

# ---------------------------------------------------------------
# 3. Execute (results split by mode)
# ---------------------------------------------------------------
mkdir -p "$SERIAL_DIR" "$PARALLEL_DIR" "$LABELS_DIR"

# Remove stale timings so old-format rows (5 cols, no n_points) can't mix
# with new-format rows (6 cols) and break the CSV parser.
rm -f "$SERIAL_DIR/timings.csv" "$PARALLEL_DIR/timings.csv" \
      "$CUDA_DIR/timings.csv"  "$RESULTS_DIR/timings.csv"

# Calculate total number of HAC invocations for the progress bar
n_modes=2
[ "$CUDA" = true ] && n_modes=3
total_runs=$(( ${#csv_files[@]} * ${#LINKAGES[@]} * n_modes ))
current=0

echo "Running hac...  ($total_runs total runs)"
draw_bar 0 "$total_runs" "starting..."

# We collect averaged rows into temporary files then build timings.csv at the end.
SERIAL_TIMING_TMP=$(mktemp)
PARALLEL_TIMING_TMP=$(mktemp)
CUDA_TIMING_TMP=$(mktemp)

# Write header once (from any first real run)
HEADER_WRITTEN=false
for dataset in "${csv_files[@]}"; do
  stem=$(basename "$dataset" .csv)
  for linkage in "${LINKAGES[@]}"; do

    (( current++ )) || true
    draw_bar "$current" "$total_runs" "${stem}  [${linkage}]  serial"
    
    # first real run also writes --save-labels; subsequent runs skip it
    ./build/hac \
      --dataset  "$dataset" \
      --linkage  "$linkage" \
      --mode     serial \
      --out-dir  "$SERIAL_DIR" \
      --save-labels > /dev/null

    # write header from the serial timings file if not done yet
    if [ "$HEADER_WRITTEN" = false ]; then
      head -n 1 "$SERIAL_DIR/timings.csv" > "$SERIAL_TIMING_TMP"
      head -n 1 "$SERIAL_DIR/timings.csv" > "$PARALLEL_TIMING_TMP"
      [ "$CUDA" = true ] && head -n 1 "$SERIAL_DIR/timings.csv" > "$CUDA_TIMING_TMP"
      HEADER_WRITTEN=true
    fi
    
    # run RUNS times and average
    averaged=$(run_averaged serial "$dataset" "$linkage" "$SERIAL_DIR")
    echo "$averaged" >> "$SERIAL_TIMING_TMP"
  
  
    (( current++ )) || true
    draw_bar "$current" "$total_runs" "${stem}  [${linkage}]  parallel"
    averaged=$(run_averaged parallel "$dataset" "$linkage" "$PARALLEL_DIR" --threads "$THREADS")
    echo "$averaged" >> "$PARALLEL_TIMING_TMP"

    if [ "$CUDA" = true ]; then
      mkdir -p "$CUDA_DIR"
      (( current++ )) || true
      draw_bar "$current" "$total_runs" "${stem}  [${linkage}]  cuda"
      averaged=$(run_averaged cuda "$dataset" "$linkage" "$CUDA_DIR" 2>> "$CUDA_DIR/cuda_timing.log")
      echo "$averaged" >> "$CUDA_TIMING_TMP"

    fi

  done
done
printf '\n'  # end the progress bar line
# copy averaged timing files into place
cp "$SERIAL_TIMING_TMP"   "$SERIAL_DIR/timings.csv"
cp "$PARALLEL_TIMING_TMP" "$PARALLEL_DIR/timings.csv"
[ "$CUDA" = true ] && cp "$CUDA_TIMING_TMP" "$CUDA_DIR/timings.csv"
rm -f "$SERIAL_TIMING_TMP" "$PARALLEL_TIMING_TMP" "$CUDA_TIMING_TMP"

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

echo "==> Plotting CUDA copy vs compute breakdown..."
python3 scripts/plot_cuda_timing.py "$RESULTS_DIR"

echo ""
echo "==> All done."
echo "    Linkage matrices : $RESULTS_DIR/serial/  $RESULTS_DIR/parallel/"
echo "    Labels           : $RESULTS_DIR/labels/"
echo "    Timings          : $RESULTS_DIR/timings.csv"
echo "    Plots            : $RESULTS_DIR/plots/"

# crafted with the help from Claude
