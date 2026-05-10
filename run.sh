# run.sh
set -e
# amount  of threads to use for parallel execution - 8
export NUM_THREADS=8
mkdir -p build && cd build && cmake .. && make -j$NUM_THREADS && cd ..

## run cross datasets
for dataset in data/*.csv; do
  for linkage in single complete average ward; do
    ./build/hac --dataset "$dataset" --linkage "$linkage" --mode serial
    ./build/hac --dataset "$dataset" --linkage "$linkage" --mode parallel
  done
done

## plt
python3 scripts/plot_dendrogram.py results/
python3 scripts/plot_timings.py results/

## clean up
if [ -d "build" ]; then
  cd build && make clean && cd ..
fi
rm -rf build