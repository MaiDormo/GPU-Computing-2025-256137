#!/bin/bash
#SBATCH --partition=edu-short
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --gres=gpu:a30.24:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:05:00
#SBATCH --job-name=hybrid_adaptive_spmv
#SBATCH --output=hybrid_adaptive_spmv-%j.out
#SBATCH --error=hybrid_adaptive_spmv-%j.err
#SBATCH --nodelist=edu01

# Define executable and base directory
EXEC=~/GPU-Computing-2025-256137/bin/spmv_gpu_hybrid_adaptive_csr.exec
DATA_DIR=~/GPU-Computing-2025-256137/data

# Print header for results
echo "=================================================="
echo "SpMV Benchmark Results"
echo "=================================================="
echo "Started at: $(date)"
echo ""

# Define datasets to test
DATASETS=(
  "662_bus/662_bus.mtx"
  "Goodwin_127/Goodwin_127.mtx"
  "ML_Geer/ML_Geer.mtx"
  "Zd_Jac3_db/Zd_Jac3_db.mtx"
  "mawi_201512020330/mawi_201512020330.mtx"
  "CurlCurl_4/CurlCurl_4.mtx"
)

nvidia-smi

# Run benchmark for each dataset
for dataset in "${DATASETS[@]}"; do
  echo "------------------------------------------------"
  echo "Testing dataset: $dataset"
  echo "------------------------------------------------"
  srun $EXEC $DATA_DIR/$dataset
  echo ""
done

echo "=================================================="
echo "Benchmark completed at: $(date)"
echo "=================================================="