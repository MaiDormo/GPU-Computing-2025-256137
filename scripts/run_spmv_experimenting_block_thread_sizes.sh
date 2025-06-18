#!/bin/bash
#SBATCH --partition=edu-medium
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --gres=gpu:a30.24:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --job-name=experiment_spmv_benchmark
#SBATCH --output=experiment_spmv_benchmark-%j.out
#SBATCH --error=experiment_spmv_benchmark-%j.err
#SBATCH --nodelist=edu01

# Set matrix file (edit as needed)
MATRIX_FILE=~/GPU-Computing-2025-256137/data/Goodwin_127/Goodwin_127.mtx
EXEC=~/GPU-Computing-2025-256137/bin/spmv_experimenting_block_thread_sizes.exec

# Block sizes and block numbers to test
BLOCK_SIZES=(64 128 256 512 1024)
BLOCK_NUMS=(1 2 4 8 16 32 64 128)

echo "Experiment: SpMV Block Size and Block Num Sweep"
echo "Matrix: $MATRIX_FILE"
echo "Date: $(date)"
echo "Hostname: $(hostname)"
nvidia-smi

for block_num in "${BLOCK_NUMS[@]}"; do
    for block_size in "${BLOCK_SIZES[@]}"; do
        echo "------------------------------------------------------------"
        echo "BLOCK_NUM=$block_num BLOCK_SIZE=$block_size"
        echo "------------------------------------------------------------"
        srun "$EXEC" "$MATRIX_FILE" "$block_size" "$block_num"
        echo ""
    done
done

echo "All experiments completed at $(date)."