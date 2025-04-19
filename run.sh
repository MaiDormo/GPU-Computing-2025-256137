#!/bin/bash
#SBATCH --partition=edu-short
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --job-name=test
#SBATCH --output=test-%j.out
#SBATCH --error=test-%j.err
srun ~/GPU-Computing-2025-256137/bin/spmv_gpu_csr.exec data/mawi_201512020330/mawi_201512020330.mtx

# srun ~/GPU-Computing-2025-256137/bin/spmv_gpu_csr.exec data/662_bus/662_bus.mtx
