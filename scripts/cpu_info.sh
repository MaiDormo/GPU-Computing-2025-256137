#!/bin/bash
#SBATCH --partition=edu-short
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=1
#SBATCH --time=00:02:00
#SBATCH --job-name=cpu-info
#SBATCH --output=cpu-info-%j.out
#SBATCH --error=cpu-info-%j.err
#SBATCH --nodelist=edu02

lscpu