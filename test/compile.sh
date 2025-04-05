#!/bin/bash



if [ $# -eq 0 ]; then
  echo "Usage: $0 filename.cu"
  exit 1
fi

module load cuda/12.1

filename=$(basename -- "$1")
name="${filename%.*}"

nvcc --gpu-architecture=sm_80 -m64 -o "$name.exec" "$1"
echo "Compiled $1 into $name.exec"