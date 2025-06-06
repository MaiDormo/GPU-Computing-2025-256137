#!/bin/bash

if [ $# -eq 0 ]; then
  echo "Usage: $0 filename.cu"
  exit 1
fi

module load CUDA/12.3.2

filename=$(basename -- "$1")
name="${filename%.*}"

nvcc -O3 --use_fast_math --gpu-architecture=sm_80 -m64 \
     -Xcompiler "-Wall -Wextra -O3" \
     -o "$name.exec" "$1" -lcusparse
echo "Compiled $1 into $name.exec"
