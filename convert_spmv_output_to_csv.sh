#!/bin/bash
# filepath: convert_spmv_output_to_csv.sh

# Usage: ./convert_spmv_output_to_csv.sh experiment_spmv_benchmark-27153.out > results.csv

INPUT_FILE="$1"

echo "block_num,block_size,columns,rows,nnz,avg_nnz_per_row,percentage_nnz,avg_time,bandwidth,gflops"

awk '
/^BLOCK_NUM=/ {
    split($0, a, "[ =]");
    block_num = a[2];
    block_size = a[4];
}
/^Number of Columns:/ { columns = $4 }
/^Number of Rows:/ { rows = $4 }
/^Number of NNZ:/ { nnz = $4 }
/^Average NNZ per Row:/ { avg_nnz_per_row = $5 }
/^Percentage of NNZ:/ { percentage_nnz = $4 }
/^Average execution time:/ { avg_time = $4 }
/^Memory bandwidth \(estimated\):/ { bandwidth = $4 }
/^Computational performance:/ { gflops = $3 }
/^First few non-zero elements of result vector:/ {
    # Remove possible trailing unit from gflops
    split(gflops, g, " "); gflops_val = g[1];
    print block_num "," block_size "," columns "," rows "," nnz "," avg_nnz_per_row "," percentage_nnz "," avg_time "," bandwidth "," gflops_val
}
' "$INPUT_FILE"