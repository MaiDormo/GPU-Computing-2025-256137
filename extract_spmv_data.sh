#!/bin/bash
# Script to extract SpMV benchmark data from various implementation output files
# Usage: ./extract_spmv_data.sh [output_file.csv]

OUTPUT_CSV="${1:-spmv_comprehensive_results.csv}"

# Create CSV header
echo "Platform,Implementation,Dataset,Matrix_Name,Rows,Columns,NNZ,Mean_NNZ_Per_Row,Min_NNZ_Per_Row,Max_NNZ_Per_Row,Median_NNZ_Per_Row,Std_Dev_NNZ,Sparsity_Ratio,Percentage_NNZ,Block_Size,Num_Blocks,Warps_Per_Block,Shared_Memory_Bytes,Execution_Time_Seconds,Bandwidth_GB_s,Performance_GFLOPS" > "$OUTPUT_CSV"

# Function to process each benchmark file
process_benchmark_file() {
    local file=$1
    local filename=$(basename "$file")
    
    echo "Processing $file..."
    
    # Determine platform (CPU or GPU)
    local platform="CPU"
    if grep -q "NVIDIA-SMI\|nvidia-smi" "$file"; then
        platform="GPU"
    fi
    
    # Extract implementation type from filename
    local implementation=""
    if [[ "$filename" =~ vector_spmv ]]; then
        implementation="vector"
    elif [[ "$filename" =~ adaptive_spmv ]]; then
        implementation="adaptive"
    elif [[ "$filename" =~ simple_spmv ]]; then
        implementation="simple"
    elif [[ "$filename" =~ value_sequential_spmv ]]; then
        implementation="value_sequential"
    elif [[ "$filename" =~ value_blocked_spmv ]]; then
        implementation="value_blocked"
    elif [[ "$filename" =~ cpu_([^_]+)_spmv ]]; then
        implementation="cpu_${BASH_REMATCH[1]}"
    elif [[ "$filename" =~ experiment_spmv ]]; then
        implementation="experiment"
    else
        # Extract from filename pattern
        implementation=$(echo "$filename" | sed -E 's/.*_([^_]+)_spmv_benchmark.*/\1/' | sed 's/_benchmark.*//')
        if [[ "$implementation" == "$filename" ]]; then
            implementation="unknown"
        fi
    fi
    
    # Process the file using AWK
    awk -v platform="$platform" -v impl="$implementation" '
    BEGIN {
        dataset = ""; matrix_name = "";
        rows = ""; cols = ""; nnz = ""; mean_nnz = ""; min_nnz = ""; max_nnz = ""; median_nnz = "";
        std_dev = ""; sparsity = ""; perc_nnz = "";
        block_size = ""; num_blocks = ""; warps_per_block = ""; shared_mem = "";
        time = ""; bandwidth = ""; gflops = "";
        in_dataset = 0;
        
        # For experiment files
        exp_block_num = ""; exp_block_size = "";
    }
    
    # Handle experiment format
    /^BLOCK_NUM=/ {
        split($0, a, "[ =]");
        exp_block_num = a[2];
        exp_block_size = a[4];
    }
    
    # Dataset detection
    /^Testing dataset:/ {
        # Output previous dataset if we have complete data
        if (in_dataset && rows != "" && time != "") {
            output_csv_line();
        }
        
        # Extract new dataset
        dataset = $3;
        gsub(/\/.*\.mtx/, "", dataset);  # Remove path and extension
        matrix_name = dataset;
        
        # Reset all variables
        reset_variables();
        in_dataset = 1;
    }
    
    # Matrix analysis line (for optimal launch config)
    /^Matrix analysis: mean_nnz_per_row/ {
        mean_nnz = $4;
    }
    
    # Launch configuration
    /^Optimal launch config:/ {
        num_blocks = $4;
        block_size = $6;
        if ($0 ~ /warps per block:/) {
            match($0, /warps per block: ([0-9]+)/, arr);
            warps_per_block = arr[1];
        }
    }
    
    /^Final launch config:/ {
        num_blocks = $4;
        block_size = $7;
        if ($0 ~ /bytes shared memory/) {
            match($0, /([0-9]+) bytes shared memory/, arr);
            shared_mem = arr[1];
        }
    }
    
    # Matrix Statistics section
    /^Matrix dimensions:/ {
        rows = $3;
        gsub(/x/, "", rows);
        cols = $5;
    }
    
    /^Total NNZ:/ { nnz = $3; }
    /^Mean NNZ per Row:/ { mean_nnz = $5; }
    /^Min NNZ per Row:/ { 
        min_nnz = $5;
        gsub(/,/, "", min_nnz);
        if ($0 ~ /Max NNZ per Row:/) {
            match($0, /Max NNZ per Row: ([0-9]+)/, arr);
            max_nnz = arr[1];
        }
    }
    /^Median NNZ per Row:/ { median_nnz = $5; }
    /^Standard Deviation NNZ per Row:/ { std_dev = $6; }
    /^Sparsity Ratio:/ { 
        sparsity = $3;
        if ($0 ~ /\(([0-9.]+)% sparse\)/) {
            match($0, /\(([0-9.]+)% sparse\)/, arr);
            sparsity = arr[1] / 100.0;
        }
    }
    
    # Performance Results section
    /^Matrix size:.*with.*non-zero elements/ {
        # Fallback extraction if matrix dimensions were missed
        if (rows == "" || cols == "" || nnz == "") {
            match($0, /([0-9]+) x ([0-9]+) with ([0-9]+)/, arr);
            if (rows == "") rows = arr[1];
            if (cols == "") cols = arr[2];
            if (nnz == "") nnz = arr[3];
        }
    }
    
    /^Percentage of NNZ:/ { 
        perc_nnz = $4;
        gsub(/%/, "", perc_nnz);
    }
    
    /^Average execution time:/ { 
        time = $4;
        # Ensure time is in seconds
        if ($5 == "ms" || $5 == "milliseconds") {
            time = time / 1000.0;
        }
    }
    
    /^Memory bandwidth \(estimated\):/ { 
        bandwidth = $4;
    }
    
    /^Computational performance:/ { 
        gflops = $3;
    }
    
    # For experiment format, also capture matrix info from different location
    /^Number of Columns:/ && impl == "experiment" { cols = $4; }
    /^Number of Rows:/ && impl == "experiment" { rows = $4; }
    /^Number of NNZ:/ && impl == "experiment" { nnz = $4; }
    /^Average NNZ per Row:/ && impl == "experiment" { mean_nnz = $5; }
    
    # End of file processing
    END {
        if (in_dataset && rows != "" && time != "") {
            output_csv_line();
        }
    }
    
    function reset_variables() {
        rows = ""; cols = ""; nnz = ""; mean_nnz = ""; min_nnz = ""; max_nnz = ""; median_nnz = "";
        std_dev = ""; sparsity = ""; perc_nnz = "";
        block_size = ""; num_blocks = ""; warps_per_block = ""; shared_mem = "";
        time = ""; bandwidth = ""; gflops = "";
    }
    
    function output_csv_line() {
        # Use experiment values if available
        if (exp_block_num != "") num_blocks = exp_block_num;
        if (exp_block_size != "") block_size = exp_block_size;
        
        printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n",
               platform, impl, dataset, matrix_name, rows, cols, nnz, mean_nnz, 
               min_nnz, max_nnz, median_nnz, std_dev, sparsity, perc_nnz,
               block_size, num_blocks, warps_per_block, shared_mem,
               time, bandwidth, gflops;
    }
    ' "$file" >> "$OUTPUT_CSV"
}

# Find and process all benchmark output files
echo "Searching for SpMV benchmark output files..."

# Look for various patterns of benchmark files, excluding the results folder
benchmark_files=$(find . -path "./results" -prune -o -name "*spmv*benchmark*.out" -print -o -name "*vector*.out" -print -o -name "*adaptive*.out" -print -o -name "*experiment*.out" -print | sort)

if [ -z "$benchmark_files" ]; then
    echo "No benchmark output files found!"
    echo "Looking for files matching patterns: *spmv*benchmark*.out, *vector*.out, *adaptive*.out, *experiment*.out"
    echo "Excluding: ./results folder"
    exit 1
fi

echo "Found $(echo "$benchmark_files" | wc -l) benchmark files:"
echo "$benchmark_files"
echo ""

# Process each file
for file in $benchmark_files; do
    if [ -f "$file" ]; then
        process_benchmark_file "$file"
    fi
done

# Clean up any duplicate headers that might have been added
grep -v "^Platform,Implementation" "$OUTPUT_CSV" > temp.csv
echo "Platform,Implementation,Dataset,Matrix_Name,Rows,Columns,NNZ,Mean_NNZ_Per_Row,Min_NNZ_Per_Row,Max_NNZ_Per_Row,Median_NNZ_Per_Row,Std_Dev_NNZ,Sparsity_Ratio,Percentage_NNZ,Block_Size,Num_Blocks,Warps_Per_Block,Shared_Memory_Bytes,Execution_Time_Seconds,Bandwidth_GB_s,Performance_GFLOPS" > "$OUTPUT_CSV"
cat temp.csv >> "$OUTPUT_CSV"
rm -f temp.csv

echo ""
echo "Data extraction complete!"
echo "Results saved to: $OUTPUT_CSV"
echo ""
echo "Summary:"
echo "Total records: $(tail -n +2 "$OUTPUT_CSV" | wc -l)"
echo "Unique implementations: $(tail -n +2 "$OUTPUT_CSV" | cut -d',' -f2 | sort -u | wc -l)"
echo "Unique datasets: $(tail -n +2 "$OUTPUT_CSV" | cut -d',' -f3 | sort -u | wc -l)"
echo ""
echo "Preview of first 5 records:"
head -n 6 "$OUTPUT_CSV" | column -t -s','