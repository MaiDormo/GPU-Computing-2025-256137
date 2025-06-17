#!/bin/bash
# Script to extract SpMV benchmark data from various implementation output files
# Usage: ./extract_spmv_data.sh [output_file.csv]

OUTPUT_CSV="${1:-spmv_comprehensive_results.csv}"

# Create CSV header
echo "Platform,Implementation,Dataset,Matrix_Name,Rows,Columns,NNZ,Mean_NNZ_Per_Row,Min_NNZ_Per_Row,Max_NNZ_Per_Row,Median_NNZ_Per_Row,Std_Dev_NNZ,Sparsity_Ratio,Percentage_NNZ,Block_Size,Num_Blocks,Elements_Per_Thread,Launch_Config,Execution_Time_Seconds,Bandwidth_GB_s,Performance_GFLOPS,Arithmetic_Intensity" > "$OUTPUT_CSV"

# Function to process each benchmark file
process_benchmark_file() {
    local file=$1
    local filename=$(basename "$file")
    
    echo "Processing $file..."
    
    # Determine platform (CPU or GPU)
    local platform="CPU"
    if grep -q "NVIDIA-SMI\|nvidia-smi\|GPU\|CUDA" "$file"; then
        platform="GPU"
    fi
    
    # Extract implementation type from filename and file content
    local implementation=""
    if [[ "$filename" =~ cpu.*simple ]] || grep -q "spmv_cpu_csr" "$file"; then
        implementation="cpu_simple"
    elif [[ "$filename" =~ cpu.*ilp ]] || grep -q "spmv_cpu_csr_ilp" "$file"; then
        implementation="cpu_ilp"
    elif [[ "$filename" =~ gpu.*simple ]] || grep -q "GPU Simple CSR" "$file"; then
        implementation="gpu_simple"
    elif [[ "$filename" =~ gpu.*vector ]] || grep -q "GPU Vector CSR" "$file"; then
        implementation="gpu_vector"
    elif [[ "$filename" =~ gpu.*adaptive ]] || grep -q "GPU Adaptive CSR" "$file"; then
        implementation="gpu_adaptive"
    elif [[ "$filename" =~ hybrid.*adaptive ]] || grep -q "Hybrid Adaptive CSR" "$file"; then
        implementation="gpu_hybrid_adaptive"
    elif [[ "$filename" =~ value.*sequential ]] || grep -q "Value Sequential" "$file"; then
        implementation="gpu_value_sequential"
    elif [[ "$filename" =~ value.*blocked ]] || grep -q "Value Parallel Blocked" "$file"; then
        implementation="gpu_value_blocked"
    elif [[ "$filename" =~ experiment ]] || grep -q "spmv_experimenting_block_thread_sizes" "$file"; then
        implementation="gpu_experiment"
    else
        # Try to extract from content
        if grep -q "CSR SpMV" "$file"; then
            if grep -q "CPU" "$file"; then
                implementation="cpu_csr"
            else
                implementation="gpu_csr"
            fi
        else
            implementation="unknown"
        fi
    fi
    
    # Process the file using AWK
    awk -v platform="$platform" -v impl="$implementation" '
    BEGIN {
        dataset = ""; matrix_name = "";
        rows = ""; cols = ""; nnz = ""; mean_nnz = ""; min_nnz = ""; max_nnz = ""; median_nnz = "";
        std_dev = ""; sparsity = ""; perc_nnz = "";
        block_size = ""; num_blocks = ""; elements_per_thread = ""; launch_config = "";
        time = ""; bandwidth = ""; gflops = ""; arithmetic_intensity = "";
        in_dataset = 0;
    }
    
    # Dataset detection from different formats
    /^Testing dataset:/ || /^------------------------------------------------/ {
        if ($0 ~ /Testing dataset:/) {
            # Output previous dataset if we have complete data
            if (in_dataset && rows != "" && time != "") {
                output_csv_line();
            }
            
            # Extract new dataset
            dataset = $3;
            gsub(/.*\//, "", dataset);  # Remove path
            gsub(/\.mtx.*/, "", dataset);  # Remove extension
            matrix_name = dataset;
            
            # Reset all variables
            reset_variables();
            in_dataset = 1;
        }
        next;
    }
    
    # Matrix dimensions - handle different formats
    /^Matrix size:|^Dimensions:/ {
        if ($0 ~ /Matrix size:.*x.*with.*non-zero/) {
            match($0, /([0-9]+) x ([0-9]+) with ([0-9]+)/, arr);
            rows = arr[1]; cols = arr[2]; nnz = arr[3];
        } else if ($0 ~ /Dimensions:.*x.*NNZ:/) {
            match($0, /([0-9]+) x ([0-9]+), NNZ: ([0-9]+)/, arr);
            rows = arr[1]; cols = arr[2]; nnz = arr[3];
        }
    }
    
    # Matrix analysis section
    /^Matrix analysis:/ {
        if ($0 ~ /mean_nnz_per_row/) {
            match($0, /mean_nnz_per_row=([0-9.]+)/, arr);
            mean_nnz = arr[1];
        } else if ($0 ~ /avg NNZ per row/) {
            match($0, /([0-9.]+) avg NNZ per row/, arr);
            mean_nnz = arr[1];
        }
    }
    
    # Matrix statistics - detailed format
    /^  Mean NNZ per row:/ { mean_nnz = $5; }
    /^  Std deviation:/ { std_dev = $3; }
    /^  Max NNZ per row:/ { max_nnz = $5; }
    /^  Min NNZ per row:/ { min_nnz = $5; }
    /^  Empty rows:/ { 
        empty_rows = $3;
        # Extract percentage if available
        if ($0 ~ /\(([0-9.]+)%\)/) {
            match($0, /\(([0-9.]+)%\)/, arr);
            empty_rows_percent = arr[1];
        }
    }
    
    # Launch configuration - handle multiple formats
    /^Launch config:|^Launch configuration:|^Selected adaptive block size:/ {
        if ($0 ~ /Launch config:.*blocks.*threads/) {
            match($0, /([0-9]+) blocks.*([0-9]+) threads/, arr);
            num_blocks = arr[1];
            block_size = arr[2];
            launch_config = sprintf("%d blocks, %d threads", num_blocks, block_size);
        } else if ($0 ~ /Selected adaptive block size:/) {
            match($0, /Selected adaptive block size: ([0-9]+)/, arr);
            block_size = arr[1];
        }
    }
    
    # Elements per thread
    /^Elements per thread:/ {
        elements_per_thread = $4;
    }
    
    # Experiment format - specific handling
    /^BLOCK_NUM=.*BLOCK_SIZE=/ {
        match($0, /BLOCK_NUM=([0-9]+).*BLOCK_SIZE=([0-9]+)/, arr);
        num_blocks = arr[1]; block_size = arr[2];
        launch_config = sprintf("BLOCK_NUM=%d BLOCK_SIZE=%d", num_blocks, block_size);
    }
    
    # Performance metrics
    /^Average execution time:/ { 
        time = $4;
        # Convert ms to seconds if needed
        if ($5 == "ms" || $5 == "milliseconds") {
            time = time / 1000.0;
        } else if ($5 == "seconds" || $4 ~ /s$/) {
            gsub(/s$/, "", time);
        }
    }
    
    /^Memory bandwidth \(estimated\):/ || /^Bandwidth:/ { 
        bandwidth = $4;
        if ($0 ~ /Memory bandwidth/) bandwidth = $4;
        else if ($0 ~ /Bandwidth:/) bandwidth = $2;
        gsub(/GB\/s/, "", bandwidth);
    }
    
    /^Computational performance:/ || /^Performance:/ { 
        gflops = $3;
        if ($0 ~ /Computational performance:/) gflops = $3;
        else if ($0 ~ /Performance:/) gflops = $2;
        gsub(/GFLOPS/, "", gflops);
    }
    
    /^Arithmetic intensity:/ {
        arithmetic_intensity = $3;
        gsub(/FLOP\/Byte/, "", arithmetic_intensity);
    }
    
    # Handle SpMV performance print format
    /^SpMV Implementation:/ {
        getline; # Skip separator line
        getline; # Matrix file line
        if (getline > 0 && $0 ~ /Matrix size:/) {
            match($0, /([0-9]+) x ([0-9]+), ([0-9]+) NNZ/, arr);
            if (rows == "") rows = arr[1];
            if (cols == "") cols = arr[2];
            if (nnz == "") nnz = arr[3];
        }
    }
    
    # Alternative format detection
    /^Number of Rows:/ { if (rows == "") rows = $4; }
    /^Number of Columns:/ { if (cols == "") cols = $4; }
    /^Number of NNZ:/ { if (nnz == "") nnz = $4; }
    /^Average NNZ per Row:/ { if (mean_nnz == "") mean_nnz = $5; }
    
    # Calculate derived metrics if missing
    function calculate_derived_metrics() {
        if (sparsity == "" && rows != "" && cols != "" && nnz != "") {
            total_elements = rows * cols;
            sparsity = 1.0 - (nnz / total_elements);
            perc_nnz = (nnz / total_elements) * 100.0;
        }
        if (mean_nnz == "" && rows != "" && nnz != "") {
            mean_nnz = nnz / rows;
        }
    }
    
    # End of file processing
    END {
        if (in_dataset && rows != "" && time != "") {
            calculate_derived_metrics();
            output_csv_line();
        }
    }
    
    function reset_variables() {
        rows = ""; cols = ""; nnz = ""; mean_nnz = ""; min_nnz = ""; max_nnz = ""; median_nnz = "";
        std_dev = ""; sparsity = ""; perc_nnz = "";
        block_size = ""; num_blocks = ""; elements_per_thread = ""; launch_config = "";
        time = ""; bandwidth = ""; gflops = ""; arithmetic_intensity = "";
    }
    
    function output_csv_line() {
        printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n",
               platform, impl, dataset, matrix_name, rows, cols, nnz, mean_nnz, 
               min_nnz, max_nnz, median_nnz, std_dev, sparsity, perc_nnz,
               block_size, num_blocks, elements_per_thread, launch_config,
               time, bandwidth, gflops, arithmetic_intensity;
    }
    ' "$file" >> "$OUTPUT_CSV"
}

# Find and process all benchmark output files
echo "Searching for SpMV benchmark output files..."

# Look for various patterns of benchmark files
benchmark_files=$(find . -maxdepth 1 \( \
    -name "*spmv*benchmark*.out" -o \
    -name "*cpu*simple*.out" -o \
    -name "*cpu*ilp*.out" -o \
    -name "*gpu*simple*.out" -o \
    -name "*vector*.out" -o \
    -name "*adaptive*.out" -o \
    -name "*hybrid*.out" -o \
    -name "*experiment*.out" -o \
    -name "*value*.out" \
\) -type f | sort)

if [ -z "$benchmark_files" ]; then
    echo "No benchmark output files found in current directory!"
    echo "Looking for files matching patterns:"
    echo "  *spmv*benchmark*.out"
    echo "  *cpu*simple*.out, *cpu*ilp*.out"
    echo "  *gpu*simple*.out, *vector*.out, *adaptive*.out"
    echo "  *hybrid*.out, *experiment*.out, *value*.out"
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
temp_file=$(mktemp)
head -n 1 "$OUTPUT_CSV" > "$temp_file"
tail -n +2 "$OUTPUT_CSV" | grep -v "^Platform,Implementation" >> "$temp_file"
mv "$temp_file" "$OUTPUT_CSV"

echo ""
echo "Data extraction complete!"
echo "Results saved to: $OUTPUT_CSV"
echo ""
echo "Summary:"
total_records=$(tail -n +2 "$OUTPUT_CSV" | wc -l)
unique_implementations=$(tail -n +2 "$OUTPUT_CSV" | cut -d',' -f2 | sort -u | wc -l)
unique_datasets=$(tail -n +2 "$OUTPUT_CSV" | cut -d',' -f3 | sort -u | wc -l)

echo "Total records: $total_records"
echo "Unique implementations: $unique_implementations"
echo "Unique datasets: $unique_datasets"
echo ""

if [ $total_records -gt 0 ]; then
    echo "Implementations found:"
    tail -n +2 "$OUTPUT_CSV" | cut -d',' -f2 | sort | uniq -c | sort -nr
    echo ""
    echo "Datasets found:"
    tail -n +2 "$OUTPUT_CSV" | cut -d',' -f3 | sort | uniq -c | sort -nr
    echo ""
    echo "Preview of first 3 records:"
    head -n 4 "$OUTPUT_CSV" | column -t -s',' | head -n 4
else
    echo "Warning: No data records were extracted!"
    echo "Check that your benchmark output files contain the expected format."
fi