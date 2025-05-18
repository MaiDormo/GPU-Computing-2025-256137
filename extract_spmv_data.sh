#!/bin/bash
# filepath: extract_spmv_data.sh
# Script to extract SpMV benchmark data from output files and format as CSV

# Output CSV file
OUTPUT_CSV="spmv_benchmark_results.csv"

# Create CSV header
echo "Platform,Implementation,Dataset,Rows,Columns,NNZ,AvgNNZPerRow,MinNNZ,MaxNNZ,PercentageNNZ,ExecutionTime_ms,Bandwidth_GB_s,Performance_GFLOPS" > "$OUTPUT_CSV"

# Process each benchmark file
process_benchmark_file() {
    local file=$1
    local filename=$(basename "$file")
    
    # Determine if it's CPU or GPU by checking for nvidia-smi output
    local platform="CPU"  # Default to CPU
    if grep -q "NVIDIA-SMI" "$file"; then
        platform="GPU"
    fi
    
    # Extract implementation type from filename
    local implementation=""
    if [[ "$filename" =~ cpu_([^_]+)_spmv ]]; then
        implementation="${BASH_REMATCH[1]}"
    elif [[ "$filename" =~ gpu_([^_]+)_spmv ]]; then
        implementation="${BASH_REMATCH[1]}"
    elif [[ "$filename" == *"simple_spmv"* ]]; then
        implementation="simple"
    elif [[ "$filename" == *"value_sequential_spmv"* ]]; then
        implementation="value_sequential"
    elif [[ "$filename" == *"strided"* ]]; then
        implementation="strided"
    elif [[ "$filename" == *"vector"* ]]; then
        implementation="vector"
    elif [[ "$filename" == *"ilp"* ]]; then
        implementation="ilp"
    elif [[ "$filename" == *"omp"* ]]; then
        implementation="omp"
    else
        # Default fallback
        implementation=$(echo "$file" | sed -E 's/.*_([^_]+)_spmv_benchmark.*/\1/g')
        # If still no match, use filename
        if [[ "$implementation" == "$file" ]]; then
            implementation=$(basename "$file" | sed 's/_spmv_benchmark.*//g')
        fi
    fi

    # Extract datasets and their metrics
    awk -v platform="$platform" -v impl="$implementation" '
    BEGIN {
        dataset = ""; 
        in_matrix = 0;
        rows = ""; cols = ""; nnz = ""; avg_nnz = ""; min_nnz = ""; max_nnz = ""; perc_nnz = "";
        time = ""; bandwidth = ""; perf = "";
        current_section = "";
    }
    
    # Start of a new dataset section and end of previous one
    /^Testing dataset:/ {
        # Output previous dataset if we have one
        if (in_matrix && rows != "" && time != "") {
            printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n", 
                   platform, impl, dataset, rows, cols, nnz, avg_nnz, min_nnz, max_nnz, 
                   perc_nnz, time, bandwidth, perf;
            
            # Reset for next dataset
            rows = ""; cols = ""; nnz = ""; avg_nnz = ""; min_nnz = ""; max_nnz = ""; perc_nnz = "";
            time = ""; bandwidth = ""; perf = "";
        }
        
        # New dataset starts
        dataset = $3;
        in_matrix = 1;
        current_section = "matrix_info";
    }
    
    # Pattern 1: Matrix metadata section
    /^Number of Columns:/ && in_matrix { cols = $4; }
    /^Number of Rows:/ && in_matrix { rows = $4; }
    /^Number of NNZ:/ && in_matrix { nnz = $4; }
    /^Average NNZ per Row:/ && in_matrix { avg_nnz = $5; }
    /^Min NNZ:/ && in_matrix { min_nnz = $3; }
    /^Max NNZ:/ && in_matrix { max_nnz = $3; }
    /^Percentage of NNZ:/ && in_matrix { 
        perc_nnz = $4; 
        gsub(/%/, "", perc_nnz);
    }
    
    # Pattern 2: Performance section indicators
    /^SpMV Performance/ { current_section = "performance"; }
    
    # Matrix size line (fallback if we didnt get matrix info before)
    /^Matrix size:/ && current_section == "performance" && (rows == "" || cols == "" || nnz == "") { 
        split($0, parts, "[ x]");
        for (i=1; i<=length(parts); i++) {
            if (parts[i] == "size:") {
                rows = parts[i+1];
                cols = parts[i+3];
            }
            if (parts[i] == "with") {
                nnz = parts[i+1];
                break;
            }
        }
    }
    
    # Use consistent pattern for time extraction
    /^Average execution time:/ {
        time = $4;
        if ($5 == "seconds") {
            time = time * 1000;  # Convert to ms
        }
    }
    
    # Bandwidth and GFLOPS extraction
    /^Memory bandwidth/ { bandwidth = $(NF-1); }
    /^Computational performance:/ { perf = $(NF-1); }
    
    # Flag to indicate we are at the results section
    /^First (few |non-zero |)elements? of result vector/ { 
        current_section = "results";
    }
    
    # At end of file, output any remaining dataset
    END {
        if (in_matrix && rows != "" && time != "") {
            printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n", 
                   platform, impl, dataset, rows, cols, nnz, avg_nnz, min_nnz, max_nnz, 
                   perc_nnz, time, bandwidth, perf;
        }
    }
    ' "$file" >> "$OUTPUT_CSV"
}

# Find all benchmark output files
benchmark_files=$(find . -name "*spmv_benchmark*.out")

# Process each file
for file in $benchmark_files; do
    echo "Processing $file..."
    process_benchmark_file "$file"
done

# Clean up the CSV by fixing any missing data
# Datasets have the same matrix properties regardless of implementation
awk 'BEGIN { FS=OFS="," }
    NR==1 {header=$0; print; next}
    {
        # Store matrix properties by dataset
        if ($3 in datasets) {
            # Fill in any missing values from previous records of same dataset
            if ($4 == "") $4 = datasets[$3]["rows"]
            if ($5 == "") $5 = datasets[$3]["cols"]
            if ($6 == "") $6 = datasets[$3]["nnz"]
            if ($7 == "") $7 = datasets[$3]["avg_nnz"]
            if ($8 == "") $8 = datasets[$3]["min_nnz"]
            if ($9 == "") $9 = datasets[$3]["max_nnz"]
            if ($10 == "") $10 = datasets[$3]["perc_nnz"]
        } else {
            # First time seeing this dataset, store values
            datasets[$3]["rows"] = $4
            datasets[$3]["cols"] = $5
            datasets[$3]["nnz"] = $6
            datasets[$3]["avg_nnz"] = $7
            datasets[$3]["min_nnz"] = $8
            datasets[$3]["max_nnz"] = $9
            datasets[$3]["perc_nnz"] = $10
        }
        print
    }' "$OUTPUT_CSV" > temp.csv && mv temp.csv "$OUTPUT_CSV"

echo "Data extraction complete. Results saved to $OUTPUT_CSV"

# Print preview of CSV
echo "Preview of CSV data:"
head -n 5 "$OUTPUT_CSV"