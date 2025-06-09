#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "../include/spmv_utils.h"
#include "../include/spmv_type.h"

int adaptive_row_selection(const int *csr_row_ptr, int rows, int *row_blocks, int warp_size, int block_size) {
    const int MAX_ROWS_PER_BLOCK = block_size / warp_size;
    const int MAX_NNZ_PER_WARP = warp_size * 4;
    const int MAX_NNZ_PER_BLOCK = MAX_NNZ_PER_WARP * MAX_ROWS_PER_BLOCK;
    const int VERY_DENSE_ROW = MAX_NNZ_PER_WARP * 4;

    // Input validation
    if (!csr_row_ptr || !row_blocks || rows <= 0 || warp_size <= 0 || block_size <= 0) {
        return -1;
    }

    row_blocks[0] = 0;
    int block_idx = 1;
    int current_block_rows = 0;
    int current_block_nnz = 0;
    
    for (int row = 0; row < rows; row++) {
        int row_nnz = csr_row_ptr[row + 1] - csr_row_ptr[row];

        // Case 1: Very dense row - put in its own block
        if (row_nnz > VERY_DENSE_ROW) {
            // If we've accumulated rows, finish the previous block
            if (current_block_rows > 0) {
                row_blocks[block_idx++] = row;
                current_block_rows = 0;
                current_block_nnz = 0;
            }

            // Create a dedicated block for this row
            row_blocks[block_idx++] = row + 1;
            continue;
        }
        
        // Case 2: Adding this row would exceed block limits
        if ((current_block_rows + 1) > MAX_ROWS_PER_BLOCK || 
            (current_block_nnz + row_nnz > MAX_NNZ_PER_BLOCK)) {
            row_blocks[block_idx++] = row;
            current_block_rows = 1;
            current_block_nnz = row_nnz;
        } else {
            current_block_rows++;
            current_block_nnz += row_nnz;
        }
    }

    // Handle any remaining rows
    if (current_block_rows > 0) {
        row_blocks[block_idx++] = rows;
    }

    return block_idx;
}

// Comparison function for qsort
static int compare_ints(const void *a, const void *b) {
    int ia = *(const int*)a;
    int ib = *(const int*)b;
    return (ia > ib) - (ia < ib);
}

static struct MAT_STATS calculate_matrix_stats(const struct CSR *csr_matrix) {
    struct MAT_STATS stats = {0}; // Initialize all fields to 0
    
    if (!csr_matrix || !csr_matrix->row_pointers || csr_matrix->num_rows <= 0) {
        return stats; // Return zero-initialized stats
    }
    
    int rows = csr_matrix->num_rows;
    int cols = csr_matrix->num_cols;
    const int *csr_row_ptr = csr_matrix->row_pointers;
    
    int *row_nnz = malloc(rows * sizeof(int));
    if (!row_nnz) {
        return stats; // Return zero-initialized stats on allocation failure
    }
    
    // Calculate NNZ per row
    for (int i = 0; i < rows; i++) {
        row_nnz[i] = csr_row_ptr[i + 1] - csr_row_ptr[i];
        stats.total_nnz += row_nnz[i];
    }
    
    // Basic statistics
    stats.mean_nnz_per_row = (double)stats.total_nnz / rows;
    stats.min_nnz_per_row = row_nnz[0];
    stats.max_nnz_per_row = row_nnz[0];
    
    // Find min/max and calculate variance
    double variance = 0.0;
    for (int i = 0; i < rows; i++) {
        if (row_nnz[i] < stats.min_nnz_per_row) stats.min_nnz_per_row = row_nnz[i];
        if (row_nnz[i] > stats.max_nnz_per_row) stats.max_nnz_per_row = row_nnz[i];
        
        double diff = row_nnz[i] - stats.mean_nnz_per_row;
        variance += diff * diff;
    }
    
    stats.std_dev_nnz_per_row = sqrt(variance / rows);
    
    // Calculate median
    qsort(row_nnz, rows, sizeof(int), compare_ints);
    stats.median_nnz_per_row = row_nnz[rows / 2];
    
    // Calculate sparsity ratio
    long long total_elements = (long long)rows * cols;
    stats.sparsity_ratio = 1.0 - ((double)stats.total_nnz / total_elements);
    
    free(row_nnz);
    return stats;
}

void print_matrix_stats(const struct CSR *csr_matrix) {
    if (!csr_matrix) {
        printf("Error: NULL CSR matrix provided\n");
        return;
    }
    
    struct MAT_STATS matrix_stats = calculate_matrix_stats(csr_matrix);
    
    if (matrix_stats.total_nnz == 0) {
        printf("Error: Unable to calculate matrix statistics\n");
        return;
    }
    
    printf("\n=== Matrix Statistics ===\n");
    printf("Matrix dimensions: %d x %d\n", csr_matrix->num_rows, csr_matrix->num_cols);
    printf("Total NNZ: %d\n", matrix_stats.total_nnz);
    printf("Mean NNZ per Row: %.2f\n", matrix_stats.mean_nnz_per_row);
    printf("Standard Deviation NNZ per Row: %.2f\n", matrix_stats.std_dev_nnz_per_row);
    printf("Min NNZ per Row: %d, Max NNZ per Row: %d\n", 
           matrix_stats.min_nnz_per_row, matrix_stats.max_nnz_per_row);
    printf("Median NNZ per Row: %d\n", matrix_stats.median_nnz_per_row);
    printf("Sparsity Ratio: %.4f (%.2f%% sparse)\n", 
           matrix_stats.sparsity_ratio, matrix_stats.sparsity_ratio * 100.0);
    printf("========================\n\n");
}

void print_spmv_performance(
    const char* implementation_name,
    const char* matrix_path,
    int n, 
    int m, 
    int nnz, 
    double avg_time, 
    double bandwidth, 
    double gflops, 
    const dtype* result_vector,
    int max_samples
) {
    // Input validation
    if (!implementation_name || !matrix_path || !result_vector || 
        n <= 0 || m <= 0 || nnz < 0 || avg_time <= 0 || max_samples <= 0) {
        printf("Error: Invalid parameters for performance printing\n");
        return;
    }
    
    // Extract just the matrix name from the path
    const char* matrix_name = strrchr(matrix_path, '/');
    if (matrix_name) {
        matrix_name++; // Skip the '/'
    } else {
        matrix_name = matrix_path; // Use the full path if no '/' found
    }
    
    // Additional statistics
    double avg_nnz_per_row = (double)nnz / n;
    double percentage_nnz = ((double)nnz / ((long long)n * m)) * 100.0;
    
    printf("\n=== SpMV Performance Results ===\n");
    printf("Implementation: %s\n", implementation_name);
    printf("Matrix: %s\n", matrix_name);
    printf("Matrix size: %d x %d with %d non-zero elements\n", n, m, nnz);
    printf("Average NNZ per Row: %.2f\n", avg_nnz_per_row);
    printf("Percentage of NNZ: %.4f%%\n", percentage_nnz);
    printf("Average execution time: %.6f seconds\n", avg_time);
    printf("Memory bandwidth (estimated): %.4f GB/s\n", bandwidth);
    printf("Computational performance: %.6f GFLOPS\n", gflops);
    
    // Print sample of result vector
    printf("\nFirst few non-zero elements of result vector:\n");
    int count = 0;
    for (int i = 0; i < n && count < max_samples; i++) {
        if (result_vector[i] != 0.0) {
            printf("%.6f ", result_vector[i]);
            count++;
        }
    }
    
    if (count == 0) {
        printf("Result vector is all zeros or first %d elements are zero.", max_samples);
    }
    printf("\n");
    printf("===============================\n\n");
}