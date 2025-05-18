#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "../include/spmv_utils.h"

void print_matrix_stats(struct CSR *csr_data) {
    double avg_nnz_per_row = (double)csr_data->num_non_zeros / csr_data->num_rows;
    //account for possible overflow
    double total_elements = (double)csr_data->num_cols * (double)csr_data->num_rows;
    double percentage_nnz = (double)csr_data->num_non_zeros / total_elements; 

    int min_nnz = csr_data->num_non_zeros;
    int max_nnz = 0;
    for (int i = 0; i < csr_data->num_rows; i++) {
        int nnz = csr_data->row_pointers[i+1] - csr_data->row_pointers[i];
        min_nnz = min_nnz > nnz ? nnz : min_nnz;
        max_nnz = max_nnz < nnz ? nnz : max_nnz;
    }
    
    printf("Number of Columns: %d\n", csr_data->num_cols);
    printf("Number of Rows: %d\n", csr_data->num_rows);
    printf("Number of NNZ: %d\n", csr_data->num_non_zeros);
    printf("Average NNZ per Row: %f\n", avg_nnz_per_row);
    printf("Min NNZ: %d\n", min_nnz);
    printf("Max NNZ: %d\n", max_nnz);
    printf("Percentage of NNZ: %f%%\n", percentage_nnz*100);
}

int adaptive_row_selection(const int *csr_row_ptr, int rows, int *row_blocks, int warp_size, int block_size) {
    const int MAX_ROWS_PER_BLOCK = block_size / warp_size;
    const int MAX_NNZ_PER_WARP = warp_size * 4;
    const int MAX_NNZ_PER_BLOCK = MAX_NNZ_PER_WARP * MAX_ROWS_PER_BLOCK;
    const int VERY_DENSE_ROW = MAX_NNZ_PER_WARP * 4;

    row_blocks[0] = 0;
    int block_idx = 1;
    int current_block_rows = 0;
    int current_block_nnz = 0;
    
    for (int row = 0; row < rows; row++) {
        int row_nnz = csr_row_ptr[row+1] - csr_row_ptr[row];

        // Case 1: Very dense row - put in its own block
        if (row_nnz > VERY_DENSE_ROW) {
            // if we've accumulated rows, finish the previous block
            if (current_block_rows > 0) {
                row_blocks[block_idx++] = row;
                current_block_rows = 0;
                current_block_nnz = 0;
            }

            //create a dedicated block for this row
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
    // Extract just the matrix name from the path
    const char* matrix_name = strrchr(matrix_path, '/');
    if (matrix_name) {
        matrix_name++; // Skip the '/'
    } else {
        matrix_name = matrix_path; // Use the full path if no '/' found
    }
    
    // Basic matrix stats
    printf("Number of Columns: %d\n", m);
    printf("Number of Rows: %d\n", n);
    printf("Number of NNZ: %d\n", nnz);
    
    // Additional statistics if we have a CSR struct
    double avg_nnz_per_row = (double)nnz / n;
    printf("Average NNZ per Row: %f\n", avg_nnz_per_row);
    
    // Percentage of non-zeros (careful with large matrices to avoid overflow)
    double total_elements = (double)n * (double)m;
    double percentage_nnz = ((double)nnz / total_elements) * 100.0;
    printf("Percentage of NNZ: %f%%\n", percentage_nnz);
    
    // Performance metrics
    printf("\nSpMV Performance %s:\n", implementation_name);
    printf("Matrix size: %d x %d with %d non-zero elements\n", n, m, nnz);

    printf("Average execution time: %.6f seconds\n", avg_time);
    
    printf("Memory bandwidth (estimated): %.4f GB/s\n", bandwidth);
    printf("Computational performance: %.6f GFLOPS\n", gflops);
    
    // Print sample of result vector
    printf("\nFirst few non-zero elements of result vector:\n");
    int count = 0;
    for (int i = 0; i < n && count < max_samples; i++) {
        if (result_vector[i] != 0.0) {
            printf("%f ", result_vector[i]);
            count++;
        }
    }
    
    if (count == 0) {
        printf("Result vector is all zeros or first %d elements are zero.", max_samples);
    }
    printf("\n");
}