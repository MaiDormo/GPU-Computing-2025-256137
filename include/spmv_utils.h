#ifndef SPMV_UTILS_H
#define SPMV_UTILS_H


#include "spmv_type.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Determine block distribution for adaptive CSR SpMV
 *
 * @param csr_row_ptr Row pointers array from CSR matrix
 * @param rows Number of rows
 * @param row_blocks Array to store row block indices (preallocated)
 * @param warp_size Size of a warp
 * @param block_size Size of a block
 * @return Number of row blocks
 */
int adaptive_row_selection(const int *csr_row_ptr, int rows, int *row_blocks, int warp_size, int block_size);

/**
 * Print standardized performance information for SpMV operations
 * 
 * @param implementation_name Name of the SpMV implementation (e.g., "GPU Simple CSR")
 * @param matrix_path Path to the matrix file used
 * @param n Number of rows
 * @param m Number of columns
 * @param nnz Number of non-zero elements
 * @param avg_time Average execution time in seconds
 * @param bandwidth Memory bandwidth in GB/s
 * @param gflops Computational performance in GFLOPS
 * @param result_vector Pointer to the result vector to print samples from
 * @param max_samples Maximum number of non-zero samples to print
 */
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
);

#ifdef __cplusplus
}
#endif

#endif // SPMV_UTILS_H