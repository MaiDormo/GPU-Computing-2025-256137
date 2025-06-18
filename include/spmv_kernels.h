#ifndef SPMV_KERNELS_H
#define SPMV_KERNELS_H

#include "spmv_type.h"

#define WARP_SIZE 32
#define BLOCK_SIZE 1024

/**
 * Simple CSR SpMV kernel that maps one thread to one row
 * Works best for matrices with few non-zeros per row
 */
__global__ void spmv_simple(const dtype *csr_values, const int *csr_row_ptr, 
                          const int *csr_col_indices, const dtype *vec, 
                          dtype *res, int num_rows);

/**
 * @brief kernel that maps each thread to each non-zero element
 * and later uses atomic add in order to calculate the result
 */
__global__ void value_parallel_sequential_spmv(const dtype *csr_values, const int *csr_row_ptr, 
    const int *csr_col_indices, const dtype *vec, 
    dtype *res, int nnz, int num_rows);

/**
 * @brief kernel that maps each thread to a consecutive block of element
 * so that each thread reads the lenght of stride element and only cares about those
 */

__global__ void value_parallel_blocked_spmv(const dtype *csr_values, const int *csr_row_ptr, 
    const int *csr_col_indices, const dtype *vec,
    dtype *res, int nnz, int num_rows, int stride);

//------------------------ NOT PART OF THE FIRST DELIVERABLE ----------------

/**
 * Standard vector-based CSR SpMV kernel
 * Each warp processes one row
 */
__global__ void vector_csr(const dtype *csr_values, const int *csr_row_ptr, 
                          const int *csr_col_indices, const dtype *vec, 
                          dtype *res, int n);

/**
 * Adaptive CSR SpMV kernel that handles both dense and sparse rows efficiently
 * Uses either a warp per row or an entire block per row depending on density
 */
__global__ void adaptive_csr(const dtype *csr_values, const int *csr_row_ptr,
                           const int *csr_col_indices, const dtype *vec,
                           dtype *res, const int *row_blocks, int n);

/**
 * @brief Double-buffered vector CSR kernel
 */
__global__ void vector_csr_double_buffer(const dtype *csr_values, const int *csr_row_ptr, 
                                        const int *csr_col_indices, const dtype *vec, 
                                        dtype *res, int n);

/**
 * @brief Hybrid adaptive kernel that employs both scalar and 
 * vector approaches based on matrix statistics and row lengths.
 */
__global__ void hybrid_adaptive_spmv_optimized(const dtype *csr_values, const int *csr_row_ptr,
                                              const int *csr_col_indices, const dtype *vec,
                                              dtype *res, int n, const int *short_rows, 
                                              const int *long_rows, int num_short, 
                                              int num_long, int short_blocks);




#endif // SPMV_KERNELS_H