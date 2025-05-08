#ifndef SPMV_UTILS_H
#define SPMV_UTILS_H


#include "spmv_type.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Print statistics about a CSR matrix
 * 
 * @param csr_data The CSR matrix data
 */
void print_matrix_stats(struct CSR *csr_data);

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

#ifdef __cplusplus
}
#endif

#endif // SPMV_UTILS_H