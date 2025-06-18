#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "../include/csr_conversion.h"

int coo_to_csr(const struct COO *coo_data, struct CSR *csr_data) {
    int n_rows = coo_data->num_rows;
    int nnz = coo_data->num_non_zeros;
    const int* a_row = coo_data->a_row;
    const int* a_col = coo_data->a_col;
    const dtype* a_val = coo_data->a_val;

    dtype* csr_values = csr_data->values;
    int* csr_col_indices = csr_data->col_indices;
    int* csr_row_ptr = csr_data->row_pointers;

    // Basic validation
    if (n_rows <= 0 || nnz < 0) return -1;
    if (nnz > 0 && (!a_row || !a_col || !a_val)) return -1;
    if (!csr_values || !csr_col_indices || !csr_row_ptr) return -1;

    // Initialize row pointers (assuming already zeroed by calloc)
    for (int i = 0; i < nnz; i++) {
        if(a_row[i] >= n_rows || a_row[i] < 0) {
             fprintf(stderr, "Error: Row index %d out of bounds (0-%d) at nnz index %d.\n", a_row[i], n_rows-1, i);
             return -1; // Invalid row index
        }
        csr_row_ptr[a_row[i] + 1]++;
    }

    for (int i = 0; i < n_rows; i++) {
        csr_row_ptr[i + 1] += csr_row_ptr[i];
    }

    int * temp_row_counts = (int *)calloc(n_rows, sizeof(int));
    if (!temp_row_counts) return -1;

    for (int i = 0; i < nnz; i++) {
        int row = a_row[i];
        int dest_indx = csr_row_ptr[row] + temp_row_counts[row];
        if (dest_indx >= nnz) {
             fprintf(stderr, "Error: Destination index %d out of bounds (%d) during CSR construction.\n", dest_indx, nnz);
             free(temp_row_counts);
             return -1; // Index out of bounds
        }
        csr_values[dest_indx] = a_val[i];
        csr_col_indices[dest_indx] = a_col[i];
        temp_row_counts[row]++;
    }
    free(temp_row_counts);

    // Sort columns within rows
    #pragma omp parallel for
    for (int i = 0; i < n_rows; i++) {
        int row_start = csr_row_ptr[i];
        int row_end = csr_row_ptr[i + 1];
        if (row_end <= row_start + 1) continue; // Skip empty or single-element rows

        for (int j = row_start + 1; j < row_end; j++) {
            int current_col = csr_col_indices[j];
            dtype current_val = csr_values[j];
            int k = j - 1;
            while (k >= row_start && csr_col_indices[k] > current_col) {
                csr_col_indices[k + 1] = csr_col_indices[k];
                csr_values[k + 1] = csr_values[k];
                k--;
            }
            csr_col_indices[k + 1] = current_col;
            csr_values[k + 1] = current_val;
        }
    }

    csr_data->num_rows = coo_data->num_rows;
    csr_data->num_cols = coo_data->num_cols;
    csr_data->num_non_zeros = coo_data->num_non_zeros;

    return 0;
}