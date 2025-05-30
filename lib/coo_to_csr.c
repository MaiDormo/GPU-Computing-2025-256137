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



int coo_to_csr_padding(const struct COO *coo_data, struct CSR *csr_data) {
    int n_rows = coo_data->num_rows;
    int nnz = coo_data->num_non_zeros;
    const int* a_row = coo_data->a_row;
    const int* a_col = coo_data->a_col;
    const dtype* a_val = coo_data->a_val;
    
    dtype* old_csr_values = csr_data->values;
    int* old_csr_col_indices = csr_data->col_indices;
    int* old_csr_row_ptr = csr_data->row_pointers;

    if (n_rows <= 0 || nnz < 0) return -1;
    if (nnz > 0 && (!a_row || !a_col || !a_val)) return -1;
    if (!old_csr_values || !old_csr_col_indices || !old_csr_row_ptr) return -1;

    for (int i = 0; i < n_rows + 1; ++i) {
        old_csr_row_ptr[i] = 0;
    }
    for (int i = 0; i < nnz; i++) {
        if(a_row[i] >= n_rows || a_row[i] < 0) {
             fprintf(stderr, "Error: Row index %d out of bounds (0-%d) at nnz index %d.\n", a_row[i], n_rows-1, i);
             return -1; 
        }
        old_csr_row_ptr[a_row[i] + 1]++;
    }

    // Calculate new_nnz_val without storing padding_for_row array
    int new_nnz_val = 0;
    for (int i = 0; i < n_rows; i++) { // For row i (0 to n_rows-1)
        int elements_in_row = old_csr_row_ptr[i+1]; // Count for row i (old_csr_row_ptr still holds counts here)
        int remainder = elements_in_row % 32;
        int current_row_padding = (remainder == 0) ? 0 : (32 - remainder);
        new_nnz_val += elements_in_row + current_row_padding;
    }

    // Convert old_csr_row_ptr to cumulative sum for unpadded data
    for (int i = 0; i < n_rows; i++) {
        old_csr_row_ptr[i+1] += old_csr_row_ptr[i];
    }

    int * temp_row_counts = (int *)calloc(n_rows,sizeof(int));
    if (!temp_row_counts) {
        // No padding_for_row to free here anymore
        return -1;
    }

    for (int i = 0; i < nnz; i++) {
        int row = a_row[i];
        int dest_indx = old_csr_row_ptr[row] + temp_row_counts[row];
        if (dest_indx >= nnz) {
             fprintf(stderr, "Error: Destination index %d out of bounds (%d) during CSR construction.\n", dest_indx, nnz);
             free(temp_row_counts);
             // No padding_for_row to free here anymore
             return -1; 
        }
        old_csr_values[dest_indx] = a_val[i];
        old_csr_col_indices[dest_indx] = a_col[i];
        temp_row_counts[row]++;
    }
    free(temp_row_counts);

    #pragma omp parallel for
    for (int i = 0; i < n_rows; i++) {
        int row_start = old_csr_row_ptr[i];
        int row_end = old_csr_row_ptr[i + 1];
        if (row_end <= row_start + 1) continue; 

        for (int j = row_start + 1; j < row_end; j++) {
            int current_col = old_csr_col_indices[j];
            dtype current_val = old_csr_values[j];
            int k = j - 1;
            while (k >= row_start && old_csr_col_indices[k] > current_col) {
                old_csr_col_indices[k + 1] = old_csr_col_indices[k];
                old_csr_values[k + 1] = old_csr_values[k];
                k--;
            }
            old_csr_col_indices[k + 1] = current_col;
            old_csr_values[k + 1] = current_val;
        }
    }

    dtype* new_csr_values_padded = (dtype*)calloc(new_nnz_val, sizeof(dtype));
    int* new_csr_col_indices_padded = (int*)calloc(new_nnz_val, sizeof(int));
    int* new_csr_row_ptr_padded = (int*)calloc(n_rows + 1, sizeof(int));

    if (!new_csr_values_padded || !new_csr_col_indices_padded || !new_csr_row_ptr_padded) {
        free(new_csr_values_padded);
        free(new_csr_col_indices_padded);
        free(new_csr_row_ptr_padded);
        // No padding_for_row to free here anymore
        return -1; 
    }

    new_csr_row_ptr_padded[0] = 0;
    int current_padded_idx = 0;

    for (int i = 0; i < n_rows; i++) { 
        int original_row_start = old_csr_row_ptr[i];
        int original_row_end = old_csr_row_ptr[i+1];
        int num_original_elements = original_row_end - original_row_start;

        for (int k = 0; k < num_original_elements; k++) {
            new_csr_values_padded[current_padded_idx] = old_csr_values[original_row_start + k];
            new_csr_col_indices_padded[current_padded_idx] = old_csr_col_indices[original_row_start + k];
            current_padded_idx++;
        }
        
        // Recalculate padding for the current row
        int remainder = num_original_elements % 32;
        int num_padding_elements_for_row = (remainder == 0) ? 0 : (32 - remainder);
        
        for (int k = 0; k < num_padding_elements_for_row; k++) {
            new_csr_values_padded[current_padded_idx] = 0.0; 
            new_csr_col_indices_padded[current_padded_idx] = 0;   
            current_padded_idx++;
        }
        new_csr_row_ptr_padded[i+1] = current_padded_idx;
    }

    // No padding_for_row to free here anymore

    free(csr_data->values);
    free(csr_data->col_indices);
    free(csr_data->row_pointers);

    csr_data->num_non_zeros = new_nnz_val;
    csr_data->num_rows = coo_data->num_rows;
    csr_data->num_cols = coo_data->num_cols;
    csr_data->values = new_csr_values_padded;
    csr_data->col_indices = new_csr_col_indices_padded;
    csr_data->row_pointers = new_csr_row_ptr_padded;

    return 0; 
}


// Helper functions for quicksort
void swap_rows(float* signatures, int* order, int i, int j) {
    float temp_sig = signatures[i];
    signatures[i] = signatures[j];
    signatures[j] = temp_sig;
    
    int temp_order = order[i];
    order[i] = order[j];
    order[j] = temp_order;
}

int partition_rows(float* signatures, int* order, int low, int high) {
    float pivot = signatures[high];
    int i = low - 1;
    
    for (int j = low; j < high; j++) {
        if (signatures[j] <= pivot) {
            i++;
            swap_rows(signatures, order, i, j);
        }
    }
    
    swap_rows(signatures, order, i + 1, high);
    return i + 1;
}

void quicksort_rows(float* signatures, int* order, int low, int high) {
    if (low < high) {
        int pi = partition_rows(signatures, order, low, high);
        
        quicksort_rows(signatures, order, low, pi - 1);
        quicksort_rows(signatures, order, pi + 1, high);
    }
}

int coo_to_csr_reordered(const struct COO *coo_data, struct CSR *csr_data) {
    int n_rows = coo_data->num_rows;
    int m_cols = coo_data->num_cols;
    int nnz = coo_data->num_non_zeros;
    const int* a_row = coo_data->a_row;
    const int* a_col = coo_data->a_col;
    const dtype* a_val = coo_data->a_val;
    
    // Basic validation
    if (n_rows <= 0 || nnz < 0) return -1;
    if (nnz > 0 && (!a_row || !a_col || !a_val)) return -1;
    if (!csr_data->values || !csr_data->col_indices || !csr_data->row_pointers) return -1;

    // 1. Count non-zeros per row and build signatures for each row
    int* row_nnz = (int*)calloc(n_rows, sizeof(int));
    int* row_order = (int*)malloc(n_rows * sizeof(int));
    float* row_signatures = (float*)calloc(n_rows, sizeof(float));
    if (!row_nnz || !row_order || !row_signatures) {
        free(row_nnz); free(row_order); free(row_signatures);
        return -1;
    }
    
    // Initialize row order and count non-zeros
    for (int i = 0; i < n_rows; i++) {
        row_order[i] = i;
    }
    
    for (int i = 0; i < nnz; i++) {
        int row = a_row[i];
        int col = a_col[i];
        row_nnz[row]++;
        row_signatures[row] += (float)col / m_cols; // Weighted average of column positions
    }
    
    // 2. Sort rows by their signatures using quicksort for better performance
    quicksort_rows(row_signatures, row_order, 0, n_rows - 1);
    
    // 3. Build row pointer array for the reordered matrix
    csr_data->row_pointers[0] = 0;
    for (int i = 0; i < n_rows; i++) {
        int orig_row = row_order[i];
        csr_data->row_pointers[i+1] = csr_data->row_pointers[i] + row_nnz[orig_row];
    }
    
    // 4. Create mapping from original row indices to new positions
    int* row_position_map = (int*)malloc(n_rows * sizeof(int));
    if (!row_position_map) {
        free(row_nnz); free(row_order); free(row_signatures);
        return -1;
    }
    
    for (int i = 0; i < n_rows; i++) {
        row_position_map[row_order[i]] = i;
    }
    
    // Create temporary arrays for sorted values and columns
    dtype* temp_values = (dtype*)malloc(nnz * sizeof(dtype));
    int* temp_cols = (int*)malloc(nnz * sizeof(int));
    if (!temp_values || !temp_cols) {
        free(row_nnz); free(row_order); free(row_signatures); free(row_position_map);
        free(temp_values); free(temp_cols);
        return -1;
    }
    
    // Place values in reordered positions using our mapping
    int* next_row_pos = (int*)calloc(n_rows, sizeof(int));
    if (!next_row_pos) {
        free(row_nnz); free(row_order); free(row_signatures); free(row_position_map);
        free(temp_values); free(temp_cols);
        return -1;
    }
    
    for (int i = 0; i < nnz; i++) {
        int orig_row = a_row[i];
        int new_row_idx = row_position_map[orig_row]; // O(1) lookup instead of O(n)
        
        int pos = csr_data->row_pointers[new_row_idx] + next_row_pos[new_row_idx]++;
        temp_values[pos] = a_val[i];
        temp_cols[pos] = a_col[i];
    }
    
    free(row_position_map); // Free early since we don't need it anymore
    free(next_row_pos);
    
    // 5. Sort columns within each row using insertion sort (better for small arrays)
    #pragma omp parallel for
    for (int i = 0; i < n_rows; i++) {
        int row_start = csr_data->row_pointers[i];
        int row_end = csr_data->row_pointers[i + 1];
        
        // Use insertion sort - much faster than bubble sort for small ranges
        for (int j = row_start + 1; j < row_end; j++) {
            int current_col = temp_cols[j];
            dtype current_val = temp_values[j];
            int k = j - 1;
            
            while (k >= row_start && temp_cols[k] > current_col) {
                temp_cols[k + 1] = temp_cols[k];
                temp_values[k + 1] = temp_values[k];
                k--;
            }
            
            temp_cols[k + 1] = current_col;
            temp_values[k + 1] = current_val;
        }
    }
    
    // Copy from temp arrays to output
    memcpy(csr_data->values, temp_values, nnz * sizeof(dtype));
    memcpy(csr_data->col_indices, temp_cols, nnz * sizeof(int));
    
    // Set metadata
    csr_data->num_rows = n_rows;
    csr_data->num_cols = m_cols;
    csr_data->num_non_zeros = nnz;
    
    // Cleanup
    free(row_nnz);
    free(row_order);
    free(row_signatures);
    free(temp_values);
    free(temp_cols);
    
    return 0;
}