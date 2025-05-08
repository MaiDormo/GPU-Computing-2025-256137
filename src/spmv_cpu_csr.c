#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <wchar.h>
#include <time.h>
#include <sys/time.h>

#include "../include/my_time_lib.h"
#include "../include/read_file_lib.h"
#include "../include/spmv_type.h"

// Convert from COO to CSR format using the defined structs
int coo_to_csr(const struct COO *coo_data, struct CSR *csr_data) {
    int n_rows = coo_data->num_rows;
    int nnz = coo_data->num_non_zeros;
    const int* a_row = coo_data->a_row;
    const int* a_col = coo_data->a_col;
    const dtype* a_val = coo_data->a_val;

    dtype* csr_values = csr_data->values;
    int* csr_col_indices = csr_data->col_indices;
    int* csr_row_ptr = csr_data->row_pointers;

    // Error handling
    if (n_rows <= 0 || nnz < 0) return -1;
    if (nnz > 0 && (!a_row || !a_col || !a_val)) return -1;
    if (!csr_values || !csr_col_indices || !csr_row_ptr) return -1;

    // Count number of elements in each row
    // (we assume csr_row_pointer to be already initialized with zeros)
    for (int i = 0; i < nnz; i++) {
        if(a_row[i] >= n_rows || a_row[i] < 0) return -1;
        csr_row_ptr[a_row[i] + 1]++;
    }

    // Cumulative sum to get row pointers
    for (int i = 0; i < n_rows; i++) {
        csr_row_ptr[i + 1] += csr_row_ptr[i];
    }

    // Copy values and column indices to their correct positions
    int * temp_row_counts = (int *)calloc(n_rows, sizeof(int));
    if (!temp_row_counts) return -1;

    for (int i = 0; i < nnz; i++) {
        int row = a_row[i];
        int dest_indx = csr_row_ptr[row] + temp_row_counts[row];

        csr_values[dest_indx] = a_val[i];
        csr_col_indices[dest_indx] = a_col[i];
        temp_row_counts[row]++;
    }

    free(temp_row_counts);

    // Sort column indices and values within each row (if needed)
    for (int i = 0; i < n_rows; i++) {
        int row_start = csr_row_ptr[i];
        int row_end = csr_row_ptr[i + 1];

        for (int j = row_start + 1; j < row_end; j++) {
            int col = csr_col_indices[j];
            dtype val = csr_values[j];
            int k = j - 1;

            while (k >= row_start && csr_col_indices[k] > col) {
                csr_col_indices[k + 1] = csr_col_indices[k];
                csr_values[k + 1] = csr_values[k];
                k--;
            }

            csr_col_indices[k + 1] = col;
            csr_values[k + 1] = val;
        }
    }

    // Update metadata
    csr_data->num_rows = coo_data->num_rows;
    csr_data->num_cols = coo_data->num_cols;
    csr_data->num_non_zeros = coo_data->num_non_zeros;

    return 0;
}

// Matrix-vector product function (SpMV) using CSR format
void spmv(const struct CSR *csr_data, const dtype *vec, dtype *res) {
    const int n = csr_data->num_rows;
    const dtype *csr_values = csr_data->values;
    const int *csr_row_ptr = csr_data->row_pointers;
    const int *csr_col_indices = csr_data->col_indices;
    
    // Perform SpMV
    for (int i = 0; i < n; i++) {
        dtype sum = 0.0;
        const int start = csr_row_ptr[i];
        const int end = csr_row_ptr[i+1];

        for (int j = start; j < end; j++) {
            sum += csr_values[j] * vec[csr_col_indices[j]];
        }
        res[i] = sum;
    } 
}

int main(int argc, char ** argv) {
    if (argc != 2) {
        printf("Usage: <./spmv> <path/to/file.mtx>\n");
        return -1;
    }

    // --- Host Data Structures ---
    struct COO coo_data;
    struct CSR csr_data;
    dtype *vec = NULL;
    dtype *res = NULL;

    // --- Read Matrix in COO format ---
    read_from_file_and_init(argv[1], &coo_data);
    int n = coo_data.num_rows;
    int m = coo_data.num_cols;
    int nnz = coo_data.num_non_zeros;

    // --- Allocate Host Memory ---
    vec = (dtype*)malloc(m * sizeof(dtype));
    res = (dtype*)malloc(n * sizeof(dtype));
    csr_data.values = (dtype*)malloc(nnz * sizeof(dtype));
    csr_data.col_indices = (int*)malloc(nnz * sizeof(int));
    csr_data.row_pointers = (int*)calloc(n + 1, sizeof(int));

    // --- Initialize Vectors ---
    for (int i = 0; i < m; i++) {
        vec[i] = 1.0;
    }
    memset(res, 0, n * sizeof(dtype));

    // --- Convert COO to CSR ---
    if (coo_to_csr(&coo_data, &csr_data) != 0) {
        fprintf(stderr, "Error during COO to CSR conversion.\n");
        // Free all allocated memory before exiting
        free(coo_data.a_val); free(coo_data.a_row); free(coo_data.a_col);
        free(vec); free(res);
        free(csr_data.values); free(csr_data.col_indices); free(csr_data.row_pointers);
        return -1;
    }

    // Free original COO data as we don't need it anymore
    free(coo_data.a_val); free(coo_data.a_row); free(coo_data.a_col);

    // Run SpMV multiple times to get accurate timing
    const int NUM_RUNS = 50;
    double total_time = 0.0;
    double times[NUM_RUNS];

    // Warmup run
    spmv(&csr_data, vec, res);

    // Timed runs
    for (int run = 0; run < NUM_RUNS; run++) {
        memset(res, 0, n * sizeof(dtype)); 

        TIMER_DEF(0);
        TIMER_START(0);

        spmv(&csr_data, vec, res);

        TIMER_STOP(0);
        times[run] = TIMER_ELAPSED(0)*1e-6;
    }

    for (int i = 0; i < NUM_RUNS; i++) {
        total_time += times[i];
    }

    double avg_time = total_time / NUM_RUNS;
    
    // Calculate bandwidth and computation metrics
    size_t bytes_read = (size_t)nnz * (sizeof(dtype) + sizeof(int)) +    // values and col indices
                        (size_t)(n + 1) * sizeof(int) +                  // row pointers
                        (size_t)nnz * sizeof(dtype);                     // vector reads (worst case, one vec element per nnz)
    
    size_t bytes_written = (size_t)n * sizeof(dtype);                    // result vector
    size_t total_bytes = bytes_read + bytes_written;

    double bandwidth = total_bytes / (avg_time * 1.0e9);  // GB/s
    double flops = 2.0 * nnz;  // Each non-zero element requires a multiply and add
    double gflops = flops / (avg_time * 1.0e9);  // GFLOPS

    printf("\nSpMV Performance CSR (Single-Threaded):\n");
    printf("Matrix size: %d x %d with %d non-zero elements\n", n, m, nnz);
    printf("Average execution time: %.6f seconds\n", avg_time);
    printf("Memory bandwidth: %.2f GB/s\n", bandwidth);
    printf("Computational performance: %.2f GFLOPS\n", gflops);

    // First few elements of result vector
    printf("\nFirst non zero element of result vector (or fewer if n < 10):\n");
    for (int i = 0; i < n; i++) {
        if (res[i] == 0.0) // Using a small epsilon might be better for float comparison
            continue;
        printf("%f ", res[i]);
        break;
    }
    printf("\n");

    // Cleanup
    free(csr_data.values);
    free(csr_data.col_indices);
    free(csr_data.row_pointers);
    free(res);
    free(vec);
    return 0;
}