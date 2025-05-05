#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <wchar.h>
#include <time.h>
#include <sys/time.h>
#include <omp.h>

#include "../include/my_time_lib.h"
#include "../include/read_file_lib.h"

int coo_to_csr(int n_rows, int nnz, 
    const int* a_row, const int* a_col, const double* a_val,
    double* csr_values, int* csr_col_indices, int* csr_row_ptr) {

    //error handling
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

    // Copy values and column indicies to their correct positions
    int * temp_row_counts = calloc(n_rows,sizeof(int));
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
    // This step uses insertion sort for each row - good for small row lengths
    for (int i = 0; i < n_rows; i++) {
        int row_start = csr_row_ptr[i];
        int row_end = csr_row_ptr[i + 1];

        for (int j = row_start + 1; j < row_end; j++) {
            int col = csr_col_indices[j];
            double val = csr_values[j];
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

    return 0;
}

// Matrix-vector product function (SpMV)
void spmv(const double *csr_values, const int *csr_row_ptr, const int *csr_col_indices,
          const double *vec, double *res, int n) {
    // Clear result vector
    // memset(res, 0, n * sizeof(double));

    // Perform SpMV
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        double sum = 0.0;
        const int start = csr_row_ptr[i];
        const int end = csr_row_ptr[i+1];

        #pragma omp simd reduction(+:sum)
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

    int * a_row;
    int * a_col;
    double * a_val;
    int n, m, n_val;

    _read_from_file_and_init(argv[1], &a_val, &a_row, &a_col, &n, &m, &n_val);


    double * vec = aligned_alloc(64, m * sizeof(double));
    #pragma omp parallel for simd
    for (int i = 0; i < m; i++) {
        vec[i] = 1.0;
    }

    double * res = (double*)malloc(n*sizeof(double));

    double * csr_values = aligned_alloc(64, n_val * sizeof(double));
    int * csr_col_indices = aligned_alloc(64, n_val * sizeof(int));
    int * csr_row_ptr = (int*)calloc(n + 1,sizeof(int));

    coo_to_csr(n,n_val,a_row,a_col,a_val,csr_values,csr_col_indices,csr_row_ptr);

    free(a_row);
    free(a_col);
    free(a_val);

    // Run SpMV multiple times to get accurate timing
    const int NUM_RUNS = 100;
    double total_time = 0.0;

    double times[NUM_RUNS];

    // Warmup run
    spmv(csr_values, csr_row_ptr, csr_col_indices, vec, res, n);

    // Timed runs
    for (int run = 0; run < NUM_RUNS; run++) {
        TIMER_DEF(0);
        TIMER_START(0);

        spmv(csr_values, csr_row_ptr, csr_col_indices, vec, res, n);

        TIMER_STOP(0);
        times[run] = TIMER_ELAPSED(0)*1e-6;
        // total_time += times[run];
    }

    for (int i = 0; i < NUM_RUNS; i++) {
        total_time += times[i];
    }

    double avg_time = total_time / NUM_RUNS;
    // Calculate bandwidth and computation metrics
    size_t bytes_read = n_val * (sizeof(double) + sizeof(int)) +    // values and col indices
                        (n + 1) * sizeof(int) +                     // row pointers
                        n_val * sizeof(double);                     // vector reads (worst case)
    
    size_t bytes_written = n * sizeof(double);                      // result vector
    size_t total_bytes = bytes_read + bytes_written;

    double bandwidth = total_bytes / (avg_time * 1.0e9);  // GB/s
    double flops = 2.0 * n_val;  // Each non-zero element requires a multiply and add
    double gflops = flops / (avg_time * 1.0e9);  // GFLOPS

    printf("\nSpMV Performance CSR:\n");
    printf("Matrix size: %d x %d with %d non-zero elements\n", n, m, n_val);
    printf("Average execution time: %.6f seconds\n", avg_time);
    printf("Memory bandwidth: %.2f GB/s\n", bandwidth);
    printf("Computational performance: %.2f GFLOPS\n", gflops);

    // First few elements of result vector
    printf("\nFirst non zero element of result vector (or fewer if n < 10):\n");
    for (int i = 0; i < n; i++) {
        if (res[i] == 0.0)
            continue;
        printf("%f ", res[i]);
        break;
    }
    printf("\n");


    free(csr_values);
    free(csr_col_indices);
    free(csr_row_ptr);
    free(res);
    free(vec);
    return 0;
}
