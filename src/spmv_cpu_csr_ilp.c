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
#include "../include/csr_conversion.h"
#include "../include/spmv_utils.h"

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
        int j = start;

        for (; j + 3 < end; j += 4) {
            sum += csr_values[j] * vec[csr_col_indices[j]];
            sum += csr_values[j + 1] * vec[csr_col_indices[j + 1]];
            sum += csr_values[j + 2] * vec[csr_col_indices[j + 2]];
            sum += csr_values[j + 3] * vec[csr_col_indices[j + 3]];
        }
        
        //treat final elements
        for(; j < end; j++) {
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
    const int NUM_RUNS = 5;
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
    
    double bandwidth, gflops;
    calculate_bandwidth(n,m,nnz,csr_data.col_indices, avg_time, &bandwidth, &gflops);

    print_spmv_performance(
        "CSR", 
        argv[1],
        n, 
        m, 
        nnz, 
        avg_time, 
        bandwidth, 
        gflops, 
        res,
        10  // Print up to 10 samples
    );

    // Cleanup
    free(csr_data.values);
    free(csr_data.col_indices);
    free(csr_data.row_pointers);
    free(res);
    free(vec);
    return 0;
}