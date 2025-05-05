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

//(n*m)*(m*1)=(n*1)

// Matrix-vector product function (SpMV)
void spmv(const double *a_val, const int *a_row, const int *a_col,
          const double *vec, double *res, int n_val, int n) {
    // Clear result vector
    memset(res, 0, n * sizeof(double));

    // Perform SpMV
    for (int i = 0; i < n_val; i++) {
        res[a_row[i]] += a_val[i] * vec[a_col[i]];
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


    double * vec = (double*)malloc(m*sizeof(double));
    #pragma omp for simd
    for (int i = 0; i < n; i++) {
        vec[i] = 1.0;
    }

    double * res = (double*)malloc(n*sizeof(double));

    // Run SpMV multiple times to get accurate timing
    const int NUM_RUNS = 100;
    double total_time = 0.0;

    double times[NUM_RUNS];

    // Warmup run
    spmv(a_val, a_row, a_col, vec, res, n_val, n);

    // Timed runs
    // #pragma omp parallel for
    for (int run = 0; run < NUM_RUNS; run++) {
        TIMER_DEF(0);
        TIMER_START(0);

        spmv(a_val, a_row, a_col, vec, res, n_val, n);

        TIMER_STOP(0);
        times[run] = TIMER_ELAPSED(0)*1e-6;
        // total_time += times[run];
    }

    for (int i = 0; i < NUM_RUNS; i++) {
        total_time += times[i];
    }

    double avg_time = total_time / NUM_RUNS;
    // Calculate bandwidth and computation metrics
    // For SpMV: we read a_val, a_row, a_col, vec and write to res
    size_t bytes_read = n_val * (sizeof(double) + 2 * sizeof(int)) + n * sizeof(double);
    size_t bytes_written = n * sizeof(double);
    size_t total_bytes = bytes_read + bytes_written;

    double bandwidth = total_bytes / (avg_time * 1.0e9);  // GB/s
    double flops = 2.0 * n_val;  // Each non-zero element requires a multiply and add
    double gflops = flops / (avg_time * 1.0e9);  // GFLOPS

    printf("\nSpMV Performance C00:\n");
    printf("Matrix size: %d x %d with %d non-zero elements\n", n, m, n_val);
    printf("Average execution time: %.6f seconds\n", avg_time);
    printf("Memory bandwidth: %.2f GB/s\n", bandwidth);
    printf("Computational performance: %.2f GFLOPS\n", gflops);

    // First few elements of result vector
    printf("\nFirst 10 elements of result vector (or fewer if n < 10):\n");
    for (int i = 0; i < n; i++) {
        if (res[i] == 0.0)
            continue;
        printf("%f ", res[i]);
        break;
    }
    printf("\n");

    free(a_val);
    free(a_col);
    free(a_row);
    free(res);
    free(vec);
    return 0;
}
