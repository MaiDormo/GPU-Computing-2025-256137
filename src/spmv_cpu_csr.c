#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <wchar.h>
#include <time.h>
#include <sys/time.h>
#include <omp.h>

#include "../include/my_time_lib.h"

//(n*m)*(m*1)=(n*1)

void read_from_file_and_init(char * file_path, double ** a_val, int ** a_row, int ** a_col, int * mat_rows, int * mat_cols, int * vec_size) {
    FILE * file;
    const size_t BUFFER_SIZE = 16 * 1024 * 1024; // 16MB

    // Open file for reading
    file = fopen(file_path, "r");
    // Check if file opened successfully
    if (!file) {
        perror("Error opening file!\n");
        exit(1);
    }

    // Set up a buffer for file I/O (16MB buffer)
    char * buffer = (char*)malloc(BUFFER_SIZE);
    if (!buffer) {
        perror("Failed to allocate file buffer");
        fclose(file);
        exit(1);
    }
    setvbuf(file, buffer, _IOFBF, BUFFER_SIZE);

    char line_buffer[1024];
    // Ignore lines that start with '%'
    while (fgets(line_buffer, sizeof(line_buffer), file)) {
        if (line_buffer[0] != '%') {
            break;  // Found a non-comment line
        }
    }

    // Read header
    int n, m, n_val;
    // We also take m, even though we do not use it, cause its a squared matrix
    //
    if (sscanf(line_buffer, "%d %d %d", &n, &m, &n_val) != 3) {
        perror("Error reading graph metadata");
        free(buffer);
        fclose(file);
        exit(-1);
    }
    *mat_rows = n;
    *mat_cols = m;
    *vec_size = n_val;

    double * val = (double*)malloc(n_val*sizeof(double));
    int * row = (int*)malloc(n_val*sizeof(int));
    int * col = (int*)malloc(n_val*sizeof(int));

    // Check if allocations succeeded
    if (!val || !row || !col) {
        perror("Failed to allocate memory for matrix data");
        free(buffer);
        free(val);   // These are safe even if NULL
        free(row);
        free(col);
        fclose(file);
        exit(1);
    }

    int r, c;
    double v;
    for (int i = 0; i < n_val; i++) {
        if (fscanf(file, "%d %d %lf", &r, &c, &v) != 3){
            fprintf(stderr, "Error reading entry %d\n", i);
            free(buffer);
            free(row);
            free(col);
            free(val);
            fclose(file);
            exit(1);
        }

        row[i] = r-1;
        col[i] = c-1;
        val[i] = v;
    }

    // Passing pointers
    *a_row = row;
    *a_col = col;
    *a_val = val;

    free(buffer);
    fclose(file);
}

int coo_to_csr(int n_rows, int nnz, const int* a_row, const int* a_col, const double* a_val,
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

void print_vector(const double * vec, const int n) {
    for (int i = 0; i < n; i++) {
        printf("%f ", vec[i]);
    }
    printf("\n");
}

// Matrix-vector product function (SpMV)
void spmv(const double *csr_values, const int *csr_row_ptr,
          const double *vec, double *res, int n) {
    // Clear result vector
    memset(res, 0, n * sizeof(double));

    // Perform SpMV
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        for (int j = csr_row_ptr[i]; j < csr_row_ptr[i+1]; j++) {
            res[i] += csr_values[j] * vec[i];
        }
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

    read_from_file_and_init(argv[1], &a_val, &a_row, &a_col, &n, &m, &n_val);


    double * vec = (double*)malloc(m*sizeof(double));
    #pragma omp for simd
    for (int i = 0; i < n; i++) {
        vec[i] = 1.0;
    }

    double * res = (double*)malloc(n*sizeof(double));

    double * csr_values = malloc(n_val * sizeof(double));
    int * csr_col_indices = malloc(n_val * sizeof(int));
    int * csr_row_ptr = calloc(n + 1,sizeof(int));

    coo_to_csr(n,n_val,a_row,a_col,a_val,csr_values,csr_col_indices,csr_row_ptr);

    free(a_row);
    free(a_col);
    free(a_val);

    // Run SpMV multiple times to get accurate timing
    const int NUM_RUNS = 100;
    double total_time = 0.0;

    double times[NUM_RUNS];

    // Warmup run
    spmv(csr_values, csr_row_ptr, vec, res, n);

    // Timed runs
    for (int run = 0; run < NUM_RUNS; run++) {
        TIMER_DEF(0);
        TIMER_START(0);

        spmv(csr_values, csr_row_ptr, vec, res, n);

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

    printf("\nSpMV Performance CSR:\n");
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


    free(csr_values);
    free(csr_col_indices);
    free(csr_row_ptr);
    free(res);
    free(vec);
    return 0;
}
