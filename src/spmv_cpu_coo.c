#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <wchar.h>
#include <time.h>
#include <sys/time.h>

// Function to get current time in microseconds
double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

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

void print_vector(const double * vec, const int n) {
    for (int i = 0; i < n; i++) {
        printf("%f ", vec[i]);
    }
    printf("\n");
}

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

    read_from_file_and_init(argv[1], &a_val, &a_row, &a_col, &n, &m, &n_val);

    double * vec = (double*)malloc(m*sizeof(double));
    for (int i = 0; i < n; i++) {
        vec[i] = 1.0;
    }

    double * res = (double*)calloc(m, sizeof(double));

    // Run SpMV multiple times to get accurate timing
    const int NUM_RUNS = 10;
    double total_time = 0.0;

    // Warmup run
    spmv(a_val, a_row, a_col, vec, res, n_val, n);

    // Timed runs
    for (int run = 0; run < NUM_RUNS; run++) {
        double start_time = get_time();

        spmv(a_val, a_row, a_col, vec, res, n_val, n);

        double end_time = get_time();
        total_time += (end_time - start_time);
    }

    double avg_time = total_time / NUM_RUNS;

    // Calculate bandwidth and computation metrics
    // For SpMV: we read a_val, a_row, a_col, vec and write to res
    size_t bytes_read = n_val * (sizeof(double) + 2 * sizeof(int)) + n * sizeof(double);
    size_t bytes_written = n * sizeof(double);
    size_t total_bytes = bytes_read + bytes_written;

    double bandwidth = total_bytes / (avg_time * 1e9);  // GB/s
    double flops = 2.0 * n_val;  // Each non-zero element requires a multiply and add
    double gflops = flops / (avg_time * 1e9);  // GFLOPS

    printf("\nSpMV Performance:\n");
    printf("Matrix size: %d x %d with %d non-zero elements\n", n, m, n_val);
    printf("Average execution time: %.6f seconds\n", avg_time);
    printf("Memory bandwidth: %.2f GB/s\n", bandwidth);
    printf("Computational performance: %.2f GFLOPS\n", gflops);

    // First few elements of result vector
    printf("\nFirst 10 elements of result vector (or fewer if n < 10):\n");
    for (int i = 0; i < m; i++) {
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
