#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <wchar.h>
#include <time.h>
#include <sys/time.h>
#include <omp.h>
//#include <limits.h>

#define dtype double

// #include "../include/my_time_lib.h"
// #include "../include/read_file_lib.h"

/*
## Thread & Block Limits
| Specification | Value |
|---------------|-------|
| Warp Size | 32 |
| Max Threads per Multiprocessor | 1536 |
| Max Threads per Block | 1024 |
| Max Thread Block Dimensions | (1024, 1024, 64) |
| Max Grid Dimensions | (2147483647, 65535, 65535) |
*/

// --- Struct Definitions ---
// Structure for CSR format
struct CSR {
    dtype * values;
    int * col_indices;
    int * row_pointers;
    int num_rows;
    int num_cols;
    int num_non_zeros;
};

// Structure for COO format
struct COO {
    dtype * a_val;
    int * a_col;
    int * a_row;
    int num_rows;
    int num_cols;
    int num_non_zeros;
};

// --- Function Prototypes ---
void read_from_file_and_init(char * file_path, struct COO *coo_data);
int coo_to_csr(const struct COO *coo_data, struct CSR *csr_data);
__global__ void multi_warp_spmv(const dtype *csr_values, const int *csr_row_ptr, const int *csr_col_indices,
                                const dtype *vec, dtype *res, int n); // Keep n for row boundary check


// --- File Reading and Initialization ---
void read_from_file_and_init(char * file_path, struct COO *coo_data) {
    FILE * file;
    const size_t BUFFER_SIZE = 1 * 1024 * 1024; // 1MB

    file = fopen(file_path, "r");
    if (!file) {
        perror("Error opening file!\n");
        exit(1);
    }

    char * buffer = (char*)malloc(BUFFER_SIZE);
    if (!buffer) {
        perror("Failed to allocate file buffer");
        fclose(file);
        exit(1);
    }
    setvbuf(file, buffer, _IOFBF, BUFFER_SIZE);

    char line_buffer[1024];
    while (fgets(line_buffer, sizeof(line_buffer), file)) {
        if (line_buffer[0] != '%') break;
    }

    int n, m, nnz;
    if (sscanf(line_buffer, "%d %d %d", &n, &m, &nnz) != 3) {
        perror("Error reading graph metadata");
        free(buffer); fclose(file); exit(-1);
    }
    coo_data->num_rows = n;
    coo_data->num_cols = m;
    coo_data->num_non_zeros = nnz;

    coo_data->a_val = (dtype*)malloc(nnz * sizeof(dtype));
    coo_data->a_row = (int*)malloc(nnz * sizeof(int));
    coo_data->a_col = (int*)malloc(nnz * sizeof(int));

    if (!coo_data->a_val || !coo_data->a_row || !coo_data->a_col) {
        perror("Failed to allocate memory for COO matrix data");
        free(buffer); fclose(file);
        free(coo_data->a_val); free(coo_data->a_row); free(coo_data->a_col);
        exit(1);
    }

    int r, c;
    dtype v;
    for (int i = 0; i < nnz; i++) {
        if (fscanf(file, "%d %d %lf", &r, &c, &v) != 3){
            fprintf(stderr, "Error reading entry %d\n", i);
            free(buffer); fclose(file);
            free(coo_data->a_val); free(coo_data->a_row); free(coo_data->a_col);
            exit(1);
        }
        coo_data->a_row[i] = r - 1;
        coo_data->a_col[i] = c - 1;
        coo_data->a_val[i] = v;
    }

    free(buffer);
    fclose(file);
}

// --- COO to CSR Conversion ---
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

    // Sort within rows
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


// --- SpMV Kernel ---
__global__ void multi_warp_spmv(const dtype *csr_values, const int *csr_row_ptr, const int *csr_col_indices,
                                const dtype *vec, dtype *res, int n) { // n is num_rows

    const int warp_size = 32;
    const int block_threads = blockDim.x;
    const int tid = threadIdx.x;
    const int warps_per_block = block_threads / warp_size;
    const int block_start_row = blockIdx.x * warps_per_block;
    const int warp_id = tid / warp_size;
    const int lane_id = tid % warp_size;
    const int actual_row = block_start_row + warp_id;

    if (actual_row < n) {
        dtype thread_sum = 0.0;
        int row_start = csr_row_ptr[actual_row];
        int row_end = csr_row_ptr[actual_row + 1];

        for (int j = row_start + lane_id; j < row_end; j += warp_size) {
            int col = csr_col_indices[j];
            thread_sum += csr_values[j] * __ldg(&vec[col]);
        }

        #pragma unroll
        for (int offset = warp_size / 2; offset > 0; offset /= 2) {
            thread_sum += __shfl_down_sync(0xFFFFFFFF, thread_sum, offset);
        }

        if (lane_id == 0) {
            res[actual_row] = thread_sum;
        }
    }
}

// --- Main Function ---
int main(int argc, char ** argv) {
    if (argc != 2) {
        printf("Usage: <./bin/spmv_*> <path/to/file.mtx>\n");
        return -1;
    }

    // --- Host Data Structures ---
    struct COO h_coo;
    struct CSR h_csr;
    dtype *h_vec = NULL;
    dtype *h_res = NULL;

    // --- Read Matrix ---
    read_from_file_and_init(argv[1], &h_coo);
    int n = h_coo.num_rows;
    int m = h_coo.num_cols;
    int nnz = h_coo.num_non_zeros;

    // --- Allocate Host Memory ---
    h_vec = (dtype*)malloc(m * sizeof(dtype));
    h_res = (dtype*)malloc(n * sizeof(dtype));
    h_csr.values = (dtype*)malloc(nnz * sizeof(dtype));
    h_csr.col_indices = (int*)malloc(nnz * sizeof(int));
    h_csr.row_pointers = (int*)calloc(n + 1, sizeof(int)); // Zero initialization is important

    if (!h_vec || !h_res || !h_csr.values || !h_csr.col_indices || !h_csr.row_pointers) {
        perror("Failed to allocate host memory");
        // Free any successfully allocated memory before exiting
        free(h_coo.a_val); free(h_coo.a_row); free(h_coo.a_col);
        free(h_vec); free(h_res);
        free(h_csr.values); free(h_csr.col_indices); free(h_csr.row_pointers);
        return -1;
    }

    // --- Initialize Host Vectors ---
    for (int i = 0; i < m; i++) h_vec[i] = 1.0;
    memset(h_res, 0, n * sizeof(dtype));

    // --- Convert COO to CSR ---
    if (coo_to_csr(&h_coo, &h_csr) != 0) {
         fprintf(stderr, "Error during COO to CSR conversion.\n");
         // Free memory and exit
         free(h_coo.a_val); free(h_coo.a_row); free(h_coo.a_col);
         free(h_vec); free(h_res);
         free(h_csr.values); free(h_csr.col_indices); free(h_csr.row_pointers);
         return -1;
    }

    // Free original COO data (host)
    free(h_coo.a_val); h_coo.a_val = NULL;
    free(h_coo.a_row); h_coo.a_row = NULL;
    free(h_coo.a_col); h_coo.a_col = NULL;

    // --- Device Data Structures ---
    struct CSR d_csr; // Holds device pointers
    dtype *d_vec = NULL, *d_res = NULL;

    // --- Allocate Device Memory ---
    cudaMalloc(&d_vec, m * sizeof(dtype));
    cudaMalloc(&d_res, n * sizeof(dtype));
    cudaMalloc(&d_csr.values, nnz * sizeof(dtype));
    cudaMalloc(&d_csr.col_indices, nnz * sizeof(int));
    cudaMalloc(&d_csr.row_pointers, (n + 1) * sizeof(int));
    d_csr.num_rows = n;
    d_csr.num_cols = m;
    d_csr.num_non_zeros = nnz;


    // --- Copy Data to Device ---
    cudaMemcpy(d_vec, h_vec, m * sizeof(dtype), cudaMemcpyHostToDevice);
    cudaMemcpy(d_res, h_res, n * sizeof(dtype), cudaMemcpyHostToDevice); // Copy initial zeros
    cudaMemcpy(d_csr.values, h_csr.values, nnz * sizeof(dtype), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csr.col_indices, h_csr.col_indices, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csr.row_pointers, h_csr.row_pointers, (n + 1) * sizeof(int), cudaMemcpyHostToDevice);


    // --- Kernel Launch Configuration ---
    const int warp_size = 32;
    const int thread_per_block = 1024; // Choose appropriate block size
    const int warps_per_block = thread_per_block / warp_size;
    const int block_num = (n + warps_per_block - 1) / warps_per_block;
    // Adjust shared memory size if caching is re-introduced
    const size_t dynamic_shared_mem = 0; //!TODO configure shared_mem

    // --- Timing Setup ---
    const int NUM_RUNS = 50;
    dtype total_time = 0.0;
    dtype times[NUM_RUNS];
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    // --- Warmup Run ---
    multi_warp_spmv<<<block_num, thread_per_block, dynamic_shared_mem>>>(
        d_csr.values, d_csr.row_pointers, d_csr.col_indices, d_vec, d_res, n);
    cudaDeviceSynchronize(); // Check for launch errors

    // --- Timed Runs ---
    for (int run = 0; run < NUM_RUNS; run++) {
        cudaEventRecord(start);

        multi_warp_spmv<<<block_num, thread_per_block, dynamic_shared_mem>>>(
            d_csr.values, d_csr.row_pointers, d_csr.col_indices, d_vec, d_res, n);

        cudaEventRecord(end);
        cudaEventSynchronize(end);

        float millisec = 0.0;
        cudaEventElapsedTime(&millisec, start, end);
        times[run] = millisec * 1e-3;
    }

    // --- Copy Result Back ---
    cudaMemcpy(h_res, d_res, n * sizeof(dtype), cudaMemcpyDeviceToHost);

    // --- Performance Calculation ---
    cudaEventDestroy(start);
    cudaEventDestroy(end); // Destroy events even if errors occurred after creation

    for (int i = 0; i < NUM_RUNS; i++) {
        total_time += times[i];
    }
    dtype avg_time = total_time / NUM_RUNS;

    size_t bytes_read = (size_t)nnz * (sizeof(dtype) + sizeof(int)) + // values and col indices
                        (size_t)(n + 1) * sizeof(int) +               // row pointers
                        (size_t)m * sizeof(dtype);                    // vector reads (worst case estimate)
    size_t bytes_written = (size_t)n * sizeof(dtype);                 // result vector
    size_t total_bytes = bytes_read + bytes_written;

    dtype bandwidth = total_bytes / (avg_time * 1.0e9);  // GB/s
    dtype flops = 2.0 * nnz;
    dtype gflops = flops / (avg_time * 1.0e9);  // GFLOPS

    // --- Print Results ---
    printf("\nSpMV Performance CSR:\n");
    printf("Matrix size: %d x %d with %d non-zero elements\n", n, m, nnz);
    printf("Average execution time: %.3f ms\n", avg_time * 1.0e3);
    printf("Memory bandwidth (estimated): %.2f GB/s\n", bandwidth);
    printf("Computational performance: %.2f GFLOPS\n", gflops);

    printf("\nFirst few non-zero elements of result vector:\n");
    int count = 0;
    for (int i = 0; i < n && count < 10; i++) {
        if (h_res[i] != 0.0) {
            printf("%f ", h_res[i]);
            count++;
        }
    }
     if (count == 0) printf("Result vector is all zeros or first 10 elements are zero.");
    printf("\n");

    // --- Cleanup ---
    cudaFree(d_vec);
    cudaFree(d_res);
    cudaFree(d_csr.values);
    cudaFree(d_csr.col_indices);
    cudaFree(d_csr.row_pointers);

    free(h_vec);
    free(h_res);
    free(h_csr.values);
    free(h_csr.col_indices);
    free(h_csr.row_pointers);

    return 0;
}