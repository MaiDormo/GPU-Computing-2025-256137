#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>

#include "../include/read_file_lib.h"
#include "../include/spmv_type.h"
#include "../include/csr_conversion.h"
#include "../include/spmv_utils.h"
#include "../include/spmv_kernels.h"

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

    // --- Adaptive Block Size Selection ---
    int adaptive_block_size;
    double avg_nnz_per_row = (double)nnz / n;
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    // For extremely sparse matrices (< 2 NNZ per row)
    if (avg_nnz_per_row < 2.0) {
        adaptive_block_size = 128;  // Smaller blocks for better occupancy with sparse data
    } else if (avg_nnz_per_row < 8.0) {
        adaptive_block_size = 256;  // Medium sparse
    } else if (avg_nnz_per_row < 32.0) {
        adaptive_block_size = 512;  // Medium dense
    } else {
        adaptive_block_size = 1024; // Dense matrices
    }

    // Ensure it's a multiple of warp size and within device limits
    adaptive_block_size = ((adaptive_block_size + 31) / 32) * 32;
    if (adaptive_block_size > prop.maxThreadsPerBlock) {
        adaptive_block_size = prop.maxThreadsPerBlock;
    }

    printf("Matrix analysis: %.2f avg NNZ per row\n", avg_nnz_per_row);
    printf("Selected adaptive block size: %d (was %d)\n", adaptive_block_size, BLOCK_SIZE);

    // --- Updated Kernel Launch Configuration ---
    const int elements_per_thread = 8;
    const int total_threads_needed = (nnz + elements_per_thread - 1) / elements_per_thread;
    const int block_num = (total_threads_needed + adaptive_block_size - 1) / adaptive_block_size;

    printf("Launch configuration: %d blocks, %d threads per block\n", block_num, adaptive_block_size);
    printf("Elements per thread: %d, Total threads needed: %d\n", elements_per_thread, total_threads_needed);

    // --- Timing Setup ---
    const int NUM_RUNS = 50;
    dtype total_time = 0.0;
    dtype times[NUM_RUNS];
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    // --- Warmup Run ---
    value_parallel_blocked_spmv_v3<<<block_num, adaptive_block_size>>>(
        d_csr.values, d_csr.row_pointers, d_csr.col_indices, d_vec, d_res, d_csr.num_non_zeros, n, elements_per_thread
    );
    
    cudaDeviceSynchronize();
    
    // --- Timed Runs ---
    for (int run = 0; run < NUM_RUNS; run++) {
        // Reset result vector before each run
        cudaMemset(d_res, 0, n * sizeof(dtype));

        cudaEventRecord(start);

        value_parallel_blocked_spmv_v3<<<block_num, adaptive_block_size>>>(
            d_csr.values, d_csr.row_pointers, d_csr.col_indices, d_vec, d_res, d_csr.num_non_zeros, n, elements_per_thread
        );
            
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
    cudaEventDestroy(end);

    for (int i = 0; i < NUM_RUNS; i++) {
        total_time += times[i];
    }
    dtype avg_time = total_time / NUM_RUNS;

    double bandwidth, gflops;
    calculate_bandwidth(n,m,nnz,h_csr.col_indices, avg_time, &bandwidth, &gflops);

    // --- Print Results ---
    print_spmv_performance(
        "Value Parallel Blocked v3 (Hash Table)", 
        argv[1],
        n, 
        m, 
        nnz, 
        avg_time, 
        bandwidth, 
        gflops, 
        h_res,
        10  // Print up to 10 samples
    );

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