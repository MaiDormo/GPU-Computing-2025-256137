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
         free(h_csr.values); free(h_csr.col_indices); free(h_csr.row_pointers);;
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
    const int block_num = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // const size_t shared_mem = BLOCK_SIZE * sizeof(dtype);

    // --- Timing Setup ---
    const int NUM_RUNS = 50;
    dtype total_time = 0.0;
    dtype times[NUM_RUNS];
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    // --- Warmup Run ---
    spmv_simple<<<block_num, BLOCK_SIZE>>>(
        d_csr.values, d_csr.row_pointers, d_csr.col_indices, d_vec, d_res, n
    );
    
    cudaDeviceSynchronize();
    
    // --- Timed Runs ---
    for (int run = 0; run < NUM_RUNS; run++) {
        cudaEventRecord(start);

        spmv_simple<<<block_num, BLOCK_SIZE>>>(
            d_csr.values, d_csr.row_pointers, d_csr.col_indices, d_vec, d_res, n
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

    // Calculate memory bandwidth more accurately for SpMV
    // For CSR SpMV, memory access pattern is:
    // 1. Read all row_pointers (accessed sequentially)
    // 2. Read all values and col_indices (accessed sequentially) 
    // 3. Read vector elements (potentially random access pattern)
    // 4. Write result vector (sequential)

    size_t bytes_read_vals = (size_t)nnz * sizeof(dtype);           // matrix values
    size_t bytes_read_cols = (size_t)nnz * sizeof(int);            // column indices  
    size_t bytes_read_row_ptr = (size_t)(n + 1) * sizeof(int);     // row pointers
    
    // For vector reads, each column index causes a vector element read
    // This is more accurate than counting unique columns because:
    // 1. Cache misses depend on access pattern, not just unique elements
    // 2. Simple CSR kernel may not optimize for reused vector elements
    size_t bytes_read_vec = (size_t)nnz * sizeof(dtype);           // vector reads (one per nnz)
    
    size_t bytes_written = (size_t)n * sizeof(dtype);              // result vector
    
    // Total memory traffic
    size_t total_bytes = bytes_read_vals + bytes_read_cols + 
                        bytes_read_row_ptr + bytes_read_vec + bytes_written;

    // Memory bandwidth calculation
    double bandwidth = total_bytes / (avg_time * 1.0e9);  // GB/s
    
    // Computational intensity
    double flops = 2.0 * nnz;  // Each non-zero: 1 multiply + 1 add
    double gflops = flops / (avg_time * 1.0e9);  // GFLOPS
    
    // Calculate arithmetic intensity for roofline analysis
    double arithmetic_intensity = flops / (double)total_bytes;  // FLOPS/Byte

    // --- Print Matrix Statistics ---
    print_matrix_stats(&h_csr);

    // --- Print Results with Additional Metrics ---
    printf("\n=== Simple CSR Performance Results ===\n");
    printf("Matrix: %s\n", argv[1]);
    printf("Dimensions: %d x %d, NNZ: %d\n", n, m, nnz);
    printf("Average time: %.6f seconds\n", avg_time);
    printf("Memory bandwidth: %.2f GB/s\n", bandwidth);
    printf("Compute performance: %.2f GFLOPS\n", gflops);
    printf("Arithmetic intensity: %.3f FLOPS/Byte\n", arithmetic_intensity);
    printf("Memory breakdown:\n");
    printf("  Matrix values: %.2f MB\n", bytes_read_vals / (1024.0 * 1024.0));
    printf("  Column indices: %.2f MB\n", bytes_read_cols / (1024.0 * 1024.0));
    printf("  Row pointers: %.2f MB\n", bytes_read_row_ptr / (1024.0 * 1024.0));
    printf("  Vector reads: %.2f MB\n", bytes_read_vec / (1024.0 * 1024.0));
    printf("  Result writes: %.2f MB\n", bytes_written / (1024.0 * 1024.0));
    printf("  Total memory: %.2f MB\n", total_bytes / (1024.0 * 1024.0));

    // Also call the standard print function for consistency
    print_spmv_performance(
        "Simple CSR", 
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