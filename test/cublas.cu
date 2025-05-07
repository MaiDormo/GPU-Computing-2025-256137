#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <wchar.h>
#include <time.h>
#include <sys/time.h>
#include <cusparse.h>
#include <omp.h>

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

void read_from_file_and_init(char * file_path, double ** a_val, int ** a_row, int ** a_col, 
    int * mat_rows, int * mat_cols, int * vec_size) {
    FILE * file;
    const size_t BUFFER_SIZE = 1 * 1024 * 1024; // 1MB

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

//implement a kernel that does the conversion (at least the sorting)

int coo_to_csr(int n_rows, int nnz, 
    const int* a_row, const int* a_col, const double* a_val,
    double* csr_values, int* csr_col_indices, int* csr_row_ptr) {

    //error handling
    if (n_rows <= 0 || nnz < 0) return -1;
    if (nnz > 0 && (!a_row || !a_col || !a_val)) return -1;
    if (!csr_values || !csr_col_indices || !csr_row_ptr) return -1;

    // Count number of elements in each row
    // (Assuming csr_row_pointer to be already initialized with zeros)
    for (int i = 0; i < nnz; i++) {
        if(a_row[i] >= n_rows || a_row[i] < 0) return -1;
        csr_row_ptr[a_row[i] + 1]++;
    }

    // Cumulative sum to get row pointers
    for (int i = 0; i < n_rows; i++) {
        csr_row_ptr[i + 1] += csr_row_ptr[i];
    }

    // Copy values and column indicies to their correct positions
    int * temp_row_counts = (int *)calloc(n_rows,sizeof(int));
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

int main(int argc, char ** argv) {
    if (argc != 2) {
        printf("Usage: <./bin/spmv> <path/to/file.mtx>\n");
        return -1;
    }

    const double alpha = 1.0;
    const double beta = 0.0;

    int * a_row;
    int * a_col;
    double * a_val;
    int n, m, n_val;

    read_from_file_and_init(argv[1], &a_val, &a_row, &a_col, &n, &m, &n_val);
    
    // Host memory allocations
    double *h_vec = (double*)malloc(m * sizeof(double));
    double *h_res = (double*)malloc(n * sizeof(double));
    double *h_csr_values = (double*)malloc(n_val * sizeof(double));
    int *h_csr_col_indices = (int*)malloc(n_val * sizeof(int));
    int *h_csr_row_ptr = (int*)calloc(n+1, sizeof(int)); // Initialize to zero

    // Initialize vector with ones
    for (int i = 0; i < m; i++) {
        h_vec[i] = 1.0;
    }

    // Initialize result array with zeros
    memset(h_res, 0, n * sizeof(double));

    // Convert COO to CSR format in host memory
    coo_to_csr(n, n_val, a_row, a_col, a_val, h_csr_values, h_csr_col_indices, h_csr_row_ptr);

    // Free original COO data as we don't need it anymore
    free(a_row);
    free(a_col);
    free(a_val);

    // Device memory allocations
    double *d_vec, *d_res, *d_csr_values;
    int *d_csr_col_indices, *d_csr_row_ptr;
    
    cudaMalloc(&d_vec, m * sizeof(double));
    cudaMalloc(&d_res, n * sizeof(double));
    cudaMalloc(&d_csr_values, n_val * sizeof(double));
    cudaMalloc(&d_csr_col_indices, n_val * sizeof(int));
    cudaMalloc(&d_csr_row_ptr, (n+1) * sizeof(int));

    // Copy data to device
    cudaMemcpy(d_vec, h_vec, m * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_res, h_res, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csr_values, h_csr_values, n_val * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csr_col_indices, h_csr_col_indices, n_val * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csr_row_ptr, h_csr_row_ptr, (n+1) * sizeof(int), cudaMemcpyHostToDevice);

    // Initialize cuSPARSE
    cusparseHandle_t handle;
    cusparseCreate(&handle);

    // Create matrix/vector descriptors using the new generic API
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;
    void* dBuffer = NULL;
    size_t bufferSize = 0;

    // Create sparse matrix A in CSR format
    cusparseCreateCsr(&matA, n, m, n_val,
                    d_csr_row_ptr, d_csr_col_indices, d_csr_values,
                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

    // Create dense vector X
    cusparseCreateDnVec(&vecX, m, d_vec, CUDA_R_64F);

    // Create dense vector Y
    cusparseCreateDnVec(&vecY, n, d_res, CUDA_R_64F);

    // Allocate an external buffer if needed
    cusparseSpMV_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
        CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize);
    cudaMalloc(&dBuffer, bufferSize);


    // Run SpMV multiple times to get accurate timing
    const int NUM_RUNS = 50;
    double total_time = 0.0;
    double times[NUM_RUNS];

    // Create event timing objects
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    // Warmup run - using cuSPARSE SpMV operation
    cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
        CUSPARSE_SPMV_ALG_DEFAULT, dBuffer);
    cudaDeviceSynchronize();
    
    // Timed runs
    for (int run = 0; run < NUM_RUNS; run++) {   
        //start the event
        cudaEventRecord(start);

        // Call cuSPARSE SpMV
        cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
            CUSPARSE_SPMV_ALG_DEFAULT, dBuffer);
        
        //close event
        cudaEventRecord(end);
        
        //wait to synch
        cudaEventSynchronize(end);
        
        //extract time
        float millisec = 0.0;
        cudaEventElapsedTime(&millisec, start, end);
        times[run] = millisec * 1e-3;
    }

    // Copy result back after all runs
    cudaMemcpy(h_res, d_res, n * sizeof(double), cudaMemcpyDeviceToHost);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    for (int i = 0; i < NUM_RUNS; i++) {
        total_time += times[i];
    }

    double avg_time = total_time / NUM_RUNS;
    // Calculate bandwidth and computation metrics
    size_t bytes_read = n_val * (sizeof(double) + sizeof(int)) +    // values and col indices
                        (n + 1) * sizeof(int) +                     // row pointers
                        m * sizeof(double);                     // vector reads (worst case)
    
    size_t bytes_written = n * sizeof(double);                      // result vector
    size_t total_bytes = bytes_read + bytes_written;

    double bandwidth = total_bytes / (avg_time * 1.0e9);  // GB/s
    double flops = 2.0 * n_val;  // Each non-zero element requires a multiply and add
    double gflops = flops / (avg_time * 1.0e9);  // GFLOPS

    printf("\nSpMV Performance cuSPARSE CSR:\n");
    printf("Matrix size: %d x %d with %d non-zero elements\n", n, m, n_val);
    printf("Average execution time: %.6f seconds\n", avg_time);
    printf("Memory bandwidth: %.2f GB/s\n", bandwidth);
    printf("Computational performance: %.2f GFLOPS\n", gflops);

    // First few elements of result vector
    printf("\nFirst non zero element of result vector (or fewer if n < 10):\n");
    int nnz_counter = 0;
    for (int i = 0; i < n; i++) {
        if (h_res[i] == 0.0)
            continue;
        else if (nnz_counter++ < 10) {
            printf("%f ",h_res[i]);
        } else {
            break;
        }
    }
    printf("\n");

    // Clean up cuSPARSE resources
    cusparseDestroySpMat(matA);
    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecY);
    if (dBuffer) cudaFree(dBuffer);
    cusparseDestroy(handle);

    // Free device memory
    cudaFree(d_vec);
    cudaFree(d_res);
    cudaFree(d_csr_values);
    cudaFree(d_csr_col_indices);
    cudaFree(d_csr_row_ptr);

    // Free host memory
    free(h_vec);
    free(h_res);
    free(h_csr_values);
    free(h_csr_col_indices);
    free(h_csr_row_ptr);
    
    return 0;
}