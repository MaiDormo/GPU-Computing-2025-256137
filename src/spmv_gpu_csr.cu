#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <wchar.h>
#include <time.h>
#include <sys/time.h>
#include <omp.h>
//#include <limits.h>

#include "../include/read_file_lib.h"
#include "../include/spmv_type.h"


#define WARP_SIZE 32
#define BLOCK_SIZE 1024

// --- Function Prototypes ---
int coo_to_csr(const struct COO *coo_data, struct CSR *csr_data);
__global__ void vector_csr(const dtype *csr_values, const int *csr_row_ptr, const int *csr_col_indices,
                                const dtype *vec, dtype *res, int n);




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

void print_matrix_stats(struct CSR *csr_data) {
    //number of rows
    //number of cols
    //number of nnz
   
    //nnz per row
    //percentage of non zero compared to all values

    double avg_nnz_per_row = csr_data->num_non_zeros / csr_data->num_rows;
    double percentage_nnz = csr_data->num_non_zeros / (csr_data->num_cols * csr_data->num_rows); 

    int min_nnz = csr_data->num_non_zeros;
    int max_nnz = 0;
    for (int i = 0; i < csr_data->num_rows; i++) {
        int nnz = csr_data->row_pointers[i+1] - csr_data->row_pointers[i];
        min_nnz = min_nnz > nnz ? nnz : min_nnz;
        max_nnz = max_nnz < nnz ? nnz : max_nnz;
    }
    
    printf("Number of Columns: %d\n", csr_data->num_cols);
    printf("Number of Rows: %d\n", csr_data->num_rows);
    printf("Number of NNZ: %d\n", csr_data->num_non_zeros);
    printf("Average NNZ per Row: %f\n", avg_nnz_per_row);
    printf("Min NNZ: %d\n", min_nnz);
    printf("Max NNZ: %d\n", max_nnz);
    printf("Percentage of NNZ: %f%\n", percentage_nnz*100);
}



// --- SpMV Kernel ---
__global__ void vector_csr(const dtype *csr_values, const int *csr_row_ptr, const int *csr_col_indices,
                          const dtype *vec, dtype *res, int n) { // n is num_rows

    const int tid = threadIdx.x;
    const int warps_per_block = BLOCK_SIZE / WARP_SIZE;
    const int block_start_row = blockIdx.x * warps_per_block;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int actual_row = block_start_row + warp_id;

    if (actual_row < n) {
        dtype thread_sum = 0.0;
        int row_start = csr_row_ptr[actual_row];
        int row_end = csr_row_ptr[actual_row + 1];

        // Improved vector access using texture memory
        for (int j = row_start + lane_id; j < row_end; j += WARP_SIZE) {
            int col = csr_col_indices[j];
            thread_sum += csr_values[j] * __ldg(&vec[col]);
        }

        #pragma unroll
        for (int delta = WARP_SIZE / 2; delta > 0; delta >>= 1) {
            thread_sum += __shfl_down_sync(0xFFFFFFFF, thread_sum, delta);
        }

        if (lane_id == 0) {
            res[actual_row] = thread_sum;
        }
    }
}







__global__ void vector_csr_unrolled(const dtype *csr_values, const int *csr_row_ptr, const int *csr_col_indices,
                                const dtype *vec, dtype *res, int n) { // n is num_rows

    const int tid = threadIdx.x;
    const int warps_per_block = BLOCK_SIZE / WARP_SIZE;
    const int block_start_row = blockIdx.x * warps_per_block;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int actual_row = block_start_row + warp_id;

    __shared__ volatile dtype vals[BLOCK_SIZE];

    if (actual_row < n) {
        dtype thread_sum = 0.0;
        int row_start = csr_row_ptr[actual_row];
        int row_end = csr_row_ptr[actual_row + 1];

        for (int j = row_start + lane_id; j < row_end; j += WARP_SIZE) {
            int col = csr_col_indices[j];
            thread_sum += csr_values[j] * __ldg(&vec[col]);
        }

        vals[tid] = thread_sum;
        __syncthreads();


        // Reduce partial sums loop unrolled
        if (lane_id < 16) vals[tid] += vals[tid + 16];
        if (lane_id < 8) vals[tid] += vals[tid + 8];
        if (lane_id < 4) vals[tid] += vals[tid + 4];
        if (lane_id < 2) vals[tid] += vals[tid + 2];
        if (lane_id < 1) vals[tid] += vals[tid + 1];
        __syncthreads();

        if (lane_id == 0) {
            res[actual_row] = vals[tid];
        }
    }
}


__global__ void adaptive_csr(const dtype *csr_values, const int *csr_row_ptr,
                            const int *csr_col_indices, const dtype *vec,
                            dtype *res, const int *row_blocks, int n) {
   __shared__ volatile dtype SHARED_MEM[BLOCK_SIZE];

    const int tid = threadIdx.x;
    const int WARP_NUM = tid / WARP_SIZE;
    const int WARP_ID = tid % WARP_SIZE;

    int block_row_start = row_blocks[blockIdx.x];
    int block_row_end = row_blocks[blockIdx.x + 1];
    int num_rows = block_row_end - block_row_start;
    if (num_rows > 1) {
        //that means we have a warp per row
        int warp_row_index = block_row_start + WARP_NUM;
        int warp_row_start = csr_row_ptr[warp_row_index];
        int warp_row_end = csr_row_ptr[warp_row_index + 1];
        int nnz = warp_row_end - warp_row_start;
        dtype thread_sum = 0.0;
        
        for (int i = warp_row_start + WARP_ID; i < warp_row_end; i += WARP_SIZE) {
            thread_sum += csr_values[i] * __ldg(&vec[csr_col_indices[i]]);
        }

        //warp reduction
        #pragma unroll
        for (int delta = WARP_SIZE / 2; delta > 0; delta >>= 1) {
            thread_sum += __shfl_down_sync(0xFFFFFFFF, thread_sum, delta);
        }

        if (WARP_ID == 0) {
            res[warp_row_index] = thread_sum;
        }
        
    } else {
        //that means we have a block per row
        int row_idx = block_row_start;
        int row_start = csr_row_ptr[block_row_start];
        int row_end = csr_row_ptr[block_row_end];
        int nnz = row_end - row_start;
        dtype thread_sum = 0.0;
        for (int i = row_start + tid; i < row_end; i += blockDim.x) {
            thread_sum += csr_values[i] * __ldg(&vec[csr_col_indices[i]]);
        }

        //warp reduction
        #pragma unroll
        for (int delta = WARP_SIZE / 2; delta > 0; delta >>= 1) {
            thread_sum += __shfl_down_sync(0xFFFFFFFF, thread_sum, delta);
        }

        //one thread of each warp saves on shared memory
        if (WARP_ID == 0) {
            SHARED_MEM[WARP_NUM] = thread_sum;
        }

        if (tid < WARP_SIZE) {
            #pragma unroll
            for (int delta = WARP_SIZE / 2; delta > 0; delta >>= 1) {
                thread_sum += __shfl_down_sync(0xFFFFFFFF, thread_sum, delta);
            }

            if (tid == 0)
                res[row_idx] = thread_sum;
        }
    }
}

int adaptive_row_selection(const int *csr_row_ptr, int rows, int *row_blocks) {

    // inside this function i would like to compute how many rows are dispatched per block,
    // accounting for the fact that a warp is 32 threads and a block has 1024 threads, 
    // -> so at most a block can handle 8 rows!
    const int MAX_ROWS_PER_BLOCK = BLOCK_SIZE / WARP_SIZE;
    const int MAX_NNZ_PER_WARP = WARP_SIZE * 4;
    const int MAX_NNZ_PER_BLOCK = MAX_NNZ_PER_WARP * MAX_ROWS_PER_BLOCK;
    const int VERY_DENSE_ROW = MAX_NNZ_PER_WARP * 4;

    row_blocks[0] = 0;
    int block_idx = 1;
    int current_block_rows = 0;
    int current_block_nnz = 0;
    
    for (int row = 0; row < rows; row++) {
        int row_nnz = csr_row_ptr[row+1] - csr_row_ptr[row];

        // Case 1: Very dense row - put in its own block
        if (row_nnz > VERY_DENSE_ROW) {
            // if we've accumulated rows, finish the previous block
            if (current_block_rows > 0) {
                row_blocks[block_idx++] = row;
                current_block_rows = 0;
                current_block_nnz = 0;
            }

            //create a dedicated block for this row
            row_blocks[block_idx++] = row + 1;
            continue;

        }
        
        // Case 2: Adding this row would exceed block limits
        if ((current_block_rows + 1) > MAX_ROWS_PER_BLOCK || 
            (current_block_nnz + row_nnz > MAX_NNZ_PER_BLOCK)) {
                row_blocks[block_idx++] = row;
                current_block_rows = 1;
                current_block_nnz = row_nnz;
        } else {
            current_block_rows++;
            current_block_nnz += row_nnz;
        }

    }

    // Handle any remaining rows
    if (current_block_rows > 0) {
        row_blocks[block_idx++] = rows;
    }

    return block_idx;
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
    int *h_block_rows = (int*)calloc(n, sizeof(int));


    if (!h_vec || !h_res || !h_csr.values || !h_csr.col_indices || !h_csr.row_pointers || !h_block_rows) {
        perror("Failed to allocate host memory");
        // Free any successfully allocated memory before exiting
        free(h_coo.a_val); free(h_coo.a_row); free(h_coo.a_col);
        free(h_vec); free(h_res);
        free(h_csr.values); free(h_csr.col_indices); free(h_csr.row_pointers);
        free(h_block_rows);
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
         free(h_block_rows);
         return -1;
    }

    // Free original COO data (host)
    free(h_coo.a_val); h_coo.a_val = NULL;
    free(h_coo.a_row); h_coo.a_row = NULL;
    free(h_coo.a_col); h_coo.a_col = NULL;

    int countRowBlocks = adaptive_row_selection(h_csr.row_pointers, n, h_block_rows);

    printf("Number of rowBlocks: %d\n", countRowBlocks);

    // --- Device Data Structures ---
    struct CSR d_csr; // Holds device pointers
    dtype *d_vec = NULL, *d_res = NULL;
    int * d_block_rows;

    // --- Allocate Device Memory ---
    cudaMalloc(&d_vec, m * sizeof(dtype));
    cudaMalloc(&d_res, n * sizeof(dtype));
    cudaMalloc(&d_csr.values, nnz * sizeof(dtype));
    cudaMalloc(&d_csr.col_indices, nnz * sizeof(int));
    cudaMalloc(&d_csr.row_pointers, (n + 1) * sizeof(int));
    cudaMalloc(&d_block_rows, countRowBlocks * sizeof(int));
    d_csr.num_rows = n;
    d_csr.num_cols = m;
    d_csr.num_non_zeros = nnz;

    // --- Copy Data to Device ---
    cudaMemcpy(d_vec, h_vec, m * sizeof(dtype), cudaMemcpyHostToDevice);
    cudaMemcpy(d_res, h_res, n * sizeof(dtype), cudaMemcpyHostToDevice); // Copy initial zeros
    cudaMemcpy(d_csr.values, h_csr.values, nnz * sizeof(dtype), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csr.col_indices, h_csr.col_indices, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csr.row_pointers, h_csr.row_pointers, (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_block_rows, h_block_rows, countRowBlocks * sizeof(int), cudaMemcpyHostToDevice);

    // --- Kernel Launch Configuration ---
    const int warps_per_block = BLOCK_SIZE / WARP_SIZE;
    const int rows_per_block = warps_per_block / 2;
    const int block_num = (n + warps_per_block - 1) / warps_per_block;
    const size_t shared_mem = BLOCK_SIZE * sizeof(dtype); //!TODO configure shared_mem


    print_matrix_stats(&h_csr);

    // --- Timing Setup ---
    const int NUM_RUNS = 50;
    dtype total_time = 0.0;
    dtype times[NUM_RUNS];
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    // --- Warmup Run ---
    // vector_csr_unrolled<<<block_num, thread_per_block, dynamic_shared_mem>>>(
        // d_csr.values, d_csr.row_pointers, d_csr.col_indices, d_vec, d_res, n);
    vector_csr<<<block_num, BLOCK_SIZE, shared_mem>>>(d_csr.values, d_csr.row_pointers, d_csr.col_indices, d_vec, d_res, n);

    // adaptive_csr<<<block_num, BLOCK_SIZE, shared_mem>>>(d_csr.values, d_csr.row_pointers, d_csr.col_indices, d_vec, d_res, d_block_rows, n);
    
    cudaDeviceSynchronize();

    // --- Timed Runs ---
    for (int run = 0; run < NUM_RUNS; run++) {
        cudaEventRecord(start);

        // vector_csr_unrolled<<<block_num, thread_per_block, dynamic_shared_mem>>>(
            // d_csr.values, d_csr.row_pointers, d_csr.col_indices, d_vec, d_res, n);
        
        vector_csr<<<block_num, BLOCK_SIZE, shared_mem>>>(
            d_csr.values, d_csr.row_pointers, d_csr.col_indices, d_vec, d_res, n);

        // adaptive_csr<<<block_num, BLOCK_SIZE, shared_mem>>>(
        //     d_csr.values, d_csr.row_pointers, d_csr.col_indices, d_vec, d_res, d_block_rows, n
        // );
            

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
                        (size_t)m * sizeof(dtype) +      // vector reads (worst case estimate)
                        (size_t)countRowBlocks * sizeof(int);
    size_t bytes_written = (size_t)n * sizeof(dtype);                 // result vector
    size_t total_bytes = bytes_read + bytes_written;

    double bandwidth = total_bytes / (avg_time * 1.0e9);  // GB/s
    double flops = 2.0 * nnz;
    double gflops = flops / (avg_time * 1.0e9);  // GFLOPS

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
    cudaFree(d_block_rows);

    free(h_vec);
    free(h_res);
    free(h_csr.values);
    free(h_csr.col_indices);
    free(h_csr.row_pointers);
    free(h_block_rows);

    return 0;
}