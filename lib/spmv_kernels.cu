#include "../include/spmv_kernels.h"

__global__ void spmv_simple(const dtype *csr_values, const int *csr_row_ptr, 
                          const int *csr_col_indices, const dtype *vec, 
                          dtype *res, int num_rows) {
    // Each thread processes one row
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < num_rows) {
        // Get the range of this row's elements
        int row_start = csr_row_ptr[row];
        int row_end = csr_row_ptr[row + 1];
        
        // Process each non-zero element in the row
        dtype sum = 0.0;
        for (int j = row_start; j < row_end; j++) {
            int col = csr_col_indices[j];
            sum += csr_values[j] * vec[col];
        }
        
        // Write the result
        res[row] = sum;
    }
}

__global__ void value_parallel_sequential_spmv(const dtype *csr_values, const int *csr_row_ptr, 
                              const int *csr_col_indices, const dtype *vec, 
                              dtype *res, int nnz, int num_rows) {
    // Each thread processes one non-zero value
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < nnz) {
        // Find which row this element belongs to (binary search)
        int row = 0;
        int left = 0;
        int right = num_rows - 1;
        
        while (left <= right) {
            int mid = (left + right) >> 1;
            if (csr_row_ptr[mid] <= idx) {
                row = mid;
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        
        // Get column and compute contribution
        int col = csr_col_indices[idx];
        dtype val = csr_values[idx] * vec[col];
        
        // Add contribution to result vector using atomic operation
        atomicAdd(&res[row], val);
    }
}

__global__ void value_parallel_blocked_spmv(const dtype *csr_values, const int *csr_row_ptr, 
    const int *csr_col_indices, const dtype *vec,
    dtype *res, int nnz, int num_rows, int stride) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // Calculate stride as total number of threads (distance between memory addresses accessed by sequential threads)
    
    int start_idx = tid * stride;
    
    // Simple strided access: each thread starts at its ID and jumps by stride
    for (int i = 0; i < stride; i++) {
        int idx = start_idx + i;
        if (idx >= nnz) break;
        // Find which row this element belongs to (binary search)
        int row = 0;
        int left = 0;
        int right = num_rows - 1;

        while (left <= right) {
            int mid = (left + right) >> 1;
            if (csr_row_ptr[mid] <= idx) {
                row = mid;
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        
        int col = csr_col_indices[idx];
        dtype val = csr_values[idx] * vec[col];
        atomicAdd(&res[row], val);
    }
}

//------------------- NOT PART OF THE FIRST DELIVERABLE --------------------------------

__global__ void vector_csr(const dtype *csr_values, const int *csr_row_ptr, const int *csr_col_indices,
    const dtype *vec, dtype *res, int n) { 
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

        for (int j = row_start + lane_id; j < row_end; j += WARP_SIZE) {
            int col = csr_col_indices[j];
            thread_sum += csr_values[j] * __ldg(&vec[col]);
        }

        #pragma unroll
        for (int delta = WARP_SIZE >> 1; delta > 0; delta >>= 1) {
            thread_sum += __shfl_down_sync(0xFFFFFFFF, thread_sum, delta);
        }

        if (lane_id == 0) {
            res[actual_row] = thread_sum;
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
        // Warp per row with improved memory access
        int warp_row_index = block_row_start + WARP_NUM;
        if (warp_row_index < block_row_end) {
            int warp_row_start = csr_row_ptr[warp_row_index];
            int warp_row_end = csr_row_ptr[warp_row_index + 1];
            dtype thread_sum = 0.0;
            
            // Coalesced memory access pattern
            for (int i = warp_row_start + WARP_ID; i < warp_row_end; i += WARP_SIZE) {
                thread_sum += csr_values[i] * __ldg(&vec[csr_col_indices[i]]);
            }

            // Optimized warp reduction
            #pragma unroll
            for (int delta = WARP_SIZE >> 1; delta > 0; delta >>= 1) {
                thread_sum += __shfl_down_sync(0xFFFFFFFF, thread_sum, delta);
            }

            if (WARP_ID == 0) {
                res[warp_row_index] = thread_sum;
            }
        }
    } else {
        // Block per row with enhanced reduction
        int row_idx = block_row_start;
        int row_start = csr_row_ptr[row_idx];
        int row_end = csr_row_ptr[row_idx + 1];
        dtype thread_sum = 0.0;
        
        // Improved memory access pattern
        for (int i = row_start + tid; i < row_end; i += blockDim.x) {
            thread_sum += csr_values[i] * __ldg(&vec[csr_col_indices[i]]);
        }

        __syncthreads();

        // Two-level reduction: first within warps, then across warps
        #pragma unroll
        for (int delta = WARP_SIZE >> 1; delta > 0; delta >>= 1) {
            thread_sum += __shfl_down_sync(0xFFFFFFFF, thread_sum, delta);
        }

        // Final reduction across warps using only the first warp
        if (WARP_NUM == 0) { 
            dtype warp_sum = (WARP_ID < blockDim.x / WARP_SIZE) ? SHARED_MEM[WARP_ID] : 0.0;
            
            #pragma unroll
            for (int delta = WARP_SIZE >> 1; delta > 0; delta >>= 1) {
                warp_sum += __shfl_down_sync(0xFFFFFFFF, warp_sum, delta);
            }

            if (WARP_ID == 0) {
                res[row_idx] = warp_sum;
            }
        }
    }
}

