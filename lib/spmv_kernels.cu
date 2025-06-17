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

#define HASH_SIZE 256

__device__ __forceinline__ int hash_function(int row) {
    return row % HASH_SIZE;
}

//this is complete ass
__global__ void value_parallel_blocked_spmv_v3(const dtype *csr_values, const int *csr_row_ptr, 
    const int *csr_col_indices, const dtype *vec,
    dtype *res, int nnz, int num_rows, int stride) {
    
    // Hash table in shared memory
    __shared__ dtype hash_sums[HASH_SIZE];
    __shared__ int hash_rows[HASH_SIZE];
    __shared__ int hash_locks[HASH_SIZE];
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int local_tid = threadIdx.x;
    int start_idx = tid * stride;
    
    // Initialize hash table
    for (int i = local_tid; i < HASH_SIZE; i += blockDim.x) {
        hash_sums[i] = 0.0;
        hash_rows[i] = -1;
        hash_locks[i] = 0;
    }
    __syncthreads();
    
    // Phase 1: Accumulate in shared hash table
    for (int i = 0; i < stride; i++) {
        int idx = start_idx + i;
        if (idx >= nnz) break;
        
        // Binary search for row
        int row = 0;
        int left = 0;
        int right = num_rows - 1;
        
        while (left < right) {
            int mid = (left + right) >> 1;
            if (csr_row_ptr[mid + 1] > idx) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        row = left;
        
        int col = csr_col_indices[idx];
        dtype val = csr_values[idx] * __ldg(&vec[col]);
        
        // Insert into hash table with linear probing
        int hash_idx = hash_function(row);
        bool inserted = false;
        
        for (int probe = 0; probe < HASH_SIZE && !inserted; probe++) {
            int current_idx = (hash_idx + probe) % HASH_SIZE;
            
            // Try to acquire lock
            if (atomicCAS(&hash_locks[current_idx], 0, 1) == 0) {
                if (hash_rows[current_idx] == -1) {
                    // Empty slot
                    hash_rows[current_idx] = row;
                    hash_sums[current_idx] = val;
                    inserted = true;
                } else if (hash_rows[current_idx] == row) {
                    // Same row
                    hash_sums[current_idx] += val;
                    inserted = true;
                }
                // Release lock
                __threadfence_block();
                hash_locks[current_idx] = 0;
            }
        }
        
        // Fallback to global atomic if hash table full
        if (!inserted) {
            atomicAdd(&res[row], val);
        }
    }
    
    __syncthreads();
    
    // Phase 2: Flush hash table to global memory
    for (int i = local_tid; i < HASH_SIZE; i += blockDim.x) {
        if (hash_rows[i] != -1 && hash_sums[i] != 0.0) {
            atomicAdd(&res[hash_rows[i]], hash_sums[i]);
        }
    }
}


__global__ void vector_csr(const dtype *csr_values, const int *csr_row_ptr, 
                                  const int *csr_col_indices, const dtype *vec, 
                                  dtype *res, int n) {
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    
    // Simple: one row per warp
    const int row = blockIdx.x * (blockDim.x >> 5) + warp_id;
    
    if (row < n) {
        const int row_start = csr_row_ptr[row];
        const int row_end = csr_row_ptr[row + 1];
        
        dtype thread_sum = 0.0;
        
        // Process all elements in this row
        for (int j = row_start + lane_id; j < row_end; j += 32) {
            thread_sum += csr_values[j] * __ldg(&vec[csr_col_indices[j]]);
        }
        
        // Warp reduction
        thread_sum += __shfl_down_sync(0xFFFFFFFF, thread_sum, 16);
        thread_sum += __shfl_down_sync(0xFFFFFFFF, thread_sum, 8);
        thread_sum += __shfl_down_sync(0xFFFFFFFF, thread_sum, 4);
        thread_sum += __shfl_down_sync(0xFFFFFFFF, thread_sum, 2);
        thread_sum += __shfl_down_sync(0xFFFFFFFF, thread_sum, 1);
        
        if (lane_id == 0) {
            res[row] = thread_sum;
        }
    }
}

//now it kinda don't work on mawi...
// __global__ void vector_csr_double_buffer(const dtype *csr_values, const int *csr_row_ptr, 
//                                         const int *csr_col_indices, const dtype *vec, 
//                                         dtype *res, int n) {
//     // Shared memory for prefetching vector values
//     extern __shared__ dtype shared_vec[];
//     __shared__ int shared_cols[256];  // Cache frequently accessed columns
//     __shared__ volatile int cache_size;
    
//     const int tid = threadIdx.x;
//     const int warps_per_block = blockDim.x / WARP_SIZE;
//     const int warp_id = tid / WARP_SIZE;
//     const int lane_id = tid % WARP_SIZE;
    
//     // Initialize shared memory cache
//     if (tid == 0) {
//         cache_size = 0;
//     }
//     __syncthreads();
    
//     // Improved row assignment for better coalescing
//     const int rows_per_warp = 2;  // Reduced for better cache utilization
//     const int total_rows_per_block = warps_per_block * rows_per_warp;
//     const int block_start_row = blockIdx.x * total_rows_per_block;
    
//     // Process rows in a more coalesced manner
//     for (int r = 0; r < rows_per_warp; r++) {
//         int actual_row = block_start_row + warp_id + r * warps_per_block;
        
//         if (actual_row < n) {
//             dtype thread_sum = 0.0;
//             int row_start = csr_row_ptr[actual_row];
//             int row_end = csr_row_ptr[actual_row + 1];
//             int row_length = row_end - row_start;

//             // Early exit for empty rows
//             if (row_length == 0) {
//                 if (lane_id == 0) {
//                     res[actual_row] = 0.0;
//                 }
//                 continue;
//             }

//             // Improved memory access pattern
//             if (row_length >= WARP_SIZE) {
//                 // Prefetch column indices for better coalescing
//                 const int prefetch_size = min(WARP_SIZE, row_length);
                
//                 // Process in chunks for better memory coalescing
//                 for (int chunk_start = row_start; chunk_start < row_end; chunk_start += WARP_SIZE) {
//                     int j = chunk_start + lane_id;
//                     if (j < row_end) {
//                         int col = csr_col_indices[j];
//                         dtype val = csr_values[j];
                        
//                         // Use texture cache hint for vector access
//                         dtype vec_val = __ldg(&vec[col]);
//                         thread_sum += val * vec_val;
                        
//                         // Opportunistically cache frequently accessed columns
//                         if (cache_size < 256) {
//                             // Simple heuristic: cache if col index is "small" (likely reused)
//                             if (col < 1000 && lane_id == 0) {
//                                 int cache_idx = atomicAdd((int*)&cache_size, 1);
//                                 if (cache_idx < 256) {
//                                     shared_vec[cache_idx] = vec_val;
//                                     shared_cols[cache_idx] = col;
//                                 }
//                             }
//                         }
//                     }
//                 }
//             } else {
//                 // Optimized short row handling with better branching
//                 if (lane_id < row_length) {
//                     int j = row_start + lane_id;
//                     int col = csr_col_indices[j];
//                     dtype val = csr_values[j];
                    
//                     // Check cache first for short rows
//                     dtype vec_val;
//                     bool found_in_cache = false;
                    
//                     // Unrolled cache lookup for better performance
//                     #pragma unroll 8
//                     for (int k = 0; k < min(cache_size, 256); k += 8) {
//                         if (k < cache_size && shared_cols[k] == col) {
//                             vec_val = shared_vec[k];
//                             found_in_cache = true;
//                             break;
//                         }
//                     }
                    
//                     if (!found_in_cache) {
//                         vec_val = __ldg(&vec[col]);
//                     }
                    
//                     thread_sum = val * vec_val;
//                 }
//             }

//             // Optimized warp shuffle reduction
//             #pragma unroll
//             for (int offset = 16; offset > 0; offset >>= 1) {
//                 thread_sum += __shfl_down_sync(0xFFFFFFFF, thread_sum, offset);
//             }

//             // Coalesced write to result
//             if (lane_id == 0) {
//                 res[actual_row] = thread_sum;
//             }
//         }
//     }
// }

__global__ void vector_csr_double_buffer(const dtype *csr_values, const int *csr_row_ptr, 
                                        const int *csr_col_indices, const dtype *vec, 
                                        dtype *res, int n) {
    const int tid = threadIdx.x;
    const int warps_per_block = blockDim.x >> 5;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    
    // Process multiple rows per warp for better memory utilization
    const int rows_per_warp = 2;
    
    // Correct calculation: each block processes warps_per_block * rows_per_warp rows
    const int rows_per_block = warps_per_block * rows_per_warp;
    const int block_start_row = blockIdx.x * rows_per_block;
    
    for (int r = 0; r < rows_per_warp; r++) {
        // Correct row assignment: consecutive rows for each warp
        int actual_row = block_start_row + warp_id * rows_per_warp + r;
        
        if (actual_row < n) {
            dtype thread_sum = 0.0;
            int row_start = csr_row_ptr[actual_row];
            int row_end = csr_row_ptr[actual_row + 1];
            int row_length = row_end - row_start;

            // Early exit for empty rows
            if (row_length == 0) {
                if (lane_id == 0) {
                    res[actual_row] = 0.0;
                }
                continue;
            }

            for (int j = row_start + lane_id; j < row_end; j += WARP_SIZE) {
                int col = csr_col_indices[j];
                thread_sum += csr_values[j] * __ldg(&vec[col]);
            }

            #pragma unroll
            for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
                thread_sum += __shfl_down_sync(0xFFFFFFFF, thread_sum, offset);
            }

            if (lane_id == 0) {
                res[actual_row] = thread_sum;
            }
        }
    }
}


__global__ void adaptive_csr(const dtype *csr_values, const int *csr_row_ptr,
                                  const int *csr_col_indices, const dtype *vec,
                                  dtype *res, const int *row_blocks, int n) {
    __shared__ volatile dtype SHARED_MEM[BLOCK_SIZE >> 5]; // One slot per warp
    
    const int tid = threadIdx.x;
    const int WARP_NUM = tid >> 5; // Warp index within the block
    const int WARP_ID = tid & 31;  // Lane index within the warp
    
    int block_row_start = row_blocks[blockIdx.x];
    int block_row_end = row_blocks[blockIdx.x + 1];
    int num_rows_in_block = block_row_end - block_row_start;
    
    if (num_rows_in_block > 1) {
        // Warp per row strategy (existing code for this path)
        int warp_row_index = block_row_start + WARP_NUM;
        if (warp_row_index < block_row_end) {
            int row_val_start = csr_row_ptr[warp_row_index];
            int row_val_end = csr_row_ptr[warp_row_index + 1];
            dtype thread_sum = 0.0;
            
            for (int i = row_val_start + WARP_ID; i < row_val_end; i += WARP_SIZE) {
                thread_sum += csr_values[i] * __ldg(&vec[csr_col_indices[i]]);
            }
            
            #pragma unroll
            for (int delta = WARP_SIZE >> 1; delta > 0; delta >>= 1) {
                thread_sum += __shfl_down_sync(0xFFFFFFFF, thread_sum, delta);
            }
            
            if (WARP_ID == 0) {
                res[warp_row_index] = thread_sum;
            }
        }
    } else {
        // Block per row strategy
        int row_idx = block_row_start;
        int row_val_start = csr_row_ptr[row_idx];
        int row_val_end = csr_row_ptr[row_idx + 1];
        dtype thread_sum = 0.0;
        
        for (int i = row_val_start + tid; i < row_val_end; i += blockDim.x) {
            thread_sum += csr_values[i] * __ldg(&vec[csr_col_indices[i]]);
        }

        // Intra-warp reduction
        #pragma unroll
        for (int delta = WARP_SIZE >> 1; delta > 0; delta >>= 1) {
            thread_sum += __shfl_down_sync(0xFFFFFFFF, thread_sum, delta);
        }

        // Lane 0 of each warp writes its partial sum to shared memory
        if (WARP_ID == 0) {
            SHARED_MEM[WARP_NUM] = thread_sum;
        }
        __syncthreads(); // IMPORTANT: Ensure all warps have written to SHARED_MEM

        // The first warp (WARP_NUM == 0) sums the partial sums from SHARED_MEM
        if (WARP_NUM == 0) {
            dtype final_sum_val = 0.0;
            // WARP_ID iterates from 0 to WARP_SIZE-1.
            // We need to sum up to num_warps_in_block = blockDim.x / WARP_SIZE.
            if (WARP_ID < (blockDim.x / WARP_SIZE)) {
                final_sum_val = SHARED_MEM[WARP_ID];
            }

            // Reduce 'final_sum_val' within the first warp
            // The number of elements to reduce is num_warps_in_block.
            // Max num_warps_in_block is typically 32 for 1024 threads/block.
            #pragma unroll
            for (int delta = WARP_SIZE / 2; delta > 0; delta >>= 1) { // Max num_warps is 32, so WARP_SIZE/2 is fine.
                                                                    // For more general num_warps, adjust loop bound or use safer shuffle.
                final_sum_val += __shfl_down_sync(0xFFFFFFFF, final_sum_val, delta);
            }

            if (WARP_ID == 0) { // Thread 0 of the first warp writes the final sum
                res[row_idx] = final_sum_val;
            }
        }
    }
}