#include <math.h>
#include <time.h>
#include <iostream>
#include <float.h>

#define dtype double

/*
## Memory Specifications L40S
| Specification | Value |
|---------------|-------|
| Memory Clock Rate | 9001 Mhz |
| Memory Bus Width | 384-bit |
| L2 Cache Size | 100663296 bytes |
| Constant Memory | 65536 bytes |
| Shared Memory per Block | 49152 bytes |
| Shared Memory per Multiprocessor | 102400 bytes |
*/


//---------------------------KERNELS------------------------------------------
__global__ void add_shared_memory(const int n, dtype *x, dtype *y) {
    // Get shared memory array for this block
    extern __shared__ dtype sharedData[];
    
    // Calculate global index
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;
    
    const int local_idx = threadIdx.x;
    
    const int block_size = blockDim.x;
    
    if (idx < n*n) {
        sharedData[local_idx] = x[idx];
        sharedData[local_idx + block_size] = y[idx];
        sharedData[local_idx + block_size] += sharedData[local_idx];
        
        // Ensure computation is complete before writing back
        // __syncthreads();
        
        // Store result back to global memory
        y[idx] = sharedData[local_idx + block_size];
    }
}

__global__ void add_2d_thread_blocks(const int n, dtype *x, dtype *y) {
    // Calculate the global thread coordinates (x, y)
    // blockIdx.{x,y} gives the 2D index of the block in the grid
    // blockDim.{x,y} gives the dimensions (number of threads) of the block
    // threadIdx.{x,y} gives the 2D index of the thread within its block
    const int unique_x = threadIdx.x + blockDim.x * blockIdx.x;
    const int unique_y = threadIdx.y + blockDim.y * blockIdx.y;
    int unique_id = unique_x + n * unique_y;
    if (unique_x < n && unique_y < n) {
        y[unique_id] += x[unique_id];
    }
}

__global__ void add_shared_memory_and_2d_thread_blocks(const int n, dtype *x, dtype *y) {
    
    extern __shared__ dtype sharedData[];

    const int local_id = threadIdx.y * blockDim.x + threadIdx.x;

    const int unique_x = threadIdx.x + blockDim.x * blockIdx.x;
    const int unique_y = threadIdx.y + blockDim.y * blockIdx.y;
    const int unique_id = n * unique_y + unique_x;

    const int block_elem_count = blockDim.x * blockDim.y;

    if (unique_x < n && unique_y < n) {
        
        sharedData[local_id] = x[unique_id];
        sharedData[local_id + block_elem_count] = y[unique_id];
        sharedData[local_id + block_elem_count] += sharedData[local_id];
        
        // __syncthreads();
        
        y[unique_id] = sharedData[local_id + block_elem_count];
    }
}


//---------------------------------BENCHMARKING-------------------------------------------------------

void add_shared_memory_bench(const int runs, cudaEvent_t start, cudaEvent_t end, 
    const int thread_per_block, const int block_num, int N, dtype *x, dtype *y, float times[]) {
    float millis = 0.0;
    
    // Calculate shared memory based on threads per block, not arbitrary dim
    const size_t shared_mem_size = thread_per_block * 2 * sizeof(dtype);
    
    for (int i = 0; i < runs; i++) {
        cudaEventRecord(start);
        // No need to pass dim parameter anymore
        add_shared_memory<<<block_num, thread_per_block, shared_mem_size>>>(N, x, y);
        cudaEventRecord(end);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&millis, start, end);
        times[i] = millis;
    }
}

void add_2d_thread_blocks_bench(const int runs, cudaEvent_t start, cudaEvent_t end, 
    const dim3 thread_per_block, const dim3 block_num, int N, dtype *x, dtype *y , float times[]) {
    float millis = 0.0;

    for (int i = 0; i < runs; i++) {
        cudaEventRecord(start);
        add_2d_thread_blocks<<<block_num, thread_per_block>>>(N, x, y);
        cudaEventRecord(end);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&millis, start, end);
        times[i] = millis;
    }
}

void add_shared_mem_2d_bench(const int runs, cudaEvent_t start, cudaEvent_t end, 
    const dim3 thread_per_block, const dim3 block_num, int N, dtype *x, dtype *y , float times[]) {
    float millis = 0.0;

    const size_t shared_mem_size = thread_per_block.x * thread_per_block.y * 2 * sizeof(dtype);
    
    for (int i = 0; i < runs; i++) {
        cudaEventRecord(start);
        add_shared_memory_and_2d_thread_blocks<<<block_num, thread_per_block, shared_mem_size>>>(N, x, y);
        cudaEventRecord(end);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&millis, start, end);
        times[i] = millis;
    }
}

//---------------------------------COMPUTING RESULT AND PRINTING--------------------------


void print_stat(const float times[], const int runs, const double values[], const char * unit) {
    
    if (values == NULL) {
        // Calculate statistics for times

        printf("\tAll runs: ");
        for (int i = 0; i < runs; i++) {
            printf("%.2f",times[i]);
            if (i < runs - 1) printf(", ");
            
        }
        printf(" %s \n", unit);
    } else {
        // Calculate statistics for values
        
        printf("\tAll runs: ");
        for (int i = 0; i < runs; i++) {
            printf("%.2f",values[i]);
            if (i < runs - 1) printf(", ");
        }
        printf(" %s \n", unit);
    }
}


void compute_result(const float times[], int runs, int N, const char* test_name) {
    
    size_t bytes_read = N * N * sizeof(dtype) * 2;
    size_t bytes_written = N * N * sizeof(dtype);
    size_t total_bytes = bytes_read + bytes_written;

    std::cout << "\n===== " << test_name << " Results =====\n";

    double values[runs];

    print_stat(times,runs,NULL,"ms");

    //compute the bandwidth (TB/s)
    for (int i = 0; i < runs; i++) {
        values[i] = total_bytes / (times[i] * 1.0e9); // TB/s
    }

    print_stat(times,runs,values,"TB/s");

    //compute the GFLOPS
    for (int i = 0; i < runs; i++) {
        values[i] = (N * N) / (times[i] * 1.0e9);
    }

    print_stat(times,runs,values,"TFLOPS");
    
}


void add_shared_mem_compute_result(const float shared_mem_times[], int runs, int N) {
    compute_result(shared_mem_times, runs, N, "Shared Memory");
}

void add_2d_thread_blocks_compute_result(const float thread_block_2d_times[], int runs, int N) {
    compute_result(thread_block_2d_times, runs, N, "2D Block/Grid");
}

void add_2d_shared_mem_compute_result(const float block_2d_shared_mem_times[], int runs, int N) {
    compute_result(block_2d_shared_mem_times, runs, N, "2D Block/Grid and Shared Memory");
}


//----------------------------VALIDATION--------------------------------------------------


void check_and_reset_data(int N, dtype *x, dtype *y, int runs, char * computation) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int pos = i * N + j;
            if (y[pos] / (runs+1) != x[pos]) {
                printf("Error! computation not correct: %f != %f * (%d+1), at pos: %d, for computation: %s\n", y[pos], x[pos], runs, pos, computation);
                exit(-1);
            }
            y[pos] = pos;
            x[pos] = pos;   
        }
    }
}

//----------------------------------------------------------------------------------------

int main(void) {

    const int N = 2048;
    
    // Shared Memory 
    const int runs = 20;
    const int thread_per_block = 32*32; //1024
    const int block_nums = (N*N + thread_per_block - 1) / thread_per_block;
    float times_shared_mem[runs];
    
    // 2D thread blocks
    const int size_block = N < 32 ? N : 32;
    dim3 dimBlock(size_block,size_block);
    const int size_grid = (N + size_block - 1) / size_block;
    dim3 dimGrid(size_grid,size_grid);
    float times_2d_block[runs];


    // 2D and shared_mem
    float times_2d_shared_mem[runs];
    

    
    // Allocate Unified Memory accessible from CPU or GPU
    dtype *x, *y;
    cudaMallocManaged(&x, N * N * sizeof(dtype));
    cudaMallocManaged(&y, N * N * sizeof(dtype));
    
    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int pos = i * N + j;
            x[pos] = pos;
            y[pos] = pos; 
        }
    }

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    //---------------BENCHMARKING-------------------

    add_shared_memory_bench(runs,start,end,thread_per_block,block_nums,N,x,y,times_shared_mem);
    check_and_reset_data(N,x,y,runs,"Shared Mem");
    

    add_2d_thread_blocks_bench(runs,start,end,dimBlock,dimGrid,N,x,y,times_2d_block);
    check_and_reset_data(N,x,y,runs, "2D Blocks/Grid");

    add_shared_mem_2d_bench(runs,start,end,dimBlock,dimGrid,N,x,y,times_2d_shared_mem);
    check_and_reset_data(N,x,y,runs, "Shared Mem and 2D Blocks/Grid");
    
    //---------------COMPUTE RESULTS------------------

    add_shared_mem_compute_result(times_shared_mem,runs,N);
    printf("\n");
    

    add_2d_thread_blocks_compute_result(times_2d_block,runs,N);
    printf("\n");

    add_2d_shared_mem_compute_result(times_2d_shared_mem,runs,N);
    printf("\n");
    
    
    // Free memory
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    cudaFree(x);
    cudaFree(y);
}