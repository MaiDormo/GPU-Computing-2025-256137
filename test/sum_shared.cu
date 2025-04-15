#include <iostream>
#include <iomanip>
#include <math.h>
#include <time.h>

#define dtype double

/*
## Metrics Asked For Lab
| Metric | Value |
|--------|-------|
| Operations per Second | 2430 MHz (Memory clock rate × 2 due to DDR) |
| Total Number of Byte Exchanged | 384 Byte (Memory bus width / 8 (bit in a Byte)) |
| Streaming Multiprocessor | 56 (as Referenced Before) |
*/

__global__ void add_shared_memory(const int n, dtype *x, dtype *y) {
    //Max double that can be loaded inside the shared memory (78*78)
	//data inside the shared memory
	extern __shared__ dtype sharedData[];
    
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    const int dim = n < 36 ? n : 36;

    //load to shared memory
    const int pos_in_shared_vec = idx % dim;
    sharedData[pos_in_shared_vec] = x[idx]; //load x
    sharedData[pos_in_shared_vec + dim] = y[idx]; //load y
    // each thread computes
    sharedData[pos_in_shared_vec + dim] = sharedData[pos_in_shared_vec + dim] + sharedData[pos_in_shared_vec];
    // synch thread
    __syncthreads(); //reset values
    //send_back result
    y[idx] = sharedData[pos_in_shared_vec + dim];
}

__global__ void add_2d_thread_blocks(const int n, dtype *x, dtype *y) {

}

__global__ void add_shared_memory_and_2d_thread_blocks(const int n, dtype *x, dtype *y) {

}


void add_shared_memory_bench(const int runs, cudaEvent_t start, cudaEvent_t end, 
    const int thread_per_block, const int block_num, int N, dtype *x, dtype *y , float times[]) {
    float millis = 0.0;
    for (int i = 0; i < runs; i++) {
        cudaEventRecord(start);
        add_shared_memory<<<block_num, thread_per_block>>>(N, x, y);
        cudaEventRecord(end);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&millis, start, end);
        times[runs] = millis;
    }
}

void print_stat(const float times[], const int runs, const int N, const double values[], const char * unit) {

    
    // Print separator
    std::cout << std::string(12, '-');
    for (int j = 0; j < runs; j++) {
        std::cout << "+" << std::string(12, '-');
    }
    std::cout << std::endl;

    // Print results for each block size
    for (int i = 0; i < runs; i++) {
        if (values == NULL) {
            std::cout << "| " << std::fixed << std::setprecision(2)
                 << std::setw(8) << times[i] << unit << " ";
        } else {
            std::cout << "| " << std::fixed << std::setprecision(2)
                     << std::setw(6) << values[i] << unit << " ";
        }
    }
    std::cout << std::endl;
}


void compute_result(const float times[], int runs, int N, const char* test_name) {
    
    size_t bytes_read = N * N * sizeof(dtype) * 2;
    size_t bytes_written = N * N * sizeof(dtype);
    size_t total_bytes = bytes_read + bytes_written;

    std::cout << "\n===== " << test_name << " Results =====\n";

    double values[runs];

    print_stat(times,runs,N,NULL,"ms");

    //compute the bandwidth (GB/s)
    for (int i = 0; i < runs; i++) {
        values[i] = total_bytes /  (times[i] * 1.0e6); // GB/s
    }

    print_stat(times,runs,N,values,"GB/s");

    //compute the GFLOPS
    for (int i = 0; i < runs; i++) {
        values[i] = (N * N) / (times[i] * 1.0e6);
    }

    print_stat(times,runs,N,values,"GFLOPS");
    
}


void add_shared_mem_compute_result(const float shared_mem_times[], int runs, int N) {
    compute_result(shared_mem_times, runs, N, "Shared Memory");
}



int main(int argc, char **argv) {

    const int N = 4096 * 4;
    
    // For benchmarking
    const int runs = 6;
    const int thread_per_block = 36*36;
    const int block_nums = (N*N + thread_per_block - 1) / thread_per_block;
    float times[runs];
    
    // Allocate Unified Memory accessible from CPU or GPU
    dtype *x, *y;
    cudaMallocManaged(&x, N * N * sizeof(dtype));
    cudaMallocManaged(&y, N * N * sizeof(dtype));
    
    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int pos = i * N + j;
            x[pos] = pos;
            x[pos] = pos; 
        }
    }

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    //---------------BENCHMARKING-------------------

    add_shared_memory_bench(runs,start,end,thread_per_block,block_nums,N,x,y,times);

    
    //---------------COMPUTE RESULTS------------------

    add_shared_mem_compute_result(times,runs,N);
    printf("\n");

    //------------------------------------------------

    
    // Free memory
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    cudaFree(x);
    cudaFree(y);
}
