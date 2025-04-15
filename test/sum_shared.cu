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
    const int x_chunk_dim = dim;
    const int y_chunk_dim = dim;

    //load to shared memory
    const int pos_in_shared_vec = idx % dim;
    sharedData[pos_in_shared_vec] = x[idx]; //load x
    sharedData[pos_in_shared_vec + dim] = y[idx]; //load y
    // each thread computes
    sharedData[pos_in_shared_vec + dim] = sharedData[pos_in_shared_vec + dim] + sharedData[pos_in_shared_vec];
    // synch thread
    __syncthreads();
    //send_back result
    y[idx] = sharedData[pos_in_shared_vec + dim];
}

__global__ void add_2d_thread_blocks(const int n, dtype *x, dtype *y) {

}

__global__ void add_shared_memory_and_2d_thread_blocks(const int n, dtype *x, dtype *y) {

}


void bench(int len, cudaEvent_t start, const int grid_sizes[], const int block_sizes[], int N, dtype *x, dtype *y, cudaEvent_t end, float times[]) {
    float millis = 0.0;
    for (int i = 0; i < len; i++) {
        for (int j = 0; j < len; j++) {
            cudaEventRecord(start);
            add_linear_access<<<grid_sizes[j], block_sizes[i]>>>(N, x, y);
            cudaEventRecord(end);
            cudaEventSynchronize(end);
            cudaEventElapsedTime(&millis, start, end);
            times[j + i * len] = millis;
        }
    }
}


void linear_access_bench(cudaEvent_t start, cudaEvent_t end, const int grid_sizes[], 
    const int block_sizes[], float times[], int len, dtype* x, dtype* y, int N) {
    
    bench(len, start, grid_sizes, block_sizes, N, x, y, end, times);
}

void consecutive_access_bench(cudaEvent_t start, cudaEvent_t end, const int grid_sizes[], 
    const int block_sizes[], float times[], int len, dtype* x, dtype* y, int N) {
    
    bench(len, start, grid_sizes, block_sizes, N, x, y, end, times);
}

void print_stat(const int grid_sizes[], const int block_sizes[], const float times[], const int len, const int N,
    const double values[], const char * unit) {
    // Header row with grid sizes
    std::cout << std::left << std::setw(12) << "Block/Grid";
    for (int j = 0; j < len; j++) {
        std::cout << "| " << std::left << std::setw(10) << grid_sizes[j] << " ";
    }
    std::cout << std::endl;
    
    // Print separator
    std::cout << std::string(12, '-');
    for (int j = 0; j < len; j++) {
        std::cout << "+" << std::string(12, '-');
    }
    std::cout << std::endl;

    // Print results for each block size
    for (int i = 0; i < len; i++) {
        std::cout << std::left << std::setw(12) << block_sizes[i];
        for (int j = 0; j < len; j++) {
            if (values == NULL) {
                std::cout << "| " << std::fixed << std::setprecision(2)
                     << std::setw(8) << times[i*len+j] << unit << " ";
            } else {
                std::cout << "| " << std::fixed << std::setprecision(2)
                         << std::setw(6) << values[i*len+j] << unit << " ";
            }
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}


void compute_result(const int grid_sizes[], const int block_sizes[], 
    const float times[], int len, int N, const char* test_name) {
    
    size_t bytes_read = N * sizeof(dtype) * 2;
    size_t bytes_written = N * sizeof(dtype);
    size_t total_bytes = bytes_read + bytes_written;

    std::cout << "\n===== " << test_name << " Results =====\n";

    double values[len*len];

    print_stat(grid_sizes,block_sizes,times,len,N,NULL, "ms");

    //compute the bandwidth (GB/s)
    for (int i = 0; i < len; i++) {
        for (int j = 0; j < len; j++) {
            values[i*len+j] = total_bytes /  (times[i*len+j] * 1.0e6); // GB/s
        }
    }

    print_stat(grid_sizes,block_sizes,times,len,N,values, "GB/s");

    //compute the GFLOPS
    for (int i = 0; i < len; i++) {
        for (int j = 0; j < len; j++) {
            values[i*len+j] = N / (times[i*len+j] * 1.0e6);
        }
    }

    print_stat(grid_sizes,block_sizes,times,len,N,values, "GFLOPS");
    
}


void linear_compute_result(const int grid_sizes[], const int block_sizes[], 
    const float linear_times[], int len, int N) {
    compute_result(grid_sizes, block_sizes, linear_times, len, N, "Linear");
}

void consecutive_compute_result(const int grid_sizes[], const int block_sizes[], 
    const float consecutive_times[], int len, int N) {
    compute_result(grid_sizes, block_sizes, consecutive_times, len, N, "Consecutive");
}



int main(int argc, char **argv) {

    const int N = 4096 * 16;
    
    // For benchmarking
    const int len = 6;
    const int grid_sizes[] = {1,3,7,14,28,56};
    const int block_sizes[] = {32,64,128,256,512,1024};
    float linear_times[len*len];
    float consecutive_times[len*len];
    
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

    linear_access_bench(start,end,grid_sizes,block_sizes,linear_times,len,x,y,N);

    //reset values
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int pos = i * N + j;
            x[pos] = pos;
            x[pos] = pos; 
        }
    }

    consecutive_access_bench(start,end,grid_sizes,block_sizes,consecutive_times,len,x,y,N);
    
    //---------------COMPUTE RESULTS------------------

    linear_compute_result(grid_sizes,block_sizes,linear_times,len,N);
    printf("\n");
    consecutive_compute_result(grid_sizes,block_sizes,consecutive_times,len,N);
    printf("\n");

    //------------------------------------------------

    
    // Free memory
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    cudaFree(x);
    cudaFree(y);
}
