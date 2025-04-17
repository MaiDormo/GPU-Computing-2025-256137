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

__global__ void add_linear_access(const int n, dtype *x, dtype* y) {
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;
    const int stride = gridDim.x * blockDim.x;
    for (int j = idx ; j < n; j += stride) {
        y[j] = x[j] + y[j];
    }
}

__global__ void add_consecutive_access(const int n, dtype *x, dtype *y) {
    const int stride = n / (gridDim.x);
    const int len = n / (blockDim.x * gridDim.x);
    const int start = threadIdx.x * len + stride * blockIdx.x;
    
    for (int j = start; j < start + len; j++) {
        y[j] = x[j] + y[j];
    }
}

void linear_access_bench(cudaEvent_t start, cudaEvent_t end, const int grid_sizes[], 
    const int block_sizes[], float times[], int len, dtype* x, dtype* y, int N) {
    
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

void consecutive_access_bench(cudaEvent_t start, cudaEvent_t end, const int grid_sizes[], 
    const int block_sizes[], float times[], int len, dtype* x, dtype* y, int N) {
    
    float millis = 0.0;
    for (int i = 0; i < len; i++) {
        for (int j = 0; j < len; j++) {
            cudaEventRecord(start);
            add_consecutive_access<<<grid_sizes[j], block_sizes[i]>>>(N, x, y);
            cudaEventRecord(end);
            cudaEventSynchronize(end);
            cudaEventElapsedTime(&millis, start, end);
            times[j + i * len] = millis;
        }
    }
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

    //compute the bandwidth (TB/s)
    for (int i = 0; i < len; i++) {
        for (int j = 0; j < len; j++) {
            values[i*len+j] = total_bytes / times[i*len+j] * 1.0e-9; // GB/s
        }
    }

    print_stat(grid_sizes,block_sizes,times,len,N,values, "TB/s");

    //compute the TFLOPS
    for (int i = 0; i < len; i++) {
        for (int j = 0; j < len; j++) {
            values[i*len+j] = N / times[i*len+j] * 1.0e-9;
        }
    }

    print_stat(grid_sizes,block_sizes,times,len,N,values, "TFLOPS");
    
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

    const int N = 4096 * 1000;
    
    // For benchmarking
    const int len = 6;
    const int grid_sizes[] = {1,3,7,14,28,56};
    const int block_sizes[] = {32,64,128,256,512,1024};
    float linear_times[len*len];
    float consecutive_times[len*len];
    
    // Allocate Unified Memory accessible from CPU or GPU
    dtype *x, *y;
    cudaMallocManaged(&x, N * sizeof(dtype));
    cudaMallocManaged(&y, N * sizeof(dtype));
    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
        x[i] = i;
        y[i] = i;
    }

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    //---------------BENCHMARKING-------------------

    linear_access_bench(start,end,grid_sizes,block_sizes,linear_times,len,x,y,N);

    //reset values
    for (int i = 0; i < N; i++) {
        x[i] = i;
        y[i] = i;
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
