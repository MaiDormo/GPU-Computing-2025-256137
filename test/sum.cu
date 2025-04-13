#include<iostream>
#include <math.h>
#include <time.h>

#define dtype float


// Kernel function to add the elements of two arrays
__global__ void add(int n, float *x, float *y) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        y[index] = x[index] + y[index];
    }
}

__global__ void add_elem_wise(int n, dtype *x, dtype* y) {

    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    for (int j = idx; j < n; j += gridDim.x*blockDim.x) {
        y[j] = x[j] + y[j];
    }
}

int main(int argc, char ** argv) {

    srand(time(NULL));
    int N = 4096;
    dtype *x, *y;
    // Allocate Unified Memory accessible from CPU or GPU
    cudaMallocManaged(&x, N * sizeof(dtype));
    cudaMallocManaged(&y, N * sizeof(dtype));
    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++)
    {
        x[i] = i;
        y[i] = i;
    }

    for (int i = 0; i < 20; i++) {
        printf("%f ", x[i]);
    }
    printf("\n");

    for (int i = 0; i < 20; i++) {
        printf("%f ", y[i]);
    }
    printf("\n");

    const int threads_per_block = 256;
    const int number_of_blocks = (N + threads_per_block - 1) / (threads_per_block * 2);

    // Run kernel on 1M elements on the GPU
    // add<<<number_of_blocks, threads_per_block>>>(N, x, y);

    add_elem_wise<<<number_of_blocks,threads_per_block>>>(N,x,y);
    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();
    
    for (int i = 0; i < 20; i++) {
        printf("%f ", y[i]);
    }
    printf("\n");


    // Free memory
    cudaFree(x);
    cudaFree(y);
}
