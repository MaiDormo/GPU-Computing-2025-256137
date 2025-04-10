#include<iostream>
#include <math.h>
#include <time.h>


// Kernel function to add the elements of two arrays
__global__ void add(int n, float *x, float *y) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        y[index] = x[index] + y[index];
    }
}

int main(int argc, char ** argv) {

    if (argc != 2) {
        printf("Usage: ./<sum.exec> <N>");
        exit(-1);
    }

    srand(time(NULL));
    int N = atoi(argv[1]);
    float *x, *y;
    // Allocate Unified Memory accessible from CPU or GPU
    cudaMallocManaged(&x, N * sizeof(float));
    cudaMallocManaged(&y, N * sizeof(float));
    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++)
    {
        x[i] = (rand()%100+1)*0.31;
        y[i] = (rand()%100+1)*0.73;
    }

    for (int i = 0; i < N; i++) {
        printf("%f ", x[i]);
    }
    printf("\n");

    for (int i = 0; i < N; i++) {
        printf("%f ", y[i]);
    }
    printf("\n");

    const int threads_per_block = 256;
    const int number_of_blocks = (N + threads_per_block - 1) / threads_per_block;

    // Run kernel on 1M elements on the GPU
    add<<<number_of_blocks, threads_per_block>>>(N, x, y);
    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();
    
    for (int i = 0; i < N; i++) {
        printf("%f ", y[i]);
    }
    printf("\n");


    // Free memory
    cudaFree(x);
    cudaFree(y);
}
