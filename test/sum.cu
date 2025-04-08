#include<iostream>
#include <math.h>
#include <time.h>
    // Kernel function to add the elements of two arrays
__global__ void add(int n, float *x, float *y) {
    for (int i = 0; i < n; i++)
        y[i] = x[i] + y[i];
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
    // Run kernel on 1M elements on the GPU
    add<<<1, 1>>>(N, x, y);
    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();
    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(y[i] - 3.0f));
    std::cout << "Max error: " << maxError << std::endl;
    // Free memory
    cudaFree(x);
    cudaFree(y);
}
