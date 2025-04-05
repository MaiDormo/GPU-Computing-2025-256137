#include <stdio.h>
#include <stdlib.h>


__global__ void print_from_gpu(void) {
    printf("Hello World! from thread [%d,%d] From device\n", blockIdx.x, threadIdx.x);
}


int main(void) {
    printf("Hello World from host!\n");
    print_from_gpu<<<14,32>>>();
    cudaDeviceSynchronize();
    return 0;
}