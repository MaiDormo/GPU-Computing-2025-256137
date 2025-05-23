#include <mpi.h>
#include <stdio.h>
#include <time.h>


int main(int argc, char** argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);
    
    int rank, size;
    // Get process ID
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // Get total number of processes;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int vec[5] = {1,2,3,4,5};

    int tag = 0;
    char message[100];

    MPI_Status status;
    int step = 0;
    clock_t start, end;
    int elem = 0;
    int partial_sum = 0;    
    while (step < 5) {
        if (rank == 0) {
            start = clock();
            MPI_Send(&vec[step], 1, MPI_INT, 1, tag, MPI_COMM_WORLD);
            MPI_Recv(&elem, 1, MPI_INT, 1, tag, MPI_COMM_WORLD, &status);
            end = clock();
            printf("Time passed: %lf ms\n", (double)(end-start)/CLOCKS_PER_SEC * 1000);
            partial_sum += vec[step] + elem;
            step++;
        } else {
            MPI_Recv(&elem, 1, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);
            MPI_Send(&vec[step], 1, MPI_INT, 0, tag, MPI_COMM_WORLD);
            partial_sum += vec[step] + elem;
            step++;
        }
    }

    printf("Summ of all elements %d\n", partial_sum);

    MPI_Finalize();
    return 0;

}