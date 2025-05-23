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
    int elem = 0;
    int local_sum = 0;
    int recv_sum = 0;    
    
    //-------------------
    for (int i = 0; i < 5; i++)
        local_sum += vec[i];

    MPI_Allreduce(&local_sum, &recv_sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    printf("Received sum: %d\n", recv_sum);
    //-------------------------
    
    
    //-----------------
    int rec_vec[5];
    MPI_Allreduce(&vec, &rec_vec, 5, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    local_sum = 0;
    for (int i = 0; i < 5; i++)
        local_sum += rec_vec[i];
    printf("Computed sum from vector: %d\n", local_sum);
    //--------------------
    
    MPI_Finalize();
    return 0;

}