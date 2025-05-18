#include <stdio.h>
#include <stdlib.h>
#include "../include/read_file_lib.h"
#include "../include/spmv_type.h"



// --- File Reading and Initialization using Structs ---
void read_from_file_and_init(char * file_path, struct COO *coo_data) {
    FILE * file;
    const size_t BUFFER_SIZE = 1 * 1024 * 1024; // 1MB

    file = fopen(file_path, "r");
    if (!file) {
        perror("Error opening file!\n");
        exit(1);
    }

    char * buffer = (char*)malloc(BUFFER_SIZE);
    if (!buffer) {
        perror("Failed to allocate file buffer");
        fclose(file);
        exit(1);
    }
    setvbuf(file, buffer, _IOFBF, BUFFER_SIZE);

    char line_buffer[1024];
    while (fgets(line_buffer, sizeof(line_buffer), file)) {
        if (line_buffer[0] != '%') break;
    }

    int n, m, nnz;
    if (sscanf(line_buffer, "%d %d %d", &n, &m, &nnz) != 3) {
        perror("Error reading graph metadata");
        free(buffer); fclose(file); exit(-1);
    }
    coo_data->num_rows = n;
    coo_data->num_cols = m;
    coo_data->num_non_zeros = nnz;

    coo_data->a_val = (dtype*)malloc(nnz * sizeof(dtype));
    coo_data->a_row = (int*)malloc(nnz * sizeof(int));
    coo_data->a_col = (int*)malloc(nnz * sizeof(int));

    if (!coo_data->a_val || !coo_data->a_row || !coo_data->a_col) {
        perror("Failed to allocate memory for COO matrix data");
        free(buffer); fclose(file);
        free(coo_data->a_val); free(coo_data->a_row); free(coo_data->a_col);
        exit(1);
    }

    int r, c;
    dtype v;
    for (int i = 0; i < nnz; i++) {
        if (fscanf(file, "%d %d %lf", &r, &c, &v) != 3){
            fprintf(stderr, "Error reading entry %d\n", i);
            free(buffer); fclose(file);
            free(coo_data->a_val); free(coo_data->a_row); free(coo_data->a_col);
            exit(1);
        }
        coo_data->a_row[i] = r - 1;
        coo_data->a_col[i] = c - 1;
        coo_data->a_val[i] = v;
    }

    free(buffer);
    fclose(file);
}

//-----------------------------------------------------------------------------------------------------------------------------

void _read_from_file_and_init(char * file_path, double ** a_val, int ** a_row, int ** a_col, int * mat_rows, int * mat_cols, int * vec_size) {
    FILE * file;
    const size_t BUFFER_SIZE = 1 * 1024 * 1024; // 1MB

    // Open file for reading
    file = fopen(file_path, "r");
    // Check if file opened successfully
    if (!file) {
        perror("Error opening file!\n");
        exit(1);
    }

    // Set up a buffer for file I/O (16MB buffer)
    char * buffer = (char*)malloc(BUFFER_SIZE);
    if (!buffer) {
        perror("Failed to allocate file buffer");
        fclose(file);
        exit(1);
    }
    setvbuf(file, buffer, _IOFBF, BUFFER_SIZE);

    char line_buffer[1024];
    // Ignore lines that start with '%'
    while (fgets(line_buffer, sizeof(line_buffer), file)) {
        if (line_buffer[0] != '%') {
            break;  // Found a non-comment line
        }
    }

    // Read header
    int n, m, n_val;
    // We also take m, even though we do not use it, cause its a squared matrix
    //
    if (sscanf(line_buffer, "%d %d %d", &n, &m, &n_val) != 3) {
        perror("Error reading graph metadata");
        free(buffer);
        fclose(file);
        exit(-1);
    }
    *mat_rows = n;
    *mat_cols = m;
    *vec_size = n_val;

    double * val = (double*)malloc(n_val*sizeof(double));
    int * row = (int*)malloc(n_val*sizeof(int));
    int * col = (int*)malloc(n_val*sizeof(int));

    // Check if allocations succeeded
    if (!val || !row || !col) {
        perror("Failed to allocate memory for matrix data");
        free(buffer);
        free(val);   // These are safe even if NULL
        free(row);
        free(col);
        fclose(file);
        exit(1);
    }

    int r, c;
    double v;
    for (int i = 0; i < n_val; i++) {
        if (fscanf(file, "%d %d %lf", &r, &c, &v) != 3){
            fprintf(stderr, "Error reading entry %d\n", i);
            free(buffer);
            free(row);
            free(col);
            free(val);
            fclose(file);
            exit(1);
        }

        row[i] = r-1;
        col[i] = c-1;
        val[i] = v;
    }

    // Passing pointers
    *a_row = row;
    *a_col = col;
    *a_val = val;

    free(buffer);
    fclose(file);
}