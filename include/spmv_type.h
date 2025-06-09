#ifndef CSR_STRUCT_H
#define CSR_STRUCT_H

#define dtype float

// --- Struct Definitions ---
// Structure for CSR format
struct CSR {
    dtype * values;
    int * col_indices;
    int * row_pointers;
    int num_rows;
    int num_cols;
    int num_non_zeros;
};

// Structure for COO format
struct COO {
    dtype * a_val;
    int * a_col;
    int * a_row;
    int num_rows;
    int num_cols;
    int num_non_zeros;
};


struct MAT_STATS {
    double mean_nnz_per_row;
    double std_dev_nnz_per_row;
    int min_nnz_per_row;
    int max_nnz_per_row;
    int median_nnz_per_row;
    double sparsity_ratio;
    int total_nnz;
};

#endif