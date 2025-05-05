#ifndef READ_FILE_LIB_H
#define READ_FILE_LIB_H

#include "spmv_type.h"

#ifdef __cplusplus
extern "C" {
#endif

void read_from_file_and_init(char * file_path, struct COO *coo_data);
void _read_from_file_and_init(char * file_path, double ** a_val, int ** a_row, int ** a_col, int * mat_rows, int * mat_cols, int * vec_size);

#ifdef __cplusplus
}
#endif

#endif