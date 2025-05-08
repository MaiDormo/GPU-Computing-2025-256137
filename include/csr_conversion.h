#ifndef CSR_CONVERSION_H
#define CSR_CONVERSION_H

#include "spmv_type.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Converts a matrix from COO format to CSR format
 *
 * @param coo_data Input matrix in COO format
 * @param csr_data Output matrix in CSR format (pre-allocated)
 * @return 0 on success, -1 on error
 */
int coo_to_csr(const struct COO *coo_data, struct CSR *csr_data);

#ifdef __cplusplus
}
#endif

#endif // CSR_CONVERSION_H