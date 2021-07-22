//#include <pybind11/pybind11.h>

//#include <cassert>
//#include <chrono>
//#include <cmath>
//#include <cstdio>
//#include <cublas_v2.h>
//#include <cuda.h>
//#include <string>
/**
 * A is 2*2 matrix 3,3,3,3
 * B is 2*2 identity matrix
 */

#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>         // cusparseSparseToDense
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE
//Souorce: copied from https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuSPARSE/dense2sparse_csr/dense2sparse_csr_example.c
#define CHECK_CUDA(func)                                               \
    {                                                                  \
        cudaError_t status = (func);                                   \
        if (status != cudaSuccess)                                     \
        {                                                              \
            printf("CUDA API failed at line %d with error: %s (%d)\n", \
                   __LINE__, cudaGetErrorString(status), status);      \
            return EXIT_FAILURE;                                       \
        }                                                              \
    }

#define CHECK_CUSPARSE(func)                                               \
    {                                                                      \
        cusparseStatus_t status = (func);                                  \
        if (status != CUSPARSE_STATUS_SUCCESS)                             \
        {                                                                  \
            printf("CUSPARSE API failed at line %d with error: %s (%d)\n", \
                   __LINE__, cusparseGetErrorString(status), status);      \
            return EXIT_FAILURE;                                           \
        }                                                                  \
    }

/**
 * @brief convert the given dense mtx into the csr format using cusparse lib
 * 
 * @param h_dense the dense matrix represented in array of float
 * @param num_rows the number of rows of that dense matrix
 * @param num_cols  the numebr of cols of that dense matrix
 * @param h_csr_offsets output, the offset of csr mtx
 * @param h_csr_columns output, the column number of csr mtx
 * @param h_csr_values output, the float value of the csr mtx
 */

int dense_2_csr(float *h_dense, int num_rows, int num_cols, int **h_csr_offsets,
                 int **h_csr_columns, float **h_csr_values, int *nz)
{
    //leading dimension (the offset)
    int ld = num_cols;
    int dense_size = num_rows * num_cols;
    
    //device memory
    int   *d_csr_offsets, *d_csr_columns;
    float *d_csr_values,  *d_dense;
    CHECK_CUDA( cudaMalloc((void**) &d_dense, dense_size * sizeof(float)))
    CHECK_CUDA( cudaMalloc((void**) &d_csr_offsets,
                           (num_rows + 1) * sizeof(int)) )
    CHECK_CUDA( cudaMemcpy(d_dense, h_dense, dense_size * sizeof(float),
                           cudaMemcpyHostToDevice) )

    //cusparse
    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matB;
    cusparseDnMatDescr_t matA;
    void*                dBuffer    = NULL;
    size_t               bufferSize = 0;
    CHECK_CUSPARSE( cusparseCreate(&handle) )
    // Create dense matrix A
    CHECK_CUSPARSE( cusparseCreateDnMat(&matA, num_rows, num_cols, ld, d_dense,
                                        CUDA_R_32F, CUSPARSE_ORDER_ROW) )
    // Create sparse matrix B in CSR format
    CHECK_CUSPARSE( cusparseCreateCsr(&matB, num_rows, num_cols, 0,
                                      d_csr_offsets, NULL, NULL,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )
    // allocate an external buffer if needed
    CHECK_CUSPARSE( cusparseDenseToSparse_bufferSize(
                                        handle, matA, matB,
                                        CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                        &bufferSize) )
    CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )

    // execute dense to sparse conversion
    CHECK_CUSPARSE( cusparseDenseToSparse_analysis(handle, matA, matB,
                                        CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                        dBuffer) )
    // get number of non-zero elements
    int64_t num_rows_tmp, num_cols_tmp, nnz;
    CHECK_CUSPARSE( cusparseSpMatGetSize(matB, &num_rows_tmp, &num_cols_tmp,
                                         &nnz) )

    // allocate CSR column indices and values
    CHECK_CUDA( cudaMalloc((void**) &d_csr_columns, nnz * sizeof(int))   )
    CHECK_CUDA( cudaMalloc((void**) &d_csr_values,  nnz * sizeof(float)) )


    // reset offsets, column indices, and values pointers
    CHECK_CUSPARSE( cusparseCsrSetPointers(matB, d_csr_offsets, d_csr_columns,
                                           d_csr_values) )
    // execute Sparse to Dense conversion
    CHECK_CUSPARSE( cusparseDenseToSparse_convert(handle, matA, matB,
                                        CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                        dBuffer) )

    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseDestroyDnMat(matA) )
    CHECK_CUSPARSE( cusparseDestroySpMat(matB) )
    CHECK_CUSPARSE( cusparseDestroy(handle) )

    *h_csr_offsets = malloc((num_rows + 1) * sizeof(int));
    *h_csr_columns = malloc(nnz * sizeof(int));
    *h_csr_values = malloc(nnz * sizeof(float));

    CHECK_CUDA( cudaMemcpy(*h_csr_offsets, d_csr_offsets,
                           (num_rows + 1) * sizeof(int),
                           cudaMemcpyDeviceToHost) )
    CHECK_CUDA( cudaMemcpy(*h_csr_columns, d_csr_columns, nnz * sizeof(int),
                           cudaMemcpyDeviceToHost) )
    CHECK_CUDA( cudaMemcpy(*h_csr_values, d_csr_values, nnz * sizeof(float),
                           cudaMemcpyDeviceToHost) )

    *nz = (int)nnz;

    // device memory deallocation
    CHECK_CUDA( cudaFree(dBuffer) )
    CHECK_CUDA( cudaFree(d_csr_offsets) )
    CHECK_CUDA( cudaFree(d_csr_columns) )
    CHECK_CUDA( cudaFree(d_csr_values) )
    CHECK_CUDA( cudaFree(d_dense) )
    return EXIT_SUCCESS;
}

// void csr_dense_mm()

int main(){
    float A[] = {1.0f, 0.0f, 1.0f, 0.0f};
    int num_rows = 2;
    int num_cols = 2;
    int nnz;
    int * csr_offsets;
    int * csr_columns;
    float * csr_values;

    dense_2_csr(A, num_rows, num_cols, &csr_offsets, &csr_columns, &csr_values, &nnz);
    for(int i=0; i<2; i++){
        fprintf(stderr, "col: %d | val: %f \n",csr_columns[i], csr_values[i]);
    }
    fprintf(stderr, "the offsets\n");
    for(int i=0; i<3; i++){
        fprintf(stderr, "%d\n", csr_offsets[i]);
    }
}