#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <chrono>
#include <string>
#include <cuda.h>
#include <cublas_v2.h>
namespace py = pybind11;
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

#define DEBUG


float test_cublas_sgemm(int m, int n, int k, py::array_t<float> arr_A, py::array_t<float> arr_B) {
// float test_cublas_sgemm(int m, int n, int k, float * arr_A, float * arr_B) {
	//remember the mtx is col based!!!
	//init the variables	
	typedef std::chrono::high_resolution_clock Clock;
	typedef std::chrono::milliseconds milliseconds;
	float *A, *B;
	float *d_A, *d_B, *d_C;
#ifdef DEBUG
	//define the output variable C
	float *C;
	C =(float *) malloc(sizeof(float) * m * n);
#endif


 	// get the elements inside the numpy passed in array
	py::buffer_info buf_A = arr_A.request();
	py::buffer_info buf_B = arr_B.request();
	A = (float *) buf_A.ptr;
	B = (float *) buf_B.ptr;
	
	// A = arr_A;
	// B = arr_B;

	//cuda code
	cublasHandle_t handle;
	if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS){
		fprintf(stderr, "fail handle");
	}

	CHECK_CUDA(cudaMalloc((void **)&d_A, sizeof(float) * m * k))
	CHECK_CUDA(cudaMalloc((void **)&d_B, sizeof(float) * n * k))
	CHECK_CUDA(cudaMalloc((void **)&d_C, sizeof(float) * m * n))

	CHECK_CUDA(cudaMemcpy(d_A, A, sizeof(float) * m * k, cudaMemcpyHostToDevice))
	CHECK_CUDA(cudaMemcpy(d_B, B, sizeof(float) * n * k, cudaMemcpyHostToDevice))



	const float a = 1.0,  b = 0.0;
	Clock::time_point start = Clock::now();
	
	cublasStatus_t ret = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &a, d_A, m, d_B, k, &b, d_C, m);
	cudaDeviceSynchronize();
	Clock::time_point end = Clock::now();

#ifdef DEBUG
	//copy the result back to host memory ofr latter printing
	CHECK_CUDA(cudaMemcpy(C, d_C, sizeof(float) * m * n, cudaMemcpyDeviceToHost))
#endif



	CHECK_CUDA(cudaFree(d_A))
	CHECK_CUDA(cudaFree(d_B))
	CHECK_CUDA(cudaFree(d_C))	

	cublasDestroy(handle);

	milliseconds ms = std::chrono::duration_cast<milliseconds> (end-start);

#ifdef DEBUG
	fprintf(stderr, "printing the multiplication result col by col, matrix is %d X %d\n\n", m, n);
	for (int i=0; i<m*n; i++){
		fprintf(stderr, "%f \n", C[i]);
	}
#endif
	return ms.count();
}


//Pybind call
PYBIND11_MODULE(cpp_lib, m){
	m.def("cuBLAS", &test_cublas_sgemm, "the function returning the RT of cuBLAS");
}

// int main(){
// 	float m = 2.0;
// 	float n = 2.0;
// 	float k = 2.0;
// 	float arr_A[] = {1,2,3,4};
// 	float arr_B[] = {1,0,0,1};
// 	test_cublas_sgemm(m,n,k,arr_A, arr_B);	
// }