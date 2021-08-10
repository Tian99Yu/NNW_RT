#include <pybind11/pybind11.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <chrono>
#include <string>
#include <cuda.h>
#include <cublas_v2.h>


void generate_random_sparsity(float *A, int w, int h, float sparsity){
	fprintf(stderr, "generate_random_sparsity(): w = %d, h = %d\n", w, h);
	int s = w*h;
	// int x = (rand() | rand() << 16 )% (w * h);
	int x = rand() % s;
	int nz = ceil((1-sparsity) * w * h);

	cudaMemset(A, 0, w*h*sizeof(float));

	for (int i = 0; i < nz; i ++) {
		while (A[x] != 0)
			x = rand() % s;
		A[x] = (1.0 * rand()) / RAND_MAX;
	}

}


float test_cusparse_gemm(int m, int n, int k, float sparsity) {
	typedef std::chrono::high_resolution_clock Clock;
	typedef std::chrono::milliseconds milliseconds;
	float *A, *B, *C;

	cublasHandle_t handle;
	if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS){
		fprintf(stderr, "fail handle");
	}

	cudaMallocManaged(&A, sizeof(float) * m * k);
	cudaMallocManaged(&B, sizeof(float) * k * n);
	cudaMallocManaged(&C, sizeof(float) * m * n);

	generate_random_sparsity(A, m, k, sparsity);
	generate_random_sparsity(B, k, n, sparsity);
	// memset(C, 0, m*n*sizeof(float));
	cudaMemset(C, 0, m*n*sizeof(float));

	const float a = 1.0,  b = 0.0;
	Clock::time_point start = Clock::now();
	
	cublasStatus_t ret = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &a, A, m, B, k, &b, C, m);
	cudaDeviceSynchronize();

	Clock::time_point end = Clock::now();
	cudaFree(A);
	cudaFree(B);
	cudaFree(C);
	cublasDestroy(handle);

	milliseconds ms = std::chrono::duration_cast<milliseconds> (end-start);

	return ms.count();
}

float test_cublas_sgemm(int m, int n, int k, float sparsity) {
	typedef std::chrono::high_resolution_clock Clock;
	typedef std::chrono::milliseconds milliseconds;
	float *A, *B, *C;

	cublasHandle_t handle;
	if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS){
		fprintf(stderr, "fail handle");
	}

	cudaMallocManaged(&A, sizeof(float) * m * k);
	cudaMallocManaged(&B, sizeof(float) * k * n);
	cudaMallocManaged(&C, sizeof(float) * m * n);

	generate_random_sparsity(A, m, k, sparsity);
	generate_random_sparsity(B, k, n, sparsity);
	// memset(C, 0, m*n*sizeof(float));
	cudaMemset(C, 0, m*n*sizeof(float));

	const float a = 1.0,  b = 0.0;
	Clock::time_point start = Clock::now();
	
	cublasStatus_t ret = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &a, A, m, B, k, &b, C, m);
	cudaDeviceSynchronize();

	Clock::time_point end = Clock::now();
	cudaFree(A);
	cudaFree(B);
	cudaFree(C);
	cublasDestroy(handle);

	milliseconds ms = std::chrono::duration_cast<milliseconds> (end-start);

	return ms.count();
}


PYBIND11_MODULE(runtime_compare, m){
	m.def("test_cublas_sgemm", &test_cublas_sgemm, "get the time of cublas operation given the matrix size and sparsity");
}
