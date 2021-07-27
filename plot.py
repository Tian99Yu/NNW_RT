import numpy as np
from scipy import sparse
from cpp_lib import *
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import time
from scipy.stats.mstats import gmean


def tensorflow_runtime(A, B, reps=10):
	"""
	Given the sparse matrix A and dense matrix B return the runtime of the 
	tf.sparse.sparse_dense_matmul function
	"""
	t_s_A = tf.sparse.from_dense(A)
	times = []
	for _ in range(reps):
		start = time.perf_counter()
		tf.sparse.sparse_dense_matmul(t_s_A, B)
		end = time.perf_counter()
		times.append(end - start)
	return 1000.0 * gmean(times)

# def cublas_runtime(A, B, reps=10):
# 	"""
# 	Given the sparse matrix A and dense matrix B return the runtime of the 
# 	cublas code (cpp code binded in python)
# 	"""
# 	dense_time = []
# 	sparse_A, A, B = gen_test_input(x,x,x,0.9)
# 	for _ in range(reps):
# 		dense_time.append(cuBLAS(x,x,x,A,B))



def get_sparse_info(mtx, m, n):
	"""
	given a dense np array mtx (1d array with m*n as length)
	where m, n is the row and col
	Return the csr format data as a tuple: (value, col index, row ptr)  
	"""

	sparse_mtx = sparse.csr_matrix(mtx.reshape(m, n))
	return sparse_mtx.data, sparse_mtx.indices, sparse_mtx.indptr, sparse_mtx.nnz


def gen_test_input(m,n,k,sparsity):
	"""
	return A as a sparse matrix and B as a dense
	A is of shape (m, k)
	B is of shape (k, n)
	"""
	sparse_A = sparse.random(m, k, 1 - sparsity, format = 'csr', dtype=np.float32)
	return sparse_A,sparse_A.A.reshape(-1), np.random.randn(k, n).reshape(-1).astype(np.float32)

def experiment():
	#first experiment: square mtx matmul
	dense_time, sparse_time = [], []
	for x in np.linspace(10, 1 << 13, num=10, dtype=np.int32):
		sparse_A, A, B = gen_test_input(x,x,x,0.9)
		dense_time.append(cuBLAS(x,x,x,A,B))
		sparse_time.append(cuSPARSE(x,x,x, sparse_A.nnz, sparse_A.indptr, sparse_A.indices, sparse_A.data, B))
	df = pd.DataFrame({"cuBLAS": dense_time, "cuSPARSE": sparse_time}, index=list(np.linspace(10, 1 << 13, num=10, dtype=np.int32)))
	df.plot.line()	
	plt.title("cuSparse -- cuBlas compare")
	plt.xlabel("squared matrix dimension")
	plt.ylabel("run time (ms)")
	plt.savefig("blas_sparse.pdf")	


if __name__ == "__main__":
	experiment()