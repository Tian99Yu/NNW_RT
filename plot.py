import numpy as np
from scipy import sparse
from cpp_lib import *



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
	return sparse.random(m, k, 1 - sparsity, format = 'csr').A.reshape(-1), np.random.randn(k, n).reshape(-1)

def experiment():
	#first experiment: square mtx matmul
	dense_time, sparse_time = [], []
	for x in np.linspace(10, 1 << 13, num=10, dtype=np.int32):
		A, B = gen_test_input(x,x,x,0.9)
		time.append(cuBLAS(x,x,x,))

