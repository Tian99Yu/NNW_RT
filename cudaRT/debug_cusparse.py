import math 
import numpy as np
from scipy import sparse
from cpp_lib import *
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import time
from scipy.stats.mstats import gmean
from tqdm import tqdm
from plot import *

A, B = gen_test_input(10, 8, 1, 0.9)
sparse_A = A
m, k = A.A.shape
n = B.shape[1]
print(A.A)
param = [m,n,k, sparse_A.nnz, sparse_A.indptr, sparse_A.indices, sparse_A.data, B.flatten()]
acc = ""
for i in param:
	acc += "{} \n".format(i)
print(acc)
cuSPARSE(m,n,k, sparse_A.nnz, sparse_A.indptr, sparse_A.indices, sparse_A.data, B.flatten())