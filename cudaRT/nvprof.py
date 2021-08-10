import numpy as np
from scipy import sparse
from cpp_lib import *
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import time
from scipy.stats.mstats import gmean
#import the functions from plot.py (in the same folder)
from plot import *

if __name__ == "__main__":
	x, sparsity = 1<<11, 0.9
	A, B = gen_test_input(x,x,x,sparsity) 
	# cublas_runtime(A,B)
	# cusparse_runtime(A, B)
	tensorflow_runtime(A, B)
