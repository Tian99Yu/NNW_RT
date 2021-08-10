from runtime_compare import *
import numpy as np
import matplotlib.pyplot as plt
time = []
dim = []
for x in np.linspace(10, 1 << 13, num=10, dtype=np.int32):
	
	time.append(test_cublas_sgemm(x, x, x, 1))
	dim.append(x)

plt.plot(dim, time)
plt.show()

