SHIT := pybind_array.cpp
SRC := cublas_test.cu
CUDNN_PATH := cudnn
INCLUDES := `python -m pybind11 --includes` -I $(CUDNN_PATH)/include
NVCC := /usr/local/cuda-11/bin/nvcc
CUDA_TOOLKIT := /usr/local/cuda-11/
NVCC_INC := -I/usr/local/cuda-11/include
LIBS         := -lcudart -lcusparse -lcublas
CXX_FLAGS := -std=c++11 -g -O2
LIBRARY_DIR := cpp_lib.cu

all:
	# g++ -shared -std=c++11 -fPIC `python -m pybind11 --includes` $(SRC) -o test`python3-config --extension-suffix`
	nvcc $(CXX_FLAGS) $(SRC) -shared -Xcompiler -fPIC $(INCLUDES) -o runtime_compare`python3-config --extension-suffix` -lcublas


cusparse:
	${NVCC} ${NVCC_INC} $(CXX_FLAGS) cuSparse.c -o cuSparse_test.o ${LIBS}
	./cuSparse_test.o
clean:
	rm *.so
	rm *.o
cpp_lib:
	${NVCC} ${NVCC_INC} $(CXX_FLAGS) $(LIBRARY_DIR) $(LIBS) -o cpp_lib.o
	./cpp_lib.o
library:
	${NVCC} ${NVCC_INC} $(CXX_FLAGS) $(LIBRARY_DIR) -shared -Xcompiler -fPIC $(INCLUDES) -o cpp_lib`python3-config --extension-suffix` $(LIBS)
