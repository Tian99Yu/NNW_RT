SRC := cublas_test.cu
CUDNN_PATH := cudnn
INCLUDES := `python -m pybind11 --includes` -I $(CUDNN_PATH)/include
NVCC := /usr/local/cuda-11/bin/nvcc
CUDA_TOOLKIT := /usr/local/cuda-11/
NVCC_INC := -I/usr/local/cuda-11/include
LIBS         := -lcudart -lcusparse
all:
	# g++ -shared -std=c++11 -fPIC `python -m pybind11 --includes` $(SRC) -o test`python3-config --extension-suffix`
	nvcc -std=c++11 -O2 $(SRC) -shared -Xcompiler -fPIC $(INCLUDES) -o runtime_compare`python3-config --extension-suffix` -lcublas
cusparse:
	${NVCC} ${NVCC_INC} cuSparse.c -o cuSparse_test.o ${LIBS}
	./cuSparse_test.o
clean:
	rm *.so
	rm *.o
