SRC := cublas_test.cu
CUDNN_PATH := cudnn
INCLUDES := `python -m pybind11 --includes` -I $(CUDNN_PATH)/include

all:
	# g++ -shared -std=c++11 -fPIC `python -m pybind11 --includes` $(SRC) -o test`python3-config --extension-suffix`
	nvcc -std=c++11 -O2 $(SRC) -shared -Xcompiler -fPIC $(INCLUDES) -o runtime_compare`python3-config --extension-suffix` -lcublas

clean:
	rm *.so
