#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

void shit(py::array_t<float> arr){
	py::buffer_info buf1 = arr.request();
	float *ptr = (float *) buf1.ptr;
	for (int i=0; i<3; i++){
		fprintf(stderr, "array element %d: %f\n", i, ptr[i]);
	}
}

PYBIND11_MODULE(shit, m){
	m.def("shit", &shit, "get the time of cublas operation given the matrix size and sparsity");
}