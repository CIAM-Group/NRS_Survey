.DEFAULT_GOAL := all
PYBIND11_INCLUDES := $(shell python3 -m pybind11 --includes)
SUFFIX := $(shell python3-config --extension-suffix)

all:
	clang++ -O3 -finline-functions -Wall -shared -std=c++17 -fPIC \
	$(PYBIND11_INCLUDES) interface.cpp -o interface$(SUFFIX)

clean:
	rm *.so
