# .DEFAULT_GOAL := all
# PYBIND11_INCLUDES := $(shell python3 -m pybind11 --includes)
# SUFFIX := $(shell python3-config --extension-suffix)

# all:
# 	clang++ -O3 -finline-functions -Wall -shared -std=c++17 -fPIC \
# 	$(PYBIND11_INCLUDES) interface.cpp -o interface$(SUFFIX)

# clean:
# 	rm *.so

# ! 当前环境没有 clang++，改用 g++

.DEFAULT_GOAL := all

# 编译器：默认用 g++，以后想用 clang++ 再覆盖就行
CXX ?= g++

PYBIND11_INCLUDES := $(shell python3 -m pybind11 --includes)
SUFFIX := $(shell python3-config --extension-suffix)

all:
	$(CXX) -O3 -finline-functions -Wall -shared -std=c++17 -fPIC \
		$(PYBIND11_INCLUDES) interface.cpp -o interface$(SUFFIX)

clean:
	rm -f *.so
