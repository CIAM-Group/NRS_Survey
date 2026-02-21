#pragma once

#include "pybind11/cast.h"
#include "pybind11/gil.h"
#include "pybind11/pytypes.h"
#include "par_utils.hpp"
#include "tsp.hpp"
#include "tsp_interface.hpp"
#include "mis_interface.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <sys/types.h>

namespace py = pybind11; 


