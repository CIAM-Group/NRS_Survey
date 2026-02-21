#include <pybind11/pybind11.h>


#include "interface.hpp"
#include "mis_interface.hpp"
#include "mcl_interface.hpp"


PYBIND11_MODULE(interface, m) {
    m.def("tsp_two_opt_inplace", &tsp_two_opt_inplace_interface, "");
    m.def("tsp_random_two_opt_inplace", &tsp_random_two_opt_inplace_interface, "");
    m.def("tsp_double_two_opt", &tsp_double_two_opt_interface, "");
    m.def("tsp_greedy_insert", &tsp_greedy_insert_interface, "");
    m.def("tsp_eval_cost", &tsp_eval_cost, "");

    py::class_<std::shared_ptr<std::vector<std::vector<std::vector<int>>>>>(m, "MISBatchedNeighborsInt32");
    m.def("mis_edges2neighbors", &mis_edges2neighbors_interface_int32, "");
    m.def("mis_edges2neighbors_nocast", &mis_edges2neighbors_interface_int32_nocast, py::return_value_policy::take_ownership);
    m.def("mis_partially_greedy_insert", &mis_partially_greedy_insert_interface_int32, "");

    m.def("mcl_partially_greedy_insert", &mcl_partially_greedy_insert_interface<int>, "");
}
