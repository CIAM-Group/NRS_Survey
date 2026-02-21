#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "mcl.hpp"


namespace py = pybind11;


template<typename NodeIndexDType>
inline void
mcl_partially_greedy_insert_interface(
    py::array_t<bool> & solution,
    py::array_t<int> & num_clique_neighbors,
    py::array_t<int> & num_inserted_nodes,
    const py::array_t<NodeIndexDType> & candidate_nodes,
    const std::shared_ptr<std::vector<std::vector<std::vector<NodeIndexDType>>>> neighbors_ptr,
    const py::array_t<int> & num_nodes,
    const py::array_t<int> & target_num_nodes,
    const int num_workers
) {
    assert(solution.flags() & py::array::c_style);
    assert(num_clique_neighbors.flags() & py::array::c_style);
    assert(num_inserted_nodes.flags() & py::array::c_style);
    assert(candidate_nodes.flags() & py::array::c_style);
    assert(num_nodes.flags() & py::array::c_style);
    assert(target_num_nodes.flags() & py::array::c_style);

    const int batch_size = solution.shape()[0];
    const int solution_stride_batch = solution.shape()[1];
    const int num_candidate_nodes = candidate_nodes.shape()[1];

    auto solution_ptr = static_cast<bool *>(solution.request().ptr);
    auto num_clique_neighbors_ptr = static_cast<int *>(num_clique_neighbors.request().ptr);
    auto num_inserted_nodes_ptr = static_cast<int *>(num_inserted_nodes.request().ptr);
  
    auto candidate_nodes_ptr = candidate_nodes.data();
    auto num_nodes_ptr = num_nodes.data();
    auto target_num_nodes_ptr = target_num_nodes.data();

    auto & neighbors = *neighbors_ptr;

    auto task_fn = [&](const int task_id) {
        mcl::partially_greedy_insert(
            solution_ptr + task_id * solution_stride_batch, 
            num_clique_neighbors_ptr + task_id * solution_stride_batch, 
            num_inserted_nodes_ptr[task_id], 
            candidate_nodes_ptr + task_id * num_candidate_nodes, 
            neighbors[task_id], 
            num_nodes_ptr[task_id], 
            num_candidate_nodes,
            target_num_nodes_ptr[task_id]
        );
    };

    parallelize(task_fn, batch_size, std::min(batch_size, num_workers));
}
