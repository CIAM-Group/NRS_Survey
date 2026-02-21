#pragma once


#include "pybind11/cast.h"
#include "pybind11/gil.h"
#include "pybind11/pytypes.h"
#include "pybind11/stl.h"
#include "par_utils.hpp"
#include "mis.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <random>
#include <sys/types.h>
#include <vector>

namespace py = pybind11;

PYBIND11_MAKE_OPAQUE(std::shared_ptr<std::vector<std::vector<std::vector<int>>>>);
PYBIND11_MAKE_OPAQUE(std::shared_ptr<std::vector<std::vector<std::vector<int16_t>>>>);


template<typename NodeIndexDType>
inline auto
mis_partially_greedy_insert_interface(
    py::array_t<bool> & solution,
    py::array_t<bool> & mask,
    py::array_t<int> & num_inserted_nodes,
    py::array_t<int> & num_masked_nodes,
    const py::array_t<NodeIndexDType> & candidate_nodes,
    const std::shared_ptr<std::vector<std::vector<std::vector<NodeIndexDType>>>> neighbors_ptr,
    const py::array_t<int> & num_nodes,
    const py::array_t<int> & target_num_nodes,
    const int num_workers
) {
    assert(solution.flags() & py::array::c_style);
    assert(mask.flags() & py::array::c_style);
    assert(num_inserted_nodes.flags() & py::array::c_style);
    assert(num_masked_nodes.flags() & py::array::c_style);
    assert(candidate_nodes.flags() & py::array::c_style);
    assert(num_nodes.flags() & py::array::c_style);
    assert(target_num_nodes.flags() & py::array::c_style);

    assert(solution.ndim() == 2);

    const int batch_size = solution.shape()[0];
    const int solution_stride_batch = solution.shape()[1];
    const int num_candidate_nodes = candidate_nodes.shape()[1];

    auto solution_ptr = static_cast<bool *>(solution.request().ptr);
    auto mask_ptr = static_cast<bool *>(mask.request().ptr);
    auto num_inserted_nodes_ptr = static_cast<int *>(num_inserted_nodes.request().ptr);
    auto num_masked_nodes_ptr = static_cast<int *>(num_masked_nodes.request().ptr);
    auto candidate_nodes_ptr = candidate_nodes.data();
    auto num_nodes_ptr = num_nodes.data();
    auto target_num_nodes_ptr = target_num_nodes.data();

    auto & neighbors = *neighbors_ptr;

    py::gil_scoped_release release;

    auto task_fn = [&](const int task_id) {
        mis::partially_greedy_insert(
            solution_ptr + task_id * solution_stride_batch, 
            mask_ptr + task_id * solution_stride_batch, 
            num_inserted_nodes_ptr[task_id], 
            num_masked_nodes_ptr[task_id], 
            candidate_nodes_ptr + task_id * num_candidate_nodes, 
            neighbors[task_id], 
            num_nodes_ptr[task_id], 
            num_candidate_nodes,
            target_num_nodes_ptr[task_id]
        );
    };

    parallelize(task_fn, batch_size, std::min(batch_size, num_workers));
}


inline auto
mis_partially_greedy_insert_interface_int32(
    py::array_t<bool> & solution,
    py::array_t<bool> & mask,
    py::array_t<int> & num_inserted_nodes,
    py::array_t<int> & num_masked_nodes,
    const py::array_t<int> & candidate_nodes,
    const std::shared_ptr<std::vector<std::vector<std::vector<int>>>> neighbors_ptr,
    const py::array_t<int> & num_nodes,
    const py::array_t<int> & target_num_nodes,
    const int num_workers
) {
    mis_partially_greedy_insert_interface(
        solution, mask, num_inserted_nodes, 
        num_masked_nodes, candidate_nodes, neighbors_ptr,
        num_nodes, target_num_nodes,
        num_workers
    );
}


template<typename NodeIndexDType>
inline auto
mis_edges2neighbors_interface(
    const py::array_t<NodeIndexDType> & edges,
    const py::array_t<int> & num_nodes,
    const py::array_t<int> & num_edges,
    const int num_workers
) {
    assert(edges.flags() & py::array::c_style);
    assert(num_nodes.flags() & py::array::c_style);
    assert(num_edges.flags() & py::array::c_style);

    assert(edges.ndim() == 3);
    assert(num_nodes.ndim() == 1);
    assert(num_edges.ndim() == 1);

    auto edges_ptr = edges.data();
    auto num_nodes_ptr = num_nodes.data();
    auto num_edges_ptr = num_edges.data();

    const int edges_stride_batch = edges.shape()[1] * 2;

    const int batch_size = edges.shape()[0];
    
    py::gil_scoped_release release;

    auto batched_neighbors = std::vector<std::vector<std::vector<NodeIndexDType>>>(batch_size);

    auto task_fn = [&](const int task_id) {
        batched_neighbors[task_id] = mis::edges2neighbors(
            edges_ptr + task_id * edges_stride_batch,
            num_nodes_ptr[task_id],
            num_edges_ptr[task_id]
        );
    };

    parallelize(task_fn, batch_size, std::min(batch_size, num_workers));

    py::gil_scoped_acquire acquire;

    return batched_neighbors;
}


inline auto
mis_edges2neighbors_interface_int32(
    const py::array_t<int> & edges,
    const py::array_t<int> & num_nodes,
    const py::array_t<int> & num_edges,
    const int num_workers
) {
    return mis_edges2neighbors_interface(edges, num_nodes, num_edges, num_workers);
}


inline auto
mis_edges2neighbors_interface_int32_nocast(
    const py::array_t<int> & edges,
    const py::array_t<int> & num_nodes,
    const py::array_t<int> & num_edges,
    const int num_workers
) {
    auto result = mis_edges2neighbors_interface(edges, num_nodes, num_edges, num_workers);
    return std::make_shared<decltype(result)>(result);
}



inline auto
mis_partially_mask_interface(
    py::array_t<bool> & solution,
    const py::array_t<int> & seeds,
    const py::array_t<int> & num_nodes,
    const py::array_t<int> & num_nodes_to_keep,
    const int num_workers
) {
    assert(solution.ndim() == 2);

    assert(solution.flags() & py::array::c_style);
    assert(seeds.flags() & py::array::c_style);
    assert(num_nodes.flags() & py::array::c_style);
    assert(num_nodes_to_keep.flags() & py::array::c_style);

    const int batch_size = solution.shape()[0];
    const int solution_stride_batch = solution.shape()[1];

    auto solution_ptr = static_cast<bool *>(solution.request().ptr);
    auto seeds_ptr = seeds.data();
    auto num_nodes_ptr = num_nodes.data();
    auto num_nodes_to_keep_ptr = num_nodes_to_keep.data();

    py::gil_scoped_release release;

    auto task_fn = [&](const int task_id) {
        mis::partially_mask(
            solution_ptr + task_id * solution_stride_batch, 
            seeds_ptr[task_id], 
            num_nodes_ptr[task_id], 
            num_nodes_to_keep_ptr[task_id]
        );
    };

    parallelize(task_fn, batch_size, std::min(batch_size, num_workers));
}

