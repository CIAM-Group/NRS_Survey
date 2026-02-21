#pragma once

#include "pybind11/cast.h"
#include "pybind11/gil.h"
#include "pybind11/pytypes.h"
#include "par_utils.hpp"
#include "tsp.hpp"

#include <algorithm>
#include <cassert>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <random>
#include <sys/types.h>
#include <vector>

namespace py = pybind11;


inline void
tsp_two_opt_inplace_interface(
    py::array_t<int> & tours, const py::array_t<float> & dist_mat,
    const int num_steps, const int num_workers
) {
    assert(tours.flags() & py::array::c_style);
    assert(dist_mat.flags() & py::array::c_style);

    const bool batched = tours.ndim() == 2;

    if (!batched) {
        auto tours_ptr = static_cast<int *>(tours.request().ptr);
        const auto dist_mat_ptr = dist_mat.data();
        const int num_nodes = dist_mat.shape()[0];

        pybind11::gil_scoped_release release;

        tsp::two_opt(tours_ptr, dist_mat_ptr, num_nodes, num_steps);
    } else {
        const int batch_size = tours.shape()[0];
        auto tours_ptr = static_cast<int *>(tours.request().ptr);
        const auto dist_mat_ptr = dist_mat.data();
        const int num_nodes = dist_mat.shape()[1];

        pybind11::gil_scoped_release release;

        auto task_fn = [&](const int task_id) {
            tsp::two_opt(
                tours_ptr + task_id * num_nodes,
                dist_mat_ptr + task_id * num_nodes * num_nodes,
                num_nodes, num_steps
            );
        };

        parallelize(task_fn, batch_size, std::min(batch_size, num_workers));
    }
}


inline void
tsp_random_two_opt_inplace_interface(
    const int seed, py::array_t<int> & tours,
    const py::array_t<int> & num_steps, const int num_workers
) {
    assert(tours.flags() & py::array::c_style);
    assert(num_steps.flags() & py::array::c_style);

    const bool batched = tours.ndim() == 2;
    
    if (!batched) {
        const int num_nodes = tours.shape()[0];
        auto tours_ptr = static_cast<int *>(tours.request().ptr);
        const int _num_steps = static_cast<int *>(num_steps.request().ptr)[0];
        pybind11::gil_scoped_release release;
        tsp::random_two_opt(seed, tours_ptr, num_nodes, _num_steps);
    } else {
        const int batch_size = tours.shape()[0];
        const int num_nodes = tours.shape()[1];
        auto tours_ptr = static_cast<int *>(tours.request().ptr);
        auto num_steps_ptr = static_cast<int *>(num_steps.request().ptr);
        pybind11::gil_scoped_release release;
        std::vector<unsigned long> seeds;
        seeds.reserve(batch_size);
        std::mt19937 seed_generator(seed);
        for (int i = 0; i < batch_size; ++i) {
            seeds.push_back(seed_generator());
        }
        auto task_fn = [&](const int task_id) {
            tsp::random_two_opt(
                seeds[task_id],
                tours_ptr + task_id * num_nodes,
                num_nodes, num_steps_ptr[task_id]
            );
        };
        parallelize(task_fn, batch_size, std::min(batch_size, num_workers));
    }
}


inline py::tuple
tsp_double_two_opt_interface(
    const int seed, py::array_t<int> & tours, const py::array_t<float> & dist_mat,
    const int steps, const py::array_t<int> & random_steps, 
    const int num_workers
) {
    assert(tours.flags() & py::array::c_style);
    assert(dist_mat.flags() & py::array::c_style);
    assert(random_steps.flags() & py::array::c_style);

    const bool batched = tours.ndim() == 2;

    
    if (!batched) {
        const int num_nodes = tours.shape()[0];
        auto tours_ptr = static_cast<int *>(tours.request().ptr);
        const int _random_steps = static_cast<int *>(random_steps.request().ptr)[0];
        auto dist_mat_ptr = static_cast<float *>(dist_mat.request().ptr);

        auto two_opt_result = py::array_t<int>(num_nodes);
        auto random_two_opt_result = py::array_t<int>(num_nodes);
        auto two_opt_result_ptr = static_cast<int *>(two_opt_result.request().ptr);
        auto random_two_opt_result_ptr = static_cast<int *>(random_two_opt_result.request().ptr);

        pybind11::gil_scoped_release release;

        std::copy(tours_ptr, tours_ptr + num_nodes, two_opt_result_ptr);
        tsp::two_opt(two_opt_result_ptr, dist_mat_ptr, num_nodes, steps);
        std::copy(two_opt_result_ptr, two_opt_result_ptr + num_nodes, random_two_opt_result_ptr);
        tsp::random_two_opt(seed, random_two_opt_result_ptr, num_nodes, _random_steps);

        pybind11::gil_scoped_acquire acquire;

        return py::make_tuple(two_opt_result, random_two_opt_result);
    } else {
        const int batch_size = tours.shape()[0];
        const int num_nodes = tours.shape()[1];
        auto tours_ptr = static_cast<int *>(tours.request().ptr);
        auto random_steps_ptr = static_cast<int *>(random_steps.request().ptr);
        auto dist_mat_ptr = static_cast<float *>(dist_mat.request().ptr);

        auto two_opt_result = py::array_t<int>({batch_size, num_nodes});
        auto random_two_opt_result = py::array_t<int>({batch_size, num_nodes});
        auto two_opt_result_ptr = static_cast<int *>(two_opt_result.request().ptr);
        auto random_two_opt_result_ptr = static_cast<int *>(random_two_opt_result.request().ptr);
        
        pybind11::gil_scoped_release release;

        std::vector<unsigned long> seeds;
        seeds.reserve(batch_size);
        std::mt19937 seed_generator(seed);
        for (int i = 0; i < batch_size; ++i) {
            seeds.push_back(seed_generator());
        }
        auto task_fn = [&](const int task_id) {
            std::copy(
                tours_ptr + num_nodes * task_id, 
                (tours_ptr + num_nodes * task_id) + num_nodes,
                two_opt_result_ptr + num_nodes * task_id
            );
            tsp::two_opt(
                two_opt_result_ptr + num_nodes * task_id, 
                dist_mat_ptr + num_nodes * num_nodes * task_id, 
                num_nodes, steps
            );
            std::copy(
                two_opt_result_ptr + num_nodes * task_id, 
                (two_opt_result_ptr + num_nodes * task_id) + num_nodes, 
                random_two_opt_result_ptr + num_nodes * task_id
            );
            tsp::random_two_opt(
                seeds[task_id],
                random_two_opt_result_ptr + task_id * num_nodes,
                num_nodes, random_steps_ptr[task_id]
            );
        };
        parallelize(task_fn, batch_size, std::min(batch_size, num_workers));

        pybind11::gil_scoped_acquire acquire;

        return py::make_tuple(two_opt_result, random_two_opt_result);
    }
}


inline py::array_t<int>
tsp_greedy_insert_interface(
    const py::array_t<int> & candidate_edges, const int num_nodes, const int num_workers
) {
    assert(candidate_edges.flags() & py::array::c_style);

    const bool batched = candidate_edges.ndim() == 3;
    auto candidate_edges_ptr = candidate_edges.data();
    
    if (!batched) {
        py::array_t<int> tour(num_nodes);
        auto tour_ptr = static_cast<int *>(tour.request().ptr);
        const int num_candidate_edges = candidate_edges.shape()[0];

        pybind11::gil_scoped_release release;

        tsp::greedy_insert(tour_ptr, candidate_edges_ptr, num_nodes, num_candidate_edges);
        return tour;
    } else {
        const int batch_size = candidate_edges.shape()[0];
        const int num_candidate_edges = candidate_edges.shape()[1];
        py::array_t<int> tours({batch_size, num_nodes});
        auto tours_ptr = static_cast<int *>(tours.request().ptr);

        pybind11::gil_scoped_release release;

        auto task_fn = [&](const int task_id) {
            tsp::greedy_insert(
                tours_ptr + task_id * num_nodes, 
                candidate_edges_ptr + task_id * num_candidate_edges * 2, 
                num_nodes, num_candidate_edges
            );
        };

        parallelize(task_fn, batch_size, std::min(batch_size, num_workers));

        return tours;
    }

}







inline auto
tsp_eval_cost(
    const py::array_t<int> & tour,
    const py::array_t<float> & dist_mat,
    const int num_workers
) {
    assert(tour.flags() & py::array::c_style);
    assert(dist_mat.flags() & py::array::c_style);

    const bool batched = tour.ndim() == 2;
    auto tour_ptr = tour.data();
    auto dist_mat_ptr = dist_mat.data();

    if (!batched) {
        const int num_nodes = dist_mat.shape()[0];

        auto cost = py::array_t<float>({});
        auto cost_ptr = static_cast<float *>(cost.request().ptr); 

        py::gil_scoped_release release;

        tsp::eval_cost(*cost_ptr, tour_ptr, dist_mat_ptr, num_nodes);

        return cost;
    } else {
        const int batch_size = dist_mat.shape()[0];
        const int num_nodes = dist_mat.shape()[1];

        auto cost = py::array_t<float>(batch_size);
        auto cost_ptr = static_cast<float *>(cost.request().ptr);
        
        py::gil_scoped_release release;

        auto task_fn = [&](const int task_id) {
            tsp::eval_cost(
                cost_ptr[task_id], 
                tour_ptr + task_id * num_nodes, 
                dist_mat_ptr + task_id * num_nodes * num_nodes, 
                num_nodes
            );
        };
        
        parallelize(task_fn, batch_size, std::min(batch_size, num_workers));

        return cost;
    }
}

