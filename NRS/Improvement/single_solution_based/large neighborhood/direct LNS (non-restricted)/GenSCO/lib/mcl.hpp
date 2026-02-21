#pragma once

#include <algorithm>
#include <cassert>
#include <limits>
#include <random>
#include <vector>


namespace mcl {

template<typename NodeIndexDType>
inline void
partially_greedy_insert(
    bool * solution,
    int * num_clique_neighbors,
    int & num_inserted_nodes,
    const NodeIndexDType * candidate_nodes,
    const std::vector<std::vector<NodeIndexDType>> & neighbors,
    const int num_nodes,
    const int num_candidate_nodes,
    int target_num_nodes
) {
    assert(static_cast<int>(neighbors.size()) == num_nodes);

    if (target_num_nodes < 0) {
        target_num_nodes = std::numeric_limits<int>().max();
    }

    for (int i = 0; i < num_candidate_nodes; ++i) {
        if (num_inserted_nodes >= target_num_nodes) {
            break;
        }

        const NodeIndexDType & candidate_node_idx = candidate_nodes[i];
        if (candidate_node_idx >= num_nodes) {
            continue;
        }
        if (num_clique_neighbors[candidate_node_idx] < num_inserted_nodes) {    
            continue;       // this will also filter out those already in the clique
        } 
        // insert
        solution[candidate_node_idx] = true;
        ++num_inserted_nodes;

        // update
        for (const auto & neighbor_idx: neighbors[candidate_node_idx]) {
            ++num_clique_neighbors[neighbor_idx];
        }
    }
}

// TODO: fully support partial insertion for mcl


}   // namespace mcl 
