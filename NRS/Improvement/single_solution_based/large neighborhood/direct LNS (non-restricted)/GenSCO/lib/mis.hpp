#pragma once

#include <algorithm>
#include <cassert>
#include <limits>
#include <random>
#include <vector>


namespace mis {

template<typename NodeIndexDType>
inline void
partially_greedy_insert(
    bool * solution,
    bool * mask,
    int & num_inserted_nodes,
    int & num_masked_nodes,
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
        if (num_inserted_nodes >= target_num_nodes || num_masked_nodes >= num_nodes) {
            break;
        }

        const NodeIndexDType & candidate_node_idx = candidate_nodes[i];
        if (candidate_node_idx >= num_nodes) {
            continue;
        }
        if (mask[candidate_node_idx]) {
            continue;
        } 
        // insert
        solution[candidate_node_idx] = true;
        ++num_inserted_nodes;
        mask[candidate_node_idx] = true;
        ++num_masked_nodes;
        for (const auto & neighbor_idx: neighbors[candidate_node_idx]) {
            if (!mask[neighbor_idx]) {
                ++num_masked_nodes;
            }
            mask[neighbor_idx] = true;
        }
    }
}


template<typename NodeIndexDType>
inline auto
edges2neighbors(
    const NodeIndexDType * edges,
    const int num_nodes,
    const int num_edges
) {
    std::vector<std::vector<NodeIndexDType>> neighbors(num_nodes);
    for (int edge_idx = 0; edge_idx < num_edges; ++edge_idx) {
        const auto & i = edges[2 * edge_idx];
        const auto & j = edges[2 * edge_idx + 1];

        neighbors[i].emplace_back(j);
        neighbors[j].emplace_back(i);
    }
    return neighbors;
}


template<typename NodeIndexDType>
inline auto
init_insertion_state(
    bool * mask,
    int & num_masked_nodes,
    const bool * solution,
    const std::vector<std::vector<NodeIndexDType>> & neighbors,
    const int num_nodes
) {
    assert(static_cast<int>(neighbors.size()) == num_nodes);

    std::fill(mask, mask + num_nodes, false);
    num_masked_nodes = 0;
    
    for (int node_idx = 0; node_idx < num_nodes; ++node_idx) {
        if (solution[node_idx]) {
            mask[node_idx] = true;
            ++num_masked_nodes;
            for (const auto & neighbor_idx: neighbors[node_idx]) {
                if (!mask[neighbor_idx]) {
                    ++num_masked_nodes;
                }
                mask[neighbor_idx] = true;
            }
        }
    }
}


inline void
partially_mask(
    bool * solution,
    std::default_random_engine & generator,
    const int num_nodes,
    const int num_nodes_to_keep
) {
    std::vector<int> indices;

    for (int node_idx = 0; node_idx < num_nodes; ++node_idx) {
        if (solution[node_idx]) {
            indices.emplace_back(node_idx);
        }
    }

    std::shuffle(indices.begin(), indices.end(), generator);

    for (int i = num_nodes_to_keep; i < static_cast<int>(indices.size()); ++i) {
        solution[indices[i]] = false;
    }
}


inline void
partially_mask(
    bool * solution,
    const int seed,
    const int num_nodes,
    const int num_nodes_to_keep
) {
    auto generator = std::default_random_engine(seed);
    partially_mask(solution, generator, num_nodes, num_nodes_to_keep);
}


}   // namespace mis
