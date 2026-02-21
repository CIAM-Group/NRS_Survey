#pragma once

#include <algorithm>
#include <cassert>
#include <limits>
#include <tuple>
#include <random>
#include <vector>

#include "params.hpp"


namespace tsp {

inline std::tuple<int, int, float>
find_best_two_opt(const int * tour, const float * dist_mat, const int num_nodes) {
    float best_cost_reduction = std::numeric_limits<float>().min();
    int best_i = 0, best_j = 0;
    for (int i = 0; i < num_nodes - 2; ++i) {
        for (int j = i + 2; j < num_nodes; ++j) {
            const float cost_reduction = (
                dist_mat[tour[i] * num_nodes + tour[i + 1]] + dist_mat[tour[j] * num_nodes + tour[(j + 1) % num_nodes]]
                - dist_mat[tour[i] * num_nodes + tour[j]] - dist_mat[tour[i + 1] * num_nodes + tour[(j + 1) % num_nodes]]
            );
            if constexpr (TWO_OPT_TYPE == TwoOptType::Best) {
                if (cost_reduction > best_cost_reduction) {
                    best_cost_reduction = cost_reduction;
                    best_i = i;
                    best_j = j;
                }
            } else {
                if (cost_reduction > 0.) {
                    return {i, j, cost_reduction};
                }
            }
        }
    }
    return {best_i, best_j, best_cost_reduction};
}


inline void apply_two_opt(int * tour, const int i, const int j) {
    std::reverse(tour + i + 1, tour + j + 1);
}


inline void two_opt(
    int * tour, const float * dist_mat, 
    const int num_nodes, const int num_steps
) {
    if (num_steps < 0) {
        return;
    }

    for (int step_ctr = 0; step_ctr < num_steps; ++step_ctr) {
        auto [i, j, cost_reduction] = tsp::find_best_two_opt(tour, dist_mat, num_nodes);
        if (cost_reduction < 1e-5) {
            break;
        } else {
            tsp::apply_two_opt(tour, i, j);
        }
    }
}


inline void random_two_opt(
    std::mt19937 & generator, int * tour, const int num_nodes, const int num_steps
) {
    assert(num_nodes >= 5);

    auto i_dist = std::uniform_int_distribution<int>(0, num_nodes - 1);
    auto diff_dist = std::uniform_int_distribution<int>(2, num_nodes - 2);
    for (int step = 0; step < num_steps; ++step) {
        int i = i_dist(generator);
        int j = (i + diff_dist(generator)) % num_nodes;
        if (i > j) {
            std::swap(i, j);
        }
        tsp::apply_two_opt(tour, i, j);
    }
}


inline void random_two_opt(
    const int seed, int * tour, const int num_nodes, const int num_steps
) {
    auto generator = std::mt19937(seed);
    tsp::random_two_opt(generator, tour, num_nodes, num_steps);
}


inline void random_two_opt(
    const unsigned long seed, int * tour, const int num_nodes, const int num_steps
) {
    auto generator = std::mt19937(seed);
    tsp::random_two_opt(generator, tour, num_nodes, num_steps);
}


inline void neighbors2sol(
    int * sol,
    const int * neighbors,
    const int num_nodes
) {
    bool has_visited[num_nodes];
    std::fill(has_visited, has_visited + num_nodes, false);

    sol[0] = 0;
    int current_node = 0;
    for (int i = 1; i < num_nodes; ++i) {
        has_visited[current_node] = true;
        if (!has_visited[neighbors[2 * current_node]]) {
            sol[i] = neighbors[2 * current_node];
            current_node = neighbors[2 * current_node];
        } else {
            assert(!has_visited[neighbors[2 * current_node + 1]]);
            sol[i] = neighbors[2 * current_node + 1];
            current_node = neighbors[2 * current_node + 1];
        }
    }
}


inline void edges2neighbors(
    int * neighbors, const int * edges,
    const int num_nodes, const int num_edges
) {
    std::fill(neighbors, neighbors + 2 * num_nodes, -1);

    auto set_neighbor = [&neighbors](const int & i, const int & j) {
        if (neighbors[2 * j] == -1) {
            neighbors[2 * j] = i;
        } else {
            assert(neighbors[2 * j + 1] == -1);
            neighbors[2 * j + 1] = i;
        }
        if (neighbors[2 * i] == -1) {
            neighbors[2 * i] = j;
        } else {
            assert(neighbors[2 * i + 1] == -1);
            neighbors[2 * i + 1] = j;
        }
    };

    for (int edge_idx = 0; edge_idx < num_edges; ++edge_idx) {
        set_neighbor(edges[2 * edge_idx], edges[2 * edge_idx + 1]);
    }
}





inline void
fill_solution_naive(
    int * edges,
    int * neighbors,
    int * subtour_id,
    const int num_nodes,
    int num_inserted_edges,
    int num_edges_to_fill
);


inline void greedy_insert(
    int * tour, const int * candidate_edges, const int num_nodes, const int num_candidate_edges
) {
    std::vector<int> subtour_id;
    subtour_id.resize(num_nodes);
    std::fill(subtour_id.begin(), subtour_id.end(), 0);     // `0` means has not been inserted

    auto update_id = [&subtour_id, &num_nodes](const int from, const int to) {
        for (int i = 0; i < num_nodes; ++i) {
            if (subtour_id[i] == from) {
                subtour_id[i] = to;
            }
        }
    };

    std::vector<int> neighbors;
    neighbors.resize(2 * num_nodes);
    std::fill(neighbors.begin(), neighbors.end(), -1);  // `-1` means undetermined

    auto set_neighbor = [&neighbors](const int & i, const int & j) {
        if (neighbors[2 * i] == -1) {
            neighbors[2 * i] = j;
        } else {
            assert(neighbors[2 * i + 1] == -1);
            neighbors[2 * i + 1] = j;
        }
        if (neighbors[2 * j] == -1) {
            neighbors[2 * j] = i;
        } else {
            assert(neighbors[2 * j + 1] == -1);
            neighbors[2 * j + 1] = i;
        }
    };

    int next_available_subtour = 1;
    int num_inserted_edges = 0;
    for (int edge_idx = 0; edge_idx < num_candidate_edges; ++edge_idx) {
        const int i = candidate_edges[2 * edge_idx];
        const int j = candidate_edges[2 * edge_idx + 1];

        if (neighbors[2 * i + 1] != -1 || neighbors[2 * j + 1] != -1) {
            continue;
        }
        if (i == j) {
            continue;
        }

        if (subtour_id[i] == 0) {
            ++num_inserted_edges;
            if (subtour_id[j] == 0) {
                subtour_id[i] = next_available_subtour;
                subtour_id[j] = next_available_subtour;
                ++next_available_subtour;
            } else {
                subtour_id[i] = subtour_id[j];
            }
            set_neighbor(i, j);
        } else {
            if (subtour_id[j] == 0) {
                ++num_inserted_edges;
                subtour_id[j] = subtour_id[i];
                set_neighbor(i, j);
            } else {
                // both have been inserted
                if (subtour_id[i] == subtour_id[j]) {
                    // same subtour, cannot insert
                    continue;
                } else {
                    ++num_inserted_edges;
                    update_id(subtour_id[j], subtour_id[i]);
                    set_neighbor(i, j);
                }
            }
        }

        if (num_inserted_edges == num_nodes - 1) {
            break;
        }
    }

    if constexpr (ENABLE_INSERTION_FALLBACK) {
        if (!(num_inserted_edges == num_nodes - 1)) {
            std::vector<int> edges(2 * num_nodes);
            fill_solution_naive(
                edges.data(), neighbors.data(), 
                subtour_id.data(), num_nodes, 
                num_inserted_edges, 
                num_nodes - num_inserted_edges
            );
            neighbors2sol(tour, neighbors.data(), num_nodes);
            return;
        }
    } else {
        assert(num_inserted_edges == num_nodes - 1);
    }

    // neighbors -> tour
    int start_node = 0;
    while (neighbors[2 * start_node + 1] != -1) {
        ++start_node;
    }
    tour[0] = start_node;
    const int start_node_neighbor = neighbors[2 * start_node];
    if (neighbors[2 * start_node_neighbor] == start_node) {
        std::swap(neighbors[2 * start_node_neighbor], neighbors[2 * start_node_neighbor + 1]);
    } 
    int current_node = start_node;

    bool has_visited[num_nodes];
    std::fill(has_visited, has_visited + num_nodes, false);

    for (int i = 1; i < num_nodes; ++i) {
        has_visited[current_node] = true;
        if (!has_visited[neighbors[2 * current_node]]) {
            tour[i] = neighbors[2 * current_node];
            current_node = neighbors[2 * current_node];
        } else {
            assert(!has_visited[neighbors[2 * current_node + 1]]);
            tour[i] = neighbors[2 * current_node + 1];
            current_node = neighbors[2 * current_node + 1];
        }
    }
}


inline void partially_greedy_insert(
    int * edges, 
    int * subtour_id,
    int * neighbors,
    const int * candidate_edges, 
    const int num_nodes, 
    const int num_candidate_edges,
    const int num_inserted_edges,
    const int num_edges_to_insert
) {
    assert(num_nodes >= num_edges_to_insert + num_inserted_edges);

    auto update_id = [&subtour_id, &num_nodes](const int from, const int to) {
        for (int i = 0; i < num_nodes; ++i) {
            if (subtour_id[i] == from) {
                subtour_id[i] = to;
            }
        }
    };

    auto set_neighbor = [&neighbors](const int & i, const int & j) {
        if (neighbors[2 * i] == -1) {
            neighbors[2 * i] = j;
        } else {
            assert(neighbors[2 * i + 1] == -1);
            neighbors[2 * i + 1] = j;
        }
        if (neighbors[2 * j] == -1) {
            neighbors[2 * j] = i;
        } else {
            assert(neighbors[2 * j + 1] == -1);
            neighbors[2 * j + 1] = i;
        }
    };

    int next_available_subtour = num_inserted_edges + 1;
    int num_inserted_edges_this_round = 0;
    
    for (int edge_idx = 0; edge_idx < num_candidate_edges; ++edge_idx) {
        const int i = candidate_edges[2 * edge_idx];
        const int j = candidate_edges[2 * edge_idx + 1];

        if (neighbors[2 * i + 1] != -1 || neighbors[2 * j + 1] != -1) {
            continue;
        }
        if (i == j) {
            continue;
        }

        if (subtour_id[i] == 0) {
            edges[2 * (num_inserted_edges + num_inserted_edges_this_round)] = i;
            edges[2 * (num_inserted_edges + num_inserted_edges_this_round) + 1] = j;
            ++num_inserted_edges_this_round;
            if (subtour_id[j] == 0) {
                subtour_id[i] = next_available_subtour;
                subtour_id[j] = next_available_subtour;
                ++next_available_subtour;
            } else {
                subtour_id[i] = subtour_id[j];
            }
            set_neighbor(i, j);
        } else {
            if (subtour_id[j] == 0) {
                edges[2 * (num_inserted_edges + num_inserted_edges_this_round)] = i;
                edges[2 * (num_inserted_edges + num_inserted_edges_this_round) + 1] = j;
                ++num_inserted_edges_this_round;
                subtour_id[j] = subtour_id[i];
                set_neighbor(i, j);
            } else {
                // both have been inserted
                if (subtour_id[i] == subtour_id[j]) {
                    // same subtour, cannot insert
                    continue;
                } else {
                    edges[2 * (num_inserted_edges + num_inserted_edges_this_round)] = i;
                    edges[2 * (num_inserted_edges + num_inserted_edges_this_round) + 1] = j;
                    ++num_inserted_edges_this_round;
                    update_id(subtour_id[j], subtour_id[i]);
                    set_neighbor(i, j);
                }
            }
        }

        if (num_inserted_edges + num_inserted_edges_this_round == num_nodes - 1) {
            break;
        }
        if (num_inserted_edges_this_round >= num_edges_to_insert) {
            break;
        }
    }
    if constexpr (ENABLE_INSERTION_FALLBACK) {
        if (!(num_inserted_edges + num_inserted_edges_this_round == num_nodes - 1 || num_inserted_edges_this_round == num_edges_to_insert)) {
            fill_solution_naive(
                edges, neighbors, 
                subtour_id, num_nodes, 
                num_inserted_edges + num_inserted_edges_this_round, 
                num_edges_to_insert - num_inserted_edges_this_round
            );
            return;
        }
    } else {
        assert(num_inserted_edges + num_inserted_edges_this_round == num_nodes - 1 || num_inserted_edges_this_round == num_edges_to_insert);
    }
    
    if (num_inserted_edges + num_inserted_edges_this_round == num_nodes - 1 && num_inserted_edges_this_round < num_edges_to_insert) {
        assert(num_inserted_edges_this_round + 1 == num_edges_to_insert);
        // find the last edge
        int node_lefted_1 = 0;
        while (neighbors[2 * node_lefted_1 + 1] != -1) {
            ++node_lefted_1;
        }
        int node_lefted_2 = num_nodes - 1;
        while (neighbors[2 * node_lefted_2 + 1] != -1) {
            --node_lefted_2;
        }
        assert(node_lefted_1 < node_lefted_2);
        set_neighbor(node_lefted_1, node_lefted_2);
        edges[2 * num_nodes - 2] = node_lefted_1;
        edges[2 * num_nodes - 1] = node_lefted_2;
    }
}


inline void
eval_cost(
    float & cost,
    const int * tour,
    const float * dist_mat,
    const int num_nodes
) {
    cost = 0.;
    for (int i = 0; i < num_nodes - 1; ++i) {
        cost += dist_mat[tour[i] * num_nodes + tour[i + 1]];
    }
    cost += dist_mat[tour[num_nodes - 1] * num_nodes + tour[0]];
}


inline void
fill_solution_naive(
    int * edges,
    int * neighbors,
    int * subtour_id,
    const int num_nodes,
    int num_inserted_edges,
    int num_edges_to_fill
) {
    assert(num_inserted_edges + num_edges_to_fill <= num_nodes);

    std::vector<int> free_node_indices;
    for (int i = 0; i < num_nodes; ++i) {
        if (neighbors[2 * i] == -1 || neighbors[2 * i + 1] == -1) {
            free_node_indices.emplace_back(i);
        }
    }

    const int num_free_nodes = static_cast<int>(free_node_indices.size());
    std::vector<int> candidate_edges;
    candidate_edges.reserve(num_free_nodes * (num_free_nodes - 1));

    for (int idx = 0; idx < num_free_nodes - 1; ++idx) {
        const int & node_x = free_node_indices[idx];
        for (int idx_2 = idx + 1; idx_2 < num_free_nodes; ++idx_2) {
            const int & node_y = free_node_indices[idx_2];
            candidate_edges.emplace_back(node_x);
            candidate_edges.emplace_back(node_y);
        }
    }    

    assert(candidate_edges.size() == num_free_nodes * (num_free_nodes - 1));

    partially_greedy_insert(
        edges, subtour_id, neighbors, candidate_edges.data(), num_nodes,
        candidate_edges.size() / 2, num_inserted_edges, num_edges_to_fill
    );
}

}   // namespace tsp
