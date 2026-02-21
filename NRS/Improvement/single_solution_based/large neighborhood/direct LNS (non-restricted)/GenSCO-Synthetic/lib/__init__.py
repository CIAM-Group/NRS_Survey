import numpy as np
import typing as tp

from .interface import (
    tsp_two_opt_inplace as _tsp_two_opt_inplace_impl,
    tsp_random_two_opt_inplace as _tsp_random_two_opt_inplace_impl,
    tsp_double_two_opt as _tsp_double_two_opt_impl,
    tsp_greedy_insert as _tsp_greedy_insert_impl,
    tsp_eval_cost as _tsp_eval_cost_impl,
    
    mis_edges2neighbors as _mis_edges2neighbors_impl,
    mis_edges2neighbors_nocast as _mis_edges2neighbors_nocast_impl,
    mis_partially_greedy_insert as _mis_partially_greedy_insert_impl,

    mcl_partially_greedy_insert as _mcl_partially_greedy_insert_impl,
)

MISBatchedNeighbors = tp.Any


def tsp_eval_cost(
    tour: np.ndarray,
    dist_mat: np.ndarray,
    num_workers: int,
) -> np.ndarray:
    return _tsp_eval_cost_impl(
        tour, dist_mat, num_workers,
    )


def tsp_two_opt(
    tours: np.ndarray, dist_mat: np.ndarray, num_steps: int, num_workers: int = 1, inplace: bool = False,
) -> np.ndarray:
    if not inplace:
        tours = tours.copy()
    _tsp_two_opt_inplace_impl(tours, dist_mat, num_steps, num_workers)
    return tours


def tsp_random_two_opt(
    seed: int, tours: np.ndarray, num_steps: np.ndarray, num_workers: int = 1, inplace: bool = False,
) -> np.ndarray:
    if not inplace:
        tours = tours.copy()
    _tsp_random_two_opt_inplace_impl(seed, tours, num_steps, num_workers)
    return tours


def tsp_double_two_opt(
    seed: int, tours: np.ndarray, dist_mat: np.ndarray,
    steps: int, random_steps: np.ndarray, 
    num_workers: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    return _tsp_double_two_opt_impl(seed, tours, dist_mat, steps, random_steps, num_workers)


def tsp_greedy_insert(
    candidate_edges: np.ndarray, num_nodes: int, num_workers: int = 1,
) -> np.ndarray:
    return _tsp_greedy_insert_impl(candidate_edges, num_nodes, num_workers)
    

def mis_edges2neighbors(
    edges: np.ndarray,
    num_nodes: np.ndarray,
    num_edges: np.ndarray,
    num_worker: int = 1,
    *,
    tolist: bool = True,
) -> list[list[list[int]]] | MISBatchedNeighbors:
    if tolist:
        return _mis_edges2neighbors_impl(
            edges, num_nodes, num_edges, num_worker,
        )
    else:
        return _mis_edges2neighbors_nocast_impl(
            edges, num_nodes, num_edges, num_worker,
        )


def mis_partially_greedy_insert(
    solution: np.ndarray,
    mask: np.ndarray,
    num_inserted_nodes: np.ndarray,
    num_masked_nodes: np.ndarray,
    candidate_nodes: np.ndarray,
    neighbors: MISBatchedNeighbors,
    num_nodes: np.ndarray,
    target_num_nodes: np.ndarray,
    num_workers: int = 1,
):
    _mis_partially_greedy_insert_impl(
        solution, mask,
        num_inserted_nodes, num_masked_nodes,
        candidate_nodes, neighbors,
        num_nodes, target_num_nodes,
        num_workers,
    )


class MISPartialInsertion:
    def __init__(
        self,
        neighbors: MISBatchedNeighbors,
        num_nodes: np.ndarray,
        num_nodes_padded: int,
        seed: int = 0,
        num_workers: int = 1,
    ):
        self.neighbors = neighbors
        self.num_nodes = num_nodes
        self.num_workers = num_workers

        batch_size = num_nodes.shape[0]

        self.solution = np.zeros([batch_size, num_nodes_padded], dtype='bool')
        self.mask = np.zeros([batch_size, num_nodes_padded], dtype='bool')
        self.num_inserted_nodes = np.zeros([batch_size], dtype=np.int32)
        self.num_masked_nodes = np.zeros([batch_size], dtype=np.int32)

        self.generator = np.random.default_rng(seed)
        self.batch_size = batch_size

    def insert(
        self,
        candidate_nodes: np.ndarray,
        target_num_nodes: np.ndarray,
    ):
        mis_partially_greedy_insert(
            self.solution, self.mask,
            self.num_inserted_nodes, self.num_masked_nodes,
            candidate_nodes, self.neighbors, self.num_nodes,
            target_num_nodes, self.num_workers,
        )


def mis_greedy_insert(
    neighbors: MISBatchedNeighbors,
    num_nodes: np.ndarray,
    num_nodes_padded: int,
    candidate_nodes: np.ndarray,
    num_workers: int = 1,
):
    insertion = MISPartialInsertion(
        neighbors,
        num_nodes,
        num_nodes_padded,
        num_workers=num_workers,
    )
    batch_size = num_nodes.shape[0]
    insertion.insert(
        candidate_nodes,
        target_num_nodes=np.full([batch_size], -1, dtype=np.int32),
    )
    return insertion.solution, insertion.num_inserted_nodes


def mcl_greedy_insert(
    neighbors: MISBatchedNeighbors,
    num_nodes: np.ndarray,
    num_nodes_padded: int,
    candidate_nodes: np.ndarray,
    num_workers: int = 1,
):
    batch_size = candidate_nodes.shape[0]
    
    solution = np.zeros([batch_size, num_nodes_padded], dtype='bool')
    num_clique_neighbors = np.zeros([batch_size, num_nodes_padded], dtype=np.int32)
    num_inserted_nodes = np.zeros([batch_size], dtype=np.int32)
    
    _mcl_partially_greedy_insert_impl(
        solution,
        num_clique_neighbors,
        num_inserted_nodes,
        candidate_nodes,
        neighbors,
        num_nodes,
        np.full([batch_size], fill_value=-1, dtype=np.int32),
        num_workers,
    )
    return solution, num_inserted_nodes

