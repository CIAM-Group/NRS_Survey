"""
BQ-NCO
Copyright (c) 2023-present NAVER Corp.
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license
"""

import numpy
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.spatial.distance import pdist, squareform
import numpy as np
from torch.utils.data.dataloader import default_collate


class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self


def make_vrplib_data(filename):
    import vrplib

    node_coords = []
    demands = []
    capacitys = []
    costs = []
    names = []
    edge_weight_types = []

    from tqdm import tqdm

    for line in tqdm(open(filename, "r",encoding='utf-8').readlines()):
        line = line.split(", ")

        name_index = int(line.index('[\'name\''))
        depot_index = int(line.index('\'depot\''))
        customer_index = int(line.index('\'customer\''))
        capacity_index = int(line.index('\'capacity\''))
        demand_index = int(line.index('\'demand\''))
        cost_index = int(line.index('\'cost\''))

        try:
            ew_index = int(line.index('\'edge_weight_type\''))
            ew_type = line[ew_index + 1].strip("'\" \n\t[]")
        except ValueError:
            ew_type = 'Exact'

        depot = [[float(line[depot_index + 1]), float(line[depot_index + 2])]]
        customer = [[float(line[idx]), float(line[idx + 1])] for idx in
                    range(customer_index + 1, demand_index, 2)]

        loc = depot + customer
        # 包括 depot 的 location，在第一个

        capacity = int(float(line[capacity_index + 1]))
        # demand = [0] + [int(line[idx]) for idx in range(demand_index + 1, cost_index)]
        demand = [int(line[idx]) for idx in range(demand_index + 1, capacity_index)]
        # [0] + 包括depot的demand，其为 0，在第一个

        cost = float(line[cost_index + 1])

        node_coords.append(loc)
        demands.append(demand)
        capacitys.append(capacity)
        costs.append(cost)
        names.append(line[name_index+1][1:-1])
        edge_weight_types.append(ew_type) 
        # print(node_coords,demands,capacitys,costs,names)
        # assert False

    # 每一行的数据表示一个instance，每一个instance的size不一样
    node_coords = np.array(node_coords,dtype=object)
    demands = np.array(demands,dtype=object)
    capacitys = np.array(capacitys)
    costs = np.array(costs)
    names = np.array(names)
    edge_weight_types = np.array(edge_weight_types)
    # print(instance_data.shape)

    return node_coords, demands, capacitys, costs, names, edge_weight_types

class DataSet(Dataset):

    def __init__(self, node_coords, demands, capacities, remaining_capacities, tour_lens=None, via_depots=None, edge_weight_types=None, nodes_original=None):
        self.node_coords = node_coords
        self.demands = demands
        self.capacities = capacities
        self.remaining_capacities = remaining_capacities
        self.via_depots = via_depots
        self.tour_lens = tour_lens
        self.edge_weight_types = edge_weight_types
        self.nodes_original = nodes_original

    def __len__(self):
        return len(self.node_coords)

    def __getitem__(self, item):
        node_coords = self.node_coords[item]
        demands = self.demands[item]
        capacity = self.capacities[item]
        if self.tour_lens is not None:
            tour_len = self.tour_lens[item]
        else:
            tour_len = numpy.array([])

        if self.remaining_capacities is not None:
            via_depots = self.via_depots[item]
            current_capacities = self.remaining_capacities[item]
        else:
            via_depots = numpy.array([])
            current_capacities = numpy.array([])
        
        if self.edge_weight_types is not None:
            ew_type = self.edge_weight_types[item]
        else:
            ew_type = 'Exact'
        
        # Calculate distance matrix on ORIGINAL coordinates if available (for correct rounding)
        if self.nodes_original is not None:
            coords_for_dist = self.nodes_original[item]
        else:
            coords_for_dist = node_coords

        dist = pdist(coords_for_dist, metric='euclidean')
        if ew_type == 'EUC_2D':
            distance_matrix = squareform(np.floor(dist + 0.5))
        elif ew_type == 'CEIL_2D':
            distance_matrix = squareform(np.ceil(dist))
        else:
            # For Exact, we might be using normalized coords if nodes_original is not provided
            # But if nodes_original IS provided, we should use it for consistency?
            # Usually Exact on normalized coords is fine for training. 
            # But for testing against 'tour_len' (which is real cost), we need Real distances.
            # Assuming 'tour_len' is always Real Cost.
            distance_matrix = squareform(dist)
        
        # From list to tensors as a DotDict
        item_dict = DotDict()
        item_dict.dist_matrices = torch.Tensor(distance_matrix)
        item_dict.node_coords = torch.Tensor(node_coords)
        item_dict.demands = torch.Tensor(demands)
        item_dict.capacities = torch.tensor(capacity).float()
        item_dict.remaining_capacities = torch.Tensor(current_capacities)
        item_dict.tour_len = torch.tensor(tour_len)
        item_dict.via_depots = torch.Tensor(via_depots).long()
        return item_dict


def load_dataset(filename, batch_size, shuffle=False, what="test"):
    print(filename)

    node_coords, demands, capacitys, costs, names, edge_weight_types = make_vrplib_data(filename)

    nodes_tmp = []
    nodes_original = []
    factors = []
    for i in range(len(node_coords)):
        tmp0 = np.array(node_coords[i])
        tmp1 = tmp0
        tmp2 = tmp0[[0]]
        tmp3 = np.concatenate((tmp1, tmp2), axis=0)

        # Original coords with depot at end - for exact distance calculation
        nodes_original.append(tmp3)

        factors.append(np.max(tmp3) - np.min(tmp3))
        tmp3 = (tmp3 - np.min(tmp3)) / (np.max(tmp3) - np.min(tmp3))

        nodes_tmp.append(tmp3)
    print(factors)
    data = {}

    # data['coords'] = nodes_tmp
    # data['demands'] = demands
    # data["tour_lens"] = costs

    demand_tmp = []
    for i in range(len(demands)):
        tmp0 = np.array(demands[i])
        tmp1 = tmp0
        tmp2 = tmp0[[0]]
        tmp3 = np.concatenate((tmp1, tmp2), axis=0)
        demand_tmp.append(tmp3)

    data["nodes_coord"] = nodes_tmp
    data["nodes_original"] = nodes_original

    data["demands"] = demands
    data["capacities"] = capacitys
    data["tour_lens"] = costs
    data["reorder"] = False

    # cvrp_node_coords, cvrp_demands, cvrp_capacitys, \
    # vrplib_cost, vrplib_name = make_vrplib_data(filename)
    # data = {}
    #
    # nodes_tmp = []
    # factors = []
    # for i in range(len(cvrp_node_coords)):
    #     tmp0 = np.array(cvrp_node_coords[i])
    #     tmp1 = tmp0
    #     tmp2 = tmp0[[0]]
    #     tmp3 = np.concatenate((tmp1, tmp2), axis=0)
    #
    #     factors.append(np.max(tmp3) - np.min(tmp3))
    #     tmp3 = (tmp3 - np.min(tmp3)) / (np.max(tmp3) - np.min(tmp3))
    #
    #     nodes_tmp.append(tmp3)
    #
    # demand_tmp = []
    # for i in range(len(cvrp_demands)):
    #     tmp0 = np.array(cvrp_demands[i])
    #     tmp1 = tmp0
    #     tmp2 = tmp0[[0]]
    #     tmp3 = np.concatenate((tmp1, tmp2), axis=0)
    #     demand_tmp.append(tmp3)
    #
    # data["coords"] = nodes_tmp
    # data["demands"] = demand_tmp
    # data["capacities"] = cvrp_capacitys
    # data["tour_lens"] = vrplib_cost
    # data["reorder"] = False

    if what == "train":
        assert data["reorder"]

    # node_coords = data["coords"]
    # demands = data["demands"]
    # capacities = data["capacities"]

    print('capacities',capacitys)

    # in training dataset we have via_depots and remaining capacities but not tour lens
    tour_lens = data["tour_lens"] if "tour_lens" in data.keys() else None
    remaining_capacities = data["remaining_capacities"] if "remaining_capacities" in data.keys() else None
    via_depots = data["via_depots"] if "via_depots" in data.keys() else None

    collate_fn = collate_func_with_sample if what == "train" else None

    dataset = DataLoader(DataSet(nodes_tmp, demand_tmp, capacitys,
                                 remaining_capacities=remaining_capacities,
                                 tour_lens=costs,
                                 via_depots=via_depots,
                                 edge_weight_types=edge_weight_types,
                                 nodes_original=nodes_original), batch_size=batch_size,
                         drop_last=False, shuffle=shuffle, collate_fn=collate_fn)
    return dataset,factors,names


def collate_func_with_sample(l_dataset_items):
    """
    assemble minibatch out of dataset examples.
    For instances of TOUR-CVRP of graph size N (i.e. nb_nodes=N+1 including return to beginning node),
    this function also takes care of sampling a SUB-problem (PATH-TSP) of size 3 to N+1
    """
    nb_nodes = len(l_dataset_items[0].dist_matrices)
    begin_idx = np.random.randint(0, nb_nodes - 3)  # between _ included and nb_nodes + 1 excluded

    l_dataset_items_new = []
    for d in l_dataset_items:
        d_new = {}
        for k, v in d.items():
            if k == "dist_matrices":
                v_ = v[begin_idx:, begin_idx:]
            elif k == "remaining_capacities":
                v_ = v[begin_idx]
            elif k == "capacities":
                v_ = v
            else:
                v_ = v[begin_idx:, ...]

            d_new.update({k + '_s': v_})
        l_dataset_items_new.append({**d, **d_new})

    return default_collate(l_dataset_items_new)

# def make_vrplib_data(filename):
#     import vrplib
#
#     node_coords = []
#     demands = []
#     capacitys = []
#     costs = []
#     names = []
#
#     from tqdm import tqdm
#
#     for line in tqdm(open(filename, "r").readlines(), ascii=True):
#         line = line.split(", ")
#
#         name_index = int(line.index('[\'name\''))
#         depot_index = int(line.index('\'depot\''))
#         customer_index = int(line.index('\'customer\''))
#         capacity_index = int(line.index('\'capacity\''))
#         demand_index = int(line.index('\'demand\''))
#         cost_index = int(line.index('\'cost\''))
#
#         depot = [[float(line[depot_index + 1]), float(line[depot_index + 2])]]
#         customer = [[float(line[idx]), float(line[idx + 1])] for idx in
#                     range(customer_index + 1, demand_index, 2)]
#
#         loc = depot + customer
#         # 包括 depot 的 location，在第一个
#
#         capacity = int(float(line[capacity_index + 1]))
#         # demand = [0] + [int(line[idx]) for idx in range(demand_index + 1, cost_index)]
#         demand = [int(line[idx]) for idx in range(demand_index + 1, capacity_index)]
#         # [0] + 包括depot的demand，其为 0，在第一个
#
#         cost = float(line[cost_index + 1])
#
#         node_coords.append(loc)
#         demands.append(demand)
#         capacitys.append(capacity)
#         costs.append(cost)
#         names.append(line[name_index+1][1:-1])
#         # print(node_coords,demands,capacitys,costs,names)
#         # assert False
#
#     # 每一行的数据表示一个instance，每一个instance的size不一样
#     node_coords = np.array(node_coords,dtype=object)
#     demands = np.array(demands,dtype=object)
#     capacitys = np.array(capacitys)
#     costs = np.array(costs)
#     names = np.array(names)
#     # print(instance_data.shape)
#
#     return node_coords, demands, capacitys, costs, names
