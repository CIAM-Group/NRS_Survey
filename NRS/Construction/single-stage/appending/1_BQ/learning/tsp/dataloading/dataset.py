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


class DataSet(Dataset):

    def __init__(self, node_coords, tour_lens=None, edge_weight_types=None, nodes_original=None, names=None):
        self.node_coords = node_coords
        self.tour_lens = tour_lens
        self.edge_weight_types = edge_weight_types
        self.nodes_original = nodes_original
        self.names = names

    def __len__(self):
        return len(self.node_coords)

    def __getitem__(self, item):
        node_coords = self.node_coords[item]
        
        ew_type = 'Exact'
        if self.edge_weight_types is not None:
            ew_type = self.edge_weight_types[item]
            
        if self.nodes_original is not None:
             coords_for_dist = self.nodes_original[item]
        else:
             coords_for_dist = node_coords
             
        dist = pdist(coords_for_dist, metric='euclidean')
        
        if ew_type == 'EUC_2D':
            dist_matrix = squareform(np.floor(dist + 0.5))
        elif ew_type == 'CEIL_2D':
            dist_matrix = squareform(np.ceil(dist))
        else:
            dist_matrix = squareform(dist)

        # From list to tensors as a DotDict
        item_dict = DotDict()
        item_dict.dist_matrices = torch.Tensor(dist_matrix)
        item_dict.nodes_coord = torch.Tensor(node_coords)
        if self.tour_lens is not None:
            item_dict.tour_len = self.tour_lens[item]
        else:
            item_dict.tour_len = torch.Tensor([])
            
        if self.names is not None:
            item_dict.name = self.names[item]

        return item_dict


def load_dataset(filename, batch_size, shuffle=False, what="test"):
    # data = np.load(filename)

    tsplib_problems, tsplib_cost, tsplib_name, tsplib_ew_type = make_tsplib_data(filename)
    nodes_tmp = []
    nodes_original = []
    factors = []
    for i in range(len(tsplib_problems)):
        tmp0 = np.array(tsplib_problems[i])
        tmp1 = tmp0
        tmp2 = tmp0[[0]]
        tmp3 = np.concatenate((tmp1, tmp2), axis=0)
        
        nodes_original.append(tmp3)
        
        factors.append(np.max(tmp3) - np.min(tmp3))
        tmp3 = (tmp3 - np.min(tmp3))/(np.max(tmp3)-np.min(tmp3))


        nodes_tmp.append(tmp3)
    print(factors)
    data = {}

    data['coords'] = nodes_tmp

    data["tour_lens"] = tsplib_cost


    if what == "train":
        assert data["reorder"]

    tour_lens = data["tour_lens"] if "tour_lens" in data.keys() else None

    # Do not use collate function in test dataset
    collate_fn = collate_func_with_sample_suffix if what == "train" else None

    dataset = DataLoader(DataSet(data["coords"], tour_lens=tour_lens, edge_weight_types=tsplib_ew_type, nodes_original=nodes_original, names=tsplib_name), batch_size=batch_size,
                         drop_last=False, shuffle=shuffle, collate_fn=collate_fn)
    return dataset,factors


def collate_func_with_sample_suffix(l_dataset_items):
    """
    assemble minibatch out of dataset examples.
    For instances of TOUR-TSP of graph size N (i.e. nb_nodes=N+1 including return to beginning node),
    this function also takes care of sampling a SUB-problem (PATH-TSP) of size 3 to N+1
    """
    nb_nodes = len(l_dataset_items[0].nodes_coord)
    subproblem_size = np.random.randint(4, nb_nodes + 1)
    begin_idx = nb_nodes + 1 - subproblem_size
    l_dataset_items_new = prepare_dataset_items(l_dataset_items, begin_idx, subproblem_size)
    return default_collate(l_dataset_items_new)

def prepare_dataset_items(l_dataset_items, begin_idx, subproblem_size):
    l_dataset_items_new = []
    for d in l_dataset_items:
        d_new = {}
        for k, v in d.items():
            if type(v) == numpy.float64:
                v_ = 0.
            elif len(v.shape) == 1 or k == 'nodes_coord':
                v_ = v[begin_idx:begin_idx+subproblem_size, ...]
            else:
                v_ = v[begin_idx:begin_idx+subproblem_size, begin_idx:begin_idx+subproblem_size]
            d_new.update({k+'_s': v_})
        l_dataset_items_new.append({**d, **d_new})
    return l_dataset_items_new


def sample_subproblem(nb_nodes):
    subproblem_size = np.random.randint(4, nb_nodes + 1)  # between _ included and nb_nodes + 1 excluded
    begin_idx = np.random.randint(nb_nodes - subproblem_size + 1)
    return begin_idx, subproblem_size

def make_tsplib_data(filename):
    import ast
    node_coords = []
    costs = []
    names = []
    edge_weight_types = []

    for line in open(filename, "r").readlines():
        line = line.strip()
        parsed = False
        try:
             # Try new format
             data_list = ast.literal_eval(line)
             if isinstance(data_list, list) and 'name' in data_list:
                 
                 names.append(str(data_list[data_list.index('name') + 1]))
                 costs.append(float(data_list[data_list.index('cost') + 1]))
                 
                 ew = "Exact"
                 if 'edge_weight_type' in data_list:
                     ew = data_list[data_list.index('edge_weight_type') + 1]
                 edge_weight_types.append(ew)

                 start_idx = data_list.index('customer') + 1
                 if 'end' in data_list:
                     end_idx = data_list.index('end')
                 else:
                     end_idx = len(data_list)
                 
                 coords = np.array(data_list[start_idx:end_idx], dtype=float).reshape(-1, 2)
                 node_coords.append(coords)
                 parsed = True
        except:
             pass
        
        if not parsed:
            # Fallback to old format
            line = line.replace('[', '').replace(']', '').replace("'", "")
            parts = line.split(',')
            names.append(parts[0])
            costs.append(float(parts[1]))
            edge_weight_types.append('Exact')
            node_coords.append(np.array(parts[2:], dtype=float).reshape(-1, 2))
        
    return np.array(node_coords, dtype=object), np.array(costs), np.array(names), np.array(edge_weight_types)