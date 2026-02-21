import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import random

def set_seeds(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def node_flag_tran_to_(node_flag):
    '''
    :param node_list: [B, V, 2]
    :return: [B, V+n]
    '''

    batch_size = node_flag.shape[0]
    problem_size = node_flag.shape[1]
    node = node_flag[:, :, 0]
    flag = node_flag[:, :, 1]
    depot_num = flag.sum(1)

    max_length = torch.max(depot_num)

    store_1 = torch.ones(size=(batch_size, problem_size + max_length), dtype=torch.long)

    where_is_depot_0, where_is_depot_1 = torch.where(flag == 1)

    temp1 = torch.arange(max_length)[None, :].repeat(batch_size, 1)
    temp2 = temp1 < depot_num[:, None]
    temp3 = temp1[temp2]
    where_is_depot_1 = where_is_depot_1 + temp3

    store_1[where_is_depot_0, where_is_depot_1] = 0

    mask = torch.arange(problem_size + max_length)[None, :].repeat(batch_size, 1)
    nodesss = problem_size + depot_num
    mask2 = (mask < nodesss[:, None]).long()
    store_2 = store_1 * mask2

    store_2[store_2.gt(0.1)] = node.ravel()

    zeros = torch.zeros(size=(batch_size, 1), dtype=torch.long)

    result = torch.cat((store_2, zeros), dim=1)

    return result


def tran_to_node_flag(node_list):
    '''
    :param node_list: [B, V+n]
    :return: [B, V, 2]
    '''

    batch_size = node_list.shape[0]

    index_smaller_0_shift = torch.roll(torch.le(node_list, 0), shifts=1, dims=1).long()
    index_bigger_0 = torch.gt(node_list, 0).long()

    flag_index = index_smaller_0_shift * index_bigger_0

    save_index = torch.gt(node_list, 0.1)

    save_node = node_list[save_index].reshape(batch_size, -1)
    save_flag = flag_index[save_index].reshape(batch_size, -1)

    node_flag_1 = torch.cat((save_node.unsqueeze(2), save_flag.unsqueeze(2)), dim=2)

    return node_flag_1

