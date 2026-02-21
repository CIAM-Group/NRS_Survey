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


def vectorized_insert_update(partial_solution_, remaining_cap_, insert_points_, insert_position_, insert_demands_,
                             full_capacity_):
    partial_solution = partial_solution_.clone().detach()
    remaining_cap = remaining_cap_.clone().detach()
    insert_points = insert_points_.clone().detach()
    insert_position = insert_position_.clone().detach()
    insert_demands = insert_demands_.clone().detach()
    full_capacity = full_capacity_.detach()

    # remaining_cap = , insert_points, insert_position, insert_demands,
    # full_capacity
    """
    矢量化地插入新点并更新剩余容量

    参数:
        partial_solution: 当前部分解 [batch_size, seq_len]
        remaining_cap: 当前剩余容量 [batch_size, seq_len]
        insert_points: 要插入的点 [batch_size]
        insert_position: 插入位置索引 [batch_size]
        insert_demands: 插入点的需求 [batch_size]
        full_capacity: 车辆的满载容量

    返回:
        new_sol: 更新后的部分解
        new_cap: 更新后的剩余容量
    """
    batch_size, seq_len = partial_solution.shape
    device = partial_solution.device

    def insert(batch_size, seq_len, insert_position, insert_points, partial_solution, device):
        insert_position = insert_position + 1
        # 生成扩展索引模板
        index_offset = torch.arange(batch_size, device=device) * (seq_len + 1)
        pos_expanded = insert_position + index_offset

        # 创建索引模板 [batch_size, seq_len+1]
        base_idx = torch.arange(seq_len + 1, device=device).repeat(batch_size, 1) + index_offset.view(-1, 1)
        mask = base_idx >= pos_expanded.view(-1, 1)
        new_indices = base_idx - mask.long()

        # 构建扩展后的张量
        expanded_sol = torch.cat([
            partial_solution,
            torch.zeros((batch_size, 1), device=device, dtype=partial_solution.dtype)
        ], dim=1).flatten()

        # 执行并行插入
        new_sol = torch.take_along_dim(expanded_sol, new_indices.flatten(), 0)
        new_sol = new_sol.view(batch_size, seq_len + 1)
        new_sol.scatter_(1, insert_position.unsqueeze(1), insert_points.unsqueeze(1))

        return new_sol

    new_sol = insert(batch_size, seq_len, insert_position, insert_points, partial_solution, device)

    # print('new_sol\n', new_sol)

    index1 = torch.arange(batch_size)[:, None]

    # print('remaining_cap\n', remaining_cap)
    # print('insert_position\n', insert_position)

    insert_tour_remaining_cap = remaining_cap[index1, insert_position[:, None]]

    # print(insert_tour_remaining_cap)

    # print('insert_tour_remaining_cap 1111\n', insert_tour_remaining_cap)
    # print('remaining_cap 1111\n', remaining_cap)

    # insert_tour_remaining_cap_positions = insert_tour_remaining_cap == remaining_cap

    # # 使用 == 操作符生成布尔张量
    # equal_int = insert_tour_remaining_cap_positions.int()
    # # 找到连续片段的起始和结束位置
    # diff = torch.diff(equal_int.squeeze())
    # starts = torch.nonzero(diff == 1).squeeze() + 1  # 连续片段起始位置
    # if starts.shape[0]>batch_size:

    fianl_node_index = insert_position==(remaining_cap.shape[1]-1)

    not_fianl_node_index = insert_position!=(remaining_cap.shape[1]-1)

    insert_tour_remaining_cap_positions = torch.zeros(size=(batch_size,remaining_cap.shape[1] ),dtype=torch.long)
    insert_tour_remaining_cap_positions[fianl_node_index,-1]=1

    # print(insert_tour_remaining_cap)
    # print(not_fianl_node_index)
    # print(insert_tour_remaining_cap[not_fianl_node_index])

    if not_fianl_node_index.any():
        # print(insert_tour_remaining_cap)
        # print(not_fianl_node_index)
        # print(insert_tour_remaining_cap[not_fianl_node_index])
        #
        # print(insert_tour_remaining_cap_positions[not_fianl_node_index].shape)
        # print(filtrate(remaining_cap[not_fianl_node_index],
        #                                                insert_tour_remaining_cap[not_fianl_node_index].reshape(batch_size,1),
        #                                                insert_position[not_fianl_node_index][:, None],
        #                                                partial_solution[not_fianl_node_index])[not_fianl_node_index].shape)
        #
        # print(insert_tour_remaining_cap.shape, not_fianl_node_index.shape)
        insert_tour_remaining_cap_positions[not_fianl_node_index] = filtrate(
            remaining_cap[not_fianl_node_index], insert_tour_remaining_cap[not_fianl_node_index].reshape(-1,1),
                                                       insert_position[not_fianl_node_index][:, None],
                                                       partial_solution[not_fianl_node_index])#[not_fianl_node_index]

    # print('insert_tour_remaining_cap_positions\n', insert_tour_remaining_cap_positions.long())
    #
    # print('insert_tour_remaining_cap')
    # print(insert_tour_remaining_cap)
    # print(insert_demands)

    insert_tour_remaining_cap_positions2 = insert_tour_remaining_cap_positions.clone().detach().long()
    # print('insert_tour_remaining_cap_positions2\n', insert_tour_remaining_cap_positions2.long())

    insert_tour_remaining_cap_positions2[insert_tour_remaining_cap_positions2.eq(0)] = -2

    # print('insert_tour_remaining_cap_positions2\n', insert_tour_remaining_cap_positions2)
    #

    remain_caps = insert_tour_remaining_cap - insert_demands[:, None]
    # print('remain_caps',remain_caps)
    remain_caps2 = remain_caps * insert_tour_remaining_cap_positions2

    # print('================ \n')
    #
    # print('insert_tour_remaining_cap_positions 1111\n', insert_tour_remaining_cap_positions)
    #
    # print('insert_tour_remaining_cap_positions2 1111\n', insert_tour_remaining_cap_positions2)
    #
    # print('remaining_cap 0000\n', remaining_cap)
    #
    # print('remain_caps2 1111\n', remain_caps2)

    remain_caps2[insert_tour_remaining_cap_positions2.lt(0)] = -1

    remaining_cap[insert_tour_remaining_cap_positions.long().eq(1)] = remain_caps2[remain_caps2 >= 0].long()

    # print('remaining_cap\n', remaining_cap)
    #
    # print('================ \n')

    new_cap = insert(batch_size, seq_len, insert_position, remain_caps.ravel().long(), remaining_cap.long(), device)

    # print('new_cap\n', new_cap)

    if (new_sol[:, -1] != 0).any():
        new_sol = torch.cat((new_sol, torch.zeros(size=(batch_size, 1), device=device)), dim=1)
        new_cap = torch.cat((new_cap, torch.ones(size=(batch_size, 1), device=device) * full_capacity), dim=1)

    return new_sol, new_cap


def manhattan_distance(x, y):
    # x: [B,seq,2]
    # y: [B, 1, 2]
    difference = torch.abs(x - y).sum(2)

    return difference


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


def node_flag_tran_to_along_cap(node_flag, cap_flag, full_capacity):
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
    store_3 = store_2.clone().detach()

    # print('store_2 \n', store_2)
    #
    # print('store_3 \n', store_3)

    store_2[store_2.gt(0.1)] = node.ravel()

    store_3[store_3.gt(0.1)] = cap_flag[:, :, 0].ravel()

    # print('store_3 \n', store_3)

    store_4 = torch.roll(store_3, shifts=-1, dims=1)

    # print('store_4 \n', store_4)

    store_3 = torch.where(store_3 == 0, store_4, store_3)

    # print('store_3 \n', store_3)

    zeros = torch.zeros(size=(batch_size, 1), dtype=torch.long)
    ones = torch.ones(size=(batch_size, 1), dtype=torch.long) * full_capacity

    result = torch.cat((store_2, zeros), dim=1)

    cap_list = torch.cat((store_3, ones), dim=1)

    # print('result \n', result)
    # print('cap_list \n', cap_list)

    return result, cap_list


def tran_to_node_flag_along_cap(node_list, remaining_cap):
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
    save_remaining_cap = remaining_cap[save_index].reshape(batch_size, -1)
    save_flag = flag_index[save_index].reshape(batch_size, -1)

    node_flag_1 = torch.cat((save_node.unsqueeze(2), save_flag.unsqueeze(2)), dim=2)
    save_remaining_cap_flag = torch.cat((save_remaining_cap.unsqueeze(2), save_flag.unsqueeze(2)), dim=2)
    return node_flag_1, save_remaining_cap_flag


def cal_remaining_capacity(problem, solution, capacity=30):
    # solution = torch.tensor([[[1, 1], [2, 0], [3, 1], [4, 0], [5, 0], [6, 0]],
    #                          [[7, 1], [6, 0], [3, 0], [2, 1], [1, 0], [8, 0]],
    #                          [[5, 1], [6, 0], [7, 1], [4, 0], [3, 1], [2, 0]],
    #                          [[5, 1], [6, 0], [7, 1], [4, 0], [3, 1], [2, 0]],
    #                          ])

    # print('1 ---------- problem.shape', problem.shape)
    # print('2 ---------- solution.shape', solution.shape)
    # print('3 ---------- solution \n', solution)

    solution_size = solution.shape[1]

    # coor = problem[:, :, [0, 1]]

    demand = problem[:, :, 2]

    order_flag = solution[:, :, 1].clone()

    batch_size = solution.shape[0]

    visit_depot_num = torch.sum(solution[:, :, 1], dim=1)

    # all_subtour_num = torch.sum(visit_depot_num)

    # print('4 ---------- visit_depot_num \n', visit_depot_num)
    #
    # print('5 ---------- all_subtour_num \n', all_subtour_num)

    ###############
    ###### 找到 max subtour length

    # fake_solution = torch.cat((solution[:, :, 1], torch.ones(batch_size)[:, None]), dim=1)

    # print('6 ---------- fake_solution \n', fake_solution)

    # start_from_depot = fake_solution.nonzero()

    # print('7 ---------- start_from_depot \n', start_from_depot)

    # start_from_depot_1 = start_from_depot[:, 1]

    # print('8 ---------- start_from_depot_1 \n', start_from_depot_1)

    # start_from_depot_2 = torch.roll(start_from_depot_1, shifts=-1)

    # print('9 ---------- start_from_depot_2 \n', start_from_depot_2)

    # sub_tours_length = start_from_depot_2 - start_from_depot_1

    # print('10 ---------- sub_tours_length \n', sub_tours_length)

    # max_subtour_length = torch.max(sub_tours_length)

    # print('11 ---------- max_subtour_length \n', max_subtour_length)

    # print('##############################')
    # print('##############################')
    # print('##############################')

    start_from_depot2 = solution[:, :, 1].nonzero()

    # print('12 ---------- start_from_depot2 \n', start_from_depot2)

    start_from_depot3 = solution[:, :, 1].roll(shifts=-1, dims=1).nonzero()

    # print('13 ---------- start_from_depot3 \n', start_from_depot3)

    repeat_solutions_node = solution[:, :, 0].repeat_interleave(visit_depot_num, dim=0)

    # print('14 ---------- repeat_solutions_node \n', repeat_solutions_node)

    double_repeat_solution_node = repeat_solutions_node  # .repeat(1, 2)

    # print('15 ---------- double_repeat_solution_node \n', double_repeat_solution_node)

    x1 = torch.arange(double_repeat_solution_node.shape[1])[None, :].repeat(len(repeat_solutions_node),
                                                                            1) >= start_from_depot2[:, 1][:, None]

    # print('16 ---------- x1 \n', x1.long())

    x2 = torch.arange(double_repeat_solution_node.shape[1])[None, :].repeat(len(repeat_solutions_node),
                                                                            1) <= start_from_depot3[:, 1][:, None]

    # print('17 ---------- x2 \n', x2.long())

    x3 = (x1 * x2).long()

    # print('18 ---------- x3 \n', x3.long())

    sub_tourss = double_repeat_solution_node * x3

    # print('19 ---------- sub_tourss \n', sub_tourss)

    demands = torch.repeat_interleave(demand, repeats=visit_depot_num, dim=0)

    demands_ = get_encoding(demands.unsqueeze(2), sub_tourss).squeeze(2)

    demands_total_per_subtour = demands_.sum(1).unsqueeze(1)

    demands_total_per_subtour = demands_total_per_subtour * x3

    demands_total_per_subtour_ = demands_total_per_subtour[demands_total_per_subtour.ge(0.5)].reshape(batch_size,
                                                                                                      solution_size)

    remaining_capacitys = capacity - demands_total_per_subtour_

    remaining_capacitys_flag = torch.cat((remaining_capacitys.unsqueeze(2), order_flag.unsqueeze(2)), dim=2).long()

    remaining_capacitys_edges = node_flag_tran_to_(remaining_capacitys_flag)

    remaining_capacitys_edges_shift = torch.roll(remaining_capacitys_edges, dims=1, shifts=-1)

    index = remaining_capacitys_edges.eq(0)

    remaining_capacitys_edges[index] = remaining_capacitys_edges_shift[index]

    remaining_capacitys_edges[:, -1] = capacity
    # 等于0的话，”depot的demand=0“ 与 “该路径的remaining capacity” 混合起来了。

    return remaining_capacitys_edges


def get_encoding(encoded_nodes, node_index_to_pick):
    batch_size = node_index_to_pick.size(0)
    pomo_size = node_index_to_pick.size(1)
    embedding_dim = encoded_nodes.size(2)

    gathering_index = node_index_to_pick[:, :, None].expand(batch_size, pomo_size, embedding_dim)

    picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)

    return picked_nodes


def generate_label(index_gobal, random_index, padding_solution, padding_abs_partial_solu_2, abs_scatter_solu_1,
                   batch_size_V, problem_size):
    # print('batch_size_V',batch_size_V)
    # print('1 ----------- random_index \n',random_index)
    # print('2 ----------- padding_solution \n', padding_solution)
    # print('2 ----------- padding_solution_size \n', problem_size)
    # print('3 ----------- padding_abs_partial_solu_2 \n', padding_abs_partial_solu_2)
    # print('4 ----------- abs_scatter_solu_1 \n', abs_scatter_solu_1)

    # ####################################
    # # 把 abs_partial_solu_2 对齐到 solution
    # # (1) 找到 solution 中第一个点在 abs_partial_solu_2 中的位置。
    #
    # print(padding_solution[3])
    # print(padding_abs_partial_solu_2[3])
    # assert False
    #
    # solution_first_node = padding_solution[:,1]
    # print('----- solution_first_node ', solution_first_node[0])
    #
    # tmp1 = padding_abs_partial_solu_2
    # tmp2 = solution_first_node.unsqueeze(1)
    #
    # print(tmp1.shape)
    # print(tmp2.shape)
    # tmp3 = torch.arange(padding_abs_partial_solu_2.shape[1])[None,:].repeat(batch_size_V,1)
    # print(tmp1 == tmp2)
    # tmp4 = tmp3[tmp1 == tmp2]
    #
    # print(tmp4)
    # assert False
    #
    # ####################################

    ####################################
    # 1. abs_scatter_solu_1_seleted
    ####################################
    abs_scatter_solu_1_seleted = abs_scatter_solu_1[index_gobal, random_index]

    # print('5 ----------- abs_scatter_solu_1_seleted \n', abs_scatter_solu_1_seleted)

    ####################################
    # 2. abs_scatter_solu_1_unseleted
    ####################################
    index1 = torch.arange(abs_scatter_solu_1.shape[1])[None, :].repeat(batch_size_V, 1)

    tmp1 = (index1 < random_index).long()

    tmp2 = (index1 > random_index).long()

    tmp3 = tmp1 + tmp2

    abs_scatter_solu_1_unseleted = abs_scatter_solu_1[tmp3.gt(0.5)].reshape(batch_size_V,
                                                                            abs_scatter_solu_1.shape[1] - 1)

    # print('6 ---------------------- abs_scatter_solu_1_unseleted \n', abs_scatter_solu_1_unseleted)

    ####################################
    # 3. expanded_partial_solution
    ####################################
    num_scatter_unseleted = abs_scatter_solu_1_unseleted.shape[1]

    tmp1 = padding_solution.unsqueeze(1).repeat_interleave(repeats=num_scatter_unseleted, dim=1)

    tmp2 = abs_scatter_solu_1_unseleted.unsqueeze(2)

    tmp3 = tmp1 == tmp2

    index_1 = torch.arange(problem_size, dtype=torch.long)[None, :].repeat(batch_size_V, 1).unsqueeze(1). \
        repeat(1, num_scatter_unseleted, 1)

    index_2 = index_1[tmp3].reshape(batch_size_V, num_scatter_unseleted)

    new_list = padding_solution.clone().detach()

    new_list_len = problem_size - num_scatter_unseleted  # shape: [B, V-current_step]

    index_3 = torch.arange(batch_size_V, dtype=torch.long)[:, None].expand(batch_size_V, index_2.shape[1])

    new_list[index_3, index_2] = -2

    expanded_partial_solution = new_list[torch.gt(new_list, -1)].view(batch_size_V, new_list_len)

    # 使用高级索引来实现并行 roll 操作, 防止每一行前面出现两个0
    expanded_partial_solution_len = expanded_partial_solution.shape[1]

    index_col = torch.arange(expanded_partial_solution_len, dtype=torch.long)[None, :]
    bigger_than_0_index = index_col.repeat(batch_size_V, 1)[expanded_partial_solution > 0].reshape(batch_size_V, -1)

    shift_index = bigger_than_0_index[:, 0] - 1
    # 创建批量索引

    index = index_col + shift_index[:, None]
    index[index.gt(expanded_partial_solution_len - 0.5)] = index[index.gt(
        expanded_partial_solution_len - 0.5)] - expanded_partial_solution_len

    expanded_partial_solution = expanded_partial_solution[torch.arange(batch_size_V)[:, None], index]

    # print('7 ---------------------- expanded_partial_solution \n', expanded_partial_solution)

    ####################################
    # 4. rela_label
    ####################################
    tmp4 = abs_scatter_solu_1_seleted == expanded_partial_solution

    index_1 = index_col.repeat(batch_size_V, 1)

    index_2 = index_1[tmp4].reshape(batch_size_V, 1)
    index_3 = index_2 - 1
    index_3_2 = index_2 + 1
    # index_3_3 = index_2 + 2

    index4 = torch.arange(batch_size_V)[:, None]
    abs_teacher_index_before = expanded_partial_solution[index4, index_3]
    abs_teacher_index_after = expanded_partial_solution[index4, index_3_2]
    # abs_teacher_index_after2 = expanded_partial_solution[index4, index_3_3]
    # print(abs_teacher_index)

    # -----------------
    # print(' ------------- abs_teacher_index_before \n',abs_teacher_index_before)
    # print(' ------------- abs_teacher_index_after \n',abs_teacher_index_after)
    # print('3 ----------- padding_abs_partial_solu_2 \n', padding_abs_partial_solu_2)
    # 在这里遇到问题了，就是如果要选的边的第一个点是depot，但是一个解中又有很多depot，就没有一个唯一性,而且有时候甚至会遇到一条边有两个0.。。。

    judge1 = abs_teacher_index_before == padding_abs_partial_solu_2
    judge2 = abs_teacher_index_after == torch.roll(padding_abs_partial_solu_2, dims=[1], shifts=-1)

    # judge3 = abs_teacher_index_after2 == torch.roll(padding_abs_partial_solu_2,dims=[1],shifts=-2)
    # print('------------- judge1')
    # print(judge1.long())
    # print(judge2.long())

    index1 = torch.arange(judge1.shape[1])[None, :].repeat(batch_size_V, 1)
    index1[judge1.eq(1)] = 0
    index2 = torch.roll(index1, dims=1, shifts=-1)
    substract_index1 = index1 - index2
    mask1 = torch.ones(size=(batch_size_V, padding_abs_partial_solu_2.shape[1]))
    mask1[substract_index1.eq(0)] = 0
    mask1[:, -1] = 1
    tmp5 = mask1 * judge1

    index3 = torch.arange(judge1.shape[1])[None, :].repeat(batch_size_V, 1)
    index3[judge2.eq(1)] = 0
    index4 = torch.roll(index3, dims=1, shifts=-1)
    substract_index2 = index3 - index4
    mask2 = torch.ones(size=(batch_size_V, padding_abs_partial_solu_2.shape[1]))
    mask2[substract_index2.eq(0)] = 0
    mask2[:, -1] = 1
    tmp6 = mask2 * judge2

    tmp7 = tmp5 + tmp6

    # print('------------tmp7 \n', tmp7)

    # 如果 tmp5 的某个位置的元素大于1，则说明要插入这个位置。如果是最后一个位置，说明要回depot然后从depot抵达。

    index_1 = torch.arange(padding_abs_partial_solu_2.shape[1], dtype=torch.long)[None, :].repeat(batch_size_V, 1)

    index_2 = index_1[tmp7.ge(2)].reshape(batch_size_V, 1)
    rela_label = index_2

    '''
    如果 rela_label[i] == padding_abs_partial_solu_2.shape[1],
    那么第 i 个 instance 的这一步的插入点要回 depot
    '''
    # print('8 ---------------------- rela_label \n', rela_label)

    return rela_label, expanded_partial_solution, abs_scatter_solu_1_seleted, abs_scatter_solu_1_unseleted


def extend_partial_solution_def(rela_selected, padding_abs_partial_solu_2, abs_scatter_solu_1_seleted, batch_size_V):
    # print('1 ---------------- rela_selected \n',rela_selected)
    # print('2 ---------------- padding_abs_partial_solu_2 \n', padding_abs_partial_solu_2)
    # print('3 ---------------- abs_scatter_solu_1_seleted \n', abs_scatter_solu_1_seleted)

    # 分为两种情况，一种是直接和TSP一样插入一个点
    # 另一种情况是选择了最后一个点，这样的话，说明这个点是要回depot然后从depot抵达的，这种情况下，整个partial solution的padding都要变（增加1个）

    # 先按照第一种情况插入点。

    num_abs_partial_solu_2 = padding_abs_partial_solu_2.shape[1]

    temp_extend_solution = -torch.ones(num_abs_partial_solu_2 + 1)[None, :].repeat(batch_size_V, 1)

    temp_extend_solution = temp_extend_solution.long()

    index1 = torch.arange(num_abs_partial_solu_2 + 1)[None, :].repeat(batch_size_V, 1)

    tmp1 = (index1 <= rela_selected).long()

    tmp2 = (index1 > rela_selected + 1).long()

    tmp3 = tmp1 + tmp2

    temp_extend_solution[tmp3.gt(0.5)] = padding_abs_partial_solu_2.ravel()

    # 这一步是要把被insert的点放在 temp_extend_solution 的 rela_selected+1 这个index
    index3 = torch.arange(batch_size_V)[:, None]

    temp_extend_solution[index3, rela_selected + 1] = abs_scatter_solu_1_seleted

    # print('4 ---------------- temp_extend_solution \n', temp_extend_solution)

    judge = rela_selected == padding_abs_partial_solu_2.shape[1] - 1

    # print('5 ---------------- judge \n', judge)

    if judge.any():
        zero_tmp = torch.zeros(size=(batch_size_V, 1), dtype=torch.long)
        temp_extend_solution = torch.cat((temp_extend_solution, zero_tmp), dim=1)

    # print('6 ---------------- temp_extend_solution \n', temp_extend_solution)
    return temp_extend_solution


def probs_to_selected_nodes(probs_, split_line_, batch_size_):
    selected_node_student_ = probs_.argmax(dim=1)  # shape: B
    is_via_depot_student_ = selected_node_student_ >= split_line_  # Nodes with an index greater than customer_num are via depot
    not_via_depot_student_ = selected_node_student_ < split_line_

    selected_flag_student_ = torch.zeros(batch_size_, dtype=torch.int)
    selected_flag_student_[is_via_depot_student_] = 1
    selected_node_student_[is_via_depot_student_] = selected_node_student_[
                                                        is_via_depot_student_] - split_line_ + 1
    selected_flag_student_[not_via_depot_student_] = 0
    selected_node_student_[not_via_depot_student_] = selected_node_student_[not_via_depot_student_] + 1
    return selected_node_student_, selected_flag_student_  # node 的 index 从 1 开始


def get_new_data(data, selected_node_list, prob_size, B_V):
    list = selected_node_list

    new_list = torch.arange(prob_size)[None, :].repeat(B_V, 1)

    new_list_len = prob_size - list.shape[1]  # shape: [B, V-current_step]

    index_2 = list.type(torch.long)

    index_1 = torch.arange(B_V, dtype=torch.long)[:, None].expand(B_V, index_2.shape[1])

    new_list[index_1, index_2] = -2

    unselect_list = new_list[torch.gt(new_list, -1)].view(B_V, new_list_len)

    new_data = data

    emb_dim = data.shape[-1]

    new_data_len = new_list_len

    index_2_ = unselect_list.repeat_interleave(repeats=emb_dim, dim=1)

    index_1_ = torch.arange(B_V, dtype=torch.long)[:, None].expand(B_V, index_2_.shape[1])

    index_3_ = torch.arange(emb_dim)[None, :].repeat(repeats=(B_V, new_data_len))

    new_data_ = new_data[index_1_, index_2_, index_3_].view(B_V, new_data_len, emb_dim)

    return new_data_


def drawPic_VRP_v1(coor_, solution, abs_partial_solu_2, abs_scatter_solu_1, partial_end_node_coor,
                   scatter_node_coors, abs_scatter_solu_1_seleted, name='xx', optimal_tour_=None):
    # coor: shape (V,2)
    # order_node_: shape (V)
    # order_sflag_: shape (V)
    # state.data[0], self.env.abs_partial_solu_2[0],self.env.abs_scatter_solu_1[0],
    #                            partial_end_node_coor[0],scatter_node_coors,

    coor = coor_.clone().cpu().numpy()
    solution = solution.clone().cpu().numpy()
    abs_partial_solu_2 = abs_partial_solu_2.clone().cpu().numpy()
    abs_scatter_solu_1 = abs_scatter_solu_1.clone().cpu().numpy()
    partial_end_node_coor = partial_end_node_coor.clone().cpu().numpy()
    scatter_node_coors = scatter_node_coors.clone().cpu().numpy()

    arr_max = np.max(coor)
    arr_min = np.min(coor)
    arr = (coor - arr_min) / (arr_max - arr_min)

    fig, ax = plt.subplots(figsize=(20, 20))

    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.axis('off')

    plt.scatter(arr[:, 0], arr[:, 1], color='gray', linewidth=2)
    # Depot
    plt.scatter(arr[0, 0], arr[0, 1], color='red', linewidth=15, marker='v')

    order_node = solution[:, 0]
    order_flag = solution[:, 1]

    # col_counter = order_flag.sum()
    #
    # colors = plt.cm.turbo(np.linspace(0, 1, col_counter))  # turbo
    # np.random.seed(123)
    # np.random.shuffle(colors)

    tour = []
    for i in range(len(order_node)):
        if order_flag[i] == 1:
            tour.append(0)
            tour.append(order_node[i])
        if order_flag[i] == 0:
            tour.append(order_node[i])
    # 连接optimal solution的各个点
    count = -1
    tour = np.array(tour, dtype=int)
    for i in range(len(tour) - 1):
        if tour[i] == 0:
            count += 1
        start = [arr[tour[i], 0], arr[tour[i + 1], 0]]
        end = [arr[tour[i], 1], arr[tour[i + 1], 1]]
        # plt.plot(start, end, color=colors[count], linewidth=3)
        plt.plot(start, end, color='pink', linewidth=3)
    start = [arr[0, 0], arr[tour[-1], 0]]
    end = [arr[0, 1], arr[tour[-1], 1]]
    # plt.plot(start, end, color=colors[count], linewidth=3)
    plt.plot(start, end, color='pink', linewidth=3)

    order_node_partial = abs_partial_solu_2[:, 0]
    order_flag_partial = abs_partial_solu_2[:, 1]
    tour_partial = []
    for i in range(len(order_node_partial)):
        if order_flag_partial[i] == 1:
            tour_partial.append(0)
            tour_partial.append(order_node_partial[i])
        if order_flag_partial[i] == 0:
            tour_partial.append(order_node_partial[i])

    count = -1
    tour_partial = np.array(tour_partial, dtype=int)
    for i in range(len(tour_partial) - 1):
        if tour_partial[i] == 0:
            count += 1
        start = [arr[tour_partial[i], 0], arr[tour_partial[i + 1], 0]]
        end = [arr[tour_partial[i], 1], arr[tour_partial[i + 1], 1]]
        # plt.plot(start, end, color=colors[count], linewidth=3)
        plt.plot(start, end, color='blue', linewidth=3)
    start = [arr[0, 0], arr[tour_partial[-1], 0]]
    end = [arr[0, 1], arr[tour_partial[-1], 1]]
    # plt.plot(start, end, color=colors[count], linewidth=3)
    plt.plot(start, end, color='blue', linewidth=3)

    print(tour_partial)

    plt.scatter(arr[tour_partial[0], 0], arr[tour_partial[0], 1], color='orange', linewidth=15, marker='o')

    # print(abs_scatter_solu_1_seleted)
    # assert False
    # # print(partial_end_node_coor.shape)
    plt.scatter(arr[abs_scatter_solu_1_seleted[0], 0], arr[abs_scatter_solu_1_seleted[0], 1], color='green',
                linewidth=15, marker='o')

    # print(order_node)
    # print(order_flag)
    #
    # print(order_node_partial)
    # print(order_flag_partial)

    # 连接 partial solution 的各个点
    #

    b = os.path.abspath(".")
    path = b + '/figure_train'

    def make_dir(path_destination):
        isExists = os.path.exists(path_destination)
        if not isExists:
            os.makedirs(path_destination)
        return

    make_dir(path)

    plt.savefig(path + f'/{name}.pdf', bbox_inches='tight', pad_inches=0)

    # assert False
    # plt.show()


def drawPic_VRP_v2(coor_, solution,
                   expanded_padding_partial_solution, partial_end_node_coor,
                   abs_scatter_solu_1_unseleted, abs_scatter_solu_1_seleted,
                   name='xx'):
    coor = coor_.clone().cpu().numpy()
    solution = solution.clone().cpu().numpy()
    expanded_padding_partial_solution = expanded_padding_partial_solution.clone().cpu().numpy()
    abs_scatter_solu_1_unseleted = abs_scatter_solu_1_unseleted.clone().cpu().numpy()
    abs_scatter_solu_1_seleted = abs_scatter_solu_1_seleted.clone().cpu().numpy()
    partial_end_node_coor = partial_end_node_coor.clone().cpu().numpy()

    # arr_max = np.max(coor)
    # arr_min = np.min(coor)
    # arr = (coor - arr_min) / (arr_max - arr_min)

    arr = coor

    fig, ax = plt.subplots(figsize=(20, 20))

    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.axis('off')

    plt.scatter(arr[:, 0], arr[:, 1], color='gray', linewidth=2)
    # Depot
    plt.scatter(arr[0, 0], arr[0, 1], color='red', linewidth=15, marker='v')

    order_node = solution[:, 0]
    order_flag = solution[:, 1]

    # col_counter = order_flag.sum()
    #
    # colors = plt.cm.turbo(np.linspace(0, 1, col_counter))  # turbo
    # np.random.seed(123)
    # np.random.shuffle(colors)

    tour = []
    for i in range(len(order_node)):
        if order_flag[i] == 1:
            tour.append(0)
            tour.append(order_node[i])
        if order_flag[i] == 0:
            tour.append(order_node[i])
    # 连接optimal solution的各个点
    count = -1
    tour = np.array(tour, dtype=int)
    for i in range(len(tour) - 1):
        if tour[i] == 0:
            count += 1
        start = [arr[tour[i], 0], arr[tour[i + 1], 0]]
        end = [arr[tour[i], 1], arr[tour[i + 1], 1]]
        # plt.plot(start, end, color=colors[count], linewidth=3)
        plt.plot(start, end, color='pink', linewidth=3)
    start = [arr[0, 0], arr[tour[-1], 0]]
    end = [arr[0, 1], arr[tour[-1], 1]]
    # plt.plot(start, end, color=colors[count], linewidth=3)
    plt.plot(start, end, color='pink', linewidth=3)

    tour_partial = expanded_padding_partial_solution

    count = -1
    tour_partial = np.array(tour_partial, dtype=int)
    for i in range(len(tour_partial) - 1):
        if tour_partial[i] == 0:
            count += 1
        start = [arr[tour_partial[i], 0], arr[tour_partial[i + 1], 0]]
        end = [arr[tour_partial[i], 1], arr[tour_partial[i + 1], 1]]
        # plt.plot(start, end, color=colors[count], linewidth=3)
        plt.plot(start, end, color='blue', linewidth=3)
    start = [arr[0, 0], arr[tour_partial[-1], 0]]
    end = [arr[0, 1], arr[tour_partial[-1], 1]]
    # plt.plot(start, end, color=colors[count], linewidth=3)
    plt.plot(start, end, color='blue', linewidth=3)

    print(tour_partial)

    plt.scatter(arr[tour_partial[0], 0], arr[tour_partial[0], 1], color='orange', linewidth=15, marker='o')

    # print(abs_scatter_solu_1_seleted)
    # assert False
    # # print(partial_end_node_coor.shape)
    plt.scatter(arr[abs_scatter_solu_1_seleted[0], 0], arr[abs_scatter_solu_1_seleted[0], 1], color='green',
                linewidth=15, marker='o')

    # print(order_node)
    # print(order_flag)
    #
    # print(order_node_partial)
    # print(order_flag_partial)

    # 连接 partial solution 的各个点
    #

    b = os.path.abspath(".")
    path = b + '/figure_train'

    def make_dir(path_destination):
        isExists = os.path.exists(path_destination)
        if not isExists:
            os.makedirs(path_destination)
        return

    make_dir(path)

    plt.savefig(path + f'/{name}.pdf', bbox_inches='tight', pad_inches=0)

    # assert False
    # plt.show()

def ploar_distance(scatter_coor, depot_coor, last_selecte_scatter_coor):
    def to_polar(batch_points, batch_origin):
        dx = batch_points[:, :, 0] - batch_origin[:, :, 0]
        dy = batch_points[:, :, 1] - batch_origin[:, :, 1]
        # r = torch.sqrt(dx ** 2 + dy ** 2)

        theta = torch.atan2(dy, dx)
        return theta

    # 计算所有点的极坐标
    theta = to_polar(scatter_coor, depot_coor)

    # 参考点的极坐标
    reference_theta = to_polar(last_selecte_scatter_coor, depot_coor)

    # 计算角度差值
    angle_diff = torch.abs(theta - reference_theta)

    # 找到最小角度差值对应的点
    _, min_index = torch.min(angle_diff, dim=1)

    return min_index


def ploar_distance2(scatter_coor, depot_coor, last_selecte_scatter_coor):
    def to_polar(batch_points, batch_origin):
        dx = batch_points[:, :, 0] - batch_origin[:, :, 0]
        dy = batch_points[:, :, 1] - batch_origin[:, :, 1]
        # r = torch.sqrt(dx ** 2 + dy ** 2)

        theta = torch.atan2(dy, dx)
        return theta

    # 计算所有点的极坐标
    theta = to_polar(scatter_coor, depot_coor)

    # 参考点的极坐标
    reference_theta = to_polar(last_selecte_scatter_coor, depot_coor)

    # 计算角度差值
    angle_diff = torch.abs(theta - reference_theta)
    # angle_diff = theta - reference_theta

    return angle_diff


def Rearrange_solution_clockwise(problem, solution):
    # self.valida_solution_legal(problem,solution)
    # 1. 确定每个subtour的质心
    #   1.1 提取出每个subtour
    #   1.2 对每个subtour求质心
    # 2. 交换位置
    #   2.1 按照每个subtour的质心来进行邻接排序
    #   2.2 确定一个基准质心
    #   2.3 以depot为原点，算其他每个质心与该基准质心的夹角
    #   2.4 按照夹角的大小排序
    #   2.5 将排序后的 subtours 转化为原来的 flag形式

    #   1.1 提取出每个subtour
    # 对于每条子路径，单独拿出来，pandding 0 至长度 max_subtour_length
    # 对于每个 instance， 把子路径个数 padding 0 至数目 max_subtour_num
    # 把所有instance的所有子路径 cat 到同一数组，

    problem_size = solution.shape[1]
    coor = problem[:, :, [0, 1]].clone()
    order_node = solution[:, :, 0]
    order_flag = solution[:, :, 1]
    # print(order_node[0])
    # print(order_flag[0])
    # 1.
    # 找到每个instance有几条子路径，
    # 所有instance中子路径总数目是多少     all_subtour_num，
    # 所有instance中子路径中最长长度是多少  max_subtour_length
    batch_size = solution.shape[0]

    visit_depot_num = torch.sum(order_flag, dim=1)

    all_subtour_num = torch.sum(visit_depot_num)

    fake_solution = torch.cat((order_flag, torch.ones(batch_size)[:, None]), dim=1)

    start_from_depot = fake_solution.nonzero()

    start_from_depot_1 = start_from_depot[:, 1]

    start_from_depot_2 = torch.roll(start_from_depot_1, shifts=-1)

    sub_tours_length = start_from_depot_2 - start_from_depot_1

    max_subtour_length = torch.max(sub_tours_length)

    start_from_depot2 = order_flag.nonzero()

    start_from_depot3 = order_flag.roll(shifts=-1, dims=1).nonzero()

    repeat_solutions_node = order_node.repeat_interleave(visit_depot_num, dim=0)
    double_repeat_solution_node = repeat_solutions_node.repeat(1, 2)

    x1 = torch.arange(double_repeat_solution_node.shape[1])[None, :].repeat(len(repeat_solutions_node), 1) \
         >= start_from_depot2[:, 1][:, None]
    x2 = torch.arange(double_repeat_solution_node.shape[1])[None, :].repeat(len(repeat_solutions_node), 1) \
         <= start_from_depot3[:, 1][:, None]

    x3 = (x1 * x2).long()

    sub_tourss = double_repeat_solution_node * x3

    x4 = torch.arange(double_repeat_solution_node.shape[1])[None, :].repeat(len(repeat_solutions_node), 1) \
         < (start_from_depot2[:, 1][:, None] + max_subtour_length)

    x5 = x1 * x4

    # print(sub_tourss.shape)

    sub_tours_padding = sub_tourss[x5].reshape(all_subtour_num, max_subtour_length)
    subtour_lengths = (sub_tours_padding > 1).int().sum(1)

    # print('sub_tours_padding.shape',sub_tours_padding.shape)
    # print(sub_tours_padding)

    #   1.2 对每个subtour求中心

    repeated_coor = torch.repeat_interleave(coor, repeats=visit_depot_num, dim=0)
    depot_coor = repeated_coor[:, [0], :].clone()
    repeated_coor[:, 0, :] = 0
    # print(repeated_coor.shape)
    # assert False
    subtours_coor = repeated_coor.gather(dim=1, index=sub_tours_padding[:, :, None].repeat(1, 1, 2))
    subtours_coor = torch.cat((subtours_coor, depot_coor), dim=1)
    subtours_coor_sum = torch.sum(subtours_coor, dim=1)
    subtours_centroid = subtours_coor_sum / (subtour_lengths + 1)[:, None]
    subtours_centroid_total_num = subtours_centroid.shape[0]
    # print(subtours_centroid_total_num)
    # self.drawPic_VRP(self.problems[0, :, :2], self.solution[0, :, 0], self.solution[0, :, 1], name=f'subtours_centroid',
    #                  optimal_tour_=None,subtours_centroid=subtours_centroid)

    #   2.1 按照每个subtour的中心来进行邻接排序

    temp_tri = np.triu(np.ones((len(visit_depot_num), len(visit_depot_num))), k=1)
    visit_depot_num_numpy = visit_depot_num.clone().cpu().numpy()
    temp_index = np.dot(visit_depot_num_numpy, temp_tri)

    # 每一元素对应一个instance的第一个subtour的位置  tensor([ 0, 12, 25, 38, 51])
    temp_index_1 = torch.from_numpy(temp_index).long().cuda()

    # 每一元素对应一个instance的最后一个subtour的位置   tensor([12, 25, 38, 51, 64])
    temp_index_2 = visit_depot_num + temp_index_1

    # print('temp_index_1 \n',temp_index_1)
    # print('temp_index_2 \n',temp_index_2)

    x1 = torch.arange(subtours_centroid_total_num)[None, :].repeat(batch_size, 1) >= temp_index_1[:, None]
    x2 = torch.arange(subtours_centroid_total_num)[None, :].repeat(batch_size, 1) < temp_index_2[:, None]
    x3_ = (x1 * x2).int()
    x3 = x3_[:, :, None].repeat(1, 1, 2)

    # print('x3 shape',x3.shape)

    subtours_centroid_repeat = subtours_centroid[None, :, :].repeat(batch_size, 1, 1)

    # print('subtours_centroid_repeat shape', subtours_centroid_repeat.shape)

    subtours_centroid_sperate = subtours_centroid_repeat * x3
    # print('subtours_centroid_sperate[0] \n',subtours_centroid_sperate[0])
    # print(subtours_centroid_sperate[4])
    # print('subtours_centroid_sperate.shape',subtours_centroid_sperate.shape)

    #   2.2 确定一个基准质心： 即排在第一的subtour的质心
    index2 = temp_index_1.clone().unsqueeze(1).unsqueeze(2).repeat(1, 1, 2)

    based_centroids = subtours_centroid_sperate.gather(dim=1, index=index2)
    # print('based_centroids.shape',based_centroids.shape)
    #
    # print('repeated depot_coor shape',depot_coor.shape)
    # 2.2.1 所有的质心都减去depot的位置

    # print(depot_coor.shape)
    single_depot_coor = coor[:, [0], :]
    # print('single_depot_coor',single_depot_coor.shape)
    # print('single_depot_coor[0] \n',single_depot_coor[0])
    repeated_depot_coor = coor[:, [0], :].repeat(1, all_subtour_num, 1)

    # print('repeated_depot_coor',repeated_depot_coor.shape)

    all_centroid_depot_vectors = subtours_centroid_sperate - repeated_depot_coor

    # print('all_centroid_depot_vectors ',all_centroid_depot_vectors.shape)

    based_centroid_depot_vectors = based_centroids - single_depot_coor

    # print('based_centroid_depot_vectors.shape',based_centroid_depot_vectors.shape)

    repeated_based_centroid_depot_vectors = based_centroid_depot_vectors.repeat(1, all_subtour_num, 1)

    # print('repeated_based_centroid_depot_vectors.shape',repeated_based_centroid_depot_vectors.shape)
    # theta = arccos( x1*x2 / |x1||x2| )
    #   2.3 以depot为原点，算其他每个质心与该基准质心的夹角
    x1_times_x2 = (repeated_based_centroid_depot_vectors * all_centroid_depot_vectors).sum(2)

    x1_module_length = torch.sqrt((repeated_based_centroid_depot_vectors ** 2).sum(2))
    x2_module_length = torch.sqrt((all_centroid_depot_vectors ** 2).sum(2))

    # print('x1_times_x2 shape', x1_times_x2.shape,'x1_module_length shape',x1_module_length.shape,'x2_module_length shape',x2_module_length.shape)
    cos_value = x1_times_x2 / (x1_module_length * x2_module_length)
    cos_value[cos_value.ge(1)] = 1 - 1e-5
    cos_value[cos_value.le(-1)] = -1 + 1e-5  # arccos只接受[1,-1]范围内的值，防止出现 nan
    cross_value = np.cross(repeated_based_centroid_depot_vectors.cpu().numpy(),
                           all_centroid_depot_vectors.cpu().numpy())
    # print(cross_value[0])
    cross_value = torch.tensor(cross_value)
    negtivate_sign_2 = torch.ones(size=(cross_value.shape[0], cross_value.shape[1]))
    negtivate_sign_2[cross_value.lt(0)] = -1
    # print(negtivate_sign_2[0])
    # print('cos_value[0] \n',cos_value[0])
    theta_value = torch.arccos(cos_value)  # 3.1415为pi，即180°
    theta_value = torch.where(torch.isnan(theta_value), torch.full_like(theta_value, 2 * 3.1415926), theta_value)
    theta_value = negtivate_sign_2 * theta_value
    # print('theta_value shape',theta_value.shape)
    # print('theta_value[0]',theta_value[0])
    theta_value[theta_value.lt(0)] += 2 * 3.1415926
    # print('theta_value[0]',theta_value[0])
    # print(theta_value.shape)
    # x4 = torch.arange(subtours_centroid_total_num)[None, :].repeat(batch_size, 1) < temp_index_1[:, None]
    # x5 = torch.arange(subtours_centroid_total_num)[None, :].repeat(batch_size, 1) >= temp_index_2[:, None]
    # x6 = (x4 * x5)
    # print(x6.shape)
    # print(x6[0])

    #   2.4 按照夹角的大小排序 subtour
    theta_value[x3_.le(0)] = 6 * 3.1415926
    theta_value_sort_value, theta_value_sort_index = torch.sort(theta_value, dim=1)
    # print('theta_value.shape ',theta_value.shape)
    # print(theta_value_sort_value[0])
    # print(theta_value_sort_index[0])
    repeated_sub_tours_padding = sub_tours_padding.unsqueeze(0).repeat(batch_size, 1, 1)
    # print('repeated_sub_tours_padding.shape ',repeated_sub_tours_padding.shape)
    # print(repeated_sub_tours_padding[0])
    # for i in range(batch_size):
    #     temp3 = repeated_sub_tours_padding[i,i*102:(i+1)*102,:]
    #     print('temp3.gt(0).shape ',torch.unique(temp3[temp3.gt(0)]).shape)
    gather_theta_value_sort_index = theta_value_sort_index.unsqueeze(2).repeat(1, 1, max_subtour_length)

    resort_repeated_sub_tours_padding = repeated_sub_tours_padding.gather(dim=1, index=gather_theta_value_sort_index)

    # print(resort_repeated_sub_tours_padding[0])
    # print('resort_repeated_sub_tours_padding.shape',resort_repeated_sub_tours_padding.shape)
    # print('visit_depot_num.shape', visit_depot_num)
    # print(resort_repeated_sub_tours_padding.shape)
    # test_sub_tours_padding = sub_tours_padding.reshape(batch_size,102,-1)
    # for i in range(batch_size):
    #     temp = resort_repeated_sub_tours_padding[i,i*102:(i+1)*102,:]
    #     temp2 = test_sub_tours_padding[i]
    #     print('temp.gt(0).shape ',torch.unique(temp[temp.gt(0)]).shape,torch.unique(temp2[temp2.gt(0)]).shape)

    x4 = torch.arange(all_subtour_num)[None, :].repeat(batch_size, 1)
    # print(visit_depot_num)
    x5 = (x4 < visit_depot_num[:, None]).int()
    x6 = x5.unsqueeze(2).repeat(1, 1, max_subtour_length)

    resort_repeated_sub_tours_padding = resort_repeated_sub_tours_padding * x6

    resort_repeated_sub_tours_padding = resort_repeated_sub_tours_padding.reshape(batch_size, -1)

    # for i in range(batch_size):
    #     temp = resort_repeated_sub_tours_padding[i]
    #     print(temp[temp.gt(0)].shape)

    resort_sub_tours = resort_repeated_sub_tours_padding[resort_repeated_sub_tours_padding.gt(0)].reshape(batch_size,
                                                                                                          -1)

    # print(resort_sub_tours.shape)
    # print(sub_tours_length)

    repeated_sub_tours_length = sub_tours_length[sub_tours_length.gt(0)].unsqueeze(0).repeat(batch_size, 1)

    resort_repeated_sub_tours_length = repeated_sub_tours_length.gather(dim=1, index=theta_value_sort_index)
    resort_repeated_sub_tours_length = resort_repeated_sub_tours_length * x5
    max_subtour_number = visit_depot_num.max()
    # print(max_subtour_number)
    resort_repeated_sub_tours_length = resort_repeated_sub_tours_length[:, :max_subtour_number]
    # print(resort_repeated_sub_tours_length)

    # 把 resort_repeated_sub_tours_length 转化为 flag:

    temp_tri = np.triu(np.ones((batch_size, max_subtour_number.item(), max_subtour_number.item())), k=1)
    resort_repeated_sub_tours_length_numpy = resort_repeated_sub_tours_length.clone().cpu().numpy()
    temp_index = np.dot(resort_repeated_sub_tours_length_numpy, temp_tri)
    temp_index_1 = torch.from_numpy(temp_index).long().cuda()
    index1 = torch.arange(batch_size)
    temp_index_1 = temp_index_1[index1, index1]
    temp_index_1[temp_index_1.ge(problem_size)] = 0

    flag = torch.zeros(size=(batch_size, problem_size), dtype=torch.int)
    index1 = torch.arange(batch_size)[:, None].repeat(1, max_subtour_number)

    flag[index1, temp_index_1] = 1

    #   2.5 将排序后的 subtours 转化为原来的 flag形式

    # for k in range(batch_size):
    #     print(k)
    #     self.drawPic_VRP(self.problems[k, :, :2], resort_sub_tours[k], flag[k],
    #                      name=f'resortd_before_{k}', optimal_tour_=None)
    # assert False
    solution_ = torch.cat((resort_sub_tours.unsqueeze(2), flag.unsqueeze(2)), dim=2)
    return solution_


def filtrate_back(remaining_cap, current_cap, position, partial_solution):
    # remaining_cap： [batch, V]
    # current_cap: [batch, 1]
    # position: [batch, 1]

    print('remaining_cap \n', remaining_cap)
    print('current_cap \n', current_cap)
    print('partial_solution\n', partial_solution)

    print()

    # 定义张量 a 和 b
    # a = torch.tensor([[4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    #                    5, 5, 5, 5, 1, 1, 1, 1, 1, 1, 1, 22, 22, 22, 1, 1, 1, 1,
    #                    1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 22, 22, 22, 22, 40],
    #                   [4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    #                    5, 5, 5, 5, 1, 1, 1, 1, 1, 1, 1, 22, 22, 22, 1, 1, 1, 1,
    #                    1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 22, 22, 22, 40]
    #                   ])
    # b = torch.tensor([[22],[22],])
    batch_size = remaining_cap.shape[0]
    # 使用 == 操作符生成布尔张量
    equal_tensor = (remaining_cap == current_cap)

    # print(equal_tensor.long())

    #
    # # 将布尔张量转换为整数（方便处理连续片段）
    equal_int = equal_tensor.int()

    # 找到连续片段的起始和结束位置
    diff = torch.diff(equal_int.squeeze())

    diff = diff.reshape(batch_size,-1)

    diff2 = diff.clone().detach()

    print(diff.shape, batch_size)
    diff = torch.cat((torch.zeros(size=(batch_size, 1)), diff), dim=1)

    diff[diff.sum(1) <= -1, 0] = 1

    diff2 = torch.cat((diff2, torch.zeros(size=(batch_size, 1)),), dim=1)

    diff2[diff.sum(1) >= 1, -1] = -1

    # print(diff)
    # print(diff2)

    #
    starts = torch.nonzero(diff == 1)#.squeeze()  # 连续片段起始位置
    ends = torch.nonzero(diff2 == -1)#.squeeze()  # 连续片段结束位置

    # print(starts)
    # print(ends)
    # # assert False
    # print(starts.shape, ends.shape)

    # # 如果以 1 开始或以 1 结束，修正起始或结束位置
    # if equal_int[0, 0] == 1:
    #     starts = torch.cat((torch.tensor([0]), starts))
    # if equal_int[0, -1] == 1:
    #     ends = torch.cat((ends, torch.tensor([equal_int.size(1)])))

    # print(starts)
    starts_num = starts[:, 0]
    # print(starts_num)

    unbviques, repeat_num = torch.unique(starts_num, return_counts=True)
    # print(unbviques)
    # print(repeat_num)
    # print(repeat_num.shape, position.shape)

    target_index = torch.repeat_interleave(position, repeats=repeat_num, dim=0)

    # print(target_index)

    # 将起始和结束位置组合成范围

    # print(starts)
    # print(ends)
    # print(starts.shape, ends.shape)
    ranges = torch.stack((starts, ends), dim=1)
    ranges_store = ranges.clone().detach()
    # print(ranges)
    #
    # print(ranges.shape)
    # print(ranges[:,:,1].shape)
    ranges[:, :, 1] = ranges[:, :, 1] - target_index

    index = ranges[:, 0, 1] * ranges[:, 1, 1]

    # print(index)
    #
    #
    # print(ranges.shape)

    ranges = ranges_store[index <= 0]

    # print(ranges)

    start = ranges[:, 0, 1]
    end = ranges[:, 1, 1]
    #
    # print(start)
    #
    # print(end)

    zzz = torch.arange(diff2.shape[1])[None, :].repeat(remaining_cap.shape[0], 1)

    x = zzz >= start[:, None]

    y = zzz <= end[:, None]

    # xy = (x * y)[:,1:]
    xy = (x * y)

    return xy
    # print(xy.long())
    # batch_size = remaining_cap.shape[0]
    # # 使用 == 操作符生成布尔张量
    # equal_tensor = (remaining_cap == current_cap)
    #
    # # print(equal_tensor.long())
    #
    #
    # #
    # # # 将布尔张量转换为整数（方便处理连续片段）
    # equal_int = equal_tensor.int()
    #
    # # 找到连续片段的起始和结束位置
    # diff = torch.diff(equal_int.squeeze())
    #
    # print(diff)
    #
    #
    # diff2 = diff.clone().detach()
    #
    # diff = torch.cat((torch.zeros(size=(batch_size, 1)),diff ), dim=1)
    #
    # diff[diff.sum(1)<=-1,0]=1
    #
    #
    # diff2 = torch.cat((diff2, torch.zeros(size=(batch_size, 1)),),dim=1)
    #
    # diff2[(diff==-1).sum(1)==0,-1]=-1
    #
    # print(diff)
    # #
    # starts = torch.nonzero(diff == 1).squeeze() + 1  # 连续片段起始位置
    # ends = torch.nonzero(diff2 == -1).squeeze() + 1  # 连续片段结束位置
    #
    #
    #
    # print(starts.shape, ends.shape)
    #
    # # # 如果以 1 开始或以 1 结束，修正起始或结束位置
    # # if equal_int[0, 0] == 1:
    # #     starts = torch.cat((torch.tensor([0]), starts))
    # # if equal_int[0, -1] == 1:
    # #     ends = torch.cat((ends, torch.tensor([equal_int.size(1)])))
    #
    # print(starts)
    # starts_num = starts[:, 0]
    # print(starts_num)
    #
    # unbviques, repeat_num = torch.unique(starts_num, return_counts=True)
    # print(unbviques)
    # print(repeat_num)
    # print(repeat_num.shape, position.shape )
    #
    # target_index = torch.repeat_interleave(position, repeats=repeat_num, dim=0)
    #
    # # print(target_index)
    #
    # # 将起始和结束位置组合成范围
    #
    # print(starts)
    # print(ends)
    # print(starts.shape, ends.shape)
    # ranges = torch.stack((starts, ends), dim=1)
    # ranges_store = ranges.clone().detach()
    # # print(ranges)
    # #
    # # print(ranges.shape)
    # # print(ranges[:,:,1].shape)
    # ranges[:, :, 1] = ranges[:, :, 1] - target_index
    #
    # index = ranges[:, 0, 1] * ranges[:, 1, 1]
    #
    # # print(index)
    # #
    # #
    # # print(ranges.shape)
    #
    # ranges = ranges_store[index <= 0]
    #
    # # print(ranges)
    #
    # start = ranges[:, 0, 1]
    # end = ranges[:, 1, 1]
    # #
    # # print(start)
    # #
    # # print(end)
    #
    # zzz = torch.arange(diff2.shape[1])[None, :].repeat(remaining_cap.shape[0], 1)
    #
    # x = zzz >= start[:, None]
    #
    # y = zzz < end[:, None]
    #
    # # xy = (x * y)[:,1:]
    # xy = (x * y)
    #
    # # print(xy.long())
    #
    # return xy.long()


def filtrate(remaining_cap, current_cap, position, partial_solution):
    # remaining_cap： [batch, V]
    # current_cap: [batch, 1]
    # position: [batch, 1]
    batch_size = partial_solution.shape[0]
    # print('batch_size \n', batch_size)
    # print('position \n', position)
    index1 = torch.arange(1, partial_solution.shape[1] + 1)[None, :].repeat(batch_size, 1)

    index1[partial_solution != 0] = 0

    # print('index1 \n', index1)

    solution = tran_to_node_flag(partial_solution)

    visit_depot_num = index1.gt(0).sum(1) # torch.sum(solution[:, :, 1], dim=1)

    # print(visit_depot_num)

    all_subtour_num = torch.sum(visit_depot_num)

    # print(all_subtour_num)

    index0 = index1.nonzero()

    # print(all_subtour_num)

    # print('index0 \n', index0)

    start_from_depot = index1.nonzero()[:, 1]

    # print('start_from_depot \n', start_from_depot)
    #
    # print('start_from_depot.shape \n', start_from_depot.shape)

    start_from_depot2 = start_from_depot[None, :].repeat(batch_size, 1)
    start_from_depot3 = torch.roll(start_from_depot2, shifts=-1)  # start_from_depot[1:][None,:].repeat(batch_size,1)

    # print('start_from_depot2 \n', start_from_depot2)
    # print('start_from_depot3 \n',start_from_depot3)

    x1 = start_from_depot2 <= position

    # print('x1 \n', x1)

    x2 = start_from_depot3 > position

    # print('x2 \n', x2)

    x3 = (x1 * x2).long()
    # print('x3 \n', x3)

    cum_visit_depot_num = torch.cumsum(visit_depot_num, dim=0)
    visit_depot_num2 = torch.cat((torch.tensor([0]), cum_visit_depot_num[:-1]), dim=0)

    zzz = torch.arange(all_subtour_num)[None, :].repeat(batch_size, 1)

    # print(visit_depot_num2,visit_depot_num)

    z1 = (zzz >= visit_depot_num2[:, None])
    # print(z1)
    z2 = (zzz < cum_visit_depot_num[:, None])
    # print(z2)
    mask = z1 * z2

    # print('x3 \n', x3)
    #
    # print('mask \n', mask)
    #
    # print('x3 \n', x3.shape)
    #
    # print('mask \n', mask.shape)

    index2 = x3 * mask

    # print(index2)

    index3 = index2.sum(dim=0)
    # print('index3 \n', index3)

    begin = index0[index3.eq(1)]
    end = torch.roll(index0, shifts=-1, dims=0)[index3.eq(1)]

    # print('begin \n', begin)
    # print(end)

    begin = begin[:, [1]]

    end = end[:, [1]]

    zzz = torch.arange(partial_solution.shape[1])[None, :].repeat(batch_size, 1)

    # print('zzz \n', zzz)

    index5 = (zzz >= begin) * (zzz < end)

    # print('begin \n', begin)
    #
    # print(index5.long())

    return index5.long()



    # print('remaining_cap \n', remaining_cap)
    # print('current_cap \n', current_cap)
    # print('partial_solution\n', partial_solution)
    #
    # print()
    #
    # # 定义张量 a 和 b
    # # a = torch.tensor([[4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    # #                    5, 5, 5, 5, 1, 1, 1, 1, 1, 1, 1, 22, 22, 22, 1, 1, 1, 1,
    # #                    1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 22, 22, 22, 22, 40],
    # #                   [4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    # #                    5, 5, 5, 5, 1, 1, 1, 1, 1, 1, 1, 22, 22, 22, 1, 1, 1, 1,
    # #                    1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 22, 22, 22, 40]
    # #                   ])
    # # b = torch.tensor([[22],[22],])
    # batch_size = remaining_cap.shape[0]
    # # 使用 == 操作符生成布尔张量
    # equal_tensor = (partial_solution == 0)
    #
    # print(equal_tensor.long())
    #
    # #
    # # # 将布尔张量转换为整数（方便处理连续片段）
    # equal_int = equal_tensor.int()
    #
    # # 找到连续片段的起始和结束位置
    # diff = torch.diff(equal_int.squeeze())
    #
    # diff = diff.reshape(batch_size, -1)
    #
    # diff2 = diff.clone().detach()
    #
    # print(diff.shape, batch_size)
    # diff = torch.cat((torch.zeros(size=(batch_size, 1)), diff), dim=1)
    #
    # diff[diff.sum(1) <= -1, 0] = 1
    #
    # diff2 = torch.cat((diff2, torch.zeros(size=(batch_size, 1)),), dim=1)
    #
    # diff2[diff.sum(1) >= 1, -1] = -1
    #
    # # print(diff)
    # # print(diff2)
    #
    # #
    # starts = torch.nonzero(diff == 1)  # .squeeze()  # 连续片段起始位置
    # ends = torch.nonzero(diff2 == -1)  # .squeeze()  # 连续片段结束位置
    #
    # # print(starts)
    # # print(ends)
    # # # assert False
    # # print(starts.shape, ends.shape)
    #
    # # # 如果以 1 开始或以 1 结束，修正起始或结束位置
    # # if equal_int[0, 0] == 1:
    # #     starts = torch.cat((torch.tensor([0]), starts))
    # # if equal_int[0, -1] == 1:
    # #     ends = torch.cat((ends, torch.tensor([equal_int.size(1)])))
    #
    # # print(starts)
    # starts_num = starts[:, 0]
    # # print(starts_num)
    #
    # unbviques, repeat_num = torch.unique(starts_num, return_counts=True)
    # # print(unbviques)
    # # print(repeat_num)
    # # print(repeat_num.shape, position.shape)
    #
    # target_index = torch.repeat_interleave(position, repeats=repeat_num, dim=0)
    #
    # # print(target_index)
    #
    # # 将起始和结束位置组合成范围
    #
    # # print(starts)
    # # print(ends)
    # # print(starts.shape, ends.shape)
    # ranges = torch.stack((starts, ends), dim=1)
    # ranges_store = ranges.clone().detach()
    # # print(ranges)
    # #
    # # print(ranges.shape)
    # # print(ranges[:,:,1].shape)
    # ranges[:, :, 1] = ranges[:, :, 1] - target_index
    #
    # index = ranges[:, 0, 1] * ranges[:, 1, 1]
    #
    # # print(index)
    # #
    # #
    # # print(ranges.shape)
    #
    # ranges = ranges_store[index <= 0]
    #
    # # print(ranges)
    #
    # start = ranges[:, 0, 1]
    # end = ranges[:, 1, 1]
    # #
    # # print(start)
    # #
    # # print(end)
    #
    # zzz = torch.arange(diff2.shape[1])[None, :].repeat(remaining_cap.shape[0], 1)
    #
    # x = zzz >= start[:, None]
    #
    # y = zzz <= end[:, None]
    #
    # # xy = (x * y)[:,1:]
    # xy = (x * y)
    #
    # return xy
