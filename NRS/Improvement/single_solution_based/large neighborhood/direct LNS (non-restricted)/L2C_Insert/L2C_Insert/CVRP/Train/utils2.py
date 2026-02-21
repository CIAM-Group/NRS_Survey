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

def cal_remaining_capacity(problem, solution, capacity=30):

    solution_size = solution.shape[1]

    # coor = problem[:, :, [0, 1]]

    demand = problem[:, :, 2]

    order_flag = solution[:, :, 1].clone()

    batch_size = solution.shape[0]

    visit_depot_num = torch.sum(solution[:, :, 1], dim=1)

    start_from_depot2 = solution[:, :, 1].nonzero()

    start_from_depot3 = solution[:, :, 1].roll(shifts=-1, dims=1).nonzero()

    repeat_solutions_node = solution[:, :, 0].repeat_interleave(visit_depot_num, dim=0)

    double_repeat_solution_node = repeat_solutions_node #.repeat(1, 2)

    x1 = torch.arange(double_repeat_solution_node.shape[1])[None, :].repeat(len(repeat_solutions_node),
                                                                            1) >= start_from_depot2[:, 1][:, None]

    x2 = torch.arange(double_repeat_solution_node.shape[1])[None, :].repeat(len(repeat_solutions_node),
                                                                            1) <= start_from_depot3[:, 1][:, None]

    x3 = (x1 * x2).long()

    # print('18 ---------- x3 \n', x3.long())

    sub_tourss = double_repeat_solution_node * x3

    # print('19 ---------- sub_tourss \n', sub_tourss)


    demands = torch.repeat_interleave(demand, repeats=visit_depot_num, dim=0)

    # print('20 ---------- demands \n',demands.shape)

    demands_ = get_encoding(demands.unsqueeze(2),sub_tourss).squeeze(2)

    # print('21 ---------- demands_ \n',demands_)

    demands_total_per_subtour = demands_.sum(1).unsqueeze(1)
    # print('22 ---------- demands_total_per_subtour \n', demands_total_per_subtour)

    demands_total_per_subtour = demands_total_per_subtour * x3

    # print('23 ---------- demands_total_per_subtour \n', demands_total_per_subtour)

    demands_total_per_subtour_ = demands_total_per_subtour[demands_total_per_subtour.ge(0.5)].reshape(batch_size, solution_size)

    remaining_capacitys = capacity - demands_total_per_subtour_

    remaining_capacitys_flag = torch.cat((remaining_capacitys.unsqueeze(2), order_flag.unsqueeze(2)),dim=2).long()

    remaining_capacitys_edges = node_flag_tran_to_(remaining_capacitys_flag)

    remaining_capacitys_edges_shift = torch.roll(remaining_capacitys_edges,dims=1,shifts=-1)

    index = remaining_capacitys_edges.eq(0)

    remaining_capacitys_edges[index] = remaining_capacitys_edges_shift[index]

    remaining_capacitys_edges[:,-1] = capacity

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

    ####################################
    # 1. abs_scatter_solu_1_seleted
    ####################################
    abs_scatter_solu_1_seleted = abs_scatter_solu_1[index_gobal, random_index]

    ####################################
    # 2. abs_scatter_solu_1_unseleted
    ####################################
    index1 = torch.arange(abs_scatter_solu_1.shape[1])[None, :].repeat(batch_size_V, 1)

    tmp1 = (index1 < random_index).long()

    tmp2 = (index1 > random_index).long()

    tmp3 = tmp1 + tmp2

    abs_scatter_solu_1_unseleted = abs_scatter_solu_1[tmp3.gt(0.5)].reshape(batch_size_V,
                                                                            abs_scatter_solu_1.shape[1] - 1)

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

    expanded_partial_solution_len = expanded_partial_solution.shape[1]

    index_col = torch.arange(expanded_partial_solution_len, dtype=torch.long)[None, :]
    bigger_than_0_index = index_col.repeat(batch_size_V, 1)[expanded_partial_solution > 0].reshape(batch_size_V, -1)

    shift_index = bigger_than_0_index[:, 0] - 1

    index = index_col + shift_index[:, None]
    index[index.gt(expanded_partial_solution_len - 0.5)] = index[index.gt(
        expanded_partial_solution_len - 0.5)] - expanded_partial_solution_len

    expanded_partial_solution = expanded_partial_solution[torch.arange(batch_size_V)[:, None], index]

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

    judge1 = abs_teacher_index_before == padding_abs_partial_solu_2
    judge2 = abs_teacher_index_after == torch.roll(padding_abs_partial_solu_2, dims=[1], shifts=-1)

    index1 = torch.arange(judge1.shape[1])[None,:].repeat(batch_size_V,1)
    index1[judge1.eq(1)] = 0
    index2 = torch.roll(index1,dims=1,shifts=-1)
    substract_index1 = index1 - index2
    mask1 = torch.ones(size=(batch_size_V, padding_abs_partial_solu_2.shape[1]))
    mask1[substract_index1.eq(0)] = 0
    mask1[:, -1] = 1
    tmp5 = mask1*judge1

    index3 = torch.arange(judge1.shape[1])[None,:].repeat(batch_size_V,1)
    index3[judge2.eq(1)] = 0
    index4 = torch.roll(index3, dims=1, shifts=-1)
    substract_index2 = index3 - index4
    mask2 = torch.ones(size=(batch_size_V, padding_abs_partial_solu_2.shape[1]))
    mask2[substract_index2.eq(0)] = 0
    mask2[:, -1] = 1
    tmp6 = mask2 * judge2

    tmp7 = tmp5+tmp6

    index_1 = torch.arange(padding_abs_partial_solu_2.shape[1], dtype=torch.long)[None, :].repeat(batch_size_V, 1)

    index_2 = index_1[tmp7.ge(2)].reshape(batch_size_V, 1)
    rela_label = index_2

    return rela_label, expanded_partial_solution, abs_scatter_solu_1_seleted, abs_scatter_solu_1_unseleted

def extend_partial_solution_def(rela_selected, padding_abs_partial_solu_2, abs_scatter_solu_1_seleted, batch_size_V):


    num_abs_partial_solu_2 = padding_abs_partial_solu_2.shape[1]

    temp_extend_solution = -torch.ones(num_abs_partial_solu_2 + 1)[None,:].repeat(batch_size_V,1)

    temp_extend_solution = temp_extend_solution.long()

    index1 = torch.arange(num_abs_partial_solu_2+1)[None,:].repeat(batch_size_V,1)

    tmp1 = (index1 <= rela_selected).long()

    tmp2 = (index1 > rela_selected + 1).long()

    tmp3 = tmp1 + tmp2

    temp_extend_solution[tmp3.gt(0.5)] = padding_abs_partial_solu_2.ravel()

    index3 = torch.arange(batch_size_V)[:,None]

    temp_extend_solution[index3,rela_selected+1] = abs_scatter_solu_1_seleted


    judge = rela_selected==padding_abs_partial_solu_2.shape[1] - 1

    if judge.any():
        zero_tmp = torch.zeros(size=(batch_size_V,1),dtype=torch.long)
        temp_extend_solution = torch.cat((temp_extend_solution,zero_tmp),dim=1)


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
                   scatter_node_coors,abs_scatter_solu_1_seleted, name='xx', optimal_tour_=None):

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

    plt.scatter(arr[abs_scatter_solu_1_seleted[0], 0], arr[abs_scatter_solu_1_seleted[0], 1], color='green', linewidth=15, marker='o')

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

    plt.scatter(arr[abs_scatter_solu_1_seleted[0], 0], arr[abs_scatter_solu_1_seleted[0], 1], color='green',
                linewidth=15, marker='o')

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


    problem_size = solution.shape[1]
    coor = problem[:, :, [0, 1]].clone()
    order_node = solution[:, :, 0]
    order_flag = solution[:, :, 1]

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


    sub_tours_padding = sub_tourss[x5].reshape(all_subtour_num, max_subtour_length)
    subtour_lengths = (sub_tours_padding > 1).int().sum(1)

    repeated_coor = torch.repeat_interleave(coor, repeats=visit_depot_num, dim=0)
    depot_coor = repeated_coor[:, [0], :].clone()
    repeated_coor[:, 0, :] = 0

    subtours_coor = repeated_coor.gather(dim=1, index=sub_tours_padding[:, :, None].repeat(1, 1, 2))
    subtours_coor = torch.cat((subtours_coor, depot_coor), dim=1)
    subtours_coor_sum = torch.sum(subtours_coor, dim=1)
    subtours_centroid = subtours_coor_sum / (subtour_lengths + 1)[:, None]
    subtours_centroid_total_num = subtours_centroid.shape[0]

    temp_tri = np.triu(np.ones((len(visit_depot_num), len(visit_depot_num))), k=1)
    visit_depot_num_numpy = visit_depot_num.clone().cpu().numpy()
    temp_index = np.dot(visit_depot_num_numpy, temp_tri)

    temp_index_1 = torch.from_numpy(temp_index).long().cuda()

    temp_index_2 = visit_depot_num + temp_index_1

    x1 = torch.arange(subtours_centroid_total_num)[None, :].repeat(batch_size, 1) >= temp_index_1[:, None]
    x2 = torch.arange(subtours_centroid_total_num)[None, :].repeat(batch_size, 1) < temp_index_2[:, None]
    x3_ = (x1 * x2).int()
    x3 = x3_[:,:,None].repeat(1,1,2)

    subtours_centroid_repeat = subtours_centroid[None,:,:].repeat(batch_size,1,1)

    subtours_centroid_sperate = subtours_centroid_repeat * x3

    index2 = temp_index_1.clone().unsqueeze(1).unsqueeze(2).repeat(1,1,2)

    based_centroids = subtours_centroid_sperate.gather(dim=1,index=index2)

    single_depot_coor = coor[:, [0], :]

    repeated_depot_coor = coor[:, [0], :].repeat(1,all_subtour_num,1)

    all_centroid_depot_vectors = subtours_centroid_sperate - repeated_depot_coor

    based_centroid_depot_vectors = based_centroids - single_depot_coor

    repeated_based_centroid_depot_vectors = based_centroid_depot_vectors.repeat(1,all_subtour_num,1)

    x1_times_x2 = (repeated_based_centroid_depot_vectors * all_centroid_depot_vectors).sum(2)

    x1_module_length = torch.sqrt((repeated_based_centroid_depot_vectors**2).sum(2))
    x2_module_length = torch.sqrt((all_centroid_depot_vectors**2).sum(2))

    cos_value = x1_times_x2 / (x1_module_length*x2_module_length)
    cos_value[cos_value.ge(1)] =  1 - 1e-5
    cos_value[cos_value.le(-1)] = -1 + 1e-5 # arccos只接受[1,-1]范围内的值，防止出现 nan
    cross_value = np.cross(repeated_based_centroid_depot_vectors.cpu().numpy(), all_centroid_depot_vectors.cpu().numpy())

    cross_value = torch.tensor(cross_value)
    negtivate_sign_2 = torch.ones(size=(cross_value.shape[0],cross_value.shape[1]))
    negtivate_sign_2[cross_value.lt(0)] = -1

    theta_value = torch.arccos(cos_value) # 3.1415为pi，即180°
    theta_value = torch.where(torch.isnan(theta_value), torch.full_like(theta_value, 2 * 3.1415926), theta_value)
    theta_value = negtivate_sign_2*theta_value

    theta_value[theta_value.lt(0)] +=2 * 3.1415926


    theta_value[x3_.le(0)] = 6*3.1415926
    theta_value_sort_value, theta_value_sort_index = torch.sort(theta_value,dim=1)

    repeated_sub_tours_padding = sub_tours_padding.unsqueeze(0).repeat(batch_size,1,1)

    gather_theta_value_sort_index = theta_value_sort_index.unsqueeze(2).repeat(1,1,max_subtour_length)

    resort_repeated_sub_tours_padding = repeated_sub_tours_padding.gather(dim=1,index=gather_theta_value_sort_index)


    x4 = torch.arange(all_subtour_num)[None,:].repeat(batch_size,1)
    # print(visit_depot_num)
    x5 = (x4 < visit_depot_num[:,None]).int()
    x6 = x5.unsqueeze(2).repeat(1,1,max_subtour_length)

    resort_repeated_sub_tours_padding = resort_repeated_sub_tours_padding*x6

    resort_repeated_sub_tours_padding = resort_repeated_sub_tours_padding.reshape(batch_size,-1)


    resort_sub_tours = resort_repeated_sub_tours_padding[resort_repeated_sub_tours_padding.gt(0)].reshape(batch_size,-1)


    repeated_sub_tours_length = sub_tours_length[sub_tours_length.gt(0)].unsqueeze(0).repeat(batch_size,1)

    resort_repeated_sub_tours_length = repeated_sub_tours_length.gather(dim=1,index= theta_value_sort_index)
    resort_repeated_sub_tours_length = resort_repeated_sub_tours_length*x5
    max_subtour_number = visit_depot_num.max()

    resort_repeated_sub_tours_length = resort_repeated_sub_tours_length[:,:max_subtour_number]

    temp_tri = np.triu(np.ones((batch_size,max_subtour_number.item(), max_subtour_number.item())), k=1)
    resort_repeated_sub_tours_length_numpy = resort_repeated_sub_tours_length.clone().cpu().numpy()
    temp_index = np.dot(resort_repeated_sub_tours_length_numpy, temp_tri)
    temp_index_1 = torch.from_numpy(temp_index).long().cuda()
    index1 = torch.arange(batch_size)
    temp_index_1 = temp_index_1[index1,index1]
    temp_index_1[temp_index_1.ge(problem_size)]=0


    flag = torch.zeros(size=(batch_size,problem_size),dtype=torch.int)
    index1 = torch.arange(batch_size)[:,None].repeat(1,max_subtour_number)

    flag[index1,temp_index_1]=1

    solution_ = torch.cat((resort_sub_tours.unsqueeze(2),flag.unsqueeze(2)),dim=2)
    return solution_
