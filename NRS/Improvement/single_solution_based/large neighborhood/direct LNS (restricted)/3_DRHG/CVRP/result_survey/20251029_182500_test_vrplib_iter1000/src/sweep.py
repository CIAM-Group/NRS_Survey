import numpy
import torch
from CVRP.trans_flag import tran_to_node_flag


def cal_saving_mat(problems):
    distances = (problems[:, None, :, :] - problems[:, :, None, :]).norm(p=2, dim=-1)
    # ! 取整
    distances = torch.floor(distances + 0.5)

    saving = distances[:, 1:, 0] + distances[:, 0, 1:] - distances[:, 1:, 1:]
    return saving


def cal_dist_mat(problems):
    distances = (problems[:, None, :, :] - problems[:, :, None, :]).norm(p=2, dim=-1)

    # ! 取整
    distances = torch.floor(distances + 0.5)
    return distances


class Sweep:
    def __init__(self, problems) -> None:
        self.batch_size = problems.size(0)
        self.problem_size = problems.size(1) - 1
        self.coords = problems[:, :, :2]
        self.demands = problems[:, :, 2]  # [batch, n+1]
        self.capacity = problems[0, 0, 3]
        self.distances = cal_dist_mat(self.coords)
        # self.solutions = torch.zeros((self.batch_size,1))
        # self.distances_to_depot = self.distances[:, 0, 1:] #[batch, n]
        self.distances_to_depot = self.distances[:, 0, :]
        self.device = problems.device

        # v2: 用极坐标
        new_coords = self.coords - self.coords[:, 0, :].unsqueeze(1)
        self.theta = torch.atan2(new_coords[:, :, 1], new_coords[:, :, 0])

    def set_up(self):
        self.remaining_capacity = torch.ones((self.batch_size)).to(self.device) * self.capacity
        self.partial_solution = torch.zeros((self.batch_size, 1), dtype=int).to(self.device)
        self.selected_mask = torch.zeros((self.batch_size, self.problem_size + 1), dtype=bool).to(
            self.device)  # [batch, n]
        self.selected_mask[:, 0] = True

    def update(self):
        partial_solution = self.partial_solution
        # 先判断是否需要新开一个route
        out_of_capacity = self.demands > self.remaining_capacity[:, None]
        out_of_capacity[self.selected_mask] = True
        return_to_depot = out_of_capacity.all(dim=1)
        # 取出离当前点最近的未访问点
        current_at_depot = (partial_solution[:, -1] == 0)
        fake_distances = torch.where(self.selected_mask[:, None, :] | out_of_capacity[:, None, :], torch.inf,
                                     self.distances)  # 竖着置为inf
        distances_to_current = fake_distances.gather(1, partial_solution[:, -1][:, None, None].repeat(1, 1,
                                                                                                      self.problem_size + 1)).squeeze(
            1)  # [batch, n]
        nearest_to_current = torch.argmin(distances_to_current, dim=1)  # 理论上不会选到0
        # 取出离depot最远的未访问点
        fake_distance_to_depot = torch.where(self.selected_mask, 0, self.distances_to_depot)
        furthest_to_depot = torch.argmax(fake_distance_to_depot, dim=1)
        next_node = nearest_to_current
        next_node[current_at_depot] = furthest_to_depot[current_at_depot]
        next_node[return_to_depot] = 0
        # 更新remaining_capacity
        self.remaining_capacity[return_to_depot] = self.capacity
        self.remaining_capacity -= self.demands.gather(1, next_node[:, None]).squeeze(-1)  # return_to_depot: - 0
        # 更新patial solution 和 selected mask
        self.partial_solution = torch.concat([self.partial_solution, next_node[:, None]], dim=1)
        self.selected_mask[torch.arange(self.batch_size)[~return_to_depot], next_node[~return_to_depot]] = True
        done = self.selected_mask.all()
        return (done, self.partial_solution)

    def valida_solution_legal(self, problem, solution, capacity_=50):

        capacitys = {100: 50,
                     200: 80,
                     500: 100,
                     1000: 250}
        capacity = capacity_
        coor = problem[:, :, [0, 1]]
        demand = problem[:, :, 2]

        order_node = solution[:, :, 0].clone()
        order_flag = solution[:, :, 1].clone()


        # 0. 判断 solution node list 每个index是否 unique 的

        uniques = torch.unique(order_node[0])
        if len(uniques) != problem.shape[1] - 1:
            assert False, 'wrong node list!'
        # 1. 求出每条 sub tour 对应的 demand，并判断其是否超出 capacity

        # 1.
        # 找到每个instance有几条子路径，
        # 所有instance中子路径总数目是多少     all_subtour_num，
        # 所有instance中子路径中最长长度是多少  max_subtour_length
        batch_size = solution.shape[0]
        problem_size = solution.shape[1]

        visit_depot_num = torch.sum(solution[:, :, 1], dim=1)

        all_subtour_num = torch.sum(visit_depot_num)

        fake_solution = torch.cat((solution[:, :, 1], torch.ones(batch_size)[:, None].to(self.device)), dim=1)

        start_from_depot = fake_solution.nonzero()

        start_from_depot_1 = start_from_depot[:, 1]

        start_from_depot_2 = torch.roll(start_from_depot_1, shifts=-1)

        sub_tours_length = start_from_depot_2 - start_from_depot_1

        max_subtour_length = torch.max(sub_tours_length)

        # 2。
        # 对于每条子路径，单独拿出来，pandding 0 至长度 max_subtour_length
        # 对于每个 instance， 把子路径个数 padding 0 至数目 max_subtour_num
        # 3.
        # 把所有instance的所有子路径 cat 到同一数组，

        start_from_depot2 = solution[:, :, 1].nonzero()
        start_from_depot3 = solution[:, :, 1].roll(shifts=-1, dims=1).nonzero()

        repeat_solutions_node = solution[:, :, 0].repeat_interleave(visit_depot_num, dim=0)
        double_repeat_solution_node = repeat_solutions_node.repeat(1, 2)

        x1 = torch.arange(double_repeat_solution_node.shape[1])[None, :].repeat(len(repeat_solutions_node), 1).to(
            self.device) \
             >= start_from_depot2[:, 1][:, None]
        x2 = torch.arange(double_repeat_solution_node.shape[1])[None, :].repeat(len(repeat_solutions_node), 1).to(
            self.device) \
             <= start_from_depot3[:, 1][:, None]

        x3 = (x1 * x2).long()

        sub_tourss = double_repeat_solution_node * x3

        x4 = torch.arange(double_repeat_solution_node.shape[1])[None, :].repeat(len(repeat_solutions_node), 1).to(
            self.device) \
             < (start_from_depot2[:, 1][:, None] + max_subtour_length)

        x5 = x1 * x4

        sub_tours_padding = sub_tourss[x5].reshape(all_subtour_num, max_subtour_length)

        ########################----------
        ########################----------

        demands = torch.repeat_interleave(demand, repeats=visit_depot_num, dim=0)

        index = torch.arange(sub_tours_padding.shape[0])[:, None].repeat(1, sub_tours_padding.shape[1]).to(self.device)
        sub_tours_demands = demands[index, sub_tours_padding].sum(dim=1)
        if_illegal = (sub_tours_demands > capacity)

        if if_illegal.any():
            illegal_tour = torch.nonzero(if_illegal)[0, 0]
            print('found illegal')
            print(illegal_tour)  # 536
            cum_sum = torch.cumsum(visit_depot_num, dim=0)
            print(torch.nonzero(cum_sum > illegal_tour)[0, 0])  # 39
            # print(visit_depot_num)
            print(sub_tours_demands[if_illegal])
            print(demands[index, sub_tours_padding][if_illegal, :])
            assert False, 'wrong capacity!'

        return


def main(problem_pt_path, solution_save_path):
    problems = torch.load(problem_pt_path)['problem'].cuda()
    sweep_solver = Sweep(problems)
    sweep_solver.set_up()
    done = False
    curren_step = 0
    batch_size = problems.size(0)
    while not done:
        done, partial_solution = sweep_solver.update_v2()
        curren_step += 1

    solution = torch.concat([partial_solution, torch.zeros((batch_size, 1), dtype=int).to(problems.device)], dim=1)
    solution_node_flag = tran_to_node_flag(solution)
    sweep_solver.valida_solution_legal(problems, solution_node_flag, 250)

    if not solution_save_path is None:
        torch.save(solution_node_flag, solution_save_path)

    return solution_node_flag


if __name__ == "__main__":
    problem_pt_path = 'data/train_problem_solution_cost_1000.pt'
    solution_save_path = 'sweep_solution/test.pt'
    import time

    begin = time.time()
    _ = main(problem_pt_path, solution_save_path)
    duration = time.time() - begin
    print('total time: {} s'.format(duration))

