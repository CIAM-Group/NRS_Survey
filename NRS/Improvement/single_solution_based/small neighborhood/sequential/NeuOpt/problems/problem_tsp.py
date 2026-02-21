from torch.utils.data import Dataset
import torch
import pickle
import os
from utils import augmentation
from LIBUtils import *
class TSP(object):

    NAME = 'tsp'  # Travelling Salesman Problem
    
    def __init__(self, p_size, init_val_met = 'random', with_assert = False, DUMMY_RATE = 0, k = 4, with_bonus = False, with_regular = False, lib_model=False):
        self.size = p_size
        self.do_assert = with_assert
        self.init_val_met = init_val_met
        self.k_max = k
        self.state = 'eval'
        # ! 修改的代码
        self.lib_model = lib_model 
        # print(f'TSP with {self.size} nodes.', 
        #       f'MAX {self.k_max}-opt.'
        #       ' Do assert:', with_assert)
    
    def train(self):
        self.state = 'train'
        
    def eval(self):
        self.state = 'eval'
    
    def augment(self, batch, val_m, only_copy=False):
        bs, gs, dim = batch['coordinates'].size()
        if only_copy:
            coordinates = batch['coordinates'].unsqueeze(1).expand(bs,val_m,gs,dim).clone().reshape(-1,gs,dim)
        else:
            coordinates = batch['coordinates'].unsqueeze(1).expand(bs,val_m,gs,dim).clone()
            coordinates = augmentation(coordinates, val_m).reshape(-1,gs,dim)
        
        return {'coordinates': coordinates}
    
    def input_feature_encoding(self, batch):
        return batch['coordinates'].clone()
    
    def get_initial_solutions(self, batch):
        
        coordinates = batch['coordinates']
        batch_size = coordinates.size(0)
    
        def get_solution(methods):
            
            if methods == 'random':
                
                set = torch.rand(batch_size,self.size).argsort().long()
                rec = torch.zeros(batch_size, self.size).long()
                index = torch.zeros(batch_size,1).long()
                
                for i in range(self.size - 1):
                    rec.scatter_(1,set.gather(1, index + i), set.gather(1, index + i + 1))
                
                rec.scatter_(1,set[:,-1].view(-1,1), set.gather(1, index))
                return rec

            elif methods == 'greedy':
               
               candidates = torch.ones(batch_size,self.size).bool()
               rec = torch.zeros(batch_size, self.size).long()
               selected_node = torch.zeros(batch_size, 1).long()
               candidates.scatter_(1, selected_node, 0)
               
               for i in range(self.size - 1):
                   
                   d1 = coordinates.cpu().gather(1, selected_node.unsqueeze(-1).expand(batch_size, self.size, 2))
                   d2 = coordinates.cpu()
                   
                   dists = (d1 - d2).norm(p=2, dim=2)
                   dists[~candidates] = 1e5
                   
                   next_selected_node = dists.min(-1)[1].view(-1,1)
                   rec.scatter_(1,selected_node, next_selected_node)
                   candidates.scatter_(1, next_selected_node, 0)
                   selected_node = next_selected_node

               return  rec
            
            else:
                raise NotImplementedError()

        return get_solution(self.init_val_met).expand(batch_size, self.size).clone()
    
    def step(self, batch, rec, action, obj, feasible_history, t, weights = 0):
        
        bs, gs = rec.size()
        pre_bsf = obj[:,1:].clone() # batch_size, 3 (current, bsf, tsp_bsf)
        
        # k-opt step
        next_state = self.k_opt(rec, action)
        next_obj = self.get_costs(batch, next_state,lib_model=self.lib_model) # ! 修改的代码
        
        # MDP step
        now_obj = pre_bsf.clone()
        now_obj[:,0] = next_obj.clone()
        now_obj[:,1] = next_obj.clone()
        now_bsf = torch.min(pre_bsf, now_obj)
        rewards = (pre_bsf - now_bsf)
        reward = torch.cat((rewards[:,:1], # reward
                            rewards[:,:1] * 0., # regulation
                            rewards[:,:1] * 0., # bonus
                           ),-1)
        
        # return
        out = (next_state, 
               reward,
               torch.cat((next_obj[:,None], now_bsf),-1), 
               None, 
               None,
               None,
               None)
        
        return out
    
    def k_opt(self, rec, action):
        
        # action bs * (K_index, K_from, K_to)
        selected_index = action[:,:self.k_max]
        left = action[:,self.k_max:2*self.k_max]
        right = action[:,2*self.k_max:]
        
        # prepare
        rec_next = rec.clone()
        right_nodes = rec.gather(1,selected_index)
        argsort = rec.argsort()
        
        # new rec
        rec_next.scatter_(1,left,right)
        cur = left[:,:1].clone()
        for i in range(self.size - 2): # self.size - 2 is already correct
            next_cur = rec_next.gather(1,cur)
            pre_next_wrt_old = argsort.gather(1, next_cur)
            reverse_link_condition = ((cur!=pre_next_wrt_old) & ~((next_cur==right_nodes).any(-1,True)))
            next_next_cur = rec_next.gather(1,next_cur)
            rec_next.scatter_(1,next_cur,torch.where(reverse_link_condition, pre_next_wrt_old, next_next_cur))
            # if i >= self.size - 2: assert (reverse_link_condition == False).all()
            cur = next_cur
            
        return rec_next
    
    def get_order(self, rec, return_solution = False):
        
        bs,p_size = rec.size()
        visited_time = torch.zeros((bs,p_size),device = rec.device)
        pre = torch.zeros((bs),device = rec.device).long()
        for i in range(p_size - 1):
            visited_time[torch.arange(bs),rec[torch.arange(bs),pre]] = i + 1
            pre = rec[torch.arange(bs),pre]
        if return_solution:
            return visited_time.argsort() # return decoded solution in sequence
        else:
            return visited_time.long() # also return visited order
    
    def check_feasibility(self, rec):
        p_size = self.size
        assert (
            (torch.arange(p_size, out=rec.new())).view(1, -1).expand_as(rec)  == 
            rec.sort(1)[0]
        ).all(), ((
            (torch.arange(p_size, out=rec.new())).view(1, -1).expand_as(rec)  == 
            rec.sort(1)[0]
        ).sum(-1),"not visiting all nodes")
        
        real_solution = self.get_order(rec, True)
            
        assert (
            (torch.arange(p_size, out=rec.new())).view(1, -1).expand_as(rec)  == 
            real_solution.sort(1)[0]
        ).all(), ((
            (torch.arange(p_size, out=rec.new())).view(1, -1).expand_as(rec)  == 
            real_solution.sort(1)[0]
        ).sum(-1),"not valid tour")
        
    # ! 修改的代码    
    def get_costs(self, batch, rec, get_context = False, check_full_feasibility = False, lib_model=False):
        
        if lib_model:
            coor = batch['lib_data']
            coor_next = coor.gather(1, rec.long().unsqueeze(-1).expand(*rec.size(), 2))
            segment_lengths_raw =  (coor  - coor_next).norm(p=2, dim=2)
            ewt = batch['edge_weight_type'][0]
            if ewt == 'CEIL_2D':
                segment_lengths = torch.ceil(segment_lengths_raw)
            elif ewt == 'EUC_2D':
                # TSPLIB 定义：floor(x + 0.5)，避免银行家舍入
                # segment_lengths = (segment_lengths_raw + 0.5).floor()
                segment_lengths = torch.floor(segment_lengths_raw + 0.5)
            else:
                assert False, f"Edge weight type {ewt} not supported yet!"
            cost = segment_lengths.sum(1)
        else:
        
            coor = batch['coordinates']
            coor_next = coor.gather(1, rec.long().unsqueeze(-1).expand(*rec.size(), 2))
            cost = (coor  - coor_next).norm(p=2, dim=2).sum(1)
            
        #check feasibility if needed
        if self.do_assert or check_full_feasibility:
            print("checking feasibility...")
            self.check_feasibility(rec)
            print("feasibility check passed.")

        if get_context:
            return cost, None
        else:
            return cost
        
    # ! 修改的代码
    @staticmethod
    def make_dataset(*args, **kwargs):
        lib_model = kwargs.get('lib_model', False)
        if lib_model:
            path = kwargs.get('filename', None)
            scale_range = kwargs.get('scale_range', [0,1000])
            device = kwargs.get('device', 'cpu')
            return tsplib_loader(path, scale_range=scale_range, device=device)
        return TSPDataset(*args, **kwargs)

def use_saved_problems_tsp_txt(filename, total_episodes,device, start=0):
    nodes_coords = []
    solution = []
    from tqdm import tqdm
    for line in tqdm(open(filename, "r").readlines()[start:start + total_episodes], ascii=True):
        line = line.split(" ")
        num_nodes = int(line.index('output') // 2)
        nodes_coords.append(
            [[float(line[idx]), float(line[idx + 1])] for idx in range(0, 2 * num_nodes, 2)]
        )
        tour_nodes = [int(node) - 1 for node in line[line.index('output') + 1:-1]]
        solution.append(tour_nodes)

    problems = torch.tensor(nodes_coords,device=device)  # shape: (batch, problem, 2)
    solution = torch.tensor(solution,device=device)  # shape: (batch, problem)
    assert problems.size(0) == solution.size(0)
    assert problems.size(1) == solution.size(1)
    gathering_index = solution.unsqueeze(2).expand(-1, -1, 2)
    # shape: (batch, problem, 2)
    ordered_seq = problems.gather(dim=1, index=gathering_index)
    rolled_seq = ordered_seq.roll(dims=1, shifts=-1)
    segment_lengths = ((ordered_seq - rolled_seq) ** 2).sum(2).sqrt()
    # shape: (batch, problem)
    travel_distances = segment_lengths.sum(1)
    # shape: (batch,)
    optimal_score = travel_distances

    return problems,optimal_score

class TSPDataset(Dataset):
    def __init__(self, filename=None, size=100, num_samples=10000, offset=0, distribution=None, DUMMY_RATE=None):
        super(TSPDataset, self).__init__()
        self.optimal_score = None

        if filename is not None:
            if os.path.splitext(filename)[1] == '.pkl':
                with open(filename, 'rb') as f:
                    data = pickle.load(f)
                    self.data = [torch.FloatTensor(row) for row in (data[offset:offset+num_samples])]
            elif os.path.splitext(filename)[1] == '.txt':
                problems, optimal_score = use_saved_problems_tsp_txt(filename, num_samples, 'cpu', start=offset)
                self.data = [row for row in problems]
                self.optimal_score = optimal_score
            else:
                raise NotImplementedError('Only pkl and txt file supported now.')
        else:
            raise NotImplementedError('Only loading from file is pkl or txt supported now.')

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if self.optimal_score is not None:
            return {'coordinates': self.data[idx], 'optimal_score': self.optimal_score[idx]}
        return {'coordinates': self.data[idx], 'optimal_score': torch.tensor(0.0)}

# class TSPDataset(Dataset):
#     def __init__(self, filename=None, size=20, num_samples=10000, offset=0, distribution=None, DUMMY_RATE=None):
        
#         super(TSPDataset, self).__init__()
        
#         self.data = []
#         self.size = size

#         if filename is not None:
#             assert os.path.splitext(filename)[1] == '.pkl', 'file name error'
            
#             with open(filename, 'rb') as f:
#                 data = pickle.load(f)
#             self.data = [self.make_instance(args) for args in data[offset:offset+num_samples]]

#         else:
#             self.data = [{'coordinates': torch.FloatTensor(self.size, 2).uniform_(0, 1)} for i in range(num_samples)]
        
#         self.N = len(self.data)
        
#         print(f'{self.N} instances initialized.')
    
#     def make_instance(self, args):
#         return {'coordinates': torch.FloatTensor(args)}
    
#     def __len__(self):
#         return self.N

#     def __getitem__(self, idx):
#         return self.data[idx]