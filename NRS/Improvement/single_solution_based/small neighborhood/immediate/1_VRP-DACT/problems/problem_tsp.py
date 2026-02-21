from torch.utils.data import Dataset
import torch
import pickle
import os

from tqdm import tqdm

from LIBUtils import *

class TSP(object):

    NAME = 'tsp'  # Travelling Salesman Problem
    
    def __init__(self, p_size, init_val_met = 'greedy', with_assert = False, step_method = '2_opt', P = 10, DUMMY_RATE = 0, lib_model=False):
        
        self.size = p_size
        self.do_assert = with_assert
        self.step_method = step_method
        self.init_val_met = init_val_met
        self.P = P
        # ! 修改的代码
        self.lib_model = lib_model 
        # print(f'TSP with {self.size} nodes.', ' Do assert:', with_assert)
        self.train()
    
    def eval(self, perturb = True):
        self.training = False
        self.do_perturb = perturb
        
    def train(self):
        self.training = True
        self.do_perturb = False
    
    def input_feature_encoding(self, batch):
        return batch['coordinates']
        
    def get_real_mask(self, visited_time):
        pass
    
    def get_initial_solutions(self, batch):
        
        batch_size = batch['coordinates'].size(0)
    
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
                    
                    d1 = batch['coordinates'].cpu().gather(1, selected_node.unsqueeze(-1).expand(batch_size, self.size, 2))
                    d2 = batch['coordinates'].cpu()
                    
                    dists = (d1 - d2).norm(p=2, dim=2)
                    dists[~candidates] = 1e5
                    
                    next_selected_node = dists.min(-1)[1].view(-1,1)
                    rec.scatter_(1,selected_node, next_selected_node)
                    candidates.scatter_(1, next_selected_node, 0)
                    selected_node = next_selected_node
                    
                return rec
            
            else:
                raise NotImplementedError()

        return get_solution(self.init_val_met).expand(batch_size, self.size).clone()
    
    def step(self, batch, rec, action, pre_bsf, solving_state = None, best_solution = None):

        bs = action.size(0)
        pre_bsf = pre_bsf.view(bs,-1)
        
        first = action[:,0].view(bs,1)
        second = action[:,1].view(bs,1)
        
        if self.step_method  == 'swap':
            next_state = self.swap(rec, first, second)
        elif self.step_method  == '2_opt':
            next_state = self.two_opt(rec, first, second)
        elif self.step_method  == 'insert':
            next_state = self.insert(rec, first, second)
        else:
            raise NotImplementedError()
        
        # ! 修改的代码
        new_obj = self.get_costs(batch, next_state,lib_model=self.lib_model)
        
        now_bsf = torch.min(torch.cat((new_obj[:,None], pre_bsf[:,-1, None]),-1),-1)[0]
        
        reward = pre_bsf[:,-1] - now_bsf
        
        # update solving state
        solving_state[:,:1] = (1 - (reward > 0).view(-1,1).long()) * (solving_state[:,:1] + 1)
        
        if self.do_perturb:
            
            perturb_index = (solving_state[:,:1] >= self.P).view(-1)
            solving_state[:,:1][perturb_index.view(-1, 1)] *= 0
            pertrb_cnt = perturb_index.sum().item()
            
            if pertrb_cnt > 0:
                next_state[perturb_index] =  best_solution[perturb_index]

        return next_state, reward, torch.cat((new_obj[:,None], now_bsf[:,None]),-1), solving_state


    def insert(self, solution, first, second, is_perturb = False): # insert first to the back of second
        
        rec = solution.clone()
        
        # fix connection for first node
        argsort = solution.argsort()
        
        pre_first = argsort.gather(1,first)
        post_first = solution.gather(1,first)
        
        rec.scatter_(1,pre_first,post_first)
        
        # fix connection for second node
        post_second = rec.gather(1,second)
        
        rec.scatter_(1,second, first)
        rec.scatter_(1,first, post_second)
        
        return rec
    
    def two_opt(self, solution, first, second, is_perturb = False):
        
        rec = solution.clone()
        
        # fix connection for first node
        argsort = solution.argsort()
        pre_first = argsort.gather(1,first)  
        pre_first = torch.where(pre_first != second, pre_first, first)
        rec.scatter_(1,pre_first,second)
        
        # fix connection for second node
        post_second = solution.gather(1,second)
        post_second = torch.where(post_second != first, post_second, second)
        rec.scatter_(1,first, post_second)
        
        # reverse loop:
        cur = first
        for i in range(self.size):
            cur_next = solution.gather(1,cur)
            rec.scatter_(1,cur_next, torch.where(cur != second,cur,rec.gather(1,cur_next)))
            cur = torch.where(cur != second, cur_next, cur)

        return rec
    
    
    def swap(self, solution, first_, second_, is_perturb = False):
    
        solution.gather(1,first_)
        
        con = solution.gather(1,second_) == first_
        first = torch.where(con, second_, first_)
        second =  torch.where(con, first_ , second_)
         
        rec = solution
        argsort = solution.argsort()
        pre_first = argsort.gather(1,first)
      
        # put first behind second   
        rec1 = self.insert(solution, first, second, is_perturb)            
        # put second behind pre_first 
        rec = self.insert(rec1, second, pre_first, is_perturb)
        
        return rec
        
    def check_feasibility(self, rec):
        
        p_size = self.size

        assert (
            torch.arange(p_size).to(rec.device).view(1, -1).expand_as(rec)  == 
            rec.sort(1)[0]
        ).all(), "not visiting all nodes"
        
        real_rec = get_real_seq(rec)
        
        assert (
            torch.arange(p_size).to(rec.device).view(1, -1).expand_as(real_rec)  == 
            real_rec.sort(1)[0]
        ).all(), "not visiting all nodes"
    
    
    def get_swap_mask(self, visited_time):
        
        bs, gs = visited_time.size()        
        selfmask = torch.eye(gs, device = visited_time.device).view(1,gs,gs)
        masks = selfmask.expand(bs,gs,gs).bool()
        
        return masks
   
   # ! 修改的代码
    def get_costs(self, batch, rec, check_full_feasibility = False, lib_model=False):
        
        batch_size, size = rec.size()
        
        # check feasibility
        if self.do_assert or check_full_feasibility:
            print('Checking feasibility...')
            self.check_feasibility(rec)
            print('Feasibility OK!')
            
        if lib_model:
            d1 = batch['lib_data'].gather(1, rec.long().unsqueeze(-1).expand(batch_size, size, 2))
            d2 = batch['lib_data']  
            segment_lengths_raw =  (d1  - d2).norm(p=2, dim=2)
            ewt = batch['edge_weight_type'][0]
            if ewt == 'CEIL_2D':
                segment_lengths = torch.ceil(segment_lengths_raw)
            elif ewt == 'EUC_2D':
                # TSPLIB 定义：floor(x + 0.5)，避免银行家舍入
                # segment_lengths = (segment_lengths_raw + 0.5).floor()
                segment_lengths = torch.floor(segment_lengths_raw + 0.5)
            else:
                assert False, f"Edge weight type {ewt} not supported yet!"
            length = segment_lengths.sum(1)
                
        else:
            d1 = batch['coordinates'].gather(1, rec.long().unsqueeze(-1).expand(batch_size, size, 2))
            d2 = batch['coordinates']
            length =  (d1  - d2).norm(p=2, dim=2).sum(1)
        
        return length
    
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


def get_real_seq(solutions):
    batch_size, seq_length = solutions.size()
    visited_time = torch.zeros((batch_size,seq_length)).to(solutions.device)
    pre = torch.zeros((batch_size),device = solutions.device).long()
    for i in range(seq_length):
       visited_time[torch.arange(batch_size),solutions[torch.arange(batch_size),pre]] = i+1
       pre = solutions[torch.arange(batch_size),pre]
       
    visited_time = visited_time % seq_length
    return visited_time.argsort()   


    
