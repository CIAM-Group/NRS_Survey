import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from Model_Lib import (adaptation_attention_free_module,
                       select_next_node,
                       compatibility,
                       _get_encoding)

class CVRPUpperModel(nn.Module):

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.embedding_dim = self.model_params['embedding_dim']
        self.embedding = nn.Linear(3, self.embedding_dim)

        # first node & last node
        self.Wq_last = nn.Linear(self.embedding_dim+1, self.embedding_dim, bias=False)
        # for AAFM
        self.Wk = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.Wv = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)

        self.alpha_attn = nn.Parameter(torch.Tensor([1.]), requires_grad=True)
        self.alpha_com = nn.Parameter(torch.Tensor([1.]), requires_grad=True)  # for compatibility

        self.encoded_nodes = None
        self.log_scale = None
        self.k = None
        self.v = None
        self.single_head_key = None
        self.q_first = None

    def set_decoder_method(self,decoder_type):
        self.model_params['eval_type'] = decoder_type

    def pre_forward(self, reset_state):
        depot_xy = reset_state.depot_xy
        # shape: (batch, 1, 2)
        node_xy = reset_state.node_xy
        # shape: (batch, problem, 2)
        node_demand = reset_state.node_demand
        # shape: (batch, problem)
        node_xy_demand = torch.cat((node_xy, node_demand[:, :, None]), dim=2)
        # shape: (batch, problem, 3)

        depot_demand = torch.zeros(size=(depot_xy.shape[0], 1, 1), device=depot_xy.device)
        depot_xy_demand = torch.cat((depot_xy, depot_demand), dim=2)
        # shape: (batch, 1, 3)
        depot_node_xy_demand = torch.cat((depot_xy_demand, node_xy_demand), dim=1)
        # shape: (batch, problem+1, 3)
        self.encoded_nodes = self.embedding(depot_node_xy_demand)

        self.set_kv(self.encoded_nodes)
        # shape: (batch, problem+1, embedding_dim)
        self.log_scale = reset_state.log_scale

    def set_kv(self, encoded_nodes):
        # encoded_nodes.shape: (batch, problem, embedding)
        self.k = self.Wk(encoded_nodes)
        self.v = self.Wv(encoded_nodes)
        self.single_head_key = encoded_nodes
        # shape: (batch, problem, embedding)

    def forward(self, state):
        # valid_index_sorted.shape:(batch, unvisited_num)
        batch_size = state.batch_size
        problem_size = state.problem_size
        if state.current_node is None:
            return None
        else:
            unvisited_index_sorted = state.upper_unvisited_index
            # shape: (batch, max_cur_valid_nodes)
            cur_dist_unvisited = state.upper_cur_dist
            cur_ninf_mask = state.upper_cur_ninf_mask
            # shape: (batch, 1, max_cur_valid_nodes)
            encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node[:,None])
            # shape: (batch, 1, embedding)

            #  AAFM
            #######################################################
            last_node_load = torch.cat((encoded_last_node, state.load[:, None, None]), dim=2)
            # shape: (batch, 1, embedding+1)
            q = self.Wq_last(last_node_load)
            # # shape: (batch, 1, embedding_dim)
            k = _get_encoding(self.k, unvisited_index_sorted)
            v = _get_encoding(self.v, unvisited_index_sorted)
            # shape: (batch, max_cur_valid_nodes, embedding_dim)
            alpha_adaptation_bias_attn = -1 * self.log_scale * self.alpha_attn * cur_dist_unvisited
            # shape: (batch, 1, unvisited)
            AAFM_OUT = adaptation_attention_free_module(q, k, v, alpha_adaptation_bias_attn, ninf_mask=cur_ninf_mask)
            # shape: (batch, 1, embedding_dim)
            #######################################################
            # obtain unvisited nodes' scores
            alpha_adaptation_bias_com = -1 * self.log_scale * self.alpha_com * cur_dist_unvisited
            # shape: (batch, 1, unvisited)
            single_k = _get_encoding(self.single_head_key, unvisited_index_sorted)
            # shape: (batch, embedding, unvisited)

            upper_score_k = compatibility(self.model_params, AAFM_OUT, single_k, alpha_adaptation_bias_com, cur_ninf_mask)
            # shape: (batch, unvisited_num)
            selected, selected_score = select_next_node(upper_score_k, decoding_strategy=self.model_params['eval_type'])

            upper_scores = torch.zeros(size=(batch_size, problem_size+1), device=upper_score_k.device)
            upper_scores.scatter_(dim=1, index=unvisited_index_sorted, src=upper_score_k)
            # shape: (batch, problem_size)
            true_selected = torch.gather(unvisited_index_sorted, dim=-1, index=selected.unsqueeze(-1)).squeeze(-1)
            # shape: (batch,)

            return upper_scores, true_selected, selected_score
