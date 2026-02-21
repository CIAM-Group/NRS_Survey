import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from Model_Lib import (adaptation_attention_free_module,
                       select_next_node,
                       Feed_Forward_Module,
                       compatibility,
                       _get_encoding)

class TSPUpperModel(nn.Module):

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.embedding_dim = self.model_params['embedding_dim']
        self.embedding = nn.Linear(2, self.embedding_dim)

        # first node & last node
        self.Wq_first = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.Wq_last = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        # for AAFM
        self.Wk = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.Wv = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)

        self.alpha_attn = nn.Parameter(torch.Tensor([1.]), requires_grad=True)
        self.alpha_com = nn.Parameter(torch.Tensor([1.]), requires_grad=True)

        self.encoded_nodes = None
        self.log_scale = None
        self.k = None
        self.v = None
        self.single_head_key = None
        self.q_first = None

    def set_decoder_method(self,decoder_type):
        self.model_params['eval_type'] = decoder_type

    def pre_forward(self, reset_state):
        self.encoded_nodes = self.embedding(reset_state.problems)
        # shape: (batch, problem, embedding_dim)
        self.set_kv(self.encoded_nodes)
        self.log_scale = reset_state.log_scale
        # shape: (batch, problem, embedding_dim)

    def set_kv(self, encoded_nodes):
        # encoded_nodes.shape: (batch, problem, embedding)
        self.k = self.Wk(encoded_nodes)
        self.v = self.Wv(encoded_nodes)
        #self.single_head_key = encoded_nodes
        self.single_head_key = encoded_nodes
        # shape: (batch, problem, embedding)

    def set_q1(self, state):
        # encoded_q.shape: (batch, 1, embedding)
        encoded_first_node = _get_encoding(self.encoded_nodes, state.current_node[:, None])
        self.q_first = self.Wq_first(encoded_first_node)
        # shape: (batch, 1, embedding)

    def forward(self, state):
        # unvisited_index_sorted.shape:(batch, unvisited_num)
        batch_size = state.batch_size
        problem_size = state.problem_size
        if state.selected_count == 1:
            self.set_q1(state)
            # shape: (batch, 1, embedding)
        if state.current_node is None:
            return None, None, None
        else:
            unvisited_index_sorted = state.upper_unvisited_index
            # shape: (batch, max_cur_unvisited_num)
            cur_dist_unvisited = state.upper_cur_dist
            # shape: (batch, 1, max_cur_unvisited_num)
            cur_ninf_mask = state.upper_cur_ninf_mask
            # shape: (batch, 1, max_cur_unvisited_num)

            encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node[:, None])
            # shape: (batch, 1, embedding)
            q_last = self.Wq_last(encoded_last_node)
            # shape: (batch, 1, embedding_dim)

            #  AAFM
            #######################################################
            q = self.q_first + q_last
            # shape: (batch, 1, embedding_dim)
            k = _get_encoding(self.k, unvisited_index_sorted)
            v = _get_encoding(self.v, unvisited_index_sorted)
            # shape: (batch, unvisited_num, embedding_dim)
            alpha_adaptation_bias_attn = -1 * self.log_scale * self.alpha_attn * cur_dist_unvisited
            # shape: (batch, 1, unvisited)
            AAFM_OUT = adaptation_attention_free_module(q, k, v, alpha_adaptation_bias_attn,ninf_mask=cur_ninf_mask)
            # shape: (batch, 1, embedding_dim)

            #######################################################
            # obtain unvisited nodes' scores
            alpha_adaptation_bias_com = -1 * self.log_scale * self.alpha_com * cur_dist_unvisited
            # shape: (batch, 1, unvisited)
            single_k = _get_encoding(self.single_head_key, unvisited_index_sorted)
            # shape: (batch, unvisited_num, embedding_dim)

            upper_score_k = compatibility(self.model_params, AAFM_OUT, single_k, alpha_adaptation_bias_com, cur_ninf_mask)
            # shape: (batch, unvisited_num)
            selected, selected_score = select_next_node(upper_score_k, decoding_strategy=self.model_params['eval_type'])

            upper_scores = torch.zeros(size=(batch_size, problem_size), device=upper_score_k.device)
            upper_scores.scatter_(dim=1, index=unvisited_index_sorted, src=upper_score_k)
            # shape: (batch, problem_size)
            true_selected = torch.gather(unvisited_index_sorted, dim=-1, index=selected.unsqueeze(-1)).squeeze(-1)
            # shape: (batch,)


            return upper_scores, true_selected, selected_score





