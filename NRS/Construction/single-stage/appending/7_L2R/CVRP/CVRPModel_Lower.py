import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from Model_Lib import (adaptation_attention_free_module,
                       select_next_node,
                       compatibility,
                       Feed_Forward_Module,
                       AddAndLayerNormalization)

class CVRPLowerModel(nn.Module):

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.embedding_dim = self.model_params['embedding_dim']
        self.device = self.model_params['device']
        self.embedding = nn.Linear(2, self.embedding_dim,device=self.device)

        decoder_layer_num = self.model_params['decoder_layer_num']
        self.layers_valid = nn.ModuleList([CVRP_Decoder(**self.model_params) for _ in range(decoder_layer_num)])

        # first node & last node
        self.W_load = nn.Linear(1, self.embedding_dim, bias=False,device=self.device)
        self.W_demand = nn.Linear(1, self.embedding_dim, bias=False,device=self.device)
        self.Wq_first = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False,device=self.device)
        self.Wq_last = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False,device=self.device)

        self.alpha_com = nn.Parameter(torch.Tensor([1.]), requires_grad=True)  # for compatibility

        self.padding_embedded = torch.zeros(size=(1, self.embedding_dim), device=self.device)

    def set_decoder_method(self, decoder_type):
        self.model_params['eval_type'] = decoder_type

    def forward(self, state):
        # pairwise_dist.shape: (batch, 1+1+k, 1+1+k)
        # ninf_mask.shape: (batch, 1, 1+1+max_neighbor_k)
        # neighbors_index.shape: (batch, max_neighbor_k)
        # lower_demand.shape: (batch, k, 1)
        batch_size = state.batch_size

        if state.current_node is None: # First Move, depot
            true_selected = torch.zeros(size=(batch_size, ), dtype=torch.long)
            prob = torch.ones(size=(batch_size,))

        else:
            neighbors_ninf_mask = state.lower_cur_ninf_mask[..., 2:]
            # shape: (batch, 1, k)

            load = state.load[:, None, None]
            # shape: (batch, 1, 1)
            load[load == 0] = 1

            first_last_neighbor_embedded = self.embedding(state.lower_xy)
            # shape: (batch,1+1+k, embedding)

            load_embedded = self.W_load(load.clone()) # shape: (batch, 1, embedding)
            q_first = self.Wq_first(first_last_neighbor_embedded[:, [0], :]) + load_embedded
            q_last = self.Wq_last(first_last_neighbor_embedded[:, [1], :]) + load_embedded
            # shape: (batch, 1, embedding)

            lower_demand = torch.clip(state.lower_demand / load, 0, 1)
            # shape: (batch, k, 1)
            neighbors_demand_embedded = self.W_demand(lower_demand)
            # shape: (batch, k, embedding)
            neighbor_nodes = first_last_neighbor_embedded[:, 2:] + neighbors_demand_embedded
            # shape: (batch, k, embedding)
            # use placeholder to represent the padding embeddings of the nodes that do not exist
            neighbor_nodes[neighbors_ninf_mask.squeeze(-2) < 0] = self.padding_embedded

            decoder_out = torch.cat((q_first, q_last, neighbor_nodes), dim=1)
            # shape: (batch, 1+k, embedding)
            log_neighbors = torch.log2(state.neighbors_num_list.float())[:, None, None]
            # shape: (batch, 1, 1)
            scale_pairwise_dist = -1 * log_neighbors * state.lower_pairwise_dist
            # shape: (batch, 1+k, 1+k)

            for layer in self.layers_valid:
                decoder_out = layer(decoder_out, scale_pairwise_dist,
                                    state.lower_cur_ninf_mask.expand(-1,scale_pairwise_dist.size(1),-1))
                # shape: (batch, 1+k, embedding)

            q_out = decoder_out[:, [0]] + decoder_out[:, [1]]
            # shape: (batch, 1, embedding)
            neighbor_nodes_out = decoder_out[:, 2:]
            # shape: (batch, k, embedding)
            adaptation_bias_com = self.alpha_com * scale_pairwise_dist[:, 1, 2:][:, None, :]
            # shape: (batch, 1, k)
            output_probs = compatibility(self.model_params,
                                         q_out,
                                         neighbor_nodes_out,
                                         adaptation_bias_com,
                                         neighbors_ninf_mask)
            # shape: (batch,k)

            #assert not torch.isnan(probs).any(), "probs has nan, but it should not have any nans."
            # AAFM may lead to NAN
            if torch.isnan(output_probs).any():
                probs_clone = output_probs.clone()
                flag = torch.isnan(probs_clone)  # shape: (batch, k)
                row_indices = flag.any(dim=1).nonzero(as_tuple=True)[0]  # shape: (batch,)
                probs_clone[flag] = 0.0  # replace nan with a small value
                probs_clone[row_indices, 0] = 1  # if probs are nan, replace the first one with 1 and select it
            else:
                probs_clone = output_probs.clone()

            selected, prob = select_next_node(probs_clone, self.model_params['eval_type'])
            # shape: (batch, )

            # revise the selected node
            true_selected = torch.gather(state.lower_neighbors_index, dim=-1, index=selected.unsqueeze(-1)).squeeze(-1)
            # shape: (batch,)

        return true_selected, prob

########################################
# DECODER
########################################
class CVRP_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        device = self.model_params['device']

        self.Wq = nn.Linear(embedding_dim, embedding_dim, bias=False, device=device)
        self.Wk = nn.Linear(embedding_dim, embedding_dim, bias=False, device=device)
        self.Wv = nn.Linear(embedding_dim, embedding_dim, bias=False, device=device)

        self.feedForward = Feed_Forward_Module(**model_params)

        self.alpha_attn = nn.Parameter(torch.Tensor([1.]), requires_grad=True)

        self.layer_norm_1 = AddAndLayerNormalization(**model_params)
        self.layer_norm_2 = AddAndLayerNormalization(**model_params)

    def forward(self,data,scale_pairwise_dist,ninf_mask):
        # data.shape: (batch, 1+1+k, embedding)
        # pairwise_dist.shape: (batch, 1+1+k, 1+1+k)

        #  We use AAFM to replace the multi-head attention
        #######################################################
        q = self.Wq(data)
        k = self.Wk(data)
        v = self.Wv(data)
        # shape: (batch, 1+1+k, embedding)
        AAFM_OUT = adaptation_attention_free_module(q, k, v, self.alpha_attn * scale_pairwise_dist, ninf_mask=ninf_mask)
        # shape: (batch, 1+1+k, embedding)

        out1 = self.layer_norm_1(data, AAFM_OUT)
        out2 = self.feedForward(out1)
        out3 = self.layer_norm_2(out1, out2)
        # shape: (batch, 1+1+k, embedding)

        return out3