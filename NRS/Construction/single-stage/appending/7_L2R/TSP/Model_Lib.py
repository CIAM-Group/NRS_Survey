from typing import Tuple
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

def compatibility(model_params, q_out, nodes_embedded,cur_bias,ninf_mask):
    # q.shape: (batch,1, embedding)
    # neighbors_nodes.shape: (batch, m, embedding)
    # cur_bias.shape: (batch,1, m), note that cur_bias is negative values
    # nin_mask.shape: (batch, 1, m)

    score = torch.matmul(q_out, nodes_embedded.transpose(1, 2))
    # shape: (batch, 1, m)
    score_scaled = score / model_params['sqrt_embedding_dim']
    # shape: (batch, 1, m)
    score_scaled = score_scaled + cur_bias
    # shape: (batch, 1, m)
    score_clipped = model_params['logit_clipping'] * torch.tanh(score_scaled)
    # shape: (batch, 1, m)
    score_clipped = score_clipped + ninf_mask
    probs = F.softmax(score_clipped, dim=-1).squeeze(1)
    # shape: (batch, m)

    return probs

def select_next_node(probs: Tensor, decoding_strategy: str="sampling")-> Tuple[Tensor, Tensor]:
    """
    Design a novel algorithm to select the next node in each step.
    Args:
    probs: Probability distribution over nodes, shape: (batch_size, m).
    decoding_strategy: Decoding strategy to use. Available strategies: ['sampling', 'greedy'], default: 'sampling'.

    Return:
    ID of the next node to visit.
    prob of the selected node.
    """
    assert not torch.isnan(probs).any(), "probs has nan, but it should not have any nans."
    batch_size, problem_size = probs.size()
    if decoding_strategy == "sampling":
        # Check if sampling went OK, can go wrong due to bug on GPU
        # See https://discuss.pytorch.org/t/bad-behavior-of-multinomial-function/10232
        # to fix pytorch.multinomial bug on selecting 0 probability elements
        while True:
            selected = probs.reshape(batch_size, -1).multinomial(1).squeeze(dim=1)
            # shape: (batch, )
            prob = torch.gather(probs, dim=-1, index=selected.unsqueeze(-1)).squeeze(-1)
            # shape: (batch, )
            if (prob != 0).all():
                break
        assert prob.size() == (batch_size,), f"prob.size(): {prob.size()}. Expected: {(batch_size,)}"
        # shape: (batch, n_start)
    elif decoding_strategy == "greedy":
        selected = torch.argmax(probs, dim=-1)
        # (batch_size, n_start)
        prob = None # prob is not needed for greedy decoding
        # shape: (batch, n_start)
    else:
        raise NotImplementedError(f"eval_type: {decoding_strategy} is not implemented!")

    return selected, prob

def _get_encoding(encoded_nodes, node_index_to_pick):
    # encoded_nodes.shape: (batch, problem, embedding)
    # node_index_to_pick.shape: (batch, n)
    embedding_dim = encoded_nodes.size(2)

    gathering_index = node_index_to_pick[:, :, None].expand(-1, -1, embedding_dim)
    # shape: (batch, 1, embedding)

    picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)
    # shape: (batch, 1, embedding)

    return picked_nodes

#######################################
#NN SUB CLASS / FUNCTIONS
#######################################
def adaptation_attention_free_module(q, k, v, adaptation_bias, ninf_mask=None):
    """
    The core code of Adaptation Attention Free Module.

    Inspired by the paper: An Attention Free Transformer
    (url:  https://arxiv.org/pdf/2105.14103.pdf)

    Args:
        q: query, shape: (batch, n, embedding_dim)
        k: key, shape: (batch, m, embedding_dim)
        v: value, shape: (batch, m, embedding_dim)
        adaptation_bias: - alpha * log_scale * dist, shape: (batch, n, m)
        ninf_mask: shape: (batch, n, m)

    Return:
        out: shape: (batch, n, embedding_dim)

    Note:
    To prevent potential value overflows caused by exponential operations, we use "torch.nan_to_num_" to solve it.
    For more details, please refer to the official document:
    https://pytorch.org/docs/1.10/generated/torch.nan_to_num.html
    """

    sigmoid_q = torch.sigmoid(q)
    # shape: (batch, n, embedding_dim)

    if ninf_mask is not None:
        adaptation_bias = adaptation_bias + ninf_mask

    bias = torch.exp(adaptation_bias) @ torch.mul(torch.exp(k), v)
    # shape: (batch, n, embedding_dim)
    a_k = torch.exp(adaptation_bias) @ torch.exp(k)

    weighted = bias / a_k
    if torch.isinf(bias).any() or torch.isinf(a_k).any():
        weighted = torch.nan_to_num_(bias) / torch.nan_to_num_(a_k)
    if torch.isnan(weighted).any():
        torch.nan_to_num_(weighted)
    # shape: (batch, n, embedding_dim)

    out = torch.mul(sigmoid_q, weighted)
    # shape: (batch, n, embedding_dim)

    return out

class Feed_Forward_Module(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        # input.shape: (batch, problem, embedding)

        return self.W2(F.relu(self.W1(input1)))

class AddAndLayerNormalization(nn.Module):
    '''
    official docs: https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
    归一化维度：按层（layer）进行归一化。对每个样本在层内的所有特征进行归一化，
    即对每个样本的所有通道进行归一化，计算该层所有输出的均值和方差。
    '''
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.layer_norm = nn.LayerNorm(embedding_dim, elementwise_affine=True)

    def forward(self, input1, input2):
        # input.shape: (batch, 1+1+k, embedding)

        return self.layer_norm(input1 + input2)
