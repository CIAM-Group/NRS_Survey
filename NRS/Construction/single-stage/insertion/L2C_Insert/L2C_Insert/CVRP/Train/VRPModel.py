
import torch.nn as nn
import torch.nn.functional as F
from L2C_Insert.CVRP.Train.utils2 import *

class VRPModel(nn.Module):

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.mode = model_params['mode']
        self.encoder = CVRP_Encoder(**model_params)
        self.decoder = CVRP_Decoder(**model_params)

        self.encoded_nodes = None

    def forward(self, data, abs_solution, abs_scatter_solu_1, abs_partial_solu_2, random_index, current_step,
                last_node_index, raw_data_capacity, remaining_capacity_each_subtour=None):

        # solution's shape : [B, V]
        # print('0----------------- abs_solution', abs_solution[0])
        # print('0----------------- abs_partial_solu_2', abs_partial_solu_2[0])
        self.capacity = raw_data_capacity.ravel()[0].item()
        batch_size, problem_size, _ = data.shape

        self.index_gobal = torch.arange(batch_size, dtype=torch.long)[:, None]

        if self.mode == 'train':
            # 不能用flag了，边的信息由两个点组成，flag是一个点的额外信息（从depot抵达），不能同时加在两个点上。
            # 但是这样的话，很麻烦
            # 是否可以加入个双重判断，首先用node_index判断选哪个点，其次通过flag判断这个点是否从depot抵达。（不太行）
            # 但因为depot的存在，instance之间的边的数量不一样，对不齐是肯定的。那就只能加 pandding 的边然后加 mask 了。

            padding_solution = node_flag_tran_to_(abs_solution)
            padding_abs_partial_solu_2 = node_flag_tran_to_(abs_partial_solu_2)
            # print(padding_solution.shape)
            # print(padding_abs_partial_solu_2.shape)
            # print(padding_abs_partial_solu_2)

            padding_solution_size = padding_solution.shape[1]

            rela_label, expanded_padding_partial_solution, abs_scatter_solu_1_seleted, abs_scatter_solu_1_unseleted = \
                generate_label(self.index_gobal, random_index, padding_solution, padding_abs_partial_solu_2,
                               abs_scatter_solu_1,batch_size, padding_solution_size)

            # partial_end_node_coor = get_encoding(data, last_node_index.reshape(batch_size, 1))
            # drawPic_VRP_v2(data[0, :, :2], abs_solution[0],
            #                expanded_padding_partial_solution[0], partial_end_node_coor[0],
            #                abs_scatter_solu_1_unseleted[0], abs_scatter_solu_1_seleted[0],
            #                name=f'TSP{problem_size-1}_step{current_step}_label')

            # print(' ------------------ padding_abs_partial_solu_2 shape ', padding_abs_partial_solu_2.shape)

            probs = self.decoder(self.encoder(data, self.capacity), data, self.capacity, remaining_capacity_each_subtour,
                                 padding_abs_partial_solu_2, abs_scatter_solu_1_unseleted, abs_scatter_solu_1_seleted)

            # print(' ------------------ probs shape ', padding_abs_partial_solu_2.shape)
            # print(probs)
            # print(rela_label)

            prob_select = probs[torch.arange(batch_size)[:, None], rela_label].reshape(batch_size, 1)  # shape: [B, 1]

            return prob_select, expanded_padding_partial_solution, abs_scatter_solu_1_unseleted, abs_scatter_solu_1_seleted

        if self.mode == 'test':
            if current_step<=1:
                self.encoded_nodes = self.encoder(data, self.capacity)


            padding_abs_partial_solu_2 = node_flag_tran_to_(abs_partial_solu_2)

            abs_scatter_solu_1_seleted = abs_scatter_solu_1[self.index_gobal, random_index]

            index1 = torch.arange(abs_scatter_solu_1.shape[1])[None, :].repeat(batch_size, 1)

            tmp1 = (index1 < random_index).long()

            tmp2 = (index1 > random_index).long()

            tmp3 = tmp1 + tmp2

            abs_scatter_solu_1_unseleted = abs_scatter_solu_1[tmp3.gt(0.5)].reshape(batch_size,
                                                                                    abs_scatter_solu_1.shape[1] - 1)

            probs = self.decoder(self.encoded_nodes, data, self.capacity, remaining_capacity_each_subtour,
                                 padding_abs_partial_solu_2, abs_scatter_solu_1_unseleted, abs_scatter_solu_1_seleted)

            # print(' ---------- probs.shape', probs.shape)

            rela_selected = probs.argmax(dim=1).unsqueeze(1)  # shape: B

            # print(' ---------- rela_selected \n', rela_selected)
            # print('----------- abs_scatter_solu_1_seleted \n', abs_scatter_solu_1_seleted)
            extend_partial_solution = extend_partial_solution_def(rela_selected, padding_abs_partial_solu_2,
                                                                   abs_scatter_solu_1_seleted, batch_size)



            return extend_partial_solution, abs_scatter_solu_1_unseleted, abs_scatter_solu_1_seleted








class CVRP_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        self.embedding = nn.Linear(3, embedding_dim, bias=True)
        encoder_layer_num = 1
        self.layers = nn.ModuleList([DecoderLayer(**model_params) for _ in range(encoder_layer_num)])

    def forward(self, data_, capacity):
        data = data_.clone().detach()
        data = data[:, :, :3]

        data[:, :, 2] = data[:, :, 2] / capacity

        out = self.embedding(data)

        for layer in self.layers:
            out = layer(out)

        # out = embedded_input  # [B*(V-1), problem_size - current_step +2, embedding_dim]

        return out


########################################
# DECODER
########################################

class CVRP_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        decoder_layer_num = self.model_params['decoder_layer_num']


        self.embedding_last_node = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.embedding_scatter_node = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.embedding_edge = nn.Linear(embedding_dim*2+1, embedding_dim, bias=True)

        self.layers = nn.ModuleList([DecoderLayer(**model_params) for _ in range(decoder_layer_num)])
        self.Linear_final = nn.Linear(embedding_dim, 1, bias=True)


    def forward(self, encoded_ndoes, data, full_capacity, remaining_capacity_edge, padding_abs_partial_solu_2,
                abs_scatter_solu_1_unseleted, abs_scatter_solu_1_seleted):

        batch_size_V = data.shape[0]  # B


        # 获取各个点的 embedding
        enc_current_node           = get_encoding(encoded_ndoes, abs_scatter_solu_1_seleted)
        enc_unseleted_scatter_node = get_encoding(encoded_ndoes, abs_scatter_solu_1_unseleted)
        enc_partial_nodes          = get_encoding(encoded_ndoes, padding_abs_partial_solu_2)


        # 获取当前点的 demand
        current_node               = get_encoding(data, abs_scatter_solu_1_seleted)
        curren_node_demand         = current_node[:,0,2]


        # 标记 last node、 scatter node
        embedded_last_node_        = self.embedding_last_node(enc_current_node)
        enc_unseleted_scatter_node = self.embedding_scatter_node(enc_unseleted_scatter_node)


        # 把 remaining_capacity_edge 融入 每条 partial solution 边中
        remaining_capacity = remaining_capacity_edge.unsqueeze(2) / full_capacity
        enc_partial_nodes  = torch.cat((enc_partial_nodes,torch.roll(enc_partial_nodes, dims=1, shifts=-1), remaining_capacity),dim=2)
        left_encoded_node  = self.embedding_edge(enc_partial_nodes)

        # print(' 1 ----------- left_encoded_node', left_encoded_node.shape)
        # print(' 2 ----------- remaining_capacity_edge', remaining_capacity_edge.shape)


        inter = torch.cat((embedded_last_node_, enc_unseleted_scatter_node), dim=1)
        out   = torch.cat((embedded_last_node_, enc_unseleted_scatter_node, left_encoded_node), dim=1)

        # 根据 padding_abs_partial_solu_2 写 mask
        # 如果最后的 0 数目大于 1个，则倒数第一个0是返回depot的，然后倒数前几个0是要被mask的
        # 1. 倒数最后一个肯定是返回了depot

        # 生成 mask，把一些 padding 的地方进行 mask
        padding_abs_partial_solu_2_shift = torch.roll(padding_abs_partial_solu_2, dims=1, shifts=-1)
        substract_padding_partial_solu = padding_abs_partial_solu_2 - padding_abs_partial_solu_2_shift
        mask = torch.zeros(size=(batch_size_V, padding_abs_partial_solu_2.shape[1]))
        mask[substract_padding_partial_solu.eq(0)] = float('-inf')
        mask[:, -1] = 0
        tmp = torch.zeros(size=(batch_size_V, inter.shape[1]))
        mask1 = torch.cat((tmp, mask), dim=1)



        for layer in self.layers:
            out = layer(out, mask=mask1)


        num = enc_unseleted_scatter_node.shape[1] + 1
        out = out[:, num:]
        out = self.Linear_final(out).squeeze(-1)  # shape: [B*(V-1), reminding_nodes_number + 2, embedding_dim ]
        out = out + mask



        mask2 = remaining_capacity_edge < curren_node_demand[:,None]

        # print('--------- mask2 \n', mask2.long())

        out[mask2] = float('-inf')

        props = F.softmax(out, dim=-1)  # shape: [B, remind_nodes_number]

        return props


class DecoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        # self.Wq_2 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        # self.Wk_2 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        # self.Wv_2 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)
        # self.multi_head_combine_2 = nn.Linear(head_num * qkv_dim, embedding_dim)
        # self.addAndNormalization1 = Add_And_Normalization_Module(**model_params)

        self.feedForward = Feed_Forward_Module(**model_params)
        # self.feedForward_2 = Feed_Forward_Module(**model_params)
        # self.addAndNormalization2 = Add_And_Normalization_Module(**model_params)

    def forward(self, input1, mask=None):
        # input.shape: (batch, problem, EMBEDDING_DIM)

        # print(input1.shape)
        # print(input2.shape)

        q_num = input1.shape[1]

        head_num = self.model_params['head_num']

        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(input1), head_num=head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=head_num)
        # q shape: (batch, HEAD_NUM, problem, KEY_DIM)

        # print(' -------------   attn 1')
        # print('mask.shape', mask.shape)
        if mask is not None:
            mask1 = mask.unsqueeze(1).unsqueeze(1).repeat(1,head_num,q_num,1)
            out_concat = multi_head_attention(q, k, v, mask=mask1)  # shape: (B, n, head_num*key_dim)

        else:
            out_concat = multi_head_attention(q, k, v)  # shape: (B, n, head_num*key_dim)

        multi_head_out = self.multi_head_combine(out_concat)  # shape: (B, n, embedding_dim)

        out1 = input1 + multi_head_out
        out2 = self.feedForward(out1)
        out3 = out1 + out2
        # out3 = multi_head_out
        ################################################################
        # ################################################################
        # q_2 = reshape_by_heads(self.Wq_2(input2), head_num=head_num)
        # k_2 = reshape_by_heads(self.Wk_2(out3), head_num=head_num)
        # v_2 = reshape_by_heads(self.Wv_2(out3), head_num=head_num)
        # k = reshape_by_heads(input2, head_num=head_num)

        # print(' -------------   attn 2')
        # print('mask.shape', mask.shape)
        # mask2 = mask.unsqueeze(2).repeat(1, 1, q_num).unsqueeze(1).repeat(1,head_num,1,1)
        # out_concat_2 = multi_head_attention(q_2, k_2, v_2, mask=None)  # shape: (B, n, head_num*key_dim)
        #
        #
        # multi_head_out_2 = self.multi_head_combine_2(out_concat_2)  # shape: (B, n, embedding_dim)
        #
        # out1_2 = input2 + multi_head_out_2
        # out2_2 = self.feedForward_2(out1_2)

        # out3_2 = out1_2 + out2_2

        # return out3, out3_2
        return out3
        # shape: (batch, problem, EMBEDDING_DIM)


def reshape_by_heads(qkv, head_num):
    batch_s = qkv.size(0)

    n = qkv.size(1)

    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)

    q_transposed = q_reshaped.transpose(1, 2)

    return q_transposed


def multi_head_attention(q, k, v, mask=None):
    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)

    # input_s = k.size(2)

    score = torch.matmul(q, k.transpose(2, 3))  # shape: (B, head_num, n, n)

    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))

    if mask is not None:
        score_scaled = score_scaled + mask

    weights = nn.Softmax(dim=3)(score_scaled)  # shape: (B, head_num, n, n)

    out = torch.matmul(weights, v)  # shape: (B, head_num, n, key_dim)

    out_transposed = out.transpose(1, 2)  # shape: (B, n, head_num, key_dim)

    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)  # shape: (B, n, head_num*key_dim)

    return out_concat


class Feed_Forward_Module(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        return self.W2(F.relu(self.W1(input1)))



