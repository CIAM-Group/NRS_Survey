import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import os
import matplotlib.pyplot as plt


class TSPModel(nn.Module):

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.mode = model_params['mode']
        self.encoder = TSP_Encoder(**model_params)
        self.decoder = TSP_Decoder(**model_params)
        self.encoded_nodes = None



    def forward(self, data, abs_solution, abs_scatter_solu_1, abs_partial_solu_2, random_index,
                current_step, last_node_index):

        batch_size_V = data.shape[0]
        problem_size = data.shape[1]

        self.index_gobal = torch.arange(batch_size_V,dtype=torch.long)[:,None]



        if self.mode == 'train':

            self.encoded_nodes = self.encoder(data)

            abs_scatter_solu_1_seleted = abs_scatter_solu_1[self.index_gobal, random_index]

            rela_label,unselect_list,abs_scatter_solu_1_unseleted = self.generate_label(
                                            random_index, abs_solution, abs_scatter_solu_1,
                                            abs_partial_solu_2, abs_scatter_solu_1_seleted, batch_size_V, problem_size)


            probs = self.decoder(self.encoded_nodes, abs_partial_solu_2, abs_scatter_solu_1_seleted,abs_scatter_solu_1_unseleted)

            # 根据 abs_scatter_solu_1_seleted 这个点，和 abs_partial_solu_2， 生成相应的label

            # partial_end_node_coor = self.decoder._get_encoding(data, last_node_index.reshape(batch_size_V,1))

            # drawPic_v1(data[1], abs_solution[1], unselect_list[1], abs_scatter_solu_1_unseleted[1],abs_scatter_solu_1_seleted[1],
            #            partial_end_node_coor[1,0,:],name=str(current_step))

            prob = probs[torch.arange(batch_size_V)[:, None], rela_label].reshape(batch_size_V,1)  # shape: [B, 1]

            return prob, unselect_list,abs_scatter_solu_1_unseleted, abs_scatter_solu_1_seleted


        if self.mode == 'test':
            # 根据 abs_scatter_solu_1_seleted 这个点，和 abs_partial_solu_2， 生成相应的label

            abs_scatter_solu_1_seleted = abs_scatter_solu_1[self.index_gobal, random_index]

            index1 = torch.arange(abs_scatter_solu_1.shape[1])[None, :].repeat(batch_size_V, 1)

            tmp1 = (index1 < random_index).long()

            tmp2 = (index1 > random_index).long()

            tmp3 = tmp1 + tmp2

            abs_scatter_solu_1_unseleted = abs_scatter_solu_1[tmp3.gt(0.5)].reshape(batch_size_V,
                                                                                    abs_scatter_solu_1.shape[1] - 1)

            knearest = self.model_params['knearest']
            if current_step<=1 and not knearest:
                self.encoded_nodes = self.encoder(data)

            probs = self.decoder(self.encoder, self.encoded_nodes, data,
                                 abs_partial_solu_2, abs_scatter_solu_1_seleted,abs_scatter_solu_1_unseleted)

            rela_selected = probs.argmax(dim=1).unsqueeze(1)  # shape: B

            extend_partial_solution = self.extend_partial_solution(
                                                              random_index, rela_selected,abs_scatter_solu_1,
                                                              abs_partial_solu_2, abs_scatter_solu_1_seleted,
                                                              batch_size_V, problem_size)

            # drawPic_v2(data[1], abs_solution[1], extend_partial_solution[1], abs_scatter_solu_1_unseleted[1],abs_scatter_solu_1_seleted[1],
            #            name=str(current_step))
            return extend_partial_solution, abs_scatter_solu_1_unseleted, abs_scatter_solu_1_seleted

    def generate_label(self, random_index, abs_solution, abs_scatter_solu_1, abs_partial_solu_2,
                       abs_scatter_solu_1_seleted, batch_size_V, problem_size):

        index1 = torch.arange(abs_scatter_solu_1.shape[1])[None,:].repeat(batch_size_V,1)


        tmp1 = (index1 < random_index).long()

        tmp2 = (index1 > random_index ).long()

        tmp3 = tmp1 + tmp2

        abs_scatter_solu_1_unseleted = abs_scatter_solu_1[tmp3.gt(0.5)].reshape(batch_size_V,abs_scatter_solu_1.shape[1]-1)

        num_scatter_unseleted = abs_scatter_solu_1_unseleted.shape[1]

        tmp1 = abs_solution.unsqueeze(1).repeat_interleave(repeats=num_scatter_unseleted, dim=1)

        tmp2 = abs_scatter_solu_1_unseleted.unsqueeze(2)

        tmp3 = tmp1 == tmp2

        index_1 = torch.arange(problem_size, dtype=torch.long)[None, :].repeat(batch_size_V, 1).unsqueeze(1).\
                   repeat(1, num_scatter_unseleted, 1)

        index_2 = index_1[tmp3].reshape(batch_size_V, num_scatter_unseleted)

        new_list = abs_solution.clone().detach()

        new_list_len = problem_size - num_scatter_unseleted  # shape: [B, V-current_step]

        index_3 = torch.arange(batch_size_V, dtype=torch.long)[:, None].expand(batch_size_V, index_2.shape[1])

        new_list[index_3, index_2] = -2

        unselect_list = new_list[torch.gt(new_list, -1)].view(batch_size_V, new_list_len)

        # ---------------------------

        tmp4 = abs_scatter_solu_1_seleted == unselect_list
        index_1 = torch.arange(unselect_list.shape[1], dtype=torch.long)[None, :].repeat(batch_size_V, 1)

        index_2 = index_1[tmp4].reshape(batch_size_V, 1)
        index_3 = index_2 - 1

        index4 = torch.arange(batch_size_V)[:,None]
        abs_teacher_index = unselect_list[index4,index_3]
        # print(abs_teacher_index)

        # -----------------

        tmp5 = abs_teacher_index == abs_partial_solu_2
        index_1 = torch.arange(abs_partial_solu_2.shape[1], dtype=torch.long)[None, :].repeat(batch_size_V, 1)

        index_2 = index_1[tmp5].reshape(batch_size_V, 1)
        rela_label = index_2



        return rela_label,unselect_list,abs_scatter_solu_1_unseleted

    def extend_partial_solution(self, random_index, rela_selected, abs_scatter_solu_1, abs_partial_solu_2,
                       abs_scatter_solu_1_seleted, batch_size_V, problem_size):
        '''
        这个方法的目标是，
        （1）给定一个散点，散点集里移除这个点。
        （2）模型会决策这个散点插在哪条边，这个决策用 “rela_selected” 表示，然后这个边所在的 partial solution 就自然而然地 extend 了
             rela_selected: 上一步的 partial solution 中被选中的点，当前步骤的散点会插入在这里
        '''


        num_abs_partial_solu_2 = abs_partial_solu_2.shape[1]

        temp_extend_solution = -torch.ones(num_abs_partial_solu_2 + 1)[None,:].repeat(batch_size_V,1)
        temp_extend_solution = temp_extend_solution.long()

        index1 = torch.arange(num_abs_partial_solu_2+1)[None,:].repeat(batch_size_V,1)

        tmp1 = (index1 <= rela_selected).long()

        tmp2 = (index1 > rela_selected + 1).long()

        tmp3 = tmp1+tmp2

        temp_extend_solution[tmp3.gt(0.5)] = abs_partial_solu_2.ravel()


        # 这一步是要把被insert的点放在 temp_extend_solution 的 rela_selected+1 这个index
        index3 = torch.arange(batch_size_V)[:,None]
        temp_extend_solution[index3,rela_selected+1] = abs_scatter_solu_1_seleted

        return temp_extend_solution


########################################
# ENCODER
########################################
class TSP_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        encoder_layer_num = 1
        self.embedding = nn.Linear(2, embedding_dim, bias=True)
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num)])

    def forward(self, data):
        embedded_input = self.embedding(data)
        out = embedded_input
        for layer in self.layers:
            out = layer(out)
        return out


class TSP_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        encoder_layer_num = self.model_params['decoder_layer_num']

        self.embedding_last_node = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.embedding_partial_node = nn.Linear(embedding_dim*2, embedding_dim, bias=True)
        self.embedding_scatter_node = nn.Linear(embedding_dim, embedding_dim, bias=True)

        self.layers = nn.ModuleList([DecoderLayer(**model_params) for _ in range(encoder_layer_num)])

        self.Linear_final = nn.Linear(embedding_dim, 1, bias=True)

    def _get_encoding(self,encoded_nodes, node_index_to_pick):
        batch_size = node_index_to_pick.size(0)
        pomo_size = node_index_to_pick.size(1)
        embedding_dim = encoded_nodes.size(2)

        gathering_index = node_index_to_pick[:, :, None].expand(batch_size, pomo_size, embedding_dim)

        picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)

        return picked_nodes

    def forward(self, encoder, encoded_nodes, data, abs_partial_solu_2, abs_scatter_solu_1_seleted,abs_scatter_solu_1_unseleted):


        knearest = self.model_params['knearest']
        # print('knearest',knearest)
        if knearest:

            current_node_coor = _get_encoding(data, abs_scatter_solu_1_seleted)
            unseleted_scatter_coor = _get_encoding(data, abs_scatter_solu_1_unseleted)
            partial_nodes_coor = _get_encoding(data, abs_partial_solu_2)
            #
            #
            k_nearest_edges = self.model_params['k_nearest_edges']
            k_nearest_scatter = self.model_params['k_nearest_scatter']
            batch_size, partial_nodes_coor_number, _ = partial_nodes_coor.shape
            unseleted_scatter_coor_number = unseleted_scatter_coor.shape[1]

            left_partial_nodes_coor1 = partial_nodes_coor  # torch.cat((partial_nodes_coor, torch.roll(partial_nodes_coor, dims=1, shifts=-1)), dim=2)
            left_partial_nodes_coor2 = torch.roll(partial_nodes_coor, dims=1, shifts=-1)

            # 每条边对应两个点
            # 这两个点都与 scatter_node_selected算距离，取最小值
            # 然后排序。
            if partial_nodes_coor_number > k_nearest_edges:

                partial_nodes_coor = partial_nodes_coor

                distance1 = torch.norm(partial_nodes_coor - current_node_coor, dim=2)

                distance2 = torch.roll(distance1, dims=1, shifts=-1)
                distances = torch.cat((distance1.unsqueeze(2), distance2.unsqueeze(2)), dim=2)
                distances, _ = torch.min(distances, dim=2)
                _, sort_index = torch.topk(distances, dim=1, k=k_nearest_edges, largest=False)
                left_partial_nodes_coor1 = self._get_encoding(left_partial_nodes_coor1, sort_index).clone().detach()
                left_partial_nodes_coor2 = self._get_encoding(left_partial_nodes_coor2, sort_index).clone().detach()

            if unseleted_scatter_coor_number > k_nearest_scatter:
                distance3 = torch.norm(unseleted_scatter_coor - current_node_coor, dim=2)
                _, sort_index2 = torch.topk(distance3, dim=1, k=k_nearest_scatter, largest=False)
                unseleted_scatter_coor = self._get_encoding(unseleted_scatter_coor, sort_index2).clone().detach()


            if self.model_params['coor_norm']:

                current_node_coor, unseleted_scatter_coor, left_partial_nodes_coor1, left_partial_nodes_coor2 = normalize(
                    current_node_coor, unseleted_scatter_coor, left_partial_nodes_coor1, left_partial_nodes_coor2)

            lengths = [current_node_coor.shape[1], unseleted_scatter_coor.shape[1],
                       left_partial_nodes_coor1.shape[1], left_partial_nodes_coor2.shape[1]]

            all_coors = torch.cat((current_node_coor, unseleted_scatter_coor,
                                   left_partial_nodes_coor1, left_partial_nodes_coor2), dim=1)


            enc_all_node = encoder(all_coors)



            enc_current_node, enc_unseleted_scatter_node, enc_partial_nodes1, enc_partial_nodes2 = \
                torch.split(enc_all_node, lengths, dim=1)

            embedded_last_node_ = self.embedding_last_node(enc_current_node)
            # enc_partial_nodes        = self.embedding_partial_node(enc_partial_nodes)
            enc_unseleted_scatter_node = self.embedding_scatter_node(enc_unseleted_scatter_node)

            left_encoded_node = torch.cat((enc_partial_nodes1,enc_partial_nodes2), dim=2)

            left_encoded_node = self.embedding_partial_node(left_encoded_node)

            out = torch.cat((embedded_last_node_, enc_unseleted_scatter_node, left_encoded_node), dim=1)

            layer_count = 0
            # print(all_coors.shape, enc_all_node.shape, out.shape)
            for layer in self.layers:
                out = layer(out)
                layer_count += 1
            num = enc_unseleted_scatter_node.shape[1] + 1
            # num = 1
            out = out[:, num:]


            out = self.Linear_final(out).squeeze(-1)  # shape: [B*(V-1), reminding_nodes_number + 2, embedding_dim ]

            props = F.softmax(out, dim=-1)  # shape: [B, remind_nodes_number]


            if partial_nodes_coor_number > k_nearest_edges:
                new_props = torch.zeros(batch_size, partial_nodes_coor_number)
                # shape: [B*(V-1), problem_size], 作用是把 props的概率填充到new_props里, props的里概率元素与未访问节点的概率一一对应.
                # 构造torch高级索引
                index_1_ = torch.arange(batch_size, dtype=torch.long)[:, None].expand(batch_size,
                                                                                      k_nearest_edges)  # shape: [B*(V-1), n]
                index_2_ = sort_index.type(torch.long)

                new_props[index_1_, index_2_] = props.reshape(batch_size, sort_index.shape[1])

                props = new_props

        else:



            enc_current_node = _get_encoding(encoded_nodes, abs_scatter_solu_1_seleted)
            enc_unseleted_scatter_node = _get_encoding(encoded_nodes, abs_scatter_solu_1_unseleted)
            enc_partial_nodes = _get_encoding(encoded_nodes, abs_partial_solu_2)

            embedded_last_node_ = self.embedding_last_node(enc_current_node)
            # enc_partial_nodes        = self.embedding_partial_node(enc_partial_nodes)
            enc_unseleted_scatter_node = self.embedding_scatter_node(enc_unseleted_scatter_node)

            left_encoded_node = enc_partial_nodes

            left_encoded_node = torch.cat((left_encoded_node, torch.roll(left_encoded_node, dims=1, shifts=-1)), dim=2)

            left_encoded_node = self.embedding_partial_node(left_encoded_node)

            out = torch.cat((embedded_last_node_, enc_unseleted_scatter_node, left_encoded_node), dim=1)

            # print(out.shape)

            layer_count = 0

            for layer in self.layers:
                out = layer(out)
                layer_count += 1
            num = enc_unseleted_scatter_node.shape[1] + 1
            # num = 1
            out = out[:, num:]

            out = self.Linear_final(out).squeeze(-1)  # shape: [B*(V-1), reminding_nodes_number + 2, embedding_dim ]

            props = F.softmax(out, dim=-1)  # shape: [B, remind_nodes_number]


        return props



def _get_new_data(data, selected_node_list, prob_size, B_V):
    list = selected_node_list

    new_list = torch.arange(prob_size)[None, :].repeat(B_V, 1)

    new_list_len = prob_size - list.shape[1]  # shape: [B, V-current_step]

    index_2 = list.type(torch.long)

    index_1 = torch.arange(B_V, dtype=torch.long)[:, None].expand(B_V, index_2.shape[1])

    new_list[index_1, index_2] = -2

    unselect_list = new_list[torch.gt(new_list, -1)].view(B_V, new_list_len)

    # ----------------------------------------------------------------------------

    new_data = data

    emb_dim = data.shape[-1]

    new_data_len = new_list_len

    index_2_ = unselect_list.repeat_interleave(repeats=emb_dim, dim=1)

    index_1_ = torch.arange(B_V, dtype=torch.long)[:, None].expand(B_V, index_2_.shape[1])

    index_3_ = torch.arange(emb_dim)[None, :].repeat(repeats=(B_V, new_data_len))

    new_data_ = new_data[index_1_, index_2_, index_3_].view(B_V, new_data_len, emb_dim)

    return new_data_

def _get_encoding(encoded_nodes, node_index_to_pick):

    batch_size = node_index_to_pick.size(0)
    pomo_size = node_index_to_pick.size(1)
    embedding_dim = encoded_nodes.size(2)

    gathering_index = node_index_to_pick[:, :, None].expand(batch_size, pomo_size, embedding_dim)

    picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)

    return picked_nodes


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        self.pe = torch.zeros(1,max_len, d_model,requires_grad=False)
        self.pe[0, :, 0::2] = torch.sin(position * div_term)
        self.pe[0, :, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe/d_model

    def forward(self, x):
        x = x + self.pe[:,:x.size(1),:].repeat(x.size(0),1,1)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.feedForward = Feed_Forward_Module_enc(**model_params)

    def forward(self, input1):
        head_num = self.model_params['head_num']

        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(input1), head_num=head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=head_num)

        out_concat = multi_head_attention(q, k, v)

        multi_head_out = self.multi_head_combine(out_concat)

        out1 = input1 + multi_head_out
        out2 = self.feedForward(out1)
        out3 = out1 + out2
        return out3


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

        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.feedForward = Feed_Forward_Module(**model_params)





        # self.feedForward_2 = Feed_Forward_Module(**model_params)
        # self.addAndNormalization2 = Add_And_Normalization_Module(**model_params)

    def forward(self, input2):
        # input.shape: (batch, problem, EMBEDDING_DIM)
        head_num = self.model_params['head_num']

        q = reshape_by_heads(self.Wq(input2), head_num=head_num)
        k = reshape_by_heads(self.Wk(input2), head_num=head_num)
        v = reshape_by_heads(self.Wv(input2), head_num=head_num)
        # q shape: (batch, HEAD_NUM, problem, KEY_DIM)

        out_concat = multi_head_attention(q, k, v)  # shape: (B, n, head_num*key_dim)

        multi_head_out = self.multi_head_combine(out_concat)  # shape: (B, n, embedding_dim)

        out1 = input2 + multi_head_out
        out2 = self.feedForward(out1)
        out3 = out1 + out2


        return out3
        # shape: (batch, problem, EMBEDDING_DIM)


def reshape_by_heads(qkv, head_num):
    batch_s = qkv.size(0)
    n = qkv.size(1)

    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)

    q_transposed = q_reshaped.transpose(1, 2)

    return q_transposed


def multi_head_attention(q, k, v):
    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)

    input_s = k.size(2)

    score = torch.matmul(q, k.transpose(2, 3))  # shape: (B, head_num, n, n)

    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))

    weights = nn.Softmax(dim=3)(score_scaled)  # shape: (B, head_num, n, n)

    out = torch.matmul(weights, v)  # shape: (B, head_num, n, key_dim)

    out_transposed = out.transpose(1, 2)  # shape: (B, n, head_num, key_dim)

    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)  # shape: (B, n, head_num*key_dim)

    return out_concat


class Feed_Forward_Module_enc(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        # input.shape: (batch, problem, embedding)

        return self.W2(F.relu(self.W1(input1)))



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

def make_dir(path_destination):
    isExists = os.path.exists(path_destination)
    if not isExists:
        os.makedirs(path_destination)
    return

def drawPic_v1(arr_, solution, partial_tour, scatters,abs_scatter_solu_seleted, partial_end_node_coor, name='xx'):

    optimal_tour = solution.clone().cpu().numpy()
    arr = arr_.clone().cpu().numpy()


    partial_tour = partial_tour.clone().cpu().numpy()

    scatters = scatters.clone().cpu().numpy()
    partial_end_node_coor = partial_end_node_coor.clone().cpu().numpy()
    #------------------------
    # ------------------------

    fig, ax = plt.subplots(figsize=(20, 20))

    plt.scatter(arr[:, 0], arr[:, 1], color='black', linewidth=1)

    plt.scatter(partial_end_node_coor[0], partial_end_node_coor[1], color='pink', linewidth=10)

    plt.scatter(arr[abs_scatter_solu_seleted, 0], arr[abs_scatter_solu_seleted, 1], color='orange', linewidth=10)

    tour_optimal = np.array(optimal_tour, dtype=int)
    start = [arr[optimal_tour[0], 0], arr[optimal_tour[-1], 0]]
    end = [arr[optimal_tour[0], 1], arr[optimal_tour[-1], 1]]

    plt.plot(start, end, color='red', linewidth=2, )  # linestyle="dashed"

    if True:
        for i in range(len(optimal_tour) - 1):
            start_optimal = [arr[tour_optimal[i], 0], arr[tour_optimal[i + 1], 0]]
            end_optimal = [arr[tour_optimal[i], 1], arr[tour_optimal[i + 1], 1]]
            plt.plot(start_optimal, end_optimal, color='green', linewidth=1)

    # 连接各个散点
    for i in range(len(scatters) - 1):
        start = [arr[scatters[i], 0], arr[scatters[i + 1], 0]]
        end = [arr[scatters[i], 1], arr[scatters[i + 1], 1]]
        plt.plot(start, end, color='red', linewidth=2)  # ,linestyle ="dashed"
    # 连接partial_tour
    partial_tour = np.array(partial_tour, dtype=int)
    for i in range(len(partial_tour) - 1):
        start = [arr[partial_tour[i], 0], arr[partial_tour[i + 1], 0]]
        end = [arr[partial_tour[i], 1], arr[partial_tour[i + 1], 1]]
        plt.plot(start, end, color='blue', linewidth=2)  # ,linestyle ="dashed"


    plt.axis('off')
    # 连接起点和终点

    b = os.path.abspath(".")
    path = b + '/figure'
    make_dir(path)
    plt.savefig(path + f'/{name}.pdf', bbox_inches='tight', pad_inches=0)


def drawPic_v2(arr_, solution, partial_tour, scatters_unseleted, abs_scatter_solu_seleted, name='xx'):
    #             drawPic_v2(data[1], abs_solution[1], extend_partial_solution[1], abs_scatter_solu_1_unseleted[1],abs_scatter_solu_1_seleted[1],
    #                        name=str(current_step))
    optimal_tour = solution.clone().cpu().numpy()
    arr = arr_.clone().cpu().numpy()


    partial_tour = partial_tour.clone().cpu().numpy()

    scatters_unseleted = scatters_unseleted.clone().cpu().numpy()

    #------------------------
    # ------------------------

    fig, ax = plt.subplots(figsize=(20, 20))

    plt.scatter(arr[:, 0], arr[:, 1], color='black', linewidth=1)

    plt.scatter(arr[abs_scatter_solu_seleted, 0], arr[abs_scatter_solu_seleted, 1], color='orange', linewidth=10)

    tour_optimal = np.array(optimal_tour, dtype=int)
    start = [arr[optimal_tour[0], 0], arr[optimal_tour[-1], 0]]
    end = [arr[optimal_tour[0], 1], arr[optimal_tour[-1], 1]]

    plt.plot(start, end, color='red', linewidth=2, )  # linestyle="dashed"

    if True:
        for i in range(len(optimal_tour) - 1):
            start_optimal = [arr[tour_optimal[i], 0], arr[tour_optimal[i + 1], 0]]
            end_optimal = [arr[tour_optimal[i], 1], arr[tour_optimal[i + 1], 1]]
            plt.plot(start_optimal, end_optimal, color='green', linewidth=1)

    # 连接各个散点
    for i in range(len(scatters_unseleted) - 1):
        plt.scatter(arr[scatters_unseleted[i], 0], arr[scatters_unseleted[i], 1], color='red', linewidth=1)

    # 连接partial_tour
    partial_tour = np.array(partial_tour, dtype=int)
    for i in range(len(partial_tour) - 1):
        start = [arr[partial_tour[i], 0], arr[partial_tour[i + 1], 0]]
        end = [arr[partial_tour[i], 1], arr[partial_tour[i + 1], 1]]
        plt.plot(start, end, color='blue', linewidth=2)  # ,linestyle ="dashed"


    plt.axis('off')
    # 连接起点和终点

    b = os.path.abspath(".")
    path = b + '/figure'
    make_dir(path)
    plt.savefig(path + f'/test_{name}.pdf', bbox_inches='tight', pad_inches=0)




def normalize(coor1, coor2, coor3, coor4):
    # len1 = coor1.shape[1]
    # len2 = coor2.shape[1]
    # len3 = coor3.shape[1]
    lengths = [coor1.shape[1], coor2.shape[1], coor3.shape[1], coor4.shape[1]]
    all_coors = torch.cat((coor1, coor2, coor3, coor4), dim=1)
    min_vals, _ = torch.min(all_coors, dim=1, keepdim=True)
    # min_vals_2,_ = torch.min(min_vals,dim=2,keepdim=True)
    # min_vals = min_vals_2.repeat(1,1,2)

    max_vals, _ = torch.max(all_coors, dim=1, keepdim=True)
    # max_vals_2,_ = torch.max(max_vals,dim=2,keepdim=True)
    # max_vals = max_vals_2.repeat(1,1,2)

    normalized_tensor = (all_coors - min_vals) / (max_vals - min_vals)

    # min_vals= torch.min(all_coors.ravel())
    # max_vals= torch.max(all_coors.ravel())
    # normalized_tensor = (all_coors - min_vals) / (max_vals - min_vals)

    # coor1 = normalized_tensor[:,               :len1            ,:]
    # coor2 = normalized_tensor[:, len1          : len1+len2      , :]
    # coor3 = normalized_tensor[:, len1+len2     : len1+len2+len3 , :]
    # coor4 = normalized_tensor[:, len1+len2+len3:                , :]
    coor1, coor2, coor3, coor4 = torch.split(normalized_tensor, lengths, dim=1)

    return coor1, coor2, coor3, coor4
