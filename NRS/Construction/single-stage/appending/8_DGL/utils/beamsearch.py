import numpy as np
import torch


# import objgraph
import sys

class Beamsearch(object):
    """Class for managing internals of beamsearch procedure.

    References:
        General: https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/translate/beam.py
        For TSP: https://github.com/alexnowakvila/QAP_pt/blob/master/src/tsp/beam_search.py
    """

    def __init__(self, beam_size, batch_size, num_nodes,
                 dtypeFloat=torch.FloatTensor, dtypeLong=torch.LongTensor, 
                 probs_type='raw', random_start=False, problem="TSP"):
        """
        Args:
            beam_size: Beam size
            batch_size: Batch size
            num_nodes: Number of nodes in TSP tours
            dtypeFloat: Float data type (for GPU/CPU compatibility)
            dtypeLong: Long data type (for GPU/CPU compatibility)
            probs_type: Type of probability values being handled by beamsearch (either 'raw'/'logits'/'argmax'(TODO))
            random_start: Flag for using fixed (at node 0) vs. random starting points for beamsearch
        """
        # Beamsearch parameters
        self.batch_size = batch_size
        self.beam_size = beam_size
        self.num_nodes = num_nodes
        self.probs_type = probs_type
        # Set data types
        self.dtypeFloat = dtypeFloat
        self.dtypeLong = dtypeLong
        # Set beamsearch starting nodes
        self.start_nodes = torch.zeros(batch_size, beam_size).type(self.dtypeLong)
        # Set problem
        self.problem = problem
        if random_start == True:
            # Random starting nodes
            self.start_nodes = torch.randint(0, num_nodes, (batch_size, beam_size)).type(self.dtypeLong)
        # Score for each translation on the beam
        self.scores = torch.zeros(batch_size, beam_size).type(self.dtypeFloat)
        self.all_scores = []
        # Backpointers at each time-step
        self.prev_Ks = []
        # Outputs at each time-step
        self.next_nodes = [self.start_nodes]

    def advance(self, trans_probs, env,knn, current_step):
        """Advances the beam based on transition probabilities.

        Args:
            trans_probs: Probabilities of advancing from the previous step (batch_size, beam_size, num_nodes)
        """

        selected_nodes = env.selected_node_list

        # if current_step == 1:
        #     print("trans_probs", trans_probs[0])

        # Compound the previous scores (summing logits == multiplying probabilities)
        if current_step >= 2:
            beam_lk = trans_probs + self.scores.unsqueeze(2).expand_as(trans_probs)
        else:
            beam_lk = trans_probs
            beam_lk[:, 1:] = -1e20 * torch.ones(beam_lk[:, 1:].size()).type(self.dtypeFloat)

        # print("beam_lk before", beam_lk[-9,3,33].tolist())

        beam_lk = beam_lk.view(self.batch_size, -1)  # (batch_size, beam_size * num_nodes)

        # print("beam_lk after", beam_lk[-9].tolist())

        if torch.isnan(beam_lk).any():
            print("beam_lk: ", beam_lk)
            exit(0)
        
        bestScores, bestScoresId = beam_lk.topk(self.beam_size, 1, True, True)

        

        # Update scores
        self.scores = bestScores

        # Update backpointers
        if self.problem == "TSP":
            prev_k = torch.div(bestScoresId, self.num_nodes, rounding_mode='trunc')
        elif self.problem == "CVRP":
            prev_k = torch.div(bestScoresId, self.num_nodes*2, rounding_mode='trunc')

        self.prev_Ks.append(prev_k)
        # Update outputs
        if self.problem == "TSP":
            new_nodes = bestScoresId - prev_k * self.num_nodes
        elif self.problem == "CVRP":
            new_nodes = bestScoresId - prev_k * self.num_nodes * 2

            # print("new_nodes: ", new_nodes.shape)

        # if current_step == 1:
        # print("new_nodes", new_nodes)

        self.next_nodes.append(new_nodes)

        # Re-index selected nodes


        selected_nodes = selected_nodes.view(self.batch_size, self.beam_size, -1)
        perm_selected_nodes = prev_k.unsqueeze(2).expand_as(selected_nodes)
        selected_nodes = selected_nodes.gather(1, perm_selected_nodes)
        selected_nodes = selected_nodes.view(self.batch_size*self.beam_size, -1)

        env.perm_attr("data", prev_k)

        if self.problem == "CVRP":
            env.perm_attr("capacity", prev_k)
            env.perm_attr("ninf_mask", prev_k)
            env.perm_attr("avg_unselect_distence", prev_k)
            env.perm_attr("std_dev_unselect_distence", prev_k)
            env.perm_attr("selected_flag", prev_k)
            # env.perm_attr("sum_demand_aggregation", prev_k)
            # env.perm_attr("knn_count", prev_k)

        if self.problem == "TSP":
            env.perm_attr("avg_unselect_distence", prev_k)
            env.perm_attr("std_dev_unselect_distence", prev_k)

        return selected_nodes


