import time
import torch
from logging import getLogger
from tqdm import tqdm
from TSPEnv import TSPEnv as Env
from utils.utils import *
from TSPModel_Upper import TSPUpperModel as UpperModel
from TSPModel_Lower import TSPLowerModel as LowerModel
from TSProblemDef import get_saved_tsp_data


class TSPTester:
    def __init__(self,
                 env_params,
                 model_params,
                 tester_params):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.tester_params = tester_params

        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()


        # cuda
        USE_CUDA = self.tester_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.tester_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
        self.device = device

        self.env_params['device'] = device
        self.model_params['device'] = device
        # ENV and MODEL
        self.env = Env(**self.env_params)
        self.upper_model = UpperModel(**self.model_params)
        self.lower_model = LowerModel(**self.model_params)

        # Restore
        model_load = tester_params['model_load']
        if model_load['epoch'] == 'best':
            checkpoint_fullname = '{path}/best_model.pt'.format(**model_load)
        else:
            checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)

        self.logger.info("Load model from: {}".format(checkpoint_fullname))

        checkpoint = torch.load(checkpoint_fullname, map_location=device)
        self.upper_model.load_state_dict(checkpoint['upper_model_state_dict'])
        self.lower_model.load_state_dict(checkpoint['lower_model_state_dict'])
        total_params = list(self.upper_model.parameters()) + list(self.lower_model.parameters())

        total = sum([param.nelement() for param in total_params])
        self.logger.info("Number of parameters: %.2fM" % (total / 1e6))

        # utility
        self.time_estimator = TimeEstimator()

    def run(self):
        self.time_estimator.reset()

        score_AM = AverageMeter()

        test_num_episode = self.tester_params['test_episodes']
        self.batch_selected_node_list = None
        optimal = 1.0
        if self.tester_params['test_data_load']['enable']:
            file_name = self.tester_params['test_data_load']['filename']
            solution_file = self.tester_params['test_data_load']['solution_filename']
            problems, solutions, optimal = get_saved_tsp_data(file_name, test_num_episode, self.device,start=0, solution_file=solution_file)
            self.env.input_saved_tsp_data(problems,solutions,self.device)
            self.logger.info("Saved dataset loaded successfully!!!")
            self.logger.info("Data loaded from: {0}".format(file_name))
            if isinstance(optimal, torch.Tensor):
                self.logger.info('problem_size: {0} ,optimal : {1:.4f}'.format(self.env.saved_node_xy.shape[1], optimal.mean().item()))
            else:
                self.logger.info('problem_size: {0} ,optimal : {1:.4f}'.format(self.env.saved_node_xy.shape[1], optimal))

        episode = 0
        all_scores_tensor = torch.zeros(size=(test_num_episode, ), device=self.device)
        self.start_time = time.time()
        while episode < test_num_episode:

            remaining = test_num_episode - episode
            batch_size = min(self.tester_params['test_batch_size'], remaining)

            scores = self._test_one_batch(batch_size)
            # shape: (batch,)
            
            score_AM.update(scores.mean().item(), batch_size)
            all_scores_tensor[episode:episode+batch_size] = scores

            episode += batch_size

            ############################
            # Logs
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(episode, test_num_episode)
            self.logger.info(" episode {:3d}/{:3d}, Elapsed[{}], Remain[{}], avg.score:{:.3f}".format(
                episode, test_num_episode, elapsed_time_str, remain_time_str, scores.mean().item()))
            
            self.logger.info("All AVG.SCORE:{:.4f} ".format(score_AM.avg))

            all_done = (episode == test_num_episode)

            if all_done:
                end_time = time.time()
                self.logger.info(" *** Test Done *** ")
                
                if isinstance(optimal, torch.Tensor):
                    if optimal.device != all_scores_tensor.device:
                        optimal = optimal.to(all_scores_tensor.device)
                    gaps = (all_scores_tensor - optimal) / optimal * 100
                    gap = gaps.mean().item()
                    optimal_mean = optimal.mean().item()
                else:
                    optimal_mean = optimal
                    gap = (score_AM.avg - optimal_mean) * 100 / optimal_mean

                distribution = self.env_params['distribution']

                self.logger.info("===============================================================")
                self.logger.info(" optimal score: {0:.4f} ".format(optimal_mean))
                self.logger.info("distribution: {0}".format(distribution))

                self.logger.info(" SCORE:{:.4f}, 2 decimal places:{:.2f}".format(score_AM.avg, score_AM.avg))
                self.logger.info(" problem_size: {0} ,model gap:{1:.3f}%, 2 decimal places:{2:.2f}%".format(
                    self.env_params['problem_size'], gap, gap))

                self.logger.info("total time: {:.2f} sec".format(end_time - self.start_time))
                self.logger.info("total time: {:.2f} min".format((end_time - self.start_time) / 60))
                self.logger.info("avg time: {:.2f} sec".format((end_time - self.start_time) / test_num_episode))
                self.logger.info("avg time: {:.2f} mins".format((end_time - self.start_time) / test_num_episode / 60))

    def _test_one_batch(self, batch_size):

            # Ready
            ###############################################
            self.upper_model.eval()
            self.lower_model.eval()
            self.upper_model.set_decoder_method('greedy')
            self.env.load_problems_tsp(batch_size, problem_size=self.env_params['problem_size'])
            # reset peak memory stats
            torch.cuda.reset_peak_memory_stats(device=self.device)

            reset_state, _, _ = self.env.reset()
            with torch.no_grad():
                self.upper_model.pre_forward(reset_state)
                # AM Rollout
                ###############################################
                state, reward, done = self.env.pre_step()
                with tqdm(total=0) as pbar:
                    while not done:
                        if state.current_node is not None:
                            step1_start = time.time()
                            state = self.env.get_upper_input()
                            upper_scores,_,_ = self.upper_model(state)
                            self.env.update_cur_scores(upper_scores=upper_scores)
                        state = self.env.get_lower_transformed_neighbors()
                        low_selected, _ = self.lower_model(state)
                        # shape: (batch,)
                        state, reward, done = self.env.step(low_selected)
                        # shape: (batch,)
                        pbar.total += 1
                        pbar.update(1)

            batch_memory = torch.cuda.max_memory_allocated(device=self.device) / 1024 / 1024
            self.logger.info("batch_memory:{:.2f}MB".format(batch_memory))
            self.logger.info("batch_memory_GB:{:.2f}GB".format(batch_memory / 1024))
            self.logger.info("avg_memory:{:.2f}MB".format(batch_memory / batch_size))

            # Return
            ###############################################
            scores = -reward.float()  # negative sign to make positive value
            # shape: (batch,)

            return scores

