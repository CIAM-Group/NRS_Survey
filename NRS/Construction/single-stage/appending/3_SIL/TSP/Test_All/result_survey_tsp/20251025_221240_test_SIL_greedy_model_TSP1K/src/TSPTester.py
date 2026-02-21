
import torch

import os
from logging import getLogger

from TSP.Test_All.TSPEnv import TSPEnv as Env
from TSP.Test_All.TSPModel import TSPModel as Model

from utils.utils import *
import random



tsplib_cost = {
    # TSPLIB, 77+4, all optimal, http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/STSP.html
    "tsplib_euc_a280": 2579,
    # "ali535": 202339,
    # "att48": 10628,
    # "att532": 27686,
    # "bayg29": 1610,
    # "bays29": 2020,
    "tsplib_euc_berlin52": 7542,
    "tsplib_euc_bier127": 118282,
    # "brazil58": 25395,
    "tsplib_euc_brd14051": 469385,
    # "brg180": 1950,
    # "burma14": 3323,
    "tsplib_euc_ch130": 6110,
    "tsplib_euc_ch150": 6528,
    "tsplib_euc_d198": 15780,
    "tsplib_euc_d493": 35002,
    "tsplib_euc_d657": 48912,
    "tsplib_euc_d1291": 50801,
    "tsplib_euc_d1655": 62128,
    "tsplib_euc_d2103": 80450,
    "tsplib_euc_d15112": 1573084,
    "tsplib_euc_d18512": 645238,
    # "dantzig42": 699,
    # "dsj1000": 18659688, # (EUC_2D)
    "tsplib_ceil_dsj1000": 18660188, # (CEIL_2D)
    "tsplib_euc_eil51": 426,
    "tsplib_euc_eil76": 538,
    "tsplib_euc_eil101": 629,
    "tsplib_euc_fl417": 11861,
    "tsplib_euc_fl1400": 20127,
    "tsplib_euc_fl1577": 22249,
    "tsplib_euc_fl3795": 28772,
    "tsplib_euc_fnl4461": 182566,
    # "fri26": 937,
    "tsplib_euc_gil262": 2378,
    # "gr17": 2085,
    # "gr21": 2707,
    # "gr24": 1272,
    # "gr48": 5046,
    # "gr96": 55209,
    # "gr120": 6942,
    # "gr137": 69853,
    # "gr202": 40160,
    # "gr229": 134602,
    # "gr431": 171414,
    # "gr666": 294358,
    # "hk48": 11461,
    "tsplib_euc_kroA100": 21282,
    "tsplib_euc_kroB100": 22141,
    "tsplib_euc_kroC100": 20749,
    "tsplib_euc_kroD100": 21294,
    "tsplib_euc_kroE100": 22068,
    "tsplib_euc_kroA150": 26524,
    "tsplib_euc_kroB150": 26130,
    "tsplib_euc_kroA200": 29368,
    "tsplib_euc_kroB200": 29437,
    "tsplib_euc_lin105": 14379,
    "tsplib_euc_lin318": 42029,
    # "tsplib_euc_linhp318": 41345,
    "tsplib_euc_nrw1379": 56638,
    "tsplib_euc_p654": 34643,
    # "pa561": 2763,
    "tsplib_euc_pcb442": 50778,
    "tsplib_euc_pcb1173": 56892,
    "tsplib_euc_pcb3038": 137694,
    "tsplib_ceil_pla7397": 23260728, # (CEIL_2D)
    "tsplib_ceil_pla33810": 66048945, # (CEIL_2D)
    "tsplib_ceil_pla85900": 142382641, # (CEIL_2D)
    "tsplib_euc_pr76": 108159,
    "tsplib_euc_pr107": 44303,
    "tsplib_euc_pr124": 59030,
    "tsplib_euc_pr136": 96772,
    "tsplib_euc_pr144": 58537,
    "tsplib_euc_pr152": 73682,
    "tsplib_euc_pr226": 80369,
    "tsplib_euc_pr264": 49135,
    "tsplib_euc_pr299": 48191,
    "tsplib_euc_pr439": 107217,
    "tsplib_euc_pr1002": 259045,
    "tsplib_euc_pr2392": 378032,
    "tsplib_euc_rat99": 1211,
    "tsplib_euc_rat195": 2323,
    "tsplib_euc_rat575": 6773,
    "tsplib_euc_rat783": 8806,
    "tsplib_euc_rd100": 7910,
    "tsplib_euc_rd400": 15281,
    "tsplib_euc_rl1304": 252948,
    "tsplib_euc_rl1323": 270199,
    "tsplib_euc_rl1889": 316536,
    "tsplib_euc_rl5915": 565530,
    "tsplib_euc_rl5934": 556045,
    "tsplib_euc_rl11849": 923288,
    # "si175": 21407,
    # "si535": 48450,
    # "si1032": 92650,
    "tsplib_euc_st70": 675,
    # "swiss42": 1273,
    "tsplib_euc_ts225": 126643,
    "tsplib_euc_tsp225": 3916,
    "tsplib_euc_u159": 42080,
    "tsplib_euc_u574": 36905,
    "tsplib_euc_u724": 41910,
    "tsplib_euc_u1060": 224094,
    "tsplib_euc_u1432": 152970,
    "tsplib_euc_u1817": 57201,
    "tsplib_euc_u2152": 64253,
    "tsplib_euc_u2319": 234256,
    # "ulysses16": 6859,
    # "ulysses22": 7013,
    "tsplib_euc_usa13509": 19982859,
    "tsplib_euc_vm1084": 239297,
    "tsplib_euc_vm1748": 336556,

    # National TSP, 27, 2 non-optimal, https://www.math.uwaterloo.ca/tsp/world/summary.html
    'national_ar9152': 837_479,
    'national_bm33708': 959_289, # gap 0.031%
    'national_ca4663': 1_290_319,
    'national_ch71009': 4_566_506, # gap 0.024%
    'national_dj38': 6_656,
    'national_eg7146': 172_386,
    'national_fi10639': 520_527,
    'national_gr9882': 300_899,
    'national_ho14473': 177_092,
    'national_ei8246': 206_171,
    'national_it16862': 557_315,
    'national_ja9847': 491_924,
    'national_kz9976': 1_061_881,
    'national_lu980': 11_340,
    'national_mo14185': 427_377,
    'national_nu3496': 96_132,
    'national_mu1979': 86_891,
    "national_pm8079": 114_855,
    'national_qa194': 9_352,
    'national_rw1621': 26_051,
    'national_sw24978': 855_597,
    'national_tz6117': 394_718,
    'national_uy734': 79_114,
    'national_vm22775': 569_288,
    'national_wi29': 27_603,
    'national_ym7663': 238_314,
    'national_zi929': 95_345,

    # VLSI, 102-4, non-optimal when size >= 14233 (xrb14233), https://www.math.uwaterloo.ca/tsp/vlsi/summary.html
    'vlsi_xqf131': 564,
    'vlsi_xqg237': 1_019,
    'vlsi_pma343': 1_368,
    'vlsi_pka379': 1_332,
    'vlsi_bcl380': 1_621,
    'vlsi_pbl395': 1_281,
    'vlsi_pbk411': 1_343,
    'vlsi_pbn423': 1_365,
    'vlsi_pbm436': 1_443,
    'vlsi_xql662': 2_513,
    'vlsi_rbx711': 3_115,
    'vlsi_rbu737': 3_314,
    'vlsi_dkg813': 3_199,
    'vlsi_lim963': 2_789,
    'vlsi_pbd984': 2_797,
    'vlsi_xit1083': 3_558,
    'vlsi_dka1376': 4_666,
    'vlsi_dca1389': 5_085,
    'vlsi_dja1436': 5_257,
    'vlsi_icw1483': 4_416,
    'vlsi_fra1488': 4_264,
    'vlsi_rbv1583': 5_387,
    'vlsi_rby1599': 5_533,
    'vlsi_fnb1615': 4_956,
    'vlsi_djc1785': 6_115,
    'vlsi_dcc1911': 6_396,
    'vlsi_dkd1973': 6_421,
    'vlsi_djb2036': 6_197,
    'vlsi_dcb2086': 6_600,
    'vlsi_bva2144': 6_304,
    'vlsi_xqc2175': 6_830,
    'vlsi_bck2217': 6_764,
    'vlsi_xpr2308': 7_219,
    'vlsi_ley2323': 8_352,
    'vlsi_dea2382': 8_017,
    'vlsi_rbw2481': 7_724,
    'vlsi_pds2566': 7_643,
    'vlsi_mlt2597': 8_071,
    'vlsi_bch2762': 8_234,
    'vlsi_irw2802': 8_423,
    'vlsi_lsm2854': 8_014,
    'vlsi_dbj2924': 10_128,
    'vlsi_xva2993': 8_492,
    'vlsi_pia3056': 8_258,
    'vlsi_dke3097': 10_539,
    'vlsi_lsn3119': 9_114,
    'vlsi_lta3140': 9_517,
    'vlsi_fdp3256': 10_008,
    'vlsi_beg3293': 9_772,
    'vlsi_dhb3386': 11_137,
    'vlsi_fjs3649': 9_272,
    'vlsi_fjr3672': 9_601,
    'vlsi_dlb3694': 10_959,
    'vlsi_ltb3729': 11_821,
    'vlsi_xqe3891': 11_995,
    'vlsi_xua3937': 11_239,
    'vlsi_dkc3938': 12_503,
    'vlsi_dkf3954': 12_538,
    'vlsi_bgb4355': 12_723,
    'vlsi_bgd4396': 13_009,
    'vlsi_frv4410': 10_711,
    'vlsi_bgf4475': 13_221,
    'vlsi_xqd4966': 15_316,
    'vlsi_fqm5087': 13_029,
    'vlsi_fea5557': 15_445,
    'vlsi_xsc6880': 21_535,
    'vlsi_bnd7168': 21_834,
    'vlsi_lap7454': 19_535,
    'vlsi_ida8197': 22_338,
    'vlsi_dga9698': 27_724,
    'vlsi_xmc10150': 28_387,
    'vlsi_xvb13584': 37_083,
    'vlsi_xrb14233': 45_462, # gap 0.026%
    'vlsi_xia16928': 52_850, # gap 0.023%
    'vlsi_pjh17845': 48_092, # gap 0.019%
    'vlsi_frh19289': 55_798, # gap 0.013%
    'vlsi_fnc19402': 59_287, # gap 0.020%
    'vlsi_ido21215': 63_517, # gap 0.028%
    'vlsi_fma21553': 66_527, # gap unknown
    'vlsi_lsb22777': 60_977, # gap unknown
    'vlsi_xrh24104': 69_294, # gap unknown
    'vlsi_bbz25234': 69_335, # gap unknown
    'vlsi_irx28268': 72_607, # gap unknown
    'vlsi_fyg28534': 78_562, # gap unknown
    'vlsi_icx28698': 78_087, # gap unknown
    'vlsi_boa28924': 79_622, # gap unknown
    'vlsi_ird29514': 80_353, # gap unknown
    'vlsi_pbh30440': 88_313, # gap unknown
    'vlsi_xib32892': 96_757, # gap unknown
    'vlsi_fry33203': 97_240, # gap unknown
    'vlsi_bby34656': 99_159, # gap unknown
    'vlsi_pba38478': 108_318, # gap unknown
    'vlsi_ics39603': 106_819, # gap unknown
    'vlsi_rbz43748': 125_183, # gap unknown
    'vlsi_fht47608': 125_104, # gap unknown
    'vlsi_fna52057': 147_789, # gap unknown
    'vlsi_bna56769': 158_078, # gap unknown
    'vlsi_dan59296': 165_371, # gap unknown
    # 'vlsi_sra104815': 251_342, # gap unknown
    # 'vlsi_ara238025': 578_761, # gap unknown
    # 'vlsi_lra498378': 2_168_039, # gap unknown
    # 'vlsi_lrb744710': 1_611_232, # gap unknown

    # DIMACS 8th Challenge, non-optimal, http://dimacs.rutgers.edu/archive/Challenges/TSP/opts.html and http://webhotel4.ruc.dk/~keld/research/LKH/DIMACS_results.html
    "challenge8_C1k.0": 11387430, # gap 0.54%
    "challenge8_C1k.1": 11376735, # gap 0.41%
    "challenge8_C1k.2": 10855033, # gap 0.42%
    "challenge8_C1k.3": 11886457, # gap 0.53%
    "challenge8_C1k.4": 11499958, # gap 0.58%
    "challenge8_C1k.5": 11394911, # gap 0.58%
    "challenge8_C1k.6": 10166701, # gap 0.73%
    "challenge8_C1k.7": 10664660, # gap 0.58%
    "challenge8_C1k.8": 11605723, # gap 0.34%
    "challenge8_C1k.9": 10906997, # gap 0.66%
    "challenge8_C3k.0": 19198258, # gap 0.62%
    "challenge8_C3k.1": 19017805, # gap 0.61%
    "challenge8_C3k.2": 19547551, # gap 0.70%
    "challenge8_C3k.3": 19108508, # gap 0.57%
    "challenge8_C3k.4": 18864046, # gap 0.57%
    "challenge8_C10k.0": 33_001_034, # gap 0.668%
    "challenge8_C10k.1": 33_186_248, # gap 0.690%
    "challenge8_C10k.2": 33_155_424, # gap 0.694%
    "challenge8_C31k.0": 59_545_390, # gap 0.636%
    "challenge8_C31k.1": 59_293_266, # gap 0.770%
    "challenge8_C100k.0": 104_617_752, # gap 0.675%
    "challenge8_C100k.1": 105_390_777, # gap 0.695%
    # "challenge8_C316k.0": 186_870_839 # gap 0.697%

}




class TSPTester():
    def __init__(self,
                 env_params,
                 model_params,
                 tester_params,):

        self.env_params = env_params
        self.model_params = model_params
        self.tester_params = tester_params

        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()

        seed = 123
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

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

        self.env = Env(**self.env_params)
        self.model = Model(**self.model_params)

        keys = []
        numbers = []
        values = []
        import re
        # 遍历字典的每一对键和值
        for key, value in tsplib_cost.items():
            # 1. 将键和值分别添加到列表中
            keys.append(key)
            values.append(value)

            # 2. 从键中提取数字
            # re.search(r'\d+', key) 会在字符串中查找第一个连续的数字序列
            match = re.search(r'\d+', key)
            if match:
                # 如果找到数字，将其转换为整数并添加到列表
                number = int(match.group(0))
                numbers.append(number)
            else:
                # 如果键中没有数字，可以添加一个默认值，例如 None
                numbers.append(None)

        self.inst_names = np.array(keys)
        self.inst_problem_sizes = np.array(numbers)
        self.inst_opt_values = np.array(values)
        self.inst_total_num = len(self.inst_names)

        sort_indices = np.argsort(self.inst_problem_sizes)

        self.inst_names = self.inst_names[sort_indices]
        self.inst_problem_sizes = self.inst_problem_sizes[sort_indices]
        self.inst_opt_values = self.inst_opt_values[sort_indices]

        bb = os.path.abspath("../../..").replace('\\', '/')
        # test  TSPLIBReader

        validate_name = []
        validate_problem_sizes = []
        validate_opt_values = []

        for key, opt_val in tsplib_cost.items():
            filename = f'/public/home/bayp/exp_survey_202509/0_data_survey/survey_bench_tsp/{key}.tsp'
            name, dimension, locs = self.env.TSPLIBReader(filename)
            if name is not None and dimension is not None:
                validate_name.append(key)
                validate_problem_sizes.append(dimension)
                validate_opt_values.append(opt_val)

        self.inst_names = np.array(validate_name)
        self.inst_problem_sizes = np.array(validate_problem_sizes)
        self.inst_opt_values = np.array(validate_opt_values)

        # 按真实 DIMENSION 升序排序
        sort_indices = np.argsort(self.inst_problem_sizes)
        self.inst_names = self.inst_names[sort_indices]
        self.inst_problem_sizes = self.inst_problem_sizes[sort_indices]
        self.inst_opt_values = self.inst_opt_values[sort_indices]

        # ✅ 保留，统计数量
        self.inst_total_num = len(self.inst_names)





        model_load = tester_params['model_load']
        checkpoint_fullname = '{path}'.format(**model_load)

        checkpoint = torch.load(checkpoint_fullname, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        torch.set_printoptions(precision=20)

        self.time_estimator = TimeEstimator()
        self.time_estimator_2 =  TimeEstimator()
        total = sum([param.nelement() for param in self.model.parameters()])
        print("Number of parameter: %.5fM" % (total / 1e6))

        # ====== 新增：统计容器（分桶 + 总体）======
        self.bucket_gaps_no_aug = {  # 三个桶：与 ICAM 一致
            'lt_1000': [],
            'lt_10000': [],
            'lt_100000': [],
        }
        self.bucket_times = {
            'lt_1000': [],
            'lt_10000': [],
            'lt_100000': [],
        }
        self.all_gaps_no_aug = []      # 所有实例的 gap（no aug）
        self.instance_times = []       # 所有实例的单实例运行时长（秒）
        self.all_instance_num = 0      # 总实例数（包含失败的）
        self.all_solved_instance_num = 0  # 成功求解的实例数


    def run(self):
        self.time_estimator.reset()
        self.time_estimator_2.reset()


        self.env.load_raw_data(self.tester_params['test_episodes'] )

        k_nearest = self.env_params['k_nearest']

        decode_method = self.env_params['decode_method']

        score_AM = AverageMeter()
        score_student_AM = AverageMeter()
        aug_score_AM = AverageMeter()

        test_num_episode = self.inst_total_num # self.tester_params['test_episodes']
        episode = 0
        problems_le_5000 = []
        problems_gt_5000 = []

        while episode < test_num_episode:

            remaining = test_num_episode - episode
            batch_size = min(self.tester_params['test_batch_size'], remaining)

            score, score_student_mean, aug_score,problems_size = self._test_one_batch(episode,batch_size,k_nearest,decode_method,clock=self.time_estimator_2)

            print('max_memory_allocated',torch.cuda.max_memory_allocated(device=self.device ) / 1024 / 1024,'MB')

            if self.env.test_in_tsplib:
                if problems_size <= 1000:
                    problems_le_5000.append((score_student_mean - score) / score)
                elif 5000 < problems_size:
                    problems_gt_5000.append((score_student_mean - score) / score)

                print('problems_le_5000 mean gap:', np.mean(problems_le_5000), len(problems_le_5000))
                print('problems_gt_5000 mean gap:', np.mean(problems_gt_5000), len(problems_gt_5000))

            score_AM.update(score, batch_size)
            score_student_AM.update(score_student_mean, batch_size)
            aug_score_AM.update(aug_score, batch_size)

            episode += batch_size

            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(episode, test_num_episode)
            self.logger.info("episode {:3d}/{:3d}, Elapsed[{}], Remain[{}], score:{:.3f},Score_studetnt: {:.4f}, aug_score:{:.3f}".format(
                episode, test_num_episode, elapsed_time_str, remain_time_str, score,score_student_mean, aug_score))

            all_done = (episode == test_num_episode)

            if all_done and not self.env.test_in_tsplib:
                self.logger.info(" *** Test_All Done *** ")
                self.logger.info(" NO-AUG SCORE: {:.4f} ".format(score_AM.avg))
                self.logger.info(" NO-AUG SCORE student: {:.4f} ".format(score_student_AM.avg))

                self.logger.info(" Gap: {:.4f}%".format((score_student_AM.avg-score_AM.avg) / score_AM.avg * 100))
                gap_ = (score_student_AM.avg-score_AM.avg) / score_AM.avg * 100



        return score_AM.avg, score_student_AM.avg, gap_

    def run_lib(self):
        import time
        self.time_estimator.reset()
        self.time_estimator_2.reset()

        self.env.load_raw_data(self.tester_params['test_episodes'])

        k_nearest = self.env_params['k_nearest']
        decode_method = self.env_params['decode_method']

        score_AM = AverageMeter()
        score_student_AM = AverageMeter()
        aug_score_AM = AverageMeter()

        test_num_episode = self.inst_total_num
        episode = 0

        # ====== 新增：整体计时 ======
        total_start_time = time.time()

        while episode < test_num_episode:

            remaining = test_num_episode - episode
            batch_size = min(self.tester_params['test_batch_size'], remaining)

            # —— TSPLIB 场景下：一例一批；拿到名称与规模，便于日志/分桶
            inst_name = self.inst_names[episode]
            inst_size = int(self.inst_problem_sizes[episode])

            # 统计总实例数（包含失败）
            self.all_instance_num += 1

            # ====== 新增：单实例计时（含距离缩放与测试）======
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            inst_start = time.time()

            try:
                # 原有测试流程
                score, score_student_mean, aug_score, problems_size = self._test_one_batch(
                    episode, batch_size, k_nearest, decode_method, clock=self.time_estimator_2
                )

                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                inst_time = time.time() - inst_start

                # 成功实例数 +1，并记录时间
                self.all_solved_instance_num += 1
                self.instance_times.append(inst_time)

                # 计算 gap（%）
                no_aug_gap = (score_student_mean - score) * 100.0 / score

                # 分桶：与 ICAM 相同
                if inst_size < 1000:
                    bucket_key = 'lt_1000'
                elif 1000 <= inst_size < 10000:
                    bucket_key = 'lt_10000'
                else:  # 10000 <= inst_size <= 100000
                    bucket_key = 'lt_100000'

                self.bucket_gaps_no_aug[bucket_key].append(no_aug_gap)
                self.bucket_times[bucket_key].append(inst_time)
                self.all_gaps_no_aug.append(no_aug_gap)

                # —— 原有统计更新
                score_AM.update(score, batch_size)
                score_student_AM.update(score_student_mean, batch_size)
                aug_score_AM.update(aug_score, batch_size)

                # —— 日志：单实例
                self.logger.info(
                    f"[OK] inst:{inst_name}, n:{inst_size}, gap:{no_aug_gap:.3f}%, time:{inst_time:.3f}s"
                )

            except Exception as e:
                # 失败：记录错误但不中断
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                inst_time = time.time() - inst_start
                self.instance_times.append(inst_time)

                self.logger.info(
                    f"[ERR] inst:{inst_name}, n:{inst_size}, skip. err={repr(e)}, time:{inst_time:.3f}s"
                )
                # 跳过本实例，不更新均值统计
            finally:
                # 推进 episode
                episode += batch_size

                # 原有进度日志
                elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(episode, test_num_episode)
                self.logger.info(
                    "episode {:3d}/{:3d}, Elapsed[{}], Remain[{}], score:{:.3f},Score_studetnt: {:.4f}, aug_score:{:.3f}".format(
                        episode, test_num_episode, elapsed_time_str, remain_time_str,
                        score_AM.avg if score_AM.count>0 else float('nan'),
                        score_student_AM.avg if score_student_AM.count>0 else float('nan'),
                        aug_score_AM.avg if aug_score_AM.count>0 else float('nan'))
                )

        # ====== 汇总输出 ======
        total_elapsed = time.time() - total_start_time
        avg_time_all = (total_elapsed / self.all_solved_instance_num) if self.all_solved_instance_num > 0 else 0.0
        overall_avg_gap = np.mean(self.all_gaps_no_aug) if len(self.all_gaps_no_aug) > 0 else 0.0

        # 分桶平均 gap & 时间
        def _mean(xs): return float(np.mean(xs)) if len(xs) > 0 else 0.0
        b = self.bucket_gaps_no_aug
        t = self.bucket_times
        self.logger.info("#################  Summary  #################")
        self.logger.info("All instances: {}/{} solved, total time: {:.2f}s, avg time/inst: {:.2f}s".format(
            self.all_solved_instance_num, self.all_instance_num, total_elapsed, avg_time_all
        ))
        self.logger.info("[0,1000): count={}, avg_gap={:.3f}%, avg_time={:.3f}s".format(
            len(b['lt_1000']), _mean(b['lt_1000']), _mean(t['lt_1000'])
        ))
        self.logger.info("[1000,10000): count={}, avg_gap={:.3f}%, avg_time={:.3f}s".format(
            len(b['lt_10000']), _mean(b['lt_10000']), _mean(t['lt_10000'])
        ))
        self.logger.info("[10000,100000]: count={}, avg_gap={:.3f}%, avg_time={:.3f}s".format(
            len(b['lt_100000']), _mean(b['lt_100000']), _mean(t['lt_100000'])
        ))
        self.logger.info("Overall avg gap (no aug): {:.3f}%".format(overall_avg_gap))

        # 保持原有返回
        return score_AM.avg, score_student_AM.avg, (score_student_AM.avg - score_AM.avg) / score_AM.avg * 100 if score_AM.count>0 else float('nan')


    def decide_whether_to_repair_solution(self,
                                            before_complete_solution, before_repair_sub_solution,
                                          after_repair_sub_solution,before_reward, after_reward,
                                          first_node_index, length_of_subpath, double_solution):

        the_whole_problem_size  = int(double_solution.shape[1]/2)


        other_part_1 = double_solution[:,:first_node_index]
        other_part_2 = double_solution[:,first_node_index+length_of_subpath:]
        origin_sub_solution = double_solution[:, first_node_index : first_node_index+length_of_subpath]

        jjj, _ = torch.sort(origin_sub_solution, dim=1, descending=False)

        index = torch.arange(jjj.shape[0])[:,None].repeat(1,jjj.shape[1])

        kkk_2 = jjj[index,after_repair_sub_solution]

        if_repair = before_reward>after_reward

        double_solution[if_repair] = torch.cat((other_part_1[if_repair],
                                                        kkk_2[if_repair],
                                                        other_part_2[if_repair]),dim=1)
        after_repair_complete_solution = double_solution[:,first_node_index:first_node_index+the_whole_problem_size]



        return after_repair_complete_solution

    def _test_one_batch(self, episode, batch_size,k_nearest,decode_method,clock=None):

        self.model.eval()


        max_memory_allocated_before = torch.cuda.max_memory_allocated(device=self.device) / 1024 / 1024

        print('max_memory_allocated before', max_memory_allocated_before, 'MB')

        torch.cuda.reset_peak_memory_stats(device=self.device)

        with torch.no_grad():

            self.env.load_problems(episode, batch_size, self.inst_names[episode],
                                   self.inst_problem_sizes[episode],self.inst_opt_values[episode],only_test=True)


            self.origin_problem = self.env.problems
            reset_state, _, _ = self.env.reset(self.env_params['mode'])

            if self.env.test_in_tsplib:

                optimal_length, name = self.env._get_travel_distance_2(self.origin_problem, self.env.solution,
                                                                       test_in_tsplib=self.env.test_in_tsplib,
                                                                       need_optimal=True)

                self.optimal_length = optimal_length[0]

                self.name = name

            else:
                self.optimal_length = self.env._get_travel_distance_2(self.origin_problem, self.env.solution)

            IF_random_insertion = self.env_params['random_insertion']

            if IF_random_insertion:
                from utils.insertion import random_insertion

                dataset = self.origin_problem.clone().cpu().numpy()
                problem_size = dataset.shape[1]
                width = 1
                print('random insertion begin!')
                orders = [torch.randperm(problem_size) for i in range(width)]
                pi_all = [random_insertion(instance, orders[order_id])[0] for order_id in range(len(orders)) for
                          instance in
                          dataset]
                pi_all = np.array(pi_all, dtype=np.int64)
                best_select_node_list = torch.tensor(pi_all)

            else:

                B_V = batch_size * 1
                current_step = 0
                state, reward, reward_student, done = self.env.pre_step()
                # from tqdm import tqdm
                # with tqdm(total=self.env.problem_size) as pbar:
                while not done:
                    # pbar.update(1)

                    if current_step == 0:
                        selected_teacher= torch.zeros(B_V,dtype=torch.int64)
                        selected_student = selected_teacher

                    else:
                        selected_teacher, _,_,selected_student = self.model(
                            state,self.env.selected_node_list,self.env.solution,current_step,
                            decode_method=decode_method)

                    current_step += 1

                    state, reward,reward_student, done = self.env.step(selected_teacher, selected_student)


                best_select_node_list = self.env.selected_node_list


            current_best_length = self.env._get_travel_distance_2(self.origin_problem, best_select_node_list,
                                                                  test_in_tsplib=self.env.test_in_tsplib)

            max_memory_allocated_after = torch.cuda.max_memory_allocated(device=self.device) / 1024 / 1024


            escape_time, _ = clock.get_est_string(1, 1)

            gap = ((current_best_length.mean() - self.optimal_length.mean()) / self.optimal_length.mean()).item() * 100

            if self.env.test_in_tsplib:
                self.logger.info("greedy, name:{}, gap:{:6f} %,  Elapsed[{}], stu_l:{:6f} , opt_l:{:6f}".format(
                    self.name, gap, escape_time, current_best_length.mean().item(), self.optimal_length.mean().item()))
            else:
                self.logger.info("curr00,  gap:{:6f} %, Elapsed[{}], stu_l:{:6f} , opt_l:{:6f}, Memory:{:4f}MB".format(
                    gap, escape_time,  current_best_length.mean().item(), self.optimal_length.mean().item(),max_memory_allocated_after ))

            budget = self.env_params['budget']


            origin_problem_size = self.origin_problem.shape[1]

            # torch.save(best_select_node_list, f'TSP{origin_problem_size}_step0.pt')

            origin_batch_size = batch_size

            # print('============================= origin_problem_size', origin_problem_size)

            repair_max_sub_length = self.env_params['repair_max_sub_length']
            repair_max_sub_length = min(origin_problem_size, repair_max_sub_length)

            if origin_problem_size<=1000:
                length_all = torch.randint(low=4, high=repair_max_sub_length, size=[budget])
            else:
                length_all = torch.randint(low=4, high=repair_max_sub_length + 1, size=[budget])
            first_index_all = torch.randint(low=0, high=origin_problem_size, size=[budget])

            for bbbb in range(budget):


                # print('======length_all', length_all[bbbb], '======first_index_all', first_index_all[bbbb])

                self.env.problems = self.origin_problem.clone().detach()

                best_select_node_list = self.env.random_inverse_solution(best_select_node_list)


                if_PRC = self.env_params['PRC']

                if if_PRC:
                    partial_solution_length, first_node_index, length_of_subpath, double_solution, \
                    origin_sub_solution, index4, factor = \
                        self.env.destroy_solution_PRC(self.env.problems, best_select_node_list, length_all[bbbb],
                                                      first_index_all[bbbb], )
                else:
                    partial_solution_length, first_node_index, length_of_subpath, double_solution = \
                        self.env.destroy_solution(self.env.problems, best_select_node_list)

                before_reward = partial_solution_length

                before_repair_sub_solution = self.env.solution

                self.env.batch_size = before_repair_sub_solution.shape[0]

                current_step = 0

                reset_state, _, _ = self.env.reset(self.env_params['mode'])

                state, reward, reward_student, done = self.env.pre_step()

                while not done:
                    if current_step == 0:
                        selected_teacher = self.env.solution[:, -1]
                        selected_student = self.env.solution[:, -1]

                    elif current_step == 1:
                        selected_teacher = self.env.solution[:, 0]
                        selected_student = self.env.solution[:, 0]

                    else:
                        selected_teacher, _,_,selected_student = self.model(
                            state,self.env.selected_node_list,self.env.solution,current_step,
                            decode_method=decode_method,repair = True)

                    current_step += 1
                    state, reward, reward_student, done = self.env.step(selected_teacher, selected_student)

                ahter_repair_sub_solution = torch.roll(self.env.selected_node_list,shifts=-1,dims=1)

                after_reward = reward_student

                if if_PRC:
                    after_repair_complete_solution = self.env.decide_whether_to_repair_solution_PRC(
                        ahter_repair_sub_solution, before_reward, after_reward, double_solution,
                        origin_batch_size, origin_sub_solution, index4, factor)
                else:
                    after_repair_complete_solution = self.env.decide_whether_to_repair_solution(
                        ahter_repair_sub_solution, before_reward, after_reward,
                        first_node_index, length_of_subpath, double_solution)

                best_select_node_list = after_repair_complete_solution

                current_best_length = self.env._get_travel_distance_2(self.origin_problem,
                                                                      best_select_node_list, test_in_tsplib=self.env.test_in_tsplib)
                escape_time,_ = clock.get_est_string(1, 1)

                max_memory_allocated_after = torch.cuda.max_memory_allocated(device=self.device) / 1024 / 1024

                gap = ((current_best_length.mean() - self.optimal_length.mean()) / self.optimal_length.mean()).item() * 100

                if self.env.test_in_tsplib:

                    self.logger.info("RRC step{}, name:{}, gap:{:6f} %, Elapsed[{}], stu_l:{:6f} , opt_l:{:6f}".format(
                            bbbb, self.name, gap, escape_time, current_best_length.mean().item(),
                            self.optimal_length.mean().item()))
                else:
                    self.logger.info("step{},  gap:{:6f} %, Elapsed[{}], stu_l:{:6f} , opt_l:{:6f}, Memory:{:4f}MB".format(
                       bbbb, ((current_best_length.mean() - self.optimal_length.mean()) / self.optimal_length.mean()).item() * 100, escape_time,
                    current_best_length.mean().item(), self.optimal_length.mean().item(),max_memory_allocated_after))



        return self.optimal_length.mean().item(),current_best_length.mean().item(), current_best_length.mean().item(),self.env.problem_size
