import os
import numpy as np
import itertools

import torch
from torch.utils.data import Dataset

# ! 修改的代码
# ! TSPLib Dataloader
class tsplib_loader(Dataset):
    def __init__(self, path, scale_range=[0,1000], device="cpu"):
        self.device = device
        self.path = path
        self.scale_range = scale_range  # e.g. [500, 800]
        print("scale range:", scale_range)
        self.data = {}
        self.num_sample = 0
        self.scale_min_max = [float("inf"), float("-inf")]
        for root, dirs, files in os.walk(self.path):
            for file in files:
                if file.endswith(".tsp"):
                    name, dimension, locs, edge_weight_type = TSPLIBReader(os.path.join(root, file))
                    if name is None:
                        continue
                    if self.scale_range is not None and not (self.scale_range[0] <= dimension < self.scale_range[1]):
                        continue
                    self.data[name] = {"dimension": dimension, "locations": locs, "edge_weight_type": edge_weight_type}
                    self.num_sample += 1
                    self.scale_min_max[0] = min(self.scale_min_max[0], dimension)
                    self.scale_min_max[1] = max(self.scale_min_max[1], dimension)
        
        if self.num_sample == 0:
            print(f"No TSP files found in {self.path} within the specified scale range {self.scale_range}, the scale range is {self.scale_min_max}")
            # raise ValueError(f"No TSP files found in {self.path} within the specified scale range {self.scale_range}, the scale range is {self.scale_min_max}")
        else:
            self.data = dict(sorted(self.data.items(), key=lambda item: item[1]["dimension"]))
            
    def __getitem__(self, index):
        name = list(self.data.keys())[index]
        edge_weight_type = self.data[name]["edge_weight_type"]
        original_data = torch.tensor(self.data[name]["locations"], dtype=torch.float32)
        data, _ = normalize_to_unit_board(original_data)
        return {
            "name": name,
            "edge_weight_type": edge_weight_type,
            "lib_data": original_data,
            "coordinates": data,
            "optimal_score": torch.tensor(tsplib_cost.get(name, 0), dtype=torch.float32),
            "num_sample": self.num_sample,
        }
    def __len__(self):
        return self.num_sample 


class cvrplib_loader(Dataset):
    def __init__(self, path, scale_range=[0,1000], device="cpu",DUMMY_RATE = None):
        self.device = device
        self.path = path
        self.data = {}
        self.scale_range = scale_range  # e.g. [500, 800]
        print("scale range:", scale_range)
        self.scale_min_max = [float("inf"), float("-inf")]
        self.num_sample = 0
        self.DUMMY_RATE = DUMMY_RATE
        for root, dirs, files in os.walk(self.path):
            for file in files:
                if file.endswith(".vrp"):
                    name, dimension, locs, demand, capacity, cost = CVRPLIBReader(
                        os.path.join(root, file)
                    )
                    if name is None:
                        continue
                    if self.scale_range is not None and not (self.scale_range[0] <= dimension < self.scale_range[1]):
                        continue
                    self.data[name] = {"dimension": dimension, "locations": locs, "demand": demand, "capacity": capacity, "cost": cost}
                    self.num_sample += 1
                    self.scale_min_max[0] = min(self.scale_min_max[0], dimension)
                    self.scale_min_max[1] = max(self.scale_min_max[1], dimension)


        if self.num_sample == 0:
            print(f"No CVRP files found in {self.path} within the specified scale range {self.scale_range}, the scale range is {self.scale_min_max}")
            # raise ValueError(f"No VRP files found in {self.path} within the specified scale range {self.scale_range}")
        else:
            self.data = dict(sorted(self.data.items(), key=lambda item: item[1]["dimension"]))

    def __getitem__(self, index):
        name = list(self.data.keys())[index]
        original_data = torch.tensor(self.data[name]["locations"], dtype=torch.float32)
        data, _ = normalize_to_unit_board(original_data)
        demand = (
            torch.tensor(self.data[name]["demand"], dtype=torch.float32)
            / self.data[name]["capacity"]
        )
        # shape: (num_nodes,), including depot demand
        dimension = self.data[name]["dimension"]
        dimension_with_dummy = int(np.ceil(dimension * (1 + self.DUMMY_RATE))) # the number of real nodes plus dummy nodes in cvrp
        real_size = dimension  # the number of real nodes in cvrp
        
        demand = torch.cat((torch.zeros(dimension_with_dummy - real_size), 
                            demand[1:]), dim=0)  # set dummy depot demand to 0
        # shape: (dimension_with_dummy,)
        depot_original = original_data[0].unsqueeze(0)
        depot_normed = data[0].unsqueeze(0)
        # shape: (1, 2)
        assert depot_original.shape == (1, 2)
        assert depot_normed.shape == (1, 2)
        
        loc_original = original_data[1:]
        loc_normed = data[1:]
        # shape: (dimension, 2)
        
        coordinates_original = torch.cat((depot_original.repeat(dimension_with_dummy - real_size,1), loc_original), 0)
        coordinates_normed = torch.cat((depot_normed.repeat(dimension_with_dummy - real_size,1), loc_normed), 0)
        # shape: (dimension_with_dummy, 2)
        # add dummy depot nodes
        return {
            "name": name,
            "lib_data": coordinates_original,
            "coordinates": coordinates_normed,
            "demand": demand,
            "optimal_score": torch.tensor(self.data[name]["cost"], dtype=torch.float32),
            "num_sample": self.num_sample,
            'capacity': self.data[name]["capacity"],
            "dimension": dimension,
        }

    def __len__(self):
        return self.num_sample

# ! data generation from Invit
def normalize_to_unit_board(node_coord):
    """
    normalize a tsp instance to a [0, 1]^2 unit board, prefer to have points on both x=0 and y=0
    :param tsp_instance: a (num_nodes, 2) tensor
    :return: a (num_nodes, 2) tensor, a normalized tsp instance
    """
    normalized_instance = node_coord.clone()
    normalization_factor = (normalized_instance.max(dim=0).values - normalized_instance.min(dim=0).values).max()
    normalized_instance = (normalized_instance - normalized_instance.min(dim=0).values) / normalization_factor
    return normalized_instance, normalization_factor



def TSPLIBReader(filename):
    '''
        Acquire description of a TSP problem from a TSPLIB-formatted file
        Parameters:
        - filename: the name of the TSPLIB-formatted file.  
        Returns:
        - name: the name of the TSP problem.
        - dimension: the number of nodes in the TSP problem. (int)
        - locs: the coordinates of nodes in the TSP problem. e.g.[[31020, 8718], [85228, 81588], [28825, 6767], [9301, 86527]]
    '''
    with open(filename, 'r') as f:
        dimension = 0
        started = False
        locs = []
        # ! 补充ceil / round
        edge_weight_type = None

        for line in f:
            loc = []
            if started:
                if line.startswith("EOF"):
                    break
                loc.append(float(line.strip().split()[1]))
                loc.append(float(line.strip().split()[2]))
                locs.append(loc)
            if line.startswith("NAME"):
                name = line.strip().split(" ")[-1]
            if line.startswith("DIMENSION"):
                dimension = int(line.strip().split(" ")[-1])
            if line.startswith("EDGE_WEIGHT_TYPE"):
                edge_weight_type = line.strip().split(" ")[-1]
                if line.strip().split(" ")[-1] not in ["EUC_2D", "CEIL_2D"]:
                    return None, None, None
            if line.startswith("NODE_COORD_SECTION"):
                started = True

    assert len(locs) == dimension
    return name, dimension, locs, edge_weight_type


def CVRPLIBReader(filename):
    '''
        Acquire description of a CVRP problem from a CVRPLIB-formatted file
        Parameters:
        - filename: the name of the CVRPLIB-formatted file.   
        Returns:
        - name: the name of the CVRP problem.
        - dimension: the number of nodes in the CVRP problem. (int)
        - locs: the coordinates of nodes in the CVRP problem. e.g.[[31020, 8718], [85228, 81588], [28825, 6767], [9301, 86527]]
        - demand: A list of node demands. e.g.[0, 9, 2, 2, 5, 3, 9, 4, 8, 1, 8, 3, 7, 5, 8, 6, 3, 9, 5, 6, 8]
        - capacity: The capacity of the vehicle. (int)
    '''
    with open(filename, 'r') as f:
        dimension = 0
        started_node = False
        started_demand = False
        locs = []
        demand = []
        for line in f:
            loc = []
            if started_demand:
                if line.startswith("DEPOT_SECTION"):
                    break
                demand.append(int(line.strip().split()[-1]))
            if started_node:
                if line.startswith("DEMAND_SECTION"):
                    started_node = False
                    started_demand = True
            if started_node:
                loc.append(float(line.strip().split()[1]))
                loc.append(float(line.strip().split()[2]))
                locs.append(loc)

            if line.startswith("NAME"):
                name = line.strip().split()[-1]
            if line.startswith("DIMENSION"):
                dimension = float(line.strip().split()[-1]) - 1 # depot is not counted
            if line.startswith("EDGE_WEIGHT_TYPE"):
                if line.strip().split()[-1] not in ["EUC_2D", "CEIL_2D"]:
                    return None, None, None, None, None, None
            if line.startswith("CAPACITY"):
                capacity = float(line.strip().split()[-1])
            if line.startswith("NODE_COORD_SECTION"):
                started_node = True
    cost_file = filename.replace('.vrp', '.sol')
    if os.path.exists(cost_file):
        with open(cost_file, 'r') as f:
            for line in f:
                if line.startswith("Cost"):
                    cost = float(line.split()[1])
    else:
        cost = None
    assert len(locs) == dimension + 1  # +1 for depot
    return name, int(dimension), locs, demand, capacity, cost


# 最优值 invit data_utils
tsplib_cost = {
    # TSPLIB, 77+4, all optimal, http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/STSP.html
    "a280": 2579,
    # "ali535": 202339,
    # "att48": 10628,
    # "att532": 27686,
    # "bayg29": 1610,
    # "bays29": 2020,
    "berlin52": 7542,
    "bier127": 118282,
    # "brazil58": 25395,
    "brd14051": 469385,
    # "brg180": 1950,
    # "burma14": 3323,
    "ch130": 6110,
    "ch150": 6528,
    "d198": 15780,
    "d493": 35002,
    "d657": 48912,
    "d1291": 50801,
    "d1655": 62128,
    "d2103": 80450,
    "d15112": 1573084,
    "d18512": 645238,
    # "dantzig42": 699,
    # "dsj1000": 18659688, # (EUC_2D)
    "dsj1000": 18660188, # (CEIL_2D)
    "eil51": 426,
    "eil76": 538,
    "eil101": 629,
    "fl417": 11861,
    "fl1400": 20127,
    "fl1577": 22249,
    "fl3795": 28772,
    "fnl4461": 182566,
    # "fri26": 937,
    "gil262": 2378,
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
    "kroA100": 21282,
    "kroB100": 22141,
    "kroC100": 20749,
    "kroD100": 21294,
    "kroE100": 22068,
    "kroA150": 26524,
    "kroB150": 26130,
    "kroA200": 29368,
    "kroB200": 29437,
    "lin105": 14379,
    "lin318": 42029,
    "linhp318": 41345,
    "nrw1379": 56638,
    "p654": 34643,
    # "pa561": 2763,
    "pcb442": 50778,
    "pcb1173": 56892,
    "pcb3038": 137694,
    "pla7397": 23260728, # (CEIL_2D)
    "pla33810": 66048945, # (CEIL_2D)
    "pla85900": 142382641, # (CEIL_2D)
    "pr76": 108159,
    "pr107": 44303,
    "pr124": 59030,
    "pr136": 96772,
    "pr144": 58537,
    "pr152": 73682,
    "pr226": 80369,
    "pr264": 49135,
    "pr299": 48191,
    "pr439": 107217,
    "pr1002": 259045,
    "pr2392": 378032,
    "rat99": 1211,
    "rat195": 2323,
    "rat575": 6773,
    "rat783": 8806,
    "rd100": 7910,
    "rd400": 15281,
    "rl1304": 252948,
    "rl1323": 270199,
    "rl1889": 316536,
    "rl5915": 565530,
    "rl5934": 556045,
    "rl11849": 923288,
    # "si175": 21407,
    # "si535": 48450,
    # "si1032": 92650,
    "st70": 675,
    # "swiss42": 1273,
    "ts225": 126643,
    "tsp225": 3916,
    "u159": 42080,
    "u574": 36905,
    "u724": 41910,
    "u1060": 224094,
    "u1432": 152970,
    "u1817": 57201,
    "u2152": 64253,
    "u2319": 234256,
    # "ulysses16": 6859,
    # "ulysses22": 7013,
    "usa13509": 19982859,
    "vm1084": 239297,
    "vm1748": 336556,

    # National TSP, 27, 2 non-optimal, https://www.math.uwaterloo.ca/tsp/world/summary.html
    'ar9152': 837_479,
    'bm33708': 959_289, # gap 0.031%
    'ca4663': 1_290_319,
    'ch71009': 4_566_506, # gap 0.024%
    'dj38': 6_656,
    'eg7146': 172_386,
    'fi10639': 520_527,
    'gr9882': 300_899,
    'ho14473': 177_092,
    'ei8246': 206_171,
    'it16862': 557_315,
    'ja9847': 491_924,
    'kz9976': 1_061_881,
    'lu980': 11_340,
    'mo14185': 427_377,
    'nu3496': 96_132,
    'mu1979': 86_891,
    "pm8079": 114_855,
    'qa194': 9_352,
    'rw1621': 26_051,
    'sw24978': 855_597,
    'tz6117': 394_718,
    'uy734': 79_114,
    'vm22775': 569_288,
    'wi29': 27_603,
    'ym7663': 238_314,
    'zi929': 95_345,

    # VLSI, 102-4, non-optimal when size >= 14233 (xrb14233), https://www.math.uwaterloo.ca/tsp/vlsi/summary.html
    'xqf131': 564,
    'xqg237': 1_019,
    'pma343': 1_368,
    'pka379': 1_332,
    'bcl380': 1_621,
    'pbl395': 1_281,
    'pbk411': 1_343,
    'pbn423': 1_365,
    'pbm436': 1_443,
    'xql662': 2_513,
    'rbx711': 3_115,
    'rbu737': 3_314,
    'dkg813': 3_199,
    'lim963': 2_789,
    'pbd984': 2_797,
    'xit1083': 3_558,
    'dka1376': 4_666,
    'dca1389': 5_085,
    'dja1436': 5_257,
    'icw1483': 4_416,
    'fra1488': 4_264,
    'rbv1583': 5_387,
    'rby1599': 5_533,
    'fnb1615': 4_956,
    'djc1785': 6_115,
    'dcc1911': 6_396,
    'dkd1973': 6_421,
    'djb2036': 6_197,
    'dcb2086': 6_600,
    'bva2144': 6_304,
    'xqc2175': 6_830,
    'bck2217': 6_764,
    'xpr2308': 7_219,
    'ley2323': 8_352,
    'dea2382': 8_017,
    'rbw2481': 7_724,
    'pds2566': 7_643,
    'mlt2597': 8_071,
    'bch2762': 8_234,
    'irw2802': 8_423,
    'lsm2854': 8_014,
    'dbj2924': 10_128,
    'xva2993': 8_492,
    'pia3056': 8_258,
    'dke3097': 10_539,
    'lsn3119': 9_114,
    'lta3140': 9_517,
    'fdp3256': 10_008,
    'beg3293': 9_772,
    'dhb3386': 11_137,
    'fjs3649': 9_272,
    'fjr3672': 9_601,
    'dlb3694': 10_959,
    'ltb3729': 11_821,
    'xqe3891': 11_995,
    'xua3937': 11_239,
    'dkc3938': 12_503,
    'dkf3954': 12_538,
    'bgb4355': 12_723,
    'bgd4396': 13_009,
    'frv4410': 10_711,
    'bgf4475': 13_221,
    'xqd4966': 15_316,
    'fqm5087': 13_029,
    'fea5557': 15_445,
    'xsc6880': 21_535,
    'bnd7168': 21_834,
    'lap7454': 19_535,
    'ida8197': 22_338,
    'dga9698': 27_724,
    'xmc10150': 28_387,
    'xvb13584': 37_083,
    'xrb14233': 45_462, # gap 0.026%
    'xia16928': 52_850, # gap 0.023%
    'pjh17845': 48_092, # gap 0.019%
    'frh19289': 55_798, # gap 0.013%
    'fnc19402': 59_287, # gap 0.020%
    'ido21215': 63_517, # gap 0.028%
    'fma21553': 66_527, # gap unknown
    'lsb22777': 60_977, # gap unknown
    'xrh24104': 69_294, # gap unknown
    'bbz25234': 69_335, # gap unknown
    'irx28268': 72_607, # gap unknown
    'fyg28534': 78_562, # gap unknown
    'icx28698': 78_087, # gap unknown
    'boa28924': 79_622, # gap unknown
    'ird29514': 80_353, # gap unknown
    'pbh30440': 88_313, # gap unknown
    'xib32892': 96_757, # gap unknown
    'fry33203': 97_240, # gap unknown
    'bby34656': 99_159, # gap unknown
    'pba38478': 108_318, # gap unknown
    'ics39603': 106_819, # gap unknown
    'rbz43748': 125_183, # gap unknown
    'fht47608': 125_104, # gap unknown
    'fna52057': 147_789, # gap unknown
    'bna56769': 158_078, # gap unknown
    'dan59296': 165_371, # gap unknown
    # 'sra104815': 251_342, # gap unknown
    # 'ara238025': 578_761, # gap unknown
    # 'lra498378': 2_168_039, # gap unknown
    # 'lrb744710': 1_611_232, # gap unknown

    # DIMACS 8th Challenge, non-optimal, http://dimacs.rutgers.edu/archive/Challenges/TSP/opts.html and http://webhotel4.ruc.dk/~keld/research/LKH/DIMACS_results.html
    "portcgen-1000-1000": 11387430, # gap 0.54%, C1k.0
    "portcgen-1000-10001": 11376735, # gap 0.41%, C1k.1
    "portcgen-1000-10002": 10855033, # gap 0.42%, C1k.2
    "portcgen-1000-10003": 11886457, # gap 0.53%, C1k.3
    "portcgen-1000-10004": 11499958, # gap 0.58%, C1k.4
    "portcgen-1000-10005": 11394911, # gap 0.58%, C1k.5
    "portcgen-1000-10006": 10166701, # gap 0.73%, C1k.6
    "portcgen-1000-10007": 10664660, # gap 0.58%, C1k.7
    "portcgen-1000-10008": 11605723, # gap 0.34%, C1k.8
    "portcgen-1000-10009": 10906997, # gap 0.66%, C1k.9
    "portcgen-3162-3162": 19198258, # gap 0.62%, C3k.0
    "portcgen-3162-31621": 19017805, # gap 0.61%, C3k.1
    "portcgen-3162-31622": 19547551, # gap 0.70%, C3k.2
    "portcgen-3162-31623": 19108508, # gap 0.57%, C3k.3
    "portcgen-3162-31624": 18864046, # gap 0.57%, C3k.4
    "portcgen-10000-10000": 33_001_034, # gap 0.668%, C10k.0
    "portcgen-10000-100001": 33_186_248, # gap 0.690%, C10k.1
    "portcgen-10000-100002": 33_155_424, # gap 0.694%, C10k.2
    "portcgen-31623-31623": 59_545_390, # gap 0.636%, C31k.0
    "portcgen-31623-316231": 59_293_266, # gap 0.770%, C31k.1
    "portcgen-100000-100000": 104_617_752, # gap 0.675%, C100k.0
    "portcgen-100000-1000001": 105_390_777, # gap 0.695%, C100k.1
    # "C316k.0": 186_870_839 # gap 0.697%, C316k.0
    

}
