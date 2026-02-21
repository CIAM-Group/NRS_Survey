# diffusion/co_datasets/tsp_tsplib_dataset.py
import os
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

SUPPORTED_EWT = {"EUC_2D": 1, "CEIL_2D": 2}  # 0=continuous(默认), 1=EUC_2D, 2=CEIL_2D

def _read_tsplib_tsp(path: str) -> Tuple[str, int, np.ndarray, str]:
    """
    读取 TSPLIB .tsp 文件，返回 (name, dimension, coords[n,2], edge_weight_type)
    只支持 EUC_2D / CEIL_2D；不支持的类型抛错。
    """
    name, dimension, ewt = None, None, None
    coords: List[List[float]] = []
    started = False
    with open(path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            u = line.upper()
            if u.startswith("NAME"):
                # 兼容 “NAME : xxx” / “NAME:xxx”
                _, _, rhs = line.partition(":")
                name = (rhs or line.split()[-1]).strip()
            elif u.startswith("DIMENSION"):
                _, _, rhs = line.partition(":")
                dimension = int((rhs or line.split()[-1]).strip())
            elif u.startswith("EDGE_WEIGHT_TYPE"):
                _, _, rhs = line.partition(":")
                ewt = (rhs or line.split()[-1]).strip()
                if ewt not in SUPPORTED_EWT:
                    raise ValueError(f"EDGE_WEIGHT_TYPE '{ewt}' not supported (only EUC_2D/CEIL_2D). File={path}")
            elif u.startswith("NODE_COORD_SECTION"):
                started = True
            elif u.startswith("EOF"):
                break
            elif started:
                # 自适应空白分隔
                toks = line.split()
                if len(toks) >= 3:
                    # 格式：idx x y
                    x, y = float(toks[1]), float(toks[2])
                    coords.append([x, y])
    assert name is not None and dimension is not None and ewt is not None, f"Bad TSPLIB header in {path}"
    assert len(coords) == dimension, f"coord len {len(coords)} != dimension {dimension} ({path})"
    return name, dimension, np.asarray(coords, dtype=np.float32), ewt


def _load_opt_cost_map(opt_file: Optional[str]) -> dict:
    """
    读取一个简单的 “name cost” 文本或 json（任选其一），用于提供最优值 label（可选）。
    - .json: { "a280": 2579, ... }
    - 其他文本：每行 "name cost"
    """
    if not opt_file:
        return {}
    if not os.path.exists(opt_file):
        raise FileNotFoundError(f"tsplib optimal file not found: {opt_file}")
    if opt_file.endswith(".json"):
        import json
        with open(opt_file, "r") as f:
            return {k: float(v) for k, v in json.load(f).items()}
    m = {}
    with open(opt_file, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            toks = s.split()
            if len(toks) >= 2:
                m[toks[0]] = float(toks[1])
    return m


class TSPTSPLIBDataset(Dataset):
    """
    读取目录/单文件的 .tsp，并返回：
      - idx: LongTensor([i])
      - points: FloatTensor(N,2), 归一化到 [0,1]（与 ICAM 一致）
      - ewt_code: LongTensor([0|1|2]) 0=continuous, 1=EUC_2D, 2=CEIL_2D
      - gt_cost: FloatTensor([cost]) 或 FloatTensor([])（如无最优值）
    说明：
      * 为了最小侵入，我们不提供 gt_tour（TSPLIB 没有），下游用 gt_cost 直接算 gap。
      * 如果你有 .opt.tour 或 LKH 结果，也可扩展返回 tour。
    """
    # ! 增加
    def __init__(self, path: str, optimal_cost_file: Optional[str] = None, optimal_cost_map: Optional[dict] = None):
        self.paths: List[str] = []
        if os.path.isdir(path):
            for root, _, files in os.walk(path):
                for fn in files:
                    if fn.lower().endswith(".tsp"):
                        self.paths.append(os.path.join(root, fn))
        else:
            assert path.lower().endswith(".tsp"), f"expect .tsp or dir, got: {path}"
            self.paths.append(path)
        self.paths.sort()
        # ! 最优值字典
        file_map = _load_opt_cost_map(optimal_cost_file) if optimal_cost_file else {}
        self.opt_map = {**file_map, **(optimal_cost_map or {})}  # 字典优先级：内存>文件
        # self.opt_map = _load_opt_cost_map(optimal_cost_file)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        tsp_path = self.paths[idx]
        name, n, coords, ewt = _read_tsplib_tsp(tsp_path)

        # 1) 模型输入：归一化到 [0,1]
        xy_max = coords.max(axis=0, keepdims=True)
        xy_min = coords.min(axis=0, keepdims=True)
        side = float(np.max(xy_max - xy_min))
        side = side if side > 0 else 1.0
        norm = (coords - xy_min) / side  # [N,2]

        # 2) 评测：保留原始坐标
        orig = coords  # [N,2]

        ewt_code = SUPPORTED_EWT.get(ewt, 0)
        gt_cost = self.opt_map.get(name, None)
        gt_cost_tensor = (
            torch.tensor([gt_cost], dtype=torch.float32)
            if gt_cost is not None else
            torch.tensor([], dtype=torch.float32)
        )

        return (
            torch.LongTensor(np.array([idx], dtype=np.int64)),
            torch.from_numpy(norm).float(),               # 给模型
            torch.LongTensor(np.array([ewt_code], dtype=np.int64)),
            gt_cost_tensor,
            torch.from_numpy(orig).float(),               # 给评测
        )