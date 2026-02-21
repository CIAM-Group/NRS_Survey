# vrplib.py
# Minimal CVRPLIB parser matching the CVRPLIBReader style you provided
# and returning data structures that fit your CVRPEnv/test code.

from __future__ import annotations
import os
import re
from typing import Dict, Any, List, Optional
import numpy as np


def parse_vrplib(filename: str) -> Dict[str, Any]:
    """
    严格模仿你给的 CVRPLIBReader 的读取流程：
      - startswith 判断块
      - NODE_COORD_SECTION -> 读取 (id, x, y)
      - DEMAND_SECTION -> 读取 (id, demand)
      - DEPOT_SECTION -> 读取 depot id，直到 -1
    返回原始顺序（id 从 1 开始）的 locs/demand（都包含 depot）。
    注意：这里不做 depot 重排，read_instance 再做。
    """
    with open(filename, 'r') as f:
        name = None
        dimension = 0   # customers count (文件里会给总点数或客户数，这里跟示例保持：减一在下面做)
        capacity = 0
        started_node = False
        started_demand = False
        locs: List[List[float]] = []
        demand: List[int] = []
        depot: Optional[int] = None

        for line in f:
            line = line.strip()
            if not line:
                continue

            # DEMAND_SECTION 内容
            if started_demand:
                if line.startswith("DEPOT_SECTION"):
                    started_demand = False
                    # 继续在下面读 depot
                    continue
                # e.g. "1 0"
                demand.append(int(line.split()[-1]))
                continue

            # NODE_COORD_SECTION 内容
            if started_node:
                if line.startswith("DEMAND_SECTION"):
                    started_node = False
                    started_demand = True
                    continue
                parts = line.split()
                # e.g. "1 35 35" -> 取 x/y
                locs.append([float(parts[1]), float(parts[2])])
                continue

            # 头部信息
            if line.startswith("NAME"):
                # e.g. "NAME : X-n101-k25"
                name = line.split()[-1]
            elif line.startswith("DIMENSION"):
                # 示例里用 "dimension = float(...)-1"
                # 这里先记下原值，等全部读完再做 -1（与示例保持一致：depot 不计入）
                dimension = int(float(line.split()[-1])) - 1
            elif line.startswith("EDGE_WEIGHT_TYPE"):
                # 参考示例，如果不是 EUC_2D/CEIL_2D 就放弃；这里不强制检查，按需可加
                pass
            elif line.startswith("CAPACITY"):
                capacity = int(float(line.split()[-1]))
            elif line.startswith("NODE_COORD_SECTION"):
                started_node = True
            elif line.startswith("DEPOT_SECTION"):
                # 读取 depot id，直到 -1
                # 有些文件只有一个 depot 行
                # 这里模仿示例：读到第一行 depot，忽略后续直到 -1
                # 也兼容多个 depot id 的写法（只取第一个）
                while True:
                    nxt = f.readline()
                    if not nxt:
                        break
                    nxt = nxt.strip()
                    if nxt.startswith("-1"):
                        break
                    if depot is None:
                        depot = int(nxt.split()[0])
                break  # 与示例一致，读到 depot 就结束

    # 与示例一致：期待 depot 也在 locs/demand 里，因此长度应为 dimension+1
    assert len(locs) == dimension + 1, f"Expected {dimension+1} coords, got {len(locs)}"
    assert len(demand) == dimension + 1, f"Expected {dimension+1} demands, got {len(demand)}"
    if depot is None:
        # 常见默认值为 1
        depot = 1

    return {
        "name": name,
        "dimension": int(dimension),   # customers count（不含 depot）
        "capacity": capacity,
        "locs": locs,                  # len = dimension+1，包含 depot
        "demand": demand,              # len = dimension+1，包含 depot
        "depot": depot                 # 原始 depot id（1-based）
    }


def read_instance(filename: str) -> Dict[str, Any]:
    """
    在 parse_vrplib 的基础上，按照**你当前工程的存储格式**返回：
      - node_coord: np.ndarray, shape (n+1, 2)，depot 调整到索引 0
      - demand:     np.ndarray, shape (n+1, )，demand[0] = 0
      - capacity:   int/float
      - depot:      0
      - name:       str
      - dimension:  int（客户数）
    """
    parsed = parse_vrplib(filename)
    name = parsed["name"]
    dimension = parsed["dimension"]          # customers count
    capacity = parsed["capacity"]
    locs = parsed["locs"]                    # depot + customers（原始 1-based 顺序）
    demand = parsed["demand"]
    depot_id = parsed["depot"]               # 1-based

    # 将 depot 调整到索引 0
    # locs/demand 的索引 0 对应 id=1，因此 depot 的 0-based 索引是 depot_id-1
    depot_idx0 = depot_id - 1
    n_all = len(locs)  # = dimension + 1

    # 其余点保持原始顺序（去掉 depot）
    order = [depot_idx0] + [i for i in range(n_all) if i != depot_idx0]

    node_coord = np.array([locs[i] for i in order], dtype=np.float32)       # (n+1, 2)
    demand_arr = np.array([demand[i] for i in order], dtype=np.float32)     # (n+1,)
    # depot 需求归零（环境里到 depot 会补货）
    demand_arr[0] = 0.0

    return {
        "name": name,
        "dimension": int(dimension),
        "node_coord": node_coord,   # depot at index 0
        "demand": demand_arr,       # demand[0] == 0
        "capacity": float(capacity),
        "depot": 0                  # 现在 depot 在索引 0
    }


def read_solution(filename: str) -> Dict[str, float]:
    """
    读取 .sol 文件中的 Cost。
    先按你提供示例的方式：遇到以 "Cost" 开头的行，直接取第二个 token。
    若格式含冒号或其它分隔符，则做一次兜底：从该行提取第一个数字。
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Solution file not found: {filename}")

    cost: Optional[float] = None

    # 第一遍：严格按“示例风格”来（line.startswith("Cost")，取第二个 token）
    with open(filename, 'r') as f:
        for line in f:
            line_stripped = line.strip()
            if line_stripped.startswith("Cost"):
                parts = line_stripped.split()
                # 典型情况：["Cost", "12345"] 或 ["Cost", ":", "12345"]
                if len(parts) >= 2:
                    # 优先取第二个 token，如果是 ':' 再尝试第三个
                    cand = parts[1]
                    if cand == ":" and len(parts) >= 3:
                        cand = parts[2]
                    try:
                        cost = float(cand)
                        break
                    except ValueError:
                        # 兜底：从整行找数字
                        nums = re.findall(r"[0-9]+(?:\.[0-9]+)?", line_stripped)
                        if nums:
                            cost = float(nums[0])
                            break

    # 如果第一遍没找到，做一次宽松兜底（比如大小写不同）
    if cost is None:
        with open(filename, 'r') as f:
            for line in f:
                if line.strip().lower().startswith("cost"):
                    nums = re.findall(r"[0-9]+(?:\.[0-9]+)?", line)
                    if nums:
                        cost = float(nums[0])
                        break

    if cost is None:
        raise ValueError(f"Could not parse 'Cost' from solution file: {filename}")

    return {"cost": cost}
