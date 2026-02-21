import torch

def compute_insertion_positions(coords, partial_tour, insert_city):
    B, T = partial_tour.shape
    device = coords.device

    current = torch.gather(coords, 1, partial_tour.unsqueeze(-1).expand(-1, -1, 2))  # [B, T, 2]
    next_idx = torch.roll(partial_tour, shifts=-1, dims=1)
    next_coords = torch.gather(coords, 1, next_idx.unsqueeze(-1).expand(-1, -1, 2))  # [B, T, 2]

    insert_coords = coords[torch.arange(B, device=device), insert_city].unsqueeze(1)  # [B, 1, 2]

    d1 = torch.norm(current - insert_coords, dim=-1)  # [B, T]
    d2 = torch.norm(next_coords - insert_coords, dim=-1)
    removed = torch.norm(current - next_coords, dim=-1)

    delta = d1 + d2 - removed
    return torch.argmin(delta, dim=1)  # [B]

def insert_into_tour(partial_tour, insert_city, insert_pos):
    B, T = partial_tour.shape
    device = partial_tour.device

    tour_new = torch.zeros(B, T + 1, dtype=torch.long, device=device)
    arange = torch.arange(T + 1, device=device).unsqueeze(0).expand(B, -1)  # [B, T+1]
    
    # mask for positions before insert_pos
    before_mask = arange <= insert_pos.unsqueeze(1) 
    after_mask = arange > (insert_pos.unsqueeze(1) + 1)
    at_mask = arange == (insert_pos.unsqueeze(1) + 1)

    # Step 1: 插入前的部分
    tour_new[before_mask] = partial_tour[before_mask[:, :-1]]
    # print(tour_new, 'tour_new_before')

    # Step 2: 插入城市
    tour_new[at_mask] = insert_city
    # print(tour_new, 'tour_new_at')

    # Step 3: 插入后的部分（偏移一位）
    tour_new[after_mask] = partial_tour[after_mask[:, 1:]]
    # print(tour_new, 'tour_new_after')

    return tour_new

def random_insertion_tsp(coords):
    """
    coords: Tensor [B, N, 2], batch of TSP instances
    return: tours: Tensor [B, N], batch of valid tours
    """
    B, N, _ = coords.shape
    device = coords.device

    all_nodes = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)  # [B, N]

    # Step 1: 初始化，随机选择起始路径（三个城市）
    init = torch.rand(B, N, device=device).argsort(dim=1)[:, :3]  # [B, 3]
    partial_tour = init.clone()

    # Step 2: 构建 mask，记录已访问的城市
    visited_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
    visited_mask.scatter_(1, init, True)

    for _ in range(3, N):
        # Step 3: 找到尚未插入的城市（每个 batch 选一个）
        unvisited_mask = ~visited_mask
        rand_vals = torch.rand(B, N, device=device)
        rand_vals[~unvisited_mask] = float('inf')  # 忽略已访问
        insert_city = rand_vals.argmin(dim=1)  # [B]
        visited_mask[torch.arange(B, device=device), insert_city] = True

        # Step 4: 找到最佳插入位置
        insert_pos = compute_insertion_positions(coords, partial_tour, insert_city)  # [B]

        # Step 5: 执行插入
        partial_tour = insert_into_tour(partial_tour, insert_city, insert_pos)

    return partial_tour  # [B, N]