
import torch
import numpy as np
from problems.problem_cvrp import CVRP

def test_cvrp_initial_solution():
    p_size = 100
    # Initialize CVRP
    cvrp = CVRP(p_size, init_val_met='random', with_assert=True)
    
    # Create a dummy batch with batch_size=1
    batch_size = 1
    
    print(f"CVRP Size: {cvrp.size}")
    print(f"CVRP Real Size: {cvrp.real_size}")
    print(f"CVRP Dummy Size: {cvrp.dummy_size}")
    
    # Construct batch
    coordinates = torch.rand(batch_size, cvrp.size, 2)
    demand = torch.zeros(batch_size, cvrp.size)
    
    # Set demands for real nodes to be NORMAL (small enough to fit in depots)
    # 20 nodes. Capacity 1.0.
    # If demand is 0.1, total 2.0. Need 2 trips. 10 depots available.
    real_demand = torch.ones(batch_size, cvrp.real_size) * 0.1
    demand[:, cvrp.dummy_size:] = real_demand
    
    batch = {'coordinates': coordinates, 'demand': demand}
    
    print("Testing get_initial_solutions with batch_size=1 and NORMAL demand...")
    try:
        solution = cvrp.get_initial_solutions(batch)
        print("Solution generated.")
        print("Solution shape:", solution.shape)
        print("Solution:", solution)
        
        cvrp.get_costs(batch, solution, check_full_feasibility=True)
        print("Feasibility check passed.")
        
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_cvrp_initial_solution()
