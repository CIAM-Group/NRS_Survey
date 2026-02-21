import os
import math
import re

def read_vrp_file(filepath):
    """
    Reads a VRP file and returns necessary information.
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    data = {
        'name': '',
        'capacity': 0,
        'dimension': 0,
        'nodes': {}, # id -> (x, y)
        'demands': {}, # id -> demand
        'depot_id': 1,
        'edge_weight_type': 'Exact' # default
    }

    section = None
    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith('NAME'):
            data['name'] = line.split(':')[1].strip()
        elif line.startswith('DIMENSION'):
            data['dimension'] = int(line.split(':')[1].strip())
        elif line.startswith('EDGE_WEIGHT_TYPE'):
            data['edge_weight_type'] = line.split(':')[1].strip()
        elif line.startswith('CAPACITY'):
            data['capacity'] = int(line.split(':')[1].strip())
        elif line.startswith('NODE_COORD_SECTION'):
            section = 'COORD'
        elif line.startswith('DEMAND_SECTION'):
            section = 'DEMAND'
        elif line.startswith('DEPOT_SECTION'):
            section = 'DEPOT'
        elif line.startswith('EOF'):
            break
        elif section == 'COORD':
            parts = line.split()
            if len(parts) >= 3:
                nid = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                data['nodes'][nid] = (x, y)
        elif section == 'DEMAND':
            parts = line.split()
            if len(parts) >= 2:
                nid = int(parts[0])
                demand = int(parts[1])
                data['demands'][nid] = demand
    
    return data

def read_sol_file(filepath):
    """
    Reads a solution file and returns updated cost and a list of routes.
    Each route is a list of node IDs.
    """
    routes = []
    cost = None
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        if line.startswith('Route'):
            # Example: Route #1: 3707 2869 ...
            parts = line.split(':')
            if len(parts) > 1:
                nodes_str = parts[1].strip()
                route_nodes = [int(x) for x in re.findall(r'\d+', nodes_str)]
                if route_nodes:
                    routes.append(route_nodes)
        elif line.startswith('Cost'):
            # Example: Cost 111395
            parts = line.split()
            if len(parts) > 1:
                try:
                    cost = float(parts[1])
                except ValueError:
                    pass
    return cost, routes

def calculate_distance(p1, p2, edge_weight_type='Exact'):
    """
    Calculates distance based on edge_weight_type.
    """
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    dist = math.sqrt(dx*dx + dy*dy)

    if edge_weight_type == 'EUC_2D':
        return math.floor(dist + 0.5)
    elif edge_weight_type == 'CEIL_2D':
        return math.ceil(dist)
    else:
        return dist

def calculate_total_cost(vrp_data, routes):
    """
    Calculates the total cost of the solution.
    """
    total_cost = 0
    depot_id = vrp_data['depot_id']
    if depot_id not in vrp_data['nodes']:
        depot_id = 1
        
    depot_coord = vrp_data['nodes'][depot_id]
    edge_weight_type = vrp_data.get('edge_weight_type', 'Exact')

    for route in routes:
        if not route:
            continue
        
        # From depot to first node
        first_node_id = route[0]
        total_cost += calculate_distance(depot_coord, vrp_data['nodes'][first_node_id], edge_weight_type)

        # Traverse the route
        for i in range(len(route) - 1):
            n1_id = route[i]
            n2_id = route[i+1]
            total_cost += calculate_distance(vrp_data['nodes'][n1_id], vrp_data['nodes'][n2_id], edge_weight_type)
        
        # From last node back to depot
        last_node_id = route[-1]
        total_cost += calculate_distance(vrp_data['nodes'][last_node_id], depot_coord, edge_weight_type)
        
    return total_cost

def convert_to_dataset_format(vrp_data, cost):
    """
    Converts the data into the list format expected by the dataset loader.
    """
    output_list = []
    
    # Name
    output_list.append('name')
    output_list.append(vrp_data['name'])
    
    # Depot
    depot_id = vrp_data['depot_id']
    if depot_id not in vrp_data['nodes']:
        depot_id = 1

    depot_coord = vrp_data['nodes'][depot_id]
    output_list.append('depot')
    output_list.append(int(depot_coord[0]) if depot_coord[0].is_integer() else depot_coord[0])
    output_list.append(int(depot_coord[1]) if depot_coord[1].is_integer() else depot_coord[1])
    
    # Customer coords
    output_list.append('customer')
    sorted_node_ids = sorted(vrp_data['nodes'].keys())
    for nid in sorted_node_ids:
        if nid == depot_id:
            continue
        coord = vrp_data['nodes'][nid]
        output_list.append(int(coord[0]) if coord[0].is_integer() else coord[0])
        output_list.append(int(coord[1]) if coord[1].is_integer() else coord[1])
        
    # Demand
    output_list.append('demand')
    # Depot demand (usually 0)
    output_list.append(vrp_data['demands'].get(depot_id, 0))
    # Customer demands
    for nid in sorted_node_ids:
        if nid == depot_id:
            continue
        output_list.append(vrp_data['demands'].get(nid, 0))
        
    # Capacity
    output_list.append('capacity')
    output_list.append(vrp_data['capacity'])
    
    # Cost
    output_list.append('cost')
    output_list.append(cost)
    
    # Edge Weight Type
    output_list.append('edge_weight_type')
    output_list.append(vrp_data.get('edge_weight_type', 'Exact'))

    output_list.append('end')
    
    return output_list

def get_dimension_sort_key(filepath):
    """
    Sort key based on DIMENSION in the VRP file.
    """
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if line.strip().startswith('DIMENSION'):
                    dim = int(line.strip().split(':')[1].strip())
                    return (0, dim, os.path.basename(filepath))
    except Exception:
        pass
    
    # Fallback to filename sort if dimension not found
    return (1, os.path.basename(filepath))

def main():
    # Modify this variable to target 'X' or 'XXL'
    target_subfolder = 'survey'
    source_dir = os.path.join('original_files_examples_cvrp', target_subfolder)
    output_file = f'data/CVRPlib_{target_subfolder}_instances_converted.txt'
    
    if not os.path.exists(source_dir):
        print(f"Directory {source_dir} not found.")
        return

    # Find all .vrp files recursively in the target directory
    vrp_files = []
    for root, dirs, files in os.walk(source_dir):
        for filename in files:
            if filename.endswith(".vrp"):
                vrp_files.append(os.path.join(root, filename))
                
    # Sort files by DIMENSION
    vrp_files.sort(key=get_dimension_sort_key)

    items = []
    processed_count = 0
    
    for vrp_path in vrp_files:
        filename = os.path.basename(vrp_path)
        # Assuming solution file is in same directory
        sol_filename = filename.replace(".vrp", ".sol")
        sol_path = os.path.join(os.path.dirname(vrp_path), sol_filename)
        
        if os.path.exists(sol_path):
            print(f"Processing pair: {filename} + {sol_filename}")
            try:
                vrp_data = read_vrp_file(vrp_path)
                sol_cost, routes = read_sol_file(sol_path)
                
                # If cost is found in the sol file, use it. Otherwise calculate it.
                if sol_cost is not None:
                    cost = sol_cost
                else:
                    print(f"Warning: Cost not found in {sol_filename}, calculating from routes.")
                    cost = calculate_total_cost(vrp_data, routes)
                
                dataset_entry_list = convert_to_dataset_format(vrp_data, cost)
                items.append(dataset_entry_list)
                processed_count += 1
            except Exception as e:
                print(f"Error processing {filename}: {e}")
        else:
            print(f"Warning: Solution file for {filename} not found at {sol_path}, skipping.")

    # Write to output file
    if items:
        # Write mode 'w' creates new file
        with open(output_file, 'w') as f:
            for item in items:
                f.write(str(item) + "\n")
        print(f"Successfully wrote {processed_count} instances to {output_file}")
    else:
        print("No paired .vrp and .sol files found to process.")

if __name__ == "__main__":
    main()
