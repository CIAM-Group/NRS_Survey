
import numpy as np
import ast

def test_load(filename):
    print(f"Testing load for {filename}")
    node_coords = []
    costs = []
    names = []
    edge_weight_types = []

    with open(filename, "r") as f:
        line = f.readline().strip()
        parsed = False
        try:
             data_list = ast.literal_eval(line)
             if isinstance(data_list, list) and 'name' in data_list:
                 print("Found NEW format")
                 names.append(str(data_list[data_list.index('name') + 1]))
                 costs.append(float(data_list[data_list.index('cost') + 1]))
                 ew = "Exact"
                 if 'edge_weight_type' in data_list:
                     ew = data_list[data_list.index('edge_weight_type') + 1]
                 edge_weight_types.append(ew)

                 start_idx = data_list.index('customer') + 1
                 if 'end' in data_list:
                     end_idx = data_list.index('end')
                 else:
                     end_idx = len(data_list)
                 
                 coords = np.array(data_list[start_idx:end_idx], dtype=float).reshape(-1, 2)
                 node_coords.append(coords)
                 parsed = True
        except Exception as e:
             print(f"New format failed: {e}")
        
        if not parsed:
            print("Fallback to OLD format")
            # Fallback to old format
            line = line.replace('[', '').replace(']', '').replace("'", "").replace(" ", "")
            parts = line.split(',')
            names.append(parts[0])
            costs.append(float(parts[1]))
            edge_weight_types.append('Exact')
            # Handle potential empty string if trailing comma
            parts = [p for p in parts[2:] if p]
            node_coords.append(np.array(parts, dtype=float).reshape(-1, 2))
            
    print(f"Name: {names[0]}")
    print(f"Cost: {costs[0]}")
    print(f"EW Type: {edge_weight_types[0]}")
    print(f"First coord: {node_coords[0][0]}")
    print(f"Last coord: {node_coords[0][-1]}")
    print("-" * 20)

test_load("/public/home/bayp/exp_survey_202509/Construction/single-stage/appending/1_BQ_fixed/data/TSPlib_large_scale_n8.txt")
test_load("/public/home/bayp/exp_survey_202509/Construction/single-stage/appending/1_BQ_fixed/data/TSPlib_large_n8_converted.txt")
