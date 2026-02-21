import os
import re
import math

def get_tsp_cost_dict(py_file_path):
    # Read the python file content
    with open(py_file_path, 'r') as f:
        content = f.read()
    
    # Exec to get the dictionary
    # Assuming the variable name is tsp_survey_bench_cost_all
    local_scope = {}
    try:
        exec(content, {}, local_scope)
        return local_scope.get('tsp_survey_bench_cost_all', {})
    except Exception as e:
        print(f"Error reading cost file: {e}")
        return {}

def read_tsp_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    data = {
        'name': '',
        'dimension': 0,
        'edge_weight_type': 'Exact', # Default
        'nodes': {},
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
        elif line.startswith('NODE_COORD_SECTION'):
            section = 'COORD'
        elif line.startswith('EOF'):
            break
        elif section == 'COORD':
            parts = line.split()
            # ID X Y
            if len(parts) >= 3:
                node_id = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                data['nodes'][node_id] = (x, y)
    return data

def convert_to_dataset_format(tsp_data, cost):
    output_list = []
    output_list.append('name')
    output_list.append(tsp_data['name'])
    
    output_list.append('cost')
    output_list.append(cost)
    
    output_list.append('edge_weight_type')
    output_list.append(tsp_data.get('edge_weight_type', 'Exact'))
    
    output_list.append('customer')
    # Sort by ID to ensure order. TSPLIB usually 1-n.
    for nid in sorted(tsp_data['nodes'].keys()):
        x, y = tsp_data['nodes'][nid]
        output_list.append(int(x) if x.is_integer() else x)
        output_list.append(int(y) if y.is_integer() else y)
        
    output_list.append('end')
    return output_list

def get_dimension_sort_key(filepath):
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if line.strip().startswith('DIMENSION'):
                    dim = int(line.strip().split(':')[1].strip())
                    return (0, dim, os.path.basename(filepath))
    except Exception:
        pass
    return (1, os.path.basename(filepath))

def main():
    target_folder = 'original_files_examples_tsp/survey'
    cost_file_name = 'survey_bench_opt_tsp_same_name_inside.py'
    output_file = 'data/TSP_survey_converted.txt'
    
    if not os.path.exists(target_folder):
        print(f"Directory {target_folder} not found.")
        return

    cost_file_path = os.path.join(target_folder, cost_file_name)
    cost_dict = get_tsp_cost_dict(cost_file_path)
    if not cost_dict:
        print("Warning: No costs loaded.")

    start_cwd = os.getcwd()
    
    # Find all .tsp files
    tsp_files = []
    for root, dirs, files in os.walk(target_folder):
        for filename in files:
            if filename.endswith(".tsp"):
                tsp_files.append(os.path.join(root, filename))
                
    tsp_files.sort(key=get_dimension_sort_key)

    items = []
    processed_count = 0
    
    for tsp_path in tsp_files:
        try:
            tsp_data = read_tsp_file(tsp_path)
            name = tsp_data['name']
            
            # Lookup cost
            cost = cost_dict.get(name)
            
            if cost is None:
                # Try partial match or manual check?
                # The user said "cost in the .py file". 
                # Some names in py file might differ slightly? 
                # e.g. "brd14051" vs "tsplib_euc_brd14051.tsp" -> NAME inside is "brd14051" ?
                # The read_tsp_file gets NAME from file content.
                # Let's check if successful.
                print(f"Warning: Cost not found for {name} (File: {os.path.basename(tsp_path)})")
                cost = 0 # Or skip?
            
            dataset_entry = convert_to_dataset_format(tsp_data, cost)
            items.append(dataset_entry)
            processed_count += 1
            print(f"Processed {name}, Cost: {cost}")
            
        except Exception as e:
            print(f"Error processing {tsp_path}: {e}")

    # Write output
    if items:
        with open(output_file, 'w') as f:
            for item in items:
                f.write(str(item) + "\n")
        print(f"Successfully wrote {processed_count} instances to {output_file}")

if __name__ == "__main__":
    main()
