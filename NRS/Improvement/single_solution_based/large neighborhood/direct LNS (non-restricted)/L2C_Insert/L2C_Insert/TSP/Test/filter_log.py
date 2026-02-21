import re
import ast
import os

def parse_log(file_path):
    print(f"Reading log file from: {file_path}")
    if not os.path.exists(file_path):
        print("File not found.")
        return None, None, None

    with open(file_path, 'r') as f:
        content = f.read()

    def extract_list(name, content):
        # Regex to find "name: [1, 2, 3]"
        # The log lines look like: [timestamp] file.py(line) : name: [...]
        # We search for " name: [" to be safer
        pattern = re.compile(rf"\s{name}\s*:\s*(\[.*?\])", re.DOTALL)
        match = pattern.search(content)
        if match:
            list_str = match.group(1)
            try:
                # Replace newlines if necessary, though literal_eval handles them usually
                return ast.literal_eval(list_str)
            except Exception as e:
                print(f"Error parsing {name}: {e}")
                return []
        return []

    problem_size = extract_list("problem_size", content)
    # Using aug_gap as it matches the 'gap' reported in the summary lines of the log
    gaps = extract_list("aug_gap", content) 
    times = extract_list("inst_time_sec", content)

    if not problem_size or not gaps or not times:
        print("Could not extract all required lists from the log file.")
        print(f"Found problem_size: {len(problem_size) if problem_size else 0}")
        print(f"Found aug_gap: {len(gaps) if gaps else 0}")
        print(f"Found inst_time_sec: {len(times) if times else 0}")
        return None, None, None

    return problem_size, gaps, times

def analyze(problem_size, gaps, times):
    if problem_size is None:
        return

    data = []
    skipped = 0
    for s, g, t in zip(problem_size, gaps, times):
        if g <= 100.0:
            data.append((s, g, t))
        else:
            skipped += 1
    
    print(f"Total instances: {len(problem_size)}")
    print(f"Filtered instances (gap > 100%): {skipped}")
    print(f"Remaining instances: {len(data)}")
    print("-" * 60)

    # Buckets: [0, 1000), [1000, 10000), [10000, 100000]
    # Note: The last bucket in the log output is [10000, 100000].
    # We will assume this covers everything >= 10000.
    
    buckets = [
        {"label": "[0, 1000)", "min": 0, "max": 1000, "data": []},
        {"label": "[1000, 10000)", "min": 1000, "max": 10000, "data": []},
        {"label": "[10000, 100000]", "min": 10000, "max": 100001, "data": []}
    ]
    
    overall_data = data

    for s, g, t in data:
        placed = False
        for b in buckets:
            if b["min"] <= s < b["max"]:
                b["data"].append((s, g, t))
                placed = True
                break
        if not placed:
            # If it doesn't fit in the defined buckets (e.g. > 100000), 
            # check if it should be in the last one or separate.
            # Based on log, the intention is likely 10000 to infinity, or 100000 is the max.
            # If s >= 100001, we might want to flag it or just add to last bucket if appropriate.
            # For now, let's assume the buckets cover the dataset as per the log description.
            pass

    for b in buckets:
        d = b["data"]
        count = len(d)
        if count > 0:
            avg_gap = sum(x[1] for x in d) / count
            avg_time = sum(x[2] for x in d) / count
        else:
            avg_gap = 0.0
            avg_time = 0.0
        print(f"{b['label']}: count: {count}, avg gap: {avg_gap:.3f}%, avg time: {avg_time:.3f}s")

    # Overall
    count = len(overall_data)
    if count > 0:
        avg_gap = sum(x[1] for x in overall_data) / count
        avg_time = sum(x[2] for x in overall_data) / count
    else:
        avg_gap = 0.0
        avg_time = 0.0
        
    print("-" * 60)
    print(f"All instances: count: {count}, avg gap: {avg_gap:.3f}%, avg time: {avg_time:.3f}s")

if __name__ == "__main__":
    log_file = "/public/home/bayp/exp_survey_202509/Improvement/single_solution_based/large neighborhood/direct LNS (non-restricted)/L2C_Insert/L2C_Insert/TSP/Test/result_survey/20251027_163925_RRC_budgets=[1000], RI_inites=[0], coords_norms=[0], knearest=[1], rrc_ranges=[200], k_edge_nums=[100], k_scatter_nums=[100]/log.txt"
    problem_size, gaps, times = parse_log(log_file)
    analyze(problem_size, gaps, times)
