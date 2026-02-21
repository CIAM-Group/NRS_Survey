import re
import os

def parse_log(file_path):
    print(f"Reading log file from: {file_path}")
    if not os.path.exists(file_path):
        print("File not found.")
        return None, None, None

    problem_size = []
    gaps = []
    times = []

    with open(file_path, 'r') as f:
        for line in f:
            if "[Inst]" in line and "size:" in line:
                # Regex to extract size, time, and gap
                # Example: [Inst] name:X-n101-k25, size:100, time:323.343s, gap:2.541%, ...
                match = re.search(r"size:(\d+),\s*time:([\d\.]+)s,\s*gap:([\d\.]+)%", line)
                if match:
                    try:
                        s = int(match.group(1))
                        t = float(match.group(2))
                        g = float(match.group(3))
                        problem_size.append(s)
                        times.append(t)
                        gaps.append(g)
                    except ValueError:
                        pass
    
    if not problem_size:
        print("No instances found.")
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
    
    print(f"Total instances found: {len(problem_size)}")
    print(f"Filtered instances (gap > 100%): {skipped}")
    print(f"Remaining instances: {len(data)}")
    print("-" * 60)

    # Buckets: [0, 1000), [1000, 10000), [10000, 100001)
    buckets = [
        {"label": "[0, 1000)",     "min": 0,     "max": 1000,   "data": []},
        {"label": "[1000, 10000)", "min": 1000,  "max": 10000,  "data": []},
        {"label": "[10000, 100001)", "min": 10000, "max": 100001, "data": []}
    ]
    
    overall_data = data

    for s, g, t in data:
        placed = False
        for b in buckets:
            if b["min"] <= s < b["max"]:
                b["data"].append((s, g, t))
                placed = True
                break

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
    log_file = "/public/home/bayp/exp_survey_202509/Improvement/single_solution_based/large neighborhood/direct LNS (non-restricted)/L2C_Insert/L2C_Insert/CVRP/Test/result_survey/20251029_033029_scales=[0], RRC_budgets=[1000], RI_inites=[0], coords_norms=[0], knearest=[0], rrc_ranges=[1000], k_edge_nums=[200], k_scatter_nums=[100]/log.txt"
    problem_size, gaps, times = parse_log(log_file)
    analyze(problem_size, gaps, times)
