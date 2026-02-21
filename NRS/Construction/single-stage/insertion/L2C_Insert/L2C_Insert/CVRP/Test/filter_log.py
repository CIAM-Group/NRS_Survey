import re
import sys
import os

def parse_log(file_path):
    # CVRP format: [Inst] name:X-n101-k25, size:100, time:1.105s, gap:7.807%, score_teacher:27591.0000, score_student:29745.0000
    regex = r"\[Inst\] name:.+?, size:(\d+), time:([\d\.]+)s, gap:([\d\.]+)%"
    
    results = []
    
    with open(file_path, 'r') as f:
        for line in f:
            match = re.search(regex, line)
            if match:
                dim = int(match.group(1))
                time = float(match.group(2))
                gap = float(match.group(3))
                
                if gap <= 100.0:
                    results.append({
                        'dim': dim,
                        'gap': gap,
                        'time': time
                    })
    return results

def calculate_stats(results):
    buckets = {
        '[0, 1000)': {'min': 0, 'max': 1000, 'items': []},
        '[1000, 10000)': {'min': 1000, 'max': 10000, 'items': []},
        '[10000, 100001)': {'min': 10000, 'max': 100001, 'items': []}
    }
    
    for res in results:
        dim = res['dim']
        if dim < 1000:
            buckets['[0, 1000)']['items'].append(res)
        elif dim < 10000:
            buckets['[1000, 10000)']['items'].append(res)
        else:
            buckets['[10000, 100001)']['items'].append(res)
            
    print("-" * 60)
    print(f"{'Bucket':<20} | {'Count':<5} | {'Avg Gap (%)':<12} | {'Avg Time (s)':<12}")
    print("-" * 60)
    
    total_items = []
    
    for name, bucket in buckets.items():
        items = bucket['items']
        total_items.extend(items)
        count = len(items)
        if count > 0:
            avg_gap = sum(item['gap'] for item in items) / count
            avg_time = sum(item['time'] for item in items) / count
            print(f"{name:<20} | {count:<5} | {avg_gap:<12.3f} | {avg_time:<12.3f}")
        else:
            print(f"{name:<20} | {0:<5} | {'N/A':<12} | {'N/A':<12}")

    print("-" * 60)
    
    count = len(total_items)
    if count > 0:
        avg_gap = sum(item['gap'] for item in total_items) / count
        avg_time = sum(item['time'] for item in total_items) / count
        print(f"{'Total':<20} | {count:<5} | {avg_gap:<12.3f} | {avg_time:<12.3f}")
    else:
        print(f"{'Total':<20} | {0:<5} | {'N/A':<12} | {'N/A':<12}")
    print("-" * 60)

if __name__ == "__main__":
    log_file = "/public/home/bayp/exp_survey_202509/Construction/single-stage/insertion/L2C_Insert/L2C_Insert/CVRP/Test/result_survey/20251028_001850_scales=[0], RRC_budgets=[0], RI_inites=[0], coords_norms=[0], knearest=[0], rrc_ranges=[1000], k_edge_nums=[200], k_scatter_nums=[100]/log.txt"
    
    if not os.path.exists(log_file):
        print(f"Error: File '{log_file}' not found.")
        sys.exit(1)
        
    results = parse_log(log_file)
    calculate_stats(results)