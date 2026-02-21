import re
import sys
import os

def parse_log(file_path):
    results = []
    
    # regex to match instance lines
    # Example: [2025-11-10 01:26:40] Tester_inCVRPlib.py(497) : ep   1/110, Elapsed[3.18m], Remain[5.78h], Dim:  100, Teacher:27591.0000, Student:28279.0000, Gap:2.494%, Time:190.842s
    inst_pattern = re.compile(r"Dim:\s*(\d+).*?Gap:([0-9.]+)%.*?Time:([0-9.]+)s")

    with open(file_path, 'r') as f:
        for line in f:
            # We look for lines containing "Dim:" and "Gap:" and "Time:"
            if "Dim:" in line and "Gap:" in line and "Time:" in line:
                match = inst_pattern.search(line)
                if match:
                    n = int(match.group(1))
                    gap = float(match.group(2))
                    time_val = float(match.group(3))
                    results.append({'n': n, 'gap': gap, 'time': time_val})
    return results

def main():
    log_file = "/public/home/bayp/exp_survey_202509/Improvement/single_solution_based/large neighborhood/direct LNS (restricted)/1_LEHD/CVRP/result_survey/20251110_022327_test__vrp0/log.txt"
    if len(sys.argv) > 1:
        log_file = sys.argv[1]

    if not os.path.exists(log_file):
        print(f"File {log_file} not found")
        return

    data = parse_log(log_file)
    
    # Filter gap <= 100%
    filtered_data = [d for d in data if d['gap'] <= 100.0]
    
    # Buckets matches the original log style
    buckets = [
        (0, 1000),
        (1000, 10000),
        (10000, 100001)
    ]
    
    stats = {b: {'count': 0, 'total_gap': 0.0, 'total_time': 0.0} for b in buckets}
    overall = {'count': 0, 'total_gap': 0.0, 'total_time': 0.0}
    
    for d in filtered_data:
        n = d['n']
        # overall
        overall['count'] += 1
        overall['total_gap'] += d['gap']
        overall['total_time'] += d['time']
        
        # buckets
        for b in buckets:
            if b[0] <= n < b[1]:
                stats[b]['count'] += 1
                stats[b]['total_gap'] += d['gap']
                stats[b]['total_time'] += d['time']
                break
    
    # Print results
    print(f"Original entries: {len(data)}")
    print(f"Filtered entries (gap <= 100%): {len(filtered_data)}")
    print("-" * 60)
    
    for b in buckets:
        s = stats[b]
        count = s['count']
        if count > 0:
            avg_gap = s['total_gap'] / count
            avg_time = s['total_time'] / count
        else:
            avg_gap = 0.0
            avg_time = 0.0
        
        print(f"[{b[0]}, {b[1]})  num={count}, avg_gap={avg_gap:.3f}%, avg_time={avg_time:.2f}s")

    # Overall
    cnt = overall['count']
    if cnt > 0:
        avg_gap = overall['total_gap'] / cnt
        avg_time = overall['total_time'] / cnt
    else:
        avg_gap = 0.0
        avg_time = 0.0
        
    print(f"[All] num={cnt}, avg_gap={avg_gap:.3f}%, avg_time={avg_time:.2f}s")

if __name__ == "__main__":
    main()
