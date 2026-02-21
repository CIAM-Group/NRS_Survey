import re
import sys
import os

def parse_log(file_path):
    results = []
    
    # regex to match instance lines
    # Example: [2025-10-23 01:29:55] TSPTester_inTSPlib.py(656) : [Inst] national_wi29 (n=29) opt=27603.00, stu=27603.00, gap=0.000%, time=41.32s
    inst_pattern = re.compile(r"\[Inst\].*?\(n=(\d+)\).*?gap=([0-9.]+)%.*?time=([0-9.]+)s")

    with open(file_path, 'r') as f:
        for line in f:
            if "[Inst]" in line:
                match = inst_pattern.search(line)
                if match:
                    n = int(match.group(1))
                    gap = float(match.group(2))
                    time_val = float(match.group(3))
                    results.append({'n': n, 'gap': gap, 'time': time_val})
    return results

def main():
    log_file = "/public/home/bayp/exp_survey_202509/Improvement/single_solution_based/large neighborhood/direct LNS (restricted)/1_LEHD/TSP/result_survey/20251023_022909_test_tsp_survey_RRC_budget1000/log.txt"
    if len(sys.argv) > 1:
        log_file = sys.argv[1]

    if not os.path.exists(log_file):
        # try strictly relative path if absolute logic fails, though usually current dir is fine
        # check if we are running from a weird place
        print(f"File {log_file} not found in {os.getcwd()}")
        return

    data = parse_log(log_file)
    
    # Filter gap <= 100%
    filtered_data = [d for d in data if d['gap'] <= 100.0]
    
    # Buckets from the original log file at the end
    # [0, 1000), [1000, 10000), [10000, 100001)
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
