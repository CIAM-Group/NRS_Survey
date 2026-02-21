import re
import os

def parse_log(file_path):
    print(f"Processing log file: {file_path}")
    
    data = []
    
    # Regex to match the relevant line
    # [2025-11-10 02:14:34] Tester.py(428) : Episode   1/110 done.Instance: X-n101-k25, size: 100, opt: 27591.000, student: 34663.000, gap: 25.632%, time: 329.184s
    pattern = re.compile(r"size:\s+(\d+),.*gap:\s+([0-9.]+)%,.*time:\s+([0-9.]+)s")
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if "Episode" in line and "done.Instance:" in line:
                    match = pattern.search(line)
                    if match:
                        n = int(match.group(1))
                        gap = float(match.group(2))
                        time = float(match.group(3))
                        data.append({'n': n, 'gap': gap, 'time': time})
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return

    print(f"Total entries found: {len(data)}")
    
    # Filter 
    filtered_data = [d for d in data if d['gap'] <= 100.0]
    print(f"Entries after filtering gap > 100%: {len(filtered_data)}")
    print("-" * 30)

    # Buckets definition
    # [0, 1000), [1000, 10000), [10000, 100000] (inclusive on the right for the last one based on log)
    buckets = [
        {'range': (0, 1000), 'label': '[0,1000)', 'data': []},
        {'range': (1000, 10000), 'label': '[1000,10000)', 'data': []},
        {'range': (10000, 100001), 'label': '[10000,100000]', 'data': []}, # 100001 to include 100000
    ]
    
    for item in filtered_data:
        placed = False
        for bucket in buckets:
            low, high = bucket['range']
            if low <= item['n'] < high:
                bucket['data'].append(item)
                placed = True
                break
        if not placed:
            pass

    # Calculate stats
    overall_gap_sum = 0
    overall_time_sum = 0
    overall_count = 0

    for bucket in buckets:
        count = len(bucket['data'])
        if count > 0:
            avg_gap = sum(d['gap'] for d in bucket['data']) / count
            avg_time = sum(d['time'] for d in bucket['data']) / count
            print(f"{bucket['label']}: count={count}, avg_gap={avg_gap:.3f}%, avg_time={avg_time:.3f}s")
            
            overall_count += count
            overall_gap_sum += sum(d['gap'] for d in bucket['data'])
            overall_time_sum += sum(d['time'] for d in bucket['data'])
        else:
            print(f"{bucket['label']}: count=0, avg_gap=N/A, avg_time=N/A")

    print("-" * 30)
    if overall_count > 0:
        overall_avg_gap = overall_gap_sum / overall_count
        overall_avg_time = overall_time_sum / overall_count
        print(f"Overall: count={overall_count}, avg_gap={overall_avg_gap:.3f}%, avg_time={overall_avg_time:.3f}s")
    else:
        print("Overall: No data")

if __name__ == "__main__":
    # Explicitly define the log file path
    base_dir = "/public/home/bayp/exp_survey_202509/Improvement/single_solution_based/large neighborhood/direct LNS (restricted)/2_SIL/CVRP/Test_All/result_survey_cvrp"
    # sub_dir = "20251110_020903_test_SIL_PRC1000_model_CVRP1K/log.txt"
    # sub_dir = "20251121_221237_test_SIL_PRC1000_model_CVRP5K/log.txt"
    # sub_dir = "20251121_221606_test_SIL_PRC1000_model_CVRP10K/log.txt"
    # sub_dir = "20251121_221942_test_SIL_PRC1000_model_CVRP50K/log.txt"
    sub_dir = "20251121_222206_test_SIL_PRC1000_model_CVRP100K/log.txt"

    log_file = os.path.join(base_dir, sub_dir)
    
    print(f"Log file address: {log_file}")
    parse_log(log_file)
