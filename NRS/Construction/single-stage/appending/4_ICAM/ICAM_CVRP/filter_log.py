import re
import sys
import os
import numpy as np

def analyze_log(log_path):
    print(f"Analyzing log file: {log_path}")
    
    if not os.path.exists(log_path):
        print(f"Error: File {log_path} not found.")
        # Try to find it relative to current directory if not found
        if not log_path.startswith('/'):
            # This is just a fallback attempt logic, maybe unnecessary if user provides full path
            pass
        return

    data = []
    
    current_instance = None
    current_size = None
    
    # Regex patterns
    # [2025-10-12 00:37:20] CVRPTester_LIB_Survey.py(167) : Instance name: X-n120-k6, problem_size: 119
    # Using [^,]+ to safely capture name before the comma
    size_pattern = re.compile(r"Instance name:\s*([^,]+),\s*problem_size:\s*(\d+)")
    
    # [2025-10-12 00:37:21] CVRPTester_LIB_Survey.py(232) : No aug score:13730.000, No aug gap:2.985%
    gap_pattern = re.compile(r"No aug gap:\s*([\d.]+)%")

    with open(log_path, 'r') as f:
        for line in f:
            size_match = size_pattern.search(line)
            if size_match:
                current_instance = size_match.group(1).strip()
                current_size = int(size_match.group(2))
                continue
            
            gap_match = gap_pattern.search(line)
            if gap_match and current_size is not None:
                gap = float(gap_match.group(1))
                
                # Check if we should filter
                if gap <= 100.0:
                    data.append({
                        'name': current_instance,
                        'size': current_size,
                        'gap': gap
                    })
                else:
                    print(f"Filtered out {current_instance} (Size: {current_size}) with gap: {gap}%")
                
                # Reset current instance data to avoid duplicate adds
                current_instance = None
                current_size = None

    if not data:
        print("No valid data found.")
        return

    # Define buckets based on log file summary: [0, 1000), [1000, 10000), [10000, 100000]
    buckets = {
        "0-1k": [],
        "1k-10k": [],
        "10k-100k": []
    }
    
    for item in data:
        sz = item['size']
        if sz < 1000:
            buckets["0-1k"].append(item['gap'])
        elif sz < 10000:
            buckets["1k-10k"].append(item['gap'])
        else:
            buckets["10k-100k"].append(item['gap'])

    print("\n" + "="*50)
    print(f"{'Bucket':<15} | {'Count':<10} | {'Avg Gap (%)':<15}")
    print("-" * 50)
    
    overall_gaps = []
    
    for bucket_name, gaps in buckets.items():
        count = len(gaps)
        avg_gap = np.mean(gaps) if count > 0 else 0.0
        overall_gaps.extend(gaps)
        print(f"{bucket_name:<15} | {count:<10} | {avg_gap:<15.4f}")

    print("-" * 50)
    overall_count = len(overall_gaps)
    overall_avg = np.mean(overall_gaps) if overall_count > 0 else 0.0
    print(f"{'Overall':<15} | {overall_count:<10} | {overall_avg:<15.4f}")
    print("="*50 + "\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Default path
        log_file = "./ICAM_CVRP/result_survey_cvrp/20251012_003719_no_aug_test_CVRPLIB_Survey_SeparateNorm/run_log.txt"
    else:
        log_file = sys.argv[1]
    
    analyze_log(log_file)
