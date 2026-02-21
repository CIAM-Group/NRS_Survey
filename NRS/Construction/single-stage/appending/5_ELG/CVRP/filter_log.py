import re
import numpy as np

def analyze_log(log_file):
    # Intervals
    # 0: [0, 1000)
    # 1: [1000, 10000)
    # 2: [10000, 100001)
    
    results = {
        0: {'times': [], 'gaps': []},
        1: {'times': [], 'gaps': []},
        2: {'times': [], 'gaps': []}
    }
    
    instance_sizes = {}
    
    # Regex patterns
    # Pattern for initial size listing: Instance: X-n101-k25, Problem Size: 100
    size_pattern = re.compile(r"Instance:\s+(.+?),\s+Problem Size:\s+(\d+)")
    
    # Pattern for result: Instance X-n936-k151: Time 3.4145s, Cost 153665.0, Gap 15.79%
    result_pattern = re.compile(r"Instance\s+(.+?):\s+Time\s+([\d\.]+)s,\s+Cost\s+[\d\.]+,\s+Gap\s+([\d\.]+)%")
    all_ins_avg_gap = []

    with open(log_file, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Check for size definition
            size_match = size_pattern.match(line)
            if size_match:
                name = size_match.group(1)
                size = int(size_match.group(2))
                instance_sizes[name] = size
                continue
                
            # Check for result
            result_match = result_pattern.match(line)
            if result_match:
                name = result_match.group(1)
                time_val = float(result_match.group(2))
                gap_val = float(result_match.group(3))
                
                if name in instance_sizes:
                    if gap_val > 100:
                        continue
                        
                    size = instance_sizes[name]
                    
                    if size < 1000:
                        idx = 0
                    elif size < 10000:
                        idx = 1
                    elif size < 100001:
                        idx = 2
                    else:
                        idx = -1 # Out of range if any
                        
                    if idx != -1:
                        results[idx]['times'].append(time_val)
                        results[idx]['gaps'].append(gap_val)
                        all_ins_avg_gap.append(gap_val)
                else:
                    print(f"Warning: Size not found for instance {name}")

    # Print summary
    intervals = ["[0, 1000)", "[1000, 10000)", "[10000, 100001)"]
    
    print("-" * 60)
    print(f"{'Interval':<20} | {'Count':<10} | {'Avg Time (s)':<15} | {'Avg Gap (%)':<15}")
    print("-" * 60)
    
    for idx in range(3):
        times = results[idx]['times']
        gaps = results[idx]['gaps']
        
        count = len(times)
        if count > 0:
            avg_time = np.mean(times)
            avg_gap = np.mean(gaps)
            print(f"{intervals[idx]:<20} | {count:<10} | {avg_time:<15.4f} | {avg_gap:<15.2f}")
        else:
            print(f"{intervals[idx]:<20} | {0:<10} | {'N/A':<15} | {'N/A':<15}")
    print(f"0-{max(instance_sizes.values()):<19}| {len(all_ins_avg_gap):<10} | {'N/A':<15} | {np.mean(all_ins_avg_gap):<15.2f}")
    print("-" * 60)

if __name__ == "__main__":
    import os
    
    current_path = os.path.dirname(os.path.abspath(__file__))
    log_path = f"{current_path}/elg_cvrplib_survey.log"
    analyze_log(log_path)
