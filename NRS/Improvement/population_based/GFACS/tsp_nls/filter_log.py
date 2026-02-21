import re
import numpy as np
import sys
import os

def analyze_log(log_path):
    if not os.path.exists(log_path):
        print(f"Error: File '{log_path}' not found.")
        return

    print(f"Analyzing log file: {log_path}")

    # Buckets storage
    # bucket: (gaps_list, times_list)
    stats = {
        "lt_1k":   ([], []),  # [0, 1000)
        "lt_10k":  ([], []),  # [1000, 10000)
        "lt_100k": ([], []),  # [10000, 100000]
        "all":     ([], [])   # All valid
    }

    # Regex pattern to match result lines
    # Example: 2025-12-07 00:20:42,927 | INFO | Instance wi29, dim=29, length=27603, optimal=27603, gap=0.000%, time=7.1019s
    # Note: Sometimes length/optimal might be NA, but usually gap is valid float if optimal exists. 
    # If gap=NA, we skip calculation or handle separate. The user asked to filter gap>100, implies valid numeric gap.
    
    # Matching pattern: ... Instance {name}, dim={dim}, length={len}, optimal={opt}, gap={gap}%, time={time}s
    pattern = re.compile(r"Instance\s+.*,\s+dim=(\d+),\s+length=.*,\s+optimal=.*,\s+gap=([0-9\.]+)%,\s+time=([0-9\.]+)s")

    ignored_gap_count = 0
    ignored_time_count = 0
    total_parsed = 0
    model_checkpoint = "Unknown"

    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            if "checkpoint:" in line:
                parts = line.split("checkpoint:")
                if len(parts) > 1:
                    model_checkpoint = parts[1].strip()

            if "Instance" not in line or "gap=" not in line:
                continue
            
            match = pattern.search(line)
            if match:
                total_parsed += 1
                dim = int(match.group(1))
                gap = float(match.group(2))
                time_val = float(match.group(3))

                # Filters
                if gap > 100.0:
                    ignored_gap_count += 1
                    continue
                
                if time_val > 36000.0:
                    ignored_time_count += 1
                    continue

                # Add to buckets
                is_stored = False
                if dim < 1000:
                    stats["lt_1k"][0].append(gap)
                    stats["lt_1k"][1].append(time_val)
                    is_stored = True
                elif dim < 10000:
                    stats["lt_10k"][0].append(gap)
                    stats["lt_10k"][1].append(time_val)
                    is_stored = True
                elif dim <= 100000:
                    stats["lt_100k"][0].append(gap)
                    stats["lt_100k"][1].append(time_val)
                    is_stored = True
                
                if is_stored:
                    stats["all"][0].append(gap)
                    stats["all"][1].append(time_val)

    # Helper for average
    def _avg(lst):
        return np.mean(lst) if len(lst) > 0 else 0.0

    print("-" * 60)
    print(f"Model Checkpoint: {model_checkpoint}")
    print(f"Total instances parsed: {total_parsed}")
    print(f"Ignored (Gap > 100%): {ignored_gap_count}")
    print(f"Ignored (Time > 36000s): {ignored_time_count}")
    print(f"Valid instances: {len(stats['all'][0])}")
    print("-" * 60)
    
    print(f"{'Scale Range':<20} | {'Num':<5} | {'Avg Gap (%)':<12} | {'Avg Time (s)':<12}")
    print("-" * 60)

    # Print buckets
    # [0, 1000)
    gaps, times = stats["lt_1k"]
    print(f"{'[0, 1000)':<20} | {len(gaps):<5} | {_avg(gaps):<12.3f} | {_avg(times):<12.4f}")

    # [1000, 10000)
    gaps, times = stats["lt_10k"]
    print(f"{'[1000, 10000)':<20} | {len(gaps):<5} | {_avg(gaps):<12.3f} | {_avg(times):<12.4f}")

    # [10000, 100000]
    gaps, times = stats["lt_100k"]
    print(f"{'[10000, 100000]':<20} | {len(gaps):<5} | {_avg(gaps):<12.3f} | {_avg(times):<12.4f}")

    print("-" * 60)
    # Total
    gaps, times = stats["all"]
    print(f"{'Total (All Valid)':<20} | {len(gaps):<5} | {_avg(gaps):<12.3f} | {_avg(times):<12.4f}")
    print("-" * 60)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_log.py <path_to_log_file>")
        # Default to a know log if exists for quick testing, otherwise exit

        # default_log = "./20251207_0035_GFACS_last_iter_no_ls_model200_TSP.log"
        # default_log = "./20251207_0039_GFACS_last_iter_no_ls_model500_TSP.log"
        # default_log = "./20251207_0040_GFACS_last_iter_no_ls_model1000_TSP.log"
        # default_log = "./20251207_0025_GFACS_model200_TSP.log"
        # default_log = "./20251207_0026_GFACS_model500_TSP.log"
        default_log = "./20251207_0026_GFACS_model1000_TSP.log"
        
        
        
        if os.path.exists(default_log):
             analyze_log(default_log)
        else:
             print(f"Example log '{default_log}' not found.")
    else:
        analyze_log(sys.argv[1])
