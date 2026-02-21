import re
import numpy as np
import sys
import os
import glob
import argparse

def analyze_file(log_path):
    print(f"\n{'='*80}")
    print(f"Analyzing log file: {log_path}")
    
    if not os.path.exists(log_path):
        print(f"Error: File '{log_path}' not found.")
        return

    # Buckets storage
    # bucket: (gaps_list, times_list)
    stats = {
        "lt_1k":   ([], []),  # [0, 1000)
        "lt_10k":  ([], []),  # [1000, 10000)
        "lt_100k": ([], []),  # [10000, 100000+)
        "all":     ([], [])   # All valid
    }

    # Regex pattern to match result lines
    # Example: Instance X-n101-k25, dim=100, length=42946, optimal=27591.0, gap=55.652%, time=124.5816s)
    # Handling optional trailing ')'
    pattern = re.compile(r"Instance\s+.*,\s+dim=(\d+),.*gap=([0-9\.]+)%,.*time=([0-9\.]+)s")

    ignored_gap_count = 0
    ignored_time_count = 0
    total_parsed = 0
    model_checkpoint = "Unknown"

    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            if "checkpoint:" in line.lower() or "checkpoint" in line.lower():
                 # Attempt to capture checkpoint path if present prominently
                 # Typical line: ... | INFO | Checkpoint: /path/to/model.pt
                 parts = line.split("point:", 1)
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
                else: # >= 10000
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
    print("-" * 80)
    
    print(f"{'Scale Range':<20} | {'Num':<5} | {'Avg Gap (%)':<12} | {'Avg Time (s)':<12}")
    print("-" * 80)

    # Print buckets
    # [0, 1000)
    gaps, times = stats["lt_1k"]
    print(f"{'[0, 1000)':<20} | {len(gaps):<5} | {_avg(gaps):<12.3f} | {_avg(times):<12.4f}")

    # [1000, 10000)
    gaps, times = stats["lt_10k"]
    print(f"{'[1000, 10000)':<20} | {len(gaps):<5} | {_avg(gaps):<12.3f} | {_avg(times):<12.4f}")
    
    # [10000, 100001) or > 10000
    gaps, times = stats["lt_100k"]
    print(f"{'>= 10000':<20} | {len(gaps):<5} | {_avg(gaps):<12.3f} | {_avg(times):<12.4f}")

    print("-" * 80)
    # Total
    gaps, times = stats["all"]
    print(f"{'Total (Valid)':<20} | {len(gaps):<5} | {_avg(gaps):<12.3f} | {_avg(times):<12.4f}")
    print("-" * 80)

def main():
    # Hardcoded paths
    print("Using hardcoded paths...")
    paths = [
        "cvrp_nls/survey_results/",
        "cvrp_nls/survey_results_lastgen_no_ls/"
    ]

    files_to_process = []
    for p in paths:
        if os.path.isfile(p):
            files_to_process.append(p)
        elif os.path.isdir(p):
            # scan dictionary for .log files
            found = glob.glob(os.path.join(p, "*.log"))
            files_to_process.extend(found)
        else:
            # Try glob expansion (shell might not have done it)
            found = glob.glob(p)
            files_to_process.extend(found)

    # Deduplicate and sort
    files_to_process = sorted(list(set(files_to_process)))

    if not files_to_process:
        print("No log files found.")
        return

    for f in files_to_process:
        analyze_file(f)

if __name__ == "__main__":
    main()
