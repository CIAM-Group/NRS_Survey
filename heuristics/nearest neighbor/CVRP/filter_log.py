import re
import sys
import os

def parse_and_analyze(log_file_path):
    print(f"Reading log file: {log_file_path}")
    
    if not os.path.exists(log_file_path):
        print(f"Error: File not found at {log_file_path}")
        return

    data = []
    
    # Regex patterns
    # 2025-11-14 17:59:41,351 - trainer - INFO - Instance name: X-n120-k6, problem_size: 119
    pattern_dim = re.compile(r"Instance name: .*, problem_size: (\d+)")
    
    # 2025-11-14 17:59:41,353 - trainer - INFO - No aug score:15848.000, No aug gap:18.872%
    pattern_gap = re.compile(r"No aug score:[\d.]+, No aug gap:([\d.]+)%")
    
    # 2025-11-14 17:59:41,353 - trainer - INFO - Instance time (NN only): 0.002s
    pattern_time = re.compile(r"Instance time \(NN only\): ([\d.]+)s")

    current_instance = {}
    skipped_count = 0

    with open(log_file_path, 'r') as f:
        for line in f:
            # Check for Dimension
            match_dim = pattern_dim.search(line)
            if match_dim:
                # If we were already tracking an instance but didn't finish it (unexpected), just overwrite
                current_instance = {'dim': int(match_dim.group(1))}
                continue

            # Check for Gap
            match_gap = pattern_gap.search(line)
            if match_gap and 'dim' in current_instance:
                current_instance['gap'] = float(match_gap.group(1))
                continue

            # Check for Time and finish instance
            match_time = pattern_time.search(line)
            if match_time and 'dim' in current_instance and 'gap' in current_instance:
                current_instance['time'] = float(match_time.group(1))
                
                # Filter logic
                if current_instance['gap'] <= 100.0:
                    data.append(current_instance)
                else:
                    skipped_count += 1
                
                # Reset
                current_instance = {}

    if skipped_count > 0:
        print(f"Filtered out {skipped_count} instances with GAP > 100%.")

    if not data:
        print("No valid data found after filtering.")
        return

    # Bucket definitions
    # Adjust buckets if needed, assuming same as TSP for now: [0, 1000), [1000, 10000), [10000, 100000]
    # But based on log snippet, the log header says "Test scale range: [0, 1000]", but let's keep the general buckets
    buckets = [
        {"label": "[0, 1000)",       "check": lambda d: 0 <= d < 1000},
        {"label": "[1000, 10000)",   "check": lambda d: 1000 <= d < 10000},
        {"label": "[10000, 100000]", "check": lambda d: 10000 <= d <= 100000},
    ]
    
    print("\n#################  Bucket Summary (GAP <= 100%)  #################")
    for bucket in buckets:
        bucket_data = [d for d in data if bucket["check"](d['dim'])]
        count = len(bucket_data)
        if count > 0:
            avg_gap = sum(d['gap'] for d in bucket_data) / count
            avg_time = sum(d['time'] for d in bucket_data) / count
        else:
            avg_gap = 0.0
            avg_time = 0.0

        print(f"{bucket['label']}, number: {count}, avg GAP: {avg_gap:.3f}%, avg time: {avg_time:.3f}s")

    # Overall Summary
    total_count = len(data)
    total_avg_gap = sum(d['gap'] for d in data) / total_count
    total_avg_time = sum(d['time'] for d in data) / total_count

    print("###################################  Overall Filtered Summary  ##########################################")
    print(f"All solved instances (filtered), number: {total_count}, avg GAP: {total_avg_gap:.3f}%, avg time: {total_avg_time:.3f}s")


if __name__ == "__main__":
    # Default log path for CVRP
    default_log = "/public/home/bayp/exp_survey_202509/heuristics/nearest neighbor/CVRP/result_nn_cvrp/20260113_223644_1683/run_log.txt"
    
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        if os.path.exists(default_log):
            path = default_log
        elif os.path.exists(os.path.join("CVRP", default_log)):
            path = os.path.join("CVRP", default_log)
        else:
            # Fallback
            path = "/public/home/bayp/exp_survey_202509/heuristics/nearest neighbor/CVRP/result_nn_cvrp/20260113_223644_1683/run_log.txt"

    parse_and_analyze(path)
