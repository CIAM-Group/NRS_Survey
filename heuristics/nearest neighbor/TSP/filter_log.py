import re
import sys
import os

def parse_and_analyze(log_file_path):
    print(f"Reading log file: {log_file_path}")
    
    if not os.path.exists(log_file_path):
        print(f"Error: File not found at {log_file_path}")
        return

    data = []
    # Regex pattern to match lines like:
    # [1/228] Instance: fqm5087, dim: 5087, BKS: 13029, NN cost: 17874, GAP: 37.186%, time: 0.464s
    pattern = re.compile(r"Instance: .+, dim: (\d+), .+, GAP: ([\d.]+)%, time: ([\d.]+)s")

    skipped_count = 0

    with open(log_file_path, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                dim = int(match.group(1))
                gap = float(match.group(2))
                time = float(match.group(3))
                
                if gap <= 100.0:
                    data.append({
                        'dim': dim,
                        'gap': gap,
                        'time': time
                    })
                else:
                    skipped_count += 1
    
    if skipped_count > 0:
        print(f"Filtered out {skipped_count} instances with GAP > 100%.")

    if not data:
        print("No valid data found after filtering.")
        return

    # Bucket definitions
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
    # Look for the specific log file in the likely location relative to this script or current dir
    default_log = "result_survey_tsp_nn/20251216_173955_NN_TSPLIB/run_log.txt"
    
    # Check if a path was passed as argument
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        # Check current directory
        if os.path.exists(default_log):
            path = default_log
        # Check if we are in the parent directory (one level up)
        elif os.path.exists(os.path.join("TSP", default_log)):
            path = os.path.join("TSP", default_log)
        else:
            # Fallback to the absolute path seen in context (best effort)
            path = "/public/home/bayp/exp_survey_202509/heuristics/nearest neighbor/TSP/result_survey_tsp_nn/20251216_173955_NN_TSPLIB/run_log.txt"

    parse_and_analyze(path)
