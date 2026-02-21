import re
import sys
import os

def process_log(log_path):
    print(f"Processing log file: {log_path}")
    
    if not os.path.exists(log_path):
        print(f"Error: File not found at {log_path}")
        return

    episodes = []
    
    # Define buckets (same as CVRP for consistency)
    buckets = [
        (0, 1000),
        (1000, 10000),
        (10000, 100001)
    ]
    
    bucket_stats = {b: {'count': 0, 'total_gap': 0.0, 'total_time': 0.0} for b in buckets}
    
    # Regex for TSP log
    # Example line: [Timestamp] ... : [OK] inst:national_wi29, n:29, gap:5.662%, time:0.053s
    regex_pattern = re.compile(r"\[OK\] inst:.*?, n:\s*(\d+), gap:\s*([0-9.]+)%, time:\s*([0-9.]+)s")

    valid_lines_count = 0
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            if "model_load" in line:
                match_model = re.search(r"'model_load':\s*\{\s*'path':\s*'([^']+)'", line)
                if match_model:
                     print(f"Model Path: {match_model.group(1)}")

            if "[OK] inst:" in line:
                match = regex_pattern.search(line)
                if match:
                    size = int(match.group(1))
                    gap = float(match.group(2))
                    time = float(match.group(3))
                    
                    # Filtering condition: gap must be <= 100%
                    if gap <= 100.0:
                        episodes.append({'size': size, 'gap': gap, 'time': time})
                        
                        # Add to buckets
                        for b in buckets:
                            if b[0] <= size < b[1]:
                                bucket_stats[b]['count'] += 1
                                bucket_stats[b]['total_gap'] += gap
                                bucket_stats[b]['total_time'] += time
                                break
                    valid_lines_count += 1

    print(f"Read {valid_lines_count} episode lines.")
    if not episodes:
        print("No episodes found matching criteria (gap <= 100%).")
        return

    print("\n" + "#" * 16 + " Filtered Bucket Summary (gap <= 100%) " + "#" * 16)
    
    for b in buckets:
        stats = bucket_stats[b]
        count = stats['count']
        if count > 0:
            avg_gap = stats['total_gap'] / count
            avg_time = stats['total_time'] / count
            print(f"Bucket [{b[0]}, {b[1]}): count={count}, avg_gap={avg_gap:.3f}% avg_time={avg_time:.3f}s")
        else:
             print(f"Bucket [{b[0]}, {b[1]}): count={count}, avg_gap=N/A avg_time=N/A")

    print("\n" + "#" * 16 + " Filtered Overall " + "#" * 16)
    
    total_count = len(episodes)
    overall_gap = sum(ep['gap'] for ep in episodes) / total_count
    overall_time = sum(ep['time'] for ep in episodes) / total_count
    
    print(f"Original instances: {valid_lines_count}")
    print(f"Filtered instances (gap <= 100%): {total_count}")
    print(f"Filtered Overall avg gap: {overall_gap:.3f}%")
    print(f"Filtered Overall avg time: {overall_time:.3f}s")

if __name__ == "__main__":
    # Default path based on the context
    default_log_path = "./TSP/Test_All/result_survey_tsp/20251114_004117_test_random_insertion/log.txt"
    # default_log_path = "./TSP/Test_All/result_survey_tsp/20251025_221240_test_SIL_greedy_model_TSP1K/log.txt"
    # default_log_path = "./TSP/Test_All/result_survey_tsp/20251121_011644_test_SIL_greedy_model_TSP5K/log.txt"
    # default_log_path = "./TSP/Test_All/result_survey_tsp/20251121_012409_test_SIL_greedy_model_TSP10K/log.txt"
    # default_log_path = "./TSP/Test_All/result_survey_tsp/20251121_012508_test_SIL_greedy_model_TSP50K/log.txt"
    # default_log_path = "./TSP/Test_All/result_survey_tsp/20251121_013510_test_SIL_greedy_model_TSP100K/log.txt"

    
    if len(sys.argv) > 1:
        log_path = sys.argv[1]
    else:
        log_path = default_log_path
        
    process_log(log_path)
