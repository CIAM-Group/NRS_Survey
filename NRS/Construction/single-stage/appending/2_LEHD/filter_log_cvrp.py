import re
import argparse
import sys
import numpy as np

def get_bucket(n):
    # Buckets based on similar logic to TSP: [0, 1000), [1000, 10000), [10000, 100001)
    if 0 <= n < 1000: return (0, "[0, 1000)")
    if 1000 <= n < 10000: return (1, "[1000, 10000)")
    if 10000 <= n < 100001: return (2, "[10000, 100000)")
    return (3, ">= 100000")

def analyze_log(file_path):
    print(f"Reading log file: {file_path}")
    
    # Regex to parse the CVRP log line
    # Example: ... Dim:  100, Teacher:27591.0000, Student:30613.0000, Gap:10.953%, Time:0.849s
    pattern = re.compile(r"Dim:\s*(\d+).*Gap:([\d\.]+)%.*Time:([\d\.]+)s")
    
    data = []
    
    skipped_count = 0
    total_parsed = 0
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                # Look for lines containing "Gap:" and "Dim:"
                if "Gap:" in line and "Dim:" in line:
                    match = pattern.search(line)
                    if match:
                        total_parsed += 1
                        n = int(match.group(1))
                        gap = float(match.group(2))
                        time = float(match.group(3))
                        
                        if gap > 100.0:
                            skipped_count += 1
                            continue
                            
                        bucket_sort, bucket_name = get_bucket(n)
                        data.append({
                            'n': n,
                            'gap': gap,
                            'time': time,
                            'bucket': bucket_name,
                            'sort': bucket_sort
                        })
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return

    print(f"Total instances found: {total_parsed}")
    print(f"Instances skipped (gap > 100%): {skipped_count}")
    print(f"Instances remaining: {len(data)}")
    print("-" * 65)
    
    if not data:
        print("No valid data found.")
        return

    # Aggregate results
    buckets = {}
    for item in data:
        b_name = item['bucket']
        if b_name not in buckets:
            buckets[b_name] = {'gaps': [], 'times': [], 'sort': item['sort'], 'count': 0}
        buckets[b_name]['gaps'].append(item['gap'])
        buckets[b_name]['times'].append(item['time'])
        buckets[b_name]['count'] += 1
        
    # Sort buckets
    sorted_bucket_names = sorted(buckets.keys(), key=lambda k: buckets[k]['sort'])
    
    print(f"{'Bucket':<20} | {'Count':<6} | {'Avg Gap (%)':<12} | {'Avg Time (s)':<12}")
    print("-" * 65)
    
    all_gaps = []
    all_times = []
    
    for b_name in sorted_bucket_names:
        stats = buckets[b_name]
        avg_gap = np.mean(stats['gaps'])
        avg_time = np.mean(stats['times'])
        count = stats['count']
        
        all_gaps.extend(stats['gaps'])
        all_times.extend(stats['times'])
        
        print(f"{b_name:<20} | {count:<6} | {avg_gap:12.4f} | {avg_time:12.4f}")
        
    print("-" * 65)
    
    overall_avg_gap = np.mean(all_gaps)
    overall_avg_time = np.mean(all_times)
    total_count = len(all_gaps)
    
    print(f"{'Overall':<20} | {total_count:<6} | {overall_avg_gap:12.4f} | {overall_avg_time:12.4f}")
    print("-" * 65)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze CVRP Log Filtered")
    parser.add_argument('--file', type=str, 
                        default='/public/home/bayp/exp_survey_202509/Construction/single-stage/appending/2_LEHD/CVRP/result_survey/20251029_191348_test__vrp0/log.txt',
                        help='Path to the log file')
    args = parser.parse_args()
    
    analyze_log(args.file)
