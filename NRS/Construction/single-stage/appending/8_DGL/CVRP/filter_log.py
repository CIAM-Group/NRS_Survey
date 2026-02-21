import re
import statistics

def parse_log_file(file_path):
    results = []
    # Pattern to match lines like:
    # [2025-10-13 01:05:24] test_cvrplib_survey.py(334) : Instance: X-n101-k25: Size: 100, Score: 30541.0, Gap: 10.692%, Time: 1.73s
    pattern = re.compile(r"Instance: .*?: Size: (\d+), Score: .*, Gap: ([\d\.]+)%, Time: ([\d\.]+)s")

    with open(file_path, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                size = int(match.group(1))
                gap = float(match.group(2))
                time_val = float(match.group(3))
                
                # Filter out gaps > 100%
                if gap > 100:
                    continue
                    
                results.append({'size': size, 'gap': gap, 'time': time_val})
    return results

def get_bucket_stats(results):
    # original buckets logic
    # [0, 1000)
    # [1000, 10000)
    # [10000, 100000]
    
    buckets = [
        (0, 1000),
        (1000, 10000),
        (10000, 100001) # Upper bound exclusive in logic, so 100001 to include 100000
    ]
    
    bucket_data = {b: {'gaps': [], 'times': []} for b in buckets}
    
    for r in results:
        size = r['size']
        gap = r['gap']
        time_val = r['time']
        
        placed = False
        for low, high in buckets:
            if low <= size < high:
                 bucket_data[(low, high)]['gaps'].append(gap)
                 bucket_data[(low, high)]['times'].append(time_val)
                 placed = True
                 break

    stats = {}
    total_gaps = []
    total_times = []
    
    for bucket, data in bucket_data.items():
        gaps = data['gaps']
        times = data['times']
        if gaps:
            avg_gap = statistics.mean(gaps)
            avg_time = statistics.mean(times)
            count = len(gaps)
            stats[bucket] = {'avg_gap': avg_gap, 'avg_time': avg_time, 'count': count}
            total_gaps.extend(gaps)
            total_times.extend(times)
        else:
            stats[bucket] = {'avg_gap': 0.0, 'avg_time': 0.0, 'count': 0}

    if total_gaps:
        overall_avg_gap = statistics.mean(total_gaps)
        overall_avg_time = statistics.mean(total_times)
        overall_count = len(total_gaps)
    else:
        overall_avg_gap = 0.0
        overall_avg_time = 0.0
        overall_count = 0
        
    return stats, overall_avg_gap, overall_avg_time, overall_count

def main():
    log_file = "/public/home/bayp/exp_survey_202509/Construction/single-stage/appending/8_DGL/CVRP/result_survey_cvrp/20251013_010522_test_TSPLIB_Survey/run_log.txt"
    
    try:
        results = parse_log_file(log_file)
        stats, overall_avg_gap, overall_avg_time, overall_count = get_bucket_stats(results)
        
        print(f"{'Bucket':<20} | {'Count':<10} | {'Avg Gap (%)':<15} | {'Avg Time (s)':<15}")
        print("-" * 70)
        
        sorted_buckets = sorted(stats.keys(), key=lambda x: x[0])
        
        for bucket in sorted_buckets:
            low, high = bucket
            label = f"[{low}, {high})"
            data = stats[bucket]
            print(f"{label:<20} | {data['count']:<10} | {data['avg_gap']:.4f}{'':<9} | {data['avg_time']:.4f}")
            
        print("-" * 70)
        print(f"{'Overall':<20} | {overall_count:<10} | {overall_avg_gap:.4f}{'':<9} | {overall_avg_time:.4f}")

    except FileNotFoundError:
        print(f"Error: Log file not found at {log_file}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
