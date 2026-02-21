import re
import sys

def parse_log_file(file_path):
    buckets = {}
    current_bucket = None
    all_results = []
    
    # Pre-define the buckets order based on the log file header info we saw
    # scale_range_all: [[0, 1000], [1000, 10000], [10000, 100001]]
    bucket_ranges = [ (0, 1000), (1000, 10000), (10000, 100001) ]
    for r in bucket_ranges:
        buckets[r] = {'count': 0, 'gaps': [], 'times': []}
        
    current_gap = None
    current_time = None
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Detect scale range
            # Test scale range: [0, 1000]
            match_scale = re.search(r'Test scale range: \[(\d+), (\d+)\]', line)
            if match_scale:
                start, end = int(match_scale.group(1)), int(match_scale.group(2))
                current_bucket = (start, end)
                # Ensure bucket exists
                if current_bucket not in buckets:
                    buckets[current_bucket] = {'count': 0, 'gaps': [], 'times': []}
                continue

            # Detect gap
            # score:27612.000, gap:0.033%
            match_gap = re.search(r'gap:\s*([0-9.]+)%', line)
            if match_gap:
                current_gap = float(match_gap.group(1))
                continue

            # Detect time
            # Instance time: 0.684s
            match_time = re.search(r'Instance time:\s*([0-9.]+)s', line)
            if match_time and current_gap is not None and current_bucket is not None:
                current_time = float(match_time.group(1))
                
                # Check filter condition
                if current_gap <= 100.0:
                    buckets[current_bucket]['gaps'].append(current_gap)
                    buckets[current_bucket]['times'].append(current_time)
                    buckets[current_bucket]['count'] += 1
                
                # Reset for next instance
                current_gap = None
                current_time = None

    return buckets

def print_stats(buckets):
    print(f"{'Scale Range':<20} | {'Count':<10} | {'Avg Gap (%)':<15} | {'Avg Time (s)':<15}")
    print("-" * 70)
    
    total_gaps = []
    total_times = []
    
    # Sort buckets by start range
    sorted_keys = sorted(buckets.keys(), key=lambda x: x[0])
    
    for r in sorted_keys:
        data = buckets[r]
        count = len(data['gaps'])
        if count > 0:
            avg_gap = sum(data['gaps']) / count
            avg_time = sum(data['times']) / count
            print(f"[{r[0]}, {r[1]}]".ljust(20) + f" | {count:<10} | {avg_gap:<15.4f} | {avg_time:<15.4f}")
            
            total_gaps.extend(data['gaps'])
            total_times.extend(data['times'])
        else:
            print(f"[{r[0]}, {r[1]}]".ljust(20) + f" | {0:<10} | {'N/A':<15} | {'N/A':<15}")

    print("-" * 70)
    
    total_count = len(total_gaps)
    if total_count > 0:
        total_avg_gap = sum(total_gaps) / total_count
        total_avg_time = sum(total_times) / total_count
        print(f"{'Total':<20} | {total_count:<10} | {total_avg_gap:<15.4f} | {total_avg_time:<15.4f}")
    else:
        print(f"{'Total':<20} | {0:<10} | {'N/A':<15} | {'N/A':<15}")

if __name__ == "__main__":
    log_file = "survey_tsplib_survey.log"
    buckets = parse_log_file(log_file)
    print_stats(buckets)
