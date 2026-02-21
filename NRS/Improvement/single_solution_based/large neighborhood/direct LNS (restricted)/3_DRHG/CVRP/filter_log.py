import re
import argparse
import sys

def parse_log(log_path):
    buckets = [
        {'range': (0, 1000), 'data': []},
        {'range': (1000, 10000), 'data': []},
        {'range': (10000, 100001), 'data': []}
    ]
    
    # Regex pattern
    # [2025-10-29 18:32:35] VRPTester_cvrplib.py(585) : Instance X-n120-k6 (n=119) | optimal=13332 | score=13460 | gap=0.960% | time=455.006s
    line_pattern = re.compile(r'Instance\s+.*?\s*\(n=(\d+)\)\s+\|.*?gap=([-\d.]+)%\s+\|\s+time=([-\d.]+)s')
    
    try:
        with open(log_path, 'r') as f:
            for line in f:
                match = line_pattern.search(line)
                if match:
                    n = int(match.group(1))
                    gap = float(match.group(2))
                    time_val = float(match.group(3))
                    
                    # Filter gap > 100%
                    if gap > 100.0:
                        continue
                    
                    # Add to appropriate bucket
                    match_bucket = False
                    for bucket in buckets:
                        start, end = bucket['range']
                        if start <= n < end:
                            bucket['data'].append({'gap': gap, 'time': time_val})
                            match_bucket = True
                            break
                    
    except FileNotFoundError:
        print(f"Error: File not found at {log_path}")
        sys.exit(1)
    
    return buckets

def print_stats(buckets):
    all_data = []
    
    print(f"{'Range':<20} {'Count':<10} {'Avg Gap (%)':<15} {'Avg Time (s)':<15}")
    print("-" * 60)
    
    for bucket in buckets:
        start, end = bucket['range']
        range_str = f"[{start}, {end})"
        data = bucket['data']
        count = len(data)
        
        if count > 0:
            avg_gap = sum(d['gap'] for d in data) / count
            avg_time = sum(d['time'] for d in data) / count
            all_data.extend(data)
        else:
            avg_gap = 0.0
            avg_time = 0.0
            
        print(f"{range_str:<20} {count:<10} {avg_gap:<15.3f} {avg_time:<15.3f}")
        
    print("-" * 60)
    
    # Overall
    total_count = len(all_data)
    if total_count > 0:
        overall_avg_gap = sum(d['gap'] for d in all_data) / total_count
        overall_avg_time = sum(d['time'] for d in all_data) / total_count
    else:
        overall_avg_gap = 0.0
        overall_avg_time = 0.0
        
    print(f"{'Overall':<20} {total_count:<10} {overall_avg_gap:<15.3f} {overall_avg_time:<15.3f}")

if __name__ == "__main__":
    # Default log file path
    default_log_path = "/public/home/bayp/exp_survey_202509/Improvement/single_solution_based/large neighborhood/direct LNS (restricted)/3_DRHG/CVRP/result_survey/20251029_182500_test_vrplib_iter1000/log.txt"
    
    parser = argparse.ArgumentParser(description='Analyze CVRP log file stats with filtering.')
    parser.add_argument('log_path', nargs='?', default=default_log_path, help='Path to the log file')
    args = parser.parse_args()
    
    print(f"Analyzing log file: {args.log_path}")
    buckets = parse_log(args.log_path)
    print_stats(buckets)
