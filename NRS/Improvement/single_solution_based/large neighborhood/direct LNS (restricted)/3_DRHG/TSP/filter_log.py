import re
import argparse
import sys

def parse_log(log_path):
    buckets = [
        {'range': (0, 1000), 'data': []},
        {'range': (1000, 10000), 'data': []},
        {'range': (10000, 100001), 'data': []}
    ]
    
    current_size = None
    
    # Regex patterns
    # [2025-11-07 00:36:48] TSPTester_tsplib.py(348) : problem size: 144
    size_pattern = re.compile(r'problem size:\s+(\d+)')
    
    # [2025-11-07 00:42:58] TSPTester_tsplib.py(241) : [pr144] score=58537.000, opt=58537.000, gap=0.000%, time=370.651s
    result_pattern = re.compile(r'gap=([-\d.]+)%,\s+time=([-\d.]+)s')
    
    try:
        with open(log_path, 'r') as f:
            for line in f:
                size_match = size_pattern.search(line)
                if size_match:
                    current_size = int(size_match.group(1))
                    # Don't continue here, because in some formats checking both patterns might be safer 
                    # but here the log structure seems sequential.
                    # However, strictly speaking, size is updated first, then result comes.
                    continue
                    
                result_match = result_pattern.search(line)
                if result_match and current_size is not None:
                    gap = float(result_match.group(1))
                    time_val = float(result_match.group(2))
                    
                    # Filter gap > 100%
                    if gap > 100.0:
                        continue
                    
                    # Add to appropriate bucket
                    match_bucket = False
                    for bucket in buckets:
                        start, end = bucket['range']
                        if start <= current_size < end:
                            bucket['data'].append({'gap': gap, 'time': time_val})
                            match_bucket = True
                            break
                    
                    if not match_bucket:
                        # Maybe log if size outside ranges
                        pass
    except FileNotFoundError:
        print(f"Error: File not found at {log_path}")
        sys.exit(1)
    
    return buckets

def print_stats(buckets):
    all_data = []
    
    # Header
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
    # Default log file path from context
    default_log_path = "/public/home/bayp/exp_survey_202509/Improvement/single_solution_based/large neighborhood/direct LNS (restricted)/3_DRHG/TSP/result_survey_no_finetune/20251107_003647_test_tsp_tsplib_iter1000/log.txt"
    
    parser = argparse.ArgumentParser(description='Analyze TSP log file stats with filtering.')
    parser.add_argument('log_path', nargs='?', default=default_log_path, help='Path to the log file')
    args = parser.parse_args()
    
    print(f"Analyzing log file: {args.log_path}")
    buckets = parse_log(args.log_path)
    print_stats(buckets)
