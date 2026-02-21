import re
import os

def parse_log(file_path):
    results = []
    current_instance = {}
    
    with open(file_path, 'r') as f:
        for line in f:
            # Match instance name and problem size
            # [2025-10-12 00:53:27] TSPTester_LIB_Survey.py(173) : Instance name: pr144, problem_size: 144
            size_match = re.search(r'Instance name: .+, problem_size: (\d+)', line)
            if size_match:
                # If we have a previous instance, save it if it has a gap
                if current_instance and 'gap' in current_instance:
                    results.append(current_instance)
                
                current_instance = {
                    'size': int(size_match.group(1))
                }
                continue
            
            # Match Aug gap (preferred)
            # [2025-10-12 00:53:28] TSPTester_LIB_Survey.py(241) : Aug score:63073.000, Aug gap:7.749%
            gap_match = re.search(r'Aug gap:\s*(-?[\d\.]+)', line)
            if gap_match and current_instance:
                try:
                    gap_val = float(gap_match.group(1))
                    current_instance['gap'] = gap_val
                except ValueError:
                    pass

        # Append the last one
        if current_instance and 'gap' in current_instance:
            results.append(current_instance)
            
    return results

def get_bucket(size):
    if size < 1000:
        return "[0, 1000)"
    elif 1000 <= size < 10000:
        return "[1000, 10000)"
    elif 10000 <= size <= 100000:
        return "[10000, 100000]"
    else:
        return "Other"

def main():
    # Use the directory of the script to find run_log
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_file = os.path.join(script_dir, 'run_log')
    
    if not os.path.exists(log_file):
        print(f"Error: {log_file} not found.")
        return

    data = parse_log(log_file)
    
    if not data:
        print("No data found in log file.")
        return

    filtered_data = [d for d in data if d['gap'] <= 100.0]
    excluded_count = len(data) - len(filtered_data)
    
    print(f"Total instances found: {len(data)}")
    print(f"Instances with Gap > 100% (excluded): {excluded_count}")
    print(f"Instances remaining: {len(filtered_data)}")
    print("-" * 30)

    # Bucketing
    buckets = {
        "[0, 1000)": [],
        "[1000, 10000)": [],
        "[10000, 100000]": [],
        "Other": []
    }
    
    # Sort order for display
    bucket_order = ["[0, 1000)", "[1000, 10000)", "[10000, 100000]", "Other"]

    overall_gap_sum = 0
    overall_count = 0

    for item in filtered_data:
        size = item['size']
        gap = item['gap']
        
        b = get_bucket(size)
        if b in buckets:
             buckets[b].append(gap)
        else:
             buckets["Other"].append(gap)
        
        overall_gap_sum += gap
        overall_count += 1

    print(f"{'Bucket':<20} | {'Count':<10} | {'Avg Gap (%)':<15}")
    print("-" * 51)
    
    for b_name in bucket_order:
        gaps = buckets[b_name]
        count = len(gaps)
        avg_gap = sum(gaps) / count if count > 0 else 0.0
        # Only print non-empty buckets or all if requested? 
        # Usually good to see all main buckets even if empty to confirm coverage.
        if b_name == "Other" and count == 0:
            continue
        print(f"{b_name:<20} | {count:<10} | {avg_gap:.4f}")

    print("-" * 51)
    overall_avg = overall_gap_sum / overall_count if overall_count > 0 else 0.0
    print(f"{'Overall':<20} | {overall_count:<10} | {overall_avg:.4f}")

if __name__ == "__main__":
    main()
