import re
import os

def analyze_log(log_path):
    if not os.path.exists(log_path):
        print(f"Log file not found: {log_path}")
        return

    # Define buckets
    buckets = {
        "[0,1000)": {"range": (0, 1000), "gaps": [], "times": []},
        "[1000,10000)": {"range": (1000, 10000), "gaps": [], "times": []},
        "[10000,100000)": {"range": (10000, 100000), "gaps": [], "times": []},
        ">=100000": {"range": (100000, float('inf')), "gaps": [], "times": []}
    }
    
    # Store all valid instances for total stats
    all_valid_gaps = []
    all_valid_times = []

    current_size = None
    current_gap = None
    
    # Regex patterns
    re_ckpt = re.compile(r"ckpt_path:\s*(.*)")
    re_size = re.compile(r"problem_size:\s*(\d+)")
    re_gap = re.compile(r"No aug gap:\s*([0-9.-]+)%")
    re_time = re.compile(r"Instance time \(incl\. dist\+test\):\s*([0-9.]+)s")

    model_path = "Unknown"

    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            if model_path == "Unknown":
                m_ckpt = re_ckpt.search(line)
                if m_ckpt:
                    model_path = m_ckpt.group(1).strip()

            if "Error occurred" in line:
                # Reset current instance parsing if an error occurs (though logs seem to show error blocks separately)
                current_size = None
                current_gap = None
                continue

            # Check for size
            m_size = re_size.search(line)
            if m_size:
                current_size = int(m_size.group(1))
                current_gap = None # Reset gap for new instance
                continue

            # Check for gap
            m_gap = re_gap.search(line)
            if m_gap:
                current_gap = float(m_gap.group(1))
                continue

            # Check for time
            m_time = re_time.search(line)
            if m_time and current_size is not None and current_gap is not None:
                current_time = float(m_time.group(1))
                
                # Filter: gap < 100%
                if current_gap < 100.0:
                    # Find bucket
                    matched_bucket = None
                    for b_name, b_data in buckets.items():
                        low, high = b_data["range"]
                        if low <= current_size < high:
                            matched_bucket = b_name
                            break
                    
                    if matched_bucket:
                        buckets[matched_bucket]["gaps"].append(current_gap)
                        buckets[matched_bucket]["times"].append(current_time)
                    
                    all_valid_gaps.append(current_gap)
                    all_valid_times.append(current_time)
                
                # Reset for next instance
                current_size = None
                current_gap = None

    # Print results
    print(f"Model Path: {model_path}")
    print(f"{'Bucket':<20} | {'Count':<6} | {'Avg Gap (%)':<12} | {'Avg Time (s)':<12}")
    print("-" * 60)
    
    for b_name, b_data in buckets.items():
        count = len(b_data["gaps"])
        if count > 0:
            avg_gap = sum(b_data["gaps"]) / count
            avg_time = sum(b_data["times"]) / count
            print(f"{b_name:<20} | {count:<6} | {avg_gap:<12.4f} | {avg_time:<12.4f}")
        else:
            print(f"{b_name:<20} | {0:<6} | {'N/A':<12} | {'N/A':<12}")

    print("-" * 60)
    total_count = len(all_valid_gaps)
    if total_count > 0:
        total_avg_gap = sum(all_valid_gaps) / total_count
        total_avg_time = sum(all_valid_times) / total_count
        print(f"{'Total':<20} | {total_count:<6} | {total_avg_gap:<12.4f} | {total_avg_time:<12.4f}")
    else:
        print(f"{'Total':<20} | {0:<6} | {'N/A':<12} | {'N/A':<12}")

if __name__ == "__main__":
    # log_file_path = "./result_survey_tsp/20251226_222536_no_aug_test_TSPLIB_Survey_model_TSP50/run_log.txt"
    # log_file_path = "./result_survey_tsp/20251226_223434_no_aug_test_TSPLIB_Survey_model_TSP100/run_log.txt"
    # log_file_path = "./result_survey_tsp/20251226_223833_no_aug_test_TSPLIB_Survey_model_TSP500/run_log.txt"
    # log_file_path = "./result_survey_tsp/20251226_224727_no_aug_test_TSPLIB_Survey_model_TSP1K/run_log.txt"
    log_file_path = "./result_survey_tsp/20251226_225837_no_aug_test_TSPLIB_Survey_model_TSP10K/run_log.txt"

    analyze_log(log_file_path)
