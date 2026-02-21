import re
import statistics
import os

def analyze_log(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    # Regex to extract dim and gap
    # Pattern matches lines like: ... Instance X-n101-k25, dim=100, length=28021, optimal=27591.0, gap=1.558%, ...
    pattern = re.compile(r"dim=(\d+).*?gap=([0-9.]+)%")
    
    ranges = {
        "100-299": [],
        "300-699": [],
        "700-1001": []
    }
    
    with open(file_path, 'r') as f:
        for line in f:
            # We only care about lines with results
            if "Instance" in line and "gap=" in line:
                match = pattern.search(line)
                if match:
                    dim = int(match.group(1))
                    gap = float(match.group(2))
                    
                    if 100 <= dim <= 299:
                        ranges["100-299"].append(gap)
                    elif 300 <= dim <= 699:
                        ranges["300-699"].append(gap)
                    elif 700 <= dim <= 1001:
                        ranges["700-1001"].append(gap)

    print(f"Analysis for log file: {file_path}")
    print("-" * 50)
    print(f"{'Range':<15} | {'Count':<10} | {'Avg Gap (%)':<15}")
    print("-" * 50)
    
    for r_name in ["100-299", "300-699", "700-1001"]:
        gaps = ranges[r_name]
        if gaps:
            avg_gap = statistics.mean(gaps)
            print(f"{r_name:<15} | {len(gaps):<10} | {avg_gap:.3f}")
        else:
            print(f"{r_name:<15} | {0:<10} | {'N/A'}")
    print("-" * 50)

if __name__ == "__main__":
    # You can update this path to point to other log files if needed
    # log_file = "/public/home/bayp/exp_survey_202509/Improvement/population_based/GFACS/cvrp_nls/survey_results/cvrp_test_20251204_164342_test_CVRPLIB_Survey_model200.log"
    # log_file = "/public/home/bayp/exp_survey_202509/Improvement/population_based/GFACS/cvrp_nls/survey_results/cvrp_test_20251204_164359_test_CVRPLIB_Survey_model500.log"
    log_file = "/public/home/bayp/exp_survey_202509/Improvement/population_based/GFACS/cvrp_nls/survey_results/cvrp_test_20251204_164410_test_CVRPLIB_Survey_model1K.log"
    
    
    analyze_log(log_file)
