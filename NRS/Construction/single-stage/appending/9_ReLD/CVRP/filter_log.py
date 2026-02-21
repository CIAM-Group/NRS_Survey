import os
import re

# Log file path relative to CVRP directory
LOG_FILE_REL_PATH = "./CVRP/Results_survey/results_survey_cvrp_20251012_183026_54/vrplib_test.log"

# Sizes for known large instances where size isn't in the name in standard format
SIZES = {
    'Antwerp1': 6000, 'Antwerp2': 6000,
    'Brussels1': 15000, 'Brussels2': 15000,
    'Flanders1': 30000, 'Flanders2': 30000,
    'Ghent1': 10000, 'Ghent2': 10000,
    'Leuven1': 3000, 'Leuven2': 3000,
}

def get_size(name):
    # Check manual map first
    if name in SIZES:
        return SIZES[name]
    
    # Handle X-nXXX-kYY format (e.g., X-n101-k25 -> size 100)
    # Extract number after 'n'
    match = re.search(r'-n(\d+)-', name)
    if match:
        return int(match.group(1)) - 1
    
    # Handle just nXXX if format differs
    match = re.search(r'n(\d+)', name)
    if match:
        return int(match.group(1)) - 1

    print(f"Warning: Unknown size for {name}, assuming 100 (Bucket 0-1k).")
    return 100

def main():
    if not os.path.exists(LOG_FILE_REL_PATH):
        print(f"Error: Log file not found at {LOG_FILE_REL_PATH}")
        print("Please check the path or run this script from the CVRP/ directory.")
        return

    data = []
    
    print(f"Reading log file: {LOG_FILE_REL_PATH}")
    print("-" * 60)
    
    with open(LOG_FILE_REL_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            # Regex to match lines like:
            # 2025-10-12 18:30:35,573 - Instance Antwerp1: Time 8.5447s, Cost 513191.0, Gap 7.52%
            match = re.search(r'Instance\s+([\w\.-]+):\s+Time\s+([\d\.]+)s,\s+Cost\s+[\d\.]+,\s+Gap\s+([\d\.]+)%', line)
            
            if match:
                name = match.group(1)
                time_val = float(match.group(2))
                gap_val = float(match.group(3))
                
                size = get_size(name)
                
                # Filter gap > 100%
                if gap_val > 100.0:
                    print(f"Skipping {name}: Gap {gap_val:.2f}% > 100%")
                    continue
                
                data.append({
                    'gap': gap_val,
                    'time': time_val,
                    'size': size
                })

    # Buckets
    small = [d for d in data if d['size'] < 1000]
    medium = [d for d in data if 1000 <= d['size'] < 10000]
    large = [d for d in data if d['size'] >= 10000]
    total = data
    
    def print_stats(label, subset):
        if not subset:
            print(f"Average gap {label}: nan% (count=0)")
            print(f"Average time {label}: nan (count=0)")
        else:
            avg_gap = sum(d['gap'] for d in subset) / len(subset)
            avg_time = sum(d['time'] for d in subset) / len(subset)
            print(f"Average gap {label}: {avg_gap:.2f}% (count={len(subset)})")
            print(f"Average time {label}: {avg_time:.2f}s (count={len(subset)})")
        print("-" * 60)

    print("-" * 60)
    print("Filtered Statistics (Gap <= 100%)")
    print("-" * 60)
    print_stats("[0, 1k)", small)
    print_stats("[1k, 10k)", medium)
    print_stats("[10k, +inf)", large)
    print_stats("total", total)

if __name__ == "__main__":
    main()
