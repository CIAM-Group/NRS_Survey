import re

# log_path = 'result_survey_tsp/20251207_161255_GenSCO_TSPLIB_Survey_model_TSP100/run_log.txt'
# log_path = 'result_survey_tsp/20251207_161445_GenSCO_TSPLIB_Survey_model_TSP500/run_log.txt'
log_path = 'result_survey_tsp/20251207_161605_GenSCO_TSPLIB_Survey_model_TSP1K/run_log.txt'


def parse_and_caclulate(file_path):
    # 存储结果：key为scale_range字符串，value为包含 {'gap': float, 'time': float} 的列表
    data = {}
    current_scale = None
    
    # 正则表达式匹配
    # 匹配 Scale Header, 例如: #################  Test scale range: (0, 1000)  #################
    scale_pattern = re.compile(r"Test scale range: (\(.+?\))")
    # 匹配结果行, 例如: ... Pred cost: 43569.0000, Gap: 57.842%, Time: 17.520s
    result_pattern = re.compile(r"Gap:\s+([\d\.]+)\%,\s+Time:\s+([\d\.]+)s")

    try:
        with open(file_path, 'r') as f:
            for line in f:
                # 检查是否是新的 Scale 块
                scale_match = scale_pattern.search(line)
                if scale_match:
                    current_scale = scale_match.group(1)
                    if current_scale not in data:
                        data[current_scale] = []
                    continue

                # 检查是否是结果行
                result_match = result_pattern.search(line)
                if result_match and current_scale:
                    gap = float(result_match.group(1))
                    time_val = float(result_match.group(2))
                    
                    # 可以在这里记录所有数据，后续再过滤
                    data[current_scale].append({'gap': gap, 'time': time_val})

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return

    print('logpath:', log_path)
    # 计算统计信息
    print(f"{'Scale Range':<20} | {'Total':<6} | {'Excl.':<6} | {'Valid':<6} | {'Avg Gap (%)':<15} | {'Avg Time (s)':<15}")
    print("-" * 80)
    
    all_valid_instances = []
    total_instances_count = 0

    for scale, instances in data.items():
        # 过滤 Gap > 100%
        valid_instances = [i for i in instances if i['gap'] <= 100.0]
        excluded_count = len(instances) - len(valid_instances)
        
        # 收集 valid instances 用于 overall 计算
        all_valid_instances.extend(valid_instances)
        total_instances_count += len(instances)
        
        if valid_instances:
            avg_gap = sum(i['gap'] for i in valid_instances) / len(valid_instances)
            avg_time = sum(i['time'] for i in valid_instances) / len(valid_instances)
            print(f"{scale:<20} | {len(instances):<6} | {excluded_count:<6} | {len(valid_instances):<6} | {avg_gap:<15.3f} | {avg_time:<15.3f}")
        else:
            print(f"{scale:<20} | {len(instances):<6} | {excluded_count:<6} | 0      | N/A             | N/A")

    # Overall 统计
    print("-" * 80)
    overall_valid_count = len(all_valid_instances)
    overall_excluded_count = total_instances_count - overall_valid_count
    
    if overall_valid_count > 0:
        overall_avg_gap = sum(i['gap'] for i in all_valid_instances) / overall_valid_count
        overall_avg_time = sum(i['time'] for i in all_valid_instances) / overall_valid_count
        print(f"{'Overall':<20} | {total_instances_count:<6} | {overall_excluded_count:<6} | {overall_valid_count:<6} | {overall_avg_gap:<15.3f} | {overall_avg_time:<15.3f}")
    else:
        print(f"{'Overall':<20} | {total_instances_count:<6} | {overall_excluded_count:<6} | 0      | N/A             | N/A")

if __name__ == "__main__":
    parse_and_caclulate(log_path)