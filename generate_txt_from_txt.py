import random

def sample_lines(input_file, output_file, sample_size=1000):
    # 读取原始文件的所有行
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 如果文件行数不足1000行，则使用所有行
    if len(lines) <= sample_size:
        sampled_lines = lines
    else:
        # 随机选取1000行
        sampled_lines = random.sample(lines, sample_size)
    
    # 将选取的行写入新文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(sampled_lines)

# 使用示例
input_filename = '/data/lgl/datasets/SYSU-CD/test_list.txt'  # 原始文件名
output_filename = '/data/lgl/datasets/SYSU-CD/test_list1000.txt'  # 输出文件名
sample_lines(input_filename, output_filename)