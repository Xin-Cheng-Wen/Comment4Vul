import json
from random import choice

import jsonlines

# 打开文件a和文件b
jsonl_file_path = 'valid_cdata_results_V2without_here.jsonl'
with open(jsonl_file_path, 'r') as file_b, open('../remove/test_cdata.jsonl', 'r') as file_a:
    # 创建一个字典，用于存储文件b中的idx和choices
    b_data_dict = {}
    for line in file_b:
        data_b = json.loads(line)
        idx_b = data_b['idx']
        choices = data_b['choices']
        b_data_dict[idx_b] = choices

    # 逐行遍历文件a并检查idx是否存在于文件b中
    for line in file_a:
        data_a = json.loads(line)
        idx_a = data_a['idx']
        func_a = data_a['func']

        if idx_a in b_data_dict:
            choices_b = b_data_dict[idx_a]
            choice_a = choices_b
            # print(f"idx: {idx_a}, func_a: {func_a}, choices_b: {choices_b}")
        else:
            choice_a = data_a['func']
            # print(f"idx: {idx_a}, func_a: {func_a}, choices_b: (not found in file_b)")
        result = {}
        result['target'] = data_a['target']
        result['choices'] = choice_a
        result['func'] = data_a['func']
        result['idx'] = data_a['idx']

        with jsonlines.open(jsonl_file_path.split('.jsonl')[0]+'_match.jsonl', mode='a') as f:
                f.write_all([result]) 