import openai
import jsonlines
import json
import sys
from tqdm import tqdm
import traceback
from time import sleep
import tiktoken
api_keys = [
"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",
"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",
"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",
"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",
"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",
"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",
"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",
"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",
"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",
"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",


            ]
# api_keys = [
# "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",
# "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",
# "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",
# "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",
# "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",
# "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",
# "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",
# "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",
# "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",
# "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",
# "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",
# "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",
# "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",
# "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",
# "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",
# "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",
# "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",
# "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",
# "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",
# "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",
#
#             ]
number = 1
# int(sys.argv[1])
total = 1
# int(sys.argv[2])

# int(sys.argv[3])
filename = './test_cdata.jsonl'
    # sys.argv[1]

# openai.organization = "org-48syCGOYPz9FAlQqSwfWziWn"
querys = []
with jsonlines.open(filename) as reader:
    for obj in reader:
        querys.append(obj)
querys = querys[int((number-1)*(len(querys)/total)):int((number)*(len(querys)/total))]

results = []
fail = []
'''
numbers = []
with open('idx.txt', 'r') as file:
    lines = file.readlines()

    for line in lines:
        numbers.append(int(line.strip()))
#print(numbers)
'''
encoding = tiktoken.get_encoding("cl100k_base")
# token_count = 0
idxk = 21903
for pos in tqdm(range(len(querys))):
    api_idx = pos % 10
    openai.api_key = api_keys[api_idx]
    '''
    if pos not in numbers:
        continue
    '''
    query = querys[pos]
    idx = query['idx']
    # if idx <= idxk:
    #    continue
    # if idx >= idxk+200:
    #     break
    success = 0
    fail_count = 0

    prompt = 'You are an expert C/C++ programmer.'
    # prompt_1 = 'Please comment each line of source code:\\n          '
    prompt_1 = 'Please provide concise and precise comments for each line of source code to help the model better understand the source code. Please give the commented code directly\\n'
    # code = query['func'].replace('\n', '\\\\n')
    code = 'Source code: \\n' + query['func']
    count = len(encoding.encode(prompt_1 + code))
    if count > 3000:
        continue
    # print(pos)
    while success != 1:
        try:
        # 解析 API 响应以获取 token 数量


        #
        # token_count += count
        # if pos % 10 == 0:
        #     print("Pos:"+str(pos) +" Count" + str(token_count))

            response = openai.ChatCompletion.create(model="gpt-3.5-turbo-0301", messages=[
                {"role": "system", "content":prompt},
                {"role": "user", "content": prompt_1 + code}], temperature=0,max_tokens=1024,top_p=1,frequency_penalty=0.0,presence_penalty=0.0,stop=["\n\n"])
            success=1
            result = {}
            result['target'] = query['target']
            result['choices'] = response.get("choices")[0]["message"]["content"]
            result['func'] = query['func']
            # print(result['choices'])
            # print(result['func'])
            # exit(0)
            result['idx'] = query['idx']
            # int((number-1)*(len(querys))) + pos
            with jsonlines.open(filename.split('.jsonl')[0]+'_results_V2'+'.jsonl', mode='a') as f:
                f.write_all([result])
        except Exception as e:
            print("Error:" + str(pos))
            print(e)
            sleep(1)
            fail_count+=1

        if fail_count>50:
            fail.append(pos)
            exit(0)
            break

        # success = 1
    sleep(1)
print(fail)
