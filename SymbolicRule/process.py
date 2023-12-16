import json
import os
import jsonlines
import parserTool.parse as ps
# from c_cfg import C_CFG
from parserTool.utils import remove_comments_and_docstrings
from parserTool.parse import Lang
import re
# 指定 JSONL 文件的路径
jsonl_file_path = 'valid_cdata_results_V2.jsonl'

# 打开 JSONL 文件并逐行读取
with open(jsonl_file_path, 'r') as jsonl_file:
    lines = jsonl_file.readlines()


def print_ast_node(code, node, indent=""):
    cpp_loc = code.split('\n')

    if node.type == "if_statement":
        
        # print(cpp_loc[node.start_point[0] - 1])
        # print(cpp_loc[node.start_point[0] - 1].strip().startswith("//"))
        if node.start_point[0] > 0 and cpp_loc[node.start_point[0] - 1].strip().startswith("//"):
                comment_line = cpp_loc[node.start_point[0] - 1]
                # Extract the comment content
                comment = comment_line.strip().lstrip("//")  

                    # for child in node.children:
                    #     print(child.type)
                for child in node.children:
                    # print(child.type)
                    if child.type == "parenthesized_expression" and child.start_point[0] == node.start_point[0]:
                        # del cpp_loc[child.start_point[0],child.start_point[1]:child.endpoint[1]]
                        Begin = cpp_loc[child.start_point[0]][:child.start_point[1]]
                        End = cpp_loc[child.start_point[0]][child.end_point[1]:]
                        
                        New_line = Begin + "(" + comment + ") "+ End
                        cpp_loc[node.start_point[0]] = New_line
                        code = "\n".join(cpp_loc)

    if node.type == "if":
        

        if node.start_point[0] > 0 and cpp_loc[node.start_point[0] - 1].strip().startswith("//"):
                comment_line = cpp_loc[node.start_point[0] - 1]
                # Extract the comment content
                comment = comment_line.strip().lstrip("//")  
                parent = node.parent
                if parent is not None:
                    next_sibling_index = parent.children.index(node) + 1
                    while next_sibling_index < len(parent.children):
                        next_sibling = parent.children[next_sibling_index]
                        # print(next_sibling)
                                       
                        if next_sibling.type == "parenthesized_expression" and next_sibling.start_point[0] == node.start_point[0]:
                            # del cpp_loc[child.start_point[0],child.start_point[1]:child.endpoint[1]]
                            Begin = cpp_loc[next_sibling.start_point[0]][:next_sibling.start_point[1]]
                            End = cpp_loc[next_sibling.start_point[0]][next_sibling.end_point[1]:]
                            
                            New_line = Begin + "(" + comment + ") "+ End
                            cpp_loc[node.start_point[0]] = New_line
                            code = "\n".join(cpp_loc)
                            break
                        next_sibling_index += 1





    # print(f"{indent}Type: {node.type}")
    # print(f"{indent}Value: {node}")
    # print(f"{indent}Start: {node.start_point}")
    # print(f"{indent}End: {node.end_point}")





    

    for child in node.children:
        # print(f"{indent}Child:")
        code = print_ast_node(code, child, indent + "  ")

    return code






def remove_comments(code):
    # 使用正则表达式去除单行注释
    code = re.sub(r'//.*', '', code)

    # 使用正则表达式去除多行注释
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)

    # 使用正则表达式去除空行
    code = re.sub(r'^\s*?\n', '', code, flags=re.MULTILINE)

    return code

def remove_triple_quotes(text):
    if text.startswith("```") and text.endswith("```"):
        text = text[4:-3]
    elif text.startswith("```"):
        text = text[4:]
    return text


# 从前五行中解析 JSON 数据
number = 0
for line_num, line in enumerate(lines[:], start=1):
    # print(line_num)
    sign = 0
    try:
        data = json.loads(line)
        # print(data)
        result = {}
        
        choices = data['choices']

        if "Sure, here's the commented code:" in choices:
             sign = 1
             number+=1
             print(number)
        if "Here is the commented code:" in choices:
             sign = 1
             number+=1
             print(number)


        result['target'] = data['target']
        if sign == 1:
            result['choices'] = data['func']
        else:
            choices = remove_triple_quotes(choices)
            result['choices'] = choices
        
        result['func'] = data['func']
        result['idx'] = data['idx']
        # print(data)
        # code_ast = ps.tree_sitter_ast(data, Lang.C)
        # print(code_ast)
        # 打印 AST 根节点
        # print("AST Root:")
        # updated_code = print_ast_node(data, code_ast.root_node)
        # cleaned_code = remove_comments(updated_code)
        # print(cleaned_code)
        # result['choices'] = cleaned_code

        # print(result['choices'])
        # exit(0)

        with jsonlines.open(jsonl_file_path.split('.jsonl')[0]+'without_here.jsonl', mode='a') as f:
                f.write_all([result]) 
        # print(code_ast.root_node.serialize())
        # print(f"Line {line_num}: {data}")
    except json.JSONDecodeError as e:
        print(f"Line {line_num}: Error decoding JSON - {e}")

# 关闭文件
jsonl_file.close()
