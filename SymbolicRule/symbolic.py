import json
import os
import jsonlines
import parserTool.parse as ps
# from c_cfg import C_CFG
from parserTool.utils import remove_comments_and_docstrings
from parserTool.parse import Lang
import re
# 指定 JSONL 文件的路径
jsonl_file_path = 'valid_cdata_results_V2without_here_match.jsonl'

# 打开 JSONL 文件并逐行读取
with open(jsonl_file_path, 'r') as jsonl_file:
    lines = jsonl_file.readlines()



# def move_comments_to_new_line(input_code):
#     # 使用正则表达式匹配同时包含注释和代码的行
#     pattern = r'(\s*)(.*?)(\s*//\s*(.*?))\n'
#     modified_code = re.sub(pattern, r'\1// \4\n\1\2\n', input_code)
#     modified_code = re.sub(r'^\s*?\n', '', modified_code, flags=re.MULTILINE)
#     return modified_code

def move_comments_to_new_line(code):
    lines = code.split('\n')
    new_lines = []

    for line in lines:
        # 使用正则表达式来查找代码和注释部分
        match = re.match(r'(\s*)(.*?)(\s*\/\/.*|\/\*.*\*\/)?$', line)
        if match:
            spaces, code_part, comment_part = match.groups()
            if comment_part: 
                new_lines.append(f'{spaces}{comment_part.strip()}')
            new_lines.append(f'{spaces}{code_part}')

    code = '\n'.join(new_lines)
    code = re.sub(r'^\s*?\n', '', code, flags=re.MULTILINE)
    return code


def print_ast_node(code, node, indent=""):
    cpp_loc = code.split('\n')
    # if node.start_point[0] == 74:
    # print(len(cpp_loc))
    # print(cpp_loc[13])
    # exit(0)
    # If Statement
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
                        
                        New_line = Begin + "(" + comment[1:] + ") "+ End
                        cpp_loc[node.start_point[0]] = New_line
                        code = "\n".join(cpp_loc)

        if node.start_point[0] > 1 and cpp_loc[node.start_point[0] - 2].strip().startswith("//") and cpp_loc[node.start_point[0] - 1].strip() == "}":
                # Extract the comment content
                comment_line = cpp_loc[node.start_point[0] - 2]
                comment = comment_line.strip().lstrip("//")

                # Remove the comment and the "{" line from code_lines
                for child in node.children:
                    # print(child.type)
                    if child.type == "parenthesized_expression" and child.start_point[0] == node.start_point[0]:
                        # del cpp_loc[child.start_point[0],child.start_point[1]:child.endpoint[1]]
                        Begin = cpp_loc[child.start_point[0]][:child.start_point[1]]
                        End = cpp_loc[child.start_point[0]][child.end_point[1]:]
                        
                        New_line = Begin + "(" + comment + ") "+ End
                        cpp_loc[node.start_point[0]] = New_line
                        code = "\n".join(cpp_loc)
                        
    # If Statement
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
                            
                            New_line = Begin + "(" + comment[1:] + ") "+ End
                            cpp_loc[node.start_point[0]] = New_line
                            code = "\n".join(cpp_loc)
                            break
                        next_sibling_index += 1

    if node.type == "for_statement":
        
        # print(cpp_loc[node.start_point[0] - 1])
        # print(cpp_loc[node.start_point[0] - 1].strip().startswith("//"))
        if node.start_point[0] > 0 and cpp_loc[node.start_point[0] - 1].strip().startswith("//"):
                comment_line = cpp_loc[node.start_point[0] - 1]
                
                # Extract the comment content
                comment = comment_line.strip().lstrip("//")  

                    # for child in node.children:
                    #     print(child.type)
                for child in node.children:
                    if child.type == "(" and child.start_point[0] == node.start_point[0]:
                        # del cpp_loc[child.start_point[0],child.start_point[1]:child.endpoint[1]]
                        Begin = cpp_loc[child.start_point[0]][:child.end_point[1]]
                        # print(cpp_loc[node.start_point[0]])
                    if child.type == ")" and child.start_point[0] == node.start_point[0]:
                        End = cpp_loc[child.start_point[0]][child.start_point[1]:]
                        
                        New_line = Begin + comment[1:] + End
                        cpp_loc[node.start_point[0]] = New_line
                        code = "\n".join(cpp_loc)

        if node.start_point[0] > 1 and cpp_loc[node.start_point[0] - 2].strip().startswith("//") and cpp_loc[node.start_point[0] - 1].strip() == "}":
                # Extract the comment content
                comment_line = cpp_loc[node.start_point[0] - 2]
                comment = comment_line.strip().lstrip("//")


                # Remove the comment and the "{" line from code_lines
                for child in node.children:
                    if child.type == "(" and child.start_point[0] == node.start_point[0]:
                        # del cpp_loc[child.start_point[0],child.start_point[1]:child.endpoint[1]]
                        Begin = cpp_loc[child.start_point[0]][:child.end_point[1]]
                        # print(cpp_loc[node.start_point[0]])
                    if child.type == ")" and child.start_point[0] == node.start_point[0]:
                        End = cpp_loc[child.start_point[0]][child.start_point[1]:]
                        
                        New_line = Begin + comment + End
                        cpp_loc[node.start_point[0]] = New_line
                        code = "\n".join(cpp_loc)

    if node.type == "switch_statement":

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
                        
                        New_line = Begin + " (" + comment[1:] + ") "+ End
                        cpp_loc[node.start_point[0]] = New_line
                        code = "\n".join(cpp_loc)

        if node.start_point[0] > 1 and cpp_loc[node.start_point[0] - 2].strip().startswith("//") and cpp_loc[node.start_point[0] - 1].strip() == "}":
                # Extract the comment content
                comment_line = cpp_loc[node.start_point[0] - 2]
                comment = comment_line.strip().lstrip("//")

                # Remove the comment and the "{" line from code_lines
                for child in node.children:
                    # print(child.type)
                    if child.type == "parenthesized_expression" and child.start_point[0] == node.start_point[0]:
                        # del cpp_loc[child.start_point[0],child.start_point[1]:child.endpoint[1]]
                        Begin = cpp_loc[child.start_point[0]][:child.start_point[1]]
                        End = cpp_loc[child.start_point[0]][child.end_point[1]:]
                        
                        New_line = Begin + "(" + comment + ") "+ End
                        cpp_loc[node.start_point[0]] = New_line
                        code = "\n".join(cpp_loc)
    
    if node.type == "while_statement":
        
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
                        
                        New_line = Begin + " (" + comment[1:] + ") "+ End
                        cpp_loc[node.start_point[0]] = New_line
                        code = "\n".join(cpp_loc)

        if node.start_point[0] > 1 and cpp_loc[node.start_point[0] - 2].strip().startswith("//") and cpp_loc[node.start_point[0] - 1].strip() == "}":
                # Extract the comment content
                comment_line = cpp_loc[node.start_point[0] - 2]
                comment = comment_line.strip().lstrip("//")

                # Remove the comment and the "{" line from code_lines
                for child in node.children:
                    # print(child.type)
                    if child.type == "parenthesized_expression" and child.start_point[0] == node.start_point[0]:
                        # del cpp_loc[child.start_point[0],child.start_point[1]:child.endpoint[1]]
                        Begin = cpp_loc[child.start_point[0]][:child.start_point[1]]
                        End = cpp_loc[child.start_point[0]][child.end_point[1]:]
                        
                        New_line = Begin + "(" + comment + ") "+ End
                        cpp_loc[node.start_point[0]] = New_line
                        code = "\n".join(cpp_loc)

    if node.type == "while":
        

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
                            
                            New_line = Begin + " (" + comment[1:] + ") "+ End
                            cpp_loc[node.start_point[0]] = New_line
                            code = "\n".join(cpp_loc)
                            break
                        next_sibling_index += 1

    if node.type == "else":
        
        # print(cpp_loc[node.start_point[0] - 1])
        # print(cpp_loc[node.start_point[0] - 1].strip().startswith("//"))
        if node.start_point[0] > 0 and cpp_loc[node.start_point[0] - 1].strip().startswith("//"):
            # exit(0)
            comment_line = cpp_loc[node.start_point[0] - 1]
            
            # Extract the comment content
            comment = comment_line.strip().lstrip("//")  
            parent = node.parent
            if parent is not None:
                next_sibling_index = parent.children.index(node) + 1
                else_sign = 0
                while next_sibling_index < len(parent.children):
                    next_sibling = parent.children[next_sibling_index]
                    # print(next_sibling)
                                    
                    if next_sibling.type == "compound_statement" and next_sibling.start_point[0] == node.start_point[0]:
                        # del cpp_loc[child.start_point[0],child.start_point[1]:child.endpoint[1]]
                        Begin = cpp_loc[next_sibling.start_point[0]][:next_sibling.start_point[1]]
                        End = cpp_loc[next_sibling.start_point[0]][next_sibling.start_point[1]+1:]
                        
                        New_line = Begin + " (" + comment[1:] + ") "+ End
                        cpp_loc[node.start_point[0]] = New_line
                        code = "\n".join(cpp_loc)
                        else_sign = 1
                        break
                    next_sibling_index += 1
                if else_sign ==0:
                    Begin = cpp_loc[node.start_point[0]][:node.end_point[1]]
                    End = cpp_loc[node.start_point[0]][node.end_point[1]:]
                    New_line = Begin + " (" + comment[1:] + ") "+ End
                    cpp_loc[node.start_point[0]] = New_line
                    code = "\n".join(cpp_loc)

        elif node.start_point[0] > 1 and cpp_loc[node.start_point[0] - 2].strip().startswith("//") and cpp_loc[node.start_point[0] - 1].strip() == "}":
                # Extract the comment content
                comment_line = cpp_loc[node.start_point[0] - 2]
                comment = comment_line.strip().lstrip("//")


                parent = node.parent
                if parent is not None:
                    next_sibling_index = parent.children.index(node) + 1
                    while next_sibling_index < len(parent.children):
                        next_sibling = parent.children[next_sibling_index]
                        # print(next_sibling)
                                       
                        if next_sibling.type == "compound_statement" and next_sibling.start_point[0] == node.start_point[0]:
                            # del cpp_loc[child.start_point[0],child.start_point[1]:child.endpoint[1]]
                            Begin = cpp_loc[next_sibling.start_point[0]][:next_sibling.start_point[1]]
                            End = cpp_loc[next_sibling.start_point[0]][next_sibling.start_point[1]+1:]
                            
                            New_line = Begin + "(" + comment + ") "+ End
                            cpp_loc[node.start_point[0]] = New_line
                            code = "\n".join(cpp_loc)
                            break
                        next_sibling_index += 1

    if node.type == "case":
        

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
                                       
                        if next_sibling.type == ":" and next_sibling.start_point[0] == node.start_point[0]:
                            # del cpp_loc[child.start_point[0],child.start_point[1]:child.endpoint[1]]
                            Begin = cpp_loc[next_sibling.start_point[0]][:next_sibling.start_point[1]]
                            End = cpp_loc[next_sibling.start_point[0]][next_sibling.start_point[1]:]
                            
                            New_line = Begin + " (" + comment[1:] + ")"+ End
                            cpp_loc[node.start_point[0]] = New_line
                            code = "\n".join(cpp_loc)
                            break
                        next_sibling_index += 1

    if node.type == "break_statement":
        if node.start_point[0] > 0 and cpp_loc[node.start_point[0] - 1].strip().startswith("//"):
                comment_line = cpp_loc[node.start_point[0] - 1]
                # Extract the comment content
                comment = comment_line.strip().lstrip("//")  

                    # for child in node.children:
                    #     print(child.type)
                for child in node.children:
                    # print(child.type)
                    if child.type == ";" and child.start_point[0] == node.start_point[0]:
                        # del cpp_loc[child.start_point[0],child.start_point[1]:child.endpoint[1]]
                        Begin = cpp_loc[child.start_point[0]][:child.start_point[1]]
                        End = cpp_loc[child.start_point[0]][child.start_point[1]:]
                        
                        New_line = Begin + " (" + comment[1:] + ") "+ End
                        cpp_loc[node.start_point[0]] = New_line
                        code = "\n".join(cpp_loc)

    if node.type == "return_statement":
        if node.start_point[0] > 0 and cpp_loc[node.start_point[0] - 1].strip().startswith("//"):
                comment_line = cpp_loc[node.start_point[0] - 1]
                # Extract the comment content
                comment = comment_line.strip().lstrip("//")  

                    # for child in node.children:
                    #     print(child.type)
                for child in node.children:
                    # print(child.type)
                    if child.type == ";" and child.start_point[0] == node.start_point[0]:
                        # del cpp_loc[child.start_point[0],child.start_point[1]:child.endpoint[1]]
                        Begin = cpp_loc[child.start_point[0]][:child.start_point[1]]
                        End = cpp_loc[child.start_point[0]][child.start_point[1]:]
                        
                        New_line = Begin + " (" + comment[1:] + ") "+ End
                        cpp_loc[node.start_point[0]] = New_line
                        code = "\n".join(cpp_loc)

    if node.type == "goto_statement":
        if node.start_point[0] > 0 and cpp_loc[node.start_point[0] - 1].strip().startswith("//"):
                comment_line = cpp_loc[node.start_point[0] - 1]
                # Extract the comment content
                comment = comment_line.strip().lstrip("//")  

                    # for child in node.children:
                    #     print(child.type)
                for child in node.children:
                    # print(child.type)
                    if child.type == ";" and child.start_point[0] == node.start_point[0]:
                        # del cpp_loc[child.start_point[0],child.start_point[1]:child.endpoint[1]]
                        Begin = cpp_loc[child.start_point[0]][:child.start_point[1]]
                        End = cpp_loc[child.start_point[0]][child.start_point[1]:]
                        
                        New_line = Begin + " (" + comment[1:] + ") "+ End
                        cpp_loc[node.start_point[0]] = New_line
                        code = "\n".join(cpp_loc)

    if node.type == "continue_statement":
        if node.start_point[0] > 0 and cpp_loc[node.start_point[0] - 1].strip().startswith("//"):
                comment_line = cpp_loc[node.start_point[0] - 1]
                # Extract the comment content
                comment = comment_line.strip().lstrip("//")  

                    # for child in node.children:
                    #     print(child.type)
                for child in node.children:
                    # print(child.type)
                    if child.type == ";" and child.start_point[0] == node.start_point[0]:
                        # del cpp_loc[child.start_point[0],child.start_point[1]:child.endpoint[1]]
                        Begin = cpp_loc[child.start_point[0]][:child.start_point[1]]
                        End = cpp_loc[child.start_point[0]][child.start_point[1]:]
                        
                        New_line = Begin + " (" + comment[1:] + ") "+ End
                        cpp_loc[node.start_point[0]] = New_line
                        code = "\n".join(cpp_loc)
             
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




# 从前五行中解析 JSON 数据
for line_num, line in enumerate(lines[:], start=1):
    print(line_num)

    # try:
    data = json.loads(line)
    # print(data)
    result = {}
    result['target'] = data['target']
    result['choices'] = data['choices']
    result['func'] = data['func']
    result['idx'] = data['idx']
    data = data['choices']
    # print(data)
    
    
    data = move_comments_to_new_line(data)
    # print(data)
    # exit(0)
    code_ast = ps.tree_sitter_ast(data, Lang.C)
    # print(data)
    
    # 打印 AST 根节点
    # print("AST Root:")
    
    updated_code = print_ast_node(data, code_ast.root_node)
    # print(updated_code)
    cleaned_code = remove_comments(updated_code)
    # print(cleaned_code)
    result['clean_code'] = cleaned_code

    # print(result['choices'])

    with jsonlines.open(jsonl_file_path.split('.jsonl')[0]+'V3'+'.jsonl', mode='a') as f:
            f.write_all([result]) 
        # print(code_ast.root_node.serialize())
        # print(f"Line {line_num}: {data}")

    # except json.JSONDecodeError as e:
    # print(f"Line {line_num}: Error decoding JSON - {e}")



# 关闭文件
jsonl_file.close()
