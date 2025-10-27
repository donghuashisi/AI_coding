import os
import sys
import re
import yaml
import subprocess
import copy
import pdb
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, END, START
from langgraph.types import Command
# from langchain_community.tools import (
#     ReadFileTool,
#     WriteFileTool,
#     DeleteFileTool,
# )s
from langchain.tools import tool
from langgraph.types import Command
from commonNode import *
from pathlib import Path
from coding_assistent_tool import *
from typing import TypedDict
from utils.utility import find_vtest_root

GET_ALL_DEPENDENCY = True

cur_path = Path(__file__).parent
with open(cur_path / "prompt" / "vtest_guide.md", 'r') as f:
    vtest_guide_promote = f.read()

with open(cur_path / "prompt" / "vtest_infra_readme.md", 'r') as f:
    vtest_infra_readme = f.read()

with open(cur_path / "prompt" / "most_matched_find.md", 'r') as f:
    most_matched_find = f.read()

with open(cur_path / "prompt" / "module_find.md", 'r') as f:
    module_find = f.read()

with open(cur_path / "prompt" / "keyword_design.md", 'r') as f:
    keyword_design = f.read()

with open(cur_path / "prompt" / "keyword_redsign.md", 'r') as f:
    keyword_redsign = f.read()

with open(cur_path / "prompt" / "vtest_coding_guide.md", 'r') as f:
    vtest_coding_guide = f.read()

with open(cur_path / "prompt" / "vtest_script_template.py", 'r') as f:
    vtest_script_template = f.read()


KEY_WORD_DESIGN_TRY_NUM_LIMIT = 3

class BASEAgentState(TypedDict):
    """定义RAGAgent的状态类型。
    包含一个字符串类型的字段`query`
    用于存储用户查询。
    """
    _llm: object
    llm: object
    tools: list
    agent_name: str
    messages: list
    tools_dict: dict
    user_data: dict
    ctx: object
    input_func: object
    info_func: object
    error: str
    ai_comments: dict
    current_step: str
    last_score: float
    logger: object
    user_ask: list
    find_res: list
    related_code: str
    key_word_design_try_num: int
    advertise_file_or_folder: list
    project_root_folder: str
    
@tool
def get_functions_classes_definitions(file_path: str, funcation_class_name:str) -> str:
    """
    Get the definition of a function or class in a file.
    Args:
        file_path: path to the file to be read.
        funcation_class_name: name of the function or class to be extracted.
    Returns:
        return the definition of the function or class.
    """ 
    if not os.path.exists(file_path):
        return f"File {file_path} does not exist."
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
        pattern_func = re.compile(r'^\s*def\s+' + re.escape(funcation_class_name) + r'\s*\(.*\)\s*:')
        pattern_class = re.compile(r'^\s*class\s+' + re.escape(funcation_class_name) + r'\s*(\(.*\))?\s*:')
        in_definition = False
        definition_lines = []
        indent_level = None
        for line in lines:
            if in_definition:
                current_indent = len(line) - len(line.lstrip())
                if line.strip() == '' or current_indent > indent_level:
                    definition_lines.append(line)
                else:
                    break
            elif pattern_func.match(line) or pattern_class.match(line):
                in_definition = True
                indent_level = len(line) - len(line.lstrip())
                definition_lines.append(line)
        if definition_lines:
            return "".join(definition_lines)
        else:
            return f"Function or class {funcation_class_name} not found in {file_path}."
    except Exception as e:
        return f"Error reading file {file_path}: {e}"

@tool
def find_python_func_definition_and_dependencies_tool(file_path: str, function_name: str) -> str:
    """
    Find the definition and dependencies of a Python function in a file.
    Args:
        file_path: path to the Python file.
        function_name: name of the function to find.
    Returns:
        return the function definition and its dependencies.
    """ 
    return find_python_func_definition_and_dependencies(file_path, function_name)

@tool 
def get_all_functions_class_in_file_with_grep(file_path: str, regex:str) -> str:
    """
    Get all functions and classes in python file with regex using grep command.
    Args:
        file_path: path to the file to be read.
    Returns:
        return the list of functions and classes in the file.
    """ 
    if not os.path.exists(file_path):
        return f"File {file_path} does not exist."
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
        functions_classes = []
        for i, line in enumerate(lines):
            if re.match(r'^\s*def\s+\w+\s*\(.*\)\s*:', line):
                if re.findall(regex, line):
                    functions_classes.append(f"Function: {line.strip()} (Line {i+1})")
            elif re.match(r'^\s*class\s+\w+\s*(\(.*\))?\s*:', line):
                if re.findall(regex, line):
                    functions_classes.append(f"Class: {line.strip()} (Line {i+1})")
        return "\n".join(functions_classes) if functions_classes else f"No functions or classes found in {file_path}."
    except Exception as e:
        return f"Error reading file {file_path}: {e}"

@tool 
def read_file(file_path: str, position: int = 0, length: int = 1000) -> str:
    """
    Read a file from a specific position with a given length.
    Args:
        file_path: path to the file to be read.
        position: position to start reading from.
        length: number of characters to read.
    Returns:
        return the content read from the file.
    """ 
    if not os.path.exists(file_path):
        return f"File {file_path} does not exist."
    try:
        with open(file_path, 'r') as file:
            file.seek(position)
            content = file.read(length)
            return content
    except Exception as e:
        return f"Error reading file {file_path}: {e}"

def execute_linux_command(command: str) -> str:
    """
    Execute a linux command and return the result.
    Args:
        command: linux command to be executed.
    Returns:
        return the result of the command.
    """
    if '--include=' in command:
        limit_num = 5
    else:
        limit_num = 30
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            check=True
        )
        # import pdb;pdb.set_trace()
        if result.stdout:
            find_entries = result.stdout.splitlines()
            if len(find_entries) > limit_num:
                return [False,"Find too many matched results"]
            else:
                # import pdb;pdb.set_trace()
                # find_entries = result.stdout.splitlines()
                res_elements = [i for i in find_entries if len(i) < 200]
                # for i in result.stdout.splitlines():
                #     # too long line, cut it
                #     if len(i) > 1000:
                #         pass
                #     else:
                #         res_elements.append(i)
                return [True,"\n".join(res_elements)]
        else:
            return [False,"No matched results found."]

    except subprocess.CalledProcessError as e:
        if e.returncode == 1:
            return [False,"No matches found."]
        else:
            return [False,f"Error executing command: {e.stderr}"]
    except FileNotFoundError:
        return [False,"Command not found. Please ensure the command is correct and available in the system."]
    except Exception as e:
        return [False,f"Error executing command: {e}"]

@tool 
def find_content_in_file_under_folder(command: str) -> str:
    """
    Used find command to search for content in files under a folder and return the result.
    Args:
        command: linux command to be executed.
    Returns:
        return the result of the command.
    """

    try:
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as e:
        return f"Error executing command: {e.stderr}"
    if len(result.stdout) > 300:
        return "Find too many results, do not search file with too general keyword or with only one word. make your keyword more specific."
    return result.stdout

@tool
def list_files(directory: str) -> str:
    """
    List all files in the specified directory.
    
    Args:
        directory: The path to the directory to list files from.    
    Returns:
        A string containing the list of files, separated by new lines.
    """ 
    if not os.path.exists(directory):
        return f"Directory {directory} does not exist."
    try:
        files = os.listdir(directory)
        if len(files) > 200:
            return f"Directory {directory} contains too many files ({len(files)}). Please narrow down your search."
        return "\n".join(files)
    except Exception as e:
        return f"Error listing files in directory {directory}: {e}" 

@tool
def list_files_with_grep(directory: str, keyword: str) -> str:
    """
    List all files in the specified directory that match the given keyword using grep.
    
    Args:
        directory: The path to the directory to list files from.
        keyword: The keyword to filter files.
    Returns:
        A string containing the list of matching files, separated by new lines.
    """ 
    if not os.path.exists(directory):
        return f"Directory {directory} does not exist."
    try:
        command = f"ls {directory} | grep '{keyword}'"
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if len(result.stdout) > 200:
            return f"Find too many results, do not search file with too general keyword or with only one word. make your keyword more specific."
        return result.stdout if result.stdout else f"No files found matching '{keyword}' in {directory}."
    except subprocess.CalledProcessError as e:
        return f"Error executing command: {e.stderr}"

@tool
def search_content_in_file(file_path: str, keyword: str, before_content_distance: int, after_content_distance: int) -> str:
    """
    Search for a specific keyword in a file and return the lines containing the keyword.
    
    Args:
        file_path: The path to the file to search.
        keyword: The keyword to search for in the file.
        before_content_distance: Number of lines before the matching line to include.
        after_content_distance: Number of lines after the matching line to include.
    
    Returns:
        A string containing the lines with the keyword, separated by new lines.
    """
    if not os.path.exists(file_path):
        return f"File {file_path} does not exist."
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
        find_match_line_number = 0
        for i, line in enumerate(lines):
            if keyword in line:
                find_match_line_number += 1
        if find_match_line_number > 20:
            return f"Find too many results, do not search file with too general keyword or with only one word. make your keyword more specific."
        matching_lines = []
        for i, line in enumerate(lines):
            if keyword in line:
                start = max(0, i - before_content_distance)
                end = min(len(lines), i + after_content_distance + 1)
                matching_lines.extend(lines[start:end])
                matching_lines.append("\n")  # Add a newline for separation
        return "".join(matching_lines) if matching_lines else f"No matches found for '{keyword}' in {file_path}."
    except Exception as e:
        return f"Error reading file {file_path}: {e}"

def AgentReadyNode(state):
    """Initialize the agent with the LLM and tools."""
    if 'tools' in state.keys() and len(state['tools']) > 0:
        state['_llm'] = state['llm']
        state['llm'] = state['llm'].bind_tools(state['tools'])
        state['tools_dict']= {tool.name: tool for tool in state['tools']}
    return state

async def AIAskNode(state, json_format=True, retry_num=2,timeout=90):
    """Invoke the LLM with the current messages.""" 
    # print(state['messages'][-1].content)
    try:
        response = await asyncio.wait_for(state['llm'].ainvoke(state['messages']), timeout=timeout)
    except asyncio.TimeoutError:
        log_str = f"LLM invocation timed out. Timeout after {timeout} seconds."
        state['error'] = log_str
        # print(log_str)
        return state
    except Exception as e:
        if retry_num > 2:
            log_str = f"LLM invocation error: {e}"
            state['error'] = log_str
            # print(log_str)
            return state
        else:
            return await AIAskNode(state, json_format=json_format, retry_num=retry_num+1, timeout=timeout)
    # print(response.content)
    state['messages'].append(response)
    if 'tool_calls' not in  response.additional_kwargs.keys():
        if retry_num > 2:
            log_str =  "LLM response format error after multiple retries."
            state['error'] = log_str
            # print(log_str)
            return state
        
        if json_format:
            try:
                json_reponse = yaml.safe_load(response.content.split("```")[1].split("yaml")[1])
            except Exception as e:
                log_str = f"LLM response format error: {e} seen when parsing. Need AI regenerate it."
                print(log_str)
                state['messages'].append(HumanMessage(content=f"Your response is not in the correct JSON format. Error: {e} seen when parsing. Please correct it."))
                return await AIAskNode(state, json_format=json_format, retry_num=retry_num+1, timeout=timeout)
    
    if 'tool_calls' in  response.additional_kwargs.keys():
        tool_calls = state['messages'][-1].tool_calls
        all_tool_results = []
        for t in tool_calls:
            # await ctx_notify(state['ctx'], f" ********** Calling Tool: {t['name']} with arg: {t['args']}")
            try:
                print(f"Calling Tool: {t['name']} with arg: {t['args']}")
                result = state['tools_dict'][t['name']].invoke(t['args'])
                # print(f"Tool {t['name']} returned: {result}")
            except Exception as e:
                state['error'] = f"Tool {t['name']} invocation error: {e}"
                return state
            all_tool_results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        
        state['messages'].extend(all_tool_results)
        return await AIAskNode(state, json_format=json_format, timeout=timeout)

    return state

async def A00Node(state:BASEAgentState):
    """Generating common configuration for each device in the topology."""   
    ask_list = ""
    ask_id = 1
    for i in state['user_ask']:
        ask_list += f"{ask_id}. {i}\n"
        ask_id += 1
    
    prompt_text = module_find.format(
        vtest_infra_readme, 
        ask_list,
    )

    state['messages'].append(HumanMessage(content=prompt_text))
    state = await AIAskNode(state,True)
    response = yaml.safe_load(state['messages'][-1].content.split("```")[1].split("yaml")[1])
    
    state['advertise_file_or_folder'] = response
    
    _index = 0
    for i in state['user_ask']:
        state['advertise_file_or_folder'][_index]['user_ask'] = i
        _index += 1

    state['messages'] = []
    # print(f"Find advertise_file_or_folder: {state['advertise_file_or_folder']}")
    return Command(goto='A0Node',update=state)

async def A0Node(state:BASEAgentState):
    """Generating common configuration for each device in the topology."""    
    if state['key_word_design_try_num'] == 0:
        prompt_text = keyword_design.format(
            state['user_ask'], 
        )
        state['messages'].append(HumanMessage(content=prompt_text))
    
    state = await AIAskNode(state)
    keyWordList = yaml.safe_load(state['messages'][-1].content.split("```")[1].split("yaml")[1])

    # for one ask
    for a_ask in keyWordList:
        for a_record in state['advertise_file_or_folder']:
            find_history = "\n*** Search result for ask: "+a_record['user_ask']+"\n"
            if int(a_record['id']) == int(a_ask['id']):

                for _ in a_ask['keywords']:

                    if len(a_record['related_code']) == 0:
                        grep_command = f"grep --include='*.py' -irnH '{_}' {state['project_root_folder']}tests/scripts/"
                        # print(grep_command)
                        
                        res = execute_linux_command(grep_command)
                        find_history += f"Search Keyword: {_}\n"
                        find_history += f"Search Result:\n{res[1]}\n\n"
                        a_record['find_history'] = find_history
                        
                        if res[0]:
                            a_record['find_matched'] = True
                            a_record['find_res'] = res[1].splitlines()
                            break
                    else:
                        for i in a_record['related_code']:
                            if i[0] != '/':
                                i = state['project_root_folder']+i
                            else:
                                i = state['project_root_folder']+i[1:]

                            if os.path.isdir(i):
                                grep_command = f"grep --include='*.py' -irnH '{_}' {i}"
                            elif os.path.isfile(i):
                                grep_command = f"grep -irnH '{_}' {i}"
                            else:
                                import pdb;pdb.set_trace()
                            # print(grep_command)
                            res = execute_linux_command(grep_command)
                            find_history += f"Search with keyword: {_}\n"
                            find_history += f"Result:\n{res[1]}\n\n"
                            a_record['find_history'] = find_history
                            
                            if res[0]:
                                a_record['find_matched'] = True
                                a_record['find_res'] = res[1].splitlines()
                                break
                break

    
    refind_str = '\n'
    find_history = ''
    all_matched= True
    for a_record in state['advertise_file_or_folder']: 
        if 'find_matched' not in a_record.keys():
            all_matched = False
            find_history += a_record['find_history']
            refind_str += f"{a_record['id']}.{a_record['user_ask']}\n"
    
    if all_matched is False:
        state['key_word_design_try_num'] += 1
        if state['key_word_design_try_num'] >= KEY_WORD_DESIGN_TRY_NUM_LIMIT:
            return Command(goto='A1Node',update=state)
        
        prompt_text = keyword_redsign.format(
            find_history, 
            refind_str,
        )
        # prompt_text = f"""
        #     {find_history}

        #     Based on the above search history and results, redesign 10 new keyword combinations for the following requirements that did not find matching results. 
        #         - Refer to the previously designed keyword combinations and search results, analyze why no matching results were found, and improve the keyword design.
        #         - If too many results were found, indicating previous keyword combinations were too broad or too general, try to make them more specific and targeted.
        #         - If no matching results were found, try to use more general keywords.

        #     {refind_str}

        #     Return with yaml format like below:
        #         ```yaml
        #             - id: 2
        #                 keywords:
        #                 - keyword combination 1
        #                 - keyword combination 2
        #                 - ...
        #                 - keyword combination 10
        #             - ...
        #         ```
        #     """
        state['messages'].append(HumanMessage(prompt_text))        
        return Command(goto='A0Node',update=state)
    
    else:
        # state['messages'].append(HumanMessage(find_res))
        return Command(goto='A1Node',update=state)

async def A1Node(state:BASEAgentState):
    find_res = ''
    for i in state['advertise_file_or_folder']:
        if 'find_matched' in i.keys() and i['find_matched'] is True:
            find_res += f"user_ask_id: {i['id']}\n"
            find_res += f"user_ask: {i['user_ask']}\n"
            find_res += "matched_code_snippet_list:\n"
            num = 1
            for j in i['find_res']:
                find_res+= f"{num}: {j}\n"
                num += 1
            
    prompt_text = most_matched_find.format(find_res,)
    state['messages'] = []
    state['messages'].append(HumanMessage(content=prompt_text))
    state = await AIAskNode(state,True)
    response = yaml.safe_load(state['messages'][-1].content.split("```")[1].split("yaml")[1])
    # print(f"Find most_relevant_code_snippet_index: {response}")
    for i in response:
        for a_record in state['advertise_file_or_folder']:
            if int(i['user_ask_id']) == int(a_record['id']):
                if i['most_relevant_code_snippet_index'] > 0:
                    a_record['most_relevant_code_snippet_index'] = i['most_relevant_code_snippet_index'] - 1
                    a_record['most_relevant_code_line'] = a_record['find_res'][a_record['most_relevant_code_snippet_index']]
                    res = a_record['most_relevant_code_line'].split(":")
                    if GET_ALL_DEPENDENCY is True:
                        res = find_python_func_definition_and_dependencies_by_included_codeline(res[0], int(res[1]))
                        a_record['refer_code'] = res
                    else:
                        res = get_function_definition(res[0], int(res[1]))
                        refer_code = ''
                        if isinstance(res, str):
                            refer_code =  res
                        if res[0] == 'FunctionDef':
                            refer_code = res[3]
                        elif res[0] == 'MethodDef':
                            refer_code = 'There is a class named '+ res[1] +', and a method named '+ res[2]
                            refer_code += ' in the class. The method definition is as follows:\n'
                            refer_code += res[3] + '\n'
                        a_record['refer_code'] = refer_code
    return state
    
    # if int(response['index']) == 0:
    #     state['messages'] = previous_history
    #     state['messages'].append(HumanMessage(content=f"The search results do not matched.\n {response['reason']}.\n Please re-design more specific keywords to help me find the code snippet in the code base."))
    #     state['key_word_design_try_num'] += 1
    #     if state['key_word_design_try_num'] >= KEY_WORD_DESIGN_TRY_NUM_LIMIT:
    #         return Command(goto=END,update=state)
    #     else:
    #         return Command(goto='A0Node',update=state)
    # else:
    #     find_code_line =state['find_res'][int(response['index'])-1]
    #     res = find_code_line.split(":")
    #     res = find_python_func_definition_and_dependencies_by_included_codeline(res[0], int(res[1]))
    #     state['related_code'] = res
    #     return state


async def BNode(state:BASEAgentState):
    """Generating common configuration for each device in the topology.""" 
    automate_ask = 'Automate below test case steps:\n'  
    _index = 1
    for i in user_ask:
        automate_ask += f"{_index}.{i}\n"
        _index += 1
    # user_ask = """
    # Automate below test case steps:
    #     1. create a nwpi trace on vm5 with vpn 1 and vpn 2
    #     2. shutdown interace on vm5
    #     3. stop the nwpi trace on vm5
    #     4. verify no crash or core on vm5
    # """
    related_code = ''
    for i in state['advertise_file_or_folder']:
        related_code += "\n============================================\n"
        related_code += f"User ask: {i['user_ask']}\n"
        if 'refer_code' in i.keys():
            related_code += f"Reference code:\n{i['refer_code']}\n"
        else:
            related_code += "Do not find relevant code snippet. It may be a new feature. You can implement it from scratch.\n"
        related_code += "============================================\n"

    prompt_text = f"""
        You are an expert Python automation test script developer, proficient in using the Viptela vManage Python SDK (vmanage-api) to develop automated test scripts for Viptela SD-WAN solutions.
        Vtest is a Python-based test framework specifically designed for testing Viptela SD-WAN solutions. It provides a comprehensive set of tools and libraries to facilitate the creation, execution, and management of automated test cases for Viptela devices and features.
        
        You need to help generate the test script to implement the user requirements based on the following information:
            1. User description of the automation test case they want to implement
            2. Vtest test framework guide
            3. Related code snippets from Vtest code base which may help you implement the user requirements
            4. Script template for your reference

        Here is the user description of the automation test case they want to implement:
        {automate_ask}

        Here is the Vtest test framework guide:
        {vtest_coding_guide}
        
        Here is some related code snippets from Vtest code base which may help you implement the user requirements:
        {related_code}

        Here is the Vtest script template for your reference:
        {vtest_script_template}
    """
    state['messages'] = []
    state['messages'].append(HumanMessage(content=prompt_text))
    state = await AIAskNode(state,False)
    with open("generated_code.py", 'w') as f:
        generate_code = state['messages'][-1].content.split("```")[1].split("python")[1]
        f.write(generate_code)
    print("\n\n\n\nGenerated code:")
    print(generate_code)
    # import pdb;pdb.set_trace()
    return state


graph = StateGraph(BASEAgentState)

graph.add_node("P0Node", AgentReadyNode)
graph.add_node("A00Node", A00Node)
graph.add_node("A0Node", A0Node)
graph.add_node("A1Node", A1Node)
graph.add_node("BNode", BNode)

graph.add_edge(START, 'P0Node')
graph.add_edge('P0Node', 'A00Node')
graph.add_edge('A1Node', 'BNode')
graph.add_edge('BNode', END)

VtestAgent = graph.compile(name='VtestAgent')

async def VtestAgent_run(llm, user_ask) -> str:
    """
    """
    if type(user_ask) is str:
        user_ask = [user_ask]

    res = await VtestAgent.ainvoke(
        {     
            'llm': llm,
            'messages':[],
            'tools': [],
            'user_ask': user_ask,
            "related_code": None,
            "key_word_design_try_num": 0,
            "advertise_file_or_folder": [],
            'project_root_folder': find_vtest_root(),
            # 'tools': [
            #     execute_linux_command,
            #     # list_files,
            #     search_content_in_file,
            #     # list_files_with_grep,
            #     # read_file,
            #     get_functions_classes_definitions,
            #     find_python_func_definition_and_dependencies_tool,
            #     get_all_functions_class_in_file_with_grep,
            #     # WriteFileTool(),
            # ],
            'ctx': None,
            'agent_name': 'VtestAgent',  
            'error': '',     
            "user_data": {},
            'logger': None,
        },
        config={"recursion_limit": 50}, 
    )
    return res

if __name__ == "__main__":
    from utils.utility import *
    import os
    import asyncio
    from utils.utils_llm import get_cisco_llm
    llm = get_cisco_llm()
    os.system('rm -f test.log')
    with open('test.log', 'w') as f:
        # user_ask = [
        #     # "config ipv6 address on vm5",
        #     # "config nat66 on vm5",
        #     # "create NWPI trace on vm5",
        #     # "stop NWPI trace on vm5",
        #     # "show nat table on vm5",
        #     # "show APM status on vm5",
        #     # "config qos on vm5 by configuration group",
        #     # "create nwpi trace with vpn 1 and vpn 2.",
        #     # "show qos policy on vEdge1",
        #     # 'config fnf on vm6',
        #     # "create a config-group called cg1",
        #     # "create a policy-group called pg1",
        #     # 'install utd container on vm5',
        #     'config vxlan static tunnel on vm5',
        #     'create a device template called dt1',
        #     'create a localized policy called lp1',
        #     'attach device template dt1 to vm5',
        #     'attach localized policy lp1 to vm5',
        # ]
        # user_ask = [
        #     'create a nwpi trace on vm5 with vpn 1 and vpn 2',
        #     'shutdown interace on vm5',
        #     'stop the nwpi trace on vm5',
        #     'verify no crash or core on vm5',
        # ]
    
        user_ask = [
            'add nat66 configuration on vm5',
            'verify nat66 configuration on vm5 by show command',
            'send traffic to verify nat66 on vm5',
            'shutdown interace on vm5',
            'verify no crash or core on vm5',
            'remove nat66 configuration on vm5',
        ]

        # user_ask = [
        #     'create a config-group called cg1 with on vmanager by rest api',
        #     'create a policy-group called pg1 with on vmanager by rest api',
        #     'attach config-group cg1 to vm5',
        #     'attach policy-group pg1 to vSmart',
        #     'deattach config-group cg1 from vm5',
        #     'deattach policy-group pg1 from vSmart',
        #     'remove config-group cg1 from vmanager by rest api',
        #     'remove policy-group pg1 from vmanager by rest api',
        # ]

        res = asyncio.run(VtestAgent_run(llm, user_ask)) 
        for i in res['advertise_file_or_folder']:
            f.write("============================================\n")
            f.write(f"User ask: {i['user_ask']}\n")
            if 'refer_code' in i.keys():
                f.write(f"Reference code:\n{i['refer_code']}\n")
            else:
                f.write("Do not find relevant code snippet. It should be a new feature. Please implement it from scratch.\n")
            f.write("============================================\n\n")
