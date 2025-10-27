import os
import re
import ast


builtin_names = set(dir(__import__('builtins')))

class PythonModule():
    def __init__(self, file_path):
        self.file_path = file_path
        with open(file_path, "r", encoding="utf-8") as f:
            self.code_source = f.read()
        self.tree = ast.parse(self.code_source, file_path)
        self.source_lines = self.code_source.splitlines(keepends=True)
        self.imported_module_parsed = False
        self.imported_objs_and_modules_map = {}
        self.imported_modules = set()
        self.imported_objs = set()

class GlobalVariableDefinitionFinder(ast.NodeVisitor):
    def __init__(self, source_lines):
        self.source_lines = source_lines
        self.global_definitions = {}

    def visit_Assign(self, node):
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            var_name = node.targets[0].id
            start_line = node.lineno
            # Python 3.8+ 支持 end_lineno
            end_line = getattr(node, 'end_lineno', node.lineno)
            definition_lines = self.source_lines[start_line - 1:end_line]
            definition_text = "".join(definition_lines).strip()
            if var_name not in self.global_definitions:
                self.global_definitions[var_name] = {
                    "start_line": start_line,
                    "end_line": end_line,
                    "definition": definition_text
                }
        self.generic_visit(node)

class FuncAnalyzer(ast.NodeVisitor):
    def __init__(self,func_name):
        self.func_node = None
        self.local_define_vars = set()
        self.used_all_vars = set()
        self.func_input_param_vars = set()
        self.global_decls = set()
        self.nonlocal_decls = set()
        self.func_name = func_name
    
    def visit_FunctionDef(self, node):
        if node.name == self.func_name:
            self.func_node = node
            self.func_input_param_vars = {arg.arg for arg in node.args.args}
            # Python 3.8+ only: node.end_lineno
            self.func_start = node.lineno
            self.func_end = getattr(node, "end_lineno", node.body[-1].lineno)
            self.generic_visit(node)
    
    def visit_Assign(self, node):
        if self.func_node and self.func_start <= node.lineno <= self.func_end:
            # traceName = 'synthetic_not_sampled' node.targets[0] is traceName
            # print(F"visit_Assign at line {node.lineno}")
            for t in node.targets:
                if isinstance(t, ast.Name):
                    self.local_define_vars.add(t.id)
                elif isinstance(t, (ast.Tuple, ast.List)):
                    self.local_define_vars.update(e.id for e in t.elts if isinstance(e, ast.Name))
        self.generic_visit(node)

    def visit_For(self, node):
        if self.func_node and self.func_start <= node.lineno <= self.func_end:
            # print(F"visit_For at line {node.lineno}")
            if isinstance(node.target, ast.Name):
                self.local_define_vars.add(node.target.id)
            elif isinstance(node.target, (ast.Tuple, ast.List)):
                self.local_define_vars.update(e.id for e in node.target.elts if isinstance(e, ast.Name))
        self.generic_visit(node)

    def visit_With(self, node):
        if self.func_node and self.func_start <= node.lineno <= self.func_end:
            # print(F"visit_With at line {node.lineno}")
            for item in node.items:
                if item.optional_vars and isinstance(item.optional_vars, ast.Name):
                    self.local_define_vars.add(item.optional_vars.id)
        self.generic_visit(node)

    def visit_ExceptHandler(self, node):
        if self.func_node and self.func_start <= node.lineno <= self.func_end:
            # print(F"visit_ExceptHandler at line {node.lineno}")
            if node.name and isinstance(node.name, str):
                self.local_define_vars.add(node.name)
        self.generic_visit(node)

    def visit_Name(self, node):
        if self.func_node and self.func_start <= node.lineno <= self.func_end:
            # print(F"visit_Name at line {node.lineno}")
            if isinstance(node.ctx, ast.Load):
                self.used_all_vars.add(node.id)
        self.generic_visit(node)
    
    def visit_Global(self, node):
        if self.func_node and self.func_start <= node.lineno <= self.func_end:
            # print(F"visit_Global at line {node.lineno}")
            self.global_decls.update(node.names)
        self.generic_visit(node)
    
    def visit_Nonlocal(self, node):
        if self.func_node and self.func_start <= node.lineno <= self.func_end:
            # print(F"visit_Nonlocal at line {node.lineno}")
            self.nonlocal_decls.update(node.names)
        self.generic_visit(node)

def python_proc_parser(pModuleObj, func_name, find_content, involved_import_obj):
    tree = pModuleObj.tree

    if not pModuleObj.imported_module_parsed:
        for node in tree.body:
            if isinstance(node, ast.Import):
                for n in node.names:
                    pModuleObj.imported_modules.add(n.name)
                    pModuleObj.imported_objs.add(n.name)
                    pModuleObj.imported_objs_and_modules_map[n.name] = n.name


            elif isinstance(node, ast.ImportFrom):
                pModuleObj.imported_modules.add(node.module)
                for n in node.names:
                    pModuleObj.imported_objs.add(n.asname or n.name)
                    pModuleObj.imported_objs_and_modules_map[n.asname or n.name] = node.module

            # global variable assignment at module level should not be treated as external object
            # elif isinstance(node, ast.Assign):
            #     for t in node.targets:
            #         if isinstance(t, ast.Name):
            #             pModuleObj.imported_objs.add(t.id)

        pModuleObj.imported_module_parsed = True

    # Step 2: 解析函数体
    analyzer = FuncAnalyzer(func_name)
    analyzer.visit(tree)
    if analyzer.func_node is None:
        return
    
    # 获取函数源代码
    func_lines = pModuleObj.source_lines[analyzer.func_start-1:analyzer.func_end]
    if func_name not in find_content:
        find_content[func_name] = {}
        find_content[func_name]['start_line'] = analyzer.func_start
        find_content[func_name]['end_line'] = analyzer.func_end
        find_content[func_name]['definition'] = "".join(func_lines)

    # referenced_external_obj is the set of global objects/function in this python file
    # it may be function or variable
    referenced_external_obj = analyzer.used_all_vars   - analyzer.func_input_param_vars - analyzer.local_define_vars - builtin_names - pModuleObj.imported_objs
   
    # import pdb; pdb.set_trace()
    # If obj is function, we need to get its definition too:
    for _ in referenced_external_obj:
        if _ == func_name or _ in find_content.keys():
            continue
        python_proc_parser(pModuleObj, _, find_content, involved_import_obj)
    
    # If obj is variable, we need to get its definition too:
    external_vars = set()
    for _ in referenced_external_obj:
        if _ not in find_content.keys():
            external_vars.add(_)
    
    # external obj used in the function
    involved_import_obj.update(analyzer.used_all_vars  - analyzer.local_define_vars - analyzer.func_input_param_vars - builtin_names - external_vars)

    # import pdb; pdb.set_trace()
    for _ in external_vars:
        if _ not in find_content.keys():
            finder = GlobalVariableDefinitionFinder(pModuleObj.source_lines)
            finder.visit(tree)
            for var_name, info in finder.global_definitions.items():
                if var_name == _:
                    if _ not in find_content:
                        find_content[_] = {}
                        find_content[_]['start_line'] = info['start_line']
                        find_content[_]['end_line'] = info['end_line']
                        find_content[_]['definition'] = info['definition']
                   


def find_python_func_definition_and_dependencies(file_path, func_name):
    """
    Find a function and its dependencies in a Python file.
    Args:
        file_path: path to the Python file.
        func_name: name of the function to find.
    Returns:
        Prints the function definition and its dependencies.
    """
    find_content = {}
    involved_import_obj = set()
    pModuleObj = PythonModule(file_path)
    python_proc_parser(pModuleObj, func_name, find_content, involved_import_obj)
    sorted_funcs = sorted(find_content.items(), key=lambda x: x[1]['start_line'])
    return_str = ""
    return_str += f"In the Python file {file_path}, the implementation and dependencies of the function {func_name} are as follows:\n"
    return_str += f"***********************************************************************\n"  
    return_str += f"Imported External objects in python file:\n"
    return_str += f"Object Name  ->  From Module\n"
    for obj, mod in pModuleObj.imported_objs_and_modules_map.items():
        return_str += f"{obj}  ->  {mod}\n"  
    return_str += f"***********************************************************************\n"  
    return_str += f"Function definition and its dependent external variables and function definitions are as follows (in code order):\n"
    to_remove = set()
    for name1, details1 in find_content.items():
        for name2, details2 in find_content.items():
            if name1 != name2:
                if details1['start_line'] >= details2['start_line'] and details1['end_line'] <= details2['end_line']:
                    to_remove.add(name1)

    for func_name, details in sorted_funcs:
        if func_name in to_remove:
            continue
        return_str += details['definition'] + "\n"
    return return_str


def get_function_definition(file_path, codeline_number):
    """
    Get the function definition that includes the specified line number.
    Args:
        file_path: path to the Python file.
        codeline_number: line number to check.  
    Returns:
        The definition of the function that includes the line number, or a message if not found.
    """
    pModuleObj = PythonModule(file_path)
    tree = pModuleObj.tree
    source_lines = pModuleObj.source_lines  
    under_class = False
    class_name = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef)):
            start_line = node.lineno
            end_line = getattr(node, 'end_lineno', node.lineno)
            if start_line <= codeline_number <= end_line:
                definition_lines = source_lines[start_line - 1:end_line]
                definition_text = "".join(definition_lines).strip()
                if under_class is False:
                    return ['FunctionDef',None, node.name, definition_text]
                else:
                    return ['MethodDef',class_name, node.name, definition_text]
        elif isinstance(node, (ast.ClassDef)):
            start_line = node.lineno
            end_line = getattr(node, 'end_lineno', node.lineno)
            if start_line <= codeline_number <= end_line:
                under_class = True
                class_name = node.name
    return f"No function or class definition found that includes line {codeline_number}."


def find_python_func_definition_and_dependencies_by_included_codeline(file_path, codeline_number):

    """
    Find the function or class definition that includes the specified line number and its dependencies in a Python file.
    Args:
        file_path: path to the Python file.
        codeline_number: line number to check.
        
    Returns:
        Prints the function or class definition and its dependencies.
    """
    res = get_function_definition(file_path, codeline_number)
    if isinstance(res, str):
        return res
    if res[0] == 'FunctionDef':
        func_name = res[2]
        return find_python_func_definition_and_dependencies(file_path, func_name)
    elif res[0] == 'MethodDef':
        return_res = 'There is a class named '+ res[1] +', and a method named '+ res[2]
        return_res += ' in the class. The method definition is as follows:\n'
        return_res += res[3] + '\n'
        return return_res


def find_most_related_funtion_and_dependencies_by_keyword(keyword, project_root='/home/tester/vtest/'):
    """
    Find the most related function or class definition that includes the specified keyword and its dependencies in a Python file.
    Args:
        file_path: path to the Python file.
        keyword: keyword to search for.     
    Returns:
        Prints the function or class definition and its dependencies.
    """
    # 执行 shell 命令grep --include='*.py' -rnH ".*config.*nat66.*vm.*" /home/tester/vtest/
    # grep_command = '''grep --include='*.py' -rnH ".*config.*nat66.*vm.*" /home/tester/vtest/'''

    grep_command = f'''grep --include='*.py' -rnH "{keyword}" {project_root}'''
    print(f"Executing command: {grep_command}")

    grep_output = os.popen(grep_command).read()
    find_res=grep_output.splitlines()
    most_related_element = find_res[0]

    # need AI to find the most related element
    res  = most_related_element.split(':')
    # print(res)

    file_path = res[0]
    code_line_number = int(res[1])
    return find_python_func_definition_and_dependencies_by_included_codeline(file_path, code_line_number)

