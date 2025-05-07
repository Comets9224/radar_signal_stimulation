# update_txt_backups.py
import os
import shutil # Not directly used for copying content anymore, but kept for potential future use
import datetime

def get_formatted_timestamp_line():
    """Generates a formatted timestamp line for prepending to files."""
    now = datetime.datetime.now()
    return f"# File processed on: {now.strftime('%Y-%m-%d %H:%M:%S')}\n\n"

def generate_code_structure_string(root_dir_param, backup_subdir_name_to_exclude, self_script_abs_path_param=None):
    """
    Generates a string representation of the directory structure of root_dir_param
    in a tree-like format.
    Excludes specified backup subdirectory, common VCS/dev folders, and optionally the script itself.
    """
    structure_lines = []
    root_dir_abs = os.path.abspath(root_dir_param)
    backup_subdir_basename = os.path.basename(backup_subdir_name_to_exclude)

    excluded_dir_basenames = {
        '.git', '.svn', 'CVS', '.hg',  # VCS
        '.idea', '.vscode', 'nbproject', # IDE
        '__pycache__',                  # Python cache
        'venv', '.venv', 'env', '.env', # Virtual environments
        'dist', 'build', 'htmlcov', 'target', 'bin', 'obj', # Build artifacts & compiled output
        'node_modules', 'bower_components', # JS dependencies
        backup_subdir_basename          # The backup directory itself
    }
    try:
        for item in os.listdir(root_dir_abs): # Dynamically add .egg-info, etc.
            if item.endswith(('.egg-info', '.dist-info', '.pytest_cache', '.mypy_cache', '.tox')):
                if os.path.isdir(os.path.join(root_dir_abs, item)):
                    excluded_dir_basenames.add(item)
    except OSError:
        pass # Ignore if listing fails

    excluded_file_basenames = {'.DS_Store', 'desktop.ini', 'Thumbs.db', '.project', '.pydevproject', '.classpath'}
    excluded_file_extensions = ('.pyc', '.pyo', '.swp', '.swo', '.bak', '.tmp', '.log')

    structure_lines.append(f"{os.path.basename(root_dir_abs)}/")

    def generate_tree_recursive(current_dir_abs, prefix=""):
        nonlocal structure_lines
        
        try:
            items = sorted(os.listdir(current_dir_abs))
        except OSError:
            structure_lines.append(f"{prefix}└── [Error accessing directory]")
            return

        dirs_to_process = []
        files_to_process = []

        for item_name in items:
            item_path = os.path.join(current_dir_abs, item_name)
            is_dir = os.path.isdir(item_path)
            
            if item_name.startswith('.') or \
               (is_dir and item_name in excluded_dir_basenames) or \
               (not is_dir and (item_name in excluded_file_basenames or item_name.endswith(excluded_file_extensions))):
                continue
            
            if not is_dir and self_script_abs_path_param and item_path == self_script_abs_path_param:
                continue

            if is_dir:
                dirs_to_process.append(item_name)
            else:
                files_to_process.append(item_name)
        
        all_items_to_list = dirs_to_process + files_to_process
        
        for i, item_name in enumerate(all_items_to_list):
            is_last_item = (i == len(all_items_to_list) - 1)
            connector = "└── " if is_last_item else "├── "
            
            structure_lines.append(f"{prefix}{connector}{item_name}")
            
            item_path = os.path.join(current_dir_abs, item_name)
            if os.path.isdir(item_path):
                new_prefix = prefix + ("    " if is_last_item else "│   ")
                generate_tree_recursive(item_path, new_prefix)
            
    generate_tree_recursive(root_dir_abs)
    return "\n".join(structure_lines)


def backup_specific_py_to_txt(source_dir=".", backup_subdir="txt_backup"):
    """
    Backs up main*.py, other 'main' prefixed .py files in root, and moulds/*.py 
    to .txt files in backup_subdir, adds timestamps, and generates code_structure.txt.
    """
    source_directory_abs = os.path.abspath(source_dir) # Define source_directory_abs early
    backup_dir_path = os.path.join(source_directory_abs, backup_subdir)

    self_script_abs_path = None
    try:
        self_script_abs_path = os.path.abspath(__file__)
    except NameError:
        print("警告: 无法确定脚本自身的文件名和路径。相关排除可能不精确或备份可能包含脚本自身。")

    timestamp_line = get_formatted_timestamp_line()

    print(f"准备从 '{source_directory_abs}' 备份特定文件和生成代码结构到 '{backup_dir_path}'...")
    files_to_backup = []
    # Keep track of files already added to avoid duplicates (e.g. if a "main*.py" is also a "main_prefix*.py")
    added_source_paths = set() 

    # --- 1. Target "main*.py" (primary main script) in the source_dir ---
    # This logic tries to find a primary main script like "main.py" or "main_multiple.py"
    main_py_filename_to_use = None
    main_py_source_path = None
    
    try:
        # List files in the source directory to check for main scripts
        root_dir_files = [f for f in os.listdir(source_directory_abs) if os.path.isfile(os.path.join(source_directory_abs, f))]
        
        potential_main_files = [f for f in root_dir_files if f.lower().startswith("main") and f.endswith(".py")]
        
        if potential_main_files:
            # Prioritize "main.py" if it exists among them
            exact_main_py = "main.py"
            if exact_main_py in potential_main_files:
                main_py_filename_to_use = exact_main_py
            else:
                # If "main.py" is not found, take the first one from the sorted list of "main*.py"
                # Sorting ensures some predictability if multiple main_*.py files exist
                main_py_filename_to_use = sorted(potential_main_files)[0]
            
            if main_py_filename_to_use:
                main_py_source_path = os.path.join(source_directory_abs, main_py_filename_to_use)
                print(f"  将尝试备份主脚本: '{main_py_filename_to_use}'")

                if self_script_abs_path and main_py_source_path == self_script_abs_path:
                    print(f"  信息: '{main_py_filename_to_use}' 是当前运行的脚本，主脚本备份将被跳过。")
                else:
                    files_to_backup.append({
                        "source_path": main_py_source_path, 
                        "original_filename": main_py_filename_to_use,
                        "relative_source_dir": "" 
                    })
                    added_source_paths.add(main_py_source_path)
                    print(f"  找到主脚本: '{main_py_filename_to_use}' 准备备份。")
        else:
            print(f"  信息: 在 '{source_directory_abs}' 中未找到符合条件的主 Python 文件 (如 main.py, main_*.py)。")

    except OSError as e:
        print(f"  错误: 无法读取源目录 '{source_directory_abs}' 中的文件列表: {e}")


    # --- 2. Target OTHER .py files in the root directory starting with "main" (case-insensitive) ---
    print(f"  正在检查根目录 '{source_directory_abs}' 中其他前缀为 'main' 的 .py 文件...")
    found_other_main_prefix_files = False
    try:
        for item_name in os.listdir(source_directory_abs):
            item_source_path = os.path.join(source_directory_abs, item_name)
            # Check if it's a file, ends with .py, starts with "main" (case-insensitive),
            # and hasn't already been added (e.g., as the primary main script)
            if os.path.isfile(item_source_path) and \
               item_name.lower().startswith("main") and \
               item_name.endswith(".py") and \
               item_source_path not in added_source_paths:
                
                if self_script_abs_path and item_source_path == self_script_abs_path:
                    print(f"  信息: '{item_name}' 是当前运行的脚本，备份将被跳过。")
                else:
                    files_to_backup.append({
                        "source_path": item_source_path, 
                        "original_filename": item_name,
                        "relative_source_dir": "" 
                    })
                    added_source_paths.add(item_source_path) # Add to set to prevent re-adding
                    print(f"  找到其他 'main' 前缀文件: '{item_name}' 准备备份。")
                    found_other_main_prefix_files = True
        if not found_other_main_prefix_files and not any(f['original_filename'].lower().startswith("main") and f['relative_source_dir'] == "" for f in files_to_backup if f['source_path'] not in added_source_paths):
             # This condition is a bit complex now, simplify the message
             if not any(f['original_filename'].lower().startswith("main") and f['relative_source_dir'] == "" for f in files_to_backup):
                print(f"  信息: 在根目录中未找到额外的前缀为 'main' 的 .py 文件进行备份。")

    except OSError as e:
        print(f"  错误: 无法读取源目录 '{source_directory_abs}' 中的文件列表以查找其他 'main' 前缀文件: {e}")


    # --- 3. Target .py files in the moulds subdirectory ---
    moulds_subdir_name = "moulds"
    moulds_dir_path = os.path.join(source_directory_abs, moulds_subdir_name)
    if os.path.exists(moulds_dir_path) and os.path.isdir(moulds_dir_path):
        print(f"  正在检查 '{moulds_subdir_name}' 目录...")
        found_in_moulds = False
        try:
            for item_name in sorted(os.listdir(moulds_dir_path)): # Sort for consistent order
                if item_name.endswith(".py"):
                    item_source_path = os.path.join(moulds_dir_path, item_name)
                    if os.path.isfile(item_source_path): # Ensure it's a file
                        if self_script_abs_path and item_source_path == self_script_abs_path:
                            print(f"  信息: '{moulds_subdir_name}/{item_name}' 是当前运行的脚本，备份将被跳过。")
                        elif item_source_path in added_source_paths: # Should not happen if moulds is distinct
                            print(f"  信息: '{moulds_subdir_name}/{item_name}' 已被添加，跳过。")
                        else:
                            files_to_backup.append({
                                "source_path": item_source_path, 
                                "original_filename": item_name,
                                "relative_source_dir": moulds_subdir_name 
                            })
                            added_source_paths.add(item_source_path)
                            print(f"  找到: '{moulds_subdir_name}/{item_name}' 准备备份。")
                            found_in_moulds = True
            if not found_in_moulds:
                print(f"  信息: 在 '{moulds_dir_path}' 中未找到 .py 文件。")
        except OSError as e:
            print(f"  错误: 无法读取 '{moulds_dir_path}' 目录: {e}")
    else:
        print(f"  跳过: '{moulds_subdir_name}' 目录在 '{source_directory_abs}' 中未找到或不是一个目录。")

    # Create backup directory (if it doesn't exist)
    if not os.path.exists(backup_dir_path):
        try:
            os.makedirs(backup_dir_path)
            print(f"\n创建备份目录: {backup_dir_path}")
        except OSError as e:
            print(f"\n错误: 无法创建备份目录 {backup_dir_path}: {e}")
            print("备份中止。")
            return
    else:
        print(f"\n备份目录已存在: {backup_dir_path}")

    backed_up_py_count = 0
    if files_to_backup:
        print(f"\n开始备份 {len(files_to_backup)} 个 .py 文件...")
        # Sort files_to_backup to ensure consistent processing order, especially for logs/output
        # Sort by relative_source_dir then original_filename
        files_to_backup.sort(key=lambda x: (x["relative_source_dir"], x["original_filename"]))

        for file_info in files_to_backup:
            source_file_path = file_info["source_path"]
            original_filename = file_info["original_filename"]
            base, _ = os.path.splitext(original_filename)
            destination_filename = base + ".txt" # Backup name remains the same
            destination_file_path = os.path.join(backup_dir_path, destination_filename)

            try:
                with open(source_file_path, 'r', encoding='utf-8', errors='replace') as f_src:
                    content = f_src.read()
                with open(destination_file_path, 'w', encoding='utf-8') as f_dest:
                    f_dest.write(timestamp_line)
                    f_dest.write(content)
                
                display_source = os.path.join(file_info["relative_source_dir"], original_filename) \
                                 if file_info["relative_source_dir"] else original_filename
                print(f"  已备份: {display_source} -> {os.path.join(backup_subdir, destination_filename)}")
                backed_up_py_count += 1
            except Exception as e:
                print(f"  错误: 无法备份 {original_filename}: {e}")
    
    structure_generated = False
    print(f"\n正在生成代码结构图 '{os.path.join(backup_subdir, 'code_structure.txt')}'...")
    try:
        code_structure_content = generate_code_structure_string(
            source_directory_abs,
            backup_subdir, 
            self_script_abs_path
        )
        structure_file_path = os.path.join(backup_dir_path, "code_structure.txt")
        with open(structure_file_path, 'w', encoding='utf-8') as f_struct:
            f_struct.write(timestamp_line)
            f_struct.write(code_structure_content)
        print(f"  已生成: {os.path.join(backup_subdir, 'code_structure.txt')}")
        structure_generated = True
    except Exception as e:
        print(f"  错误: 无法生成代码结构图: {e}")

    print("\n--- 操作总结 ---")
    py_backup_summary = "Python 文件备份: 无指定文件需要备份。"
    if files_to_backup: 
        if backed_up_py_count == len(files_to_backup):
            py_backup_summary = f"Python 文件备份: 成功 ({backed_up_py_count}/{len(files_to_backup)} 文件)。"
        else:
            py_backup_summary = f"Python 文件备份: 部分成功或失败 ({backed_up_py_count}/{len(files_to_backup)} 文件)。"
    print(py_backup_summary)
    
    structure_summary = "代码结构图: 生成失败。"
    if structure_generated:
        structure_summary = "代码结构图: 成功生成。"
    print(structure_summary)

    all_targeted_actions_successful = True
    if files_to_backup and backed_up_py_count != len(files_to_backup):
        all_targeted_actions_successful = False
    if not structure_generated: 
        all_targeted_actions_successful = False
    
    if not files_to_backup and not structure_generated : 
        print("主要操作未执行或失败 (无指定PY文件，且结构图生成失败)。")
    elif all_targeted_actions_successful and (files_to_backup or structure_generated): # Check if any action was even attempted
         print("所有目标操作均已成功完成。")
    elif not all_targeted_actions_successful and (files_to_backup or structure_generated):
        print("部分目标操作未成功完成，请检查上述日志。")
    # Case where no files were targeted but structure was generated (e.g. empty project)
    elif not files_to_backup and structure_generated:
        print("Python 文件备份: 无指定文件需要备份。")
        print("代码结构图: 成功生成。")
        print("所有目标操作均已成功完成。")


    print("\n备份程序执行完毕。")

if __name__ == "__main__":
    script_location_dir = "." 
    try:
        script_location_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError: # pragma: no cover
        print("警告: 无法自动确定脚本所在目录，将使用当前工作目录作为源目录。")
        print("       这可能导致脚本自身被包含在代码结构图或备份中（如果名称匹配）。")

    backup_specific_py_to_txt(source_dir=script_location_dir)