# update_txt_backups.py
import os
import shutil # 用于复制文件

def backup_py_to_txt(source_dir=".", backup_subdir="txt_backup"):
    """
    将指定目录下的所有 .py 文件内容复制到备份子目录中，并以 .txt 后缀保存。
    会排除运行此脚本本身。
    """
    # 获取源目录的绝对路径
    source_directory_abs = os.path.abspath(source_dir)
    backup_dir_path = os.path.join(source_directory_abs, backup_subdir)

    # 获取当前运行脚本的文件名
    try:
        self_script_name = os.path.basename(os.path.abspath(__file__))
    except NameError:
        # This can happen if __file__ is not defined.
        # In such a case, we can't reliably exclude the script itself by name.
        # We'll proceed, but it might back itself up if source_dir is its own dir.
        self_script_name = None
        print("警告: 无法确定脚本自身的文件名，可能无法自动排除。")


    # 创建备份子目录（如果不存在）
    if not os.path.exists(backup_dir_path):
        try:
            os.makedirs(backup_dir_path)
            print(f"创建备份目录: {backup_dir_path}")
        except OSError as e:
            print(f"错误: 无法创建备份目录 {backup_dir_path}: {e}")
            return

    print(f"正在从 '{source_directory_abs}' 备份 .py 文件到 '{backup_dir_path}'...")

    file_count = 0
    for filename in os.listdir(source_directory_abs):
        # 检查是否是当前脚本自身
        if self_script_name and filename == self_script_name:
            continue # 静默跳过

        if filename.endswith(".py"):
            source_file_path = os.path.join(source_directory_abs, filename)
            # 构建目标文件名，将 .py 替换为 .txt
            base, ext = os.path.splitext(filename)
            destination_filename = base + ".txt"
            destination_file_path = os.path.join(backup_dir_path, destination_filename)

            try:
                shutil.copy2(source_file_path, destination_file_path) # copy2会保留元数据
                print(f"  已复制: {filename} -> {backup_subdir}/{destination_filename}")
                file_count += 1
            except Exception as e:
                print(f"  错误: 无法复制 {filename}: {e}")

    if file_count > 0:
        print(f"\n成功备份 {file_count} 个 .py 文件。")
    else:
        # Check if any .py files existed at all (other than potentially self)
        py_files_in_source = [f for f in os.listdir(source_directory_abs) if f.endswith(".py")]
        if self_script_name in py_files_in_source and len(py_files_in_source) == 1:
            print("\n未找到其他 .py 文件进行备份 (已排除脚本自身)。")
        else:
            print("\n未找到 .py 文件进行备份。")

if __name__ == "__main__":
    # 默认从脚本所在的目录进行备份
    # 如果你想从当前工作目录备份，使用: backup_py_to_txt()
    # 或者 backup_py_to_txt(source_dir=".")

    # 通常，我们希望备份脚本所在目录的py文件
    script_location_dir = os.path.dirname(os.path.abspath(__file__))
    backup_py_to_txt(source_dir=script_location_dir)

    # input("按 Enter 键退出...") # 已移除，程序将自动关闭
    print("备份完成，程序将自动退出。") # 可选：添加一个最终的完成消息