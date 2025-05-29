# moulds/signal_generator.py
# 导入numpy库，用于数值计算，特别是数组操作
import numpy as np
# 导入json库，用于处理JSON数据格式（保存信号数据）
import json
# 导入os库，用于操作系统相关功能，如路径操作
import os

# 定义信号生成函数
def generate_signal(duration=10, num_points=1000, base_freq=5, noise_mean=0, noise_std=0.5):
    # 生成时间轴：从0到duration，共num_points个点，不包含结束点
    time_axis = np.linspace(0, duration, num_points, endpoint=False)
    # 设置信号的振幅为1.0
    amplitude = 1.0
    # 生成纯净的正弦信号：amplitude * sin(2 * pi * base_freq * time)
    clean_signal = amplitude * np.sin(2 * np.pi * base_freq * time_axis)
    # 生成高斯噪声：符合正态分布，具有指定的均值(noise_mean)和标准差(noise_std)
    noise = np.random.normal(noise_mean, noise_std, num_points)
    # 将纯净信号与噪声相加，得到带噪声的信号
    noisy_signal = clean_signal + noise
    # 计算实际的采样频率 (每秒采样点数)
    fs_actual = num_points / duration
    # 返回时间轴、带噪声的信号和实际采样频率
    return time_axis, noisy_signal, fs_actual

# 判断当前脚本是否作为主程序直接运行
if __name__ == '__main__':
    # 导入matplotlib.pyplot库，用于绘图，并简称为plt
    import matplotlib.pyplot as plt

    # 打印模块独立测试的开始信息
    print("\n--- 信号生成模块独立测试 (交互式多曲线对比模式 + 保存数据) ---")

    # 尝试设置matplotlib的字体以支持中文字符显示
    try:
        # 设置显示中文字体为'SimHei' (黑体)
        plt.rcParams['font.sans-serif'] = ['SimHei']
        # 设置正常显示负号
        plt.rcParams['axes.unicode_minus'] = False
        # 打印字体设置尝试信息
        print("[模块一] 尝试设置字体为 SimHei")
    except Exception as e:
        # 如果设置字体失败，打印错误信息
        print(f"[模块一] 设置字体 SimHei 失败: {e}")

    # 定义脚本独立测试时使用的固定信号时长（秒）
    DURATION_SEC_SCRIPT = 1.0
    # 定义脚本独立测试时使用的固定采样点数
    NUM_SAMPLES_SCRIPT = 1000
    # 打印将使用的固定参数信息，包括计算出的采样频率
    print(f"将使用固定时长: {DURATION_SEC_SCRIPT} 秒, 固定采样点数: {NUM_SAMPLES_SCRIPT} 个 (FS={NUM_SAMPLES_SCRIPT/DURATION_SEC_SCRIPT}Hz)")

    # 生成用于所有曲线的公共时间轴
    time_axis_common = np.linspace(0, DURATION_SEC_SCRIPT, NUM_SAMPLES_SCRIPT, endpoint=False)
    # 计算公共的采样频率
    fs_common = NUM_SAMPLES_SCRIPT / DURATION_SEC_SCRIPT

    # --- 图形设置 ---
    # 创建一个新的图形(figure)和一组子图(axes)，设置图形大小
    fig, ax = plt.subplots(figsize=(12, 6))
    # 设置图形的标题
    ax.set_title("生成的信号 (可交互添加)")
    # 设置x轴的标签
    ax.set_xlabel("时间 (秒)")
    # 设置y轴的标签
    ax.set_ylabel("幅值")
    # 显示网格线
    ax.grid(True)
    # --- 图形设置结束 ---

    # 初始化一个字典，用于存储所有生成的信号数据以便后续保存
    signals_data_to_save = {
        "time_axis": time_axis_common.tolist(), # 公共时间轴数据，转换为列表
        "fs": fs_common,                        # 公共采样频率
        "signals": []                           # 用于存储每条曲线数据的列表
    }
    # 初始化曲线计数器
    curve_count = 1
    # 初始化循环控制变量，决定是否继续添加曲线
    keep_adding = True

    # 开始循环，允许用户交互式添加多条信号曲线
    while keep_adding:
        # 打印当前正在输入的曲线编号
        print(f"\n--- 输入第 {curve_count} 条曲线的参数 ---")
        # 使用try-except块处理可能的输入错误
        try:
            # 提示用户输入基频，并获取输入字符串
            user_base_freq_str = input(f"基频 (Hz) [默认 5]: ")
            # 如果用户输入了内容，则转换为浮点数；否则使用默认值5.0
            user_base_freq = float(user_base_freq_str) if user_base_freq_str else 5.0

            # 提示用户输入噪声均值，并获取输入字符串
            user_noise_mean_str = input(f"噪声均值 [默认 0]: ")
            # 如果用户输入了内容，则转换为浮点数；否则使用默认值0.0
            user_noise_mean = float(user_noise_mean_str) if user_noise_mean_str else 0.0

            # 提示用户输入噪声标准差，并获取输入字符串
            user_noise_std_str = input(f"噪声标准差 [默认 0.5]: ")
            # 如果用户输入了内容，则转换为浮点数；否则使用默认值0.5
            user_noise_std = float(user_noise_std_str) if user_noise_std_str else 0.5
            # 检查噪声标准差是否为负数
            if user_noise_std < 0:
                # 如果为负，则重置为默认值0.5，并打印提示信息
                user_noise_std = 0.5
                print("标准差不能为负,已设为0.5")

        except ValueError: # 如果在转换输入为数字时发生错误
            # 打印错误提示信息
            print("输入无效，请使用数字。参数将使用默认值。")
            # 将参数设置为默认值
            user_base_freq = 5.0
            user_noise_mean = 0.0
            user_noise_std = 0.5

        # 打印将用于生成当前信号的参数
        print(f"生成: 基频={user_base_freq}Hz, 噪声均值={user_noise_mean}, 噪声标准差={user_noise_std}")

        # 调用generate_signal函数生成信号
        # 注意：这里使用脚本内部定义的DURATION_SEC_SCRIPT和NUM_SAMPLES_SCRIPT
        # 返回的时间轴(第一个返回值)会被忽略，因为我们使用公共时间轴time_axis_common
        _, current_signal, _ = generate_signal(
            duration=DURATION_SEC_SCRIPT,    # 使用脚本定义的固定时长
            num_points=NUM_SAMPLES_SCRIPT,   # 使用脚本定义的固定采样点数
            base_freq=user_base_freq,        # 用户输入的基频
            noise_mean=user_noise_mean,      # 用户输入的噪声均值
            noise_std=user_noise_std         # 用户输入的噪声标准差
        )

        # 为当前曲线创建一个标签，包含其参数信息
        label = f"F={user_base_freq}Hz, Mean={user_noise_mean}, Std={user_noise_std}"
        # 在图形上绘制当前生成的信号曲线，使用公共时间轴
        ax.plot(time_axis_common, current_signal, label=label, alpha=0.85) # alpha设置透明度

        # 将当前信号的数据添加到待保存的字典中
        signals_data_to_save["signals"].append({
            "label": label,                         # 曲线标签
            "base_freq": user_base_freq,            # 生成该信号所用的基频
            "noise_mean": user_noise_mean,          # 生成该信号所用的噪声均值
            "noise_std": user_noise_std,            # 生成该信号所用的噪声标准差
            "data": current_signal.tolist()         # 信号数据，转换为列表
        })

        # 询问用户是否要添加下一条曲线
        add_another = input("是否添加下一条曲线? (y/n) [y]: ").lower() # 获取输入并转为小写
        # 如果用户输入'n'，则停止添加曲线
        if add_another == 'n':
            keep_adding = False
        # 曲线计数器加1
        curve_count += 1

    # --- 文件保存路径修改 ---
    # 获取当前脚本文件(signal_generator.py)的绝对路径
    current_script_path = os.path.abspath(__file__)
    # 获取当前脚本所在的目录 (即 moulds 目录)
    current_script_dir = os.path.dirname(current_script_path)
    # 获取项目根目录 (moulds 目录的上级目录)
    project_root_dir = os.path.dirname(current_script_dir)
    # 构建输出文件夹的路径，位于项目根目录下的 "intermediate_output" 文件夹
    output_folder = os.path.join(project_root_dir, "intermediate_output")

    # 检查输出文件夹是否存在
    if not os.path.exists(output_folder):
        # 如果文件夹不存在，则尝试创建它
        try:
            os.makedirs(output_folder)
            # 打印已创建文件夹的信息
            print(f"已创建输出文件夹: {output_folder}")
        except OSError as e: # 如果创建文件夹时发生操作系统错误
            # 打印错误信息
            print(f"错误: 创建输出文件夹 {output_folder} 失败: {e}")
            # 作为备选方案，将输出文件夹设置为当前目录
            output_folder = "." # Fallback to current directory if creation fails

    # 构建最终输出文件的完整路径和文件名
    output_filename = os.path.join(output_folder, "signal_to_filters.txt")
    # --- 文件保存路径修改结束 ---

    # 尝试将收集到的信号数据写入文件
    try:
        # 以写入模式('w')打开文件，使用utf-8编码
        with open(output_filename, 'w', encoding='utf-8') as f:
            # 使用json.dump将字典数据写入文件，indent=4使其格式化（易读）
            json.dump(signals_data_to_save, f, indent=4)
        # 打印数据保存成功的消息
        print(f"\n所有生成的信号数据已保存到 {output_filename}")
    except IOError as e: # 如果发生输入/输出错误 (例如权限问题)
        # 打印文件写入错误信息
        print(f"错误：无法写入文件 {output_filename}: {e}")
    except Exception as e: # 捕获其他可能的未知错误
        # 打印未知错误信息
        print(f"保存数据时发生未知错误: {e}")

    # 打印结束添加曲线的消息
    print("\n已结束添加曲线。")
    # 更新图形标题为最终结果
    ax.set_title("生成的信号 (最终结果)")
    # 检查是否有信号被添加到数据中
    if signals_data_to_save["signals"]: # 只有在实际生成了信号时才显示图例
        # 在图形的右上角显示图例
        ax.legend(loc='upper right')

    # 显示图形，block=False表示不阻塞后续代码的执行
    plt.show(block=False)

    # 打印最终使用的采样率
    print(f"采样率: {fs_common:.2f} Hz")
    # 等待用户在控制台按Enter键，以便有时间查看图形
    input("按 Enter 键结束程序并关闭所有图形...")
    # 关闭所有matplotlib打开的图形窗口
    plt.close('all')
    # 打印程序结束信息
    print("程序结束。")
    # 打印模块一测试结束信息
    print("--- 模块一测试结束 ---")