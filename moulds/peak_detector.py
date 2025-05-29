# moulds/peak_detector.py
"""
峰值检测模块
"""
import numpy as np # 导入 NumPy 库，用于数值计算，特别是数组操作
import json # 导入 JSON 库，用于处理 JSON 数据格式（读取和写入文件）
import matplotlib.pyplot as plt # 导入 Matplotlib 的 pyplot 模块，用于绘图
import os # 导入 OS 库，用于与操作系统交互，如路径操作

# --- 中文显示配置 ---
def setup_chinese_font():
    """
    尝试设置 Matplotlib 图表中的中文字体显示。
    """
    try:
        # 尝试设置 sans-serif 字体为 'SimHei' (黑体)
        plt.rcParams['font.sans-serif'] = ['SimHei']
        # 尝试解决负号 '-' 显示为方块的问题
        plt.rcParams['axes.unicode_minus'] = False
        # 打印成功设置字体的消息
        print("[模块四] 尝试设置字体为 SimHei")
    except Exception as e:
        # 如果设置字体过程中发生任何异常，打印错误消息
        print(f"[模块四] 设置字体 SimHei 失败: {e}")
# --- 中文显示配置结束 ---

# 这个函数是模块的核心，被main.py调用
# 假设 main.py 调用时，data 是一个元组 (power_spectrum, frequencies)
# 并且 threshold_ratio 将使用默认值 0.5 (对应50%最大功率)
def detect_peaks(data, threshold_ratio=0.5, frequencies_param=None):
    """
    基于阈值和局部最大值条件检测功率谱中的峰值。

    参数:
    data (tuple or numpy.ndarray):
        - 如果是元组: 期望是 (power_spectrum, frequencies)。
        - 如果是 numpy.ndarray: 期望是 power_spectrum (此时 frequencies_param 必须提供)。
    threshold_ratio (float): 峰值点功率相对于最大功率的最小比例。
    frequencies_param (numpy.ndarray, optional): 对应的频率轴。仅当 data 是 power_spectrum 时需要。

    返回:
    tuple: (peak_indices, detected_peak_frequencies, detected_peak_powers)
        peak_indices (list): 检测到的峰值在功率谱数组中的索引列表。
        detected_peak_frequencies (list): 检测到的峰值对应的频率值列表 (精确到小数点后两位)。
        detected_peak_powers (list): 检测到的峰值对应的功率值列表。
    """
    # 检查输入 'data' 的类型和结构，并提取功率谱和频率数据
    if isinstance(data, tuple) and len(data) == 2:
        # 如果 data 是一个包含两个元素的元组，则解包为 power_spectrum 和 frequencies
        power_spectrum, frequencies = data
    elif isinstance(data, np.ndarray) and frequencies_param is not None:
        # 如果 data 是一个 NumPy 数组并且提供了 frequencies_param，则分别赋值
        power_spectrum = data
        frequencies = frequencies_param
    else:
        # 如果输入参数格式不正确，打印错误消息并返回空列表
        print("错误: detect_peaks - 输入参数 'data' 格式不正确。应为 (power_spectrum, frequencies) 元组或功率谱数组（此时需提供frequencies_param）。")
        return [], [], []

    # 确保 power_spectrum 和 frequencies 是 NumPy 数组类型
    if not isinstance(power_spectrum, np.ndarray): power_spectrum = np.array(power_spectrum)
    if not isinstance(frequencies, np.ndarray): frequencies = np.array(frequencies)

    # 如果功率谱或频率数组为空，则无法检测峰值，返回空列表
    if len(power_spectrum) == 0 or len(frequencies) == 0:
        return [], [], []
    # 如果功率谱和频率数组的长度不一致，打印错误消息并返回空列表
    if len(power_spectrum) != len(frequencies):
        print("错误: detect_peaks - 功率谱和频率数组长度不一致。")
        return [], [], []

    # 计算功率谱中的最大功率值，如果数组为空则设为0
    max_power = np.max(power_spectrum) if power_spectrum.size > 0 else 0
    # 如果最大功率非常小（接近于0），则认为没有显著峰值，返回空列表
    if max_power <= 1e-12:
        return [], [], []

    # 根据最大功率和阈值比例计算实际的功率阈值
    threshold = max_power * threshold_ratio

    # 初始化用于存储检测结果的列表
    peak_indices = [] # 存储峰值索引
    detected_peak_frequencies = [] # 存储峰值对应的频率
    detected_peak_powers = [] # 存储峰值对应的功率

    # 如果功率谱的点数少于3个，无法应用标准的局部最大值比较（需要前一个和后一个点）
    if len(power_spectrum) < 3:
        # 检查单个最大点是否超过阈值
        if max_power > threshold:
            # 获取最大功率点的索引
            idx_max = np.argmax(power_spectrum)
            # 确保频率数组不为空且索引有效
            if frequencies.size > idx_max :
                # 将该点视为峰值
                peak_indices.append(idx_max)
                detected_peak_frequencies.append(round(frequencies[idx_max], 2)) # 频率保留两位小数
                detected_peak_powers.append(power_spectrum[idx_max])
        # 返回检测到的（可能为空的）峰值信息
        return peak_indices, detected_peak_frequencies, detected_peak_powers

    # 遍历功率谱数据点（不包括第一个和最后一个点，因为它们没有完整的邻居）
    for i in range(1, len(power_spectrum) - 1):
        # 获取当前点、前一个点和后一个点的功率值
        current_power = power_spectrum[i]
        # 检查当前点的功率是否大于阈值
        if current_power > threshold:
            # 检查当前点是否是局部最大值（即大于其左右邻居）
            if current_power > power_spectrum[i - 1] and current_power > power_spectrum[i + 1]:
                # 如果满足条件，则将该点记录为峰值
                peak_indices.append(i) # 记录索引
                detected_peak_frequencies.append(round(frequencies[i], 2)) # 记录频率，保留两位小数
                detected_peak_powers.append(current_power) # 记录功率

    # 边界检查：处理功率谱的第一个点和最后一个点（可选，但为了完整性）
    if len(power_spectrum) >= 2: # 至少需要两个点才能比较边界点与其邻居
        # 检查第一个点
        if power_spectrum[0] > threshold and power_spectrum[0] > power_spectrum[1]:
            # 如果第一个点大于阈值且大于第二个点，并且尚未被记录
            if 0 not in peak_indices:
                # 将第一个点作为峰值插入到结果列表的开头
                peak_indices.insert(0,0)
                detected_peak_frequencies.insert(0,round(frequencies[0], 2))
                detected_peak_powers.insert(0,power_spectrum[0])
        # 检查最后一个点
        if power_spectrum[-1] > threshold and power_spectrum[-1] > power_spectrum[-2]:
            # 获取最后一个点的索引
            last_idx = len(power_spectrum)-1
            # 如果最后一个点大于阈值且大于倒数第二个点，并且尚未被记录
            if last_idx not in peak_indices:
                # 将最后一个点作为峰值追加到结果列表的末尾
                peak_indices.append(last_idx)
                detected_peak_frequencies.append(round(frequencies[last_idx], 2))
                detected_peak_powers.append(power_spectrum[last_idx])

    # 返回所有检测到的峰值信息
    return peak_indices, detected_peak_frequencies, detected_peak_powers


# 当该脚本作为主程序独立运行时，执行以下代码块
if __name__ == '__main__':
    # 打印模块独立测试开始的提示信息
    print("\n--- 峰值检测模块独立测试 (从文件读取频谱数据) ---")
    # 调用函数设置中文字体，以便图表能正确显示中文
    setup_chinese_font()

    # --- 路径设置 ---
    # 获取当前脚本文件所在的目录
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    # 获取项目根目录（假设当前脚本在 moulds 子目录下）
    project_root_dir = os.path.dirname(current_script_dir)
    # 构建中间输出文件夹的路径
    intermediate_output_folder = os.path.join(project_root_dir, "intermediate_output")

    # 检查中间输出文件夹是否存在
    if not os.path.exists(intermediate_output_folder):
        try:
            # 如果不存在，则创建该文件夹
            os.makedirs(intermediate_output_folder)
            # 打印创建成功的消息
            print(f"已创建输出文件夹: {intermediate_output_folder}")
        except OSError as e:
            # 如果创建文件夹失败，打印错误消息
            print(f"错误: 创建输出文件夹 {intermediate_output_folder} 失败: {e}")
            # 将输出文件夹路径回退到当前目录，以防后续操作失败
            intermediate_output_folder = "."

    # 定义输入文件的完整路径，该文件包含频谱数据
    input_filename = os.path.join(intermediate_output_folder, "spec_to_peak.txt")
    # 定义输出文件的完整路径，该文件将保存峰值检测结果供可视化模块使用
    output_filename_for_visual = os.path.join(intermediate_output_folder, "peak_to_visual.txt")
    # --- 路径设置结束 ---

    # 设置用于峰值检测的阈值比例，这里与 main.py 的默认值保持一致 (0.5 即 50%)
    # 如果独立测试时需要不同的阈值，可以在此修改
    script_threshold_ratio = 0.5

    # 尝试从输入文件加载频谱数据
    try:
        # 以只读模式打开文件，指定 UTF-8 编码
        with open(input_filename, 'r', encoding='utf-8') as f:
            # 使用 json.load 从文件加载 JSON 数据
            loaded_data_from_spec = json.load(f)
        # 打印加载成功的消息
        print(f"从 {input_filename} 加载数据成功。")
    except FileNotFoundError:
        # 如果文件未找到，打印错误消息并退出程序
        print(f"错误: 文件 {input_filename} 未找到。程序退出。")
        exit()
    except json.JSONDecodeError as e:
        # 如果文件内容不是有效的 JSON 格式，打印错误消息并退出程序
        print(f"错误: 文件 {input_filename} JSON格式无效: {e}。程序退出。")
        exit()
    except Exception as e:
        # 如果发生其他未知错误，打印错误消息并退出程序
        print(f"加载数据时发生未知错误: {e}。程序退出。")
        exit()

    # 校验加载的数据是否包含必要的键
    # 期望的键: "fs", "filter_params", "time_axis", "signals_with_spectra"
    if not loaded_data_from_spec or \
       "fs" not in loaded_data_from_spec or \
       "signals_with_spectra" not in loaded_data_from_spec or \
       "time_axis" not in loaded_data_from_spec: # 检查 'time_axis' 是否也存在
        # 如果数据格式不正确，打印错误消息并退出程序
        print(f"错误: 加载的数据格式不正确。文件 '{input_filename}' 应包含 'fs', 'time_axis', 和 'signals_with_spectra'。程序退出。")
        exit()

    # 从加载的数据中提取所需信息
    FS_FROM_FILE = loaded_data_from_spec["fs"] # 采样率
    signals_with_spectra_list = loaded_data_from_spec["signals_with_spectra"] # 包含频谱数据的信号列表
    filter_params_info = loaded_data_from_spec.get("filter_params", {}) # 滤波器参数，如果不存在则为空字典
    time_axis_from_input = loaded_data_from_spec.get("time_axis", []) # 时间轴数据，如果不存在则为空列表

    # 打印加载的采样率和滤波器参数信息
    print(f"采样率: {FS_FROM_FILE} Hz")
    if filter_params_info: # 如果滤波器参数存在
        print(f"源于滤波器的参数: {filter_params_info}")
    # 打印独立测试时使用的峰值检测阈值比例
    print(f"用于峰值检测的阈值比例 (独立测试时): {script_threshold_ratio*100:.0f}% of max power")

    # 检查信号列表是否为空
    if not signals_with_spectra_list:
        # 如果没有频谱数据，打印消息并退出程序
        print("文件中没有频谱数据。程序退出。")
        exit()

    # --- 为输出文件 peak_to_visual.txt 准备数据结构 ---
    output_data_for_visual = {
        "fs": FS_FROM_FILE, # 存储采样率
        "filter_params": filter_params_info, # 存储滤波器参数
        "time_axis": time_axis_from_input, # 传递时间轴数据
        "peak_detection_results": [] # 初始化一个空列表，用于存储每个信号的峰值检测结果
    }
    # --- 数据结构准备结束 ---

    # 初始化一个空列表，用于存储生成的图表对象
    output_plots = []

    # 遍历从文件中加载的每个信号包
    for i, signal_package_from_spec in enumerate(signals_with_spectra_list):
        # 从信号包中提取频率数据，并转换为 NumPy 数组；如果不存在则为空数组
        frequencies = np.array(signal_package_from_spec.get("frequencies", []))
        # 从信号包中提取功率谱数据，并转换为 NumPy 数组；如果不存在则为空数组
        power_spectrum = np.array(signal_package_from_spec.get("power_spectrum", []))
        # 获取原始信号标签，如果不存在则为 "N/A"
        original_label = signal_package_from_spec.get("original_label", "N/A")
        # 获取滤波后信号（或频谱对应信号）的标签，如果不存在则生成一个默认标签
        filtered_label = signal_package_from_spec.get("filtered_label", f"Spectrum {i+1}")

        # 继承原始和滤波后的时域信号数据
        original_signal_data = signal_package_from_spec.get("original_signal_data", [])
        filtered_signal_data = signal_package_from_spec.get("filtered_signal_data", [])

        # 检查频率或功率谱数据是否为空
        if frequencies.size == 0 or power_spectrum.size == 0:
            # 如果为空，打印警告消息并跳过该信号的峰值检测
            print(f"警告: 频谱数据 '{filtered_label}' 为空，跳过峰值检测。")
            # 仍然为 visualizer.py 保存一个条目，但峰值列表为空
            output_data_for_visual["peak_detection_results"].append({
                "original_label": original_label, # 原始信号标签
                "filtered_label": filtered_label, # 滤波后信号标签
                "original_signal_data": original_signal_data, # 传递原始时域信号数据
                "filtered_signal_data": filtered_signal_data, # 传递滤波后时域信号数据
                "spectrum_frequencies": frequencies.tolist(), # 频谱频率（将为空）
                "power_spectrum_data": power_spectrum.tolist(), # 功率谱数据（将为空）
                "detected_peaks": [], # 检测到的峰值（空列表）
                "threshold_ratio_used": script_threshold_ratio, # 使用的阈值比例
                "notes": "Input spectrum data was empty for peak detection." # 备注信息
            })
            continue # 继续处理下一个信号

        # 打印正在处理的信号信息
        print(f"\n正在检测信号 '{filtered_label}' (源自: '{original_label}') 的峰值...")

        # 调用峰值检测函数
        # 在独立测试时，直接传递功率谱和频率数据组成的元组
        peak_indices, detected_freqs, detected_powers = detect_peaks(
            (power_spectrum, frequencies), threshold_ratio=script_threshold_ratio
            # 此处不需要 frequencies_param，因为 data 是元组
        )

        # 打印检测到的峰值数量
        print(f"  检测到 {len(detected_freqs)} 个峰值:")
        # 初始化一个列表，用于存储要输出到文件的峰值详细信息
        detected_peaks_for_output_file = []
        if detected_freqs: # 如果检测到了峰值
            # 遍历检测到的每个峰值
            for idx, freq, power_val in zip(peak_indices, detected_freqs, detected_powers):
                # 打印峰值的频率、功率和在频谱数组中的索引
                print(f"    频率: {freq:.2f} Hz, 功率: {power_val:.4e}, 索引: {idx}")
                # 将峰值信息添加到输出列表中
                detected_peaks_for_output_file.append({
                    "frequency": freq, # 峰值频率
                    "power": power_val, # 峰值功率
                    "index_in_spectrum": int(idx) # 峰值在频谱中的索引 (确保为整数)
                })

        # --- 将所有数据（继承的 + 新的峰值信息）添加到输出结构中 ---
        output_data_for_visual["peak_detection_results"].append({
            "original_label": original_label, # 原始信号标签
            "filtered_label": filtered_label, # 滤波后信号标签
            "original_signal_data": original_signal_data, # 传递原始时域信号数据
            "filtered_signal_data": filtered_signal_data, # 传递滤波后时域信号数据
            "spectrum_frequencies": frequencies.tolist(),   # 传递频谱频率数据
            "power_spectrum_data": power_spectrum.tolist(), # 传递功率谱数据
            "detected_peaks": detected_peaks_for_output_file, # 存储检测到的峰值详细信息
            "threshold_ratio_used": script_threshold_ratio # 存储此次检测使用的阈值比例
        })
        # --- 结果添加结束 ---

        # --- 绘图 (为每个信号的频谱及其峰值创建一个图) ---
        # 创建一个新的图表和子图
        fig, ax = plt.subplots(figsize=(12, 7))
        # 绘制功率谱曲线
        ax.plot(frequencies, power_spectrum, label=f'功率谱: {filtered_label}', color='dodgerblue', alpha=0.7)

        if peak_indices: # 如果检测到了峰值索引
            # 确保峰值索引对于当前频率和功率谱是有效的
            valid_peak_indices = [idx for idx in peak_indices if idx < len(frequencies) and idx < len(power_spectrum)]
            if valid_peak_indices: # 如果存在有效的峰值索引
                # 在图上用红色圆点标出检测到的峰值
                ax.scatter(frequencies[valid_peak_indices], power_spectrum[valid_peak_indices],
                           color='red', marker='o', s=80, label='检测到的峰值', zorder=5)

        # 获取当前功率谱的最大值
        current_max_power = np.max(power_spectrum) if power_spectrum.size > 0 else 0
        if current_max_power > 1e-12 : # 如果存在显著的最大功率
            # 计算阈值线的值
            threshold_line_val = current_max_power * script_threshold_ratio
            # 在图上绘制阈值线（灰色虚线）
            ax.axhline(threshold_line_val, color='gray', linestyle='--',
                       label=f'检测阈值 ({script_threshold_ratio*100:.0f}% of max_power={current_max_power:.2e})')

        # 构建图表标题字符串
        title_str = f"峰值检测: {filtered_label}\n"
        if filter_params_info: # 如果有滤波器参数信息
            # 将滤波器参数添加到标题中
            fp_str = f"(源滤波器: {filter_params_info.get('lowcut','?')} - {filter_params_info.get('highcut','?')} Hz, {filter_params_info.get('order','?')}阶)"
            title_str += fp_str

        # 设置图表标题
        ax.set_title(title_str)
        # 设置 x 轴标签
        ax.set_xlabel("频率 (Hz)")
        # 设置 y 轴标签
        ax.set_ylabel("功率")
        # 显示图例，位置在右上角，字体稍小
        ax.legend(loc='upper right', fontsize='small')
        # 显示网格线
        ax.grid(True)

        # 初始化用于绘图的显著频率最大值
        significant_freq_max_plot = 0
        if current_max_power > 1e-12 and frequencies.size > 0: # 如果存在显著功率和频率数据
            # 定义一个用于绘图的功率阈值（例如最大功率的0.1%）
            power_threshold_plot = current_max_power * 0.001
            # 找到功率大于此阈值的点的索引
            significant_indices_plot = np.where(power_spectrum >= power_threshold_plot)[0]
            if significant_indices_plot.size > 0: # 如果存在这样的点
                # 获取最后一个显著功率点的索引
                max_sig_idx = significant_indices_plot[-1]
                if max_sig_idx < len(frequencies): # 边界检查
                    # 获取对应的频率作为显著频率的最大值
                    significant_freq_max_plot = frequencies[max_sig_idx]

        # 设置 x 轴的显示范围上限，默认为奈奎斯特频率
        display_xlim_max_plot = FS_FROM_FILE / 2
        if significant_freq_max_plot > 0: # 如果计算出了显著频率最大值
            # 将 x 轴上限设置为显著频率最大值的1.5倍，但不超过奈奎斯特频率
            display_xlim_max_plot = min(significant_freq_max_plot * 1.5, FS_FROM_FILE / 2)
            # 确保 x 轴上限至少为20Hz，以显示低频部分
            display_xlim_max_plot = max(display_xlim_max_plot, 20)
        else: # 如果没有显著功率，显示一个较小的默认范围
            display_xlim_max_plot = min(30, FS_FROM_FILE / 2) # 例如，最多显示到30Hz或奈奎斯特频率
        # 设置 x 轴的显示范围从0到计算出的上限
        ax.set_xlim(0, display_xlim_max_plot)

        # 设置 y 轴的显示范围
        if current_max_power > 1e-12: # 如果存在显著功率
            # y 轴范围略大于功率谱范围，以提供一些边距
             ax.set_ylim(bottom=-0.05 * current_max_power, top=current_max_power * 1.15)
        else: # 如果功率谱很小或为0
            # 设置一个默认的 y 轴范围
            ax.set_ylim(bottom=0, top=1.0)

        # 调整图表布局，确保标题等元素不会重叠
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        # 将生成的图表对象添加到列表中
        output_plots.append(fig)
        # --- 绘图结束 ---

    # --- 保存峰值检测结果到文件 peak_to_visual.txt ---
    # 检查是否有峰值检测结果需要保存
    if output_data_for_visual["peak_detection_results"]:
        try:
            # 以写入模式打开输出文件，指定 UTF-8 编码
            with open(output_filename_for_visual, 'w', encoding='utf-8') as f_out:
                # 将 `output_data_for_visual` 字典以 JSON 格式写入文件，使用缩进美化输出
                json.dump(output_data_for_visual, f_out, indent=2)
            # 打印保存成功的消息
            print(f"\n峰值检测结果已保存到: {output_filename_for_visual}")
        except Exception as e:
            # 如果保存失败，打印错误消息
            print(f"错误: 保存峰值检测结果到 {output_filename_for_visual} 失败: {e}")
    else:
        # 如果没有结果可保存，打印相应消息
        print(f"\n没有峰值检测结果可保存到 {output_filename_for_visual}。")
    # --- 文件保存结束 ---

    # 如果生成了任何图表
    if output_plots:
        # 打印生成的图表数量
        print(f"\n共生成 {len(output_plots)} 个峰值检测图。")
        # 显示所有生成的图表，非阻塞模式
        plt.show(block=False)
        # 等待用户按 Enter 键
        input("按 Enter 键结束程序并关闭所有图形...")
        # 关闭所有 Matplotlib 图形窗口
        plt.close('all')
        # 打印程序结束的消息
        print("程序结束。")
    else:
        # 如果没有生成图表，打印相应消息
        print("\n没有生成任何峰值检测图。")

    # 打印模块独立测试结束的提示信息
    print("--- 峰值检测模块独立测试结束 ---")