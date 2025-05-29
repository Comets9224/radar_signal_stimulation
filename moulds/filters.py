# moulds/filters.py
# 模块的顶层说明，指明这是滤波器模块
"""
滤波器模块
"""
# 导入 scipy.signal 库中的滤波器设计、应用和频率响应函数
from scipy.signal import butter, filtfilt, freqz, sosfiltfilt, sosfreqz
# 导入 numpy 库，用于数值计算，特别是数组操作
import numpy as np
# 导入 json 库，用于处理 JSON 数据格式（读写文件）
import json
# 导入 matplotlib.pyplot 库，用于绘图
import matplotlib.pyplot as plt
# 导入 os 库，用于操作系统相关功能，如路径操作
import os

# --- 中文显示配置 ---
# 尝试设置 matplotlib 使用 SimHei 字体以正确显示中文
try:
    # 设置 sans-serif 字体系列为 SimHei
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 设置坐标轴负号正常显示 (当使用 SimHei 等中文字体时，负号可能显示为方框)
    plt.rcParams['axes.unicode_minus'] = False
    # 打印字体设置尝试成功的消息
    print("[模块二] 尝试设置字体为 SimHei")
# 捕获设置字体时可能发生的任何异常
except Exception as e:
    # 打印字体设置失败的消息及错误详情
    print(f"[模块二] 设置字体 SimHei 失败: {e}")
# --- 中文显示配置结束 ---

def butter_bandpass(lowcut, highcut, fs, order=5):
    """
    设计一个Butterworth带通滤波器的二阶节(SOS)表示。

    参数:
    lowcut (float): 低截止频率 (Hz)。这是通带的下边界。
    highcut (float): 高截止频率 (Hz)。这是通带的上边界。
    fs (float): 采样频率 (Hz)。信号的采样率。
    order (int, optional): 滤波器阶数。默认为5。阶数越高，过渡带越窄，但可能引入更多延迟或振荡。

    返回:
    numpy.ndarray: 滤波器的SOS系数数组。每行包含一个二阶节的系数 [b0, b1, b2, a0, a1, a2]。
                   如果参数无效或设计失败，则返回None。
    """
    # 计算奈奎斯特频率，它是理论上可以无失真恢复信号的最大频率，为采样频率的一半
    nyq = 0.5 * fs
    # 检查截止频率是否在有效范围内：必须大于0，小于奈奎斯特频率，且低截止频率小于高截止频率
    if not (0 < lowcut < nyq and 0 < highcut < nyq and lowcut < highcut):
        # 如果截止频率设置不合理，打印警告信息，说明具体参数值
        print(f"警告: 滤波器截止频率设置不合理。lowcut={lowcut}, highcut={highcut}, nyq={nyq}, fs={fs}")
        # 打印合理设置的要求
        print("  要求: 0 < lowcut < nyq, 0 < highcut < nyq, lowcut < highcut")
        # 返回 None 表示滤波器设计失败
        return None
    # 将低截止频率归一化到 (0, 1) 区间，其中1对应奈奎斯特频率
    low = lowcut / nyq
    # 将高截止频率归一化到 (0, 1) 区间
    high = highcut / nyq
    # 再次检查归一化后的截止频率是否仍在 (0, 1) 区间内，且归一化低截止小于归一化高截止
    if not (0 < low < 1 and 0 < high < 1 and low < high):
        # 如果归一化截止频率不合理，打印警告信息
        print(f"警告: 归一化截止频率不合理。low_norm={low}, high_norm={high}")
        # 返回 None 表示滤波器设计失败
        return None
    # 尝试使用 scipy.signal.butter 函数设计Butterworth滤波器
    try:
        # 'order': 滤波器阶数
        # '[lowcut, highcut]': 截止频率数组 (Hz)
        # 'btype='band'': 设计带通滤波器
        # 'fs=fs': 指定采样频率 (Hz), butter函数内部会进行归一化处理
        # 'output='sos'': 输出格式为二阶节 (Second-Order Sections)，这种形式在数值上更稳定
        sos = butter(order, [lowcut, highcut], btype='band', fs=fs, output='sos')
        # 返回设计好的SOS系数
        return sos
    # 捕获在滤波器设计过程中可能发生的 ValueError (例如，参数组合无效)
    except ValueError as e:
        # 打印滤波器设计失败的错误信息及异常详情
        print(f"错误: scipy.signal.butter 设计滤波器失败: {e}")
        # 打印导致错误的具体参数
        print(f"  参数: order={order}, Wn=[{lowcut}, {highcut}], btype='band', fs={fs}, output='sos'")
        # 返回 None 表示滤波器设计失败
        return None

# 这个函数是模块的核心功能，预期会被其他模块（如main.py）调用
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    应用Butterworth带通滤波器（使用零相位滤波sosfiltfilt）到输入信号。

    参数:
    data (numpy.ndarray or list): 输入信号数据。可以是一维数组或列表。
    lowcut (float): 低截止频率 (Hz)。
    highcut (float): 高截止频率 (Hz)。
    fs (float): 采样频率 (Hz)。
    order (int, optional): 滤波器阶数。默认为5。

    返回:
    numpy.ndarray: 滤波后的信号数据。如果滤波器设计或应用失败，将返回原始输入数据。
    """
    # 确保输入数据 data 是一个 numpy 数组，如果不是，则进行转换
    if not isinstance(data, np.ndarray):
        data = np.array(data) # 将列表等转换为numpy数组

    # 调用 butter_bandpass 函数设计滤波器，获取SOS系数
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    # 检查滤波器设计是否成功 (即sos是否为None)
    if sos is None:
        # 如果设计失败，打印错误信息
        print("错误 (butter_bandpass_filter): 滤波器设计失败 (sos为None)，返回原始数据。")
        # 在设计失败时返回原始数据，以避免程序中断
        return data
    # 进一步检查SOS系数是否有效 (例如，非空、不包含NaN值)
    if not isinstance(sos, np.ndarray) or sos.size == 0 or np.any(np.isnan(sos)):
        # 如果SOS系数无效，打印错误信息
        print("错误 (butter_bandpass_filter): 滤波器SOS系数无效 (空或NaN)，返回原始数据。")
        # 返回原始数据
        return data
    # 尝试应用滤波器
    try:
        # 使用 sosfiltfilt 函数进行滤波。这是一个前向-后向滤波器，结果是零相位失真。
        # 它将滤波器应用于数据，然后再次反向应用，以校正相位延迟。
        y = sosfiltfilt(sos, data)
        # 返回滤波后的信号
        return y
    # 捕获在应用滤波器过程中可能发生的 ValueError (例如，数据或SOS系数问题)
    except ValueError as e:
        # 打印应用滤波器失败的错误信息及异常详情
        print(f"错误 (butter_bandpass_filter): scipy.signal.sosfiltfilt 应用滤波器失败: {e}")
        # 提示可能的原因
        print("  可能是由于滤波器SOS系数或数据问题。返回原始数据。")
        # 在应用失败时返回原始数据
        return data

# --- 以下函数主要用于 if __name__ == '__main__' 条件块中的独立测试 ---
def compute_power_spectrum(signal, fs):
    """
    计算给定信号的功率谱。

    参数:
    signal (numpy.ndarray): 输入的一维信号数据。
    fs (float): 信号的采样频率 (Hz)。

    返回:
    tuple (numpy.ndarray, numpy.ndarray):
        - xf_half (numpy.ndarray): 频率轴 (单边，从0到奈奎斯特频率)。
        - power_spectrum_final (numpy.ndarray): 对应的功率谱值。
        如果输入信号为空，则返回两个空numpy数组。
    """
    # 获取信号的长度 (数据点数)
    N = len(signal)
    # 如果信号为空 (长度为0)，打印警告并返回空数组
    if N == 0:
        print("警告: compute_power_spectrum 接收到空信号。")
        return np.array([]), np.array([]) # 返回两个空数组
    # 计算信号的快速傅里叶变换 (FFT)
    yf = np.fft.fft(signal)
    # 计算 FFT 对应的频率点。np.fft.fftfreq(N, d) 中 d 是采样间隔 (1/fs)
    xf = np.fft.fftfreq(N, 1 / fs)
    # 由于实信号的FFT结果是对称的，我们通常只关心正频率部分 (单边频谱)
    # 取 FFT 结果的绝对值，并截取前半部分 (包括直流分量和奈奎斯特频率点，如果N为偶数)
    yf_half_abs = np.abs(yf[:N // 2 + 1])
    # 对应地，截取频率轴的前半部分
    xf_half = xf[:N // 2 + 1]
    # 对于偶数长度N，np.fft.fftfreq 可能将奈奎斯特频率点表示为负值，这里确保其为正
    if N > 0 and N % 2 == 0 and xf_half[-1] < 0:
            xf_half[-1] = -xf_half[-1] # 将其转换为正值
    # 计算幅度谱：将FFT结果的绝对值除以N
    amp_spectrum = yf_half_abs / N
    # 对幅度谱进行校正，以反映单边频谱的能量
    if N > 0: # 确保信号长度不为0
        if N % 2 == 0: # 如果信号长度为偶数
            if N//2 > 0: # 确保有除了直流和奈奎斯特之外的频率分量
                 # 将除了直流分量(索引0)和奈奎斯特频率分量(索引N//2)之外的幅度乘以2
                 amp_spectrum[1:N//2] *= 2
        else: # 如果信号长度为奇数
            if len(amp_spectrum) > 1: # 确保有除了直流之外的频率分量
                # 将除了直流分量(索引0)之外的所有幅度乘以2
                amp_spectrum[1:] *= 2
    # 计算功率谱：幅度谱的平方
    power_spectrum_final = amp_spectrum**2
    # 返回单边频率轴和对应的功率谱值
    return xf_half, power_spectrum_final

def generate_signal_from_params(t_axis, label, fs_signal):
    """
    根据标签字符串中包含的参数生成一个测试信号。
    此函数主要用于在独立测试脚本中生成具有特定特征（如频率、幅度、噪声）的扩展时域信号。

    参数:
    t_axis (numpy.ndarray): 时间轴数组。
    label (str): 包含信号参数的标签字符串。
                 格式示例: "F=5Hz,Amplitude=1,Mean=0,Std=0.1"。
                 参数键: 'F' (频率), 'Amplitude' (幅度), 'Mean' (均值/直流偏置), 'Std' (噪声标准差)。
    fs_signal (float): 信号的采样频率 (Hz)。此参数当前在此函数实现中未使用，但保留以兼容旧接口。

    返回:
    numpy.ndarray: 生成的信号。如果参数解析失败，将使用默认值生成信号。
    """
    # 初始化一个空字典来存储从标签中解析出的参数
    params = {}
    # 尝试从标签字符串中解析参数
    try:
        # 按逗号分割标签字符串，得到各个参数部分 (如 "F=5Hz")
        parts = label.split(',')
        # 遍历每个参数部分
        for part in parts:
            # 按等号分割键值对 (如 "F" 和 "5Hz")
            key_val = part.split('=')
            # 确保成功分割为键和值两部分
            if len(key_val) == 2:
                # 获取键，并去除首尾可能存在的空格
                key = key_val[0].strip()
                # 获取值字符串，并去除首尾空格
                val_str = key_val[1].strip()
                # 如果值字符串中包含 "Hz" (通常用于频率)，则移除它以便转换为数值
                if 'Hz' in val_str:
                    val_str = val_str.replace('Hz', '')
                # 将值字符串转换为浮点数，并存入参数字典
                params[key] = float(val_str)
    # 捕获在解析过程中可能发生的任何异常 (如格式错误、无法转换类型等)
    except Exception as e:
        # 打印解析失败的警告信息，并提示将使用默认值
        print(f"  [生成长信号警告] 无法从标签 '{label}' 解析参数: {e}. 将使用默认值。")
    # 从参数字典中获取频率 ('F')，如果未找到则使用默认值 0.0 Hz
    frequency = params.get('F', 0.0)
    # 从参数字典中获取幅度 ('Amplitude')，如果未找到则使用默认值 1.0
    amplitude = params.get('Amplitude', 1.0)
    # 从参数字典中获取均值/直流偏置 ('Mean')，如果未找到则使用默认值 0.0
    mean_val = params.get('Mean', 0.0)
    # 从参数字典中获取噪声标准差 ('Std')，如果未找到则使用默认值 0.0 (即无噪声)
    std_dev = params.get('Std', 0.0)
    # 生成正弦信号部分，包含均值 (直流偏置)
    signal_with_dc = mean_val + amplitude * np.sin(2 * np.pi * frequency * t_axis)
    # 生成高斯白噪声，均值为0，标准差为 std_dev，长度与时间轴一致
    noise_component = np.random.normal(0, std_dev, len(t_axis))
    # 返回最终信号：正弦信号与噪声分量之和
    return signal_with_dc + noise_component
# --- 独立测试辅助函数结束 ---


# 当脚本作为主程序直接运行时 (而不是被导入为模块时)，执行以下代码块
if __name__ == '__main__':
    # 打印模块独立测试开始的提示信息
    print("\n--- 滤波器模块独立测试 (从文件读取数据并滤波) ---")

    # --- 路径设置 ---
    # 获取当前脚本文件 (filters.py) 的绝对路径
    # os.path.abspath(__file__) -> /path/to/your/project/moulds/filters.py
    # os.path.dirname(...) -> /path/to/your/project/moulds
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    # 获取项目根目录 (通常是当前脚本所在目录的上级目录)
    # os.path.dirname(current_script_dir) -> /path/to/your/project
    project_root_dir = os.path.dirname(current_script_dir)
    # 构建中间输出文件夹的完整路径，该文件夹位于项目根目录下
    intermediate_output_folder = os.path.join(project_root_dir, "intermediate_output")

    # 检查并确保输出文件夹存在
    if not os.path.exists(intermediate_output_folder):
        # 如果文件夹不存在，尝试创建它
        try:
            os.makedirs(intermediate_output_folder) # 创建文件夹，包括任何必要的父目录
            # 打印文件夹创建成功的消息
            print(f"已创建输出文件夹: {intermediate_output_folder}")
        # 捕获在创建文件夹时可能发生的 OSError (例如，权限问题)
        except OSError as e:
            # 打印创建文件夹失败的错误信息
            print(f"错误: 创建输出文件夹 {intermediate_output_folder} 失败: {e}")
            # 如果创建失败，则将输出文件夹回退到当前工作目录 (".")
            intermediate_output_folder = "."

    # 定义输入文件的完整路径 (包含待滤波信号数据)
    input_filename = os.path.join(intermediate_output_folder, "signal_to_filters.txt")
    # 定义输出文件的完整路径 (用于存储滤波后的信号数据，供后续频谱分析模块使用)
    output_filename_for_spec = os.path.join(intermediate_output_folder, "filters_to_spec.txt")
    # --- 路径设置结束 ---

    # 定义独立测试时使用的滤波器参数 (这些参数模拟了main.py调用此模块时的典型设置)
    LOWCUT_FILTER_SCRIPT = 4.0  # 低截止频率 (Hz)
    HIGHCUT_FILTER_SCRIPT = 6.0 # 高截止频率 (Hz)
    ORDER_FILTER_SCRIPT = 5     # 滤波器阶数
    # FS_FILTER (采样频率) 将从加载的输入文件中读取，而不是在此处硬编码

    # 定义其他用于独立测试的参数
    NORMALIZE_SPECTRUM_DISPLAY = False # 控制频谱图是否进行归一化显示 (True/False)
    EXTENDED_TIME_DURATION = 5.0       # 扩展时域图的显示时长 (秒)


    # 尝试从输入文件加载JSON数据
    try:
        # 以只读模式 ('r') 打开文件，指定编码为 'utf-8' 以正确处理中文字符
        with open(input_filename, 'r', encoding='utf-8') as f:
            # 使用 json.load 从文件对象 f 中解析 JSON 数据
            loaded_data = json.load(f)
        # 打印数据加载成功的消息
        print(f"从 {input_filename} 加载数据成功。")
    # 捕获文件未找到的错误
    except FileNotFoundError:
        # 打印文件未找到的错误信息并终止程序
        print(f"错误: 文件 {input_filename} 未找到。程序退出。")
        exit() # 退出程序
    # 捕获JSON数据格式无效的错误
    except json.JSONDecodeError:
        # 打印JSON格式无效的错误信息并终止程序
        print(f"错误: 文件 {input_filename} JSON格式无效。程序退出。")
        exit() # 退出程序
    # 捕获在加载数据过程中可能发生的其他未知异常
    except Exception as e:
        # 打印未知错误信息及异常详情，并终止程序
        print(f"加载数据时发生未知错误: {e}。程序退出。")
        exit() # 退出程序

    # 检查加载的数据是否有效，并包含必需的键 ("time_axis", "signals", "fs")
    if not loaded_data or "time_axis" not in loaded_data or "signals" not in loaded_data or "fs" not in loaded_data:
        # 如果数据格式不正确，打印错误信息并终止程序
        print("错误: 加载的数据格式不正确。缺少 time_axis, signals, 或 fs。程序退出。")
        exit() # 退出程序

    # 从加载的数据中提取时间轴，并将其转换为 numpy 数组
    time_axis_original_from_file = np.array(loaded_data["time_axis"])
    # 从加载的数据中提取信号信息列表 (每个元素是一个包含 "data" 和 "label" 的字典)
    signals_info_list = loaded_data["signals"]
    # 从加载的数据中提取采样频率 (FS_FILTER)
    FS_FILTER = loaded_data["fs"]

    # 检查信号信息列表是否为空
    if not signals_info_list:
        # 如果文件中没有信号数据，打印提示信息并终止程序
        print("文件中没有信号数据。程序退出。")
        exit() # 退出程序

    # 打印从文件读取到的采样频率
    print(f"采样率 FS_FILTER: {FS_FILTER} Hz")
    # 打印本次独立测试将使用的滤波器参数
    print(f"滤波器参数 (独立测试时): 通带 {LOWCUT_FILTER_SCRIPT}-{HIGHCUT_FILTER_SCRIPT} Hz, 阶数 {ORDER_FILTER_SCRIPT}")

    # --- 初始化用于输出文件 filters_to_spec.txt 的数据结构 ---
    # 这是一个字典，将包含所有处理结果和相关元数据
    output_data_for_spec = {
        "fs": FS_FILTER,  # 存储采样频率
        "filter_params": { # 存储本次测试使用的滤波器参数
            "lowcut": LOWCUT_FILTER_SCRIPT,      # 低截止频率
            "highcut": HIGHCUT_FILTER_SCRIPT,    # 高截止频率
            "order": ORDER_FILTER_SCRIPT,        # 滤波器阶数
            "type": "butterworth_bandpass_sos"   # 滤波器类型 (指明是Butterworth带通，SOS实现)
        },
        "time_axis": time_axis_original_from_file.tolist(), # 存储原始时间轴 (转换为列表以便JSON序列化)
        "processed_signals": [] # 初始化一个空列表，用于存储每个信号的原始数据和滤波后数据
    }
    # --- 数据结构定义结束 ---

    # --- 主绘图设置 (用于显示所有信号的时域图和第一个信号的频谱分析) ---
    # 创建一个图形窗口 (fig_main) 和一组子图 (axes_main)，布局为2行2列
    # figsize参数设置图形窗口的大小 (宽度17英寸, 高度10英寸)
    fig_main, axes_main = plt.subplots(2, 2, figsize=(17, 10))
    # 设置主图形窗口的标题，包含滤波器参数和数据来源信息
    fig_main.suptitle(f"主信号分析 (滤波器: {LOWCUT_FILTER_SCRIPT}-{HIGHCUT_FILTER_SCRIPT}Hz, {ORDER_FILTER_SCRIPT}阶, 文件时长)", fontsize=16)
    # 获取第一个子图 (位置[0,0]，即左上角)，用于显示原始信号时域图
    ax_original_time_main = axes_main[0, 0]
    # 获取第二个子图 (位置[0,1]，即右上角)，用于显示滤波后信号时域图
    ax_filtered_time_main = axes_main[0, 1]
    # 获取第三个子图 (位置[1,0]，即左下角)，用于显示第一个信号的原始与滤波后频谱
    ax_combined_spectrum_main = axes_main[1, 0]
    # 获取第四个子图 (位置[1,1]，即右下角)，用于显示滤波器频率响应曲线
    ax_filter_response_main = axes_main[1, 1]

    # 设置原始信号时域图的标题、X轴标签、Y轴标签，并显示网格
    ax_original_time_main.set_title("1. 原始信号 (时域 - 所有信号, 文件时长)")
    ax_original_time_main.set_xlabel("时间 (s)"); ax_original_time_main.set_ylabel("幅值"); ax_original_time_main.grid(True)
    # 设置滤波后信号时域图的标题、X轴标签、Y轴标签，并显示网格
    ax_filtered_time_main.set_title(f"2. 滤波后信号 (时域 - 所有信号, 文件时长)")
    ax_filtered_time_main.set_xlabel("时间 (s)"); ax_filtered_time_main.set_ylabel("幅值"); ax_filtered_time_main.grid(True)
    # --- 绘图设置结束 ---

    # 获取一个颜色映射表 ('tab10')，并根据信号数量选择颜色，用于区分不同信号的绘图曲线
    colors = plt.cm.get_cmap('tab10', len(signals_info_list))

    # 遍历从文件中加载的每个信号信息 (signal_info_from_file 是包含 "data" 和 "label" 的字典)
    # enumerate 提供索引 i 和对应的值
    for i, signal_info_from_file in enumerate(signals_info_list):
        # 从当前信号信息中提取原始信号数据，并确保其为 numpy 数组
        original_signal_data = np.array(signal_info_from_file["data"])
        # 从当前信号信息中获取标签 (如果 "label" 键不存在，则使用默认标签 "信号 i+1")
        label_from_file = signal_info_from_file.get("label", f"信号 {i + 1}")

        # 当前信号使用的时间轴 (与文件中加载的原始时间轴一致)
        current_time_axis = time_axis_original_from_file
        # 检查并调整数据长度以匹配时间轴长度，以防两者不一致 (理想情况下应匹配)
        # 这部分主要为了绘图时数据和时间轴能对齐
        if len(current_time_axis) > len(original_signal_data):
            # 如果时间轴比信号数据长，截断时间轴以匹配数据长度 (仅用于绘图)
            current_time_axis_plot = current_time_axis[:len(original_signal_data)]
            # 原始信号数据用于绘图的部分 (保持不变，因为数据更短)
            original_signal_data_plot = original_signal_data
        elif len(current_time_axis) < len(original_signal_data):
            # 如果信号数据比时间轴长，打印警告，并截断信号数据以匹配时间轴长度 (仅用于绘图)
            print(f"警告: 信号 '{label_from_file}' 数据长度 ({len(original_signal_data)}) 大于时间轴长度 ({len(current_time_axis)}). 将截断数据进行绘图。")
            original_signal_data_plot = original_signal_data[:len(current_time_axis)]
            # 时间轴用于绘图的部分 (保持不变，因为时间轴更短)
            current_time_axis_plot = current_time_axis
        else:
            # 如果长度匹配，直接使用原始数据和时间轴进行绘图
            original_signal_data_plot = original_signal_data
            current_time_axis_plot = current_time_axis


        # 在主图的原始信号子图 (ax_original_time_main) 上绘制当前原始信号 (可能被截断的绘图版本)
        # 使用从颜色映射中获取的颜色 (colors(i)) 和一定的透明度 (alpha)
        ax_original_time_main.plot(current_time_axis_plot, original_signal_data_plot, label=label_from_file, color=colors(i), alpha=0.7)

        # 应用带通滤波器到完整的原始信号数据 (original_signal_data)
        # 使用脚本中定义的滤波器参数 (LOWCUT_FILTER_SCRIPT, HIGHCUT_FILTER_SCRIPT, FS_FILTER, ORDER_FILTER_SCRIPT)
        filtered_signal = butter_bandpass_filter(original_signal_data,
                                                 LOWCUT_FILTER_SCRIPT, HIGHCUT_FILTER_SCRIPT,
                                                 FS_FILTER, order=ORDER_FILTER_SCRIPT)

        # 准备用于绘图的滤波后信号数据
        # 如果原始信号在绘图时被截断，则滤波后的信号也相应截断以保持与时间轴对齐
        filtered_signal_plot = filtered_signal
        if len(current_time_axis_plot) < len(filtered_signal):
            # 截断滤波信号以匹配绘图用的时间轴长度
            filtered_signal_plot = filtered_signal[:len(current_time_axis_plot)]

        # 在主图的滤波后信号子图 (ax_filtered_time_main) 上绘制当前滤波后的信号 (可能被截断的绘图版本)
        ax_filtered_time_main.plot(current_time_axis_plot, filtered_signal_plot, label=label_from_file, color=colors(i), alpha=0.8)

        # --- 将原始数据和滤波后数据添加到输出数据结构 output_data_for_spec 中 ---
        # 这将用于后续保存到文件
        output_data_for_spec["processed_signals"].append({
            "original_label": label_from_file, # 存储原始信号的标签
            "label": f"Filtered: {label_from_file}", # 存储滤波后信号的标签 (添加 "Filtered: " 前缀)
            "original_data": original_signal_data.tolist(), # 存储完整的原始信号数据 (转换为列表以便JSON序列化)
            "filtered_data": filtered_signal.tolist()   # 存储完整的滤波后信号数据 (转换为列表)
        })
        # --- 数据添加结束 ---

        # 如果当前是第一个信号 (i == 0)，则在主图的频谱子图上绘制其原始频谱和滤波后频谱
        if i == 0:
            # 将第一个信号的完整原始数据和完整滤波后数据分别赋值给临时变量，用于频谱计算
            first_original_signal_data = original_signal_data
            first_filtered_signal_data = filtered_signal

            # 设置主图组合频谱子图 (ax_combined_spectrum_main) 的标题和轴标签
            ax_combined_spectrum_main.set_title(f"3. 信号 '{label_from_file}' 功率谱 (文件时长)")
            ax_combined_spectrum_main.set_xlabel("频率 (Hz)")
            ax_combined_spectrum_main.grid(True) # 显示网格
            # 根据 NORMALIZE_SPECTRUM_DISPLAY 参数设置 Y 轴标签 (功率 或 归一化功率)
            y_label_power_main = "功率"
            if NORMALIZE_SPECTRUM_DISPLAY: y_label_power_main = "归一化功率"
            ax_combined_spectrum_main.set_ylabel(y_label_power_main)

            # 计算第一个原始信号的功率谱
            freq_orig_main, spec_orig_main = compute_power_spectrum(first_original_signal_data, FS_FILTER)
            # 准备用于绘图的原始信号频谱数据
            spec_orig_plot_main = spec_orig_main
            # 如果 NORMALIZE_SPECTRUM_DISPLAY 为 True，且频谱数据有效 (非空，最大值大于一个很小的值以避免除零)，则进行归一化
            if NORMALIZE_SPECTRUM_DISPLAY and len(spec_orig_main) > 0 and np.max(spec_orig_main) > 1e-12:
                spec_orig_plot_main = spec_orig_main / np.max(spec_orig_main) # 归一化到 [0, 1]
            # 绘制原始信号的功率谱
            ax_combined_spectrum_main.plot(freq_orig_main, spec_orig_plot_main, label="原始信号频谱", color='blue', alpha=0.9)

            # 计算第一个滤波后信号的功率谱
            freq_filt_main, spec_filt_main = compute_power_spectrum(first_filtered_signal_data, FS_FILTER)
            # 准备用于绘图的滤波后信号频谱数据
            spec_filt_plot_main = spec_filt_main
            # 如果需要归一化且频谱有效，则进行归一化
            if NORMALIZE_SPECTRUM_DISPLAY and len(spec_filt_main) > 0 and np.max(spec_filt_main) > 1e-12:
                spec_filt_plot_main = spec_filt_main / np.max(spec_filt_main)
            # 绘制滤波后信号的功率谱
            ax_combined_spectrum_main.plot(freq_filt_main, spec_filt_plot_main, label="滤波后信号频谱", color='purple', alpha=0.9)

            # 在频谱图上绘制低截止频率的垂直虚线作为标记
            ax_combined_spectrum_main.axvline(LOWCUT_FILTER_SCRIPT, color='gray', linestyle=':', alpha=0.7, label=f'通带 ({LOWCUT_FILTER_SCRIPT}Hz)')
            # 在频谱图上绘制高截止频率的垂直虚线作为标记
            ax_combined_spectrum_main.axvline(HIGHCUT_FILTER_SCRIPT, color='gray', linestyle=':', alpha=0.7, label=f'通带 ({HIGHCUT_FILTER_SCRIPT}Hz)')
            # 获取图例的句柄和标签，用于处理重复标签 (确保图例中每个条目唯一)
            handles_main, labels_main = ax_combined_spectrum_main.get_legend_handles_labels()
            by_label_main = dict(zip(labels_main, handles_main)) # 创建标签到句柄的映射，自动去重
            # 显示图例，使用去重后的标签和句柄
            ax_combined_spectrum_main.legend(by_label_main.values(), by_label_main.keys(), fontsize='small')

            # 如果启用了频谱归一化显示，则固定 Y 轴范围为 [0, 1.1]
            if NORMALIZE_SPECTRUM_DISPLAY:
                ax_combined_spectrum_main.set_ylim(bottom=0, top=1.1)
            # 否则，根据频谱数据的最大值自动调整 Y 轴范围
            else:
                # 获取原始信号频谱绘图数据的最大值 (如果数组为空则为0)
                max_val_orig = np.max(spec_orig_plot_main) if len(spec_orig_plot_main) > 0 else 0
                # 获取滤波后信号频谱绘图数据的最大值 (如果数组为空则为0)
                max_val_filt = np.max(spec_filt_plot_main) if len(spec_filt_plot_main) > 0 else 0
                # 设置 Y 轴上限为两个频谱中较大最大值的1.1倍 (加上一个微小值以防最大值为0导致上限也为0)
                ax_combined_spectrum_main.set_ylim(bottom=0, top=max(max_val_orig, max_val_filt) * 1.1 + 1e-9)

            # 计算频谱图 X 轴 (频率轴) 的显示上限，以便更好地观察通带附近的特性
            # 通常设置为高截止频率加上通带宽度的一个倍数
            common_xlim_max_spectra_main = HIGHCUT_FILTER_SCRIPT + (HIGHCUT_FILTER_SCRIPT - LOWCUT_FILTER_SCRIPT) * 3
            # 确保 X 轴上限至少为 10.0 Hz，以提供足够的观察范围
            if common_xlim_max_spectra_main < 10: common_xlim_max_spectra_main = 10.0
            # 确保 X 轴上限不超过奈奎斯特频率 (FS_FILTER / 2)
            if common_xlim_max_spectra_main > FS_FILTER / 2: common_xlim_max_spectra_main = FS_FILTER / 2
            # 设置频谱图 X 轴的显示范围从0到计算得到的上限
            ax_combined_spectrum_main.set_xlim(0, common_xlim_max_spectra_main)

    # 在主图的第四个子图 (ax_filter_response_main) 上绘制滤波器的频率响应曲线
    # 设置该子图的标题和轴标签
    ax_filter_response_main.set_title("4. Butterworth 带通滤波器频率响应")
    ax_filter_response_main.set_xlabel('频率 (Hz)'); ax_filter_response_main.set_ylabel('增益'); ax_filter_response_main.grid(True)
    # 设计一个用于绘制频率响应的滤波器 (使用与信号滤波相同的参数)
    sos_for_main_plot = butter_bandpass(LOWCUT_FILTER_SCRIPT, HIGHCUT_FILTER_SCRIPT, FS_FILTER, order=ORDER_FILTER_SCRIPT)
    # 如果滤波器设计成功 (sos_for_main_plot 不是 None)
    if sos_for_main_plot is not None:
        # 计算滤波器的频率响应：sosfreqz 返回频率点 (w_freqz_main) 和对应的复数响应 (h_freqz_main)
        # worN=8000 指定计算8000个频率点；fs=FS_FILTER 指定采样频率以得到Hz单位的频率轴
        w_freqz_main, h_freqz_main = sosfreqz(sos_for_main_plot, worN=8000, fs=FS_FILTER)
        # 绘制幅频响应曲线 (取复数响应的绝对值作为增益)
        ax_filter_response_main.plot(w_freqz_main, abs(h_freqz_main), label=f"阶数 = {ORDER_FILTER_SCRIPT}", color='blue')
        # 绘制 -3dB 参考线 (增益为 1/sqrt(2) 或约 0.707)
        ax_filter_response_main.plot([0, FS_FILTER / 2], [np.sqrt(0.5), np.sqrt(0.5)], '--', color='gray', label='-3dB')
        # 绘制低截止频率标记线
        ax_filter_response_main.axvline(LOWCUT_FILTER_SCRIPT, color='green', linestyle='--', label=f'低截止 ({LOWCUT_FILTER_SCRIPT} Hz)')
        # 绘制高截止频率标记线
        ax_filter_response_main.axvline(HIGHCUT_FILTER_SCRIPT, color='red', linestyle='--', label=f'高截止 ({HIGHCUT_FILTER_SCRIPT} Hz)')
        # 计算并设置滤波器响应图 X 轴的显示范围，以便更好地观察通带和过渡带
        display_xlim_filter_max_main = max(HIGHCUT_FILTER_SCRIPT * 2.5, 15.0) # 上限
        display_xlim_filter_min_main = max(0, LOWCUT_FILTER_SCRIPT - (HIGHCUT_FILTER_SCRIPT - LOWCUT_FILTER_SCRIPT)) if LOWCUT_FILTER_SCRIPT > 0 else 0 # 下限
        # 确保下限不大于上限，并进行调整以保证合理的显示窗口
        if display_xlim_filter_min_main >= display_xlim_filter_max_main :
                display_xlim_filter_min_main = max(0, LOWCUT_FILTER_SCRIPT - 2) # 调整下限
                display_xlim_filter_max_main = HIGHCUT_FILTER_SCRIPT + 2       # 调整上限
        # 设置 X 轴范围，但不超过奈奎斯特频率
        ax_filter_response_main.set_xlim(display_xlim_filter_min_main, min(display_xlim_filter_max_main, FS_FILTER/2))
        # 设置 Y 轴范围 (增益通常在0到1.1之间)
        ax_filter_response_main.set_ylim(0, 1.1)
    # 如果滤波器设计失败
    else:
        # 在子图中央显示 "滤波器设计失败" 的文本提示
        ax_filter_response_main.text(0.5, 0.5, "滤波器设计失败", ha='center', va='center', fontsize=10, color='red')
    # 显示滤波器响应图的图例
    ax_filter_response_main.legend(fontsize='small')

    # 为原始信号时域图 (ax_original_time_main) 添加图例，位置在右上角，字体较小
    ax_original_time_main.legend(loc='upper right', fontsize='small')
    # 为滤波后信号时域图 (ax_filtered_time_main) 添加图例
    ax_filtered_time_main.legend(loc='upper right', fontsize='small')
    # 调整主图形窗口的子图布局，以防止标题、标签等元素重叠
    # rect=[0, 0, 1, 0.95] 为标题留出空间 (顶部5%)
    fig_main.tight_layout(rect=[0, 0, 1, 0.95])

    # --- 为其他信号 (如果存在多个信号) 分别创建频谱和滤波器响应的独立绘图窗口 ---
    # 检查从文件中加载的信号数量是否大于1
    if len(signals_info_list) > 1:
        # 从第二个信号开始遍历 (索引从1到 len-1)
        for i in range(1, len(signals_info_list)):
            # 获取当前遍历到的信号的信息 (字典)
            signal_info_from_file_other = signals_info_list[i]
            # 从信号信息中提取原始信号数据，并转换为 numpy 数组
            original_signal_data_other = np.array(signal_info_from_file_other["data"])
            # 从之前存储的 output_data_for_spec["processed_signals"] 列表中获取对应的滤波后数据
            # 这里依赖于信号处理和存储的顺序与原始列表中的顺序一致
            filtered_signal_data_other = np.array(output_data_for_spec["processed_signals"][i]["filtered_data"])
            # 获取当前信号的标签
            label_other = signal_info_from_file_other.get("label", f"信号 {i + 1}")

            # 打印为当前信号创建单独分析图的提示信息
            print(f"\n为信号 '{label_other}' 创建单独的分析图...")
            # 创建一个新的图形窗口 (fig_other) 和一组子图 (axes_other)，布局为2行1列
            fig_other, axes_other = plt.subplots(2, 1, figsize=(12, 10))
            # 设置新图形窗口的标题
            fig_other.suptitle(f"信号 '{label_other}' 分析 (滤波器: {LOWCUT_FILTER_SCRIPT}-{HIGHCUT_FILTER_SCRIPT}Hz, {ORDER_FILTER_SCRIPT}阶, 文件时长)", fontsize=14)

            # 获取第一个子图 (axes_other[0])，用于显示当前信号的组合频谱
            ax_spectrum_other = axes_other[0]
            # 设置频谱子图的标题和轴标签
            ax_spectrum_other.set_title(f"1. 信号 '{label_other}' 功率谱")
            ax_spectrum_other.set_xlabel("频率 (Hz)")
            ax_spectrum_other.grid(True)
            # 根据是否归一化设置 Y 轴标签
            y_label_power_other = "功率"
            if NORMALIZE_SPECTRUM_DISPLAY: y_label_power_other = "归一化功率"
            ax_spectrum_other.set_ylabel(y_label_power_other)

            # 计算当前原始信号的功率谱
            freq_orig_other, spec_orig_other = compute_power_spectrum(original_signal_data_other, FS_FILTER)
            # 准备用于绘图的原始信号频谱数据
            spec_orig_plot_other = spec_orig_other
            # 如果需要归一化且频谱有效，则进行归一化
            if NORMALIZE_SPECTRUM_DISPLAY and len(spec_orig_other) > 0 and np.max(spec_orig_other) > 1e-12:
                spec_orig_plot_other = spec_orig_other / np.max(spec_orig_other)
            # 绘制原始信号频谱
            ax_spectrum_other.plot(freq_orig_other, spec_orig_plot_other, label="原始信号频谱", color='blue', alpha=0.9)

            # 计算当前滤波后信号的功率谱
            freq_filt_other, spec_filt_other = compute_power_spectrum(filtered_signal_data_other, FS_FILTER)
            # 准备用于绘图的滤波后信号频谱数据
            spec_filt_plot_other = spec_filt_other
            # 如果需要归一化且频谱有效，则进行归一化
            if NORMALIZE_SPECTRUM_DISPLAY and len(spec_filt_other) > 0 and np.max(spec_filt_other) > 1e-12:
                spec_filt_plot_other = spec_filt_other / np.max(spec_filt_other)
            # 绘制滤波后信号频谱
            ax_spectrum_other.plot(freq_filt_other, spec_filt_plot_other, label="滤波后信号频谱", color='purple', alpha=0.9)

            # 在频谱图上绘制低截止频率和高截止频率的标记线
            ax_spectrum_other.axvline(LOWCUT_FILTER_SCRIPT, color='gray', linestyle=':', alpha=0.7, label=f'通带 ({LOWCUT_FILTER_SCRIPT}Hz)')
            ax_spectrum_other.axvline(HIGHCUT_FILTER_SCRIPT, color='gray', linestyle=':', alpha=0.7, label=f'通带 ({HIGHCUT_FILTER_SCRIPT}Hz)')
            # 获取图例子项并去重，然后显示图例
            handles_other, labels_other_leg = ax_spectrum_other.get_legend_handles_labels()
            by_label_other = dict(zip(labels_other_leg, handles_other))
            ax_spectrum_other.legend(by_label_other.values(), by_label_other.keys(), fontsize='small')

            # 根据是否归一化设置频谱图的 Y 轴范围
            if NORMALIZE_SPECTRUM_DISPLAY:
                ax_spectrum_other.set_ylim(bottom=0, top=1.1)
            else:
                max_val_orig_o = np.max(spec_orig_plot_other) if len(spec_orig_plot_other) > 0 else 0
                max_val_filt_o = np.max(spec_filt_plot_other) if len(spec_filt_plot_other) > 0 else 0
                ax_spectrum_other.set_ylim(bottom=0, top=max(max_val_orig_o, max_val_filt_o) * 1.1 + 1e-9)

            # 计算并设置频谱图的 X 轴显示范围
            common_xlim_max_spectra_other = HIGHCUT_FILTER_SCRIPT + (HIGHCUT_FILTER_SCRIPT - LOWCUT_FILTER_SCRIPT) * 3
            if common_xlim_max_spectra_other < 10: common_xlim_max_spectra_other = 10.0
            if common_xlim_max_spectra_other > FS_FILTER / 2: common_xlim_max_spectra_other = FS_FILTER / 2
            ax_spectrum_other.set_xlim(0, common_xlim_max_spectra_other)


            # 获取第二个子图 (axes_other[1])，用于显示滤波器频率响应
            ax_filter_resp_other = axes_other[1]
            # 设置滤波器响应子图的标题和轴标签
            ax_filter_resp_other.set_title("2. Butterworth 带通滤波器频率响应")
            ax_filter_resp_other.set_xlabel('频率 (Hz)'); ax_filter_resp_other.set_ylabel('增益'); ax_filter_resp_other.grid(True)
            # 设计用于当前信号图的滤波器SOS系数 (与主图的滤波器参数相同)
            sos_for_other_plot = butter_bandpass(LOWCUT_FILTER_SCRIPT, HIGHCUT_FILTER_SCRIPT, FS_FILTER, order=ORDER_FILTER_SCRIPT)
            # 如果滤波器设计成功
            if sos_for_other_plot is not None:
                # 计算并绘制频率响应曲线、-3dB线和截止频率标记线 (逻辑同主图的滤波器响应部分)
                w_freqz_other, h_freqz_other = sosfreqz(sos_for_other_plot, worN=8000, fs=FS_FILTER)
                ax_filter_resp_other.plot(w_freqz_other, abs(h_freqz_other), label=f"阶数 = {ORDER_FILTER_SCRIPT}", color='green')
                ax_filter_resp_other.plot([0, FS_FILTER / 2], [np.sqrt(0.5), np.sqrt(0.5)], '--', color='gray', label='-3dB')
                ax_filter_resp_other.axvline(LOWCUT_FILTER_SCRIPT, color='blue', linestyle='--', label=f'低截止 ({LOWCUT_FILTER_SCRIPT} Hz)')
                ax_filter_resp_other.axvline(HIGHCUT_FILTER_SCRIPT, color='red', linestyle='--', label=f'高截止 ({HIGHCUT_FILTER_SCRIPT} Hz)')

                # 计算并设置滤波器响应图的 X 轴和 Y 轴显示范围 (逻辑同主图)
                display_xlim_filter_max_other = max(HIGHCUT_FILTER_SCRIPT * 2.5, 15.0)
                display_xlim_filter_min_other = max(0, LOWCUT_FILTER_SCRIPT - (HIGHCUT_FILTER_SCRIPT - LOWCUT_FILTER_SCRIPT)) if LOWCUT_FILTER_SCRIPT > 0 else 0
                if display_xlim_filter_min_other >= display_xlim_filter_max_other :
                     display_xlim_filter_min_other = max(0, LOWCUT_FILTER_SCRIPT - 2)
                     display_xlim_filter_max_other = HIGHCUT_FILTER_SCRIPT + 2
                ax_filter_resp_other.set_xlim(display_xlim_filter_min_other, min(display_xlim_filter_max_other, FS_FILTER/2))
                ax_filter_resp_other.set_ylim(0, 1.1)
            # 如果滤波器设计失败
            else:
                # 显示设计失败的文本提示
                ax_filter_resp_other.text(0.5, 0.5, "滤波器设计失败", ha='center', va='center', fontsize=10, color='red')
            # 显示图例
            ax_filter_resp_other.legend(fontsize='small')

            # 调整当前图形窗口的子图布局
            fig_other.tight_layout(rect=[0, 0, 1, 0.95])

    # --- 绘制所有信号滤波后的扩展时域图 ---
    # 检查信号信息列表是否不为空 (即是否有信号被处理过)
    if signals_info_list:
        # 打印准备绘制扩展时域图的提示信息，包含设定的显示时长
        print(f"\n准备绘制所有信号滤波后的扩展时域图 ({EXTENDED_TIME_DURATION}s)...")
        # 创建一个新的图形窗口 (fig_extended) 和一个子图 (ax_extended)
        fig_extended, ax_extended = plt.subplots(1, 1, figsize=(12, 6))
        # 设置扩展时域图窗口的标题
        fig_extended.suptitle(f"所有信号滤波后时域 ({EXTENDED_TIME_DURATION}s, 滤波器: {LOWCUT_FILTER_SCRIPT}-{HIGHCUT_FILTER_SCRIPT}Hz, {ORDER_FILTER_SCRIPT}阶)", fontsize=14)
        # 设置扩展时域子图的标题和轴标签
        ax_extended.set_title(f"滤波后信号 (时域 - {EXTENDED_TIME_DURATION}s)")
        ax_extended.set_xlabel(f"时间 (s)")
        ax_extended.set_ylabel("幅值")
        ax_extended.grid(True) # 显示网格

        # 根据设定的扩展时长和采样频率，计算所需的数据点数
        num_points_extended = int(EXTENDED_TIME_DURATION * FS_FILTER)
        # 生成扩展时域图的时间轴 (从0到EXTENDED_TIME_DURATION，共num_points_extended个点)
        time_axis_extended = np.linspace(0, EXTENDED_TIME_DURATION, num_points_extended, endpoint=False)

        # 遍历每个原始信号信息 (用于根据标签重新生成扩展时长的信号)
        for i, signal_info_from_file_ext in enumerate(signals_info_list):
            # 获取当前信号的标签
            label_from_file_ext = signal_info_from_file_ext.get("label", f"信号 {i + 1}")

            # 使用 generate_signal_from_params 函数重新生成指定时长的原始信号
            # 这是因为原始文件中的信号时长可能与 EXTENDED_TIME_DURATION 不同
            original_signal_extended = generate_signal_from_params(time_axis_extended, label_from_file_ext, FS_FILTER)

            # 如果信号成功生成且不为空
            if original_signal_extended is not None and len(original_signal_extended) > 0 :
                # 对新生成的扩展时长信号进行滤波
                filtered_signal_extended = butter_bandpass_filter(original_signal_extended,
                                                                  LOWCUT_FILTER_SCRIPT, HIGHCUT_FILTER_SCRIPT,
                                                                  FS_FILTER, order=ORDER_FILTER_SCRIPT)
                # 在扩展时域图上绘制滤波后的信号，使用对应的颜色和标签
                ax_extended.plot(time_axis_extended, filtered_signal_extended, label=label_from_file_ext, color=colors(i), alpha=0.8)
            # 如果信号生成或滤波失败
            else:
                # 打印无法生成或滤波当前扩展信号的提示
                print(f"  无法为 '{label_from_file_ext}' 生成或滤波扩展信号。")
        # 显示扩展时域图的图例
        ax_extended.legend(loc='upper right', fontsize='small')
        # 调整扩展时域图的布局
        fig_extended.tight_layout(rect=[0, 0, 1, 0.95])
    # --- 扩展时域图绘制结束 ---

    # --- 保存滤波后的数据到文件 ---
    # 尝试将包含所有处理结果的 output_data_for_spec 字典保存到 JSON 文件
    try:
        # 以写入模式 ('w') 打开输出文件，指定编码为 'utf-8'
        with open(output_filename_for_spec, 'w', encoding='utf-8') as f_out:
            # 使用 json.dump 将字典数据写入文件
            # indent=2 参数使JSON文件具有良好的可读性 (格式化缩进2个空格)
            json.dump(output_data_for_spec, f_out, indent=2)
        # 打印数据保存成功的消息及文件路径
        print(f"\n滤波后的信号数据已保存到: {output_filename_for_spec}")
    # 捕获在保存文件过程中可能发生的任何异常
    except Exception as e:
        # 打印保存数据失败的错误信息及异常详情
        print(f"错误: 保存滤波后数据到 {output_filename_for_spec} 失败: {e}")
    # --- 文件保存结束 ---

    # 打印所有图形已准备就绪的提示信息
    print("\n所有图形已准备就绪。")
    # 显示所有 matplotlib 创建的图形窗口
    # block=False 参数使得程序在显示图形后不会阻塞，可以继续执行后续代码
    plt.show(block=False)
    # 等待用户在控制台按 Enter 键，以便在关闭图形前有时间查看
    input("按 Enter 键结束程序并关闭所有图形...")
    # 关闭所有已打开的 matplotlib 图形窗口
    plt.close('all')
    # 打印程序正常结束的提示信息
    print("程序结束。")
    # 打印模块二 (滤波器模块) 测试结束的提示信息
    print("--- 模块二测试结束 ---")