# main.py
"""
雷达信号处理系统主程序
"""  # 模块级文档字符串，描述该文件的主要功能

# 导入numpy库，用于进行科学计算，特别是数组操作
import numpy as np
# 导入matplotlib.pyplot库，用于数据可视化和绘图
import matplotlib.pyplot as plt

# --- 模块导入 ---
# 尝试从 'moulds' 包中导入各个子模块的功能函数
try:
    # 从 moulds.signal_generator 模块导入 generate_signal 函数，用于生成模拟信号
    from moulds.signal_generator import generate_signal
    # 从 moulds.filters 模块导入 butter_bandpass_filter 函数，用于信号滤波
    from moulds.filters import butter_bandpass_filter
    # 从 moulds.spectrum_analyzer 模块导入 compute_power_spectrum 函数，用于计算信号的功率谱
    from moulds.spectrum_analyzer import compute_power_spectrum
    # 从 moulds.peak_detector 模块导入 detect_peaks 函数，用于在频谱中检测峰值
    from moulds.peak_detector import detect_peaks
    # 从 moulds.visualizer 模块导入 plot_results 函数，用于将处理结果可视化
    from moulds.visualizer import plot_results
# 捕获在导入模块过程中可能发生的 ImportError 异常
except ImportError as e:
    # 如果发生导入错误，打印错误信息，包括具体的异常内容
    print(f"错误：无法导入一个或多个模块: {e}")
    # 打印提示信息，指导用户如何解决模块路径问题
    print(
        "请确保模块文件位于 'moulds' 文件夹下，并且 main.py 与 moulds 文件夹在同一父目录下，或者 'moulds' 的父目录在 PYTHONPATH 中。")
    # 导入失败，退出程序
    exit()

# --- 中文显示配置 ---
# 尝试配置 Matplotlib 以支持中文字符的正确显示
try:
    # 设置 Matplotlib 的 sans-serif 字体列表，优先使用 'SimHei' 字体来显示中文
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 设置 Matplotlib 的 axes.unicode_minus 参数为 False，以正常显示坐标轴上的负号
    plt.rcParams['axes.unicode_minus'] = False
    # 打印一条消息，表明已尝试设置中文字体
    print("[Main] 尝试设置matplotlib中文字体为SimHei。")
# 捕获在配置中文字体过程中可能发生的任何异常
except Exception as e:
    # 如果设置失败，打印错误信息和异常内容
    print(f"[Main] 设置中文字体失败: {e}。图形标签可能显示为英文或乱码。")


# --- 中文显示配置结束 ---

# 定义主函数，包含了雷达信号处理系统的主要逻辑
def main():
    """
    雷达信号处理系统的主执行函数。
    该函数按顺序执行以下步骤：
    1. 定义实验参数。
    2. 生成原始信号。
    3. 对信号进行带通滤波。
    4. 计算滤波后信号的功率谱。
    5. 在功率谱中检测峰值。
    6. 可视化处理结果。
    """
    # --- 实验参数定义 ---
    # SIGNAL_DURATION: 信号的总持续时间，单位为秒
    SIGNAL_DURATION = 10.0
    # SYSTEM_FS: 系统的采样频率，单位为赫兹 (Hz)
    SYSTEM_FS = 100.0
    # SIGNAL_NUM_POINTS: 信号中的数据点总数，通过采样率和持续时间计算得到
    SIGNAL_NUM_POINTS = int(SYSTEM_FS * SIGNAL_DURATION)
    # BASE_FREQ: 原始信号的基准频率，单位为赫兹 (Hz)
    BASE_FREQ = 5.0
    # NOISE_MEAN: 添加到信号中的高斯噪声的平均值
    NOISE_MEAN = 0.0
    # NOISE_STD: 添加到信号中的高斯噪声的标准差
    NOISE_STD = 0.5
    # LOWCUT_FREQ: 带通滤波器的低频截止点，单位为赫兹 (Hz)
    LOWCUT_FREQ = 4.0
    # HIGHCUT_FREQ: 带通滤波器的高频截止点，单位为赫兹 (Hz)
    HIGHCUT_FREQ = 6.0
    # FILTER_ORDER: 带通滤波器的阶数，影响滤波器的陡峭程度
    FILTER_ORDER = 5
    # PEAK_THRESHOLD_RATIO: 峰值检测阈值的比例，相对于功率谱中的最大值
    PEAK_THRESHOLD_RATIO = 0.5

    # 打印系统启动横幅
    print("--- 雷达信号处理系统 ---")
    # 打印主要的实验参数
    print(f"参数: 时长={SIGNAL_DURATION}s, 点数={SIGNAL_NUM_POINTS}, 采样率={SYSTEM_FS}Hz")
    # 打印信号相关的参数
    print(f"信号: 基频={BASE_FREQ}Hz, 噪声均值={NOISE_MEAN}, 噪声标准差={NOISE_STD}")
    # 打印滤波器相关的参数
    print(f"滤波器: 通带={LOWCUT_FREQ}-{HIGHCUT_FREQ}Hz, 阶数={FILTER_ORDER}")
    # 打印峰值检测相关的参数
    print(f"峰值检测阈值比例: {PEAK_THRESHOLD_RATIO * 100:.0f}% of max power")

    # 1. 信号生成模块调用
    # 打印当前正在执行的步骤的标题
    print("\n[1. 信号生成模块]")
    # 调用 signal_generator 模块中的 generate_signal 函数生成信号
    # 参数包括持续时间、点数、基频、噪声均值和标准差
    time_axis, original_signal, fs_generated = generate_signal(
        duration=SIGNAL_DURATION,  # 传递信号持续时间
        num_points=SIGNAL_NUM_POINTS,  # 传递信号数据点数
        base_freq=BASE_FREQ,  # 传递信号基准频率
        noise_mean=NOISE_MEAN,  # 传递噪声平均值
        noise_std=NOISE_STD  # 传递噪声标准差
    )
    # 检查 generate_signal 返回的采样率 (fs_generated) 是否与系统预设的采样率 (SYSTEM_FS) 接近
    if not np.isclose(fs_generated, SYSTEM_FS):
        # 如果采样率不匹配，打印警告信息
        print(f"警告: 信号生成模块返回的采样率 {fs_generated:.2f} Hz 与系统预设采样率 {SYSTEM_FS:.2f} Hz 不符。")
    # 打印信号生成完成的信息，包括生成信号的数据点数
    print(f"  原始信号已生成 (数据点数: {len(original_signal)})")

    # 2. 滤波器模块调用
    # 打印当前正在执行的步骤的标题
    print("\n[2. 滤波器模块]")
    # 调用 filters 模块中的 butter_bandpass_filter 函数对原始信号进行滤波
    # 参数包括原始信号数据、低截止频率、高截止频率、采样率和滤波器阶数
    filtered_signal = butter_bandpass_filter(
        data=original_signal,  # 传递原始信号数据
        lowcut=LOWCUT_FREQ,  # 传递低截止频率
        highcut=HIGHCUT_FREQ,  # 传递高截止频率
        fs=SYSTEM_FS,  # 传递系统采样率
        order=FILTER_ORDER  # 传递滤波器阶数
    )
    # 打印信号滤波完成的信息，包括滤波后信号的数据点数
    print(f"  信号已滤波 (数据点数: {len(filtered_signal)})")

    # 3. 频谱分析模块调用
    # 打印当前正在执行的步骤的标题
    print("\n[3. 频谱分析模块]")
    # 调用 spectrum_analyzer 模块中的 compute_power_spectrum 函数计算滤波后信号的功率谱
    # 参数包括滤波后的信号和采样率
    frequencies, power_spectrum = compute_power_spectrum(
        signal=filtered_signal,  # 传递滤波后的信号
        fs=SYSTEM_FS  # 传递系统采样率
    )
    # 打印频谱数据计算完成的信息，包括频率点的数量
    print(f"  频域数据已计算 (频率点数: {len(frequencies)})")

    # 4. 峰值检测模块调用
    # 打印当前正在执行的步骤的标题
    print("\n[4. 峰值检测模块]")
    # 将功率谱和对应的频率组合成一个元组，作为峰值检测函数的输入
    spectrum_data_for_peaks = (power_spectrum, frequencies)
    # 调用 peak_detector 模块中的 detect_peaks 函数在功率谱中检测峰值
    # 参数包括频谱数据和峰值检测阈值比例
    peak_indices, detected_frequencies_list, detected_peak_powers = detect_peaks(
        spectrum_data_for_peaks,  # 传递频谱数据（功率谱和频率）
        threshold_ratio=PEAK_THRESHOLD_RATIO  # 传递峰值检测阈值比例
    )

    # 初始化用于在图表上显示的峰值信息字符串，默认为未检测到峰值
    peak_info_for_plot = "未检测到显著峰值"
    # 检查 detected_frequencies_list 是否包含检测到的峰值频率
    if detected_frequencies_list:
        # 如果检测到峰值，将每个频率格式化为保留两位小数的字符串
        formatted_freqs = [f"{f:.2f}" for f in detected_frequencies_list]
        # 在控制台打印检测到的峰值频率列表
        print(f"  Detected peaks at frequencies: {formatted_freqs}")
        # 更新 peak_info_for_plot 字符串，用于图表显示
        peak_info_for_plot = f"检测到的峰值频率: {', '.join(formatted_freqs)} Hz"
    # 如果没有检测到峰值
    else:
        # 在控制台打印未检测到显著峰值的信息
        print("  未检测到显著峰值。")

    # 5. 最终可视化模块调用 (综合显示)
    # 打印当前正在执行的步骤的标题
    print("\n[5. 可视化模块]")
    # 打印准备进行可视化的信息
    print("  准备最终可视化结果...")

    # 创建描述滤波器参数的字符串，用于图表标签
    filter_info_str = f"滤波器: {LOWCUT_FREQ}-{HIGHCUT_FREQ}Hz, {FILTER_ORDER}阶"
    # 创建图表的主标题字符串，包含信号的基本信息
    main_plot_label = f"基频 {BASE_FREQ}Hz 信号处理 (噪声均值 {NOISE_MEAN}, 标准差 {NOISE_STD})"
    # 将滤波器信息和检测到的峰值信息合并，形成图表的副标题
    full_filter_info_str = f"{filter_info_str}\n{peak_info_for_plot}"

    # 调用 visualizer 模块中的 plot_results 函数来绘制所有结果图表
    plot_results(
        time_axis=time_axis,  # 传递时间轴数据
        original_signal=original_signal,  # 传递原始信号数据
        filtered_signal=filtered_signal,  # 传递滤波后信号数据
        frequencies=frequencies,  # 传递频率轴数据
        power_spectrum=power_spectrum,  # 传递功率谱数据
        peak_indices=peak_indices,  # 传递检测到的峰值索引
        base_freq_actual=BASE_FREQ,  # 传递实际的基准频率 (用于图表参考)
        signal_label=main_plot_label,  # 传递图表主标题
        filter_info_str=full_filter_info_str,  # 传递图表副标题 (包含滤波器和峰值信息)
        fs_for_plot=SYSTEM_FS  # 传递用于绘图的采样率
    )

    # 打印所有处理步骤和图形创建已完成的信息
    print("\n--- 所有处理和图形创建完毕 ---")
    # 提示用户 plt.show() 将被调用以显示图形
    print("调用 plt.show() 来显示所有已创建的图形...")
    # 调用 plt.show() 来显示所有 Matplotlib 图形窗口
    plt.show()

    # 等待用户按下 Enter 键，以便在关闭图形前查看
    input("按 Enter 键结束程序并关闭所有图形...")
    # 关闭所有打开的 Matplotlib 图形窗口
    plt.close('all')
    # 打印程序已结束的信息
    print("程序结束。")


# 检查当前脚本是否作为主程序直接运行
if __name__ == "__main__":
    # 如果是主程序，则调用 main() 函数开始执行
    main()