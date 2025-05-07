# main.py
"""
雷达信号处理系统主程序
"""
import numpy as np
import matplotlib.pyplot as plt # 导入matplotlib用于中间绘图

# 从各模块导入函数
from filters import butter_bandpass_filter
from spectrum_analyzer import compute_power_spectrum
from peak_detector import detect_peaks
from visualizer import plot_results # 用于最终的综合可视化

# --- 中文显示配置 ---
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    print("[Main] 尝试设置matplotlib中文字体为SimHei。")
except Exception as e:
    print(f"[Main] 设置中文字体失败: {e}。图形标签可能显示为英文或乱码。")
# --- 中文显示配置结束 ---

def create_intermediate_signal_figure(time_axis, signal_data, title="Intermediate Signal", figure_num=None, ylim_override=None):
    """辅助函数：创建但不立即显示单个时域信号图"""
    plt.figure(num=figure_num, figsize=(10, 4)) # 使用 figure_num 可以给窗口一个标识
    plt.plot(time_axis, signal_data)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    if ylim_override: # 如果提供了Y轴范围，就使用它
        plt.ylim(ylim_override)
    else: # 否则，让matplotlib自动调整
        # 稍微改进自动调整，避免因极小波动导致Y轴范围过小
        min_val, max_val = np.min(signal_data), np.max(signal_data)
        if np.isfinite(min_val) and np.isfinite(max_val):
            if not np.isclose(min_val, max_val): # 如果不是一条直线
                padding = (max_val - min_val) * 0.1
                plt.ylim(min_val - padding, max_val + padding)
            else: # 如果是一条直线，给一个小的默认范围
                plt.ylim(min_val - 0.5, max_val + 0.5)
        # else: pass # 如果数据有问题（如NaN, Inf），让matplotlib默认处理或报错
    plt.grid(True)
    # 不调用 plt.show()

def create_intermediate_spectrum_figure(frequencies, power_spectrum_data, title="Intermediate Power Spectrum", figure_num=None):
    """辅助函数：创建但不立即显示单个功率谱图"""
    plt.figure(num=figure_num, figsize=(10, 4))
    plt.plot(frequencies, power_spectrum_data)
    plt.title(title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power")
    if len(frequencies) > 0 :
        # 稍微调整X轴范围，确保至少显示到20Hz，或者奈奎斯特频率
        display_xlim_max = frequencies[-1]
        if display_xlim_max < 20 and frequencies[0] <=20 : # 如果奈奎斯特小于20，且0Hz在范围内
             display_xlim_max = 20
        elif frequencies[0] > display_xlim_max : # 异常情况
             display_xlim_max = 20

        plt.xlim(0, display_xlim_max)
    plt.ylim(bottom=0)
    plt.grid(True)
    # 不调用 plt.show()

def main():
    # --- 实验参数定义 ---
    SYSTEM_FS = 1000.0
    SIGNAL_DURATION = 1.0
    SIGNAL_NUM_POINTS = int(SYSTEM_FS * SIGNAL_DURATION)
    BASE_FREQ = 5.0
    NOISE_STD = 0.5
    LOWCUT_FREQ = 4.0
    HIGHCUT_FREQ = 6.0
    FILTER_ORDER = 5
    PEAK_THRESHOLD_RATIO = 0.5

    print("--- 雷达信号处理系统 ---")
    print(f"系统采样率: {SYSTEM_FS} Hz, 信号时长: {SIGNAL_DURATION}s, 点数: {SIGNAL_NUM_POINTS}")

    # 1. 信号生成模块
    print("\n[用户] 请求生成雷达信号...")
    time_axis = np.linspace(0, SIGNAL_DURATION, SIGNAL_NUM_POINTS, endpoint=False)
    clean_component = np.sin(2 * np.pi * BASE_FREQ * time_axis)
    noise_component = np.random.normal(0, NOISE_STD, SIGNAL_NUM_POINTS)
    original_signal = clean_component + noise_component
    print(f"[雷达信号生成模块] 信号已生成 (数据点数: {len(original_signal)})")
    create_intermediate_signal_figure(time_axis, original_signal, title="步骤1: 生成的原始信号", figure_num=1)


    # 2. 滤波器模块
    print("\n[用户] 请求对信号进行滤波...")
    filtered_signal = butter_bandpass_filter(
        data=original_signal,
        lowcut=LOWCUT_FREQ,
        highcut=HIGHCUT_FREQ,
        fs=SYSTEM_FS,
        order=FILTER_ORDER
    )
    print(f"[滤波器模块] 信号已滤波 (数据点数: {len(filtered_signal)})")
    # 第一次绘图，显示完整范围（包括瞬态）
    create_intermediate_signal_figure(time_axis, filtered_signal, title="步骤2a: 滤波后的信号 (完整范围)", figure_num=2)
    # 第二次绘图，放大稳态部分
    create_intermediate_signal_figure(time_axis, filtered_signal, title="步骤2b: 滤波后的信号 (稳态放大)", figure_num=20, ylim_override=(-1.5, 1.5)) # 使用不同的figure_num, Y轴范围可调整


    # 3. 频谱分析模块
    print("\n[用户] 请求对滤波后信号进行频谱分析...")
    frequencies, power_spectrum = compute_power_spectrum(
        signal=filtered_signal, # 使用滤波后的信号
        fs=SYSTEM_FS
    )
    print(f"[频谱分析模块] 频域数据已计算 (频率点数: {len(frequencies)})")
    create_intermediate_spectrum_figure(frequencies, power_spectrum, title="步骤3: 滤波后信号的功率谱", figure_num=3)


    # 4. 峰值检测模块
    print("\n[用户] 请求在功率谱上进行峰值检测...")
    peak_indices, detected_frequencies = detect_peaks(
        power_spectrum=power_spectrum,
        frequencies=frequencies,
        threshold_ratio=PEAK_THRESHOLD_RATIO
    )
    if detected_frequencies:
        formatted_freqs = [f"{f:.2f}" for f in detected_frequencies]
        print(f"[检测模块] 检测结果: 检测到的峰值位于 {', '.join(formatted_freqs)} Hz")
    else:
        print("[检测模块] 检测结果: 未检测到峰值")


    # 5. 最终可视化模块 (综合显示)
    print("\n[用户] 请求显示最终的综合处理结果...")
    print("[显示模块] 准备最终可视化结果...")
    # 调用 plot_results，它会创建自己的图形对象
    # 确保 visualizer.py 中的 plot_results 函数末尾的 plt.show() 被注释掉或移除
    plot_results(
        time_axis=time_axis,
        original_signal=original_signal,
        filtered_signal=filtered_signal,
        frequencies=frequencies,
        power_spectrum=power_spectrum,
        peak_indices=peak_indices,
        detected_peak_frequencies=detected_frequencies,
        base_freq_actual=BASE_FREQ
    )

    print("\n--- 所有处理和图形创建完毕 ---")
    print("调用 plt.show() 来显示所有已创建的图形...")
    plt.show() # 在所有绘图命令之后，统一调用一次，显示所有打开的 figure

    print("--- 程序结束 (所有图形窗口已响应 plt.show()) ---")


if __name__ == "__main__":
    main()