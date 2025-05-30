# moulds/spectrum_analyzer.py
"""
频谱分析模块
"""
import numpy as np
from scipy.fft import fft, fftfreq
import json
import matplotlib.pyplot as plt
import os # 用于路径操作

# --- 中文显示配置 ---
def setup_chinese_font():
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        print("[模块三] 尝试设置字体为 SimHei")
    except Exception as e:
        print(f"[模块三] 设置字体 SimHei 失败: {e}")
# --- 中文显示配置结束 ---

# 这个函数是模块的核心，被main.py调用
def compute_power_spectrum(signal, fs):
    """
    计算信号的功率谱。

    参数:
    signal (numpy.ndarray): 输入信号。
    fs (float): 采样频率 (Hz)。

    返回:
    tuple: (frequencies, power_spectrum)
        frequencies (numpy.ndarray): 频率轴 (0 到 Nyquist频率)。
        power_spectrum (numpy.ndarray): 对应的功率谱值。
    """
    if not isinstance(signal, np.ndarray):
        signal = np.array(signal)

    N = len(signal)
    if N == 0:
        print("警告: compute_power_spectrum 接收到空信号。")
        return np.array([]), np.array([])
    
    yf = fft(signal)
    xf = fftfreq(N, 1 / fs)

    yf_half_abs = np.abs(yf[:N // 2 + 1])
    xf_half = xf[:N // 2 + 1]
    
    # 确保奈奎斯特频率为正 (如果N为偶数且大于0)
    if N > 0 and N % 2 == 0:
        if xf_half[-1] < 0: # fftfreq 对于Nyquist点可能返回负值
            xf_half[-1] = -xf_half[-1]

    amp_spectrum = yf_half_abs / N
    if N > 0 : # 避免对空数组操作
        if N % 2 == 0:  # N is even, Nyquist is at N//2
            if N//2 > 0: # 确保 N//2 是有效索引 (N>=2)
                 amp_spectrum[1:N // 2] *= 2 # 不包括奈奎斯特频率点
        else:  # N is odd
            if len(amp_spectrum) > 1: # N>=3
                amp_spectrum[1:] *= 2
    
    power_spectrum_final = amp_spectrum ** 2
    return xf_half, power_spectrum_final


if __name__ == '__main__':
    print("\n--- 频谱分析模块独立测试 (从文件读取滤波后数据并合并绘图) ---")
    setup_chinese_font()

    # --- 路径设置 ---
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_dir = os.path.dirname(current_script_dir)
    intermediate_output_folder = os.path.join(project_root_dir, "intermediate_output")

    if not os.path.exists(intermediate_output_folder):
        try:
            os.makedirs(intermediate_output_folder)
            print(f"已创建输出文件夹: {intermediate_output_folder}")
        except OSError as e:
            print(f"错误: 创建输出文件夹 {intermediate_output_folder} 失败: {e}")
            intermediate_output_folder = "."

    input_filename = os.path.join(intermediate_output_folder, "filters_to_spec.txt")
    output_filename_for_peak = os.path.join(intermediate_output_folder, "spec_to_peak.txt")
    # --- 路径设置结束 ---

    try:
        with open(input_filename, 'r', encoding='utf-8') as f:
            loaded_data_from_filters = json.load(f)
        print(f"从 {input_filename} 加载数据成功。")
    except FileNotFoundError:
        print(f"错误: 文件 {input_filename} 未找到。程序退出。")
        exit()
    except json.JSONDecodeError as e:
        print(f"错误: 文件 {input_filename} JSON格式无效: {e}。程序退出。")
        exit()
    except Exception as e:
        print(f"加载数据时发生未知错误: {e}。程序退出。")
        exit()

    # 检查 filters_to_spec.txt 的结构
    # 期望的键: "fs", "filter_params", "time_axis", "processed_signals"
    # "processed_signals" 列表中的每个元素应包含 "original_label", "label", "original_data", "filtered_data"
    if not loaded_data_from_filters or \
       "fs" not in loaded_data_from_filters or \
       "processed_signals" not in loaded_data_from_filters or \
       "time_axis" not in loaded_data_from_filters: # 确保 time_axis 也存在
        print(f"错误: 加载的数据格式不正确。文件 '{input_filename}' 应包含 'fs', 'time_axis', 和 'processed_signals'。程序退出。")
        exit()

    FS_FROM_FILE = loaded_data_from_filters["fs"]
    # processed_signals_info 现在应该包含原始和滤波后的时域数据
    processed_signals_info = loaded_data_from_filters["processed_signals"] 
    filter_params_info = loaded_data_from_filters.get("filter_params", {})
    time_axis_from_input = loaded_data_from_filters.get("time_axis", []) # 继承时间轴

    print(f"采样率: {FS_FROM_FILE} Hz")
    if filter_params_info:
        print(f"滤波器参数 (从输入文件继承): {filter_params_info}")

    if not processed_signals_info:
        print("文件中没有已处理的信号数据。程序退出。")
        exit()

    # --- Data structure for output file spec_to_peak.txt ---
    output_data_for_peak = {
        "fs": FS_FROM_FILE,
        "filter_params": filter_params_info, # 继承滤波器参数
        "time_axis": time_axis_from_input,   # 继承时间轴
        "signals_with_spectra": []           # 将存储每个信号的完整数据包
    }
    # --- End data structure ---

    # --- 绘图设置 ---
    fig_combined_spectra, ax_combined_spectra = plt.subplots(figsize=(14, 8))
    title_str_combined = "所有滤波后信号的功率谱\n"
    if filter_params_info:
        fp_str = f"(滤波器: {filter_params_info.get('lowcut','?')} - {filter_params_info.get('highcut','?')} Hz, {filter_params_info.get('order','?')}阶)"
        title_str_combined += fp_str
    ax_combined_spectra.set_title(title_str_combined)
    ax_combined_spectra.set_xlabel("频率 (Hz)")
    ax_combined_spectra.set_ylabel("功率")
    ax_combined_spectra.grid(True)
    max_power_overall = 0
    significant_freq_max = 0
    colors = plt.cm.get_cmap('tab10', len(processed_signals_info))
    # --- 绘图设置结束 ---
    
    for i, signal_package_from_filter in enumerate(processed_signals_info):
        # 从 filters_to_spec.txt 的每个条目中获取数据
        original_label = signal_package_from_filter.get("original_label", f"原始信号 {i+1}")
        filtered_label = signal_package_from_filter.get("label", f"Filtered Signal {i+1}") # 这是 "Filtered: Original Label"
        original_data = np.array(signal_package_from_filter.get("original_data", []))
        filtered_data = np.array(signal_package_from_filter.get("filtered_data", [])) # 之前保存为 "data" 或 "filtered_data"

        if filtered_data.size == 0:
            print(f"警告: 信号 '{filtered_label}' 的滤波后数据为空，跳过频谱分析。")
            # 仍然保存条目，但频谱数据将为空
            output_data_for_peak["signals_with_spectra"].append({
                "original_label": original_label,
                "filtered_label": filtered_label,
                "original_signal_data": original_data.tolist(),
                "filtered_signal_data": filtered_data.tolist(),
                "frequencies": [],
                "power_spectrum": []
            })
            continue

        print(f"\n正在分析信号: '{filtered_label}' (源自: '{original_label}')")
        
        # 对滤波后的信号数据计算功率谱
        frequencies, power_spectrum = compute_power_spectrum(filtered_data, FS_FROM_FILE)

        if frequencies.size == 0:
            print(f"警告: 信号 '{filtered_label}' 的功率谱计算结果为空。")
            # 保存空频谱数据
            output_data_for_peak["signals_with_spectra"].append({
                "original_label": original_label,
                "filtered_label": filtered_label,
                "original_signal_data": original_data.tolist(),
                "filtered_signal_data": filtered_data.tolist(),
                "frequencies": [],
                "power_spectrum": []
            })
            continue

        # --- 将所有相关数据（包括继承的时域数据）添加到输出结构 ---
        output_data_for_peak["signals_with_spectra"].append({
            "original_label": original_label,
            "filtered_label": filtered_label,
            "original_signal_data": original_data.tolist(),   # 继承原始时域数据
            "filtered_signal_data": filtered_data.tolist(),   # 继承滤波后时域数据
            "frequencies": frequencies.tolist(),
            "power_spectrum": power_spectrum.tolist()
        })
        # --- 数据添加结束 ---

        ax_combined_spectra.plot(frequencies, power_spectrum, label=f"{filtered_label}", color=colors(i), alpha=0.8)
        
        # ... (max_power_overall 和 significant_freq_max 更新逻辑保持不变)
        current_max_power_in_signal = 0
        if np.any(power_spectrum):
            if len(power_spectrum) > 1 and np.any(power_spectrum[1:]):
                 current_max_power_in_signal = np.max(power_spectrum[1:])
            else:
                 current_max_power_in_signal = np.max(power_spectrum)
        if current_max_power_in_signal > max_power_overall:
            max_power_overall = current_max_power_in_signal
        if current_max_power_in_signal > 1e-9:
            power_threshold = current_max_power_in_signal * 0.001
            if frequencies.size > 0 and power_spectrum.size > 0:
                significant_indices = np.where(power_spectrum >= power_threshold)[0]
                if significant_indices.size > 0:
                    current_sig_freq_max = frequencies[significant_indices[-1]]
                    if current_sig_freq_max > significant_freq_max:
                        significant_freq_max = current_sig_freq_max


    # --- 图形最终化 ---
    if processed_signals_info:
        ax_combined_spectra.legend(loc='upper right', fontsize='small')
        if max_power_overall > 1e-9:
            ax_combined_spectra.set_ylim(bottom=-0.05 * max_power_overall, top=max_power_overall * 1.15)
        else:
            ax_combined_spectra.set_ylim(bottom=0, top=1.0)

        if significant_freq_max > 0:
            display_xlim_max = min(significant_freq_max * 1.5, FS_FROM_FILE / 2)
            display_xlim_max = max(display_xlim_max, 20) 
            display_xlim_max = min(display_xlim_max, FS_FROM_FILE / 2)
            ax_combined_spectra.set_xlim(0, display_xlim_max)
        else:
            ax_combined_spectra.set_xlim(0, min(20, FS_FROM_FILE / 2))
        fig_combined_spectra.tight_layout(rect=[0, 0, 1, 0.95])
        print(f"\n已在合并图中绘制 {len(output_data_for_peak['signals_with_spectra'])} 个信号的频谱。")
    else:
        plt.close(fig_combined_spectra)
        print("\n没有信号数据可用于绘制合并频谱图。")
    # --- 图形最终化结束 ---

    # --- 保存频谱数据到文件 ---
    if output_data_for_peak["signals_with_spectra"]:
        try:
            with open(output_filename_for_peak, 'w', encoding='utf-8') as f_out: # Added encoding
                json.dump(output_data_for_peak, f_out, indent=2)
            print(f"\n频谱数据已保存到: {output_filename_for_peak}")
        except Exception as e:
            print(f"错误: 保存频谱数据到 {output_filename_for_peak} 失败: {e}")
    else:
        print(f"\n没有频谱数据可保存到 {output_filename_for_peak}。")
    # --- 文件保存结束 ---

    if processed_signals_info:
        plt.show(block=False)
        input("按 Enter 键结束程序并关闭所有图形...")
        plt.close('all')
        print("程序结束。")
    
    print("--- 频谱分析模块独立测试结束 ---")