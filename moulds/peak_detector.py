# moulds/peak_detector.py
"""
峰值检测模块
"""
import numpy as np
import json
import matplotlib.pyplot as plt
import os # 用于路径操作

# --- 中文显示配置 ---
def setup_chinese_font():
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        print("[模块四] 尝试设置字体为 SimHei")
    except Exception as e:
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
    if isinstance(data, tuple) and len(data) == 2:
        power_spectrum, frequencies = data
    elif isinstance(data, np.ndarray) and frequencies_param is not None:
        power_spectrum = data
        frequencies = frequencies_param
    else:
        print("错误: detect_peaks - 输入参数 'data' 格式不正确。应为 (power_spectrum, frequencies) 元组或功率谱数组（此时需提供frequencies_param）。")
        return [], [], []

    if not isinstance(power_spectrum, np.ndarray): power_spectrum = np.array(power_spectrum)
    if not isinstance(frequencies, np.ndarray): frequencies = np.array(frequencies)

    if len(power_spectrum) == 0 or len(frequencies) == 0:
        return [], [], []
    if len(power_spectrum) != len(frequencies):
        print("错误: detect_peaks - 功率谱和频率数组长度不一致。")
        return [], [], []

    max_power = np.max(power_spectrum) if power_spectrum.size > 0 else 0
    if max_power <= 1e-12:
        return [], [], []

    threshold = max_power * threshold_ratio

    peak_indices = []
    detected_peak_frequencies = []
    detected_peak_powers = []

    if len(power_spectrum) < 3:
        if max_power > threshold: # Check if the single max point is above threshold
            idx_max = np.argmax(power_spectrum)
            # Ensure frequencies array is not empty and idx_max is valid
            if frequencies.size > idx_max :
                peak_indices.append(idx_max)
                detected_peak_frequencies.append(round(frequencies[idx_max], 2))
                detected_peak_powers.append(power_spectrum[idx_max])
        return peak_indices, detected_peak_frequencies, detected_peak_powers

    for i in range(1, len(power_spectrum) - 1):
        current_power = power_spectrum[i]
        if current_power > threshold:
            if current_power > power_spectrum[i - 1] and current_power > power_spectrum[i + 1]:
                peak_indices.append(i)
                detected_peak_frequencies.append(round(frequencies[i], 2))
                detected_peak_powers.append(current_power)
    
    # Boundary checks (optional, but good for completeness)
    if len(power_spectrum) >= 2:
        # Check first point
        if power_spectrum[0] > threshold and power_spectrum[0] > power_spectrum[1]:
            if 0 not in peak_indices:
                peak_indices.insert(0,0)
                detected_peak_frequencies.insert(0,round(frequencies[0], 2))
                detected_peak_powers.insert(0,power_spectrum[0])
        # Check last point
        if power_spectrum[-1] > threshold and power_spectrum[-1] > power_spectrum[-2]:
            last_idx = len(power_spectrum)-1
            if last_idx not in peak_indices:
                peak_indices.append(last_idx)
                detected_peak_frequencies.append(round(frequencies[last_idx], 2))
                detected_peak_powers.append(power_spectrum[last_idx])

    return peak_indices, detected_peak_frequencies, detected_peak_powers


if __name__ == '__main__':
    print("\n--- 峰值检测模块独立测试 (从文件读取频谱数据) ---")
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

    input_filename = os.path.join(intermediate_output_folder, "spec_to_peak.txt")
    output_filename_for_visual = os.path.join(intermediate_output_folder, "peak_to_visual.txt")
    # --- 路径设置结束 ---
    
    # threshold_ratio for main.py call is 0.5 (50%)
    # For standalone test, we can use a different value if needed, or stick to 0.5
    # Let's use 0.5 to align with main.py's implicit requirement for the function.
    # If a different threshold is desired for standalone testing plots, it can be set here.
    script_threshold_ratio = 0.5 # Aligned with main.py requirement for the function's default use

    try:
        with open(input_filename, 'r', encoding='utf-8') as f:
            loaded_data_from_spec = json.load(f)
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

    # 期望的键: "fs", "filter_params", "time_axis", "signals_with_spectra"
    # "signals_with_spectra" 列表中的每个元素应包含:
    # "original_label", "filtered_label", "original_signal_data", "filtered_signal_data",
    # "frequencies", "power_spectrum"
    if not loaded_data_from_spec or \
       "fs" not in loaded_data_from_spec or \
       "signals_with_spectra" not in loaded_data_from_spec or \
       "time_axis" not in loaded_data_from_spec: # Check for time_axis as well
        print(f"错误: 加载的数据格式不正确。文件 '{input_filename}' 应包含 'fs', 'time_axis', 和 'signals_with_spectra'。程序退出。")
        exit()

    FS_FROM_FILE = loaded_data_from_spec["fs"]
    signals_with_spectra_list = loaded_data_from_spec["signals_with_spectra"]
    filter_params_info = loaded_data_from_spec.get("filter_params", {})
    time_axis_from_input = loaded_data_from_spec.get("time_axis", []) # 继承时间轴

    print(f"采样率: {FS_FROM_FILE} Hz")
    if filter_params_info:
        print(f"源于滤波器的参数: {filter_params_info}")
    print(f"用于峰值检测的阈值比例 (独立测试时): {script_threshold_ratio*100:.0f}% of max power")

    if not signals_with_spectra_list:
        print("文件中没有频谱数据。程序退出。")
        exit()

    # --- Data structure for output file peak_to_visual.txt ---
    output_data_for_visual = {
        "fs": FS_FROM_FILE,
        "filter_params": filter_params_info,
        "time_axis": time_axis_from_input, # Pass through the time_axis
        "peak_detection_results": [] # Will store full data package for each signal
    }
    # --- End data structure ---

    output_plots = []
    
    for i, signal_package_from_spec in enumerate(signals_with_spectra_list):
        frequencies = np.array(signal_package_from_spec.get("frequencies", []))
        power_spectrum = np.array(signal_package_from_spec.get("power_spectrum", []))
        original_label = signal_package_from_spec.get("original_label", "N/A")
        filtered_label = signal_package_from_spec.get("filtered_label", f"Spectrum {i+1}")
        
        # Inherit original and filtered time-domain data
        original_signal_data = signal_package_from_spec.get("original_signal_data", [])
        filtered_signal_data = signal_package_from_spec.get("filtered_signal_data", [])


        if frequencies.size == 0 or power_spectrum.size == 0:
            print(f"警告: 频谱数据 '{filtered_label}' 为空，跳过峰值检测。")
            # Still save an entry for visualizer.py to find, but with empty peaks
            output_data_for_visual["peak_detection_results"].append({
                "original_label": original_label,
                "filtered_label": filtered_label,
                "original_signal_data": original_signal_data, # Pass through
                "filtered_signal_data": filtered_signal_data, # Pass through
                "spectrum_frequencies": frequencies.tolist(), # Will be empty
                "power_spectrum_data": power_spectrum.tolist(), # Will be empty
                "detected_peaks": [],
                "threshold_ratio_used": script_threshold_ratio,
                "notes": "Input spectrum data was empty for peak detection."
            })
            continue

        print(f"\n正在检测信号 '{filtered_label}' (源自: '{original_label}') 的峰值...")
        
        # 调用峰值检测函数
        # For standalone, we directly pass power_spectrum and frequencies
        peak_indices, detected_freqs, detected_powers = detect_peaks(
            (power_spectrum, frequencies), threshold_ratio=script_threshold_ratio 
            # No need for frequencies_param here as data is a tuple
        )

        print(f"  检测到 {len(detected_freqs)} 个峰值:")
        detected_peaks_for_output_file = []
        if detected_freqs:
            for idx, freq, power_val in zip(peak_indices, detected_freqs, detected_powers):
                print(f"    频率: {freq:.2f} Hz, 功率: {power_val:.4e}, 索引: {idx}")
                detected_peaks_for_output_file.append({
                    "frequency": freq,
                    "power": power_val,
                    "index_in_spectrum": int(idx)
                })
        
        # --- Add all data (inherited + new peak info) to output structure ---
        output_data_for_visual["peak_detection_results"].append({
            "original_label": original_label,
            "filtered_label": filtered_label,
            "original_signal_data": original_signal_data, # Pass through original time-domain
            "filtered_signal_data": filtered_signal_data, # Pass through filtered time-domain
            "spectrum_frequencies": frequencies.tolist(),   # Pass through spectrum frequencies
            "power_spectrum_data": power_spectrum.tolist(), # Pass through power spectrum
            "detected_peaks": detected_peaks_for_output_file,
            "threshold_ratio_used": script_threshold_ratio
            # base_freq_actual could be added here if available from signal_generator's output
            # and passed through spectrum_analyzer
        })
        # --- End add results ---

        # --- 绘图 (为每个信号的频谱及其峰值创建一个图) ---
        # (Plotting logic remains largely the same as your previous version,
        #  it will use the 'frequencies', 'power_spectrum', and 'peak_indices' calculated above)
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.plot(frequencies, power_spectrum, label=f'功率谱: {filtered_label}', color='dodgerblue', alpha=0.7)
        
        if peak_indices: # Use the direct output from detect_peaks
            # Ensure peak_indices are valid for the current frequencies and power_spectrum
            valid_peak_indices = [idx for idx in peak_indices if idx < len(frequencies) and idx < len(power_spectrum)]
            if valid_peak_indices:
                ax.scatter(frequencies[valid_peak_indices], power_spectrum[valid_peak_indices], 
                           color='red', marker='o', s=80, label='检测到的峰值', zorder=5)
        
        current_max_power = np.max(power_spectrum) if power_spectrum.size > 0 else 0
        if current_max_power > 1e-12 :
            threshold_line_val = current_max_power * script_threshold_ratio
            ax.axhline(threshold_line_val, color='gray', linestyle='--', 
                       label=f'检测阈值 ({script_threshold_ratio*100:.0f}% of max_power={current_max_power:.2e})')

        title_str = f"峰值检测: {filtered_label}\n"
        if filter_params_info:
            fp_str = f"(源滤波器: {filter_params_info.get('lowcut','?')} - {filter_params_info.get('highcut','?')} Hz, {filter_params_info.get('order','?')}阶)"
            title_str += fp_str
        
        ax.set_title(title_str)
        ax.set_xlabel("频率 (Hz)")
        ax.set_ylabel("功率")
        ax.legend(loc='upper right', fontsize='small')
        ax.grid(True)
        
        significant_freq_max_plot = 0
        if current_max_power > 1e-12 and frequencies.size > 0:
            power_threshold_plot = current_max_power * 0.001
            significant_indices_plot = np.where(power_spectrum >= power_threshold_plot)[0]
            if significant_indices_plot.size > 0:
                max_sig_idx = significant_indices_plot[-1]
                if max_sig_idx < len(frequencies): # Boundary check
                    significant_freq_max_plot = frequencies[max_sig_idx]
        
        display_xlim_max_plot = FS_FROM_FILE / 2 # Default to Nyquist
        if significant_freq_max_plot > 0:
            display_xlim_max_plot = min(significant_freq_max_plot * 1.5, FS_FROM_FILE / 2)
            display_xlim_max_plot = max(display_xlim_max_plot, 20) 
        else: # If no significant power, show a smaller default range
            display_xlim_max_plot = min(30, FS_FROM_FILE / 2)
        ax.set_xlim(0, display_xlim_max_plot)


        if current_max_power > 1e-12:
             ax.set_ylim(bottom=-0.05 * current_max_power, top=current_max_power * 1.15)
        else:
            ax.set_ylim(bottom=0, top=1.0)

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        output_plots.append(fig)
        # --- End 绘图 ---

    # --- 保存峰值检测结果到文件 peak_to_visual.txt ---
    if output_data_for_visual["peak_detection_results"]:
        try:
            with open(output_filename_for_visual, 'w', encoding='utf-8') as f_out: # Added encoding
                json.dump(output_data_for_visual, f_out, indent=2)
            print(f"\n峰值检测结果已保存到: {output_filename_for_visual}")
        except Exception as e:
            print(f"错误: 保存峰值检测结果到 {output_filename_for_visual} 失败: {e}")
    else:
        print(f"\n没有峰值检测结果可保存到 {output_filename_for_visual}。")
    # --- 文件保存结束 ---

    if output_plots:
        print(f"\n共生成 {len(output_plots)} 个峰值检测图。")
        plt.show(block=False)
        input("按 Enter 键结束程序并关闭所有图形...")
        plt.close('all')
        print("程序结束。")
    else:
        print("\n没有生成任何峰值检测图。")
        
    print("--- 峰值检测模块独立测试结束 ---")