# peak_detector.py
"""
峰值检测模块
"""
import numpy as np
import json # For loading/saving data when run as script
import matplotlib.pyplot as plt # For plotting when run as script

# --- 中文显示配置 (仅当作为脚本运行时尝试设置) ---
def setup_chinese_font():
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        print("[模块四] 尝试设置字体为 SimHei")
    except Exception as e:
        print(f"[模块四] 设置字体 SimHei 失败: {e}")
# --- 中文显示配置结束 ---


def detect_peaks(power_spectrum, frequencies, threshold_ratio=0.5):
    """
    基于阈值和局部最大值条件检测功率谱中的峰值。

    参数:
    power_spectrum (numpy.ndarray): 功率谱数据。
    frequencies (numpy.ndarray): 对应的频率轴。
    threshold_ratio (float): 峰值点功率相对于最大功率的最小比例。

    返回:
    tuple: (peak_indices, detected_peak_frequencies, detected_peak_powers)
        peak_indices (list): 检测到的峰值在功率谱数组中的索引列表。
        detected_peak_frequencies (list): 检测到的峰值对应的频率值列表 (精确到小数点后两位)。
        detected_peak_powers (list): 检测到的峰值对应的功率值列表。
    """
    if not isinstance(power_spectrum, np.ndarray): power_spectrum = np.array(power_spectrum)
    if not isinstance(frequencies, np.ndarray): frequencies = np.array(frequencies)

    if len(power_spectrum) == 0 or len(frequencies) == 0:
        return [], [], []
    if len(power_spectrum) != len(frequencies):
        # raise ValueError("功率谱和频率数组长度必须一致") # In module use, better to return empty
        print("错误: detect_peaks - 功率谱和频率数组长度不一致。")
        return [], [], []


    # Handle cases where power_spectrum might be all zeros or very small
    max_power = np.max(power_spectrum) if power_spectrum.size > 0 else 0
    if max_power <= 1e-12:  # If max power is effectively zero, no peaks above threshold
        return [], [], []

    threshold = max_power * threshold_ratio

    peak_indices = []
    detected_peak_frequencies = []
    detected_peak_powers = []

    # 遍历功率谱数据点，不包括边界点（因为需要比较前后点）
    # If spectrum has less than 3 points, local max logic won't work.
    if len(power_spectrum) < 3:
        # Handle very short spectra: if the max point is above threshold, consider it a peak
        if max_power > threshold:
            idx_max = np.argmax(power_spectrum)
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
    
    # Optional: Check boundary points if they are above threshold and are local maxima
    # (considering only one neighbor)
    # Check first point
    if len(power_spectrum) >= 2 and power_spectrum[0] > threshold and power_spectrum[0] > power_spectrum[1]:
        if 0 not in peak_indices: # Avoid duplicates if already found by a different logic
            peak_indices.insert(0,0) # Add to beginning
            detected_peak_frequencies.insert(0,round(frequencies[0], 2))
            detected_peak_powers.insert(0,power_spectrum[0])
    # Check last point
    if len(power_spectrum) >= 2 and power_spectrum[-1] > threshold and power_spectrum[-1] > power_spectrum[-2]:
        if (len(power_spectrum) -1) not in peak_indices:
            peak_indices.append(len(power_spectrum)-1)
            detected_peak_frequencies.append(round(frequencies[-1], 2))
            detected_peak_powers.append(power_spectrum[-1])


    return peak_indices, detected_peak_frequencies, detected_peak_powers


if __name__ == '__main__':
    print("\n--- 峰值检测模块独立测试 (从文件读取频谱数据) ---")
    setup_chinese_font()

    input_filename = "spec_to_peak.txt"
    output_filename_for_visual = "peak_to_visual.txt"
    default_threshold_ratio = 0.1 # Default threshold for standalone testing, can be adjusted

    try:
        with open(input_filename, 'r') as f:
            loaded_data_from_spec = json.load(f)
        print(f"从 {input_filename} 加载数据成功。")
    except FileNotFoundError:
        print(f"错误: 文件 {input_filename} 未找到。程序退出。")
        exit()
    except json.JSONDecodeError:
        print(f"错误: 文件 {input_filename} JSON格式无效。程序退出。")
        exit()
    except Exception as e:
        print(f"加载数据时发生未知错误: {e}。程序退出。")
        exit()

    if not loaded_data_from_spec or "fs" not in loaded_data_from_spec or "spectrum_data" not in loaded_data_from_spec:
        print("错误: 加载的数据格式不正确。缺少 'fs' 或 'spectrum_data'。程序退出。")
        exit()

    FS_FROM_FILE = loaded_data_from_spec["fs"]
    spectrum_data_list = loaded_data_from_spec["spectrum_data"]
    filter_params_info = loaded_data_from_spec.get("filter_params", {}) # Get filter params if they exist

    print(f"采样率: {FS_FROM_FILE} Hz")
    if filter_params_info:
        print(f"源于滤波器的参数: {filter_params_info}")
    print(f"用于峰值检测的默认阈值比例: {default_threshold_ratio}")


    if not spectrum_data_list:
        print("文件中没有频谱数据。程序退出。")
        exit()

    # --- Data structure for output file peak_to_visual.txt ---
    output_data_for_visual = {
        "fs": FS_FROM_FILE,
        "filter_params": filter_params_info,
        "peak_detection_results": []
    }
    # --- End data structure ---

    output_plots = [] # To store figure objects for later showing
    
    for i, spec_info in enumerate(spectrum_data_list):
        frequencies = np.array(spec_info.get("frequencies", []))
        power_spectrum = np.array(spec_info.get("power_spectrum", []))
        original_label = spec_info.get("original_label", "N/A")
        filtered_label = spec_info.get("filtered_label", f"Spectrum {i+1}")

        if frequencies.size == 0 or power_spectrum.size == 0:
            print(f"警告: 频谱数据 '{filtered_label}' 为空，跳过峰值检测。")
            output_data_for_visual["peak_detection_results"].append({
                "original_label": original_label,
                "filtered_label": filtered_label,
                "detected_peaks": [],
                "threshold_ratio_used": default_threshold_ratio,
                "notes": "Input spectrum data was empty."
            })
            continue

        print(f"\n正在检测信号 '{filtered_label}' (源自: '{original_label}') 的峰值...")
        
        # 调用峰值检测函数
        peak_indices, detected_freqs, detected_powers = detect_peaks(
            power_spectrum, frequencies, threshold_ratio=default_threshold_ratio
        )

        print(f"  检测到 {len(detected_freqs)} 个峰值:")
        detected_peaks_for_output = []
        if detected_freqs:
            for idx, freq, power_val in zip(peak_indices, detected_freqs, detected_powers):
                print(f"    频率: {freq:.2f} Hz, 功率: {power_val:.4e}, 索引: {idx}")
                detected_peaks_for_output.append({
                    "frequency": freq,
                    "power": power_val,
                    "index_in_spectrum": int(idx) # Ensure index is int for JSON
                })
        
        # --- Add peak detection results to output structure ---
        output_data_for_visual["peak_detection_results"].append({
            "original_label": original_label,
            "filtered_label": filtered_label,
            "detected_peaks": detected_peaks_for_output,
            "threshold_ratio_used": default_threshold_ratio
        })
        # --- End add results ---

        # --- 绘图 (为每个信号的频谱及其峰值创建一个图) ---
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.plot(frequencies, power_spectrum, label=f'功率谱: {filtered_label}', color='dodgerblue', alpha=0.7)
        
        if peak_indices:
            ax.scatter(frequencies[peak_indices], power_spectrum[peak_indices], 
                       color='red', marker='o', s=80, label='检测到的峰值', zorder=5)
        
        current_max_power = np.max(power_spectrum) if power_spectrum.size > 0 else 0
        if current_max_power > 1e-12 :
            threshold_line_val = current_max_power * default_threshold_ratio
            ax.axhline(threshold_line_val, color='gray', linestyle='--', 
                       label=f'检测阈值 ({default_threshold_ratio*100:.0f}% of max_power={current_max_power:.2e})')

        title_str = f"峰值检测: {filtered_label}\n"
        if filter_params_info:
            fp_str = f"(源滤波器: {filter_params_info.get('lowcut','?')} - {filter_params_info.get('highcut','?')} Hz, {filter_params_info.get('order','?')}阶)"
            title_str += fp_str
        
        ax.set_title(title_str)
        ax.set_xlabel("频率 (Hz)")
        ax.set_ylabel("功率")
        ax.legend(loc='upper right', fontsize='small')
        ax.grid(True)
        
        # Dynamic X-axis similar to spectrum_analyzer
        significant_freq_max_plot = 0
        if current_max_power > 1e-12:
            power_threshold_plot = current_max_power * 0.001
            significant_indices_plot = np.where(power_spectrum >= power_threshold_plot)[0]
            if significant_indices_plot.size > 0:
                significant_freq_max_plot = frequencies[significant_indices_plot[-1]]
        
        if significant_freq_max_plot > 0:
            display_xlim_max_plot = min(significant_freq_max_plot * 1.5, FS_FROM_FILE / 2)
            display_xlim_max_plot = max(display_xlim_max_plot, 20) 
            display_xlim_max_plot = min(display_xlim_max_plot, FS_FROM_FILE / 2)
            ax.set_xlim(0, display_xlim_max_plot)
        else:
            ax.set_xlim(0, min(20, FS_FROM_FILE / 2))

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
            with open(output_filename_for_visual, 'w') as f_out:
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
        input("按 Enter 键关闭所有峰值检测图窗口...")
        plt.close('all')
        print("所有峰值检测图窗口已关闭。")
    else:
        print("\n没有生成任何峰值检测图。")
        
    print("--- 峰值检测模块独立测试结束 ---")