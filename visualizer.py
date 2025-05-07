# visualizer.py
"""
可视化模块
"""
import matplotlib.pyplot as plt
import numpy as np
import platform
import json

# --- 中文显示配置 ---
# (字体配置代码保持不变)
system_name = platform.system()
try:
    if system_name == "Windows":
        plt.rcParams['font.sans-serif'] = ['SimHei']
        print("[模块五-可视化配置] 检测到 Windows 系统，尝试设置字体为 SimHei")
    elif system_name == "Darwin":
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
        print("[模块五-可视化配置] 检测到 macOS 系统，尝试设置字体为 Arial Unicode MS")
    elif system_name == "Linux":
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
        print("[模块五-可视化配置] 检测到 Linux 系统，尝试设置字体为 WenQuanYi Micro Hei")
    else:
        print(f"[模块五-可视化配置] 未知的操作系统 ({system_name})，可能无法正确显示中文。")
    plt.rcParams['axes.unicode_minus'] = False
    print("[模块五-可视化配置] 设置 axes.unicode_minus 为 False")
except Exception as e:
    print(f"[模块五-可视化配置] 设置字体时发生错误: {e}")
# --- 中文显示配置结束 ---


def plot_results(time_axis, original_signal, filtered_signal,
                 frequencies, power_spectrum,
                 peak_indices=None,
                 base_freq_actual=None,
                 signal_label="信号",
                 filter_info_str="",
                 fs_for_plot=None):
    """
    绘制双子图显示信号处理结果。
    (函数定义与之前版本基本一致)
    """
    # Ensure inputs are numpy arrays
    if not isinstance(time_axis, np.ndarray): time_axis = np.array(time_axis)
    if not isinstance(original_signal, np.ndarray): original_signal = np.array(original_signal)
    if not isinstance(filtered_signal, np.ndarray): filtered_signal = np.array(filtered_signal)
    if not isinstance(frequencies, np.ndarray): frequencies = np.array(frequencies)
    if not isinstance(power_spectrum, np.ndarray): power_spectrum = np.array(power_spectrum)
    if peak_indices is not None and not isinstance(peak_indices, np.ndarray):
        peak_indices = np.array(peak_indices, dtype=int)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle(f"信号分析结果: {signal_label}\n{filter_info_str}", fontsize=14)

    # --- 子图1: 信号时域对比 ---
    can_plot_time_domain = True
    if time_axis.size == 0:
        print(f"  警告 (plot_results for '{signal_label}'): 时间轴数据缺失。")
        can_plot_time_domain = False
    
    ax1.set_title('信号时域对比')
    ax1.set_xlabel('时间 (秒)')
    ax1.set_ylabel('幅值')
    ax1.grid(True)

    plotted_time_domain = False
    if can_plot_time_domain:
        if original_signal.size > 0:
            plot_len = min(len(time_axis), len(original_signal))
            ax1.plot(time_axis[:plot_len], original_signal[:plot_len], label='原始信号', alpha=0.7, color='blue')
            plotted_time_domain = True
        # else: # Removed redundant print, main loading will warn
            # print(f"  警告 (plot_results for '{signal_label}'): 原始时域信号数据缺失。") 
        
        if filtered_signal.size > 0:
            plot_len = min(len(time_axis), len(filtered_signal))
            ax1.plot(time_axis[:plot_len], filtered_signal[:plot_len], label='滤波后信号', color='orange')
            plotted_time_domain = True
        # else: # Removed redundant print
            # print(f"  警告 (plot_results for '{signal_label}'): 滤波后时域信号数据缺失。")
        
        if plotted_time_domain:
             ax1.legend()
        elif original_signal.size == 0 and filtered_signal.size == 0 : # Only show this if both are missing
            ax1.text(0.5, 0.5, "时域信号数据缺失", ha='center', va='center', fontsize=12, color='gray')
    else:
        ax1.text(0.5, 0.5, "时间轴和/或时域信号数据缺失", ha='center', va='center', fontsize=12, color='gray')

    # --- 子图2: 功率谱图与峰值检测 ---
    ax2.set_title('功率谱分析与峰值检测')
    ax2.set_xlabel('频率 (Hz)')
    ax2.set_ylabel('功率')
    ax2.grid(True)
    can_plot_spectrum = True
    if frequencies.size == 0 or power_spectrum.size == 0:
        print(f"  警告 (plot_results for '{signal_label}'): 频谱数据不完整。")
        can_plot_spectrum = False
    elif frequencies.size != power_spectrum.size:
        print(f"  警告 (plot_results for '{signal_label}'): 频谱频率({frequencies.size})和功率谱({power_spectrum.size})数据长度不匹配。将尝试使用最短长度绘图。")
        min_len_spec = min(len(frequencies), len(power_spectrum))
        frequencies = frequencies[:min_len_spec]
        power_spectrum = power_spectrum[:min_len_spec]
        if min_len_spec == 0: can_plot_spectrum = False


    if can_plot_spectrum:
        ax2.plot(frequencies, power_spectrum, label='滤波后信号的功率谱', color='dodgerblue')

        if peak_indices is not None and len(peak_indices) > 0:
            valid_mask = (peak_indices >= 0) & (peak_indices < len(frequencies)) & (peak_indices < len(power_spectrum))
            valid_indices = peak_indices[valid_mask]
            if len(valid_indices) > 0:
                valid_peak_freqs = frequencies[valid_indices]
                valid_peak_powers = power_spectrum[valid_indices]
                ax2.scatter(valid_peak_freqs, valid_peak_powers,
                            color='red', marker='o', s=80, label='检测到的峰值', zorder=5)
                for freq_val, power_val in zip(valid_peak_freqs, valid_peak_powers):
                    ax2.annotate(f'{freq_val:.2f} Hz\n({power_val:.2e})',
                                 (freq_val, power_val), textcoords="offset points",
                                 xytext=(0, 10), ha='center', color='red', fontsize=9)
            elif len(peak_indices) > 0:
                print(f"  信息 (plot_results for '{signal_label}'): 提供的峰值索引无效或越界。")
        # else: # Removed redundant print
            # print(f"  信息 (plot_results for '{signal_label}'): 未提供峰值索引或峰值列表为空。")


        if base_freq_actual is not None:
            ax2.axvline(base_freq_actual, color='green', linestyle='--', alpha=0.6,
                        label=f'原始基频 ({base_freq_actual:.2f} Hz)')

        nyquist_freq = fs_for_plot / 2 if fs_for_plot else (frequencies[-1] if frequencies.size > 0 else 50)
        max_power_in_spectrum = np.max(power_spectrum) if power_spectrum.size > 0 else 0
        
        significant_freq_max_plot = 0
        if max_power_in_spectrum > 1e-12 and frequencies.size > 0:
            power_threshold_plot = max_power_in_spectrum * 0.001
            significant_indices_plot = np.where(power_spectrum >= power_threshold_plot)[0]
            if significant_indices_plot.size > 0:
                max_sig_index = significant_indices_plot[-1]
                if max_sig_index < len(frequencies):
                    significant_freq_max_plot = frequencies[max_sig_index]
        
        if significant_freq_max_plot > 0:
            display_xlim_max_plot = min(significant_freq_max_plot * 1.8, nyquist_freq)
            display_xlim_max_plot = max(display_xlim_max_plot, 20 if nyquist_freq > 20 else nyquist_freq)
            ax2.set_xlim(-0.5, display_xlim_max_plot)
        else:
            ax2.set_xlim(-0.5, min(30, nyquist_freq))
        
        if max_power_in_spectrum > 1e-12:
            ax2.set_ylim(bottom=-0.05 * max_power_in_spectrum, top=max_power_in_spectrum * 1.15)
        else:
            ax2.set_ylim(bottom=0, top=1.0)
        
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, "频谱数据不足或缺失", ha='center', va='center', fontsize=12, color='gray')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


def prepare_visualization_data_from_files(
    file_signal_gen_path="signal_to_filters.txt",
    file_filters_out_path="filters_to_spec.txt",
    file_spectrum_out_path="spec_to_peak.txt",
    file_peaks_out_path="peak_to_visual.txt"
):
    """
    从多个文件中加载并整合数据，为可视化做准备。
    返回一个包含全局信息和每个信号完整数据集的字典。
    """
    print("\n--- 开始从文件整合可视化数据 ---")
    _all_data_loaded_successfully = True

    def _load_json_file(filename, description):
        nonlocal _all_data_loaded_successfully
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"从 {filename} ({description}) 加载数据成功。")
            return data
        except FileNotFoundError:
            print(f"错误: 文件 {filename} ({description}) 未找到。")
            _all_data_loaded_successfully = False
            return None
        except json.JSONDecodeError as e:
            print(f"错误: 文件 {filename} ({description}) JSON格式无效: {e}")
            _all_data_loaded_successfully = False
            return None
        except Exception as e:
            print(f"加载 {filename} ({description}) 时发生未知错误: {e}")
            _all_data_loaded_successfully = False
            return None

    data_sig_gen = _load_json_file(file_signal_gen_path, "原始信号")
    data_filters = _load_json_file(file_filters_out_path, "滤波后时域信号")
    data_spectrum = _load_json_file(file_spectrum_out_path, "频谱数据")
    data_peaks = _load_json_file(file_peaks_out_path, "峰值检测结果")

    if not _all_data_loaded_successfully:
        return None # Indicate failure

    # --- 提取全局信息 ---
    fs_global = data_sig_gen.get("fs")
    if fs_global is None:
        print(f"错误: 未能从 '{file_signal_gen_path}' 获取 'fs' (采样率)。")
        return None
        
    time_axis_global = np.array(data_sig_gen.get("time_axis", []))
    if time_axis_global.size == 0:
         print(f"警告: '{file_signal_gen_path}' 中缺少 'time_axis' 或时间轴为空。")
         if data_filters: time_axis_global = np.array(data_filters.get("time_axis", []))
         if time_axis_global.size == 0:
              print(f"警告: '{file_filters_out_path}' 中也缺少 'time_axis'。")

    filter_params_global = data_filters.get("filter_params", {}) if data_filters else {}
    
    # --- 构建数据映射 ---
    filtered_signal_map = {item["original_label"]: item for item in data_filters.get("filtered_signals", []) if "original_label" in item} if data_filters else {}
    spectrum_map = {item["original_label"]: item for item in data_spectrum.get("spectrum_data", []) if "original_label" in item} if data_spectrum else {}
    peaks_map = {item["original_label"]: item for item in data_peaks.get("peak_detection_results", []) if "original_label" in item} if data_peaks else {}

    # --- 整合每个信号的数据 ---
    consolidated_signals_data = []
    if not (data_sig_gen and "signals" in data_sig_gen):
        print(f"错误: '{file_signal_gen_path}' 中缺少 'signals' 列表。")
        return None

    for sig_gen_entry in data_sig_gen["signals"]:
        original_label = sig_gen_entry.get("label")
        if not original_label:
            continue

        single_signal_package = {
            "original_label": original_label,
            "original_signal_data": sig_gen_entry.get("data", []),
            "base_freq_actual": sig_gen_entry.get("base_freq"),
            "filtered_signal_data": [],
            "filtered_label": f"Filtered: {original_label}", # Default
            "spectrum_frequencies": [],
            "power_spectrum_data": [],
            "peak_indices": []
        }

        filt_entry = filtered_signal_map.get(original_label)
        if filt_entry:
            single_signal_package["filtered_signal_data"] = filt_entry.get("data", [])
            single_signal_package["filtered_label"] = filt_entry.get("label", single_signal_package["filtered_label"])

        spec_entry = spectrum_map.get(original_label)
        if spec_entry:
            single_signal_package["spectrum_frequencies"] = spec_entry.get("frequencies", [])
            single_signal_package["power_spectrum_data"] = spec_entry.get("power_spectrum", [])
            single_signal_package["filtered_label"] = spec_entry.get("filtered_label", single_signal_package["filtered_label"]) # Update if more specific

        peak_entry = peaks_map.get(original_label)
        if peak_entry:
            detected_peaks_info = peak_entry.get("detected_peaks", [])
            single_signal_package["peak_indices"] = [p.get("index_in_spectrum") for p in detected_peaks_info if p.get("index_in_spectrum") is not None]
            single_signal_package["filtered_label"] = peak_entry.get("filtered_label", single_signal_package["filtered_label"]) # Update if more specific
        
        consolidated_signals_data.append(single_signal_package)
        
    return {
        "fs": fs_global,
        "time_axis": time_axis_global,
        "filter_params": filter_params_global,
        "signals_to_plot": consolidated_signals_data
    }


if __name__ == '__main__':
    # 字体设置已在模块导入时（文件顶部）完成。
    
    visualization_data_package = prepare_visualization_data_from_files()

    if visualization_data_package is None:
        print("数据准备失败，无法进行可视化。")
        exit()

    FS_FROM_FILE = visualization_data_package["fs"]
    time_axis_from_file = visualization_data_package["time_axis"]
    filter_params_info = visualization_data_package["filter_params"]
    signals_to_plot_list = visualization_data_package["signals_to_plot"]

    filter_info_str_global = ""
    if filter_params_info:
        fp_str = f"源滤波器: {filter_params_info.get('lowcut','?')} - {filter_params_info.get('highcut','?')} Hz, {filter_params_info.get('order','?')}阶"
        filter_info_str_global = fp_str
    
    output_plots = []

    if not signals_to_plot_list:
        print("没有整合到可用于绘图的信号数据。")
    else:
        for i, signal_data_package in enumerate(signals_to_plot_list):
            print(f"\n正在为信号 '{signal_data_package['original_label']}' 生成可视化图...")

            fig = plot_results(
                time_axis=time_axis_from_file,
                original_signal=signal_data_package["original_signal_data"],
                filtered_signal=signal_data_package["filtered_signal_data"],
                frequencies=signal_data_package["spectrum_frequencies"],
                power_spectrum=signal_data_package["power_spectrum_data"],
                peak_indices=signal_data_package["peak_indices"],
                base_freq_actual=signal_data_package["base_freq_actual"],
                signal_label=signal_data_package["filtered_label"], # Use the determined filtered label
                filter_info_str=filter_info_str_global,
                fs_for_plot=FS_FROM_FILE
            )
            if fig:
                output_plots.append(fig)

    if output_plots:
        print(f"\n共生成 {len(output_plots)} 个可视化图。")
        plt.show(block=False)
        input("按 Enter 键关闭所有可视化图窗口...")
        plt.close('all')
        print("所有可视化图窗口已关闭。")
    else:
        print("\n没有生成任何可视化图。")
        
    print("--- 可视化模块独立测试结束 ---")