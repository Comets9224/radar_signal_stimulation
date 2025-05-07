# spectrum_analyzer.py
"""
频谱分析模块
"""
import numpy as np
from scipy.fft import fft, fftfreq
import json
import matplotlib.pyplot as plt

# --- 中文显示配置 ---
def setup_chinese_font():
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        print("[模块三] 尝试设置字体为 SimHei")
    except Exception as e:
        print(f"[模块三] 设置字体 SimHei 失败: {e}")
# --- 中文显示配置结束 ---

def compute_power_spectrum(signal, fs):
    # (函数定义保持不变)
    N = len(signal)
    if N == 0:
        print("警告: compute_power_spectrum 接收到空信号。")
        return np.array([]), np.array([])
    yf = fft(signal)
    xf = fftfreq(N, 1 / fs)
    yf_half_abs = np.abs(yf[:N // 2 + 1])
    xf_half = xf[:N // 2 + 1]
    # Ensure Nyquist frequency is positive if N is even and N > 0
    if N > 0 and N % 2 == 0:
        if xf_half[-1] < 0:
            xf_half[-1] = -xf_half[-1] # Make it positive (abs value)

    amp_spectrum = yf_half_abs / N
    if N > 0 :
        if N % 2 == 0:
            amp_spectrum[1:N // 2] *= 2
        else:
            amp_spectrum[1:] *= 2
    power_spectrum_final = amp_spectrum ** 2
    return xf_half, power_spectrum_final


if __name__ == '__main__':
    print("\n--- 频谱分析模块独立测试 (从文件读取滤波后数据并合并绘图) ---")
    setup_chinese_font()

    input_filename = "filters_to_spec.txt"
    output_filename_for_peak = "spec_to_peak.txt"

    try:
        with open(input_filename, 'r') as f:
            loaded_data_from_filters = json.load(f)
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

    if not loaded_data_from_filters or "fs" not in loaded_data_from_filters or "filtered_signals" not in loaded_data_from_filters:
        print("错误: 加载的数据格式不正确。缺少 'fs' 或 'filtered_signals'。程序退出。")
        exit()

    FS_FROM_FILE = loaded_data_from_filters["fs"]
    filtered_signals_info = loaded_data_from_filters["filtered_signals"]
    filter_params_info = loaded_data_from_filters.get("filter_params", {})

    print(f"采样率: {FS_FROM_FILE} Hz")
    if filter_params_info:
        print(f"应用于信号的滤波器参数: {filter_params_info}")

    if not filtered_signals_info:
        print("文件中没有滤波后的信号数据。程序退出。")
        exit()

    output_data_for_peak = {
        "fs": FS_FROM_FILE,
        "filter_params": filter_params_info,
        "spectrum_data": []
    }

    fig_combined_spectra, ax_combined_spectra = plt.subplots(figsize=(14, 8))
    title_str_combined = "所有滤波后信号的功率谱\n"
    if filter_params_info:
        fp_str = f"(滤波器: {filter_params_info.get('lowcut','?')} - {filter_params_info.get('highcut','?')} Hz, {filter_params_info.get('order','?')}阶)"
        title_str_combined += fp_str
    ax_combined_spectra.set_title(title_str_combined)
    ax_combined_spectra.set_xlabel("频率 (Hz)")
    ax_combined_spectra.set_ylabel("功率")
    ax_combined_spectra.grid(True)
    # ax_combined_spectra.set_xlim(0, FS_FROM_FILE / 2) # We will adjust this later
    max_power_overall = 0
    significant_freq_max = 0 # To track the max frequency with significant power
    colors = plt.cm.get_cmap('tab10', len(filtered_signals_info))
    
    for i, signal_info in enumerate(filtered_signals_info):
        signal_data = np.array(signal_info.get("data", []))
        filtered_label = signal_info.get("label", f"Filtered Signal {i+1}")
        original_label = signal_info.get("original_label", "N/A")

        if signal_data.size == 0:
            print(f"警告: 信号 '{filtered_label}' 数据为空，跳过频谱分析。")
            continue

        print(f"\n正在分析信号: '{filtered_label}' (源自: '{original_label}')")
        
        frequencies, power_spectrum = compute_power_spectrum(signal_data, FS_FROM_FILE)

        if frequencies.size == 0:
            print(f"警告: 信号 '{filtered_label}' 的功率谱计算结果为空。")
            continue

        output_data_for_peak["spectrum_data"].append({
            "original_label": original_label,
            "filtered_label": filtered_label,
            "frequencies": frequencies.tolist(),
            "power_spectrum": power_spectrum.tolist()
        })

        ax_combined_spectra.plot(frequencies, power_spectrum, label=f"{filtered_label}", color=colors(i), alpha=0.8)
        
        # Update overall max power for Y-axis scaling
        current_max_power_in_signal = 0
        if np.any(power_spectrum): # Check if power_spectrum is not all zeros
            # Try to find max power excluding potential DC spike for better scaling
            if len(power_spectrum) > 1 and np.any(power_spectrum[1:]):
                 current_max_power_in_signal = np.max(power_spectrum[1:]) # Max excluding DC
            else:
                 current_max_power_in_signal = np.max(power_spectrum) # Fallback to overall max if only DC or all zero
        
        if current_max_power_in_signal > max_power_overall:
            max_power_overall = current_max_power_in_signal

        # Update significant frequency max for X-axis scaling
        # Find frequencies where power is above a certain threshold of the signal's own max power
        if current_max_power_in_signal > 1e-9: # Only if there's some power
            # Define a threshold, e.g., 1% of the max power in this signal
            power_threshold = current_max_power_in_signal * 0.001 # 0.1% threshold
            # Find the highest frequency where power is above this threshold
            # Ensure frequencies and power_spectrum are not empty
            if frequencies.size > 0 and power_spectrum.size > 0:
                significant_indices = np.where(power_spectrum >= power_threshold)[0]
                if significant_indices.size > 0:
                    current_sig_freq_max = frequencies[significant_indices[-1]]
                    if current_sig_freq_max > significant_freq_max:
                        significant_freq_max = current_sig_freq_max

    if filtered_signals_info:
        ax_combined_spectra.legend(loc='upper right', fontsize='small')
        
        # Adjust Y-axis
        if max_power_overall > 1e-9:
            ax_combined_spectra.set_ylim(bottom=-0.05 * max_power_overall, top=max_power_overall * 1.15) # Start slightly below 0
        else:
            ax_combined_spectra.set_ylim(bottom=0, top=1.0)

        # Adjust X-axis
        if significant_freq_max > 0:
            # Set xlim to be a bit beyond the highest significant frequency, but not exceeding Nyquist
            display_xlim_max = min(significant_freq_max * 1.5, FS_FROM_FILE / 2)
            # Ensure a minimum display range if significant_freq_max is very small
            display_xlim_max = max(display_xlim_max, 20) # e.g., always show at least up to 20Hz
            display_xlim_max = min(display_xlim_max, FS_FROM_FILE / 2) # Cap at Nyquist
            ax_combined_spectra.set_xlim(0, display_xlim_max)
        else:
            # Fallback if no significant power was found (e.g., all signals are ~zero)
            ax_combined_spectra.set_xlim(0, min(20, FS_FROM_FILE / 2)) # Default to 20Hz or Nyquist

        fig_combined_spectra.tight_layout(rect=[0, 0, 1, 0.95])
        print(f"\n已在合并图中绘制 {len(output_data_for_peak['spectrum_data'])} 个信号的频谱。")
    else:
        plt.close(fig_combined_spectra)
        print("\n没有信号数据可用于绘制合并频谱图。")

    if output_data_for_peak["spectrum_data"]:
        try:
            with open(output_filename_for_peak, 'w') as f_out:
                json.dump(output_data_for_peak, f_out, indent=2)
            print(f"\n频谱数据已保存到: {output_filename_for_peak}")
        except Exception as e:
            print(f"错误: 保存频谱数据到 {output_filename_for_peak} 失败: {e}")
    else:
        print(f"\n没有频谱数据可保存到 {output_filename_for_peak}。")

    if filtered_signals_info:
        plt.show(block=False)
        input("按 Enter 键关闭频谱图窗口...")
        plt.close('all')
        print("频谱图窗口已关闭。")
    
    print("--- 频谱分析模块独立测试结束 ---")