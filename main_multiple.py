# main_multiple.py
"""
主程序脚本 - 交互式生成多个独立信号，分别处理并在同一图形窗口的不同子图中可视化
"""
import numpy as np
import matplotlib.pyplot as plt
import json # 虽然在此版本中未使用，但保留以备将来扩展
import os   # 虽然在此版本中未使用，但保留以备将来扩展

# --- 从模块导入函数 ---
try:
    from moulds.signal_generator import generate_signal
    from moulds.filters import butter_bandpass_filter
    from moulds.spectrum_analyzer import compute_power_spectrum
    from moulds.peak_detector import detect_peaks
    # visualizer.plot_results 仍然有用，但我们会直接在main中构建子图
    from moulds.visualizer import plot_results # 或者只导入其绘图的辅助逻辑（如果拆分了）
except ImportError as e:
    print(f"错误：无法导入一个或多个模块: {e}")
    exit()

# --- 中文显示配置 ---
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    print("[Main_Multiple] 尝试设置matplotlib中文字体为SimHei。")
except Exception as e:
    print(f"[Main_Multiple] 设置中文字体失败: {e}。")
# --- 中文显示配置结束 ---

# ==============================================================================
# --- 固定信号生成参数 (可在脚本开头修改) ---
# ==============================================================================
SIGNAL_DURATION = 1.0
NUM_SAMPLES = 1000
SYSTEM_FS = NUM_SAMPLES / SIGNAL_DURATION
SIGNAL_AMPLITUDE = 1.0 # 固定信号分量的幅度 (假设 generate_signal 内部使用此值或类似概念)

# --- 初始默认处理参数 (交互式输入时作为起点) ---
DEFAULT_BASE_FREQ = 5.0
DEFAULT_LOWCUT_FREQ = 4.0 # 滤波参数保持固定
DEFAULT_HIGHCUT_FREQ = 6.0 # 滤波参数保持固定
DEFAULT_FILTER_ORDER = 5
DEFAULT_PEAK_THRESHOLD_RATIO = 0.5
DEFAULT_SNR_LINEAR = 10.0 # 默认信噪比 (线性值, P_signal / P_noise)
DEFAULT_NOISE_MEAN = 0.0
# ==============================================================================

def get_interactive_float(prompt_message, default_value, min_val=None, max_val=None):
    """辅助函数获取浮点数输入，带默认值和范围检查"""
    while True:
        user_str = input(prompt_message)
        if not user_str:
            return default_value
        try:
            val = float(user_str)
            if (min_val is not None and val < min_val) or \
               (max_val is not None and val > max_val):
                print(f"  输入值超出允许范围 ({min_val if min_val is not None else '-inf'} to {max_val if max_val is not None else 'inf'})。请重试或按回车使用默认值。")
                continue
            return val
        except ValueError:
            print("  输入无效，请输入数字。请重试或按回车使用默认值。")

def get_interactive_int(prompt_message, default_value, min_val=None, max_val=None):
    """辅助函数获取整数输入，带默认值和范围检查"""
    while True:
        user_str = input(prompt_message)
        if not user_str:
            return default_value
        try:
            val = int(user_str)
            if (min_val is not None and val < min_val) or \
               (max_val is not None and val > max_val):
                print(f"  输入值超出允许范围 ({min_val if min_val is not None else '-inf'} to {max_val if max_val is not None else 'inf'})。请重试或按回车使用默认值。")
                continue
            return val
        except ValueError:
            print("  输入无效，请输入整数。请重试或按回车使用默认值。")

def main():
    print("--- 交互式多独立信号处理与统一多子图可视化 ---")
    print(f"固定信号参数: 时长={SIGNAL_DURATION}s, 点数={NUM_SAMPLES}, 采样率={SYSTEM_FS}Hz, 信号分量幅度={SIGNAL_AMPLITUDE}")
    print(f"固定滤波器通带: {DEFAULT_LOWCUT_FREQ}-{DEFAULT_HIGHCUT_FREQ}Hz")
    print(f"初始默认值: 基频={DEFAULT_BASE_FREQ}Hz, 滤波阶数={DEFAULT_FILTER_ORDER}, 检测阈值比例={DEFAULT_PEAK_THRESHOLD_RATIO*100:.0f}%, 信噪比(线性)={DEFAULT_SNR_LINEAR}")

    time_axis_common = np.linspace(0, SIGNAL_DURATION, NUM_SAMPLES, endpoint=False)
    
    processed_signal_data_list = []
    keep_adding_signals = True
    signal_counter = 1

    # 用于递增的当前默认值
    current_default_base_freq = DEFAULT_BASE_FREQ
    current_default_filter_order = DEFAULT_FILTER_ORDER
    current_default_peak_threshold = DEFAULT_PEAK_THRESHOLD_RATIO
    current_default_snr = DEFAULT_SNR_LINEAR

    while keep_adding_signals:
        print(f"\n--- 请输入第 {signal_counter} 个独立信号的参数 ---")
        
        base_freq = get_interactive_float(
            f"  基频 (Hz) [默认 {current_default_base_freq:.1f}]: ",
            current_default_base_freq, min_val=0.1
        )
        
        filter_order_current = get_interactive_int(
            f"  滤波器阶数 [默认 {current_default_filter_order}]: ",
            current_default_filter_order, min_val=1
        )

        peak_threshold_ratio_current = get_interactive_float(
            f"  峰值检测阈值比例 (0.0-1.0) [默认 {current_default_peak_threshold:.2f}]: ",
            current_default_peak_threshold, min_val=0.0, max_val=1.0
        )
        
        snr_linear_current = get_interactive_float(
            f"  信噪比 (线性 P_sig/P_noise) [默认 {current_default_snr:.1f}]: ",
            current_default_snr, min_val=0.01 # SNR不能为0或负
        )

        # 根据SNR计算noise_std
        if snr_linear_current <= 0:
            print("  信噪比必须为正。使用默认噪声标准差。")
            # Fallback noise_std if SNR is invalid, can be a fixed value or based on a default SNR
            noise_std = SIGNAL_AMPLITUDE / np.sqrt(2 * DEFAULT_SNR_LINEAR) 
            print(f"  使用默认SNR ({DEFAULT_SNR_LINEAR}) 计算噪声标准差。")
        else:
            # SNR = (SIGNAL_AMPLITUDE^2 / 2) / noise_std^2 (for sine wave)
            # noise_std^2 = (SIGNAL_AMPLITUDE^2 / 2) / SNR
            # noise_std = SIGNAL_AMPLITUDE / sqrt(2 * SNR)
            noise_std = SIGNAL_AMPLITUDE / np.sqrt(2 * snr_linear_current)
        
        noise_mean = get_interactive_float( # 噪声均值仍然可以独立设置
            f"  噪声均值 [默认 {DEFAULT_NOISE_MEAN:.1f}]: ",
            DEFAULT_NOISE_MEAN
        )

        current_signal_label_parts = [
            f"F={base_freq:.1f}Hz",
            f"Amp={SIGNAL_AMPLITUDE}", # Amp refers to the global SIGNAL_AMPLITUDE
            f"SNR={snr_linear_current:.1f}",
            f"FiltOrd={filter_order_current}",
            f"Thresh={peak_threshold_ratio_current:.2f}"
        ]
        current_signal_label = f"原始信号 {signal_counter}: " + ", ".join(current_signal_label_parts)
        print(f"  生成信号: {current_signal_label}")
        print(f"  (计算得到 噪声标准差: {noise_std:.3f} for SNR={snr_linear_current:.1f} and Amp={SIGNAL_AMPLITUDE})")

        # 1. 信号生成
        # MODIFIED: Removed 'amplitude' keyword argument as it's not expected by generate_signal
        _, original_signal, _ = generate_signal(
            duration=SIGNAL_DURATION,
            num_points=NUM_SAMPLES,
            base_freq=base_freq,
            # amplitude=SIGNAL_AMPLITUDE, # This line caused TypeError
            noise_mean=noise_mean,
            noise_std=noise_std
        )

        # 2. 滤波处理
        filtered_signal = butter_bandpass_filter(
            data=original_signal,
            lowcut=DEFAULT_LOWCUT_FREQ, # 滤波通带固定
            highcut=DEFAULT_HIGHCUT_FREQ, # 滤波通带固定
            fs=SYSTEM_FS,
            order=filter_order_current # 使用交互式阶数
        )

        # 3. 频谱分析
        frequencies, power_spectrum = compute_power_spectrum(
            signal=filtered_signal,
            fs=SYSTEM_FS
        )

        # 4. 峰值检测
        spectrum_data_for_peaks = (power_spectrum, frequencies)
        peak_indices, detected_frequencies_list, _ = detect_peaks(
            spectrum_data_for_peaks,
            threshold_ratio=peak_threshold_ratio_current # 使用交互式阈值
        )
        
        peak_info_str = "未检测到显著峰值"
        if detected_frequencies_list:
            formatted_freqs = [f"{f:.2f}" for f in detected_frequencies_list]
            peak_info_str = f"检测峰值: {', '.join(formatted_freqs)} Hz"
        print(f"  处理完毕. {peak_info_str}")

        processed_signal_data_list.append({
            "original_label": current_signal_label,
            "filtered_label": f"已处理: {', '.join(current_signal_label_parts)}",
            "time_axis": time_axis_common,
            "original_signal": original_signal,
            "filtered_signal": filtered_signal,
            "frequencies": frequencies,
            "power_spectrum": power_spectrum,
            "peak_indices": peak_indices,
            "base_freq_actual": base_freq,
            "filter_order_actual": filter_order_current,
            "peak_threshold_actual": peak_threshold_ratio_current,
            "snr_actual": snr_linear_current,
            "noise_std_actual": noise_std,
            "peak_info_str": peak_info_str
        })
        
        # 更新下一个信号的默认值
        current_default_base_freq = round(base_freq + 2.0, 1) # 递增基频
        current_default_filter_order = filter_order_current + 1 if filter_order_current < 10 else DEFAULT_FILTER_ORDER # 递增阶数，可设上限
        current_default_peak_threshold = round(min(peak_threshold_ratio_current + 0.05, 0.9), 2) # 递增阈值
        current_default_snr = round(snr_linear_current + 5.0, 1) # 递增SNR
        
        signal_counter += 1
        if len(processed_signal_data_list) >= 1:
            add_another = input("是否添加并处理下一个独立信号? (y/n) [y]: ").lower()
            if add_another == 'n':
                keep_adding_signals = False
    
    if not processed_signal_data_list:
        print("没有生成和处理任何信号，程序退出。")
        return

    # --- 5. 统一可视化 ---
    num_signals = len(processed_signal_data_list)
    fig, axes = plt.subplots(num_signals, 2, figsize=(18, 6 * num_signals), squeeze=False)
    
    fig_suptitle = (f"多信号独立处理结果\n"
                    f"固定参数: 信号时长={SIGNAL_DURATION}s, 幅度={SIGNAL_AMPLITUDE}, "
                    f"采样率={SYSTEM_FS}Hz, 滤波通带={DEFAULT_LOWCUT_FREQ}-{DEFAULT_HIGHCUT_FREQ}Hz")
    fig.suptitle(fig_suptitle, fontsize=16)

    for idx, data_package in enumerate(processed_signal_data_list):
        ax_time = axes[idx, 0]
        ax_freq = axes[idx, 1]

        signal_params_info = (
            f"F={data_package['base_freq_actual']:.1f}Hz, "
            f"SNR={data_package['snr_actual']:.1f} (NoiseStd={data_package['noise_std_actual']:.3f}), "
            f"FiltOrd={data_package['filter_order_actual']}, "
            f"PeakThresh={data_package['peak_threshold_actual']:.2f}"
        )

        ax_time.set_title(f"信号 {idx+1} 时域: {signal_params_info}")
        ax_time.set_xlabel('时间 (秒)')
        ax_time.set_ylabel('幅值')
        ax_time.grid(True)
        if data_package['time_axis'].size > 0:
            if data_package['original_signal'].size > 0:
                plot_len = min(len(data_package['time_axis']), len(data_package['original_signal']))
                ax_time.plot(data_package['time_axis'][:plot_len], data_package['original_signal'][:plot_len], label='原始信号', alpha=0.7)
            if data_package['filtered_signal'].size > 0:
                plot_len = min(len(data_package['time_axis']), len(data_package['filtered_signal']))
                ax_time.plot(data_package['time_axis'][:plot_len], data_package['filtered_signal'][:plot_len], label='滤波后信号', color='orange')
            if data_package['original_signal'].size > 0 or data_package['filtered_signal'].size > 0:
                ax_time.legend(fontsize='small')
        else:
            ax_time.text(0.5, 0.5, "时域数据缺失", ha='center', va='center')

        ax_freq.set_title(f"信号 {idx+1} 功率谱: {data_package['peak_info_str']}")
        ax_freq.set_xlabel('频率 (Hz)')
        ax_freq.set_ylabel('功率')
        ax_freq.grid(True)
        
        current_frequencies = data_package['frequencies']
        current_power_spectrum = data_package['power_spectrum']
        current_peak_indices = np.array(data_package['peak_indices'], dtype=int) if data_package['peak_indices'] else np.array([], dtype=int)

        if current_frequencies.size > 0 and current_power_spectrum.size > 0 and \
           len(current_frequencies) == len(current_power_spectrum):
            ax_freq.plot(current_frequencies, current_power_spectrum, label='功率谱', color='dodgerblue')
            if current_peak_indices.size > 0:
                valid_mask = (current_peak_indices >= 0) & (current_peak_indices < len(current_frequencies))
                valid_indices = current_peak_indices[valid_mask]
                if valid_indices.size > 0:
                    ax_freq.scatter(current_frequencies[valid_indices], current_power_spectrum[valid_indices],
                                    color='red', marker='o', s=80, label=f"检测峰 (Th={data_package['peak_threshold_actual']:.2f})", zorder=5)
                    for pk_idx in valid_indices:
                        ax_freq.annotate(f'{current_frequencies[pk_idx]:.2f} Hz',
                                         (current_frequencies[pk_idx], current_power_spectrum[pk_idx]),
                                         textcoords="offset points", xytext=(0,10), ha='center', color='red', fontsize=8)
            
            if data_package['base_freq_actual'] is not None:
                 ax_freq.axvline(data_package['base_freq_actual'], color='green', linestyle='--', alpha=0.6,
                                label=f'原始基频 ({data_package["base_freq_actual"]:.2f} Hz)')
            ax_freq.legend(fontsize='small')

            nyquist = SYSTEM_FS / 2
            max_ps_val = np.max(current_power_spectrum) if current_power_spectrum.size > 0 else 0
            sig_freq_max = 0
            if max_ps_val > 1e-9:
                thresh_for_xlim = max_ps_val * 0.001
                sig_indices = np.where(current_power_spectrum >= thresh_for_xlim)[0]
                if sig_indices.size > 0 and sig_indices[-1] < len(current_frequencies):
                    sig_freq_max = current_frequencies[sig_indices[-1]]
            
            xlim_up_candidate1 = max(30, sig_freq_max * 1.5)
            xlim_up_candidate2 = data_package['base_freq_actual'] * 2.5 if data_package['base_freq_actual'] else 30
            xlim_up = min(max(xlim_up_candidate1, xlim_up_candidate2), nyquist)
            ax_freq.set_xlim(0, xlim_up)

            if max_ps_val > 1e-9:
                ax_freq.set_ylim(bottom=-0.05 * max_ps_val, top=max_ps_val * 1.15)
            else:
                ax_freq.set_ylim(bottom=0, top=1.0)
        else:
            ax_freq.text(0.5, 0.5, "频谱数据缺失或不匹配", ha='center', va='center')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    print("\n--- 所有处理和图形创建完毕 ---")
    if processed_signal_data_list:
        print("调用 plt.show() 来显示图形...")
        plt.show(block=False) 
        input("按 Enter 键结束程序并关闭图形...") 
        plt.close('all') 
    else:
        print("没有生成任何图形。")
        
    print("程序结束。")

if __name__ == "__main__":
    main()