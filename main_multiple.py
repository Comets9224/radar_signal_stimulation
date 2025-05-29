# main_multiple.py
"""
主程序脚本 - 交互式生成多个独立信号，分别处理并在同一图形窗口的不同子图中可视化
"""
import numpy as np  # 导入NumPy库，用于数值计算，特别是数组操作
import matplotlib.pyplot as plt  # 导入Matplotlib的pyplot模块，用于绘图

# --- 从模块导入函数 ---
try:
    from moulds.signal_generator import generate_signal  # 从自定义模块导入信号生成函数
    from moulds.filters import butter_bandpass_filter  # 从自定义模块导入带通滤波器函数
    from moulds.spectrum_analyzer import compute_power_spectrum  # 从自定义模块导入功率谱计算函数
    from moulds.peak_detector import detect_peaks  # 从自定义模块导入峰值检测函数
    # visualizer.plot_results 仍然有用，但我们会直接在main中构建子图
    from moulds.visualizer import plot_results  # 从自定义模块导入绘图结果函数 (或其绘图辅助逻辑，如果已拆分)
except ImportError as e:  # 捕获导入错误
    print(f"错误：无法导入一个或多个模块: {e}")  # 打印错误信息
    exit()  # 退出程序

# --- 中文显示配置 ---
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 尝试设置matplotlib使用SimHei字体以支持中文显示
    plt.rcParams['axes.unicode_minus'] = False  # 尝试设置matplotlib正常显示负号
    print("[Main_Multiple] 尝试设置matplotlib中文字体为SimHei。")  # 打印字体设置尝试信息
except Exception as e:  # 捕获设置字体时可能发生的任何异常
    print(f"[Main_Multiple] 设置中文字体失败: {e}。")  # 打印字体设置失败的信息
# --- 中文显示配置结束 ---

# ==============================================================================
# --- 固定信号生成参数 (可在脚本开头修改) ---
# ==============================================================================
SIGNAL_DURATION = 1.0  # 定义信号的时长，单位为秒
NUM_SAMPLES = 1000  # 定义信号的采样点数
SYSTEM_FS = NUM_SAMPLES / SIGNAL_DURATION  # 计算系统的采样率，单位为赫兹 (Hz)
SIGNAL_AMPLITUDE = 1.0  # 定义信号分量的固定幅度 (假设 generate_signal 内部使用此值或类似概念)

# --- 初始默认处理参数 (交互式输入时作为起点) ---
DEFAULT_BASE_FREQ = 5.0  # 定义默认的信号基频，单位为赫兹 (Hz)
DEFAULT_LOWCUT_FREQ = 4.0  # 定义带通滤波器的默认低截止频率，单位为赫兹 (Hz) (滤波参数保持固定)
DEFAULT_HIGHCUT_FREQ = 6.0  # 定义带通滤波器的默认高截止频率，单位为赫兹 (Hz) (滤波参数保持固定)
DEFAULT_FILTER_ORDER = 5  # 定义滤波器的默认阶数
DEFAULT_PEAK_THRESHOLD_RATIO = 0.5  # 定义峰值检测的默认阈值比例 (相对于最大峰值)
DEFAULT_SNR_LINEAR = 10.0  # 定义默认的信噪比 (线性值, 信号功率 / 噪声功率)
DEFAULT_NOISE_MEAN = 0.0  # 定义噪声的默认均值


# ==============================================================================

def get_interactive_float(prompt_message, default_value, min_val=None, max_val=None):
    """
    辅助函数：从用户处获取浮点数输入，带有默认值和可选的范围检查。

    参数:
        prompt_message (str): 显示给用户的提示信息。
        default_value (float): 如果用户直接按回车，则使用的默认值。
        min_val (float, optional): 允许的最小值。默认为None（无最小限制）。
        max_val (float, optional): 允许的最大值。默认为None（无最大限制）。

    返回:
        float: 用户输入的有效浮点数或默认值。
    """
    while True:  # 无限循环，直到获得有效输入或使用默认值
        user_str = input(prompt_message)  # 显示提示信息并获取用户输入字符串
        if not user_str:  # 如果用户输入为空字符串 (直接按回车)
            return default_value  # 返回默认值
        try:
            val = float(user_str)  # 尝试将用户输入转换为浮点数
            if (min_val is not None and val < min_val) or \
                    (max_val is not None and val > max_val):  # 检查值是否在允许范围内
                # 如果值超出范围，打印错误信息
                print(
                    f"  输入值超出允许范围 ({min_val if min_val is not None else '-inf'} to {max_val if max_val is not None else 'inf'})。请重试或按回车使用默认值。")
                continue  # 继续下一次循环，要求用户重新输入
            return val  # 如果值有效且在范围内，则返回该值
        except ValueError:  # 如果转换浮点数失败 (例如，用户输入了文本)
            print("  输入无效，请输入数字。请重试或按回车使用默认值。")  # 打印无效输入错误信息


def get_interactive_int(prompt_message, default_value, min_val=None, max_val=None):
    """
    辅助函数：从用户处获取整数输入，带有默认值和可选的范围检查。

    参数:
        prompt_message (str): 显示给用户的提示信息。
        default_value (int): 如果用户直接按回车，则使用的默认值。
        min_val (int, optional): 允许的最小值。默认为None（无最小限制）。
        max_val (int, optional): 允许的最大值。默认为None（无最大限制）。

    返回:
        int: 用户输入的有效整数或默认值。
    """
    while True:  # 无限循环，直到获得有效输入或使用默认值
        user_str = input(prompt_message)  # 显示提示信息并获取用户输入字符串
        if not user_str:  # 如果用户输入为空字符串 (直接按回车)
            return default_value  # 返回默认值
        try:
            val = int(user_str)  # 尝试将用户输入转换为整数
            if (min_val is not None and val < min_val) or \
                    (max_val is not None and val > max_val):  # 检查值是否在允许范围内
                # 如果值超出范围，打印错误信息
                print(
                    f"  输入值超出允许范围 ({min_val if min_val is not None else '-inf'} to {max_val if max_val is not None else 'inf'})。请重试或按回车使用默认值。")
                continue  # 继续下一次循环，要求用户重新输入
            return val  # 如果值有效且在范围内，则返回该值
        except ValueError:  # 如果转换整数失败 (例如，用户输入了文本或浮点数)
            print("  输入无效，请输入整数。请重试或按回车使用默认值。")  # 打印无效输入错误信息


def main():
    """
    主函数：交互式地生成和处理多个独立信号，并在一个图形窗口的多个子图中统一显示结果。
    用户可以为每个信号指定不同的参数，如基频、滤波器阶数、峰值检测阈值和信噪比。
    固定的参数包括信号时长、采样点数、采样率、信号分量幅度和滤波器通带。
    """
    print("--- 交互式多独立信号处理与统一多子图可视化 ---")  # 打印程序标题
    # 打印固定的信号参数
    print(
        f"固定信号参数: 时长={SIGNAL_DURATION}s, 点数={NUM_SAMPLES}, 采样率={SYSTEM_FS}Hz, 信号分量幅度={SIGNAL_AMPLITUDE}")
    # 打印固定的滤波器通带
    print(f"固定滤波器通带: {DEFAULT_LOWCUT_FREQ}-{DEFAULT_HIGHCUT_FREQ}Hz")
    # 打印初始的默认处理参数
    print(
        f"初始默认值: 基频={DEFAULT_BASE_FREQ}Hz, 滤波阶数={DEFAULT_FILTER_ORDER}, 检测阈值比例={DEFAULT_PEAK_THRESHOLD_RATIO * 100:.0f}%, 信噪比(线性)={DEFAULT_SNR_LINEAR}")

    # 生成所有信号共用的时间轴
    time_axis_common = np.linspace(0, SIGNAL_DURATION, NUM_SAMPLES, endpoint=False)

    processed_signal_data_list = []  # 初始化一个列表，用于存储每个处理后信号的数据包
    keep_adding_signals = True  # 控制是否继续添加信号的标志，默认为True
    signal_counter = 1  # 初始化信号计数器

    # 初始化用于下一个信号的当前默认参数值，这些值会在每次循环后更新
    current_default_base_freq = DEFAULT_BASE_FREQ  # 当前默认基频
    current_default_filter_order = DEFAULT_FILTER_ORDER  # 当前默认滤波器阶数
    current_default_peak_threshold = DEFAULT_PEAK_THRESHOLD_RATIO  # 当前默认峰值检测阈值比例
    current_default_snr = DEFAULT_SNR_LINEAR  # 当前默认信噪比

    while keep_adding_signals:  # 当需要继续添加信号时循环
        print(f"\n--- 请输入第 {signal_counter} 个独立信号的参数 ---")  # 提示用户输入当前信号的参数

        # 获取当前信号的基频
        base_freq = get_interactive_float(
            f"  基频 (Hz) [默认 {current_default_base_freq:.1f}]: ",  # 提示信息
            current_default_base_freq,  # 默认值
            min_val=0.1  # 最小值限制
        )

        # 获取当前信号的滤波器阶数
        filter_order_current = get_interactive_int(
            f"  滤波器阶数 [默认 {current_default_filter_order}]: ",  # 提示信息
            current_default_filter_order,  # 默认值
            min_val=1  # 最小值限制
        )

        # 获取当前信号的峰值检测阈值比例
        peak_threshold_ratio_current = get_interactive_float(
            f"  峰值检测阈值比例 (0.0-1.0) [默认 {current_default_peak_threshold:.2f}]: ",  # 提示信息
            current_default_peak_threshold,  # 默认值
            min_val=0.0,  # 最小值限制
            max_val=1.0  # 最大值限制
        )

        # 获取当前信号的信噪比 (线性)
        snr_linear_current = get_interactive_float(
            f"  信噪比 (线性 P_sig/P_noise) [默认 {current_default_snr:.1f}]: ",  # 提示信息
            current_default_snr,  # 默认值
            min_val=0.01  # 最小值限制 (SNR不能为0或负)
        )

        # 根据信噪比计算噪声标准差 (noise_std)
        if snr_linear_current <= 0:  # 如果输入的SNR不合法 (小于等于0)
            print("  信噪比必须为正。使用默认噪声标准差。")  # 打印警告信息
            # 如果SNR无效，则使用基于默认SNR的噪声标准差作为回退
            noise_std = SIGNAL_AMPLITUDE / np.sqrt(2 * DEFAULT_SNR_LINEAR)
            print(f"  使用默认SNR ({DEFAULT_SNR_LINEAR}) 计算噪声标准差。")  # 打印使用的默认SNR信息
        else:
            # 对于正弦波，SNR = (信号功率) / (噪声功率) = (SIGNAL_AMPLITUDE^2 / 2) / noise_std^2
            # 因此，noise_std^2 = (SIGNAL_AMPLITUDE^2 / 2) / SNR
            # 所以，noise_std = SIGNAL_AMPLITUDE / sqrt(2 * SNR)
            noise_std = SIGNAL_AMPLITUDE / np.sqrt(2 * snr_linear_current)  # 计算噪声标准差

        # 获取噪声均值 (仍然可以独立设置)
        noise_mean = get_interactive_float(
            f"  噪声均值 [默认 {DEFAULT_NOISE_MEAN:.1f}]: ",  # 提示信息
            DEFAULT_NOISE_MEAN  # 默认值
        )

        # 构建当前信号的标签字符串，包含其主要参数
        current_signal_label_parts = [
            f"F={base_freq:.1f}Hz",  # 基频
            f"Amp={SIGNAL_AMPLITUDE}",  # 幅度 (使用全局固定幅度)
            f"SNR={snr_linear_current:.1f}",  # 信噪比
            f"FiltOrd={filter_order_current}",  # 滤波器阶数
            f"Thresh={peak_threshold_ratio_current:.2f}"  # 峰值检测阈值
        ]
        current_signal_label = f"原始信号 {signal_counter}: " + ", ".join(current_signal_label_parts)  # 组合成完整标签
        print(f"  生成信号: {current_signal_label}")  # 打印生成的信号信息
        # 打印计算得到的噪声标准差信息
        print(f"  (计算得到 噪声标准差: {noise_std:.3f} for SNR={snr_linear_current:.1f} and Amp={SIGNAL_AMPLITUDE})")

        # 1. 信号生成
        # 调用 generate_signal 函数生成原始信号
        # 已修改: 移除了 'amplitude' 关键字参数，因为它不被 generate_signal 函数所期望
        _, original_signal, _ = generate_signal(  # '_' 用于接收不使用的返回值 (如纯净信号)
            duration=SIGNAL_DURATION,  # 信号时长
            num_points=NUM_SAMPLES,  # 采样点数
            base_freq=base_freq,  # 信号基频
            # amplitude=SIGNAL_AMPLITUDE, # 此行曾导致TypeError，已移除
            noise_mean=noise_mean,  # 噪声均值
            noise_std=noise_std  # 噪声标准差
        )

        # 2. 滤波处理
        # 调用 butter_bandpass_filter 函数对原始信号进行带通滤波
        filtered_signal = butter_bandpass_filter(
            data=original_signal,  # 输入的原始信号数据
            lowcut=DEFAULT_LOWCUT_FREQ,  # 低截止频率 (固定)
            highcut=DEFAULT_HIGHCUT_FREQ,  # 高截止频率 (固定)
            fs=SYSTEM_FS,  # 采样率
            order=filter_order_current  # 滤波器阶数 (使用当前交互式输入值)
        )

        # 3. 频谱分析
        # 调用 compute_power_spectrum 函数计算滤波后信号的功率谱
        frequencies, power_spectrum = compute_power_spectrum(
            signal=filtered_signal,  # 输入的滤波后信号
            fs=SYSTEM_FS  # 采样率
        )

        # 4. 峰值检测
        spectrum_data_for_peaks = (power_spectrum, frequencies)  # 准备用于峰值检测的频谱数据 (功率谱和对应频率)
        # 调用 detect_peaks 函数在功率谱上检测峰值
        peak_indices, detected_frequencies_list, _ = detect_peaks(  # '_' 用于接收不使用的返回值 (如峰值幅度)
            spectrum_data_for_peaks,  # 输入的频谱数据
            threshold_ratio=peak_threshold_ratio_current  # 峰值检测阈值比例 (使用当前交互式输入值)
        )

        peak_info_str = "未检测到显著峰值"  # 初始化峰值信息字符串
        if detected_frequencies_list:  # 如果检测到了峰值
            formatted_freqs = [f"{f:.2f}" for f in detected_frequencies_list]  # 格式化检测到的频率列表
            peak_info_str = f"检测峰值: {', '.join(formatted_freqs)} Hz"  # 更新峰值信息字符串
        print(f"  处理完毕. {peak_info_str}")  # 打印处理完成信息和峰值检测结果

        # 将当前信号的所有相关数据打包成一个字典，并添加到列表中
        processed_signal_data_list.append({
            "original_label": current_signal_label,  # 原始信号的标签
            "filtered_label": f"已处理: {', '.join(current_signal_label_parts)}",  # 处理后信号的标签
            "time_axis": time_axis_common,  # 共用的时间轴
            "original_signal": original_signal,  # 原始信号数据
            "filtered_signal": filtered_signal,  # 滤波后信号数据
            "frequencies": frequencies,  # 频率轴数据
            "power_spectrum": power_spectrum,  # 功率谱数据
            "peak_indices": peak_indices,  # 检测到的峰值索引
            "base_freq_actual": base_freq,  # 实际使用的基频
            "filter_order_actual": filter_order_current,  # 实际使用的滤波器阶数
            "peak_threshold_actual": peak_threshold_ratio_current,  # 实际使用的峰值检测阈值
            "snr_actual": snr_linear_current,  # 实际使用的信噪比
            "noise_std_actual": noise_std,  # 实际计算得到的噪声标准差
            "peak_info_str": peak_info_str  # 峰值信息的字符串描述
        })

        # 更新下一个信号的默认参数值，以便用户连续输入时有递增的默认选项
        current_default_base_freq = round(base_freq + 2.0, 1)  # 递增基频，并四舍五入到一位小数
        # 递增滤波器阶数，如果小于10则加1，否则重置为默认阶数 (可设置上限)
        current_default_filter_order = filter_order_current + 1 if filter_order_current < 10 else DEFAULT_FILTER_ORDER
        current_default_peak_threshold = round(min(peak_threshold_ratio_current + 0.05, 0.9),
                                               2)  # 递增阈值，但不超过0.9，并四舍五入到两位小数
        current_default_snr = round(snr_linear_current + 5.0, 1)  # 递增SNR，并四舍五入到一位小数

        signal_counter += 1  # 信号计数器加1
        if len(processed_signal_data_list) >= 1:  # 如果至少处理了一个信号
            # 询问用户是否要添加并处理下一个独立信号
            add_another = input("是否添加并处理下一个独立信号? (y/n) [y]: ").lower()  # 获取用户输入并转为小写
            if add_another == 'n':  # 如果用户输入 'n'
                keep_adding_signals = False  # 设置标志为False，以退出循环

    if not processed_signal_data_list:  # 如果没有生成和处理任何信号
        print("没有生成和处理任何信号，程序退出。")  # 打印信息
        return  # 退出main函数

    # --- 5. 统一可视化 ---
    num_signals = len(processed_signal_data_list)  # 获取已处理信号的数量
    # 创建一个图形窗口(fig)和一组子图(axes)，每个信号占一行，每行包含时域和频域两个子图
    # squeeze=False确保axes总是一个二维数组，即使只有一个信号
    fig, axes = plt.subplots(num_signals, 2, figsize=(18, 6 * num_signals), squeeze=False)

    # 构建图形的总标题字符串
    fig_suptitle = (f"多信号独立处理结果\n"
                    f"固定参数: 信号时长={SIGNAL_DURATION}s, 幅度={SIGNAL_AMPLITUDE}, "
                    f"采样率={SYSTEM_FS}Hz, 滤波通带={DEFAULT_LOWCUT_FREQ}-{DEFAULT_HIGHCUT_FREQ}Hz")
    fig.suptitle(fig_suptitle, fontsize=16)  # 设置图形的总标题

    # 遍历每个已处理的信号数据包进行绘图
    for idx, data_package in enumerate(processed_signal_data_list):
        ax_time = axes[idx, 0]  # 获取当前信号的时域子图对象
        ax_freq = axes[idx, 1]  # 获取当前信号的频域子图对象

        # 构建当前子图标题中显示的信号参数信息字符串
        signal_params_info = (
            f"F={data_package['base_freq_actual']:.1f}Hz, "  # 基频
            f"SNR={data_package['snr_actual']:.1f} (NoiseStd={data_package['noise_std_actual']:.3f}), "  # SNR和噪声标准差
            f"FiltOrd={data_package['filter_order_actual']}, "  # 滤波器阶数
            f"PeakThresh={data_package['peak_threshold_actual']:.2f}"  # 峰值检测阈值
        )

        # --- 配置时域子图 ---
        ax_time.set_title(f"信号 {idx + 1} 时域: {signal_params_info}")  # 设置时域子图标题
        ax_time.set_xlabel('时间 (秒)')  # 设置X轴标签
        ax_time.set_ylabel('幅值')  # 设置Y轴标签
        ax_time.grid(True)  # 显示网格
        if data_package['time_axis'].size > 0:  # 检查时间轴数据是否存在
            if data_package['original_signal'].size > 0:  # 检查原始信号数据是否存在
                # 确保绘图长度不超过时间轴和信号数据的最小长度
                plot_len = min(len(data_package['time_axis']), len(data_package['original_signal']))
                # 绘制原始信号
                ax_time.plot(data_package['time_axis'][:plot_len], data_package['original_signal'][:plot_len],
                             label='原始信号', alpha=0.7)
            if data_package['filtered_signal'].size > 0:  # 检查滤波后信号数据是否存在
                # 确保绘图长度不超过时间轴和信号数据的最小长度
                plot_len = min(len(data_package['time_axis']), len(data_package['filtered_signal']))
                # 绘制滤波后信号
                ax_time.plot(data_package['time_axis'][:plot_len], data_package['filtered_signal'][:plot_len],
                             label='滤波后信号', color='orange')
            if data_package['original_signal'].size > 0 or data_package['filtered_signal'].size > 0:  # 如果有任一信号被绘制
                ax_time.legend(fontsize='small')  # 显示图例
        else:  # 如果时域数据缺失
            ax_time.text(0.5, 0.5, "时域数据缺失", ha='center', va='center')  # 在子图中央显示提示信息

        # --- 配置频域子图 ---
        ax_freq.set_title(f"信号 {idx + 1} 功率谱: {data_package['peak_info_str']}")  # 设置频域子图标题，包含峰值信息
        ax_freq.set_xlabel('频率 (Hz)')  # 设置X轴标签
        ax_freq.set_ylabel('功率')  # 设置Y轴标签
        ax_freq.grid(True)  # 显示网格

        current_frequencies = data_package['frequencies']  # 获取当前信号的频率数据
        current_power_spectrum = data_package['power_spectrum']  # 获取当前信号的功率谱数据
        # 获取峰值索引，如果为空则创建一个空numpy数组
        current_peak_indices = np.array(data_package['peak_indices'], dtype=int) if data_package[
            'peak_indices'] else np.array([], dtype=int)

        # 检查频谱数据是否有效且长度匹配
        if current_frequencies.size > 0 and current_power_spectrum.size > 0 and \
                len(current_frequencies) == len(current_power_spectrum):
            # 绘制功率谱
            ax_freq.plot(current_frequencies, current_power_spectrum, label='功率谱', color='dodgerblue')
            if current_peak_indices.size > 0:  # 如果存在峰值索引
                # 创建一个有效性掩码，确保峰值索引在频率数组的有效范围内
                valid_mask = (current_peak_indices >= 0) & (current_peak_indices < len(current_frequencies))
                valid_indices = current_peak_indices[valid_mask]  # 获取有效的峰值索引
                if valid_indices.size > 0:  # 如果存在有效的峰值
                    # 在功率谱上标记检测到的峰值
                    ax_freq.scatter(current_frequencies[valid_indices], current_power_spectrum[valid_indices],
                                    color='red', marker='o', s=80,
                                    label=f"检测峰 (Th={data_package['peak_threshold_actual']:.2f})", zorder=5)
                    for pk_idx in valid_indices:  # 遍历每个有效峰值索引
                        # 为每个峰值添加频率标注
                        ax_freq.annotate(f'{current_frequencies[pk_idx]:.2f} Hz',
                                         (current_frequencies[pk_idx], current_power_spectrum[pk_idx]),
                                         textcoords="offset points", xytext=(0, 10), ha='center', color='red',
                                         fontsize=8)

            if data_package['base_freq_actual'] is not None:  # 如果存在原始基频信息
                # 在图上绘制一条垂直虚线表示原始基频位置
                ax_freq.axvline(data_package['base_freq_actual'], color='green', linestyle='--', alpha=0.6,
                                label=f'原始基频 ({data_package["base_freq_actual"]:.2f} Hz)')
            ax_freq.legend(fontsize='small')  # 显示图例

            nyquist = SYSTEM_FS / 2  # 计算奈奎斯特频率
            max_ps_val = np.max(current_power_spectrum) if current_power_spectrum.size > 0 else 0  # 获取功率谱的最大值
            sig_freq_max = 0  # 初始化信号主要能量的最高频率
            if max_ps_val > 1e-9:  # 如果功率谱有显著值
                # 定义一个阈值，用于确定信号能量集中的频率范围
                thresh_for_xlim = max_ps_val * 0.001
                # 找到功率谱值大于该阈值的索引
                sig_indices = np.where(current_power_spectrum >= thresh_for_xlim)[0]
                if sig_indices.size > 0 and sig_indices[-1] < len(current_frequencies):  # 如果找到这样的索引
                    sig_freq_max = current_frequencies[sig_indices[-1]]  # 更新信号主要能量的最高频率

            # 动态调整频域图的X轴上限，使其能良好显示信号的主要频率成分
            xlim_up_candidate1 = max(30, sig_freq_max * 1.5)  # 候选上限1: 基于信号能量最高频率的1.5倍，或至少30Hz
            xlim_up_candidate2 = data_package['base_freq_actual'] * 2.5 if data_package[
                'base_freq_actual'] else 30  # 候选上限2: 基于原始基频的2.5倍，或至少30Hz
            # 最终X轴上限取两个候选值的较大者，但不超过奈奎斯特频率
            xlim_up = min(max(xlim_up_candidate1, xlim_up_candidate2), nyquist)
            ax_freq.set_xlim(0, xlim_up)  # 设置X轴范围

            if max_ps_val > 1e-9:  # 如果功率谱有显著值
                # 设置Y轴范围，底部留一些空间，顶部也留一些空间
                ax_freq.set_ylim(bottom=-0.05 * max_ps_val, top=max_ps_val * 1.15)
            else:  # 如果功率谱值很小或为0
                ax_freq.set_ylim(bottom=0, top=1.0)  # 设置一个默认的Y轴范围
        else:  # 如果频谱数据缺失或不匹配
            ax_freq.text(0.5, 0.5, "频谱数据缺失或不匹配", ha='center', va='center')  # 在子图中央显示提示信息

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # 自动调整子图参数，使其填充整个图像区域，并为总标题留出空间
    print("\n--- 所有处理和图形创建完毕 ---")  # 打印完成信息
    if processed_signal_data_list:  # 如果处理了至少一个信号并生成了数据
        print("调用 plt.show() 来显示图形...")  # 提示用户即将显示图形
        plt.show(block=False)  # 显示图形，block=False使其非阻塞，允许后续代码执行
        input("按 Enter 键结束程序并关闭图形...")  # 等待用户按Enter键
        plt.close('all')  # 关闭所有matplotlib图形窗口
    else:  # 如果没有生成任何图形
        print("没有生成任何图形。")  # 打印信息

    print("程序结束。")  # 打印程序结束信息


if __name__ == "__main__":  # 如果此脚本作为主程序运行
    main()  # 调用main函数