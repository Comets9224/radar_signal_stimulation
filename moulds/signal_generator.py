# moulds/signal_generator.py
"""
信号生成模块
"""
import numpy as np
import json
import os # 用于路径操作

# 函数定义保持不变 (被main.py调用时，不执行文件操作和绘图)
def generate_signal(duration=10, num_points=1000, base_freq=5, noise_mean=0, noise_std=0.5):
    time_axis = np.linspace(0, duration, num_points, endpoint=False)
    # 确保幅度为1，符合main.py的隐含要求 (虽然未明确，但通常sin前无系数即为1)
    amplitude = 1.0 
    clean_signal = amplitude * np.sin(2 * np.pi * base_freq * time_axis)
    noise = np.random.normal(noise_mean, noise_std, num_points)
    noisy_signal = clean_signal + noise
    fs_actual = num_points / duration
    return time_axis, noisy_signal, fs_actual

# --- 当这个脚本被直接运行时 ---
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # import time # time.sleep 不再需要，因为用input()阻塞

    print("\n--- 信号生成模块独立测试 (交互式多曲线对比模式 + 保存数据) ---")

    # 尝试设置中文字体
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        print("[模块一] 尝试设置字体为 SimHei")
    except Exception as e:
        print(f"[模块一] 设置字体 SimHei 失败: {e}")

    # 固定参数用于独立测试时的信号生成
    DURATION_SEC_SCRIPT = 1.0 # 独立测试时时长可以不同于main.py调用
    NUM_SAMPLES_SCRIPT = 1000
    print(f"将使用固定时长: {DURATION_SEC_SCRIPT} 秒, 固定采样点数: {NUM_SAMPLES_SCRIPT} 个 (FS={NUM_SAMPLES_SCRIPT/DURATION_SEC_SCRIPT}Hz)")

    time_axis_common = np.linspace(0, DURATION_SEC_SCRIPT, NUM_SAMPLES_SCRIPT, endpoint=False)
    fs_common = NUM_SAMPLES_SCRIPT / DURATION_SEC_SCRIPT

    # --- 图形设置 ---
    # plt.ion() # 不再需要交互式绘图后立即显示，改为最后统一show
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title("生成的信号 (可交互添加)")
    ax.set_xlabel("时间 (秒)")
    ax.set_ylabel("幅值")
    ax.grid(True)
    # --- 图形设置结束 ---

    signals_data_to_save = {
        "time_axis": time_axis_common.tolist(),
        "fs": fs_common,
        "signals": []
    }
    curve_count = 1
    keep_adding = True

    while keep_adding:
        print(f"\n--- 输入第 {curve_count} 条曲线的参数 ---")
        try:
            user_base_freq_str = input(f"基频 (Hz) [默认 5]: ")
            user_base_freq = float(user_base_freq_str) if user_base_freq_str else 5.0

            user_noise_mean_str = input(f"噪声均值 [默认 0]: ")
            user_noise_mean = float(user_noise_mean_str) if user_noise_mean_str else 0.0

            user_noise_std_str = input(f"噪声标准差 [默认 0.5]: ")
            user_noise_std = float(user_noise_std_str) if user_noise_std_str else 0.5
            if user_noise_std < 0:
                user_noise_std = 0.5
                print("标准差不能为负,已设为0.5")

        except ValueError:
            print("输入无效，请使用数字。参数将使用默认值。")
            user_base_freq = 5.0
            user_noise_mean = 0.0
            user_noise_std = 0.5

        print(f"生成: 基频={user_base_freq}Hz, 噪声均值={user_noise_mean}, 噪声标准差={user_noise_std}")
        
        # 调用 generate_signal 函数 (注意：这里返回的 time_axis 是基于 DURATION_SEC_SCRIPT 的)
        # 但我们保存和绘图都使用 time_axis_common
        _, current_signal, _ = generate_signal(
            duration=DURATION_SEC_SCRIPT, # 使用脚本的参数
            num_points=NUM_SAMPLES_SCRIPT, # 使用脚本的参数
            base_freq=user_base_freq,
            noise_mean=user_noise_mean,
            noise_std=user_noise_std
        )

        label = f"F={user_base_freq}Hz, Mean={user_noise_mean}, Std={user_noise_std}"
        ax.plot(time_axis_common, current_signal, label=label, alpha=0.85)
        # ax.legend() # 图例在循环外统一添加，避免重复
        # plt.draw() # 不需要立即绘制
        # plt.pause(0.1) # 不需要暂停

        signals_data_to_save["signals"].append({
            "label": label,
            "base_freq": user_base_freq, # 保存实际用于生成该信号的基频
            "noise_mean": user_noise_mean,
            "noise_std": user_noise_std,
            "data": current_signal.tolist()
        })

        add_another = input("是否添加下一条曲线? (y/n) [y]: ").lower()
        if add_another == 'n':
            keep_adding = False
        curve_count += 1

    # --- 文件保存路径修改 ---
    # 获取当前脚本所在的目录 (moulds)
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    # 获取项目根目录 (moulds的上级目录)
    project_root_dir = os.path.dirname(current_script_dir)
    # 构建输出文件夹路径
    output_folder = os.path.join(project_root_dir, "intermediate_output")
    
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        try:
            os.makedirs(output_folder)
            print(f"已创建输出文件夹: {output_folder}")
        except OSError as e:
            print(f"错误: 创建输出文件夹 {output_folder} 失败: {e}")
            # 可以选择退出或尝试在当前目录保存
            output_folder = "." # Fallback to current directory

    output_filename = os.path.join(output_folder, "signal_to_filters.txt")
    # --- 文件保存路径修改结束 ---

    try:
        with open(output_filename, 'w', encoding='utf-8') as f: # Added encoding
            json.dump(signals_data_to_save, f, indent=4)
        print(f"\n所有生成的信号数据已保存到 {output_filename}")
    except IOError as e:
        print(f"错误：无法写入文件 {output_filename}: {e}")
    except Exception as e:
        print(f"保存数据时发生未知错误: {e}")

    print("\n已结束添加曲线。")
    ax.set_title("生成的信号 (最终结果)")
    if signals_data_to_save["signals"]: # 只有在有信号时才显示图例
        ax.legend(loc='upper right')
    
    # plt.ioff() # 不再需要，因为没有使用 plt.ion() 后立即 plt.draw()
    plt.show(block=False) # 显示图形，但不阻塞

    print(f"采样率: {fs_common:.2f} Hz")
    input("按 Enter 键结束程序并关闭所有图形...") # 等待控制台回车
    plt.close('all') # 关闭所有图形
    print("程序结束。")
    print("--- 模块一测试结束 ---")