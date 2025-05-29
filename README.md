# 雷达(模拟)信号处理实验



## 1. 项目概述

本项目旨在设计并实现一个模块化的雷达信号处理仿真系统，演示从信号生成到目标检测的完整流程。系统核心功能包括：含噪信号生成、巴特沃斯带通滤波、功率谱分析、峰值检测以及结果可视化。项目特别强调模块化设计，每个处理阶段均可独立测试，并通过中间文件传递数据。

## 2. 项目结构

radar_signal_stimulation/
├── intermediate_output/      # 模块间传递的中间数据文件
│   ├── signal_to_filters.txt
│   ├── filters_to_spec.txt
│   ├── spec_to_peak.txt
│   └── peak_to_visual.txt
├── moulds/                   # 核心处理模块
│   ├── signal_generator.py   # 信号生成模块
│   ├── filters.py            # 滤波器模块
│   ├── spectrum_analyzer.py  # 频谱分析模块
│   ├── peak_detector.py      # 峰值检测模块
│   └── visualizer.py         # 可视化模块
├── main.py                   # 主程序：标准单信号处理流程
├── main_merge.py             # 辅助程序：多信号分量叠加处理
├── main_multiple.py          # 辅助程序：多独立信号分别处理与对比显示
├── README.md                 # 本文件
├── item_requirest.md         # (项目需求文档，未提供内容)
├── pyproject.toml            # (Python项目配置文件，用于uv/poetry等)
└── uv.lock                   # (Python依赖锁定文件，用于uv)

## 3. 依赖与环境

*   **Python**: 3.x
*   **核心库**:
    *   `numpy`
    *   `scipy`
    *   `matplotlib`
*   **中文字体**: 为确保图表中的中文正常显示，脚本内已尝试配置 `SimHei` (Windows), `Arial Unicode MS` (macOS), 或 `WenQuanYi Micro Hei` (Linux)。请确保你的系统上安装了相应或兼容的字体。

## 4. 模块详解

每个模块位于 `moulds/` 文件夹下，并设计有 `if __name__ == '__main__':` 部分，允许独立运行和测试。

### 4.1 `signal_generator.py` (信号生成模块)
*   **功能**: 生成含高斯白噪声的正弦信号。
*   **核心函数**: `generate_signal(duration, num_points, base_freq, noise_mean, noise_std)`
*   **独立运行**:
    *   交互式输入多组信号参数（基频、噪声均值、噪声标准差）。
    *   固定时长 (`DURATION_SEC_SCRIPT = 1.0s`) 和采样点数 (`NUM_SAMPLES_SCRIPT = 1000`)。
    *   **输出**: 生成的信号数据（含时间轴、采样率、各信号参数及数据）保存到 `intermediate_output/signal_to_filters.txt` (JSON格式)。
    *   **显示**: 绘制所有生成信号的时域图。

### 4.2 `filters.py` (滤波器模块)
*   **功能**: 应用巴特沃斯带通滤波器（零相位）。
*   **核心函数**: `butter_bandpass_filter(data, lowcut, highcut, fs, order)`
*   **独立运行**:
    *   **输入**: 从 `intermediate_output/signal_to_filters.txt` 加载信号数据。
    *   使用固定的滤波器参数进行滤波 (e.g., `LOWCUT_FILTER_SCRIPT = 4.0Hz`, `HIGHCUT_FILTER_SCRIPT = 6.0Hz`, `ORDER_FILTER_SCRIPT = 5`)。采样率 `FS_FILTER` 从输入文件读取。
    *   **输出**: 原始信号、滤波后信号、时间轴、采样率及滤波器参数保存到 `intermediate_output/filters_to_spec.txt` (JSON格式)。
    *   **显示**:
        *   主图 (2x2): 原始信号时域、滤波后信号时域、第一个信号的功率谱对比、滤波器响应。
        *   若有多个信号，为其余信号分别创建独立的功率谱和滤波器响应图。
        *   所有信号滤波后的扩展时域图 (`EXTENDED_TIME_DURATION = 5.0s`)。

### 4.3 `spectrum_analyzer.py` (频谱分析模块)
*   **功能**: 计算信号的功率谱。
*   **核心函数**: `compute_power_spectrum(signal, fs)`
*   **独立运行**:
    *   **输入**: 从 `intermediate_output/filters_to_spec.txt` 加载原始和滤波后的信号数据及相关参数。
    *   对每个信号的**滤波后数据**计算功率谱。
    *   **输出**: 原始信号数据、滤波后信号数据、计算得到的频率轴和功率谱、时间轴、采样率及滤波器参数保存到 `intermediate_output/spec_to_peak.txt` (JSON格式)。
    *   **显示**: 绘制所有滤波后信号的功率谱合并图。

### 4.4 `peak_detector.py` (峰值检测模块)
*   **功能**: 基于阈值和局部最大值条件检测功率谱中的峰值。
*   **核心函数**: `detect_peaks(data, threshold_ratio, frequencies_param)`
    *   `data` 可以是 `(power_spectrum, frequencies)` 元组或 `power_spectrum` 数组 (此时需提供 `frequencies_param`)。
*   **独立运行**:
    *   **输入**: 从 `intermediate_output/spec_to_peak.txt` 加载频谱数据及前序所有数据。
    *   使用固定的峰值检测阈值比例 (`script_threshold_ratio = 0.5`)。
    *   **输出**: 原始信号数据、滤波后信号数据、频谱数据、检测到的峰值信息、时间轴、采样率及滤波器参数保存到 `intermediate_output/peak_to_visual.txt` (JSON格式)。
    *   **显示**: 为每个信号的功率谱及其检测到的峰值创建一个图。

### 4.5 `visualizer.py` (可视化模块)
*   **功能**: 绘制信号处理结果的综合图（时域对比与功率谱分析）。
*   **核心函数**: `plot_results(time_axis, original_signal, filtered_signal, frequencies, power_spectrum, ...)`
*   **独立运行**:
    *   **输入**: 从 `intermediate_output/peak_to_visual.txt` 加载完整的信号处理链数据。
    *   **输出**: 无文件输出。
    *   **显示**: 为文件中的每个信号处理结果条目调用 `plot_results` 函数，生成并显示一个双子图（时域对比、功率谱与峰值）。

## 5. 主程序脚本

### 5.1 `main.py` (标准流程)
*   **功能**: 执行一个标准的、端到端的雷达信号处理流程。
*   **参数**: 使用脚本内预定义的固定参数 (e.g., `SIGNAL_DURATION = 10.0s`, `SYSTEM_FS = 100.0Hz`, `BASE_FREQ = 5.0Hz`, `LOWCUT_FREQ = 4.0Hz`, `HIGHCUT_FREQ = 6.0Hz`, `FILTER_ORDER = 5`, `PEAK_THRESHOLD_RATIO = 0.5`)。
*   **流程**:
    1.  调用 `signal_generator.generate_signal` 生成原始信号。
    2.  调用 `filters.butter_bandpass_filter` 对信号进行滤波。
    3.  调用 `spectrum_analyzer.compute_power_spectrum` 计算滤波后信号的功率谱。
    4.  调用 `peak_detector.detect_peaks` 检测功率谱中的峰值。
    5.  调用 `visualizer.plot_results` 绘制综合结果图。
*   **输出**: 显示一个包含时域对比和频域分析（含峰值）的图形窗口。

### 5.2 `main_merge.py` (多信号分量叠加处理)
*   **功能**: 允许用户交互式定义多个信号分量（基频、噪声参数），这些分量将被**叠加（混合）**成一个信号，然后对这个混合信号进行统一的滤波、频谱分析、峰值检测和可视化。
*   **参数**:
    *   固定参数: `SIGNAL_DURATION = 1.0s`, `NUM_SAMPLES = 1000`, `SIGNAL_AMPLITUDE = 1.0` (每个分量)。
    *   固定处理参数: `LOWCUT_FREQ = 4.0Hz`, `HIGHCUT_FREQ = 6.0Hz`, `FILTER_ORDER = 5`, `PEAK_THRESHOLD_RATIO = 0.5`。
    *   交互式参数: 每个信号分量的基频、噪声均值、噪声标准差。
*   **流程**:
    1.  循环交互获取用户输入的多个信号分量参数并生成各分量。
    2.  将所有分量信号叠加成一个 `original_signal_mixed`。
    3.  对此混合信号执行滤波、频谱分析、峰值检测。
    4.  调用 `visualizer.plot_results` 显示混合信号的处理结果。
*   **输出**: 显示一个针对叠加后信号的综合处理结果图。

### 5.3 `main_multiple.py` (多独立信号分别处理与对比)
*   **功能**: 允许用户交互式定义多个**独立的信号**及其处理参数（基频、SNR、滤波器阶数、峰值检测阈值）。每个信号将独立完成整个处理链，并在同一个图形窗口的不同子图中并列显示结果，便于参数对比。
*   **参数**:
    *   固定信号参数: `SIGNAL_DURATION = 1.0s`, `NUM_SAMPLES = 1000`, `SIGNAL_AMPLITUDE = 1.0`。
    *   固定滤波器通带: `DEFAULT_LOWCUT_FREQ = 4.0Hz`, `DEFAULT_HIGHCUT_FREQ = 6.0Hz`。
    *   交互式参数 (每个独立信号): 基频, 滤波器阶数, 峰值检测阈值比例, 信噪比 (SNR), 噪声均值。
*   **流程**:
    1.  循环交互获取用户输入的每个独立信号的参数。
    2.  对每个独立信号分别执行生成、滤波、频谱分析、峰值检测。
    3.  为所有处理过的信号统一创建一个多子图窗口，每个信号占据一行（包含时域对比子图和功率谱子图）。
*   **输出**: 显示一个包含所有独立处理信号结果的多子图窗口。

## 6. 如何运行与复现

### 6.1 环境准备
1.  确保已安装 Python 3.x。
2.  安装所需库:
    ```bash
    pip install numpy scipy matplotlib
    ```
3.  (可选) 确保系统有支持中文的字体 (如 SimHei)。
4.  创建 `intermediate_output` 文件夹在项目根目录下，如果它不存在的话（部分脚本会自动创建）。

### 6.2 运行标准流程 (`main.py`)
直接运行主程序：
python main.py

这将使用预设参数执行整个信号处理链，并显示最终结果图。

### 6.3 独立测试各模块 (按顺序)

这是为了验证每个模块的功能并理解数据流。

1. **生成信号**:

   ```
         python moulds/signal_generator.py 
   ```

   交互式输入参数，生成 intermediate_output/signal_to_filters.txt 并显示时域图。

2. **滤波处理**:

   ```
         python moulds/filters.py 
   ```

   读取 signal_to_filters.txt，进行滤波，生成 intermediate_output/filters_to_spec.txt 并显示相关图表。

3. **频谱分析**:

   ```
         python moulds/spectrum_analyzer.py 
   ```

   读取 filters_to_spec.txt，计算功率谱，生成 intermediate_output/spec_to_peak.txt 并显示合并频谱图。

4. **峰值检测**:

   ```
         python moulds/peak_detector.py 
   ```

   读取 spec_to_peak.txt，检测峰值，生成 intermediate_output/peak_to_visual.txt 并为每个信号显示频谱与峰值图。

5. **最终可视化 (独立)**:

   ```
         python moulds/visualizer.py 
   ```

   读取 peak_to_visual.txt，为每个信号条目显示综合结果图。

### 6.4 运行交互式辅助脚本

- **多信号分量叠加处理**:

  ```
        python main_merge.py 
  ```

  按照提示输入各信号分量参数。

- **多独立信号分别处理与对比**:

  ```
        python main_multiple.py 
  ```

  按照提示输入各独立信号及其处理参数。

## 7. 关键参数说明 (参考 main.py 和实验报告)

- SIGNAL_DURATION: 信号时长 (秒)
- SYSTEM_FS: 系统采样频率 (Hz)
- BASE_FREQ: 目标信号基频 (Hz)
- NOISE_MEAN: 噪声均值
- NOISE_STD: 噪声标准差
- LOWCUT_FREQ: 带通滤波器低截止频率 (Hz)
- HIGHCUT_FREQ: 带通滤波器高截止频率 (Hz)
- FILTER_ORDER: 滤波器阶数
- PEAK_THRESHOLD_RATIO: 峰值检测阈值比例 (相对于最大功率)

## 8. 注意事项

- 所有脚本在绘图后会等待用户在控制台按 "Enter" 键才会关闭图形并退出。
- 中间文件均以 JSON 格式存储在 intermediate_output/ 目录下。
- main_multiple.py 和 main_merge.py 提供了更灵活的参数探索和场景模拟能力。

