# XRD特征提取工具 - 硬碳专项分析
# 专注XRD全局特征提取，带手动基线调整功能

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
# 强制使用非交互式后端，解决多显示器问题
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths, savgol_filter
from scipy.integrate import simpson
from scipy.optimize import curve_fit
import os
import base64
from scipy import sparse
from scipy.sparse.linalg import spsolve
import time
from scipy import stats
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# 高斯函数用于峰拟合
def gaussian(x, a, b, c):
    return a * np.exp(-(x - b)**2 / (2 * c**2))

# 洛伦兹函数用于峰拟合
def lorentzian(x, a, b, c):
    return a / (1 + ((x - b) / c)**2)

# 数据归一化函数
def normalize_data(intensity):
    """将强度数据归一化到0-1范围"""
    min_val = np.min(intensity)
    max_val = np.max(intensity)
    if max_val - min_val > 0:
        return (intensity - min_val) / (max_val - min_val)
    else:
        return intensity

# 两种不同的背景扣除算法
def asymmetric_least_squares_precise(y, lam=1e6, p=0.01, n_iter=10):
    """
    精确模式：原始ALS背景扣除算法
    使用稠密矩阵，准确性高但速度较慢
    """
    L = len(y)
    # 构造二阶差分矩阵 (L-2) x L
    D = np.diff(np.eye(L), 2, axis=0)
    w = np.ones(L)
    z = np.zeros(L)
    
    for i in range(n_iter):
        # 构造对角权重矩阵
        W = np.diag(w)
        # 构造系统矩阵
        Z = W + lam * D.T @ D
        # 求解线性系统
        z = np.linalg.solve(Z, w * y)
        # 更新权重
        w = p * (y > z) + (1 - p) * (y <= z)
        
    return z

def asymmetric_least_squares_fast(y, lam=1e6, p=0.01, n_iter=5):
    """
    快速模式：优化的ALS背景扣除算法
    使用稀疏矩阵大幅提高性能，适合大数据集
    """
    L = len(y)
    
    # 构造稀疏二阶差分矩阵
    diag_data = np.ones(L)
    D = sparse.diags([diag_data, -2*diag_data, diag_data], [0, -1, -2], shape=(L, L-2), format='csc')
    
    w = np.ones(L)
    z = np.zeros(L)
    
    # 预计算稀疏矩阵
    DDT = lam * D.dot(D.T)
    
    for i in range(n_iter):
        # 构造稀疏对角权重矩阵
        W = sparse.diags(w, 0, shape=(L, L), format='csc')
        
        # 组合矩阵
        A = W + DDT
        
        # 求解线性系统
        try:
            z = spsolve(A, w * y)
        except Exception as e:
            st.error(f"背景扣除失败: {str(e)}")
            return y  # 返回原始数据作为后备
        
        # 更新权重
        w = p * (y > z) + (1 - p) * (y <= z)
    
    return z

# 修改后的基线处理函数 - 支持手动起点终点选择
def modified_background_correction(angle, intensity, peak_ranges, manual_points=None, mode='precise', progress_callback=None):
    """
    背景扣除函数，支持手动选择起点终点
    manual_points: 字典，格式为 {hkl: (start_angle, end_angle)}
    """
    # 初始化进度
    if progress_callback:
        progress_callback(0, f"开始背景扣除 ({mode}模式)...")
    
    # 根据模式选择算法
    if mode == 'precise':
        # 精确模式
        if progress_callback:
            progress_callback(10, "使用精确模式进行背景扣除...")
        background = asymmetric_least_squares_precise(intensity, lam=1e7, p=0.001, n_iter=10)
    else:  # 'fast'
        # 快速模式
        if progress_callback:
            progress_callback(10, "使用快速模式进行背景扣除...")
        background = asymmetric_least_squares_fast(intensity, lam=1e7, p=0.001, n_iter=5)
    
    if progress_callback:
        progress_callback(30, "背景扣除完成，修正基线中...")
    
    # 创建修正后的基线（初始为ALS基线）
    modified_background = background.copy()
    
    # 对每个峰范围进行线性基线修正
    total_ranges = len(peak_ranges)
    for idx, (low, high, hkl) in enumerate(peak_ranges):
        if progress_callback:
            progress = 30 + int((idx+1)/total_ranges*70)
            progress_callback(progress, f"修正{hkl}晶面背景 ({idx+1}/{total_ranges})")
        
        # 检查是否有手动设置的起点终点
        if manual_points and hkl in manual_points:
            manual_low, manual_high = manual_points[hkl]
            # 使用手动设置的范围
            mask = (angle >= manual_low) & (angle <= manual_high)
        else:
            # 使用默认范围
            mask = (angle >= low) & (angle <= high)
        
        range_angles = angle[mask]
        
        if len(range_angles) < 2:
            continue
        
        # 获取范围起点和终点的背景值
        start_idx = np.where(angle == range_angles[0])[0][0]
        end_idx = np.where(angle == range_angles[-1])[0][0]
        
        start_val = background[start_idx]
        end_val = background[end_idx]
        
        # 创建线性基线（从起点到终点）
        linear_bg = np.linspace(start_val, end_val, len(range_angles))
        
        # 替换该范围内的基线为线性基线
        modified_background[mask] = linear_bg
    
    # 使用修正后的基线进行背景扣除
    corrected_intensity = intensity - modified_background
    
    return corrected_intensity, modified_background, background

# 提取XRD全局特征函数
def extract_global_features(angle, intensity, corrected_intensity):
    """提取XRD数据的全局特征"""
    features = {}
    
    # 基本统计特征
    features['global_max_intensity'] = np.max(intensity)
    features['global_min_intensity'] = np.min(intensity)
    features['global_mean_intensity'] = np.mean(intensity)
    features['global_std_intensity'] = np.std(intensity)
    
    # 背景扣除后的统计特征
    features['corrected_max_intensity'] = np.max(corrected_intensity)
    features['corrected_min_intensity'] = np.min(corrected_intensity)
    features['corrected_mean_intensity'] = np.mean(corrected_intensity)
    features['corrected_std_intensity'] = np.std(corrected_intensity)
    
    # 积分面积特征
    features['total_integral_area'] = simpson(intensity, angle)
    features['corrected_integral_area'] = simpson(corrected_intensity, angle)
    
    # 峰数量特征 (在整个范围内)
    min_prominence = 0.05 * np.max(corrected_intensity)
    peaks, _ = find_peaks(corrected_intensity, prominence=min_prominence)
    features['total_peak_count'] = len(peaks)
    
    # 峰位置分布特征
    if len(peaks) > 0:
        peak_positions = angle[peaks]
        features['mean_peak_position'] = np.mean(peak_positions)
        features['std_peak_position'] = np.std(peak_positions)
        features['min_peak_position'] = np.min(peak_positions)
        features['max_peak_position'] = np.max(peak_positions)
    else:
        features['mean_peak_position'] = np.nan
        features['std_peak_position'] = np.nan
        features['min_peak_position'] = np.nan
        features['max_peak_position'] = np.nan
    
    # 峰高分布特征
    if len(peaks) > 0:
        peak_heights = corrected_intensity[peaks]
        features['mean_peak_height'] = np.mean(peak_heights)
        features['max_peak_height'] = np.max(peak_heights)
    else:
        features['mean_peak_height'] = np.nan
        features['max_peak_height'] = np.nan
    
    # 计算信噪比 (SNR)
    noise_region = np.where((angle > 80) & (angle < 85))[0]  # 假设80-85度范围主要是噪声
    if len(noise_region) > 10:
        noise_std = np.std(corrected_intensity[noise_region])
        if noise_std > 0:
            features['snr'] = np.max(corrected_intensity) / noise_std
        else:
            features['snr'] = np.nan
    else:
        features['snr'] = np.nan
    
    return features

# 计算堆叠层数
def calculate_stacking_layers(peak_position, L_value):
    """计算堆叠层数"""
    λ = 1.5406  # Cu Kα波长 (Å)
    
    # 布拉格公式计算层间距 d = λ/(2sinθ)
    θ = np.deg2rad(peak_position / 2)  # 布拉格角(弧度)
    d_spacing = λ / (2 * np.sin(θ))
    
    # 堆叠层数 = 晶粒尺寸 / 层间距
    if d_spacing > 0:
        stacking_layers = L_value / d_spacing
    else:
        stacking_layers = np.nan
    
    return stacking_layers, d_spacing

# XRD特征提取函数（带进度反馈）
def extract_xrd_features(angle, intensity, peak_ranges=None, manual_points=None, bg_mode='precise', progress_callback=None):
    # 初始化进度
    if progress_callback:
        progress_callback(0, "开始分析XRD数据...")
    
    # 0. 数据预处理 - 修改后的背景扣除
    if progress_callback:
        progress_callback(5, "背景扣除中...")
    
    if peak_ranges is None:
        peak_ranges = []
    
    # 添加进度回调到背景扣除函数
    def bg_callback(progress, message):
        if progress_callback:
            # 背景扣除占总进度的40%
            progress_callback(5 + progress*0.4, message)
    
    corrected_intensity, modified_background, original_background = modified_background_correction(
        angle, intensity, peak_ranges, manual_points=manual_points, mode=bg_mode, progress_callback=bg_callback
    )
    
    # 1. 数据平滑 (使用Savitzky-Golay滤波器)
    if progress_callback:
        progress_callback(45, "平滑数据中...")
    
    window_size = min(51, len(angle) // 10 * 2 + 1)
    if window_size < 5:
        window_size = 5
        
    smooth_intensity = savgol_filter(corrected_intensity, window_size, 3)
    
    # 2. 提取全局特征
    if progress_callback:
        progress_callback(48, "提取全局特征...")
    
    global_features = extract_global_features(angle, intensity, corrected_intensity)
    
    # 3. 特征存储字典
    features = global_features.copy()
    
    # 4. 如果没有指定峰范围，则分析整个范围
    if not peak_ranges:
        peak_ranges = [(np.min(angle), np.max(angle), 'unknown')]
    
    # 5. 遍历所有指定的峰范围
    figs = []
    total_ranges = len(peak_ranges)
    
    # 存储晶粒尺寸用于计算比值
    Lc_value = None
    La_value = None
    
    for i, (low, high, hkl) in enumerate(peak_ranges):
        # 检查是否有手动设置的范围
        if manual_points and hkl in manual_points:
            manual_low, manual_high = manual_points[hkl]
            current_low, current_high = manual_low, manual_high
            range_label = f"{manual_low:.1f}-{manual_high:.1f}°"
        else:
            current_low, current_high = low, high
            range_label = f"{low}-{high}°"
        
        # 更新进度
        progress_percent = 50 + int((i / total_ranges) * 50)
        if progress_percent > 100:
            progress_percent = 100
            
        if progress_callback:
            progress_callback(progress_percent, f"分析{hkl}晶面范围 ({range_label})...")
        
        # 创建当前峰范围的掩码
        mask = (angle >= current_low) & (angle <= current_high)
        range_angles = angle[mask]
        range_intensity = smooth_intensity[mask]
        
        if len(range_angles) < 10:
            st.warning(f"在范围 {current_low}-{current_high} 内数据点不足（{len(range_angles)}个）！")
            prefix = f"peak_{hkl}_" if hkl != 'unknown' else f"peak_{i+1}_"
            features.update({
                f"{prefix}position": np.nan,
                f"{prefix}height": np.nan,
                f"{prefix}fwhm": np.nan,
                f"{prefix}area": np.nan
            })
            continue
        
        # 6. 在当前范围内检测峰
        min_prominence = 0.05 * max(range_intensity)
        min_width = max(2, len(range_angles) * 0.01)
        
        peaks, properties = find_peaks(range_intensity, 
                                      prominence=min_prominence, 
                                      width=min_width,
                                      rel_height=0.5)
        
        if len(peaks) == 0:
            st.warning(f"在范围 {current_low}-{current_high} 内未检测到明显的峰！")
            prefix = f"peak_{hkl}_" if hkl != 'unknown' else f"peak_{i+1}_"
            features.update({
                f"{prefix}position": np.nan,
                f"{prefix}height": np.nan,
                f"{prefix}fwhm": np.nan,
                f"{prefix}area": np.nan
            })
            continue
        
        # 7. 找到当前范围内的主峰 (最高峰)
        main_peak_idx = np.argmax(properties['prominences'])
        main_peak = peaks[main_peak_idx]
        peak_position = range_angles[main_peak]
        peak_height = range_intensity[main_peak]
        
        # 8. 高斯拟合精修峰参数
        try:
            # 选择拟合范围：±3倍半高宽
            half_width = properties['widths'][main_peak_idx] / 2
            fit_start = max(0, int(main_peak - 3 * half_width))
            fit_end = min(len(range_angles), int(main_peak + 3 * half_width))
            
            fit_angles = range_angles[fit_start:fit_end]
            fit_intensity = range_intensity[fit_start:fit_end]
            
            # 初始参数估计
            p0 = [peak_height, peak_position, half_width * (range_angles[1]-range_angles[0])]
            
            # 高斯拟合
            popt, pcov = curve_fit(gaussian, fit_angles, fit_intensity, p0=p0,
                                  maxfev=5000)  # 增加最大评估次数
            
            # 更新峰参数
            peak_position = popt[1]
            peak_height = popt[0]
            fwhm = 2 * np.sqrt(2 * np.log(2)) * abs(popt[2])  # FWHM = 2.355 * σ
            
            # 计算峰面积 (基于拟合曲线)
            x_fine = np.linspace(min(fit_angles), max(fit_angles), 500)
            y_fine = gaussian(x_fine, *popt)
            peak_area = simpson(y_fine, x_fine)
            
            fit_success = True
        except Exception as e:
            st.warning(f"高斯拟合失败: {str(e)}，使用原始方法")
            widths = peak_widths(range_intensity, [main_peak], rel_height=0.5)
            fwhm = widths[0][0] * (range_angles[1]-range_angles[0])
            
            # 计算峰面积 (自适应范围：±5倍半高宽)
            half_width_points = int(5 * widths[0][0])
            start_idx = max(0, main_peak - half_width_points)
            end_idx = min(len(range_angles), main_peak + half_width_points)
            peak_area = simpson(range_intensity[start_idx:end_idx], range_angles[start_idx:end_idx])
            fit_success = False
        
        # 9. 使用晶面指数作为前缀
        prefix = f"peak_{hkl}_" if hkl != 'unknown' else f"peak_{i+1}_"
        
        # 10. 将基本特征添加到字典
        features.update({
            f"{prefix}position": peak_position,
            f"{prefix}height": peak_height,
            f"{prefix}fwhm": fwhm,
            f"{prefix}area": peak_area
        })
        
        # 11. 晶粒尺寸计算 (仅对特定晶面)
        λ = 1.5406  # Cu Kα波长 (Å)
        θ = np.deg2rad(peak_position / 2)  # 布拉格角(弧度)
        β = np.deg2rad(fwhm)  # 半高宽(弧度)
        
        # 避免过小的β值导致异常大的晶粒尺寸
        if β < np.deg2rad(0.1):  # 0.1度阈值
            st.warning(f"半高宽过小({fwhm:.4f}度)，晶粒尺寸计算可能不准确")
            β = np.deg2rad(0.5)  # 设置最小值
        
        if hkl == '002':
            # 计算Lc (沿c轴的晶粒尺寸)
            K = 0.89  # 硬碳材料推荐值
            Lc = K * λ / (β * np.cos(θ))
            features[f"{prefix}Lc"] = Lc
            Lc_value = Lc  # 存储用于比值计算
            
            # 计算堆叠层数
            stacking_layers, d_spacing = calculate_stacking_layers(peak_position, Lc)
            features[f"{prefix}stacking_layers"] = stacking_layers
            features[f"{prefix}d_spacing"] = d_spacing
            
        elif hkl == '100':
            # 计算La (沿a轴的晶粒尺寸)
            K = 1.84  # 形状因子
            La = K * λ / (β * np.cos(θ))
            features[f"{prefix}La"] = La
            La_value = La  # 存储用于比值计算
            
            # 计算堆叠层数
            stacking_layers, d_spacing = calculate_stacking_layers(peak_position, La)
            features[f"{prefix}stacking_layers"] = stacking_layers
            features[f"{prefix}d_spacing"] = d_spacing
        
        # 12. 可视化当前峰范围 - 优化图表显示
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), dpi=100)
        
        # 设置全局字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 原始数据和平滑数据
        ax1.plot(angle, intensity, 'b-', label='原始数据', alpha=0.6)
        ax1.plot(angle, original_background, 'k:', label='初始背景', linewidth=1.5, alpha=0.7)
        ax1.plot(angle, modified_background, 'k-', label='修正背景', linewidth=2)
        ax1.plot(angle, corrected_intensity, 'g-', label='扣除背景', alpha=0.8)
        ax1.plot(angle, smooth_intensity, 'r-', label='平滑数据', linewidth=1.5)
        ax1.axvline(peak_position, color='m', linestyle='--', label='峰位置')
        
        # 标记手动设置的起点终点
        if manual_points and hkl in manual_points:
            manual_low, manual_high = manual_points[hkl]
            ax1.axvline(manual_low, color='orange', linestyle='--', alpha=0.7, label='手动起点')
            ax1.axvline(manual_high, color='orange', linestyle='--', alpha=0.7, label='手动终点')
        
        ax1.set_xlabel('2θ (度)', fontsize=12)
        ax1.set_ylabel('强度 (a.u.)', fontsize=12)
        ax1.set_title(f'XRD谱线预处理 ({range_label}, {hkl}晶面)', fontsize=14, pad=20)
        ax1.legend(loc='best', fontsize=10, framealpha=0.7)
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.tick_params(axis='both', which='major', labelsize=10)
        
        # 峰区域放大图
        ax2.plot(range_angles, range_intensity, 'b-', label='平滑数据')
        ax2.plot(range_angles[main_peak], range_intensity[main_peak], 'ro', label='主峰')
        ax2.axvline(peak_position, color='g', linestyle='--', label='峰位置')
        
        # 绘制拟合曲线（如果成功）
        if fit_success:
            ax2.plot(x_fine, y_fine, 'm-', label='高斯拟合', linewidth=2)
        
        # 标记手动设置的起点终点
        if manual_points and hkl in manual_points:
            manual_low, manual_high = manual_points[hkl]
            ax2.axvline(manual_low, color='orange', linestyle='--', alpha=0.7, label='手动起点')
            ax2.axvline(manual_high, color='orange', linestyle='--', alpha=0.7, label='手动终点')
        
        ax2.set_xlabel('2θ (度)', fontsize=12)
        ax2.set_ylabel('强度 (a.u.)', fontsize=12)
        ax2.set_title(f'峰特征提取 ({range_label}, {hkl}晶面)', fontsize=14, pad=20)
        ax2.legend(loc='best', fontsize=10, framealpha=0.7)
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.tick_params(axis='both', which='major', labelsize=10)
        
        # 调整布局
        plt.tight_layout(pad=3.0)
        figs.append(fig)
    
    # 添加比值特征
    if Lc_value is not None and La_value is not None:
        features['La_Lc_ratio'] = La_value / Lc_value
    
    # 添加002峰面积与100峰面积比值
    if 'peak_002_area' in features and 'peak_100_area' in features:
        if features['peak_100_area'] > 0:
            features['A002_A100_ratio'] = features['peak_002_area'] / features['peak_100_area']
    
    # 添加002峰高与100峰高比值
    if 'peak_002_height' in features and 'peak_100_height' in features:
        if features['peak_100_height'] > 0:
            features['H002_H100_ratio'] = features['peak_002_height'] / features['peak_100_height']
    
    if progress_callback:
        progress_callback(100, "分析完成！")
    
    return features, figs

# 文件下载函数
def get_table_download_link(df, filename):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">下载CSV文件</a>'
    return href

# 格式化数据框，只对数值列应用格式化
def format_dataframe(df):
    """格式化数据框，只对数值列应用格式化"""
    # 创建一个副本
    formatted_df = df.copy()
    
    # 对数值列进行格式化
    for col in formatted_df.columns:
        if pd.api.types.is_numeric_dtype(formatted_df[col]):
            # 对数值列应用格式化
            formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
    
    return formatted_df

# 单个文件分析函数
def analyze_single_file(uploaded_file, peak_ranges, manual_points, bg_mode, progress_callback=None):
    """分析单个XRD文件"""
    try:
        # 读取文件
        file_ext = os.path.splitext(uploaded_file.name)[1].lower()
        
        if file_ext == '.csv':
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        # 检查数据格式
        if len(df.columns) < 2:
            st.error(f"文件 {uploaded_file.name} 格式错误，无法处理")
            return None, None, None
            
        # 提取角度和强度数据
        angle_col = df.columns[0]
        intensity_col = df.columns[1]
        
        angles = df[angle_col].values
        raw_intensities = df[intensity_col].values
        
        # 数据归一化
        intensities = normalize_data(raw_intensities)
        
        # 提取特征
        features, figs = extract_xrd_features(
            angles, 
            intensities, 
            peak_ranges, 
            manual_points=manual_points,
            bg_mode=bg_mode,
            progress_callback=progress_callback
        )
        
        # 添加文件名信息
        features['filename'] = uploaded_file.name
        
        return features, figs, (angles, intensities)
        
    except Exception as e:
        st.error(f"处理文件 {uploaded_file.name} 时出错: {str(e)}")
        return None, None, None

# 批量处理函数
def batch_process_files(uploaded_files, peak_ranges, manual_points, bg_mode, progress_callback=None):
    """批量处理多个XRD文件"""
    all_results = []
    all_figs = []  # 存储每个文件的图表
    all_raw_data = []  # 存储每个文件的原始数据
    
    for i, uploaded_file in enumerate(uploaded_files):
        if progress_callback:
            progress_callback((i * 100) / len(uploaded_files), f"开始处理文件 {i+1}/{len(uploaded_files)}: {uploaded_file.name}")
        
        # 分析单个文件
        features, figs, raw_data = analyze_single_file(
            uploaded_file, 
            peak_ranges, 
            manual_points,
            bg_mode=bg_mode,
            progress_callback=lambda p, m: progress_callback(
                (i * 80 + p * 0.8) / len(uploaded_files), 
                f"处理 {uploaded_file.name}: {m}"
            ) if progress_callback else None
        )
        
        if features and figs:
            all_results.append(features)
            all_figs.append((uploaded_file.name, figs))  # 存储文件名和对应的图表
            all_raw_data.append((uploaded_file.name, raw_data))  # 存储原始数据
            
        if progress_callback:
            progress = ((i + 1) * 100) / len(uploaded_files)
            progress_callback(progress, f"完成文件 {i+1}/{len(uploaded_files)}: {uploaded_file.name}")
                
    return all_results, all_figs, all_raw_data

# 显示原始数据图
def plot_raw_data(angles, intensities, filename):
    """显示原始数据归一化后的图形"""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    
    # 设置全局字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    
    ax.plot(angles, intensities, 'b-', linewidth=2, alpha=0.8)
    ax.set_xlabel('2θ (度)', fontsize=12)
    ax.set_ylabel('归一化强度 (a.u.)', fontsize=12)
    ax.set_title(f'原始XRD数据 - {filename}', fontsize=14, pad=20)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    plt.tight_layout()
    return fig

# 主应用
def main():
    # 设置页面配置
    st.set_page_config(
        page_title="XRD特征提取工具 - 硬碳专项分析",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # 简约高级的CSS样式
    st.markdown("""
    <style>
    .main-title {
        font-size: 2.5rem;
        color: #1a1a1a;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 600;
        letter-spacing: -0.5px;
    }
    .sub-title {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    .section-header {
        font-size: 1.4rem;
        color: #2c3e50;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #3498db;
        font-weight: 600;
    }
    .subsection-header {
        font-size: 1.1rem;
        color: #34495e;
        margin: 1.5rem 0 0.8rem 0;
        font-weight: 600;
    }
    .parameter-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        border-left: 4px solid #3498db;
    }
    .result-card {
        background: white;
        border-radius: 8px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        border: 1px solid #e1e8ed;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .metric-card {
        background: white;
        border-radius: 6px;
        padding: 1rem;
        margin: 0.5rem;
        border: 1px solid #e1e8ed;
        text-align: center;
    }
    .file-selector {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid #dee2e6;
    }
    .progress-container {
        background: white;
        border-radius: 8px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        border: 1px solid #e1e8ed;
    }
    .analysis-tabs {
        margin-top: 1rem;
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.6rem 1.5rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #2980b9;
        transform: translateY(-1px);
    }
    .download-button {
        background-color: #27ae60 !important;
    }
    .download-button:hover {
        background-color: #219653 !important;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    .manual-adjust-note {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 6px;
        padding: 1rem;
        margin-bottom: 1rem;
        color: #856404;
    }
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    .chart-container {
        background: white;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid #e1e8ed;
    }
    .reanalyze-section {
        background: #fff5f5;
        border: 1px solid #fed7d7;
        border-radius: 8px;
        padding: 1.2rem;
        margin-top: 1.5rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # 初始化session state
    if 'batch_results' not in st.session_state:
        st.session_state.batch_results = None
    if 'batch_figs' not in st.session_state:
        st.session_state.batch_figs = None
    if 'batch_raw_data' not in st.session_state:
        st.session_state.batch_raw_data = None
    if 'current_file_index' not in st.session_state:
        st.session_state.current_file_index = 0
    if 'reanalyze_files' not in st.session_state:
        st.session_state.reanalyze_files = {}
    
    # 主标题
    st.markdown('<h1 class="main-title">XRD智能特征提取平台</h1>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">硬碳材料专项分析 | 批量处理 | 智能特征提取</div>', unsafe_allow_html=True)
    
    # 侧边栏 - 所有设置功能
    with st.sidebar:
        st.markdown('<div class="section-header">分析设置</div>', unsafe_allow_html=True)
        
        # 文件上传
        st.markdown("#### 上传数据文件")
        uploaded_files = st.file_uploader(
            "选择XRD数据文件", 
            type=["csv", "xlsx", "xls"], 
            accept_multiple_files=True,
            label_visibility="collapsed"
        )
        
        if uploaded_files:
            st.info(f"已选择 {len(uploaded_files)} 个文件")
            
            # 峰范围设置
            st.markdown("#### 分析参数")
            default_ranges = [(15, 35, '002'), (38, 48, '100')]
            custom_ranges = []
            
            use_default = st.checkbox("使用默认峰范围", value=True)
            
            if not use_default:
                st.text_area("自定义峰范围", "15 35 002\n38 48 100", 
                           height=100, 
                           help="每行一个范围，格式：起始角度 结束角度 晶面指数")
            
            peak_ranges = default_ranges if use_default else custom_ranges
            
            # 手动基线调整
            st.markdown("#### 基线调整")
            st.markdown('<div class="manual-adjust-note">可根据XRD谱线实际情况手动调整基线范围</div>', unsafe_allow_html=True)
            
            manual_points = {}
            
            for i, (low, high, hkl) in enumerate(peak_ranges):
                col1, col2 = st.columns(2)
                with col1:
                    manual_start = st.number_input(
                        f"{hkl}起点", 
                        value=float(low),
                        min_value=float(10),
                        max_value=float(50),
                        step=0.1,
                        key=f"start_{hkl}"
                    )
                with col2:
                    manual_end = st.number_input(
                        f"{hkl}终点", 
                        value=float(high),
                        min_value=float(10),
                        max_value=float(50),
                        step=0.1,
                        key=f"end_{hkl}"
                    )
                
                if manual_start != low or manual_end != high:
                    manual_points[hkl] = (manual_start, manual_end)
            
            # 背景扣除模式
            st.markdown("#### 处理模式")
            bg_mode = st.radio(
                "背景扣除模式",
                options=['precise', 'fast'],
                format_func=lambda x: {'precise': '精确模式', 'fast': '快速模式'}[x]
            )
            
            # 开始分析按钮
            st.markdown("---")
            if st.button("开始批量分析", type="primary", use_container_width=True):
                if uploaded_files:
                    progress_container = st.container()
                    with progress_container:
                        st.markdown('<div class="progress-container">', unsafe_allow_html=True)
                        st.markdown("**分析进度**")
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    def update_progress(progress_int, message):
                        progress_float = progress_int / 100.0
                        progress_bar.progress(progress_float)
                        status_text.text(message)
                    
                    with st.spinner("正在分析XRD数据..."):
                        all_results, all_figs, all_raw_data = batch_process_files(
                            uploaded_files, 
                            peak_ranges, 
                            manual_points,
                            bg_mode=bg_mode,
                            progress_callback=update_progress
                        )
                        
                        progress_container.empty()
                        
                        if all_results:
                            st.session_state.batch_results = all_results
                            st.session_state.batch_figs = all_figs
                            st.session_state.batch_raw_data = all_raw_data
                            st.session_state.current_file_index = 0
                            st.success("分析完成！")
                        else:
                            st.error("分析失败，请检查数据和参数设置")
                else:
                    st.warning("请先上传数据文件")
    
    # 主内容区域 - 只显示分析结果
    if st.session_state.batch_results:
        st.markdown('<div class="section-header">批量分析结果</div>', unsafe_allow_html=True)
        
        # 批量汇总结果
        result_df = pd.DataFrame(st.session_state.batch_results)
        formatted_df = format_dataframe(result_df)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("处理文件数量", len(st.session_state.batch_results))
        with col2:
            st.metric("提取特征数量", len(result_df.columns) - 1)
        with col3:
            st.metric("成功率", f"{(len(st.session_state.batch_results)/len(uploaded_files))*100:.1f}%")
        
        st.dataframe(formatted_df, use_container_width=True)
        
        # 批量下载
        download_filename = f"XRD_batch_analysis_{len(uploaded_files)}_files.csv"
        st.markdown(get_table_download_link(result_df, download_filename), unsafe_allow_html=True)
        
        st.markdown('<div class="section-header">各文件详细分析结果</div>', unsafe_allow_html=True)
        
        # 文件选择器 - 解决多文件切换问题
        st.markdown('<div class="file-selector">', unsafe_allow_html=True)
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            file_names = [name for name, _ in st.session_state.batch_figs]
            selected_file = st.selectbox(
                "选择要查看的文件",
                options=file_names,
                index=st.session_state.current_file_index
            )
            st.session_state.current_file_index = file_names.index(selected_file)
        
        with col2:
            if st.button("上一个文件") and st.session_state.current_file_index > 0:
                st.session_state.current_file_index -= 1
                st.rerun()
        
        with col3:
            if st.button("下一个文件") and st.session_state.current_file_index < len(file_names) - 1:
                st.session_state.current_file_index += 1
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 当前文件详细分析结果
        current_index = st.session_state.current_file_index
        filename, file_figs = st.session_state.batch_figs[current_index]
        features_dict = st.session_state.batch_results[current_index]
        
        # 显示原始数据图
        if st.session_state.batch_raw_data:
            raw_filename, raw_data = st.session_state.batch_raw_data[current_index]
            if raw_data:
                angles, intensities = raw_data
                raw_fig = plot_raw_data(angles, intensities, filename)
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                st.markdown("#### 原始数据图")
                st.pyplot(raw_fig)
                st.markdown('</div>', unsafe_allow_html=True)
        
        # 特征展示 - 使用网格布局
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown("#### 特征分析结果")
        
        # 全局特征
        st.markdown("##### 全局特征")
        global_cols = st.columns(3)
        global_features = [
            ('total_peak_count', '总峰数量'),
            ('global_mean_intensity', '平均强度'), 
            ('corrected_mean_intensity', '校正强度'),
            ('total_integral_area', '总积分面积'),
            ('corrected_integral_area', '校正面积'),
            ('snr', '信噪比')
        ]
        
        for i, (key, name) in enumerate(global_features):
            if key in features_dict:
                with global_cols[i % 3]:
                    st.metric(name, f"{features_dict[key]:.4f}")
        
        # 晶面特征 - 并排显示
        st.markdown("##### 晶面特征")
        crystal_cols = st.columns(2)
        
        with crystal_cols[0]:
            st.markdown("**002晶面**")
            if 'peak_002_position' in features_dict:
                st.metric("峰位置", f"{features_dict['peak_002_position']:.2f}°")
            if 'peak_002_fwhm' in features_dict:
                st.metric("FWHM", f"{features_dict['peak_002_fwhm']:.2f}°")
            if 'peak_002_Lc' in features_dict:
                st.metric("Lc晶粒尺寸", f"{features_dict['peak_002_Lc']:.2f} Å")
            if 'peak_002_area' in features_dict:
                st.metric("峰面积", f"{features_dict['peak_002_area']:.4f}")
            if 'peak_002_d_spacing' in features_dict:
                st.metric("层间距", f"{features_dict['peak_002_d_spacing']:.4f} Å")
        
        with crystal_cols[1]:
            st.markdown("**100晶面**")
            if 'peak_100_position' in features_dict:
                st.metric("峰位置", f"{features_dict['peak_100_position']:.2f}°")
            if 'peak_100_fwhm' in features_dict:
                st.metric("FWHM", f"{features_dict['peak_100_fwhm']:.2f}°")
            if 'peak_100_La' in features_dict:
                st.metric("La晶粒尺寸", f"{features_dict['peak_100_La']:.2f} Å")
            if 'peak_100_area' in features_dict:
                st.metric("峰面积", f"{features_dict['peak_100_area']:.4f}")
            if 'peak_100_d_spacing' in features_dict:
                st.metric("层间距", f"{features_dict['peak_100_d_spacing']:.4f} Å")
        
        # 比值特征
        st.markdown("##### 比值特征")
        ratio_cols = st.columns(3)
        with ratio_cols[0]:
            if 'La_Lc_ratio' in features_dict:
                st.metric("La/Lc比值", f"{features_dict['La_Lc_ratio']:.4f}")
        with ratio_cols[1]:
            if 'A002_A100_ratio' in features_dict:
                st.metric("面积比(A002/A100)", f"{features_dict['A002_A100_ratio']:.4f}")
        with ratio_cols[2]:
            if 'H002_H100_ratio' in features_dict:
                st.metric("高度比(H002/H100)", f"{features_dict['H002_H100_ratio']:.4f}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 分析图表 - 并排显示
        st.markdown("#### 分析图表")
        if file_figs:
            # 显示第一组图表（如果有多个）
            cols = st.columns(2)
            with cols[0]:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                st.markdown("**XRD谱线预处理**")
                st.pyplot(file_figs[0])
                st.markdown('</div>', unsafe_allow_html=True)
            
            with cols[1]:
                if len(file_figs) > 1:
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    st.markdown("**峰特征提取**")
                    st.pyplot(file_figs[1])
                    st.markdown('</div>', unsafe_allow_html=True)
        
        # 重新分析功能
        st.markdown('<div class="reanalyze-section">', unsafe_allow_html=True)
        st.markdown("#### 重新分析此文件")
        
        with st.expander("调整分析参数"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**002晶面范围**")
                reanalyze_002_start = st.number_input(
                    "002起点角度", 
                    value=float(manual_points.get('002', (15, 35))[0]),
                    min_value=float(10),
                    max_value=float(50),
                    step=0.1,
                    key=f"reanalyze_002_start_{current_index}"
                )
                reanalyze_002_end = st.number_input(
                    "002终点角度", 
                    value=float(manual_points.get('002', (15, 35))[1]),
                    min_value=float(10),
                    max_value=float(50),
                    step=0.1,
                    key=f"reanalyze_002_end_{current_index}"
                )
            
            with col2:
                st.markdown("**100晶面范围**")
                reanalyze_100_start = st.number_input(
                    "100起点角度", 
                    value=float(manual_points.get('100', (38, 48))[0]),
                    min_value=float(10),
                    max_value=float(50),
                    step=0.1,
                    key=f"reanalyze_100_start_{current_index}"
                )
                reanalyze_100_end = st.number_input(
                    "100终点角度", 
                    value=float(manual_points.get('100', (38, 48))[1]),
                    min_value=float(10),
                    max_value=float(50),
                    step=0.1,
                    key=f"reanalyze_100_end_{current_index}"
                )
            
            reanalyze_bg_mode = st.radio(
                "背景扣除模式",
                options=['precise', 'fast'],
                format_func=lambda x: {'precise': '精确模式', 'fast': '快速模式'}[x],
                horizontal=True,
                key=f"reanalyze_bg_mode_{current_index}"
            )
            
            if st.button("重新分析", key=f"reanalyze_btn_{current_index}"):
                with st.spinner("重新分析中..."):
                    new_manual_points = {
                        '002': (reanalyze_002_start, reanalyze_002_end),
                        '100': (reanalyze_100_start, reanalyze_100_end)
                    }
                    
                    reanalyze_features, reanalyze_figs, _ = analyze_single_file(
                        uploaded_files[current_index],
                        peak_ranges,
                        new_manual_points,
                        reanalyze_bg_mode
                    )
                    
                    if reanalyze_features and reanalyze_figs:
                        st.session_state.batch_results[current_index] = reanalyze_features
                        st.session_state.batch_figs[current_index] = (filename, reanalyze_figs)
                        st.success("重新分析完成！")
                        st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    else:
        # 未分析时的提示信息
        st.markdown("""
        <div style="text-align: center; padding: 4rem 2rem; color: #666;">
            <h3>欢迎使用XRD智能特征提取平台</h3>
            <p>请在左侧边栏上传XRD数据文件并设置分析参数</p>
            <p>支持批量处理多个文件，自动提取晶粒尺寸、堆叠层数等关键特征</p>
        </div>
        """, unsafe_allow_html=True)
        
        # 功能特点展示
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h4>批量处理</h4>
                <p>支持同时分析多个XRD文件</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h4>智能分析</h4>
                <p>自动提取晶粒尺寸等特征</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h4>专业算法</h4>
                <p>基于硬碳材料优化的分析算法</p>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()