# -*- coding: utf-8 -*-
"""
用于逐小时降水临近预报的物理约束Geo-ConvLSTM微调模型。

主要设置：
1. 从配套8小时Geo-ConvLSTM基线模型的最优检查点开始微调；
2. 数据处理、模型结构和数据损失与基线保持一致；
3. 通过3×3邻域匹配、4×4和8×8多尺度下界及H05--H08累计下界施加物理约束；
4. 使用训练集残差下分位数构造单侧下界，不将P_budget视为精确降水标签；
5. 物理约束仅作用于H05--H08，不设置物理上界或干燥惩罚；
6. 微调检查点按照验证集H05--H08强降水CSI/HSS综合得分选择。

输入张量：
    X: (B, history_len, C, H, W)
    Y: (B, horizon, 1, H, W)
    M: (B, horizon, 1, H, W)

输出：
    pred: (B, horizon, 1, H, W)

依赖：
pip install xarray netCDF4 numpy pandas matplotlib torch

四个动态变量分别存放在VARIABLE_FILES中。脚本按时间和经纬度取交集，
再在内存中按年份及4--9月切分。物理变量文件必须覆盖相同区域和时间。
"""

from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# =========================================================
# 1. 配置参数
# =========================================================

PROJECT_DIR = Path(__file__).resolve().parent
VARIABLE_DATA_DIR = PROJECT_DIR / "data"
VARIABLE_FILES = {
    "tp_hourly": VARIABLE_DATA_DIR / "tp_hourly.nc",
    "u10": VARIABLE_DATA_DIR / "u10.nc",
    "v10": VARIABLE_DATA_DIR / "v10.nc",
    "d2m": VARIABLE_DATA_DIR / "d2m.nc",
}
SELECT_MONTHS = [4, 5, 6, 7, 8, 9]
TP_INPUT_UNITS = "m_per_hour"  # tp_hourly.nc为逐小时量，但单位是m
OUT_DIR = PROJECT_DIR / "outputs" / "geoconvlstm_physics_finetune_8h"

# 按年份严格隔离训练集、验证集和测试集，避免时间信息泄漏。
TRAIN_YEARS = [2020, 2021]
VAL_YEARS = [2022]
TEST_YEARS = [2023]

# 动态输入变量和预测目标。
INPUT_VARS = ["tp_hourly", "u10", "v10", "d2m"]
TARGET_VAR = "tp_hourly"

# =========================================================
# 水汽收支物理约束配置
# =========================================================
EVAP_PATH = VARIABLE_DATA_DIR / "evaporation.nc"
TCWV_PATH = VARIABLE_DATA_DIR / "tcwv.nc"
VIMD_PATH = VARIABLE_DATA_DIR / "vimd.nc"

# 当前VIMD文件保存每个时间间隔的累积量，因此不再乘以时间步长。
VIMD_TIME_INTEGRATION_MODE = "none"  # "none"或"multiply_dt"
VIMD_CONVENTION = "divergence"       # "divergence" or "convergence"

EVAP_INPUT_MODE = "hourly_amount_m"  # ERA5 e, m per interval
EVAP_SIGN_MODE = "era5_negative"              # auto: raw mean < 0 -> E=-e, else E=e
EXPECTED_DT_SECONDS = 3600.0
DT_TOLERANCE_SECONDS = 1.0

# =========================================================
# 物理约束微调配置
# =========================================================
# 必须从同结构的8小时基线模型最优权重开始微调。
BASELINE_OUT_DIR = PROJECT_DIR / "outputs" / "geoconvlstm_baseline_8h"
BASELINE_CHECKPOINT = BASELINE_OUT_DIR / "best_model.pt"

# =========================================================
# 结构化Physics约束：邻域匹配 + 多尺度 + H05-H08累计
# =========================================================
# Physics仅作用于H05-H08，数据损失仍对H01-H08完全一致。
PHYS_HORIZON_WEIGHTS = [
    0.0, 0.0, 0.0, 0.0,
    1.0, 1.0, 1.0, 1.0,
]

# 训练集残差r=P_true_patch-alpha*P_budget_patch的下分位数。
# 该分位数用于构造多尺度及累计单侧下界。
PHYS_PATCH_SIZE = 4
PHYS_PATCH_MIN_VALID_FRACTION = 0.75
PBUDGET_RESIDUAL_LOWER_QUANTILE = 0.10
PBUDGET_RESIDUAL_CHUNK_TIMES = 256
PBUDGET_RESIDUAL_SAMPLES_PER_CHUNK = 50000
PBUDGET_RESIDUAL_MAX_SAMPLES = 2000000
PBUDGET_RESIDUAL_RANDOM_SEED = 42
PBUDGET_MIN_LOWER_OFFSET_MM = 0.05

# 邻域匹配：
# 对P_budget高置信格点，不要求模型在完全同一格点达到阈值；
# 只要3×3邻域内存在对应强度预测即可。
PHYS_NEIGHBORHOOD_KERNEL = 3
PHYS_NEIGHBORHOOD_THRESHOLDS_MM = [4.0, 8.0]
PHYS_NEIGHBORHOOD_THRESHOLD_WEIGHTS = [0.6, 0.4]
PHYS_NEIGHBORHOOD_SOFTNESS_MM = 0.75

# 多尺度区域下界：同时在4×4和8×8尺度约束大尺度水量背景。
PHYS_MULTISCALE_PATCH_SIZES = [4, 8]
PHYS_MULTISCALE_WEIGHTS = [0.6, 0.4]
PHYS_MULTISCALE_PBUDGET_THRESHOLD_MM = 1.0

# H05-H08累计约束：
# 允许降水在后4个小时之间重新分配，只限制累计水量不能过度衰减。
PHYS_CUMULATIVE_PATCH_SIZES = [4, 8]
PHYS_CUMULATIVE_SCALE_WEIGHTS = [0.6, 0.4]
PHYS_CUMULATIVE_PBUDGET_THRESHOLD_MM = 4.0
PHYS_CUMULATIVE_MIN_VALID_FRACTION = 0.75

# 三类Physics子项的内部权重。
PHYS_NEIGHBORHOOD_COMPONENT_WEIGHT = 0.25
PHYS_MULTISCALE_COMPONENT_WEIGHT = 0.35
PHYS_CUMULATIVE_COMPONENT_WEIGHT = 0.40

# 总Physics权重和ramp。
PHYS_START_EPOCH = 1
PHYS_RAMP_EPOCHS = 5
PHYS_LOWER_BOUND_WEIGHT = 0.02
PHYS_HUBER_BETA = 0.20

# 不使用干燥惩罚，也不设置物理上界。

# =========================================================
# 平衡长尾与事件识别的联合损失
# =========================================================
# 以较温和的加权 Huber 代替 1/1.5/3/6/10 的加权 MSE，减少极端格点梯度爆炸。
RAIN_WEIGHT_THRESHOLDS = [0.1, 1.0, 4.0, 8.0]
RAIN_WEIGHT_VALUES = [1.0, 1.2, 2.0, 4.0, 6.0]
NORMALIZE_RAIN_WEIGHTS = True
DATA_HUBER_BETA = 1.0

# Soft CSI按预报时效和降水阈值分别计算。
SOFT_EVENT_THRESHOLDS = [1.0, 4.0, 8.0]
SOFT_EVENT_WEIGHTS = [0.2, 0.5, 0.3]
HORIZON_LOSS_WEIGHTS = [1.0] * 8
SOFT_THRESHOLD_SHARPNESS = 3.0
SOFT_CSI_LOSS_WEIGHT = 0.05

# 独立雨/无雨辅助头直接输出logits。
USE_RAIN_EVENT_HEAD = True
RAIN_EVENT_THRESHOLD = 0.1
RAIN_EVENT_LOSS_WEIGHT = 0.05
RAIN_EVENT_POS_WEIGHT = 1.0

# 是否用训练集均值校正 p_budget_positive。
# alpha = mean(tp_hourly_train) / mean(p_budget_positive_train)
CALIBRATE_PBUDGET_POSITIVE = True
PBUDGET_ALPHA_MIN = 0.05
PBUDGET_ALPHA_MAX = 3.00

# 是否启用静态地理信息嵌入。
USE_GEO_EMBEDDING = True

# 地理信息通道依次为lat_norm、lon_norm、sin_lat、cos_lat、sin_lon和cos_lon。

history_len = 8
horizon = 8
# 将空间尺寸补齐为偶数，以适配一次2倍下采样；原始0.25°网格不变。
PAD_MULTIPLE = 2

# 微调学习率低于基线模型的初始学习率。
batch_size = 16
max_epochs = 20
early_stopping_patience = 8
learning_rate = 5e-5
weight_decay = 0.0
lr_scheduler_factor = 0.5
lr_scheduler_patience = 2
min_learning_rate = 1e-6
random_seed = 42
num_workers = 0

VALIDATION_HARD_THRESHOLDS = [4.0, 8.0]
EARLY_HORIZON_START = 0
EARLY_HORIZON_END = 4
LATE_HORIZON_START = 4
LATE_HORIZON_END = 8
EVENT_SELECTION_WEIGHTS = {
    "CSI@4": 0.30,
    "CSI@8": 0.30,
    "HSS@4": 0.20,
    "HSS@8": 0.20,
}

# 模型参数
geo_embed_channels = 8
latent_channels = 64
dropout_rate = 0.0

# 评价和绘图
rain_thresholds = [0.1, 1.0, 4.0, 8.0]
num_plot_cases = 5
plot_case_indices = None  # 例如 [0, 10, 25]；None 表示自动画前 num_plot_cases 个
plot_dpi = 200

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 配置安全检查：避免8小时输出只被前4项权重静默监督。
if len(HORIZON_LOSS_WEIGHTS) != horizon:
    raise ValueError(
        f"HORIZON_LOSS_WEIGHTS长度={len(HORIZON_LOSS_WEIGHTS)}，但horizon={horizon}。"
    )
if len(RAIN_WEIGHT_VALUES) != len(RAIN_WEIGHT_THRESHOLDS) + 1:
    raise ValueError("RAIN_WEIGHT_VALUES长度应比RAIN_WEIGHT_THRESHOLDS多1。")
if len(SOFT_EVENT_THRESHOLDS) != len(SOFT_EVENT_WEIGHTS):
    raise ValueError("SOFT_EVENT_THRESHOLDS与SOFT_EVENT_WEIGHTS长度不一致。")
if len(PHYS_HORIZON_WEIGHTS) != horizon:
    raise ValueError(
        f"PHYS_HORIZON_WEIGHTS长度={len(PHYS_HORIZON_WEIGHTS)}，但horizon={horizon}。"
    )
if not (0 <= EARLY_HORIZON_START < EARLY_HORIZON_END <= horizon):
    raise ValueError("EARLY horizon slice配置非法。")
if not (0 <= LATE_HORIZON_START < LATE_HORIZON_END <= horizon):
    raise ValueError("LATE horizon slice配置非法。")
if PHYS_NEIGHBORHOOD_KERNEL % 2 == 0:
    raise ValueError("PHYS_NEIGHBORHOOD_KERNEL必须为奇数。")
if len(PHYS_NEIGHBORHOOD_THRESHOLDS_MM) != len(
    PHYS_NEIGHBORHOOD_THRESHOLD_WEIGHTS
):
    raise ValueError("邻域阈值与对应权重长度不一致。")
if len(PHYS_MULTISCALE_PATCH_SIZES) != len(PHYS_MULTISCALE_WEIGHTS):
    raise ValueError("多尺度patch尺寸与对应权重长度不一致。")
if len(PHYS_CUMULATIVE_PATCH_SIZES) != len(
    PHYS_CUMULATIVE_SCALE_WEIGHTS
):
    raise ValueError("累计约束patch尺寸与对应权重长度不一致。")
if not np.isclose(
    PHYS_NEIGHBORHOOD_COMPONENT_WEIGHT
    + PHYS_MULTISCALE_COMPONENT_WEIGHT
    + PHYS_CUMULATIVE_COMPONENT_WEIGHT,
    1.0,
):
    raise ValueError("三类物理约束子项权重之和必须为1。")


# =========================================================
# 2. 随机种子
# =========================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(random_seed)


# =========================================================
# 3. 通用工具函数
# =========================================================

def open_nc(path):
    try:
        return xr.open_dataset(path, engine="h5netcdf")
    except Exception:
        return xr.open_dataset(path)


def find_time_name(ds):
    for name in ["valid_time", "time"]:
        if name in ds.coords or name in ds.dims:
            return name
    raise ValueError("无法识别时间坐标名称。")


def find_lat_lon_names(ds):
    if "latitude" in ds.coords or "latitude" in ds.dims:
        lat_name = "latitude"
    elif "lat" in ds.coords or "lat" in ds.dims:
        lat_name = "lat"
    else:
        raise ValueError("无法识别纬度坐标名称。")

    if "longitude" in ds.coords or "longitude" in ds.dims:
        lon_name = "longitude"
    elif "lon" in ds.coords or "lon" in ds.dims:
        lon_name = "lon"
    else:
        raise ValueError("无法识别经度坐标名称。")

    return lat_name, lon_name


def _collapse_extra_dims(da, var_name):
    """处理CDS文件中可能出现的单例维或expver维，只保留time/lat/lon。"""
    for dim in list(da.dims):
        if dim in ["time", "lat", "lon"]:
            continue
        if da.sizes[dim] == 1:
            da = da.isel({dim: 0}, drop=True)
        elif dim.lower() == "expver":
            merged = da.isel({dim: 0}, drop=True)
            for i in range(1, da.sizes[dim]):
                merged = merged.combine_first(da.isel({dim: i}, drop=True))
            da = merged
        else:
            raise ValueError(f"变量 {var_name} 存在无法自动处理的额外维度 {dim}: {da.sizes[dim]}")
    return da


def load_variable_year(var_name, year):
    """从单变量全时段NC中只读取指定年份4--9月，并规范为(time,lat,lon)。"""
    path = VARIABLE_FILES[var_name]
    if not path.exists():
        raise FileNotFoundError(f"找不到 {var_name} 文件: {path}")
    ds = open_nc(path)
    try:
        candidates = [var_name]
        if var_name == "tp_hourly":
            candidates += ["tp", "total_precipitation"]
        real_name = next((v for v in candidates if v in ds.data_vars), None)
        if real_name is None:
            raise ValueError(f"{path} 中找不到 {candidates}，当前变量={list(ds.data_vars)}")

        tname = find_time_name(ds)
        lat_name, lon_name = find_lat_lon_names(ds)
        da = ds[real_name].rename({tname: "time", lat_name: "lat", lon_name: "lon"})
        da = _collapse_extra_dims(da, var_name).transpose("time", "lat", "lon")
        times = pd.to_datetime(da["time"].values)
        keep = np.where((times.year == int(year)) & np.isin(times.month, SELECT_MONTHS))[0]
        if keep.size == 0:
            raise ValueError(f"{path} 中没有 {year} 年 {SELECT_MONTHS} 月数据")
        da = da.isel(time=keep).sortby("time")
        time_values = pd.to_datetime(da["time"].values)
        _, unique_idx = np.unique(time_values.values, return_index=True)
        da = da.isel(time=np.sort(unique_idx)).astype("float32")
        if var_name == "tp_hourly":
            if TP_INPUT_UNITS == "m_per_hour":
                da = da * 1000.0
            elif TP_INPUT_UNITS != "mm_per_hour":
                raise ValueError(f"不支持的TP_INPUT_UNITS={TP_INPUT_UNITS}")
        da = da.load()
        if var_name == "tp_hourly":
            da.attrs["units"] = "mm/hour"
            print(
                f"{year} tp_hourly ready in mm/hour: "
                f"min={float(da.min(skipna=True)):.6g}, max={float(da.max(skipna=True)):.6g}"
            )
        da.name = var_name
        return da
    finally:
        ds.close()


def open_inputs_for_year(year):
    arrays = [load_variable_year(var, year) for var in INPUT_VARS]
    arrays = xr.align(*arrays, join="inner", copy=False)
    if arrays[0].sizes["time"] == 0:
        raise ValueError(f"{year} 年四个变量对齐后没有共同时间")
    shapes = {(a.sizes["lat"], a.sizes["lon"]) for a in arrays}
    if len(shapes) != 1:
        raise ValueError(f"{year} 年变量空间网格不一致: {shapes}")
    return xr.Dataset({var: da for var, da in zip(INPUT_VARS, arrays)})


def pad_3d(arr, pad_h, pad_w, value=np.nan):
    t, h, w = arr.shape
    if h > pad_h or w > pad_w:
        raise ValueError(f"原始空间尺寸 {(h, w)} 大于 pad_to {(pad_h, pad_w)}")

    out = np.full((t, pad_h, pad_w), value, dtype=np.float32)
    out[:, :h, :w] = arr.astype(np.float32)
    return out


def pad_2d(arr, pad_h, pad_w, value=0.0):
    h, w = arr.shape
    if h > pad_h or w > pad_w:
        raise ValueError(f"原始空间尺寸 {(h, w)} 大于 pad_to {(pad_h, pad_w)}")

    out = np.full((pad_h, pad_w), value, dtype=np.float32)
    out[:h, :w] = arr.astype(np.float32)
    return out


def make_geo_features(lat, lon, pad_h, pad_w):
    """构造并补齐六通道静态地理特征。"""
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    lat_norm = (lat_grid - np.nanmean(lat_grid)) / (np.nanstd(lat_grid) + 1e-6)
    lon_norm = (lon_grid - np.nanmean(lon_grid)) / (np.nanstd(lon_grid) + 1e-6)

    sin_lat = np.sin(np.deg2rad(lat_grid))
    cos_lat = np.cos(np.deg2rad(lat_grid))
    sin_lon = np.sin(np.deg2rad(lon_grid))
    cos_lon = np.cos(np.deg2rad(lon_grid))

    maps = [lat_norm, lon_norm, sin_lat, cos_lat, sin_lon, cos_lon]
    maps_pad = [
        pad_2d(item.astype(np.float32), pad_h, pad_w, value=0.0)
        for item in maps
    ]
    return np.stack(maps_pad, axis=0).astype(np.float32)


# =========================================================
# 3.1 水汽收支物理诊断工具函数
# =========================================================

_PHYS_CACHE = {}


def find_first_existing_var(ds, candidates, label):
    lower_to_real = {v.lower(): v for v in ds.data_vars}
    for cand in candidates:
        if cand in ds.data_vars:
            return cand
        if cand.lower() in lower_to_real:
            return lower_to_real[cand.lower()]
    raise ValueError(
        f"无法在 {label} 文件中找到变量。候选={candidates}, 当前变量={list(ds.data_vars)}"
    )


def standardize_phys_da(da, name):
    """统一物理变量维度名为 time, lat, lon，并转成 float32。"""
    tname = find_time_name(da.to_dataset(name="tmp"))
    lat_name, lon_name = find_lat_lon_names(da.to_dataset(name="tmp"))

    rename_dict = {}
    if tname != "time":
        rename_dict[tname] = "time"
    if lat_name != "lat":
        rename_dict[lat_name] = "lat"
    if lon_name != "lon":
        rename_dict[lon_name] = "lon"

    da = da.rename(rename_dict)
    da = da.transpose("time", "lat", "lon")
    da = da.astype("float32")
    da.name = name
    return da


def open_phys_var(path, candidates, label):
    ds = open_nc(path)
    var_name = find_first_existing_var(ds, candidates, label)
    da = standardize_phys_da(ds[var_name], label)
    print(
        f"Loaded physics {label}: var={var_name}, shape={da.shape}, "
        f"units={da.attrs.get('units', 'unknown')}, "
        f"min={float(da.min(skipna=True)):.6g}, max={float(da.max(skipna=True)):.6g}"
    )
    ds.close()
    return da


def convert_evap_to_positive_mm(e_da):
    """ERA5 evaporation 转为向上为正的 mm per interval。"""
    if EVAP_INPUT_MODE == "hourly_amount_m":
        e_mm = e_da * 1000.0
    elif EVAP_INPUT_MODE == "hourly_amount_mm":
        e_mm = e_da
    else:
        raise ValueError(f"当前脚本只为 hourly evaporation 设计，EVAP_INPUT_MODE={EVAP_INPUT_MODE}")

    mean_val = float(e_mm.mean(skipna=True))
    if EVAP_SIGN_MODE == "auto":
        e_pos = -e_mm if mean_val < 0 else e_mm
    elif EVAP_SIGN_MODE == "era5_negative":
        e_pos = -e_mm
    elif EVAP_SIGN_MODE == "positive_upward":
        e_pos = e_mm
    else:
        raise ValueError(f"Unsupported EVAP_SIGN_MODE={EVAP_SIGN_MODE}")

    return e_pos.rename("evaporation_upward_mm")


def compute_full_pbudget():
    """
    计算 full-period P_budget。
    结果保存在内存缓存中，避免每个年份重复打开和计算。
    """
    cache_key = "full_pbudget"
    if cache_key in _PHYS_CACHE:
        return _PHYS_CACHE[cache_key]

    print("\n=== 计算水汽收支 P_budget，ERA5 single-level 0.25° ===")

    e_raw = open_phys_var(EVAP_PATH, ["e", "evaporation"], "evaporation")
    tcwv = open_phys_var(TCWV_PATH, ["tcwv", "total_column_water_vapour", "total_column_water_vapor"], "tcwv")
    vimd = open_phys_var(
        VIMD_PATH,
        [
            "vimd",
            "vimdf",
            "vertical_integral_of_divergence_of_moisture_flux",
            "vertical_integral_of_moisture_divergence",
        ],
        "vimd",
    )

    e_mm = convert_evap_to_positive_mm(e_raw)

    e_mm, tcwv, vimd = xr.align(e_mm, tcwv, vimd, join="inner")

    # delta TCWV: TCWV(t) - TCWV(t-1), kg m-2 数值等价于 mm。
    # 4--9月跨年/跨季节间隔必须屏蔽，不能把长时间缺口当成1小时变化。
    d_tcwv = tcwv.diff("time").rename("delta_tcwv_mm")
    dt_seconds = np.diff(tcwv["time"].values).astype("timedelta64[s]").astype(np.float64)
    valid_dt = np.abs(dt_seconds - EXPECTED_DT_SECONDS) <= DT_TOLERANCE_SECONDS
    valid_dt_da = xr.DataArray(valid_dt, dims=["time"], coords={"time": tcwv["time"].values[1:]})
    d_tcwv = d_tcwv.where(valid_dt_da)
    e_current = e_mm.sel(time=d_tcwv["time"])
    vimd_current = vimd.sel(time=d_tcwv["time"])

    if VIMD_TIME_INTEGRATION_MODE == "none":
        vimd_term = vimd_current
    elif VIMD_TIME_INTEGRATION_MODE == "multiply_dt":
        dt_da = xr.DataArray(dt_seconds, dims=["time"], coords={"time": tcwv["time"].values[1:]})
        vimd_term = (vimd_current * dt_da).where(valid_dt_da)
    else:
        raise ValueError(f"Unsupported VIMD_TIME_INTEGRATION_MODE={VIMD_TIME_INTEGRATION_MODE}")

    vimd_term = vimd_term.rename("vimd_term_mm")

    if VIMD_CONVENTION == "divergence":
        # dW + divQ = E - P  => P = E - dW - divQ
        p_raw = e_current - d_tcwv - vimd_term
    elif VIMD_CONVENTION == "convergence":
        p_raw = e_current - d_tcwv + vimd_term
    else:
        raise ValueError(f"Unsupported VIMD_CONVENTION={VIMD_CONVENTION}")

    p_raw = p_raw.rename("p_budget_raw")
    p_pos = p_raw.where(p_raw > 0.0, 0.0).rename("p_budget_positive")
    p_mask = xr.where(np.isfinite(p_raw), 1.0, 0.0).rename("p_budget_mask")

    ds_pb = xr.Dataset(
        {
            "p_budget_raw": p_raw.astype("float32"),
            "p_budget_positive": p_pos.astype("float32"),
            "p_budget_mask": p_mask.astype("float32"),
            "evaporation_upward_mm": e_current.astype("float32"),
            "delta_tcwv_mm": d_tcwv.astype("float32"),
            "vimd_term_mm": vimd_term.astype("float32"),
        }
    )

    print("P_budget summary:")
    print(f"  p_budget_raw mean      = {float(ds_pb['p_budget_raw'].mean(skipna=True)):.6f}")
    print(f"  p_budget_positive mean = {float(ds_pb['p_budget_positive'].mean(skipna=True)):.6f}")
    print(f"  evaporation mean       = {float(ds_pb['evaporation_upward_mm'].mean(skipna=True)):.6f}")
    print(f"  delta_tcwv mean        = {float(ds_pb['delta_tcwv_mm'].mean(skipna=True)):.6f}")
    print(f"  vimd_term mean         = {float(ds_pb['vimd_term_mm'].mean(skipna=True)):.6f}")

    _PHYS_CACHE[cache_key] = ds_pb
    return ds_pb


def interp_pbudget_to_target_grid(target_times, target_lat, target_lon):
    """
    将P_budget与0.25°目标网格严格对齐，并按目标时间重索引；不做空间插值。

    返回：
        p_raw, p_pos, p_mask: shape = (T, H, W)
    """
    ds_pb = compute_full_pbudget()

    # 对齐目标时间。P_budget因dTCWV会缺少第一个小时，缺失位置保持NaN。
    target_time_index = pd.to_datetime(target_times)
    ds_sel = ds_pb.reindex(time=target_time_index)

    # 排序后只允许浮点误差量级的最近坐标；网格不一致时直接报错。
    ds_sel = ds_sel.sortby("lat").sortby("lon")
    try:
        ds_aligned = ds_sel.sel(
            lat=xr.DataArray(np.asarray(target_lat), dims="lat"),
            lon=xr.DataArray(np.asarray(target_lon), dims="lon"),
            method="nearest",
            tolerance=1e-5,
        )
    except KeyError as exc:
        raise ValueError(
            "物理变量网格与训练数据网格不一致；该脚本不执行空间插值。"
            "请确保区域、0.25°分辨率及经纬度坐标相同。"
        ) from exc

    p_raw = ds_aligned["p_budget_raw"].values.astype(np.float32)
    p_pos = ds_aligned["p_budget_positive"].values.astype(np.float32)
    p_mask = ds_aligned["p_budget_mask"].values.astype(np.float32)

    p_mask = np.where(np.isfinite(p_raw) & np.isfinite(p_pos) & (p_mask > 0), 1.0, 0.0).astype(np.float32)
    p_raw = np.where(np.isfinite(p_raw), p_raw, np.nan).astype(np.float32)
    p_pos = np.where(np.isfinite(p_pos), p_pos, np.nan).astype(np.float32)

    return p_raw, p_pos, p_mask


# =========================================================
# 4. 构建年度数据
# =========================================================

class YearData:
    def __init__(self, year):
        self.path = VARIABLE_DATA_DIR
        ds = open_inputs_for_year(year)

        self.time_name = find_time_name(ds)
        self.lat_name, self.lon_name = find_lat_lon_names(ds)

        times = pd.to_datetime(ds[self.time_name].values)
        lat = ds[self.lat_name].values
        lon = ds[self.lon_name].values

        self.times = times
        self.lat = lat
        self.lon = lon
        self.orig_h = len(lat)
        self.orig_w = len(lon)
        self.pad_h = int(np.ceil(self.orig_h / PAD_MULTIPLE) * PAD_MULTIPLE)
        self.pad_w = int(np.ceil(self.orig_w / PAD_MULTIPLE) * PAD_MULTIPLE)
        self.year = int(year)

        feature_list = []
        feature_names = []

        for var in INPUT_VARS:
            if var not in ds.data_vars:
                raise ValueError(f"{self.path} 中找不到变量 {var}，当前变量为 {list(ds.data_vars)}")

            arr = ds[var].transpose(self.time_name, self.lat_name, self.lon_name).values.astype(np.float32)

            # 降水输入使用 log1p(mm)
            if var in ["tp_hourly", "tp"]:
                arr = np.where(np.isfinite(arr), arr, np.nan)
                arr = np.maximum(arr, 0.0)
                arr = np.log1p(arr)

            arr_pad = pad_3d(arr, self.pad_h, self.pad_w, value=np.nan)
            feature_list.append(arr_pad)
            feature_names.append(var)

        # 有效区域掩膜。
        if "valid_mask" in ds.data_vars:
            valid_mask_raw = ds["valid_mask"].transpose(
                self.time_name,
                self.lat_name,
                self.lon_name,
            ).values.astype(np.float32)
        else:
            target_raw_tmp = ds[TARGET_VAR].transpose(
                self.time_name,
                self.lat_name,
                self.lon_name,
            ).values.astype(np.float32)
            valid_mask_raw = np.where(np.isfinite(target_raw_tmp), 1.0, 0.0).astype(np.float32)

        valid_mask_pad = pad_3d(valid_mask_raw, self.pad_h, self.pad_w, value=0.0)

        self.X_all = np.stack(feature_list, axis=1).astype(np.float32)  # (T, C, H, W)
        self.feature_names = feature_names

        # target
        target_raw = ds[TARGET_VAR].transpose(self.time_name, self.lat_name, self.lon_name).values.astype(np.float32)
        target_mm = np.where(np.isfinite(target_raw), target_raw, 0.0).astype(np.float32)
        target_mm = np.maximum(target_mm, 0.0)

        self.Y_mm_all = pad_3d(target_mm, self.pad_h, self.pad_w, value=0.0)[:, None, :, :].astype(np.float32)
        self.Y_log_all = np.log1p(self.Y_mm_all).astype(np.float32)
        self.M_all = valid_mask_pad[:, None, :, :].astype(np.float32)

        # 水汽收支量与目标0.25°网格严格对齐。
        p_raw_raw, p_pos_raw, p_mask_raw = interp_pbudget_to_target_grid(
            target_times=times,
            target_lat=lat,
            target_lon=lon,
        )
        self.PB_raw_all = pad_3d(
            p_raw_raw, self.pad_h, self.pad_w, value=np.nan
        )[:, None, :, :].astype(np.float32)
        self.PB_pos_all = pad_3d(
            p_pos_raw, self.pad_h, self.pad_w, value=np.nan
        )[:, None, :, :].astype(np.float32)
        self.PB_mask_all = pad_3d(
            p_mask_raw, self.pad_h, self.pad_w, value=0.0
        )[:, None, :, :].astype(np.float32)
        self.PB_mask_all = (self.PB_mask_all * self.M_all).astype(np.float32)

        self.geo = make_geo_features(lat, lon, self.pad_h, self.pad_w)

        ds.close()

        self.sample_starts = self._build_sample_starts()

    def _build_sample_starts(self):
        t = self.X_all.shape[0]
        max_start = t - history_len - horizon + 1
        starts = []
        for s in range(max_start):
            out_start = s + history_len
            out_end = out_start + horizon
            window_times = self.times[s:out_end]
            continuous = np.all(np.diff(window_times.values).astype("timedelta64[s]").astype(np.int64) == 3600)
            if continuous and self.M_all[out_start:out_end].sum() > 0:
                starts.append(s)
        return starts

    def summary(self):
        return {
            "path": str(self.path),
            "year": self.year,
            "time_range": [str(self.times[0]), str(self.times[-1])],
            "orig_shape": [self.orig_h, self.orig_w],
            "padded_shape": [self.pad_h, self.pad_w],
            "n_samples": len(self.sample_starts),
            "feature_names": self.feature_names,
            "use_geo_embedding": USE_GEO_EMBEDDING,
            "geo_shape": list(self.geo.shape),
            "mask_valid_ratio": float(self.M_all.sum() / self.M_all.size),
            "pbudget_valid_ratio": float(
                self.PB_mask_all.sum() / max(self.M_all.sum(), 1.0)
            ),
            "pbudget_raw_mean": float(
                np.nanmean(
                    np.where(self.PB_mask_all > 0, self.PB_raw_all, np.nan)
                )
            ),
            "pbudget_positive_mean": float(
                np.nanmean(
                    np.where(self.PB_mask_all > 0, self.PB_pos_all, np.nan)
                )
            ),
        }


def load_years(years):
    data = []
    for year in years:
        yd = YearData(year)
        data.append(yd)
        print(json.dumps(yd.summary(), ensure_ascii=False, indent=2))
    return data


# =========================================================
# 5. 标准化参数
# =========================================================

def compute_scalers(train_year_data):
    x_sum = None
    x_sumsq = None
    x_count = None

    y_sum = 0.0
    y_sumsq = 0.0
    y_count = 0.0

    for yd in train_year_data:
        x = yd.X_all
        finite = np.isfinite(x)

        x_zero = np.where(finite, x, 0.0)
        sum_i = x_zero.sum(axis=(0, 2, 3))
        sumsq_i = (x_zero ** 2).sum(axis=(0, 2, 3))
        count_i = finite.sum(axis=(0, 2, 3)).astype(np.float64)

        if x_sum is None:
            x_sum = sum_i.astype(np.float64)
            x_sumsq = sumsq_i.astype(np.float64)
            x_count = count_i
        else:
            x_sum += sum_i
            x_sumsq += sumsq_i
            x_count += count_i

        y = yd.Y_log_all
        m = yd.M_all > 0
        y_valid = y[m]
        y_sum += float(y_valid.sum())
        y_sumsq += float((y_valid ** 2).sum())
        y_count += float(y_valid.size)

    x_mean = x_sum / np.maximum(x_count, 1.0)
    x_var = x_sumsq / np.maximum(x_count, 1.0) - x_mean ** 2
    x_var = np.maximum(x_var, 0.0)
    x_std = np.sqrt(x_var)
    x_std = np.where(x_std < 1e-6, 1.0, x_std)

    y_mean = y_sum / max(y_count, 1.0)
    y_var = y_sumsq / max(y_count, 1.0) - y_mean ** 2
    y_var = max(y_var, 0.0)
    y_std = np.sqrt(y_var)
    if y_std < 1e-6:
        y_std = 1.0

    return {
        "x_mean": x_mean.astype(np.float32),
        "x_std": x_std.astype(np.float32),
        "y_mean": np.array([y_mean], dtype=np.float32),
        "y_std": np.array([y_std], dtype=np.float32)
    }


def compute_pbudget_alpha(train_year_data):
    """
    用训练集均值校正 p_budget_positive 的水量尺度。
    alpha = mean(tp_hourly_train) / mean(p_budget_positive_train)
    """
    if not CALIBRATE_PBUDGET_POSITIVE:
        return 1.0

    y_sum = 0.0
    pb_sum = 0.0
    count_y = 0.0
    count_pb = 0.0

    for yd in train_year_data:
        mask = (yd.M_all > 0) & (yd.PB_mask_all > 0) & np.isfinite(yd.PB_pos_all)
        if mask.sum() == 0:
            continue

        y_vals = yd.Y_mm_all[mask]
        pb_vals = yd.PB_pos_all[mask]

        y_sum += float(np.sum(y_vals))
        pb_sum += float(np.sum(pb_vals))
        count_y += float(y_vals.size)
        count_pb += float(pb_vals.size)

    mean_y = y_sum / max(count_y, 1.0)
    mean_pb = pb_sum / max(count_pb, 1.0)

    if mean_pb <= 1e-8:
        alpha = 1.0
    else:
        alpha = mean_y / mean_pb

    alpha = float(np.clip(alpha, PBUDGET_ALPHA_MIN, PBUDGET_ALPHA_MAX))
    print("\nP_budget positive calibration:")
    print(f"  mean train tp_hourly       = {mean_y:.6f}")
    print(f"  mean train p_budget_pos    = {mean_pb:.6f}")
    print(f"  p_budget_alpha             = {alpha:.6f}")
    return alpha


def _pad_spatial_to_multiple(x, multiple, value=0.0):
    """仅在右侧和下侧补齐，使H/W可被patch size整除。"""
    pad_h = (int(multiple) - x.shape[-2] % int(multiple)) % int(multiple)
    pad_w = (int(multiple) - x.shape[-1] % int(multiple)) % int(multiple)
    if pad_h == 0 and pad_w == 0:
        return x
    return F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=float(value))


def masked_patch_mean_torch(values, mask, patch_size=4, min_valid_fraction=0.75, eps=1e-6):
    """
    对(..., 1, H, W)做非重叠masked average pooling。

    返回：
        patch_mean: (..., 1, Hp, Wp)
        patch_valid: 同shape，满足有效格点比例阈值的patch为1
    """
    if values.shape != mask.shape:
        raise ValueError(f"values/mask shape mismatch: {values.shape} vs {mask.shape}")
    if values.ndim < 4 or values.shape[-3] != 1:
        raise ValueError(f"expected (...,1,H,W), got {values.shape}")

    original_leading = values.shape[:-3]
    h, w = values.shape[-2:]
    v = values.reshape(-1, 1, h, w)
    m = mask.reshape(-1, 1, h, w).to(values.dtype)

    v = _pad_spatial_to_multiple(v, patch_size, value=0.0)
    m = _pad_spatial_to_multiple(m, patch_size, value=0.0)

    area = float(int(patch_size) ** 2)
    patch_sum = F.avg_pool2d(v * m, kernel_size=patch_size, stride=patch_size) * area
    patch_count = F.avg_pool2d(m, kernel_size=patch_size, stride=patch_size) * area
    patch_mean = patch_sum / patch_count.clamp_min(eps)
    patch_valid = (patch_count >= area * float(min_valid_fraction)).to(values.dtype)

    hp, wp = patch_mean.shape[-2:]
    out_shape = tuple(original_leading) + (1, hp, wp)
    return patch_mean.reshape(out_shape), patch_valid.reshape(out_shape)


def fit_pbudget_lower_bound(train_year_data, pbudget_alpha):
    """使用训练年份的4×4区域残差下分位数拟合单侧物理下界偏移。"""
    rng = np.random.default_rng(PBUDGET_RESIDUAL_RANDOM_SEED)
    sampled_residuals = []

    for year_data in train_year_data:
        n_time = year_data.Y_mm_all.shape[0]
        for start in range(0, n_time, PBUDGET_RESIDUAL_CHUNK_TIMES):
            end = min(start + PBUDGET_RESIDUAL_CHUNK_TIMES, n_time)

            true_mm = torch.from_numpy(
                year_data.Y_mm_all[start:end]
            ).float()
            pbudget = torch.from_numpy(
                np.nan_to_num(
                    year_data.PB_pos_all[start:end],
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
            ).float()
            valid = torch.from_numpy(
                (
                    (year_data.M_all[start:end] > 0)
                    & (year_data.PB_mask_all[start:end] > 0)
                ).astype(np.float32)
            )

            pbudget = torch.clamp(
                pbudget * float(pbudget_alpha),
                min=0.0,
            )
            true_patch, true_patch_valid = masked_patch_mean_torch(
                true_mm,
                valid,
                PHYS_PATCH_SIZE,
                PHYS_PATCH_MIN_VALID_FRACTION,
            )
            pbudget_patch, pbudget_patch_valid = masked_patch_mean_torch(
                pbudget,
                valid,
                PHYS_PATCH_SIZE,
                PHYS_PATCH_MIN_VALID_FRACTION,
            )
            patch_valid = (
                (true_patch_valid > 0)
                & (pbudget_patch_valid > 0)
            )
            residual = (
                true_patch - pbudget_patch
            )[patch_valid].cpu().numpy().astype(np.float32)
            residual = residual[np.isfinite(residual)]
            if residual.size == 0:
                continue

            take = min(
                int(PBUDGET_RESIDUAL_SAMPLES_PER_CHUNK),
                residual.size,
            )
            if residual.size > take:
                indices = rng.choice(
                    residual.size,
                    size=take,
                    replace=False,
                )
                residual = residual[indices]
            sampled_residuals.append(residual)

    if not sampled_residuals:
        raise RuntimeError(
            "无法从训练集获得有效的4×4 P_budget残差样本。"
        )

    residuals = np.concatenate(sampled_residuals).astype(np.float32)
    if residuals.size > PBUDGET_RESIDUAL_MAX_SAMPLES:
        indices = rng.choice(
            residuals.size,
            size=PBUDGET_RESIDUAL_MAX_SAMPLES,
            replace=False,
        )
        residuals = residuals[indices]

    lower_offset = float(
        np.quantile(residuals, PBUDGET_RESIDUAL_LOWER_QUANTILE)
    )
    lower_offset = min(
        lower_offset,
        -float(PBUDGET_MIN_LOWER_OFFSET_MM),
    )

    result = {
        "mode": "patch_residual_lower_quantile",
        "patch_size": int(PHYS_PATCH_SIZE),
        "min_valid_fraction": float(PHYS_PATCH_MIN_VALID_FRACTION),
        "lower_quantile": float(PBUDGET_RESIDUAL_LOWER_QUANTILE),
        "lower_offset_mm": lower_offset,
        "residual_mean_mm": float(np.mean(residuals)),
        "residual_std_mm": float(np.std(residuals)),
        "residual_median_mm": float(np.median(residuals)),
        "n_samples": int(residuals.size),
        "pbudget_alpha": float(pbudget_alpha),
        "fit_years": [int(year) for year in TRAIN_YEARS],
    }

    print()
    print("训练年份4×4 P_budget残差下分位数：")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def physics_weight_ramp(epoch):
    if epoch < PHYS_START_EPOCH:
        return 0.0
    progress = (epoch - PHYS_START_EPOCH + 1) / max(int(PHYS_RAMP_EPOCHS), 1)
    return float(min(max(progress, 0.0), 1.0))


# =========================================================
# 6. Dataset
# =========================================================

class Era5NowcastDataset(Dataset):
    def __init__(self, year_data_list, scalers):
        self.year_data_list = year_data_list
        self.scalers = scalers

        self.index = []
        for yi, yd in enumerate(year_data_list):
            for s in yd.sample_starts:
                self.index.append((yi, s))

        self.x_mean = scalers["x_mean"][None, :, None, None]
        self.x_std = scalers["x_std"][None, :, None, None]
        self.y_mean = float(scalers["y_mean"][0])
        self.y_std = float(scalers["y_std"][0])

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        yi, s = self.index[idx]
        yd = self.year_data_list[yi]

        in_start = s
        in_end = s + history_len
        out_start = in_end
        out_end = out_start + horizon

        x = yd.X_all[in_start:in_end].copy()
        y = yd.Y_log_all[out_start:out_end].copy()
        y_mm = yd.Y_mm_all[out_start:out_end].copy()
        m = yd.M_all[out_start:out_end].copy()
        pb_raw = yd.PB_raw_all[out_start:out_end].copy()
        pb_pos = yd.PB_pos_all[out_start:out_end].copy()
        pb_mask = yd.PB_mask_all[out_start:out_end].copy()

        x = (x - self.x_mean) / self.x_std
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        y = (y - self.y_mean) / self.y_std
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        target_time_str = "|".join([str(t) for t in yd.times[out_start:out_end]])

        pb_raw = np.nan_to_num(pb_raw, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        pb_pos = np.nan_to_num(pb_pos, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        pb_mask = np.nan_to_num(pb_mask, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        return (
            torch.from_numpy(x),
            torch.from_numpy(y),
            torch.from_numpy(m.astype(np.float32)),
            torch.from_numpy(y_mm.astype(np.float32)),
            torch.from_numpy(pb_raw),
            torch.from_numpy(pb_pos),
            torch.from_numpy(pb_mask),
            str(yd.year),
            target_time_str
        )


# =========================================================
# 7. 模型结构
# =========================================================

class GeoEmbedding(nn.Module):
    """
    静态地理信息嵌入模块。
    输入:  (B, C_geo, H, W)
    输出:  (B, C_embed, H, W)
    """
    def __init__(self, geo_in_channels=6, geo_embed_channels=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(geo_in_channels, 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, geo_embed_channels, kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, geo):
        return self.net(geo)


class CNNEncoder(nn.Module):
    """包含一次2倍下采样的单尺度CNN编码器。"""
    def __init__(self, in_channels, latent_channels=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, latent_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.encoder(x)


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_channels = hidden_channels

        self.conv = nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding
        )

    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)

        i, f, o, g = torch.chunk(gates, chunks=4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, height, width, device):
        h = torch.zeros(batch_size, self.hidden_channels, height, width, device=device)
        c = torch.zeros(batch_size, self.hidden_channels, height, width, device=device)
        return h, c


class CNNDecoder(nn.Module):
    """
    解码回原始 H, W。
    """
    def __init__(self, latent_channels=64, out_channels=1, dropout=0.0):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(latent_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=1)
        )

    def forward(self, z, out_size):
        z = self.conv1(z)
        z = F.interpolate(
            z,
            size=out_size,
            mode="bilinear",
            align_corners=False
        )
        return self.conv2(z)


class GeoConvLSTMForecastModel(nn.Module):
    """
    Geo-ConvLSTM降水预报模型。

    模型依次使用可选地理信息嵌入、CNN编码器、ConvLSTM上下文建模、
    隐藏状态递推和CNN解码器。

    输入:
        x:   (B, history_len, C, H, W)
        geo: (B, C_geo, H, W) 或 None

    输出:
        y_pred: (B, horizon, 1, H, W)
    """
    def __init__(
        self,
        input_channels,
        use_geo_embedding=False,
        geo_in_channels=6,
        geo_embed_channels=8,
        latent_channels=64,
        horizon=8,
        dropout=0.0,
        use_rain_event_head=True
    ):
        super().__init__()

        self.horizon = horizon
        self.use_geo_embedding = use_geo_embedding
        self.use_rain_event_head = use_rain_event_head

        if use_geo_embedding:
            self.geo_embedding = GeoEmbedding(
                geo_in_channels=geo_in_channels,
                geo_embed_channels=geo_embed_channels
            )
            encoder_in_channels = input_channels + geo_embed_channels
        else:
            self.geo_embedding = None
            encoder_in_channels = input_channels

        self.encoder = CNNEncoder(
            in_channels=encoder_in_channels,
            latent_channels=latent_channels
        )

        self.convlstm = ConvLSTMCell(
            input_channels=latent_channels,
            hidden_channels=latent_channels,
            kernel_size=3
        )

        self.decoder = CNNDecoder(
            latent_channels=latent_channels,
            out_channels=1,
            dropout=dropout
        )
        self.rain_event_decoder = CNNDecoder(
            latent_channels=latent_channels,
            out_channels=1,
            dropout=dropout
        ) if use_rain_event_head else None

    def forward(self, x, geo=None):
        batch_size, sequence_length, _, height, width = x.shape

        if self.use_geo_embedding:
            if geo is None:
                raise ValueError("USE_GEO_EMBEDDING=True 时必须提供 geo。")
            geo_emb = self.geo_embedding(geo)
        else:
            geo_emb = None

        z_seq = []
        for time_index in range(sequence_length):
            x_t = x[:, time_index]
            if self.use_geo_embedding:
                x_t = torch.cat([x_t, geo_emb], dim=1)
            z_t = self.encoder(x_t)
            z_seq.append(z_t)

        _, _, latent_height, latent_width = z_seq[-1].shape

        h_state, c_state = self.convlstm.init_hidden(
            batch_size=batch_size,
            height=latent_height,
            width=latent_width,
            device=x.device
        )

        # 使用历史序列建立上下文状态。
        for z_t in z_seq:
            h_state, c_state = self.convlstm(z_t, h_state, c_state)

        # 第一个未来时刻输入零潜在张量，后续时刻输入上一隐藏状态。
        preds = []
        rain_logits = []
        z_in = torch.zeros_like(h_state)

        for _ in range(self.horizon):
            h_state, c_state = self.convlstm(z_in, h_state, c_state)
            y_hat = self.decoder(h_state, out_size=(height, width))
            preds.append(y_hat)
            if self.rain_event_decoder is not None:
                rain_logits.append(
                    self.rain_event_decoder(h_state, out_size=(height, width))
                )

            z_in = h_state

        pred_reg = torch.stack(preds, dim=1)
        if self.rain_event_decoder is not None:
            return {"reg": pred_reg, "rain_logits": torch.stack(rain_logits, dim=1)}
        return {"reg": pred_reg, "rain_logits": None}


# =========================================================
# 8. Loss 与指标
# =========================================================

def torch_inverse_y_transform(y_norm, y_mean, y_std):
    """
    y_norm -> mm, 保持可微。
    模型输出是 normalized log1p(tp_hourly)。
    """
    y_log = y_norm * y_std + y_mean
    y_mm = torch.expm1(y_log)
    return torch.clamp(y_mm, min=0.0)


def make_rain_intensity_weights(true_mm, mask):
    weights = torch.full_like(true_mm, float(RAIN_WEIGHT_VALUES[0]))
    for threshold, value in zip(RAIN_WEIGHT_THRESHOLDS, RAIN_WEIGHT_VALUES[1:]):
        weights = torch.where(true_mm >= float(threshold), torch.full_like(weights, float(value)), weights)
    if NORMALIZE_RAIN_WEIGHTS:
        valid_mean = (weights * mask).sum() / (mask.sum() + 1e-6)
        weights = weights / torch.clamp(valid_mean, min=1e-6)
    return weights


def masked_weighted_huber_loss(pred_norm, target_norm, target_mm, mask, eps=1e-6):
    weights = make_rain_intensity_weights(target_mm, mask)
    point_loss = F.smooth_l1_loss(pred_norm, target_norm, beta=DATA_HUBER_BETA, reduction="none")
    return (point_loss * weights * mask).sum() / ((weights * mask).sum() + eps)


def soft_event_probability(x_mm, threshold):
    return torch.sigmoid(float(SOFT_THRESHOLD_SHARPNESS) * (x_mm - float(threshold)))


def balanced_soft_csi_loss(pred_mm, true_mm, mask, eps=1e-6):
    """按 horizon/threshold 分开计算；无正样本组合跳过，虚警由独立事件头约束。"""
    losses, weights = [], []
    for h, horizon_weight in enumerate(HORIZON_LOSS_WEIGHTS):
        h_mask = mask[:, h:h + 1]
        h_pred = pred_mm[:, h:h + 1]
        h_true = true_mm[:, h:h + 1]
        for threshold, event_weight in zip(SOFT_EVENT_THRESHOLDS, SOFT_EVENT_WEIGHTS):
            target_event = (h_true >= float(threshold)).to(h_pred.dtype)
            if float((target_event * h_mask).sum().detach().cpu()) <= 0.0:
                continue
            pred_event = soft_event_probability(h_pred, threshold)
            hit = (pred_event * target_event * h_mask).sum()
            miss = ((1.0 - pred_event) * target_event * h_mask).sum()
            false_alarm = (pred_event * (1.0 - target_event) * h_mask).sum()
            losses.append(1.0 - hit / (hit + miss + false_alarm + eps))
            weights.append(float(horizon_weight) * float(event_weight))
    if not losses:
        return torch.zeros((), dtype=pred_mm.dtype, device=pred_mm.device)
    weight_tensor = torch.as_tensor(weights, dtype=pred_mm.dtype, device=pred_mm.device)
    return (torch.stack(losses) * weight_tensor).sum() / weight_tensor.sum().clamp_min(eps)


def rain_event_head_loss(rain_logits, true_mm, mask, eps=1e-6):
    if rain_logits is None or not USE_RAIN_EVENT_HEAD:
        return torch.zeros((), dtype=true_mm.dtype, device=true_mm.device)
    target = (true_mm >= float(RAIN_EVENT_THRESHOLD)).to(rain_logits.dtype)
    pos_weight = torch.as_tensor(float(RAIN_EVENT_POS_WEIGHT), dtype=rain_logits.dtype, device=rain_logits.device)
    point_loss = F.binary_cross_entropy_with_logits(
        rain_logits, target, reduction="none", pos_weight=pos_weight
    )
    return (point_loss * mask).sum() / (mask.sum() + eps)


def _same_padding_max_pool_2d(x, kernel_size):
    """对[B,T,C,H,W]做同尺寸空间最大池化。"""
    if int(kernel_size) <= 1:
        return x
    if int(kernel_size) % 2 == 0:
        raise ValueError("邻域kernel必须为奇数。")
    b, t, c, h, w = x.shape
    y = x.reshape(b * t * c, 1, h, w)
    y = F.max_pool2d(
        y,
        kernel_size=int(kernel_size),
        stride=1,
        padding=int(kernel_size) // 2,
    )
    return y.reshape(b, t, c, h, w)


def _weighted_mean_losses(losses, weights, reference, eps=1e-6):
    if not losses:
        return torch.zeros((), dtype=reference.dtype, device=reference.device)
    weight_tensor = torch.as_tensor(
        weights,
        dtype=reference.dtype,
        device=reference.device,
    )
    return (
        torch.stack(losses) * weight_tensor
    ).sum() / weight_tensor.sum().clamp_min(eps)


def pbudget_patch_lower_bound_loss(
    pred_mm,
    mask,
    pb_pos,
    pb_mask,
    pbudget_alpha,
    lower_bound_stats,
    eps=1e-6,
):
    """
    结构化P_budget约束，专门作用于H05-H08。

    1) 邻域匹配：
       P_budget在4/8 mm h-1以上时，只要求模型在其3×3邻域内
       出现相应强度，允许少量空间偏移。

    2) 多尺度区域单侧下界：
       在4×4与8×8区域平均尺度上，惩罚预测低于物理支持量，
       不设置上界。

    3) H05-H08累计单侧下界：
       对后4小时累计降水施加4×4与8×8尺度约束，
       允许降水在各小时之间重新分配。
    """
    valid = mask * pb_mask
    pb_cal = torch.clamp(pb_pos * float(pbudget_alpha), min=0.0)

    horizon_weights = pred_mm.new_tensor(
        PHYS_HORIZON_WEIGHTS
    ).view(1, -1, 1, 1, 1)
    late_valid = valid * horizon_weights

    # -----------------------------------------------------
    # A. 高置信强降水邻域匹配
    # -----------------------------------------------------
    pred_local_max = _same_padding_max_pool_2d(
        torch.clamp(pred_mm, min=0.0),
        PHYS_NEIGHBORHOOD_KERNEL,
    )

    neighborhood_losses = []
    neighborhood_weights = []
    neighborhood_active = pred_mm.new_zeros(())

    for threshold, threshold_weight in zip(
        PHYS_NEIGHBORHOOD_THRESHOLDS_MM,
        PHYS_NEIGHBORHOOD_THRESHOLD_WEIGHTS,
    ):
        physical_event = (
            pb_cal >= float(threshold)
        ).to(pred_mm.dtype) * late_valid

        event_count = physical_event.sum()
        if float(event_count.detach().cpu()) <= 0.0:
            continue

        local_probability = torch.sigmoid(
            (
                pred_local_max - float(threshold)
            ) / float(PHYS_NEIGHBORHOOD_SOFTNESS_MM)
        )
        event_loss = -torch.log(
            local_probability.clamp_min(eps)
        )
        event_loss = (
            event_loss * physical_event
        ).sum() / (event_count + eps)

        neighborhood_losses.append(event_loss)
        neighborhood_weights.append(float(threshold_weight))
        neighborhood_active = neighborhood_active + event_count

    neighborhood_loss = _weighted_mean_losses(
        neighborhood_losses,
        neighborhood_weights,
        pred_mm,
        eps=eps,
    )

    # -----------------------------------------------------
    # B. 4×4 + 8×8多尺度区域下界
    # -----------------------------------------------------
    lower_offset = pred_mm.new_tensor(
        float(lower_bound_stats["lower_offset_mm"])
    )
    multiscale_losses = []
    multiscale_weights = []
    multiscale_active_sum = pred_mm.new_zeros(())
    multiscale_eligible_sum = pred_mm.new_zeros(())
    lower_sum = pred_mm.new_zeros(())
    pb_sum = pred_mm.new_zeros(())
    violation_sum = pred_mm.new_zeros(())

    for patch_size, scale_weight in zip(
        PHYS_MULTISCALE_PATCH_SIZES,
        PHYS_MULTISCALE_WEIGHTS,
    ):
        pred_patch, pred_patch_valid = masked_patch_mean_torch(
            pred_mm,
            valid,
            int(patch_size),
            PHYS_PATCH_MIN_VALID_FRACTION,
        )
        pb_patch, pb_patch_valid = masked_patch_mean_torch(
            pb_cal,
            valid,
            int(patch_size),
            PHYS_PATCH_MIN_VALID_FRACTION,
        )
        patch_valid = (
            pred_patch_valid * pb_patch_valid * horizon_weights
        )
        strong_signal = (
            pb_patch >= float(
                PHYS_MULTISCALE_PBUDGET_THRESHOLD_MM
            )
        ).to(pred_mm.dtype)
        patch_weight = patch_valid * strong_signal

        lower = torch.clamp(
            pb_patch + lower_offset,
            min=0.0,
        )
        violation = F.relu(
            torch.log1p(lower)
            - torch.log1p(torch.clamp(pred_patch, min=0.0))
        )
        point_loss = F.smooth_l1_loss(
            violation,
            torch.zeros_like(violation),
            beta=PHYS_HUBER_BETA,
            reduction="none",
        )
        scale_loss = (
            point_loss * patch_weight
        ).sum() / (patch_weight.sum() + eps)

        multiscale_losses.append(scale_loss)
        multiscale_weights.append(float(scale_weight))
        multiscale_active_sum += patch_weight.sum()
        multiscale_eligible_sum += patch_valid.sum()
        lower_sum += (lower * patch_weight).sum()
        pb_sum += (pb_patch * patch_weight).sum()
        violation_sum += (violation * patch_weight).sum()

    multiscale_loss = _weighted_mean_losses(
        multiscale_losses,
        multiscale_weights,
        pred_mm,
        eps=eps,
    )

    # -----------------------------------------------------
    # C. H05-H08累计水量约束
    # -----------------------------------------------------
    late_start = LATE_HORIZON_START
    late_end = LATE_HORIZON_END
    late_valid_raw = valid[:, late_start:late_end]
    pred_late = torch.clamp(
        pred_mm[:, late_start:late_end],
        min=0.0,
    )
    pb_late = pb_cal[:, late_start:late_end]

    time_valid_fraction = late_valid_raw.mean(
        dim=1,
        keepdim=True,
    )
    cumulative_valid = (
        time_valid_fraction
        >= float(PHYS_CUMULATIVE_MIN_VALID_FRACTION)
    ).to(pred_mm.dtype)

    pred_cumulative = (
        pred_late * late_valid_raw
    ).sum(dim=1, keepdim=True)
    pb_cumulative = (
        pb_late * late_valid_raw
    ).sum(dim=1, keepdim=True)

    cumulative_losses = []
    cumulative_weights = []
    cumulative_active_sum = pred_mm.new_zeros(())

    number_late_steps = late_end - late_start
    cumulative_offset = (
        float(number_late_steps) * lower_offset
    )

    for patch_size, scale_weight in zip(
        PHYS_CUMULATIVE_PATCH_SIZES,
        PHYS_CUMULATIVE_SCALE_WEIGHTS,
    ):
        pred_cum_patch, pred_cum_valid = masked_patch_mean_torch(
            pred_cumulative,
            cumulative_valid,
            int(patch_size),
            PHYS_PATCH_MIN_VALID_FRACTION,
        )
        pb_cum_patch, pb_cum_valid = masked_patch_mean_torch(
            pb_cumulative,
            cumulative_valid,
            int(patch_size),
            PHYS_PATCH_MIN_VALID_FRACTION,
        )
        cum_patch_valid = pred_cum_valid * pb_cum_valid
        cum_signal = (
            pb_cum_patch >= float(
                PHYS_CUMULATIVE_PBUDGET_THRESHOLD_MM
            )
        ).to(pred_mm.dtype)
        cum_weight = cum_patch_valid * cum_signal

        cumulative_lower = torch.clamp(
            pb_cum_patch + cumulative_offset,
            min=0.0,
        )
        cumulative_violation = F.relu(
            torch.log1p(cumulative_lower)
            - torch.log1p(
                torch.clamp(pred_cum_patch, min=0.0)
            )
        )
        cumulative_point_loss = F.smooth_l1_loss(
            cumulative_violation,
            torch.zeros_like(cumulative_violation),
            beta=PHYS_HUBER_BETA,
            reduction="none",
        )
        cumulative_scale_loss = (
            cumulative_point_loss * cum_weight
        ).sum() / (cum_weight.sum() + eps)

        cumulative_losses.append(cumulative_scale_loss)
        cumulative_weights.append(float(scale_weight))
        cumulative_active_sum += cum_weight.sum()

    cumulative_loss = _weighted_mean_losses(
        cumulative_losses,
        cumulative_weights,
        pred_mm,
        eps=eps,
    )

    # -----------------------------------------------------
    # 组合
    # -----------------------------------------------------
    structured_loss = (
        float(PHYS_NEIGHBORHOOD_COMPONENT_WEIGHT)
        * neighborhood_loss
        + float(PHYS_MULTISCALE_COMPONENT_WEIGHT)
        * multiscale_loss
        + float(PHYS_CUMULATIVE_COMPONENT_WEIGHT)
        * cumulative_loss
    )

    total_late_points = late_valid.sum()
    active_proxy = (
        neighborhood_active
        + multiscale_active_sum
        + cumulative_active_sum
    )
    active_fraction = active_proxy / (
        total_late_points + multiscale_eligible_sum + eps
    )

    mean_violation = violation_sum / (
        multiscale_active_sum + eps
    )
    mean_lower = lower_sum / (
        multiscale_active_sum + eps
    )
    mean_pbudget = pb_sum / (
        multiscale_active_sum + eps
    )

    return {
        "loss": structured_loss,
        "neighborhood_loss": neighborhood_loss,
        "multiscale_loss": multiscale_loss,
        "cumulative_loss": cumulative_loss,
        "active_fraction": active_fraction,
        "mean_violation": mean_violation,
        "mean_lower_mm": mean_lower,
        "mean_pbudget_mm": mean_pbudget,
    }


def compute_total_loss(
    model_output,
    target_norm,
    target_mm,
    mask,
    pb_pos,
    pb_mask,
    scalers,
    epoch,
    pbudget_alpha,
    lower_bound_stats,
):
    """
    计算物理约束微调损失。

    L = L_baseline + ramp(epoch) * lambda_phys * (
          0.25 * L_neighborhood
        + 0.35 * L_multiscale
        + 0.40 * L_cumulative
    )

    基线损失与配套8小时基线脚本一致；物理约束仅作用于H05--H08。
    """
    pred_norm = model_output["reg"]
    rain_logits = model_output.get("rain_logits")
    data_loss = masked_weighted_huber_loss(
        pred_norm,
        target_norm,
        target_mm,
        mask,
    )

    y_mean = torch.as_tensor(
        float(scalers["y_mean"][0]),
        device=pred_norm.device,
        dtype=pred_norm.dtype,
    )
    y_std = torch.as_tensor(
        float(scalers["y_std"][0]),
        device=pred_norm.device,
        dtype=pred_norm.dtype,
    )
    pred_mm = torch_inverse_y_transform(pred_norm, y_mean, y_std)
    soft_csi = balanced_soft_csi_loss(pred_mm, target_mm, mask)
    rain_event = rain_event_head_loss(rain_logits, target_mm, mask)

    baseline_loss = (
        data_loss
        + SOFT_CSI_LOSS_WEIGHT * soft_csi
        + RAIN_EVENT_LOSS_WEIGHT * rain_event
    )

    zero = torch.zeros_like(data_loss)
    ramp = physics_weight_ramp(epoch)
    if ramp <= 0.0:
        return baseline_loss, {
            "total": baseline_loss.detach(),
            "data": data_loss.detach(),
            "soft_csi": soft_csi.detach(),
            "rain_event": rain_event.detach(),
            "phys_lower_bound": zero.detach(),
            "phys_ramp": zero.detach(),
            "phys_active_fraction": zero.detach(),
            "phys_mean_violation": zero.detach(),
            "phys_mean_lower_mm": zero.detach(),
            "phys_mean_pbudget_mm": zero.detach(),
            "phys_contribution": zero.detach(),
        }

    phys = pbudget_patch_lower_bound_loss(
        pred_mm=pred_mm,
        mask=mask,
        pb_pos=pb_pos,
        pb_mask=pb_mask,
        pbudget_alpha=pbudget_alpha,
        lower_bound_stats=lower_bound_stats,
    )
    phys_contribution = (
        float(ramp)
        * PHYS_LOWER_BOUND_WEIGHT
        * phys["loss"]
    )
    total_loss = baseline_loss + phys_contribution

    return total_loss, {
        "total": total_loss.detach(),
        "data": data_loss.detach(),
        "soft_csi": soft_csi.detach(),
        "rain_event": rain_event.detach(),
        "phys_lower_bound": phys["loss"].detach(),
        "phys_ramp": pred_mm.new_tensor(float(ramp)).detach(),
        "phys_active_fraction": phys["active_fraction"].detach(),
        "phys_mean_violation": phys["mean_violation"].detach(),
        "phys_mean_lower_mm": phys["mean_lower_mm"].detach(),
        "phys_mean_pbudget_mm": phys["mean_pbudget_mm"].detach(),
        "phys_contribution": phys_contribution.detach(),
    }


def inverse_y_transform(y_norm, scalers):
    y_mean = float(scalers["y_mean"][0])
    y_std = float(scalers["y_std"][0])
    y_log = y_norm * y_std + y_mean
    y_mm = np.expm1(y_log)
    y_mm = np.maximum(y_mm, 0.0)
    return y_mm.astype(np.float32)


def calc_continuous_metrics(pred, true, mask):
    pred_flat = pred.reshape(-1)
    true_flat = true.reshape(-1)
    mask_flat = mask.reshape(-1)

    valid = np.isfinite(pred_flat) & np.isfinite(true_flat) & (mask_flat > 0)

    if valid.sum() == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "Bias": np.nan, "Corr": np.nan, "N": 0}

    p = pred_flat[valid]
    t = true_flat[valid]

    mae = float(np.mean(np.abs(p - t)))
    rmse = float(np.sqrt(np.mean((p - t) ** 2)))
    bias = float(np.mean(p - t))

    if np.std(p) < 1e-8 or np.std(t) < 1e-8:
        corr = np.nan
    else:
        corr = float(np.corrcoef(p, t)[0, 1])

    return {"MAE": mae, "RMSE": rmse, "Bias": bias, "Corr": corr, "N": int(valid.sum())}


def calc_categorical_metrics(pred, true, mask, thresholds):
    pred_flat = pred.reshape(-1)
    true_flat = true.reshape(-1)
    mask_flat = mask.reshape(-1)

    valid = np.isfinite(pred_flat) & np.isfinite(true_flat) & (mask_flat > 0)

    p = pred_flat[valid]
    t = true_flat[valid]

    out = {}
    eps = 1e-6

    for thr in thresholds:
        pe = p >= thr
        te = t >= thr

        hit = int(np.sum(pe & te))
        miss = int(np.sum((~pe) & te))
        fa = int(np.sum(pe & (~te)))
        cn = int(np.sum((~pe) & (~te)))

        csi = hit / (hit + miss + fa + eps)
        pod = hit / (hit + miss + eps)
        far = fa / (hit + fa + eps)
        precision = hit / (hit + fa + eps)
        bias_score = (hit + fa) / (hit + miss + eps)

        hss_denom = ((hit + miss) * (miss + cn) + (hit + fa) * (fa + cn) + eps)
        hss = 2.0 * (hit * cn - miss * fa) / hss_denom

        out[str(thr)] = {
            "H": hit,
            "M": miss,
            "F": fa,
            "CN": cn,
            "CSI": float(csi),
            "POD": float(pod),
            "FAR": float(far),
            "Precision": float(precision),
            "BiasScore": float(bias_score),
            "HSS": float(hss)
        }

    return out


def metrics_by_horizon(pred, true, mask):
    results = {"overall": {}, "by_horizon": {}}
    results["overall"]["continuous"] = calc_continuous_metrics(pred, true, mask)
    results["overall"]["categorical"] = calc_categorical_metrics(pred, true, mask, rain_thresholds)

    for h in range(pred.shape[1]):
        key = f"H{h + 1:02d}"
        results["by_horizon"][key] = {
            "continuous": calc_continuous_metrics(pred[:, h:h + 1], true[:, h:h + 1], mask[:, h:h + 1]),
            "categorical": calc_categorical_metrics(
                pred[:, h:h + 1],
                true[:, h:h + 1],
                mask[:, h:h + 1],
                rain_thresholds,
            ),
        }

    return results


# =========================================================
# 9. 训练与预测
# =========================================================

def make_geo_batch(geo_tensor, batch_size):
    if not USE_GEO_EMBEDDING:
        return None
    return geo_tensor.repeat(batch_size, 1, 1, 1).to(device)


def _new_streaming_validation_stats():
    return {
        "n": 0,
        "sum_abs": 0.0,
        "sum_sq": 0.0,
        "sum_err": 0.0,
        "sum_pred": 0.0,
        "sum_true": 0.0,
        "sum_pred2": 0.0,
        "sum_true2": 0.0,
        "sum_pred_true": 0.0,
        "categorical": {
            str(float(thr)): {"H": 0, "M": 0, "F": 0, "CN": 0}
            for thr in VALIDATION_HARD_THRESHOLDS
        },
    }


def _update_streaming_validation_stats(stats, pred_mm, true_mm, mask):
    valid = mask > 0
    if not bool(valid.any()):
        return

    p = pred_mm[valid].double()
    t = true_mm[valid].double()
    err = p - t

    stats["n"] += int(p.numel())
    stats["sum_abs"] += float(err.abs().sum().item())
    stats["sum_sq"] += float((err ** 2).sum().item())
    stats["sum_err"] += float(err.sum().item())
    stats["sum_pred"] += float(p.sum().item())
    stats["sum_true"] += float(t.sum().item())
    stats["sum_pred2"] += float((p ** 2).sum().item())
    stats["sum_true2"] += float((t ** 2).sum().item())
    stats["sum_pred_true"] += float((p * t).sum().item())

    for thr in VALIDATION_HARD_THRESHOLDS:
        pe = pred_mm >= float(thr)
        te = true_mm >= float(thr)
        key = str(float(thr))
        stats["categorical"][key]["H"] += int((pe & te & valid).sum().item())
        stats["categorical"][key]["M"] += int(((~pe) & te & valid).sum().item())
        stats["categorical"][key]["F"] += int((pe & (~te) & valid).sum().item())
        stats["categorical"][key]["CN"] += int(((~pe) & (~te) & valid).sum().item())


def _finalize_streaming_validation_stats(stats):
    eps = 1e-6
    n = max(int(stats["n"]), 1)
    mae = stats["sum_abs"] / n
    rmse = (stats["sum_sq"] / n) ** 0.5
    bias = stats["sum_err"] / n

    cov = stats["sum_pred_true"] - stats["sum_pred"] * stats["sum_true"] / n
    var_p = stats["sum_pred2"] - stats["sum_pred"] ** 2 / n
    var_t = stats["sum_true2"] - stats["sum_true"] ** 2 / n
    corr = cov / max((max(var_p, 0.0) * max(var_t, 0.0)) ** 0.5, eps)

    categorical = {}
    for key, counts in stats["categorical"].items():
        h = counts["H"]
        m = counts["M"]
        f = counts["F"]
        cn = counts["CN"]
        csi = h / (h + m + f + eps)
        pod = h / (h + m + eps)
        far = f / (h + f + eps)
        bias_score = (h + f) / (h + m + eps)
        hss_denom = (h + m) * (m + cn) + (h + f) * (f + cn) + eps
        hss = 2.0 * (h * cn - m * f) / hss_denom
        categorical[key] = {
            "H": h,
            "M": m,
            "F": f,
            "CN": cn,
            "CSI": float(csi),
            "POD": float(pod),
            "FAR": float(far),
            "BiasScore": float(bias_score),
            "HSS": float(hss),
        }

    return {
        "continuous": {
            "MAE": float(mae),
            "RMSE": float(rmse),
            "Bias": float(bias),
            "Corr": float(corr),
            "N": int(stats["n"]),
        },
        "categorical": categorical,
    }


def validation_event_score(hard_metrics):
    c4 = hard_metrics["categorical"]["4.0"]["CSI"]
    c8 = hard_metrics["categorical"]["8.0"]["CSI"]
    h4 = hard_metrics["categorical"]["4.0"]["HSS"]
    h8 = hard_metrics["categorical"]["8.0"]["HSS"]
    return (
        EVENT_SELECTION_WEIGHTS["CSI@4"] * c4
        + EVENT_SELECTION_WEIGHTS["CSI@8"] * c8
        + EVENT_SELECTION_WEIGHTS["HSS@4"] * h4
        + EVENT_SELECTION_WEIGHTS["HSS@8"] * h8
    )


def train_one_epoch(
    model,
    loader,
    optimizer,
    geo_tensor,
    scalers,
    epoch,
    pbudget_alpha,
    lower_bound_stats,
):
    model.train()
    sums = {
        "total": 0.0,
        "data": 0.0,
        "soft_csi": 0.0,
        "rain_event": 0.0,
        "phys_lower_bound": 0.0,
        "phys_ramp": 0.0,
        "phys_active_fraction": 0.0,
        "phys_mean_violation": 0.0,
        "phys_mean_lower_mm": 0.0,
        "phys_mean_pbudget_mm": 0.0,
        "phys_contribution": 0.0,
    }

    for x, y, mask, y_mm, _, pb_pos, pb_mask, _, _ in loader:
        x = x.to(device)
        y = y.to(device)
        mask = mask.to(device)
        y_mm = y_mm.to(device)
        pb_pos = pb_pos.to(device)
        pb_mask = pb_mask.to(device)
        geo_batch = make_geo_batch(geo_tensor, x.size(0))

        optimizer.zero_grad()
        model_output = model(x, geo=geo_batch)
        loss, parts = compute_total_loss(
            model_output,
            y,
            y_mm,
            mask,
            pb_pos,
            pb_mask,
            scalers=scalers,
            epoch=epoch,
            pbudget_alpha=pbudget_alpha,
            lower_bound_stats=lower_bound_stats,
        )
        loss.backward()
        optimizer.step()

        batch_size_current = x.size(0)
        for key in sums:
            sums[key] += float(parts[key].item()) * batch_size_current

    return {
        key: value / len(loader.dataset)
        for key, value in sums.items()
    }


def evaluate_loss(
    model,
    loader,
    geo_tensor,
    scalers,
    epoch,
    pbudget_alpha,
    lower_bound_stats,
):
    model.eval()
    sums = {
        "total": 0.0,
        "data": 0.0,
        "soft_csi": 0.0,
        "rain_event": 0.0,
        "phys_lower_bound": 0.0,
        "phys_ramp": 0.0,
        "phys_active_fraction": 0.0,
        "phys_mean_violation": 0.0,
        "phys_mean_lower_mm": 0.0,
        "phys_mean_pbudget_mm": 0.0,
        "phys_contribution": 0.0,
    }
    hard_stats_overall = _new_streaming_validation_stats()
    hard_stats_early = _new_streaming_validation_stats()
    hard_stats_late = _new_streaming_validation_stats()

    with torch.no_grad():
        for x, y, mask, y_mm, _, pb_pos, pb_mask, _, _ in loader:
            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device)
            y_mm = y_mm.to(device)
            pb_pos = pb_pos.to(device)
            pb_mask = pb_mask.to(device)
            geo_batch = make_geo_batch(geo_tensor, x.size(0))

            model_output = model(x, geo=geo_batch)
            _, parts = compute_total_loss(
                model_output,
                y,
                y_mm,
                mask,
                pb_pos,
                pb_mask,
                scalers=scalers,
                epoch=epoch,
                pbudget_alpha=pbudget_alpha,
                lower_bound_stats=lower_bound_stats,
            )
            batch_size_current = x.size(0)
            for key in sums:
                sums[key] += float(parts[key].item()) * batch_size_current

            y_mean = torch.as_tensor(
                float(scalers["y_mean"][0]),
                device=device,
                dtype=model_output["reg"].dtype,
            )
            y_std = torch.as_tensor(
                float(scalers["y_std"][0]),
                device=device,
                dtype=model_output["reg"].dtype,
            )
            pred_mm = torch_inverse_y_transform(
                model_output["reg"],
                y_mean,
                y_std,
            )

            _update_streaming_validation_stats(
                hard_stats_overall,
                pred_mm,
                y_mm,
                mask,
            )
            _update_streaming_validation_stats(
                hard_stats_early,
                pred_mm[:, EARLY_HORIZON_START:EARLY_HORIZON_END],
                y_mm[:, EARLY_HORIZON_START:EARLY_HORIZON_END],
                mask[:, EARLY_HORIZON_START:EARLY_HORIZON_END],
            )
            _update_streaming_validation_stats(
                hard_stats_late,
                pred_mm[:, LATE_HORIZON_START:LATE_HORIZON_END],
                y_mm[:, LATE_HORIZON_START:LATE_HORIZON_END],
                mask[:, LATE_HORIZON_START:LATE_HORIZON_END],
            )

    result = {
        key: value / len(loader.dataset)
        for key, value in sums.items()
    }
    result["hard_metrics"] = _finalize_streaming_validation_stats(
        hard_stats_overall
    )
    result["hard_metrics_early"] = _finalize_streaming_validation_stats(
        hard_stats_early
    )
    result["hard_metrics_late"] = _finalize_streaming_validation_stats(
        hard_stats_late
    )
    return result


def predict_all(model, loader, scalers, geo_tensor):
    model.eval()

    pred_all = []
    true_all = []
    mask_all = []
    pbudget_raw_all = []
    pbudget_pos_all = []
    pbudget_mask_all = []
    year_all = []
    target_time_all = []

    orig_h = loader.dataset.year_data_list[0].orig_h
    orig_w = loader.dataset.year_data_list[0].orig_w

    with torch.no_grad():
        for x, _, m, y_mm, pb_raw, pb_pos, pb_mask, years, target_time_strs in loader:
            x = x.to(device)
            geo_batch = make_geo_batch(geo_tensor, x.size(0))

            pred_norm = model(x, geo=geo_batch)["reg"].cpu().numpy()
            pred_mm = inverse_y_transform(pred_norm, scalers)

            # 裁剪回原始空间范围。
            pred_mm = pred_mm[:, :, :, :orig_h, :orig_w]
            y_mm_np = y_mm.numpy()[:, :, :, :orig_h, :orig_w]
            m_np = m.numpy()[:, :, :, :orig_h, :orig_w]
            pb_raw_np = pb_raw.numpy()[:, :, :, :orig_h, :orig_w]
            pb_pos_np = pb_pos.numpy()[:, :, :, :orig_h, :orig_w]
            pb_mask_np = pb_mask.numpy()[:, :, :, :orig_h, :orig_w]

            pred_all.append(pred_mm)
            true_all.append(y_mm_np)
            mask_all.append(m_np)
            pbudget_raw_all.append(pb_raw_np)
            pbudget_pos_all.append(pb_pos_np)
            pbudget_mask_all.append(pb_mask_np)

            year_all.extend(str(year) for year in years)
            target_time_all.extend(str(value) for value in target_time_strs)

    return (
        np.concatenate(pred_all, axis=0),
        np.concatenate(true_all, axis=0),
        np.concatenate(mask_all, axis=0),
        np.concatenate(pbudget_raw_all, axis=0),
        np.concatenate(pbudget_pos_all, axis=0),
        np.concatenate(pbudget_mask_all, axis=0),
        np.array(year_all),
        np.array(target_time_all, dtype=object),
    )


# =========================================================
# 10. 绘图
# =========================================================

def plot_loss_curve(train_losses, val_losses, out_dir):
    plt.figure(figsize=(7, 4))
    plt.plot(train_losses, label="Train loss")
    plt.plot(val_losses, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curve.png", dpi=plot_dpi)
    plt.close()


def plot_prediction_case(pred, true, mask, out_dir, sample_idx=0):
    n_h = pred.shape[1]

    true_case = true[sample_idx, :, 0]
    pred_case = pred[sample_idx, :, 0]
    mask_case = mask[sample_idx, :, 0] > 0

    true_show = np.where(mask_case, true_case, np.nan)
    pred_show = np.where(mask_case, pred_case, np.nan)
    err_show = pred_show - true_show

    vmax_tp = np.nanmax([np.nanmax(true_show), np.nanmax(pred_show), 1.0])

    vmax_err = np.nanmax(np.abs(err_show))
    if not np.isfinite(vmax_err) or vmax_err < 1e-6:
        vmax_err = 1.0

    err_norm = TwoSlopeNorm(vmin=-vmax_err, vcenter=0.0, vmax=vmax_err)

    _, axes = plt.subplots(3, n_h, figsize=(4.2 * n_h, 10.5), squeeze=False)

    for h in range(n_h):
        im0 = axes[0, h].imshow(true_show[h], origin="upper", vmin=0, vmax=vmax_tp, cmap="viridis")
        axes[0, h].set_title(f"Lead {h + 1}: True")
        axes[0, h].set_xticks([])
        axes[0, h].set_yticks([])
        plt.colorbar(im0, ax=axes[0, h], fraction=0.046, pad=0.04)

        im1 = axes[1, h].imshow(pred_show[h], origin="upper", vmin=0, vmax=vmax_tp, cmap="viridis")
        axes[1, h].set_title(f"Lead {h + 1}: Pred")
        axes[1, h].set_xticks([])
        axes[1, h].set_yticks([])
        plt.colorbar(im1, ax=axes[1, h], fraction=0.046, pad=0.04)

        im2 = axes[2, h].imshow(err_show[h], origin="upper", cmap="RdBu_r", norm=err_norm)
        axes[2, h].set_title(f"Lead {h + 1}: Error")
        axes[2, h].set_xticks([])
        axes[2, h].set_yticks([])
        plt.colorbar(im2, ax=axes[2, h], fraction=0.046, pad=0.04)

    plt.suptitle(f"Test sample {sample_idx}: true, prediction, and error", fontsize=15)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    out_file = out_dir / f"test_case_{sample_idx:04d}_lead1-8_true_pred_error.png"
    plt.savefig(out_file, dpi=plot_dpi)
    plt.close()
    print("预测图已保存：", out_file)


def plot_area_mean(pred, true, mask, out_dir, lead_idx=0):
    pred_h = pred[:, lead_idx, 0]
    true_h = true[:, lead_idx, 0]
    mask_h = mask[:, lead_idx, 0]

    pred_mean = np.sum(pred_h * mask_h, axis=(1, 2)) / (np.sum(mask_h, axis=(1, 2)) + 1e-6)
    true_mean = np.sum(true_h * mask_h, axis=(1, 2)) / (np.sum(mask_h, axis=(1, 2)) + 1e-6)

    plt.figure(figsize=(10, 4))
    plt.plot(true_mean, label="True")
    plt.plot(pred_mean, label="Pred")
    plt.xlabel("Test sample index")
    plt.ylabel("Area-mean precipitation, mm/h")
    plt.title(f"Area-mean prediction, lead {lead_idx + 1}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"test_area_mean_lead{lead_idx + 1}.png", dpi=plot_dpi)
    plt.close()


# =========================================================
# 11. 主程序
# =========================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Device:", device)
    print()
    print(
        "Variable files:",
        {key: str(value) for key, value in VARIABLE_FILES.items()},
    )
    print("Train/Val/Test years:", TRAIN_YEARS, VAL_YEARS, TEST_YEARS)

    print()
    print("读取训练年份数据...")
    train_years = load_years(TRAIN_YEARS)
    print()
    print("读取验证年份数据...")
    val_years = load_years(VAL_YEARS)
    print()
    print("读取测试年份数据...")
    test_years = load_years(TEST_YEARS)

    scalers = compute_scalers(train_years)
    pbudget_alpha = compute_pbudget_alpha(train_years)
    lower_bound_stats = fit_pbudget_lower_bound(
        train_years,
        pbudget_alpha,
    )
    with open(
        OUT_DIR / "pbudget_lower_bound.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            lower_bound_stats,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("Scalers:")
    print("x_mean:", scalers["x_mean"])
    print("x_std:", scalers["x_std"])
    print("y_mean:", scalers["y_mean"])
    print("y_std:", scalers["y_std"])

    train_ds = Era5NowcastDataset(train_years, scalers)
    val_ds = Era5NowcastDataset(val_years, scalers)
    test_ds = Era5NowcastDataset(test_years, scalers)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    input_channels = train_years[0].X_all.shape[1]
    geo_channels = train_years[0].geo.shape[0]
    geo_tensor = torch.from_numpy(train_years[0].geo[None]).float()
    if USE_GEO_EMBEDDING:
        geo_tensor = geo_tensor.to(device)

    model = GeoConvLSTMForecastModel(
        input_channels=input_channels,
        use_geo_embedding=USE_GEO_EMBEDDING,
        geo_in_channels=geo_channels,
        geo_embed_channels=geo_embed_channels,
        latent_channels=latent_channels,
        horizon=horizon,
        dropout=dropout_rate,
        use_rain_event_head=USE_RAIN_EVENT_HEAD,
    ).to(device)

    if not BASELINE_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"找不到基线检查点：{BASELINE_CHECKPOINT}。"
            "请先运行geoconvlstm_baseline_8h.py，"
            "或修改BASELINE_CHECKPOINT。"
        )
    baseline_state = torch.load(
        BASELINE_CHECKPOINT,
        map_location=device,
    )
    model.load_state_dict(baseline_state, strict=True)
    print("已载入基线检查点：", BASELINE_CHECKPOINT)

    print()
    print("Model:")
    print(model)
    print("input_channels:", input_channels)
    print("feature_names:", train_years[0].feature_names)
    print("USE_GEO_EMBEDDING:", USE_GEO_EMBEDDING)
    print("geo_channels:", geo_channels)
    print("fine_tune_learning_rate:", learning_rate)
    print("PHYS_NEIGHBORHOOD_KERNEL:", PHYS_NEIGHBORHOOD_KERNEL)
    print("PHYS_MULTISCALE_PATCH_SIZES:", PHYS_MULTISCALE_PATCH_SIZES)
    print("PHYS_CUMULATIVE_PATCH_SIZES:", PHYS_CUMULATIVE_PATCH_SIZES)
    print("PHYS_LOWER_BOUND_WEIGHT:", PHYS_LOWER_BOUND_WEIGHT)
    print("PHYS_HORIZON_WEIGHTS:", PHYS_HORIZON_WEIGHTS)
    print("lower_bound_stats:", lower_bound_stats)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=lr_scheduler_factor,
        patience=lr_scheduler_patience,
        min_lr=min_learning_rate,
    )

    # 基线检查点仅用于微调起点和诊断，不参与最优微调轮次的候选比较。
    baseline_val_parts = evaluate_loss(
        model,
        val_loader,
        geo_tensor,
        scalers,
        epoch=0,
        pbudget_alpha=pbudget_alpha,
        lower_bound_stats=lower_bound_stats,
    )
    baseline_selection_loss = baseline_val_parts["total"]
    baseline_hard = baseline_val_parts["hard_metrics"]
    baseline_hard_early = baseline_val_parts["hard_metrics_early"]
    baseline_hard_late = baseline_val_parts["hard_metrics_late"]
    baseline_event_score = validation_event_score(baseline_hard_late)

    torch.save(
        model.state_dict(),
        OUT_DIR / "baseline_initial_model.pt",
    )

    print()
    print(
        "Baseline validation loss = "
        f"{baseline_selection_loss:.6f}"
    )
    print(
        "Baseline overall hard metrics: "
        f"RMSE={baseline_hard['continuous']['RMSE']:.6f}, "
        f"CSI4={baseline_hard['categorical']['4.0']['CSI']:.6f}, "
        f"CSI8={baseline_hard['categorical']['8.0']['CSI']:.6f}"
    )
    print(
        "Baseline early H01-H04: "
        f"RMSE={baseline_hard_early['continuous']['RMSE']:.6f}"
    )
    print(
        "Baseline late H05-H08: "
        f"RMSE={baseline_hard_late['continuous']['RMSE']:.6f}, "
        f"CSI4={baseline_hard_late['categorical']['4.0']['CSI']:.6f}, "
        f"CSI8={baseline_hard_late['categorical']['8.0']['CSI']:.6f}, "
        f"HSS4={baseline_hard_late['categorical']['4.0']['HSS']:.6f}, "
        f"HSS8={baseline_hard_late['categorical']['8.0']['HSS']:.6f}, "
        f"late_event_score={baseline_event_score:.6f}"
    )

    best_selection_loss = float("inf")
    best_event_score = -float("inf")
    best_hard = None
    best_hard_early = None
    best_hard_late = None
    best_epoch = -1
    wait = 0

    history_keys = [
        "total",
        "data",
        "soft_csi",
        "rain_event",
        "phys_lower_bound",
        "phys_ramp",
        "phys_active_fraction",
        "phys_mean_violation",
        "phys_mean_lower_mm",
        "phys_mean_pbudget_mm",
        "phys_contribution",
    ]
    train_history = {key: [] for key in history_keys}
    val_history = {key: [] for key in history_keys}
    validation_hard_history = []

    print()
    print("开始物理约束微调...")
    for epoch in range(1, max_epochs + 1):
        train_parts = train_one_epoch(
            model,
            train_loader,
            optimizer,
            geo_tensor,
            scalers,
            epoch,
            pbudget_alpha,
            lower_bound_stats,
        )
        val_parts = evaluate_loss(
            model,
            val_loader,
            geo_tensor,
            scalers,
            epoch,
            pbudget_alpha,
            lower_bound_stats,
        )

        # 学习率调度仍依据与基线共有的可微目标。
        val_baseline_objective = (
            val_parts["data"]
            + SOFT_CSI_LOSS_WEIGHT * val_parts["soft_csi"]
            + RAIN_EVENT_LOSS_WEIGHT * val_parts["rain_event"]
        )
        scheduler.step(val_baseline_objective)
        current_lr = optimizer.param_groups[0]["lr"]

        val_hard = val_parts["hard_metrics"]
        val_hard_early = val_parts["hard_metrics_early"]
        val_hard_late = val_parts["hard_metrics_late"]
        val_event_score = validation_event_score(val_hard_late)

        for key in history_keys:
            train_history[key].append(train_parts[key])
            val_history[key].append(val_parts[key])

        validation_hard_history.append(
            {
                "epoch": int(epoch),
                "baseline_objective": float(val_baseline_objective),
                "late_event_score": float(val_event_score),
                "hard_metrics_overall": val_hard,
                "hard_metrics_early_h01_h04": val_hard_early,
                "hard_metrics_late_h05_h08": val_hard_late,
                "late_csi4_gain_from_baseline": float(
                    val_hard_late["categorical"]["4.0"]["CSI"]
                    - baseline_hard_late["categorical"]["4.0"]["CSI"]
                ),
                "late_csi8_gain_from_baseline": float(
                    val_hard_late["categorical"]["8.0"]["CSI"]
                    - baseline_hard_late["categorical"]["8.0"]["CSI"]
                ),
                "learning_rate": float(current_lr),
            }
        )

        print(
            f"Epoch {epoch:03d}/{max_epochs} | "
            f"train_total={train_parts['total']:.6f} "
            f"data={train_parts['data']:.6f} "
            f"csi={train_parts['soft_csi']:.6f} "
            f"event={train_parts['rain_event']:.6f} "
            f"phys={train_parts['phys_lower_bound']:.6f} "
            f"active={train_parts['phys_active_fraction']:.3f} "
            f"contrib={train_parts['phys_contribution']:.6f} "
            f"ramp={train_parts['phys_ramp']:.3f} | "
            f"val_data={val_parts['data']:.6f} "
            f"baseline_obj={val_baseline_objective:.6f} "
            f"overall_RMSE={val_hard['continuous']['RMSE']:.6f} "
            f"early_RMSE={val_hard_early['continuous']['RMSE']:.6f} "
            f"late_RMSE={val_hard_late['continuous']['RMSE']:.6f} "
            f"late_CSI4={val_hard_late['categorical']['4.0']['CSI']:.6f} "
            f"late_CSI8={val_hard_late['categorical']['8.0']['CSI']:.6f} "
            f"late_HSS4={val_hard_late['categorical']['4.0']['HSS']:.6f} "
            f"late_HSS8={val_hard_late['categorical']['8.0']['HSS']:.6f} "
            f"late_event_score={val_event_score:.6f} | "
            f"lr={current_lr:.2e}"
        )

        better_event_model = (
            best_epoch < 0
            or val_event_score > best_event_score + 1e-8
        )
        if better_event_model:
            best_selection_loss = val_baseline_objective
            best_event_score = val_event_score
            best_hard = val_hard
            best_hard_early = val_hard_early
            best_hard_late = val_hard_late
            best_epoch = epoch
            wait = 0
            torch.save(
                model.state_dict(),
                OUT_DIR / "best_model.pt",
            )
            print(
                f"  -> 已保存第{epoch}轮微调检查点，"
                f"H05-H08事件得分={val_event_score:.6f}。"
            )
        else:
            wait += 1

        if wait >= early_stopping_patience:
            print(
                f"Early stopping at epoch {epoch}. "
                f"Best physics epoch = {best_epoch}, "
                f"best event score = {best_event_score:.6f}"
            )
            break

    torch.save(model.state_dict(), OUT_DIR / "last_model.pt")
    plot_loss_curve(
        train_history["total"],
        val_history["total"],
        OUT_DIR,
    )

    best_state = torch.load(OUT_DIR / "best_model.pt", map_location=device)
    model.load_state_dict(best_state, strict=True)

    print()
    print("开始测试集预测...")
    (
        pred_mm,
        true_mm,
        test_mask,
        pbudget_raw_test,
        pbudget_pos_test,
        pbudget_mask_test,
        test_year_labels,
        test_target_times,
    ) = predict_all(model, test_loader, scalers, geo_tensor)

    metrics = metrics_by_horizon(pred_mm, true_mm, test_mask)
    phys_valid_mask = test_mask * pbudget_mask_test
    pbudget_pos_cal_test = np.maximum(
        pbudget_pos_test * pbudget_alpha,
        0.0,
    ).astype(np.float32)

    metrics["physics_diagnostics"] = {
        "fine_tuning": True,
        "baseline_checkpoint": str(BASELINE_CHECKPOINT),
        "best_finetune_epoch": int(best_epoch),
        "checkpoint_selection_rule": (
            "选择H05-H08强降水CSI/HSS综合得分最高的微调轮次；"
            "基线检查点不参与候选比较，不设置MAE/RMSE回退条件"
        ),
        "baseline_validation_selection_loss": float(
            baseline_selection_loss
        ),
        "best_validation_selection_loss": float(
            best_selection_loss
        ),
        "baseline_validation_hard_metrics_overall": baseline_hard,
        "best_validation_hard_metrics_overall": best_hard,
        "baseline_validation_hard_metrics_early_h01_h04": (
            baseline_hard_early
        ),
        "best_validation_hard_metrics_early_h01_h04": best_hard_early,
        "baseline_validation_hard_metrics_late_h05_h08": (
            baseline_hard_late
        ),
        "best_validation_hard_metrics_late_h05_h08": best_hard_late,
        "baseline_validation_late_event_score": float(
            baseline_event_score
        ),
        "best_validation_late_event_score": float(best_event_score),
        "validation_selection_policy": {
            "focus": "H05-H08 strong-rain CSI/HSS",
            "event_selection_weights": EVENT_SELECTION_WEIGHTS,
            "early_horizon_slice_python": [
                int(EARLY_HORIZON_START),
                int(EARLY_HORIZON_END),
            ],
            "late_horizon_slice_python": [
                int(LATE_HORIZON_START),
                int(LATE_HORIZON_END),
            ],
        },
        "pbudget_alpha": float(pbudget_alpha),
        "lower_bound_statistics": lower_bound_stats,
        "physics_patch_size": int(PHYS_PATCH_SIZE),
        "physics_horizon_weights": [
            float(value) for value in PHYS_HORIZON_WEIGHTS
        ],
        "physics_lower_bound_weight": float(
            PHYS_LOWER_BOUND_WEIGHT
        ),
        "pbudget_raw_vs_true": metrics_by_horizon(
            pbudget_raw_test,
            true_mm,
            phys_valid_mask,
        ),
        "pbudget_positive_calibrated_vs_true": metrics_by_horizon(
            pbudget_pos_cal_test,
            true_mm,
            phys_valid_mask,
        ),
        "pred_vs_pbudget_positive_calibrated": metrics_by_horizon(
            pred_mm,
            pbudget_pos_cal_test,
            phys_valid_mask,
        ),
    }
    metrics["loss_history_components"] = {
        "train_" + key: values
        for key, values in train_history.items()
    }
    metrics["loss_history_components"].update(
        {
            "val_" + key: values
            for key, values in val_history.items()
        }
    )
    metrics["loss_history_components"][
        "validation_hard_history"
    ] = validation_hard_history

    print()
    print("Overall continuous metrics:")
    print(metrics["overall"]["continuous"])
    print()
    print("By-horizon continuous metrics:")
    for key, value in metrics["by_horizon"].items():
        print(key, value["continuous"])

    with open(
        OUT_DIR / "test_metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)

    np.savez_compressed(
        OUT_DIR / "test_predictions.npz",
        pred_test_mm=pred_mm,
        true_test_mm=true_mm,
        test_mask=test_mask,
        test_year_labels=test_year_labels,
        test_target_times=test_target_times,
        pbudget_raw_test=pbudget_raw_test,
        pbudget_positive_test=pbudget_pos_test,
        pbudget_positive_calibrated_test=pbudget_pos_cal_test,
        pbudget_mask_test=pbudget_mask_test,
        pbudget_alpha=np.array([pbudget_alpha], dtype=np.float32),
        pbudget_lower_offset_mm=np.array(
            [lower_bound_stats["lower_offset_mm"]],
            dtype=np.float32,
        ),
        physics_patch_size=np.array(
            [PHYS_PATCH_SIZE],
            dtype=np.int32,
        ),
        physics_horizon_weights=np.array(
            PHYS_HORIZON_WEIGHTS,
            dtype=np.float32,
        ),
        x_mean=scalers["x_mean"],
        x_std=scalers["x_std"],
        y_mean=scalers["y_mean"],
        y_std=scalers["y_std"],
        input_vars=np.array(train_years[0].feature_names),
        use_geo_embedding=np.array([USE_GEO_EMBEDDING]),
        geo_channels=np.array([geo_channels]),
        lat=test_years[0].lat,
        lon=test_years[0].lon,
        **{
            "train_" + key: np.array(values)
            for key, values in train_history.items()
        },
        **{
            "val_" + key: np.array(values)
            for key, values in val_history.items()
        },
    )

    plot_area_mean(pred_mm, true_mm, test_mask, OUT_DIR, lead_idx=0)
    if plot_case_indices is None:
        indices = list(range(min(num_plot_cases, pred_mm.shape[0])))
    else:
        indices = [
            index
            for index in plot_case_indices
            if 0 <= index < pred_mm.shape[0]
        ]
    for index in indices:
        plot_prediction_case(
            pred_mm,
            true_mm,
            test_mask,
            OUT_DIR,
            sample_idx=index,
        )

    with open(
        OUT_DIR / "validation_hard_history.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "baseline_overall": baseline_hard,
                "baseline_early_h01_h04": baseline_hard_early,
                "baseline_late_h05_h08": baseline_hard_late,
                "baseline_late_event_score": float(
                    baseline_event_score
                ),
                "best_epoch": int(best_epoch),
                "best_overall": best_hard,
                "best_early_h01_h04": best_hard_early,
                "best_late_h05_h08": best_hard_late,
                "best_late_event_score": float(best_event_score),
                "history": validation_hard_history,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    config = {
        "VARIABLE_DATA_DIR": str(VARIABLE_DATA_DIR),
        "VARIABLE_FILES": {
            key: str(value) for key, value in VARIABLE_FILES.items()
        },
        "SELECT_MONTHS": SELECT_MONTHS,
        "TP_INPUT_UNITS": TP_INPUT_UNITS,
        "OUT_DIR": str(OUT_DIR),
        "TRAIN_YEARS": TRAIN_YEARS,
        "VAL_YEARS": VAL_YEARS,
        "TEST_YEARS": TEST_YEARS,
        "INPUT_VARS": INPUT_VARS,
        "TARGET_VAR": TARGET_VAR,
        "USE_GEO_EMBEDDING": USE_GEO_EMBEDDING,
        "history_len": history_len,
        "horizon": horizon,
        "PAD_MULTIPLE": PAD_MULTIPLE,
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "early_stopping_patience": early_stopping_patience,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "lr_scheduler_factor": lr_scheduler_factor,
        "lr_scheduler_patience": lr_scheduler_patience,
        "min_learning_rate": min_learning_rate,
        "random_seed": random_seed,
        "num_workers": num_workers,
        "latent_channels": latent_channels,
        "geo_embed_channels": geo_embed_channels,
        "dropout_rate": dropout_rate,
        "rain_thresholds": rain_thresholds,
        "RAIN_WEIGHT_THRESHOLDS": RAIN_WEIGHT_THRESHOLDS,
        "RAIN_WEIGHT_VALUES": RAIN_WEIGHT_VALUES,
        "NORMALIZE_RAIN_WEIGHTS": NORMALIZE_RAIN_WEIGHTS,
        "DATA_HUBER_BETA": DATA_HUBER_BETA,
        "SOFT_EVENT_THRESHOLDS": SOFT_EVENT_THRESHOLDS,
        "SOFT_EVENT_WEIGHTS": SOFT_EVENT_WEIGHTS,
        "HORIZON_LOSS_WEIGHTS": HORIZON_LOSS_WEIGHTS,
        "SOFT_THRESHOLD_SHARPNESS": SOFT_THRESHOLD_SHARPNESS,
        "SOFT_CSI_LOSS_WEIGHT": SOFT_CSI_LOSS_WEIGHT,
        "USE_RAIN_EVENT_HEAD": USE_RAIN_EVENT_HEAD,
        "RAIN_EVENT_THRESHOLD": RAIN_EVENT_THRESHOLD,
        "RAIN_EVENT_LOSS_WEIGHT": RAIN_EVENT_LOSS_WEIGHT,
        "RAIN_EVENT_POS_WEIGHT": RAIN_EVENT_POS_WEIGHT,
        "BASELINE_OUT_DIR": str(BASELINE_OUT_DIR),
        "BASELINE_CHECKPOINT": str(BASELINE_CHECKPOINT),
        "EVAP_PATH": str(EVAP_PATH),
        "TCWV_PATH": str(TCWV_PATH),
        "VIMD_PATH": str(VIMD_PATH),
        "VIMD_TIME_INTEGRATION_MODE": VIMD_TIME_INTEGRATION_MODE,
        "VIMD_CONVENTION": VIMD_CONVENTION,
        "EVAP_INPUT_MODE": EVAP_INPUT_MODE,
        "EVAP_SIGN_MODE": EVAP_SIGN_MODE,
        "EXPECTED_DT_SECONDS": EXPECTED_DT_SECONDS,
        "DT_TOLERANCE_SECONDS": DT_TOLERANCE_SECONDS,
        "CALIBRATE_PBUDGET_POSITIVE": CALIBRATE_PBUDGET_POSITIVE,
        "PBUDGET_ALPHA_MIN": PBUDGET_ALPHA_MIN,
        "PBUDGET_ALPHA_MAX": PBUDGET_ALPHA_MAX,
        "pbudget_alpha": float(pbudget_alpha),
        "PHYS_HORIZON_WEIGHTS": PHYS_HORIZON_WEIGHTS,
        "PHYS_PATCH_SIZE": PHYS_PATCH_SIZE,
        "PHYS_PATCH_MIN_VALID_FRACTION": (
            PHYS_PATCH_MIN_VALID_FRACTION
        ),
        "PBUDGET_RESIDUAL_LOWER_QUANTILE": (
            PBUDGET_RESIDUAL_LOWER_QUANTILE
        ),
        "PBUDGET_RESIDUAL_CHUNK_TIMES": PBUDGET_RESIDUAL_CHUNK_TIMES,
        "PBUDGET_RESIDUAL_SAMPLES_PER_CHUNK": (
            PBUDGET_RESIDUAL_SAMPLES_PER_CHUNK
        ),
        "PBUDGET_RESIDUAL_MAX_SAMPLES": PBUDGET_RESIDUAL_MAX_SAMPLES,
        "PBUDGET_RESIDUAL_RANDOM_SEED": PBUDGET_RESIDUAL_RANDOM_SEED,
        "PBUDGET_MIN_LOWER_OFFSET_MM": PBUDGET_MIN_LOWER_OFFSET_MM,
        "PHYS_NEIGHBORHOOD_KERNEL": PHYS_NEIGHBORHOOD_KERNEL,
        "PHYS_NEIGHBORHOOD_THRESHOLDS_MM": (
            PHYS_NEIGHBORHOOD_THRESHOLDS_MM
        ),
        "PHYS_NEIGHBORHOOD_THRESHOLD_WEIGHTS": (
            PHYS_NEIGHBORHOOD_THRESHOLD_WEIGHTS
        ),
        "PHYS_NEIGHBORHOOD_SOFTNESS_MM": (
            PHYS_NEIGHBORHOOD_SOFTNESS_MM
        ),
        "PHYS_MULTISCALE_PATCH_SIZES": PHYS_MULTISCALE_PATCH_SIZES,
        "PHYS_MULTISCALE_WEIGHTS": PHYS_MULTISCALE_WEIGHTS,
        "PHYS_MULTISCALE_PBUDGET_THRESHOLD_MM": (
            PHYS_MULTISCALE_PBUDGET_THRESHOLD_MM
        ),
        "PHYS_CUMULATIVE_PATCH_SIZES": (
            PHYS_CUMULATIVE_PATCH_SIZES
        ),
        "PHYS_CUMULATIVE_SCALE_WEIGHTS": (
            PHYS_CUMULATIVE_SCALE_WEIGHTS
        ),
        "PHYS_CUMULATIVE_PBUDGET_THRESHOLD_MM": (
            PHYS_CUMULATIVE_PBUDGET_THRESHOLD_MM
        ),
        "PHYS_CUMULATIVE_MIN_VALID_FRACTION": (
            PHYS_CUMULATIVE_MIN_VALID_FRACTION
        ),
        "PHYS_NEIGHBORHOOD_COMPONENT_WEIGHT": (
            PHYS_NEIGHBORHOOD_COMPONENT_WEIGHT
        ),
        "PHYS_MULTISCALE_COMPONENT_WEIGHT": (
            PHYS_MULTISCALE_COMPONENT_WEIGHT
        ),
        "PHYS_CUMULATIVE_COMPONENT_WEIGHT": (
            PHYS_CUMULATIVE_COMPONENT_WEIGHT
        ),
        "PHYS_START_EPOCH": PHYS_START_EPOCH,
        "PHYS_RAMP_EPOCHS": PHYS_RAMP_EPOCHS,
        "PHYS_LOWER_BOUND_WEIGHT": PHYS_LOWER_BOUND_WEIGHT,
        "PHYS_HUBER_BETA": PHYS_HUBER_BETA,
        "VALIDATION_HARD_THRESHOLDS": VALIDATION_HARD_THRESHOLDS,
        "EARLY_HORIZON_START": EARLY_HORIZON_START,
        "EARLY_HORIZON_END": EARLY_HORIZON_END,
        "LATE_HORIZON_START": LATE_HORIZON_START,
        "LATE_HORIZON_END": LATE_HORIZON_END,
        "EVENT_SELECTION_WEIGHTS": EVENT_SELECTION_WEIGHTS,
        "lower_bound_statistics": lower_bound_stats,
        "best_epoch": int(best_epoch),
        "best_validation_late_event_score": float(best_event_score),
    }

    with open(
        OUT_DIR / "run_config.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(config, file, ensure_ascii=False, indent=2)

    print()
    print("结果已保存到：")
    print(OUT_DIR)
    print(
        "包括：baseline_initial_model.pt、best_model.pt、last_model.pt、"
        "pbudget_lower_bound.json、validation_hard_history.json、"
        "test_predictions.npz、test_metrics.json、run_config.json和预测图。"
    )


if __name__ == "__main__":
    main()
