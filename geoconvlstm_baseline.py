# -*- coding: utf-8 -*-
"""
用于逐小时降水临近预报的 Geo-ConvLSTM 基线模型。

主要设置：
1. 输入变量为 tp_hourly、u10、v10 和 d2m；
2. 使用连续8小时历史数据预测未来8小时降水；
3. 模型由地理信息嵌入、CNN编码器、ConvLSTM和CNN解码器组成；
4. 第一个未来时刻以零潜在张量作为ConvLSTM输入，后续时刻使用上一时刻的隐藏状态；
5. 训练、验证和测试年份严格分离，避免时间信息泄漏。

输入张量：
    X: (B, history_len, C, H, W)
    Y: (B, horizon, 1, H, W)
    M: (B, horizon, 1, H, W)

输出：
    pred: (B, horizon, 1, H, W)

依赖：
pip install xarray netCDF4 numpy pandas matplotlib torch

四个变量分别存放在VARIABLE_FILES中。脚本按时间和经纬度取交集，
再在内存中按年份及4--9月切分，不改写原始NetCDF文件。
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
OUT_DIR = PROJECT_DIR / "outputs" / "geoconvlstm_baseline_8h"

# 按年份严格隔离训练集、验证集和测试集，避免时间信息泄漏。
TRAIN_YEARS = [2020, 2021]
VAL_YEARS = [2022]
TEST_YEARS = [2023]

# 动态输入变量和预测目标。
INPUT_VARS = ["tp_hourly", "u10", "v10", "d2m"]
TARGET_VAR = "tp_hourly"

# =========================================================
# 平衡长尾与事件识别的联合损失
# =========================================================
# 以较温和的加权 Huber 代替 1/1.5/3/6/10 的加权 MSE，减少极端格点梯度爆炸。
RAIN_WEIGHT_THRESHOLDS = [0.1, 1.0, 4.0, 8.0]
RAIN_WEIGHT_VALUES = [1.0, 1.2, 2.0, 4.0, 6.0]
NORMALIZE_RAIN_WEIGHTS = True
DATA_HUBER_BETA = 1.0

# Soft CSI按预报时效和降水阈值分别计算，避免短时效样本掩盖长时效样本。
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

# 是否启用静态地理信息嵌入。
USE_GEO_EMBEDDING = True

# 地理信息通道依次为lat_norm、lon_norm、sin_lat、cos_lat、sin_lon和cos_lon。
history_len = 8
horizon = 8
# 将空间尺寸补齐为偶数，以适配一次2倍下采样；原始0.25°网格不变。
PAD_MULTIPLE = 2

# 训练参数
batch_size = 16
max_epochs = 50
early_stopping_patience = 8
learning_rate = 5e-4
weight_decay = 0.0
lr_scheduler_factor = 0.5
lr_scheduler_patience = 3
min_learning_rate = 1e-5
random_seed = 42
num_workers = 0

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

if len(HORIZON_LOSS_WEIGHTS) != horizon:
    raise ValueError(
        f"HORIZON_LOSS_WEIGHTS长度={len(HORIZON_LOSS_WEIGHTS)}，但horizon={horizon}。"
    )
if len(RAIN_WEIGHT_VALUES) != len(RAIN_WEIGHT_THRESHOLDS) + 1:
    raise ValueError("RAIN_WEIGHT_VALUES长度应比RAIN_WEIGHT_THRESHOLDS多1。")
if len(SOFT_EVENT_THRESHOLDS) != len(SOFT_EVENT_WEIGHTS):
    raise ValueError("SOFT_EVENT_THRESHOLDS与SOFT_EVENT_WEIGHTS长度不一致。")


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

        # 删除重复时间，随后load使关闭源文件后数据仍可使用。
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
        x = (x - self.x_mean) / self.x_std
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        y = (y - self.y_mean) / self.y_std
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        target_time_str = "|".join([str(t) for t in yd.times[out_start:out_end]])

        return (
            torch.from_numpy(x),
            torch.from_numpy(y),
            torch.from_numpy(m.astype(np.float32)),
            torch.from_numpy(y_mm.astype(np.float32)),
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


def compute_total_loss(model_output, target_norm, target_mm, mask, scalers):
    """计算加权Huber、Soft CSI和雨事件辅助头组成的基线损失。"""
    pred_norm = model_output["reg"]
    rain_logits = model_output.get("rain_logits")
    data_loss = masked_weighted_huber_loss(
        pred_norm, target_norm, target_mm, mask
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

    total_loss = (
        data_loss
        + SOFT_CSI_LOSS_WEIGHT * soft_csi
        + RAIN_EVENT_LOSS_WEIGHT * rain_event
    )
    return total_loss, {
        "total": total_loss.detach(),
        "data": data_loss.detach(),
        "soft_csi": soft_csi.detach(),
        "rain_event": rain_event.detach(),
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


def train_one_epoch(model, loader, optimizer, geo_tensor, scalers):
    model.train()
    sums = {
        "total": 0.0,
        "data": 0.0,
        "soft_csi": 0.0,
        "rain_event": 0.0,
    }

    for x, y, mask, y_mm, _, _ in loader:
        x = x.to(device)
        y = y.to(device)
        mask = mask.to(device)
        y_mm = y_mm.to(device)
        geo_batch = make_geo_batch(geo_tensor, x.size(0))

        optimizer.zero_grad()
        model_output = model(x, geo=geo_batch)
        loss, parts = compute_total_loss(
            model_output,
            y,
            y_mm,
            mask,
            scalers=scalers,
        )
        loss.backward()
        optimizer.step()

        batch_size_current = x.size(0)
        for key in sums:
            sums[key] += float(parts[key].item()) * batch_size_current

    return {key: value / len(loader.dataset) for key, value in sums.items()}


def evaluate_loss(model, loader, geo_tensor, scalers):
    model.eval()
    sums = {
        "total": 0.0,
        "data": 0.0,
        "soft_csi": 0.0,
        "rain_event": 0.0,
    }

    with torch.no_grad():
        for x, y, mask, y_mm, _, _ in loader:
            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device)
            y_mm = y_mm.to(device)
            geo_batch = make_geo_batch(geo_tensor, x.size(0))

            model_output = model(x, geo=geo_batch)
            _, parts = compute_total_loss(
                model_output,
                y,
                y_mm,
                mask,
                scalers=scalers,
            )
            batch_size_current = x.size(0)
            for key in sums:
                sums[key] += float(parts[key].item()) * batch_size_current

    return {key: value / len(loader.dataset) for key, value in sums.items()}


def predict_all(model, loader, scalers, geo_tensor):
    model.eval()

    pred_all = []
    true_all = []
    mask_all = []
    year_all = []
    target_time_all = []

    orig_h = loader.dataset.year_data_list[0].orig_h
    orig_w = loader.dataset.year_data_list[0].orig_w

    with torch.no_grad():
        for x, _, mask, y_mm, years, target_time_strs in loader:
            x = x.to(device)
            geo_batch = make_geo_batch(geo_tensor, x.size(0))

            pred_norm = model(x, geo=geo_batch)["reg"].cpu().numpy()
            pred_mm = inverse_y_transform(pred_norm, scalers)

            # 裁剪回原始空间范围。
            pred_all.append(pred_mm[:, :, :, :orig_h, :orig_w])
            true_all.append(y_mm.numpy()[:, :, :, :orig_h, :orig_w])
            mask_all.append(mask.numpy()[:, :, :, :orig_h, :orig_w])
            year_all.extend(str(year) for year in years)
            target_time_all.extend(str(value) for value in target_time_strs)

    return (
        np.concatenate(pred_all, axis=0),
        np.concatenate(true_all, axis=0),
        np.concatenate(mask_all, axis=0),
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
    print("\nVariable files:", {key: str(value) for key, value in VARIABLE_FILES.items()})
    print("Train/Val/Test years:", TRAIN_YEARS, VAL_YEARS, TEST_YEARS)

    print("\n读取训练年份数据...")
    train_years = load_years(TRAIN_YEARS)
    print("\n读取验证年份数据...")
    val_years = load_years(VAL_YEARS)
    print("\n读取测试年份数据...")
    test_years = load_years(TEST_YEARS)

    scalers = compute_scalers(train_years)
    print("\nScalers:")
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

    print("\nModel:")
    print(model)
    print("input_channels:", input_channels)
    print("feature_names:", train_years[0].feature_names)
    print("USE_GEO_EMBEDDING:", USE_GEO_EMBEDDING)
    print("geo_channels:", geo_channels)

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

    best_val = float("inf")
    best_epoch = 0
    wait = 0

    train_losses = []
    val_losses = []
    train_data_losses = []
    val_data_losses = []
    train_soft_csi_losses = []
    val_soft_csi_losses = []
    train_rain_event_losses = []
    val_rain_event_losses = []

    print("\n开始训练...")
    for epoch in range(1, max_epochs + 1):
        train_parts = train_one_epoch(
            model,
            train_loader,
            optimizer,
            geo_tensor,
            scalers,
        )
        val_parts = evaluate_loss(model, val_loader, geo_tensor, scalers)
        val_loss = val_parts["total"]
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        train_losses.append(train_parts["total"])
        val_losses.append(val_parts["total"])
        train_data_losses.append(train_parts["data"])
        val_data_losses.append(val_parts["data"])
        train_soft_csi_losses.append(train_parts["soft_csi"])
        val_soft_csi_losses.append(val_parts["soft_csi"])
        train_rain_event_losses.append(train_parts["rain_event"])
        val_rain_event_losses.append(val_parts["rain_event"])

        print(
            f"Epoch {epoch:03d}/{max_epochs} | "
            f"train_total={train_parts['total']:.6f} "
            f"data={train_parts['data']:.6f} "
            f"csi={train_parts['soft_csi']:.6f} "
            f"event={train_parts['rain_event']:.6f} | "
            f"val_total={val_parts['total']:.6f} "
            f"data={val_parts['data']:.6f} "
            f"csi={val_parts['soft_csi']:.6f} "
            f"event={val_parts['rain_event']:.6f} | "
            f"lr={current_lr:.2e}"
        )

        if val_loss < best_val - 1e-7:
            best_val = val_loss
            best_epoch = epoch
            wait = 0
            torch.save(model.state_dict(), OUT_DIR / "best_model.pt")
        else:
            wait += 1

        if wait >= early_stopping_patience:
            print(
                f"Early stopping at epoch {epoch}. "
                f"Best epoch = {best_epoch}, best val = {best_val:.6f}"
            )
            break

    torch.save(model.state_dict(), OUT_DIR / "last_model.pt")
    plot_loss_curve(train_losses, val_losses, OUT_DIR)

    best_state = torch.load(OUT_DIR / "best_model.pt", map_location=device)
    model.load_state_dict(best_state, strict=True)

    print("\n开始测试集预测...")
    (
        pred_mm,
        true_mm,
        test_mask,
        test_year_labels,
        test_target_times,
    ) = predict_all(model, test_loader, scalers, geo_tensor)

    metrics = metrics_by_horizon(pred_mm, true_mm, test_mask)
    metrics["training_summary"] = {
        "best_epoch": int(best_epoch),
        "best_validation_loss": float(best_val),
    }
    metrics["loss_history_components"] = {
        "train_total": train_losses,
        "val_total": val_losses,
        "train_data": train_data_losses,
        "val_data": val_data_losses,
        "train_soft_csi": train_soft_csi_losses,
        "val_soft_csi": val_soft_csi_losses,
        "train_rain_event": train_rain_event_losses,
        "val_rain_event": val_rain_event_losses,
    }

    print("\nOverall continuous metrics:")
    print(metrics["overall"]["continuous"])
    print("\nBy-horizon continuous metrics:")
    for key, value in metrics["by_horizon"].items():
        print(key, value["continuous"])

    with open(OUT_DIR / "test_metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)

    np.savez_compressed(
        OUT_DIR / "test_predictions.npz",
        pred_test_mm=pred_mm,
        true_test_mm=true_mm,
        test_mask=test_mask,
        test_year_labels=test_year_labels,
        test_target_times=test_target_times,
        train_losses=np.array(train_losses),
        val_losses=np.array(val_losses),
        train_data_losses=np.array(train_data_losses),
        val_data_losses=np.array(val_data_losses),
        train_soft_csi_losses=np.array(train_soft_csi_losses),
        val_soft_csi_losses=np.array(val_soft_csi_losses),
        train_rain_event_losses=np.array(train_rain_event_losses),
        val_rain_event_losses=np.array(val_rain_event_losses),
        x_mean=scalers["x_mean"],
        x_std=scalers["x_std"],
        y_mean=scalers["y_mean"],
        y_std=scalers["y_std"],
        input_vars=np.array(train_years[0].feature_names),
        use_geo_embedding=np.array([USE_GEO_EMBEDDING]),
        geo_channels=np.array([geo_channels]),
        lat=test_years[0].lat,
        lon=test_years[0].lon,
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

    config = {
        "VARIABLE_DATA_DIR": str(VARIABLE_DATA_DIR),
        "VARIABLE_FILES": {key: str(value) for key, value in VARIABLE_FILES.items()},
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
        "best_epoch": int(best_epoch),
        "best_validation_loss": float(best_val),
    }
    with open(OUT_DIR / "run_config.json", "w", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=2)

    print("\n结果已保存到：")
    print(OUT_DIR)
    print(
        "包括：best_model.pt、last_model.pt、test_predictions.npz、"
        "test_metrics.json、run_config.json和预测图。"
    )


if __name__ == "__main__":
    main()
