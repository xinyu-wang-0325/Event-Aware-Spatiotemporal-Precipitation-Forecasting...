# -*- coding: utf-8 -*-
"""绘制单个 ERA5 时刻的柱水汽收支组成及其与降水的空间对比。

脚本采用与 ``geoconvlstm_physics_finetune_8h.py`` 一致的定义：

    P_budget = E_up - ΔTCWV - VIMD

其中，ERA5 蒸发量转换为向上为正，TCWV 的相邻小时差值数值等价于
毫米水深，VIMD 按配置解释为每小时累积量或瞬时通量率。

输出包括水汽收支组成图、降水与非负 P_budget 的对比图，以及记录配置、
统计量和空间指标的 JSON 摘要。默认从项目 ``data`` 目录读取 NetCDF 文件，
并将结果写入 ``outputs/water_budget_example``。
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
import xarray as xr


# =========================================================
# 1. 配置
# =========================================================

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"

TP_PATH = DATA_DIR / "tp_hourly.nc"
TCWV_PATH = DATA_DIR / "tcwv.nc"
VIMD_PATH = DATA_DIR / "vimd.nc"
EVAP_PATH = DATA_DIR / "evaporation.nc"

OUT_DIR = PROJECT_DIR / "outputs" / "water_budget_example"

SELECT_YEARS = [2020, 2021, 2022, 2023]
SELECT_MONTHS = [4, 5, 6, 7, 8, 9]

RANDOM_SEED = 42

# "rain_event": 在随机候选中优先选择存在明显降水的一小时，避免整幅图几乎全为0。
# "all": 从所有共同时间中完全均匀随机选择。
SELECTION_MODE = "rain_event"
MIN_TP_MAX_MM = 4.0
CANDIDATE_POOL_SIZE = 512

# 与 Physics 主体代码保持一致。
TP_INPUT_UNITS = "m_per_hour"
EVAP_INPUT_MODE = "hourly_amount_m"
EVAP_SIGN_MODE = "era5_negative"

# 当前 VIMD 文件按每小时时间间隔的水量存放，因此不再乘以时间步长。
# 可选："none" 或 "multiply_dt"
VIMD_TIME_INTEGRATION_MODE = "none"
VIMD_CONVENTION = "divergence"
EXPECTED_DT_SECONDS = 3600.0
DT_TOLERANCE_SECONDS = 1.0

# Physics 主体代码会用训练集估计 alpha，使非负 P_budget 与训练集平均降水量匹配。
# 默认从 Physics 的 run_config.json 读取该系数；如需独立运行，可在此显式指定。
PHYSICS_RUN_CONFIG_PATH = (
    PROJECT_DIR
    / "outputs"
    / "geoconvlstm_physics_finetune_8h"
    / "run_config.json"
)
PBUDGET_ALPHA_OVERRIDE: float | None = None
PBUDGET_ALPHA_FALLBACK = 1.0

RAIN_THRESHOLDS_MM = [0.1, 1.0, 4.0, 8.0]

FIG_DPI = 180
ROBUST_PERCENTILE = 99.5


# =========================================================
# 2. 通用数据工具
# =========================================================

def validate_configuration() -> None:
    """在读取大体量数据前检查可直接发现的配置错误。"""
    if not SELECT_YEARS or not SELECT_MONTHS:
        raise ValueError("SELECT_YEARS 和 SELECT_MONTHS 不能为空。")
    if any(month < 1 or month > 12 for month in SELECT_MONTHS):
        raise ValueError("SELECT_MONTHS 只能包含 1 至 12。")
    if SELECTION_MODE not in {"all", "rain_event"}:
        raise ValueError("SELECTION_MODE 必须为 'all' 或 'rain_event'。")
    if CANDIDATE_POOL_SIZE <= 0:
        raise ValueError("CANDIDATE_POOL_SIZE 必须为正整数。")
    if MIN_TP_MAX_MM < 0.0:
        raise ValueError("MIN_TP_MAX_MM 不能为负数。")
    if EXPECTED_DT_SECONDS <= 0.0 or DT_TOLERANCE_SECONDS < 0.0:
        raise ValueError("时间间隔及其容差配置无效。")
    if TP_INPUT_UNITS not in {"m_per_hour", "mm_per_hour"}:
        raise ValueError("TP_INPUT_UNITS 配置无效。")
    if EVAP_INPUT_MODE not in {"hourly_amount_m", "hourly_amount_mm"}:
        raise ValueError("EVAP_INPUT_MODE 配置无效。")
    if EVAP_SIGN_MODE not in {
        "era5_negative",
        "positive_upward",
        "auto",
    }:
        raise ValueError("EVAP_SIGN_MODE 配置无效。")
    if VIMD_TIME_INTEGRATION_MODE not in {"none", "multiply_dt"}:
        raise ValueError("VIMD_TIME_INTEGRATION_MODE 配置无效。")
    if VIMD_CONVENTION not in {"divergence", "convergence"}:
        raise ValueError("VIMD_CONVENTION 配置无效。")
    if not 0.0 < ROBUST_PERCENTILE <= 100.0:
        raise ValueError("ROBUST_PERCENTILE 必须位于 (0, 100]。")
    if not RAIN_THRESHOLDS_MM or any(
        threshold < 0.0 for threshold in RAIN_THRESHOLDS_MM
    ):
        raise ValueError("RAIN_THRESHOLDS_MM 必须包含非负阈值。")


def open_nc(path: Path) -> xr.Dataset:
    if not path.exists():
        raise FileNotFoundError(f"找不到文件: {path}")

    try:
        return xr.open_dataset(path, engine="h5netcdf")
    except (ImportError, OSError, ValueError):
        return xr.open_dataset(path)


def path_for_summary(path: Path) -> str:
    """项目内路径写为相对路径，项目外路径保留绝对形式。"""
    resolved_path = path.resolve()
    try:
        return str(resolved_path.relative_to(PROJECT_DIR.resolve()))
    except ValueError:
        return str(resolved_path)


def load_pbudget_alpha() -> tuple[float, str]:
    """读取 Physics 训练得到的 P_budget 校准系数。"""
    if PBUDGET_ALPHA_OVERRIDE is not None:
        value = PBUDGET_ALPHA_OVERRIDE
        source = "PBUDGET_ALPHA_OVERRIDE"
    elif PHYSICS_RUN_CONFIG_PATH.exists():
        try:
            with PHYSICS_RUN_CONFIG_PATH.open("r", encoding="utf-8") as file:
                config = json.load(file)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"无法读取 Physics 配置文件: {PHYSICS_RUN_CONFIG_PATH}"
            ) from exc

        if "pbudget_alpha" not in config:
            raise KeyError(
                "Physics 配置文件中缺少 'pbudget_alpha': "
                f"{PHYSICS_RUN_CONFIG_PATH}"
            )
        value = config["pbudget_alpha"]
        source = path_for_summary(PHYSICS_RUN_CONFIG_PATH)
    else:
        value = PBUDGET_ALPHA_FALLBACK
        source = "PBUDGET_ALPHA_FALLBACK"
        warnings.warn(
            "未找到 Physics 的 run_config.json，使用未校准回退值 "
            f"alpha={PBUDGET_ALPHA_FALLBACK:.6g}。"
        )

    try:
        alpha = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"P_budget 校准系数不是有效数值: {value!r}") from exc

    if not np.isfinite(alpha) or alpha <= 0.0:
        raise ValueError(f"P_budget 校准系数必须为有限正数，当前为 {alpha!r}。")

    return alpha, source


def find_time_name(ds: xr.Dataset) -> str:
    for name in ["valid_time", "time"]:
        if name in ds.coords or name in ds.dims:
            return name
    raise ValueError(
        f"无法识别时间坐标。当前坐标={list(ds.coords)}, dims={list(ds.dims)}"
    )


def find_lat_lon_names(ds: xr.Dataset) -> tuple[str, str]:
    lat_name = next(
        (
            name
            for name in ["latitude", "lat"]
            if name in ds.coords or name in ds.dims
        ),
        None,
    )
    lon_name = next(
        (
            name
            for name in ["longitude", "lon"]
            if name in ds.coords or name in ds.dims
        ),
        None,
    )

    if lat_name is None or lon_name is None:
        raise ValueError(
            "无法识别经纬度坐标。"
            f"coords={list(ds.coords)}, dims={list(ds.dims)}"
        )

    return lat_name, lon_name


def find_first_existing_var(
    ds: xr.Dataset,
    candidates: list[str],
    label: str,
) -> str:
    lower_to_real = {name.lower(): name for name in ds.data_vars}

    for candidate in candidates:
        if candidate in ds.data_vars:
            return candidate
        if candidate.lower() in lower_to_real:
            return lower_to_real[candidate.lower()]

    raise ValueError(
        f"{label} 文件中找不到变量。"
        f"候选={candidates}, 当前变量={list(ds.data_vars)}"
    )


def collapse_extra_dims(da: xr.DataArray, label: str) -> xr.DataArray:
    """处理单例维和ERA5可能出现的expver维。"""
    for dim in list(da.dims):
        if dim in {"time", "lat", "lon"}:
            continue

        if da.sizes[dim] == 1:
            da = da.isel({dim: 0}, drop=True)
        elif dim.lower() == "expver":
            merged = da.isel({dim: 0}, drop=True)
            for idx in range(1, da.sizes[dim]):
                merged = merged.combine_first(
                    da.isel({dim: idx}, drop=True)
                )
            da = merged
        else:
            raise ValueError(
                f"{label} 存在无法自动处理的额外维度 "
                f"{dim}={da.sizes[dim]}"
            )

    return da


def standardize_da(
    ds: xr.Dataset,
    candidates: list[str],
    label: str,
) -> xr.DataArray:
    var_name = find_first_existing_var(ds, candidates, label)
    time_name = find_time_name(ds)
    lat_name, lon_name = find_lat_lon_names(ds)

    rename_dict = {}
    if time_name != "time":
        rename_dict[time_name] = "time"
    if lat_name != "lat":
        rename_dict[lat_name] = "lat"
    if lon_name != "lon":
        rename_dict[lon_name] = "lon"

    da = ds[var_name].rename(rename_dict)
    da = collapse_extra_dims(da, label)
    da = da.transpose("time", "lat", "lon")
    da = da.sortby("time")

    # 去除重复时间。
    times = pd.DatetimeIndex(pd.to_datetime(da["time"].values))
    _, unique_indices = np.unique(times.values, return_index=True)
    da = da.isel(time=np.sort(unique_indices))

    print(
        f"{label}: variable={var_name}, shape={da.shape}, "
        f"units={da.attrs.get('units', 'unknown')}"
    )
    return da


def convert_tp_to_mm(tp: xr.DataArray) -> xr.DataArray:
    if TP_INPUT_UNITS == "m_per_hour":
        out = tp * 1000.0
    elif TP_INPUT_UNITS == "mm_per_hour":
        out = tp
    else:
        raise ValueError(f"不支持 TP_INPUT_UNITS={TP_INPUT_UNITS}")

    out = out.where(np.isfinite(out))
    out = xr.where(out >= 0.0, out, 0.0)
    out.attrs["units"] = "mm h-1"
    return out.rename("tp_hourly_mm")


def convert_evap_to_upward_mm(evap: xr.DataArray) -> xr.DataArray:
    if EVAP_INPUT_MODE == "hourly_amount_m":
        evap_mm = evap * 1000.0
    elif EVAP_INPUT_MODE == "hourly_amount_mm":
        evap_mm = evap
    else:
        raise ValueError(
            f"不支持 EVAP_INPUT_MODE={EVAP_INPUT_MODE}"
        )

    if EVAP_SIGN_MODE == "era5_negative":
        upward = -evap_mm
    elif EVAP_SIGN_MODE == "positive_upward":
        upward = evap_mm
    elif EVAP_SIGN_MODE == "auto":
        mean_value = float(evap_mm.mean(skipna=True).values)
        upward = -evap_mm if mean_value < 0.0 else evap_mm
    else:
        raise ValueError(
            f"不支持 EVAP_SIGN_MODE={EVAP_SIGN_MODE}"
        )

    upward.attrs["units"] = "mm h-1"
    return upward.rename("evaporation_upward_mm")


def integrate_vimd(
    vimd: xr.DataArray,
    dt_seconds: float,
) -> xr.DataArray:
    if VIMD_TIME_INTEGRATION_MODE == "none":
        out = vimd
    elif VIMD_TIME_INTEGRATION_MODE == "multiply_dt":
        out = vimd * float(dt_seconds)
    else:
        raise ValueError(
            "VIMD_TIME_INTEGRATION_MODE必须为 "
            "'none' 或 'multiply_dt'"
        )

    out.attrs["units"] = "mm h-1 equivalent"
    return out.rename("vimd_term_mm")


def common_valid_times(
    arrays: list[xr.DataArray],
    tcwv_times: pd.DatetimeIndex,
) -> pd.DatetimeIndex:
    common = pd.DatetimeIndex(
        pd.to_datetime(arrays[0]["time"].values)
    )

    for da in arrays[1:]:
        current = pd.DatetimeIndex(
            pd.to_datetime(da["time"].values)
        )
        common = common.intersection(current)

    common = common.sort_values()

    keep = (
        np.isin(common.year, SELECT_YEARS)
        & np.isin(common.month, SELECT_MONTHS)
    )
    common = common[keep]

    # ΔTCWV需要前一小时存在。
    tcwv_set = set(tcwv_times.values)
    valid = [
        timestamp
        for timestamp in common
        if (
            timestamp - pd.to_timedelta(EXPECTED_DT_SECONDS, unit="s")
        ).to_datetime64() in tcwv_set
    ]

    result = pd.DatetimeIndex(valid)
    if len(result) == 0:
        raise ValueError(
            "筛选后没有同时具备tp、tcwv、vimd、evaporation，"
            "且具有前一小时TCWV的数据时刻。"
        )

    return result


def choose_random_time(
    tp_da: xr.DataArray,
    valid_times: pd.DatetimeIndex,
) -> pd.Timestamp:
    rng = np.random.default_rng(RANDOM_SEED)

    if SELECTION_MODE == "all":
        return pd.Timestamp(rng.choice(valid_times.values))

    if SELECTION_MODE != "rain_event":
        raise ValueError(
            "SELECTION_MODE必须为 'all' 或 'rain_event'"
        )

    pool_size = min(CANDIDATE_POOL_SIZE, len(valid_times))
    candidate_indices = rng.choice(
        len(valid_times),
        size=pool_size,
        replace=False,
    )
    candidate_times = valid_times[candidate_indices]

    # 一次只读取候选时刻，避免扫描完整四年数据。
    candidate_tp = convert_tp_to_mm(
        tp_da.sel(time=candidate_times.values)
    )
    spatial_dims = [
        dim for dim in candidate_tp.dims if dim != "time"
    ]
    max_by_time = candidate_tp.max(
        dim=spatial_dims,
        skipna=True,
    ).load().values

    eligible = np.where(max_by_time >= MIN_TP_MAX_MM)[0]

    if eligible.size > 0:
        selected_local = int(rng.choice(eligible))
        selected = pd.Timestamp(
            candidate_times[selected_local]
        )
        print(
            f"从{pool_size}个随机候选中选到降水时刻："
            f"max(tp)={max_by_time[selected_local]:.3f} mm/h"
        )
        return selected

    finite_indices = np.flatnonzero(np.isfinite(max_by_time))
    if finite_indices.size == 0:
        raise ValueError("随机候选时刻的降水场均不包含有限数值。")

    fallback = int(
        finite_indices[np.argmax(max_by_time[finite_indices])]
    )
    selected = pd.Timestamp(candidate_times[fallback])
    warnings.warn(
        f"随机候选中没有 max(tp)>={MIN_TP_MAX_MM} mm/h 的时刻，"
        f"改用候选中降水最大的时刻，"
        f"max(tp)={max_by_time[fallback]:.3f} mm/h。"
    )
    return selected


# =========================================================
# 3. 统计和绘图工具
# =========================================================

def finite_values(da: xr.DataArray) -> np.ndarray:
    values = np.asarray(da.values, dtype=np.float64).reshape(-1)
    return values[np.isfinite(values)]


def field_stats(da: xr.DataArray) -> dict:
    values = finite_values(da)
    if values.size == 0:
        return {
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "p01": None,
            "p50": None,
            "p99": None,
            "n": 0,
        }

    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "p01": float(np.percentile(values, 1.0)),
        "p50": float(np.percentile(values, 50.0)),
        "p99": float(np.percentile(values, 99.0)),
        "n": int(values.size),
    }


def pair_metrics(
    prediction: xr.DataArray,
    truth: xr.DataArray,
) -> dict:
    pred, true = xr.align(prediction, truth, join="inner")
    p = np.asarray(pred.values, dtype=np.float64).reshape(-1)
    t = np.asarray(true.values, dtype=np.float64).reshape(-1)

    valid = np.isfinite(p) & np.isfinite(t)
    p = p[valid]
    t = t[valid]

    if p.size == 0:
        raise ValueError("tp与P_budget没有有效共同格点。")

    if np.std(p) < 1e-12 or np.std(t) < 1e-12:
        corr = None
    else:
        corr = float(np.corrcoef(p, t)[0, 1])

    metrics = {
        "MAE": float(np.mean(np.abs(p - t))),
        "RMSE": float(np.sqrt(np.mean((p - t) ** 2))),
        "Bias_pred_minus_true": float(np.mean(p - t)),
        "Corr": corr,
        "N": int(p.size),
        "categorical": {},
    }

    for threshold in RAIN_THRESHOLDS_MM:
        pred_event = p >= threshold
        true_event = t >= threshold

        hit = int(np.sum(pred_event & true_event))
        miss = int(np.sum((~pred_event) & true_event))
        false_alarm = int(np.sum(pred_event & (~true_event)))

        denom = hit + miss + false_alarm
        csi = hit / denom if denom > 0 else np.nan

        metrics["categorical"][str(threshold)] = {
            "H": hit,
            "M": miss,
            "F": false_alarm,
            "CSI": float(csi) if np.isfinite(csi) else None,
        }

    return metrics


def robust_upper(
    arrays: list[xr.DataArray],
    percentile: float = ROBUST_PERCENTILE,
    minimum: float = 1e-6,
) -> float:
    pieces = [finite_values(array) for array in arrays]
    pieces = [piece for piece in pieces if piece.size > 0]

    if not pieces:
        return 1.0

    values = np.concatenate(pieces)
    upper = float(np.percentile(values, percentile))
    return max(upper, minimum)


def robust_abs_bound(
    da: xr.DataArray,
    percentile: float = ROBUST_PERCENTILE,
    minimum: float = 1e-6,
) -> float:
    values = finite_values(da)
    if values.size == 0:
        return 1.0

    bound = float(np.percentile(np.abs(values), percentile))
    return max(bound, minimum)


def plot_map(
    ax: plt.Axes,
    da: xr.DataArray,
    title: str,
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
    centered: bool = False,
) -> None:
    lon = np.asarray(da["lon"].values)
    lat = np.asarray(da["lat"].values)
    values = np.asarray(da.values)

    kwargs = {
        "shading": "auto",
        "cmap": cmap,
    }

    if centered:
        bound = robust_abs_bound(da) if vmax is None else abs(vmax)
        kwargs["norm"] = TwoSlopeNorm(
            vmin=-bound,
            vcenter=0.0,
            vmax=bound,
        )
    else:
        kwargs["vmin"] = vmin
        kwargs["vmax"] = vmax

    image = ax.pcolormesh(lon, lat, values, **kwargs)
    ax.set_title(title)
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_xlim(float(np.min(lon)), float(np.max(lon)))
    ax.set_ylim(float(np.min(lat)), float(np.max(lat)))
    ax.grid(alpha=0.15, linewidth=0.5)

    plt.colorbar(
        image,
        ax=ax,
        fraction=0.046,
        pad=0.035,
    )


# =========================================================
# 4. 主程序
# =========================================================

def main() -> None:
    validate_configuration()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pbudget_alpha, pbudget_alpha_source = load_pbudget_alpha()

    datasets: list[xr.Dataset] = []

    try:
        ds_tp = open_nc(TP_PATH)
        ds_tcwv = open_nc(TCWV_PATH)
        ds_vimd = open_nc(VIMD_PATH)
        ds_evap = open_nc(EVAP_PATH)
        datasets.extend([ds_tp, ds_tcwv, ds_vimd, ds_evap])

        tp = standardize_da(
            ds_tp,
            ["tp_hourly", "tp", "total_precipitation"],
            "tp_hourly",
        )
        tcwv = standardize_da(
            ds_tcwv,
            [
                "tcwv",
                "total_column_water_vapour",
                "total_column_water_vapor",
            ],
            "tcwv",
        )
        vimd = standardize_da(
            ds_vimd,
            [
                "vimd",
                "vimdf",
                "vertically_integrated_moisture_divergence",
                "vertical_integral_of_divergence_of_moisture_flux",
                "vertical_integral_of_moisture_divergence",
            ],
            "vimd",
        )
        evaporation = standardize_da(
            ds_evap,
            ["e", "evaporation"],
            "evaporation",
        )

        tcwv_times = pd.DatetimeIndex(
            pd.to_datetime(tcwv["time"].values)
        )
        valid_times = common_valid_times(
            [tp, tcwv, vimd, evaporation],
            tcwv_times=tcwv_times,
        )

        print(
            f"有效共同时间数量={len(valid_times)}, "
            f"范围={valid_times[0]} 至 {valid_times[-1]}"
        )

        selected_time = choose_random_time(tp, valid_times)
        previous_time = selected_time - pd.to_timedelta(
            EXPECTED_DT_SECONDS,
            unit="s",
        )

        print(f"\nSelected time: {selected_time}")
        print(f"Previous TCWV time: {previous_time}")

        tp_t = convert_tp_to_mm(
            tp.sel(time=selected_time.to_datetime64())
        )
        tcwv_t = tcwv.sel(
            time=selected_time.to_datetime64()
        )
        tcwv_previous = tcwv.sel(
            time=previous_time.to_datetime64()
        )
        vimd_t = vimd.sel(
            time=selected_time.to_datetime64()
        )
        evaporation_t = evaporation.sel(
            time=selected_time.to_datetime64()
        )

        (
            tp_t,
            tcwv_t,
            tcwv_previous,
            vimd_t,
            evaporation_t,
        ) = xr.align(
            tp_t,
            tcwv_t,
            tcwv_previous,
            vimd_t,
            evaporation_t,
            join="inner",
            copy=False,
        )

        if tp_t.sizes["lat"] == 0 or tp_t.sizes["lon"] == 0:
            raise ValueError("变量空间网格严格对齐后为空。")

        # 载入所选一个小时，后续关闭NC文件也不会影响绘图。
        tp_t = tp_t.load()
        tcwv_t = tcwv_t.load()
        tcwv_previous = tcwv_previous.load()
        vimd_t = vimd_t.load()
        evaporation_t = evaporation_t.load()

        actual_dt_seconds = float(
            (
                selected_time - previous_time
            ).total_seconds()
        )
        if abs(
            actual_dt_seconds - EXPECTED_DT_SECONDS
        ) > DT_TOLERANCE_SECONDS:
            warnings.warn(
                f"实际时间间隔={actual_dt_seconds}s，"
                f"预期={EXPECTED_DT_SECONDS}s。"
            )

        delta_tcwv = (
            tcwv_t - tcwv_previous
        ).rename("delta_tcwv_mm")
        delta_tcwv.attrs["units"] = "mm h-1 equivalent"

        evaporation_upward = convert_evap_to_upward_mm(
            evaporation_t
        )
        vimd_term = integrate_vimd(
            vimd_t,
            dt_seconds=actual_dt_seconds,
        )

        if VIMD_CONVENTION == "divergence":
            pbudget_raw = (
                evaporation_upward
                - delta_tcwv
                - vimd_term
            )
        elif VIMD_CONVENTION == "convergence":
            pbudget_raw = (
                evaporation_upward
                - delta_tcwv
                + vimd_term
            )
        else:
            raise ValueError(
                "VIMD_CONVENTION必须为 "
                "'divergence' 或 'convergence'"
            )

        pbudget_raw = pbudget_raw.rename("p_budget_raw")
        pbudget_raw.attrs["units"] = "mm h-1"

        pbudget_positive = xr.where(
            pbudget_raw > 0.0,
            pbudget_raw,
            0.0,
        ).rename("p_budget_positive")

        pbudget_calibrated = (
            pbudget_alpha * pbudget_positive
        ).rename("p_budget_positive_calibrated")
        pbudget_calibrated.attrs["units"] = "mm h-1"

        comparison_difference = (
            pbudget_calibrated - tp_t
        ).rename("pbudget_minus_tp")

        # -----------------------------
        # 图1：方程各组成项
        # -----------------------------
        timestamp_text = selected_time.strftime(
            "%Y-%m-%d %H:%M UTC"
        )
        timestamp_file = selected_time.strftime(
            "%Y%m%d_%H00"
        )

        fig, axes = plt.subplots(
            2,
            3,
            figsize=(18, 10),
            constrained_layout=True,
        )

        tp_upper = robust_upper([tp_t])
        plot_map(
            axes[0, 0],
            tp_t,
            "tp_hourly (mm h$^{-1}$)",
            cmap="Blues",
            vmin=0.0,
            vmax=tp_upper,
        )

        tcwv_values = finite_values(tcwv_t)
        if tcwv_values.size == 0:
            raise ValueError("所选时刻的 TCWV 场不包含有限数值。")
        tcwv_min = float(np.percentile(tcwv_values, 0.5))
        tcwv_max = float(np.percentile(tcwv_values, 99.5))
        if tcwv_max <= tcwv_min:
            tcwv_max = tcwv_min + max(abs(tcwv_min) * 1e-6, 1e-6)
        plot_map(
            axes[0, 1],
            tcwv_t,
            "TCWV (kg m$^{-2}$)",
            cmap="viridis",
            vmin=tcwv_min,
            vmax=tcwv_max,
        )

        plot_map(
            axes[0, 2],
            delta_tcwv,
            "$\\Delta$TCWV (mm h$^{-1}$ equiv.)",
            cmap="RdBu_r",
            centered=True,
        )

        plot_map(
            axes[1, 0],
            vimd_term,
            "VIMD term (mm h$^{-1}$ equiv.)",
            cmap="RdBu_r",
            centered=True,
        )

        plot_map(
            axes[1, 1],
            evaporation_upward,
            "Evaporation, upward positive (mm h$^{-1}$)",
            cmap="RdBu_r",
            centered=True,
        )

        plot_map(
            axes[1, 2],
            pbudget_raw,
            "Raw P_budget (mm h$^{-1}$)",
            cmap="RdBu_r",
            centered=True,
        )

        fig.suptitle(
            f"ERA5 moisture-budget components — {timestamp_text}",
            fontsize=15,
        )

        components_path = (
            OUT_DIR / f"components_{timestamp_file}.png"
        )
        fig.savefig(
            components_path,
            dpi=FIG_DPI,
            bbox_inches="tight",
        )
        plt.close(fig)

        # -----------------------------
        # 图2：tp与物理约束直接比较
        # -----------------------------
        shared_upper = robust_upper(
            [tp_t, pbudget_calibrated]
        )
        difference_bound = robust_abs_bound(
            comparison_difference
        )

        fig, axes = plt.subplots(
            1,
            3,
            figsize=(18, 5.3),
            constrained_layout=True,
        )

        plot_map(
            axes[0],
            tp_t,
            "Observed tp_hourly",
            cmap="Blues",
            vmin=0.0,
            vmax=shared_upper,
        )
        plot_map(
            axes[1],
            pbudget_calibrated,
            (
                "Calibrated positive P_budget\n"
                f"$\\alpha$={pbudget_alpha:.4f}"
            ),
            cmap="Blues",
            vmin=0.0,
            vmax=shared_upper,
        )
        plot_map(
            axes[2],
            comparison_difference,
            "Calibrated P_budget - tp_hourly",
            cmap="RdBu_r",
            vmax=difference_bound,
            centered=True,
        )

        fig.suptitle(
            (
                "Direct spatial comparison with a shared precipitation "
                f"color scale — {timestamp_text}"
            ),
            fontsize=14,
        )

        comparison_path = (
            OUT_DIR / f"tp_vs_pbudget_{timestamp_file}.png"
        )
        fig.savefig(
            comparison_path,
            dpi=FIG_DPI,
            bbox_inches="tight",
        )
        plt.close(fig)

        # -----------------------------
        # 统计摘要
        # -----------------------------
        summary = {
            "selected_time": selected_time.isoformat(),
            "previous_tcwv_time": previous_time.isoformat(),
            "selection_mode": SELECTION_MODE,
            "random_seed": RANDOM_SEED,
            "paths": {
                "tp_hourly": path_for_summary(TP_PATH),
                "tcwv": path_for_summary(TCWV_PATH),
                "vimd": path_for_summary(VIMD_PATH),
                "evaporation": path_for_summary(EVAP_PATH),
            },
            "configuration": {
                "TP_INPUT_UNITS": TP_INPUT_UNITS,
                "EVAP_INPUT_MODE": EVAP_INPUT_MODE,
                "EVAP_SIGN_MODE": EVAP_SIGN_MODE,
                "VIMD_TIME_INTEGRATION_MODE":
                    VIMD_TIME_INTEGRATION_MODE,
                "VIMD_CONVENTION": VIMD_CONVENTION,
                "pbudget_alpha": pbudget_alpha,
                "pbudget_alpha_source": pbudget_alpha_source,
                "actual_dt_seconds": actual_dt_seconds,
            },
            "field_statistics": {
                "tp_hourly_mm": field_stats(tp_t),
                "tcwv": field_stats(tcwv_t),
                "delta_tcwv_mm": field_stats(delta_tcwv),
                "vimd_term_mm": field_stats(vimd_term),
                "evaporation_upward_mm":
                    field_stats(evaporation_upward),
                "pbudget_raw_mm": field_stats(pbudget_raw),
                "pbudget_positive_calibrated_mm":
                    field_stats(pbudget_calibrated),
            },
            "raw_pbudget_vs_tp": pair_metrics(
                pbudget_raw,
                tp_t,
            ),
            "calibrated_positive_pbudget_vs_tp": pair_metrics(
                pbudget_calibrated,
                tp_t,
            ),
            "outputs": {
                "components_figure": path_for_summary(components_path),
                "comparison_figure": path_for_summary(comparison_path),
            },
        }

        summary_path = (
            OUT_DIR / f"summary_{timestamp_file}.json"
        )
        with summary_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                summary,
                file,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )

        print("\n=== Selected-hour comparison ===")
        print(
            json.dumps(
                summary[
                    "calibrated_positive_pbudget_vs_tp"
                ],
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
        )
        print("\nSaved:")
        print(f"  {components_path}")
        print(f"  {comparison_path}")
        print(f"  {summary_path}")

    finally:
        for dataset in reversed(datasets):
            dataset.close()


if __name__ == "__main__":
    main()
