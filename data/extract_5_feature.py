#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Hi-C 5维特征提取：
insulation_100kb, insulation_150kb, directionality_index, up_mean, down_mean

Scaler规则：
- GM12878：仅 chr1-chr17（排除 chr7、chr10）拟合一个全局 RobustScaler
- chr18-chr22 和其他细胞系只 transform
- 其他细胞系必须传入 GM12878 的 scaler_path
"""

import gc
import re
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")

FEATURE_COLS = [
    "insulation_100kb",
    "insulation_150kb",
    "directionality_index",
    "up_mean",
    "down_mean",
]

TRAIN_CHROMS = [f"chr{i}" for i in range(1, 18) if i not in (7, 10)]
SCALER_NAME = "gm12878_training_global_robust_scaler.npz"


def chrom_from_name(name: str) -> Optional[str]:
    m = re.search(r"chr(?:om)?([0-9]+|X|Y|M|MT)", str(name), re.I)
    if not m:
        return None
    chrom = m.group(1).upper()
    return "MT" if chrom == "M" else chrom


def chrom_sort_key(chrom: str):
    return (0, int(chrom)) if chrom.isdigit() else (1, {"X": 23, "Y": 24, "MT": 25}.get(chrom, 99))


def clean(values: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(np.asarray(values, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0).copy()
    x[x < 0] = 0.0
    return x


def load_matrix(path: str, use_memmap: bool = True) -> Optional[np.ndarray]:
    try:
        matrix = np.load(path, mmap_mode="r" if use_memmap else None, allow_pickle=False)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            print(f"✗ 非方阵: {path}")
            return None
        return matrix
    except Exception as e:
        print(f"✗ 加载失败 {path}: {e}")
        return None


def compute_insulation(
    matrix: np.ndarray,
    window_bins: int,
    mode: str = "mean",
    ignore_zero: bool = False,
) -> np.ndarray:
    n = matrix.shape[0]
    raw = np.full(n, np.nan)

    for i in range(window_bins, n - window_bins):
        x = clean(matrix[i - window_bins:i, i + 1:i + window_bins + 1])
        x = x[x > 0] if ignore_zero else x.ravel()
        if x.size == 0:
            continue

        if mode == "sum":
            raw[i] = x.sum()
        elif mode == "median":
            raw[i] = np.median(x)
        else:
            raw[i] = x.mean()

    valid = np.isfinite(raw)
    out = np.zeros(n)
    if not valid.any():
        return out

    chrom_mean = raw[valid].mean()
    if np.isfinite(chrom_mean) and chrom_mean > 0:
        out[valid] = np.log2((raw[valid] + 1e-10) / (chrom_mean + 1e-10))
    else:
        out[valid] = raw[valid] - chrom_mean

    return np.nan_to_num(out)


def compute_di(matrix: np.ndarray, window_bins: int) -> np.ndarray:
    n = matrix.shape[0]
    out = np.zeros(n)

    for i in range(window_bins, n - window_bins):
        up = clean(matrix[i, i - window_bins:i])
        down = clean(matrix[i, i + 1:i + window_bins + 1])
        A = up.mean() if up.size else 0.0
        B = down.mean() if down.size else 0.0
        out[i] = (B - A) / np.sqrt(A + B + 1e-10)

    return np.nan_to_num(out)


def compute_local_mean(matrix: np.ndarray, window_bins: int) -> Tuple[np.ndarray, np.ndarray]:
    n = matrix.shape[0]
    up_mean = np.zeros(n)
    down_mean = np.zeros(n)

    for i in range(n):
        up = clean(matrix[i, max(0, i - window_bins):i])
        down = clean(matrix[i, i + 1:min(n, i + window_bins + 1)])
        if up.size:
            up_mean[i] = up.mean()
        if down.size:
            down_mean[i] = down.mean()

    return np.nan_to_num(up_mean), np.nan_to_num(down_mean)


def extract_chromosome(
    npy_path: str,
    chrom: str,
    resolution: int,
    insulation_mode: str,
    insulation_ignore_zero: bool,
    use_memmap: bool,
) -> Optional[pd.DataFrame]:
    matrix = load_matrix(npy_path, use_memmap)
    if matrix is None:
        return None

    n = matrix.shape[0]
    starts = np.arange(n, dtype=np.int64) * resolution
    df = pd.DataFrame({
        "chrom": f"chr{chrom}",
        "start": starts,
        "end": starts + resolution,
    })

    df["insulation_100kb"] = compute_insulation(
        matrix, 100_000 // resolution, insulation_mode, insulation_ignore_zero
    )
    df["insulation_150kb"] = compute_insulation(
        matrix, 150_000 // resolution, insulation_mode, insulation_ignore_zero
    )
    df["directionality_index"] = compute_di(matrix, 200_000 // resolution)

    up_mean, down_mean = compute_local_mean(matrix, 150_000 // resolution)
    df["up_mean"] = up_mean
    df["down_mean"] = down_mean

    df[FEATURE_COLS] = (
        df[FEATURE_COLS]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype(np.float64)
    )

    del matrix
    gc.collect()

    print(f"✓ chr{chrom}: {n:,} bins")
    return df


def extract_raw_tables(
    input_dir: str,
    resolution: int,
    insulation_mode: str,
    insulation_ignore_zero: bool,
    use_memmap: bool,
) -> Dict[str, pd.DataFrame]:
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise ValueError(f"输入目录不存在: {input_dir}")

    files = []
    for path in input_dir.glob("*.npy"):
        chrom = chrom_from_name(path.name)
        if chrom is not None:
            files.append((path, chrom))

    files.sort(key=lambda x: chrom_sort_key(x[1]))
    if not files:
        raise ValueError(f"未找到染色体 .npy 文件: {input_dir}")

    tables = {}
    for path, chrom in files:
        df = extract_chromosome(
            str(path),
            chrom,
            resolution,
            insulation_mode,
            insulation_ignore_zero,
            use_memmap,
        )
        if df is not None:
            tables[f"chr{chrom}"] = df

    return tables


def feature_matrix(df: pd.DataFrame) -> np.ndarray:
    return df[FEATURE_COLS].to_numpy(dtype=np.float64)


def fit_gm12878_scaler(raw_tables: Dict[str, pd.DataFrame]) -> RobustScaler:
    missing = [chrom for chrom in TRAIN_CHROMS if chrom not in raw_tables]
    if missing:
        raise ValueError(f"缺少 GM12878 training chromosomes: {missing}")

    train_x = np.concatenate(
        [feature_matrix(raw_tables[chrom]) for chrom in TRAIN_CHROMS],
        axis=0,
    )

    scaler = RobustScaler()
    scaler.fit(train_x)  # 唯一 fit
    return scaler


def save_scaler(scaler: RobustScaler, path: Union[str, Path]) -> None:
    np.savez_compressed(
        path,
        feature_names=np.array(FEATURE_COLS),
        center=scaler.center_,
        scale=scaler.scale_,
    )


def load_scaler(path: Union[str, Path]) -> RobustScaler:
    with np.load(path, allow_pickle=False) as data:
        if list(data["feature_names"]) != FEATURE_COLS:
            raise ValueError("scaler 特征顺序与当前代码不一致")

        scaler = RobustScaler()
        scaler.center_ = data["center"].astype(np.float64)
        scaler.scale_ = data["scale"].astype(np.float64)
        scaler.n_features_in_ = len(FEATURE_COLS)

    return scaler


def transform_tables(
    raw_tables: Dict[str, pd.DataFrame],
    scaler: RobustScaler,
) -> Dict[str, pd.DataFrame]:
    scaled = {}

    for chrom, df in raw_tables.items():
        out = df.copy()
        out.loc[:, FEATURE_COLS] = scaler.transform(feature_matrix(df))
        scaled[chrom] = out

    return scaled


def extract_all_chromosomes(
    input_dir: str,
    feature_dir: Union[str, Path],
    resolution: int,
    insulation_mode: str = "mean",
    insulation_ignore_zero: bool = False,
    use_memmap: bool = True,
    scaler_path: Optional[Union[str, Path]] = None,
) -> Dict[str, pd.DataFrame]:

    feature_dir = Path(feature_dir)
    feature_dir.mkdir(parents=True, exist_ok=True)

    raw_tables = extract_raw_tables(
        input_dir,
        resolution,
        insulation_mode,
        insulation_ignore_zero,
        use_memmap,
    )

    if "gm12878" in str(input_dir).lower():
        scaler = fit_gm12878_scaler(raw_tables)
        scaler_path = Path(scaler_path) if scaler_path else feature_dir / SCALER_NAME
        save_scaler(scaler, scaler_path)
        print(f"✓ scaler 已保存: {scaler_path}")
    else:
        if scaler_path is None:
            raise ValueError("其他细胞系必须提供 GM12878 的 scaler_path，禁止重新 fit。")
        scaler = load_scaler(scaler_path)

    scaled_tables = transform_tables(raw_tables, scaler)

    for chrom, df in scaled_tables.items():
        df.to_csv(feature_dir / f"{chrom}_hic_features.csv", index=False)

    print(f"✓ 完成: {len(scaled_tables)} 条染色体 -> {feature_dir}")
    return scaled_tables
