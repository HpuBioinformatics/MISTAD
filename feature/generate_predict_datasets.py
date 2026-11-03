#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
无标签滑动窗口预测样本生成模块
================================

输入：feature_extraction 输出的标准化特征 CSV
      （chrN_hic_features.csv，列含 chrom/start/end + 8 维特征）
输出：chrN_ws{window_size}.npz，字段为
      features (N, window_size, 8) / chroms (N,) / starts (N,) / ends (N,)
"""

import gc
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# 与 data/extract_5_feature.py 的 FEATURE_COLS 保持一致
FEATURE_COLS = [
    "insulation_100kb",
    "insulation_150kb",
    "directionality_index",
    "up_mean",
    "down_mean",
]


def load_feature_csvs(feature_dir: Path) -> Dict[str, pd.DataFrame]:
    """读取 feature_dir 下全部 chr*_hic_features.csv。"""
    feat_files = sorted(feature_dir.glob("chr*_hic_features.csv"))
    if not feat_files:
        raise FileNotFoundError(f"未找到特征 CSV: {feature_dir}/chr*_hic_features.csv")

    required_cols = ["chrom", "start", "end", *FEATURE_COLS]
    tables: Dict[str, pd.DataFrame] = {}

    for feat_path in feat_files:
        df = pd.read_csv(feat_path)
        if any(col not in df.columns for col in required_cols) or df.empty:
            print(f"  ✗ 跳过异常文件: {feat_path.name}")
            continue
        chrom_name = str(df["chrom"].iloc[0])
        tables[chrom_name] = df
        print(f"  ✓ 读取 {feat_path.name}: {len(df):,} bins")

    if not tables:
        raise RuntimeError("没有成功读取任何染色体特征 CSV")
    return tables


def generate_sliding_windows(
    feature_arr: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    chrom_name: str,
    window_size: int,
) -> Optional[Dict[str, np.ndarray]]:
    """生成无标签滑动窗口预测样本（中心 bin 坐标随窗口一起输出）。"""
    if window_size <= 0 or window_size % 2 == 0:
        raise ValueError(f"window_size 必须是正奇数，收到 {window_size}")

    n_bins, n_features = feature_arr.shape
    if n_bins < window_size:
        print(f"    ✗ bin 数量 {n_bins} 小于窗口大小 {window_size}，跳过")
        return None

    radius = window_size // 2
    n_samples = n_bins - 2 * radius

    windows = np.empty((n_samples, window_size, n_features), dtype=np.float32)
    center_starts = np.empty(n_samples, dtype=np.int64)
    center_ends = np.empty(n_samples, dtype=np.int64)

    for sample_idx, center_idx in enumerate(range(radius, n_bins - radius)):
        windows[sample_idx] = feature_arr[center_idx - radius:center_idx + radius + 1]
        center_starts[sample_idx] = starts[center_idx]
        center_ends[sample_idx] = ends[center_idx]

    chroms = np.full(n_samples, chrom_name, dtype=f"U{max(1, len(chrom_name))}")

    return {
        "features": windows,
        "chroms": chroms,
        "starts": center_starts,
        "ends": center_ends,
    }


def generate_all_samples(
    scaled_tables: Dict[str, pd.DataFrame],
    sample_base: Path,
    resolution: int,
    cellline: str,
    window_sizes: List[int],
) -> None:
    """对每个 window_size，逐染色体生成滑动窗口样本并保存 npz。"""
    res_kb = resolution // 1000

    for window_size in window_sizes:
        out_dir = sample_base / f"ws{window_size}_{res_kb}kb_{cellline}"
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[样本生成] ws{window_size} -> {out_dir}")

        total = 0
        for chrom_name, df in scaled_tables.items():
            feature_arr = df[FEATURE_COLS].to_numpy(dtype=np.float32)
            starts = df["start"].to_numpy(dtype=np.int64)
            ends = df["end"].to_numpy(dtype=np.int64)

            if len(starts) > 1 and np.any(np.diff(starts) < 0):
                order = np.argsort(starts)
                feature_arr, starts, ends = feature_arr[order], starts[order], ends[order]

            feature_arr = np.nan_to_num(
                feature_arr, nan=0.0, posinf=0.0, neginf=0.0
            ).astype(np.float32)

            result = generate_sliding_windows(feature_arr, starts, ends, chrom_name, window_size)
            if result is None:
                continue

            out_path = out_dir / f"{chrom_name}_ws{window_size}.npz"
            np.savez_compressed(out_path, **result)

            total += len(result["starts"])
            print(f"  ✓ {chrom_name}: {len(result['starts']):,} 样本 -> {out_path.name}")

            del result
            gc.collect()

        print(f"[样本生成] ws{window_size} 总样本数: {total:,}")


if __name__ == "__main__":
    import argparse

    repo_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description="从特征CSV生成无标签滑动窗口预测样本（chrN_ws21.npz）"
    )
    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=repo_root / "feature" / "GM12878_25kb_feature",
        help="输入：chr*_hic_features.csv 所在目录",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "datasets" / "predict_ws21",
        help="输出目录（生成 ws21_GM12878/chrN_ws21.npz）",
    )
    parser.add_argument(
        "--window-sizes",
        type=int,
        nargs="+",
        default=[21],
        help="窗口大小列表（正奇数）",
    )
    args = parser.parse_args()

    tables = load_feature_csvs(args.feature_dir)
    generate_all_samples(
        scaled_tables=tables,
        sample_base=args.output_dir,
        resolution=25000,
        cellline="GM12878",
        window_sizes=args.window_sizes,
    )

