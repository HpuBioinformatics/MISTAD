#!/usr/bin/env python
"""
Hi-C 5维特征 + 共识边界 Gaussian 软标签 训练样本集生成器
================================================================

基于共识边界 bed（filtered_boundaries.bed）中的 TAD 边界位置:
- 特征: hic_long_8dim_25kb/chr*_hic_features.csv (8维 Hi-C 特征)
- 标签: data/raw/TAD_boundary/filtered_boundaries.bed (共识 TAD 边界)
- 软标签: 以共识边界为中心，使用 Gaussian 分布 (sigma=2.5 bins)

软标签生成逻辑:
- 对每个 bin，找到最近的共识边界
- soft_label = gaussian(distance_to_nearest_boundary, sigma=2.5)
- hard_label = soft_label > 0.5

输出样本集 (datasets/ws21/):
1. ws15/ — 滑动窗口 size=15, stride=1
2. ws21/ — 滑动窗口 size=21, stride=1
3. single_bin/ — 每个 bin 独立为一个样本 (window_size=1)

每组每个染色体保存为 .npz 文件，格式:
- features: (N, window_size, 8) 滑动窗口特征矩阵
- labels: (N,) 中心 bin 硬标签 (0/1)
- window_hard_labels: (N, window_size) 整个窗口的硬标签 (0/1)
- soft_labels: (N, window_size) 整个窗口的 Gaussian 软标签
- chroms: (N,) 染色体名称字符串数组
- starts: (N,) 中心 bin 起始坐标 (bp)
- ends: (N,) 中心 bin 终止坐标 (bp)
- positive_indices: (P,) 正样本(硬标签=1)索引
- negative_indices: (N-P,) 负样本索引

使用方法:
python generate_hic_gaussian_dataset_long.py
"""

import os
import sys
import gc
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================
# 路径配置
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent

# 特征 CSV（由 data/run_extract_feature.py 生成）
FEATURE_DIR = PROJECT_ROOT / "GM12878_25kb_feature"
# TAD 边界 bed（多 TAD caller 共识边界，经蛋白覆盖置信过滤，3884 个边界）
CONSENSUS_BOUNDARY_BED = REPO_ROOT / "data" / "raw" / "TAD_boundary" / "filtered_boundaries.bed"

# 输出目录（set_name="ws21" 会写到 datasets/ws21，与训练脚本
# data_base=REPO_ROOT/datasets + dataset="ws21" 直接对应）
OUTPUT_BASE = REPO_ROOT / "datasets"

RESOLUTION = 25000  # 25kb 分辨率

# Gaussian 软标签参数
SIGMA = 2.5  # Gaussian sigma (bins)

# 硬标签阈值 - 使用共识边界，可以适当放宽阈值
HARD_LABEL_THRESHOLD = 0.99  # soft_label > 0.95 才是硬正样本（比之前0.99放宽）

# 8 维特征列名（与特征 CSV 一致）
FEATURE_COLS = [
    "insulation_100kb",
    "insulation_150kb",
    "directionality_index",
    "up_mean",
    "down_mean",
]

# 三组样本的窗口大小配置
WINDOW_CONFIGS = {
    "ws21": 21,
}


# ============================================================
# 工具函数
# ============================================================

def gaussian_value(distance, sigma):
    """计算 Gaussian 值"""
    return float(np.exp(-(distance ** 2) / (2.0 * sigma ** 2)))


def compute_gaussian_soft_labels(
    bin_starts: np.ndarray,
    boundary_positions: Set[int],
    sigma: float = SIGMA,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    为每个 bin 计算 Gaussian 软标签和硬标签。

    Args:
        bin_starts: (M,) 各 bin 的起始坐标 (bp)
        boundary_positions: 共识边界的 start 坐标集合
        sigma: Gaussian sigma (bins)

    Returns:
        soft_labels: (M,) Gaussian 软标签 [0, 1]
        hard_labels: (M,) 硬标签 (0 或 1)
    """
    M = len(bin_starts)

    if not boundary_positions:
        print(f" ⚠️ 无 共识边界数据，所有标签设为 0")
        return np.zeros(M, dtype=np.float32), np.zeros(M, dtype=np.float32)

    # 将边界位置转为 numpy 数组并排序（用于快速搜索）
    boundaries = np.array(sorted(boundary_positions), dtype=np.int64)
    n_bounds = len(boundaries)

    soft_labels = np.zeros(M, dtype=np.float32)

    for j, start in enumerate(bin_starts):
        start_int = int(start)

        # 找到最近的边界
        # 使用二分搜索找到插入点
        idx = np.searchsorted(boundaries, start_int)

        min_dist = float("inf")

        # 检查左侧边界
        if idx > 0:
            dist = abs(start_int - int(boundaries[idx - 1]))

            # 距离转换为 bins (25kb per bin)
            dist_bins = dist // RESOLUTION
            min_dist = min(min_dist, dist_bins)

        # 检查右侧边界
        if idx < n_bounds:
            dist = abs(start_int - int(boundaries[idx]))
            dist_bins = dist // RESOLUTION
            min_dist = min(min_dist, dist_bins)

        # 计算 Gaussian 软标签
        soft_labels[j] = gaussian_value(min_dist, sigma)

    # 硬标签
    hard_labels = (soft_labels > HARD_LABEL_THRESHOLD).astype(np.float32)

    n_pos = int(hard_labels.sum())

    print(
        f" 正样本(soft>{HARD_LABEL_THRESHOLD}): "
        f"{n_pos:,} / {M:,} ({n_pos / M * 100:.2f}%)"
    )
    print(f" 软标签范围: [{soft_labels.min():.4f}, {soft_labels.max():.4f}]")

    return soft_labels, hard_labels


# ============================================================
# 数据加载
# ============================================================

def load_consensus_boundaries(bed_path: Path) -> Dict[str, Set[int]]:
    """
    加载 共识边界 BED 文件，返回每个染色体的边界 start 坐标集合。

    边界坐标修正：边界 bed 中使用 bin 起始位置（0-based），
    而 TAD 域约定边界位于 bin 末尾（即 start + resolution），相当于 bin 编号 +1。
    因此将每个边界位置往后推 1 bin（+RESOLUTION bp），使 soft label 峰值
    与 TAD 域文件（25KB_TAD/chr*.txt）中的边界约定一致。

    Returns:
        dict: {chrom_name: set of start_positions}（已偏移 +1 bin）
    """
    boundaries = {}

    if not bed_path.exists():
        print(f" ✗ 共识边界 BED 文件不存在: {bed_path}")
        return boundaries

    df = pd.read_csv(bed_path, sep="\t", header=0)

    total = 0

    for _, row in df.iterrows():
        chrom = str(row["chrom"]).strip()
        start = int(row["start"])

        # 往后推 1 bin，使边界约定与 TAD 域文件一致
        shifted_start = start + RESOLUTION

        if chrom not in boundaries:
            boundaries[chrom] = set()

        boundaries[chrom].add(shifted_start)
        total += 1

    print(f" ✓ 加载 {total} 个 共识边界（已偏移 +1 bin），覆盖 {len(boundaries)} 条染色体")

    for chrom in sorted(boundaries.keys()):
        print(f" - {chrom}: {len(boundaries[chrom])} 个边界")

    return boundaries


def load_features(feat_path: Path) -> Optional[pd.DataFrame]:
    """加载单条染色体的 8 维 Hi-C 特征 CSV"""
    try:
        df = pd.read_csv(feat_path)

        required = ["chrom", "start", "end"] + FEATURE_COLS
        missing = [c for c in required if c not in df.columns]

        if missing:
            print(f" ✗ 缺少列: {missing}")
            return None

        return df

    except Exception as e:
        print(f" ✗ 加载失败: {e}")
        return None


# ============================================================
# 滑动窗口生成
# ============================================================

def generate_sliding_windows(
    feature_arr: np.ndarray,
    hard_labels: np.ndarray,
    soft_labels: np.ndarray,
    starts: np.ndarray,
    chrom_name: str,
    window_size: int,
) -> Optional[Dict]:
    """
    生成滑动窗口训练样本。

    Args:
        feature_arr: (M, 8) 已对齐的特征
        hard_labels: (M,) 硬标签
        soft_labels: (M,) Gaussian 软标签
        starts: (M,) start 坐标 (bp)
        chrom_name: str 染色体名
        window_size: int 窗口大小（奇数）

    Returns:
        dict 包含 features, labels, soft_labels, window_hard_labels, chroms, starts, ends, pos/neg indices
    """
    M = feature_arr.shape[0]

    if M < window_size:
        print(f" ✗ bins 数 ({M}) < window_size ({window_size})，跳过")
        return None

    radius = window_size // 2

    windows = []
    center_hard = []
    window_hard = []  # 新增：整个窗口的硬标签
    window_soft = []
    center_starts = []
    center_chroms = []

    for center in range(radius, M - radius):
        w_start = center - radius
        w_end = center + radius + 1

        windows.append(feature_arr[w_start:w_end])
        center_hard.append(hard_labels[center])
        window_hard.append(hard_labels[w_start:w_end])  # 整个窗口的硬标签
        window_soft.append(soft_labels[w_start:w_end])  # 整个窗口的软标签
        center_starts.append(starts[center])
        center_chroms.append(chrom_name)

    result = {
        "features": np.array(windows, dtype=np.float32),  # (N, window_size, 8)
        "labels": np.array(center_hard, dtype=np.float32),  # (N,) 中心bin硬标签
        "window_hard_labels": np.array(window_hard, dtype=np.float32),  # (N, window_size) 整个窗口硬标签
        "soft_labels": np.array(window_soft, dtype=np.float32),  # (N, window_size) 整个窗口软标签
        "chroms": np.array(center_chroms, dtype=f"U{len(chrom_name)}"),  # (N,)
        "starts": np.array(center_starts, dtype=np.int64),  # (N,)
        "ends": np.array(center_starts, dtype=np.int64) + RESOLUTION,  # (N,)
        "positive_indices": np.where(np.array(center_hard) == 1.0)[0].astype(np.int32),
        "negative_indices": np.where(np.array(center_hard) == 0.0)[0].astype(np.int32),
    }

    return result


def process_chromosome_for_window(
    feat_path: Path,
    consensus_boundaries: Dict[str, Set[int]],
    output_dir: Path,
    window_size: int,
    set_name: str,
) -> Optional[Dict]:
    """处理单个染色体，生成指定窗口大小的样本并保存"""
    chrom_id = feat_path.stem.replace("_hic_features", "")

    print(f"\n [{set_name}] 处理 {chrom_id} (window={window_size})...")

    # 加载特征
    feat_df = load_features(feat_path)

    if feat_df is None:
        return None

    # 提取特征数组
    feat_starts = feat_df["start"].values.astype(np.int64)
    feat_vals = feat_df[FEATURE_COLS].values.astype(np.float32)
    chrom_name = str(feat_df["chrom"].iloc[0])

    # 计算 共识边界 Gaussian 软标签
    print(f" 计算 Gaussian 软标签 (sigma={SIGMA})...")

    soft_lbl, hard_lbl = compute_gaussian_soft_labels(
        feat_starts,
        consensus_boundaries.get(chrom_name, set()),
        sigma=SIGMA,
    )

    print(f" 对齐后: {len(feat_vals)} bins")

    # 生成滑动窗口
    result = generate_sliding_windows(
        feat_vals,
        hard_lbl,
        soft_lbl,
        feat_starts,
        chrom_name,
        window_size,
    )

    if result is None:
        return None

    # 保存
    out_path = output_dir / f"{chrom_id}_ws{window_size}.npz"

    np.savez_compressed(
        out_path,
        features=result["features"],
        labels=result["labels"],
        window_hard_labels=result["window_hard_labels"],  # 新增：窗口硬标签
        soft_labels=result["soft_labels"],
        chroms=result["chroms"],
        starts=result["starts"],
        ends=result["ends"],
        positive_indices=result["positive_indices"],
        negative_indices=result["negative_indices"],
    )

    n_total = len(result["labels"])
    n_pos = len(result["positive_indices"])
    n_neg = len(result["negative_indices"])

    summary = {
        "chrom": chrom_id,
        "set_name": set_name,
        "window_size": window_size,
        "n_samples": n_total,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "feature_shape": result["features"].shape,
    }

    print(f" ✓ 保存: {out_path.name}")
    print(f" 样本数: {n_total:,} (正:{n_pos:,} 负:{n_neg:,})")
    print(f" 特征形状: {result['features'].shape}")

    return summary


# ============================================================
# 主流程
# ============================================================

def main():
    print("=" * 70)
    print("Hi-C 5维特征 + 共识边界 Gaussian 软标签 训练样本集生成器")
    print("=" * 70)

    print(f" 特征目录: {FEATURE_DIR}")
    print(f" 共识边界标签: {CONSENSUS_BOUNDARY_BED}")
    print(f" 输出目录: {OUTPUT_BASE}")
    print(f" 分辨率: {RESOLUTION // 1000}kb")
    print(f" Sigma: {SIGMA}")
    print(f" 硬标签阈值: {HARD_LABEL_THRESHOLD}")
    print(f" 特征列: {FEATURE_COLS}")
    print(f" 样本集: {list(WINDOW_CONFIGS.keys())}")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. 加载 共识边界
    # --------------------------------------------------------
    print("\n[1/4] 加载 共识边界...")

    consensus_boundaries = load_consensus_boundaries(CONSENSUS_BOUNDARY_BED)

    if not consensus_boundaries:
        print(" ✗ 未加载到任何 共识边界，退出")
        return

    # --------------------------------------------------------
    # 2. 发现所有可用的特征文件
    # --------------------------------------------------------
    print("\n[2/4] 扫描特征文件...")

    feat_files = sorted(FEATURE_DIR.glob("chr*_hic_features.csv"))

    print(f" ✓ 特征文件: {len(feat_files)} 个")

    # 找到有特征的染色体
    paired_chroms = []

    for fp in feat_files:
        chrom_key = fp.stem.replace("_hic_features", "")
        paired_chroms.append((fp, chrom_key))

    print(f" ✓ 可处理染色体: {len(paired_chroms)} 条")

    for _, c in paired_chroms:
        print(f" - {c}")

    # --------------------------------------------------------
    # 3. 为每组窗口配置创建输出目录
    # --------------------------------------------------------
    print("\n[3/4] 创建输出目录...")

    output_dirs = {}

    for set_name, ws in WINDOW_CONFIGS.items():
        d = OUTPUT_BASE / set_name
        d.mkdir(parents=True, exist_ok=True)

        output_dirs[set_name] = d

        print(f" ✓ {set_name}/ (window_size={ws}) → {d}")

    # --------------------------------------------------------
    # 4. 逐染色体、逐窗口配置生成样本
    # --------------------------------------------------------
    print("\n[4/4] 生成训练样本...")

    all_summaries = {name: [] for name in WINDOW_CONFIGS}

    for idx, (feat_path, chrom_id) in enumerate(paired_chroms):
        print(f"\n{'─' * 60}")
        print(f" 染色体 [{idx + 1}/{len(paired_chroms)}]: {chrom_id}")
        print(f"{'─' * 60}")

        for set_name, window_size in WINDOW_CONFIGS.items():
            summary = process_chromosome_for_window(
                feat_path=feat_path,
                consensus_boundaries=consensus_boundaries,
                output_dir=output_dirs[set_name],
                window_size=window_size,
                set_name=set_name,
            )

            if summary:
                all_summaries[set_name].append(summary)

            gc.collect()

    # --------------------------------------------------------
    # 5. 打印汇总报告
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("✓ 全部 共识边界 Gaussian 软标签样本集生成完成")
    print("=" * 70)

    for set_name, summaries in all_summaries.items():
        ws = WINDOW_CONFIGS[set_name]

        total_samples = sum(s["n_samples"] for s in summaries)
        total_pos = sum(s["n_pos"] for s in summaries)
        total_neg = sum(s["n_neg"] for s in summaries)
        n_chroms = len(summaries)

        print(f"\n ┌─ {set_name} (window_size={ws})")
        print(f" │ 染色体数: {n_chroms}")
        print(f" │ 总样本数: {total_samples:,}")
        print(f" │ 正样本: {total_pos:,} ({total_pos / max(total_samples, 1) * 100:.2f}%)")
        print(f" │ 负样本: {total_neg:,} ({total_neg / max(total_samples, 1) * 100:.2f}%)")
        print(f" │ 正负比: {total_pos}:{total_neg}")
        print(f" │ 输出目录: {output_dirs[set_name]}")
        print(f" └─")

    print(f"\n{'=' * 70}")


if __name__ == "__main__":
    main()


