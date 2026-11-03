#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Hi-C 特征提取模块
==================

从 KR 归一化稠密 .npy 矩阵提取 8 维特征
（Crane-style insulation / DI / cosine / PageRank / local profile），
并对全部染色体统一做一次 RobustScaler 标准化。

注意：输入必须是包含全部染色体 .npy 矩阵的目录，不支持只传入单条染色体
的文件路径，以保证 RobustScaler 一定是在全染色体样本上 fit 的。
"""

import gc
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import RobustScaler, normalize

warnings.filterwarnings("ignore")


# ============================================================
# 基因组距离常量（分辨率无关）
# ============================================================

GENOMIC_INSULATION_100KB = 100_000
GENOMIC_INSULATION_150KB = 150_000
GENOMIC_DI_WINDOW = 200_000
GENOMIC_COSINE_BANDWIDTH = 750_000
GENOMIC_PAGERANK_BANDWIDTH = 1_500_000
GENOMIC_LOCAL_WINDOW = 150_000


# ============================================================
# 分辨率相关 bin 参数（25kb 分辨率）
# ============================================================

RESOLUTION = 25_000

INSULATION_100KB_BINS = 4
INSULATION_150KB_BINS = 6
DI_WINDOW_BINS = 8
COSINE_BANDWIDTH_BINS = 30
COSINE_TOP_K_BY_CONTACT_SUM = 30
PAGERANK_BANDWIDTH_BINS = 60
LOCAL_WINDOW_BINS = 6


FEATURE_COLS = [
    "insulation_100kb",
    "insulation_150kb",
    "directionality_index",
    "cos_sim_1",
    "cos_sim_2",
    "pagerank",
    "up_mean",
    "down_mean",
]


# ============================================================
# GRCh38 / hg38 染色体长度
# ============================================================

CHROM_LENGTHS = {
    "1": 248_956_422, "2": 242_193_529, "3": 198_295_559, "4": 190_214_555,
    "5": 181_538_259, "6": 170_805_979, "7": 159_345_973, "8": 145_138_636,
    "9": 138_394_717, "10": 133_797_422, "11": 135_086_622, "12": 133_275_309,
    "13": 114_364_328, "14": 107_043_718, "15": 101_991_189, "16": 90_338_345,
    "17": 83_257_441, "18": 80_373_285, "19": 58_617_616, "20": 64_444_167,
    "21": 46_709_983, "22": 50_818_468, "X": 156_040_895, "Y": 57_227_415,
}


def configure_resolution(resolution: int) -> None:
    """根据 resolution 设置所有 bin 参数（bin 数 = 基因组距离 / 分辨率）。"""
    global RESOLUTION
    global INSULATION_100KB_BINS, INSULATION_150KB_BINS
    global DI_WINDOW_BINS, COSINE_BANDWIDTH_BINS, COSINE_TOP_K_BY_CONTACT_SUM
    global PAGERANK_BANDWIDTH_BINS, LOCAL_WINDOW_BINS

    RESOLUTION = resolution
    INSULATION_100KB_BINS = GENOMIC_INSULATION_100KB // resolution
    INSULATION_150KB_BINS = GENOMIC_INSULATION_150KB // resolution
    DI_WINDOW_BINS = GENOMIC_DI_WINDOW // resolution
    COSINE_BANDWIDTH_BINS = GENOMIC_COSINE_BANDWIDTH // resolution
    COSINE_TOP_K_BY_CONTACT_SUM = COSINE_BANDWIDTH_BINS
    PAGERANK_BANDWIDTH_BINS = GENOMIC_PAGERANK_BANDWIDTH // resolution
    LOCAL_WINDOW_BINS = GENOMIC_LOCAL_WINDOW // resolution


# ============================================================
# 通用工具函数
# ============================================================

def normalize_chrom_id(chrom_id: str) -> str:
    """统一为不带 chr 前缀的形式：chr1 -> 1，chrX_hicKR_dense -> X。"""
    text = str(chrom_id).strip()
    match = re.search(r"chr(?:om)?([0-9]+|X|Y|M|MT)", text, flags=re.IGNORECASE)

    if match:
        chrom = match.group(1).upper()
    else:
        chrom = text
        if chrom.lower().startswith("chr"):
            chrom = chrom[3:]
        chrom = re.split(r"[_\-.]", chrom, maxsplit=1)[0]
        chrom = chrom.upper()

    if chrom == "M":
        chrom = "MT"
    return chrom


def safe_chrom_output_name(chrom_id: str) -> str:
    """返回带 chr 前缀的染色体名称。"""
    return f"chr{normalize_chrom_id(chrom_id)}"


def parse_chrom_from_npy_filename(filename: str) -> Optional[str]:
    """从 .npy 文件名解析染色体，如 chr1_10kb_hicKR.npy -> 1。"""
    stem = Path(filename).stem
    match = re.search(r"chr(?:om)?([0-9]+|X|Y|M|MT)", stem, flags=re.IGNORECASE)

    if not match:
        return None

    chrom = match.group(1).upper()
    if chrom == "M":
        chrom = "MT"
    return chrom


def chromosome_sort_key(chrom_id: str):
    """chr1, chr2, ..., chrX, chrY 排序。"""
    chrom = normalize_chrom_id(chrom_id)
    if chrom.isdigit():
        return 0, int(chrom)
    special_order = {"X": 23, "Y": 24, "MT": 25}
    return 1, special_order.get(chrom, chrom)


def clean_contact_values(values: np.ndarray) -> np.ndarray:
    """清洗局部 contact 数组，返回可写 float64 副本。"""
    cleaned = np.array(values, dtype=np.float64, copy=True)
    cleaned = np.nan_to_num(cleaned, nan=0.0, posinf=0.0, neginf=0.0)
    cleaned[cleaned < 0] = 0.0
    return cleaned


# ============================================================
# .npy 文件读取（仅支持目录，取消单条染色体输入模式）
# ============================================================

def load_dense_matrix(npy_path: str, use_memmap: bool = True) -> Optional[np.ndarray]:
    """加载 KR 归一化稠密 Hi-C 方阵，默认内存映射读取。"""
    try:
        matrix = np.load(npy_path, mmap_mode="r" if use_memmap else None, allow_pickle=False)

        if not isinstance(matrix, np.ndarray) or matrix.ndim != 2:
            print(f"  ✗ 输入不是二维 numpy 矩阵: {npy_path}")
            return None
        if matrix.shape[0] != matrix.shape[1]:
            print(f"  ✗ 输入不是方阵: shape={matrix.shape}")
            return None
        if not np.issubdtype(matrix.dtype, np.number):
            print(f"  ✗ 矩阵不是数值类型: dtype={matrix.dtype}")
            return None

        print(f"  ✓ 已加载: {npy_path} shape={matrix.shape} dtype={matrix.dtype}")
        return matrix

    except Exception as exc:
        print(f"  ✗ 加载 .npy 矩阵失败: {exc}")
        return None


def get_npy_files(input_dir: str) -> List[Tuple[str, str]]:
    """
    扫描目录下全部染色体 .npy 矩阵，返回 [(npy_path, chrom_id), ...]。

    只支持目录输入：全染色体标准化要求所有染色体一次性参与
    RobustScaler 的 fit，因此不再支持单个文件的单染色体计算模式。
    """
    input_obj = Path(input_dir)
    if not input_obj.is_dir():
        raise ValueError(
            f"输入必须是包含全部染色体 .npy 矩阵的目录（已取消单染色体输入模式）: {input_dir}"
        )

    results: List[Tuple[str, str]] = []
    for file_path in input_obj.iterdir():
        if not file_path.is_file() or file_path.suffix.lower() != ".npy":
            continue
        chrom = parse_chrom_from_npy_filename(file_path.name)
        if chrom is None:
            print(f"  ⚠️ 跳过无法解析染色体的文件: {file_path.name}")
            continue
        results.append((str(file_path), chrom))

    results.sort(key=lambda item: chromosome_sort_key(item[1]))
    return results


# ============================================================
# Crane-style insulation
# ============================================================

def compute_crane_insulation_score(
    matrix: np.ndarray,
    window_bins: int,
    insulation_mode: str = "mean",
    ignore_zero: bool = False,
) -> np.ndarray:
    """
    Crane-style insulation：matrix[y-w:y, y+1:y+w+1]。
    归一化：log2(raw_score / chromosome_mean_raw_score)。
    """
    n_bins = matrix.shape[0]
    raw_scores = np.full(n_bins, np.nan, dtype=np.float64)
    w = int(window_bins)

    for y in range(w, n_bins - w):
        row_start, row_end = y - w, y
        col_start, col_end = y + 1, y + w + 1

        if row_start < 0 or col_end > n_bins:
            continue

        window = clean_contact_values(matrix[row_start:row_end, col_start:col_end])
        if window.size == 0:
            continue

        values = window[window > 0] if ignore_zero else window.ravel()
        if values.size == 0:
            continue

        if insulation_mode == "sum":
            score = float(np.sum(values))
        elif insulation_mode == "median":
            score = float(np.median(values))
        else:
            score = float(np.mean(values))

        if np.isfinite(score):
            raw_scores[y] = score

    valid_mask = np.isfinite(raw_scores)
    valid_scores = raw_scores[valid_mask]

    if valid_scores.size == 0:
        return np.zeros(n_bins, dtype=np.float64)

    chrom_mean_score = float(np.mean(valid_scores))
    result = np.zeros(n_bins, dtype=np.float64)

    if not np.isfinite(chrom_mean_score) or chrom_mean_score <= 0:
        result[valid_mask] = raw_scores[valid_mask] - chrom_mean_score
    else:
        eps = 1e-10
        result[valid_mask] = np.log2(
            (raw_scores[valid_mask] + eps) / (chrom_mean_score + eps)
        )

    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


# ============================================================
# Directionality Index
# ============================================================

def compute_directionality_index(
    matrix: np.ndarray,
    window_bins: int = DI_WINDOW_BINS,
) -> np.ndarray:
    """DI = (B - A) / sqrt(A + B + eps)，A=上游均值，B=下游均值。"""
    n_bins = matrix.shape[0]
    di = np.zeros(n_bins, dtype=np.float64)
    eps = 1e-10

    for i in range(window_bins, n_bins - window_bins):
        upstream = clean_contact_values(matrix[i, i - window_bins:i])
        downstream = clean_contact_values(matrix[i, i + 1:i + window_bins + 1])

        A = float(np.mean(upstream)) if upstream.size > 0 else 0.0
        B = float(np.mean(downstream)) if downstream.size > 0 else 0.0

        if np.isfinite(A) and np.isfinite(B):
            di[i] = (B - A) / np.sqrt(A + B + eps)

    return np.nan_to_num(di, nan=0.0, posinf=0.0, neginf=0.0)


# ============================================================
# Cosine Similarity
# ============================================================

def cosine_row_shift_topk(
    matrix: np.ndarray,
    row_index: int,
    shift: int,
    bandwidth: int,
    top_k: Optional[int] = None,
) -> float:
    """第 row_index 行与 row_index+shift 行在共同基因组窗口内的余弦相似度。"""
    n_bins = matrix.shape[0]
    other_index = row_index + shift

    if other_index < 0 or other_index >= n_bins:
        return 0.0

    common_start = max(0, row_index - bandwidth, other_index - bandwidth)
    common_end = min(n_bins, row_index + bandwidth + 1, other_index + bandwidth + 1)

    if common_end <= common_start:
        return 0.0

    vector_1 = clean_contact_values(matrix[row_index, common_start:common_end])
    vector_2 = clean_contact_values(matrix[other_index, common_start:common_end])

    if top_k is not None and 0 < top_k < vector_1.size:
        contact_sum = vector_1 + vector_2
        top_indices = np.argpartition(contact_sum, -top_k)[-top_k:]
        vector_1 = vector_1[top_indices]
        vector_2 = vector_2[top_indices]

    norm_1 = np.linalg.norm(vector_1)
    norm_2 = np.linalg.norm(vector_2)

    if norm_1 <= 0 or norm_2 <= 0:
        return 0.0

    similarity = float(np.dot(vector_1, vector_2) / (norm_1 * norm_2))
    return similarity if np.isfinite(similarity) else 0.0


def compute_cosine_similarity(
    matrix: np.ndarray,
    bandwidth: int = COSINE_BANDWIDTH_BINS,
    top_k: Optional[int] = COSINE_TOP_K_BY_CONTACT_SUM,
) -> Tuple[np.ndarray, np.ndarray]:
    """cos_sim_1: 当前 bin 与后 1 个 bin；cos_sim_2: 当前 bin 与后 2 个 bin。"""
    n_bins = matrix.shape[0]
    cos_sim_1 = np.zeros(n_bins, dtype=np.float64)
    cos_sim_2 = np.zeros(n_bins, dtype=np.float64)

    for i in range(n_bins):
        cos_sim_1[i] = cosine_row_shift_topk(matrix, i, 1, bandwidth, top_k)
        cos_sim_2[i] = cosine_row_shift_topk(matrix, i, 2, bandwidth, top_k)

    return (
        np.nan_to_num(cos_sim_1, nan=0.0, posinf=0.0, neginf=0.0),
        np.nan_to_num(cos_sim_2, nan=0.0, posinf=0.0, neginf=0.0),
    )


# ============================================================
# PageRank
# ============================================================

def calculate_pagerank(
    hic_matrix: np.ndarray,
    bandwidth: int = PAGERANK_BANDWIDTH_BINS,
    damping: float = 0.85,
    max_iter: int = 30,
    tol: float = 1e-6,
) -> np.ndarray:
    """带状稀疏邻接矩阵 PageRank。"""
    n_bins = hic_matrix.shape[0]
    if n_bins == 0:
        return np.array([], dtype=np.float64)

    max_band = min(bandwidth, n_bins - 1)
    offsets: List[int] = []
    diagonals: List[np.ndarray] = []

    for offset in range(-max_band, max_band + 1):
        diagonal = clean_contact_values(np.diagonal(hic_matrix, offset=offset))
        if diagonal.size == 0:
            continue
        offsets.append(offset)
        diagonals.append(diagonal)

    if not diagonals:
        return np.ones(n_bins, dtype=np.float64) / n_bins

    adjacency = sparse.diags(
        diagonals=diagonals, offsets=offsets,
        shape=(n_bins, n_bins), format="csr", dtype=np.float64,
    )

    row_sums = np.asarray(adjacency.sum(axis=1)).ravel()
    dangling_mask = row_sums <= 0

    transition = normalize(adjacency, norm="l1", axis=1, copy=True)

    rank = np.ones(n_bins, dtype=np.float64) / n_bins
    teleport = np.ones(n_bins, dtype=np.float64) / n_bins

    for _ in range(max_iter):
        dangling_mass = float(rank[dangling_mask].sum())
        next_rank = (
            damping * (transition.T @ rank)
            + damping * dangling_mass * teleport
            + (1.0 - damping) * teleport
        )
        total = float(next_rank.sum())
        if total > 0:
            next_rank = next_rank / total
        difference = np.linalg.norm(next_rank - rank, ord=1)
        rank = next_rank
        if difference < tol:
            break

    return np.nan_to_num(rank, nan=0.0, posinf=0.0, neginf=0.0)


# ============================================================
# Local Profile
# ============================================================

def compute_local_profile(
    matrix: np.ndarray,
    window_bins: int = LOCAL_WINDOW_BINS,
) -> Tuple[np.ndarray, np.ndarray]:
    """当前 bin 与上游 / 下游的局部平均互作。"""
    n_bins = matrix.shape[0]
    up_mean = np.zeros(n_bins, dtype=np.float64)
    down_mean = np.zeros(n_bins, dtype=np.float64)

    for i in range(n_bins):
        upstream = clean_contact_values(matrix[i, max(0, i - window_bins):i])
        downstream = clean_contact_values(matrix[i, i + 1:min(n_bins, i + window_bins + 1)])

        if upstream.size > 0:
            up_mean[i] = float(np.mean(upstream))
        if downstream.size > 0:
            down_mean[i] = float(np.mean(downstream))

    return (
        np.nan_to_num(up_mean, nan=0.0, posinf=0.0, neginf=0.0),
        np.nan_to_num(down_mean, nan=0.0, posinf=0.0, neginf=0.0),
    )


# ============================================================
# 单染色体 raw 特征提取（核心函数，未做任何修改）
# ============================================================

def extract_raw_hic_features_for_chromosome(
    npy_path: str,
    chrom_id: str,
    resolution: int = RESOLUTION,
    insulation_mode: str = "mean",
    insulation_ignore_zero: bool = False,
    use_memmap: bool = True,
) -> Optional[pd.DataFrame]:
    """从 KR 稠密矩阵提取 8 维 raw 特征（不做 RobustScaler）。"""
    normalized_chrom = normalize_chrom_id(chrom_id)
    chrom_name = safe_chrom_output_name(normalized_chrom)

    print(f"\n{'=' * 72}")
    print(f"提取 {chrom_name} 的 {resolution // 1000}kb raw Hi-C 特征")
    print(f"{'=' * 72}")

    try:
        print("[1/6] 加载 KR 矩阵...")
        matrix = load_dense_matrix(npy_path=npy_path, use_memmap=use_memmap)
        if matrix is None:
            return None

        n_bins = matrix.shape[0]
        print(f"  ✓ 染色体: {chrom_name}")
        print(f"  ✓ bin 数: {n_bins:,}")
        print(f"  ✓ 分辨率: {resolution:,} bp")

        print("[2/6] 创建 genomic bins...")
        starts = np.arange(n_bins, dtype=np.int64) * resolution

        chrom_length = CHROM_LENGTHS.get(normalized_chrom)

        if chrom_length is not None:
            expected_bins = int(np.ceil(chrom_length / resolution))
            if n_bins == expected_bins:
                ends = np.minimum(starts + resolution, chrom_length)
            else:
                print("  ⚠️ 矩阵 bin 数与 hg38 染色体长度不完全匹配：")
                print(f"     矩阵 bin 数: {n_bins:,}")
                print(f"     hg38 预期 bin 数: {expected_bins:,}")
                print("     将按照矩阵大小生成完整坐标。")
                ends = starts + resolution
        else:
            ends = starts + resolution

        master_df = pd.DataFrame({
            "chrom": np.full(n_bins, chrom_name, dtype=object),
            "start": starts,
            "end": ends,
        })

        print(f"  ✓ 坐标数量: {len(master_df):,}")

        print(f"[3/6] 计算 Crane-style insulation：100kb({INSULATION_100KB_BINS} bins) 和 150kb({INSULATION_150KB_BINS} bins)...")

        master_df["insulation_100kb"] = compute_crane_insulation_score(
            matrix=matrix, window_bins=INSULATION_100KB_BINS,
            insulation_mode=insulation_mode, ignore_zero=insulation_ignore_zero,
        )

        master_df["insulation_150kb"] = compute_crane_insulation_score(
            matrix=matrix, window_bins=INSULATION_150KB_BINS,
            insulation_mode=insulation_mode, ignore_zero=insulation_ignore_zero,
        )

        gc.collect()

        print(f"[4/6] 计算 Directionality Index({DI_WINDOW_BINS} bins) 和 Cosine Similarity(bw={COSINE_BANDWIDTH_BINS}, top-k={COSINE_TOP_K_BY_CONTACT_SUM})...")

        master_df["directionality_index"] = compute_directionality_index(
            matrix=matrix, window_bins=DI_WINDOW_BINS,
        )

        cos_sim_1, cos_sim_2 = compute_cosine_similarity(
            matrix=matrix,
            bandwidth=COSINE_BANDWIDTH_BINS,
            top_k=COSINE_TOP_K_BY_CONTACT_SUM,
        )

        master_df["cos_sim_1"] = cos_sim_1
        master_df["cos_sim_2"] = cos_sim_2

        del cos_sim_1, cos_sim_2
        gc.collect()

        print(f"[5/6] 计算 PageRank({PAGERANK_BANDWIDTH_BINS} bins) 和局部互作强度({LOCAL_WINDOW_BINS} bins)...")

        master_df["pagerank"] = calculate_pagerank(
            hic_matrix=matrix, bandwidth=PAGERANK_BANDWIDTH_BINS,
        )

        up_mean, down_mean = compute_local_profile(
            matrix=matrix, window_bins=LOCAL_WINDOW_BINS,
        )

        master_df["up_mean"] = up_mean
        master_df["down_mean"] = down_mean

        del up_mean, down_mean, matrix
        gc.collect()

        print("[6/6] 清洗 raw features...")
        master_df[FEATURE_COLS] = (
            master_df[FEATURE_COLS]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )

        for feature_col in FEATURE_COLS:
            master_df[feature_col] = master_df[feature_col].astype(np.float64)

        print(f"{'=' * 72}")
        print(f"✓ {chrom_name} raw 特征提取完成")
        print(f"  样本数: {len(master_df):,}")
        print(f"  特征数: {len(FEATURE_COLS)}")
        print(f"  分辨率: {resolution:,} bp")
        print(f"{'=' * 72}")

        return master_df

    except Exception as exc:
        print(f"✗ 处理 {chrom_name} 失败: {exc}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================
# 全染色体 RobustScaler
# ============================================================

def apply_global_robust_scaler(
    chrom_feature_tables: Dict[str, pd.DataFrame],
) -> Tuple[Dict[str, pd.DataFrame], RobustScaler]:
    """所有染色体合并后统一 fit 一个 RobustScaler。"""
    print(f"\n{'=' * 72}")
    print("开始全染色体 RobustScaler 标准化")
    print(f"{'=' * 72}")

    if not chrom_feature_tables:
        raise ValueError("没有可用于 RobustScaler 的特征表")

    feature_blocks: List[np.ndarray] = []

    for chrom_name, dataframe in chrom_feature_tables.items():
        block = (
            dataframe[FEATURE_COLS]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .to_numpy(dtype=np.float64)
        )
        feature_blocks.append(block)
        print(f"  加入 scaler fit: {chrom_name}, rows={len(block):,}")

    all_features = np.concatenate(feature_blocks, axis=0)
    del feature_blocks
    gc.collect()

    print(f"  ✓ 合并样本数: {len(all_features):,}")

    scaler = RobustScaler()
    scaler.fit(all_features)

    del all_features
    gc.collect()

    scaled_tables: Dict[str, pd.DataFrame] = {}

    for chrom_name, dataframe in chrom_feature_tables.items():
        scaled_df = dataframe.copy()

        cleaned_features = (
            scaled_df[FEATURE_COLS]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .to_numpy(dtype=np.float64)
        )

        scaled_values = scaler.transform(cleaned_features)
        scaled_df.loc[:, FEATURE_COLS] = scaled_values

        scaled_tables[chrom_name] = scaled_df
        print(f"  ✓ 已标准化: {chrom_name}, rows={len(scaled_df):,}")

    print(f"{'=' * 72}")
    print("✓ 全染色体 RobustScaler 完成")
    print(f"{'=' * 72}")

    return scaled_tables, scaler


def save_scaler_parameters(scaler: RobustScaler, output_dir: Path) -> None:
    """保存 RobustScaler 的中心值和缩放值。"""
    scaler_file = output_dir / "global_robust_scaler_parameters.npz"

    center = getattr(scaler, "center_", np.zeros(len(FEATURE_COLS), dtype=np.float64))
    scale = getattr(scaler, "scale_", np.ones(len(FEATURE_COLS), dtype=np.float64))

    np.savez_compressed(
        scaler_file,
        feature_names=np.array(FEATURE_COLS, dtype="U64"),
        center=np.asarray(center, dtype=np.float64),
        scale=np.asarray(scale, dtype=np.float64),
    )

    print(f"  ✓ 已保存 Scaler 参数: {scaler_file}")


# ============================================================
# 全染色体特征提取入口（取消单染色体输入模式）
# ============================================================

def extract_all_chromosomes(
    input_dir: str,
    feature_dir: Path,
    resolution: int,
    insulation_mode: str = "mean",
    insulation_ignore_zero: bool = False,
    use_memmap: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    扫描目录内全部染色体 .npy -> 逐染色体提取 raw 特征
    -> 全染色体统一 RobustScaler -> 保存 scaled CSV。

    返回 {chrom_name: scaled_dataframe}，供样本生成直接复用。
    """
    configure_resolution(resolution)

    print(f"[特征提取] 输入目录: {input_dir}")
    print(f"[特征提取] 输出目录: {feature_dir}")
    print(f"[特征提取] 分辨率:   {resolution:,} bp")

    npy_files = get_npy_files(input_dir)
    if not npy_files:
        raise ValueError(f"未找到可识别的染色体 .npy 矩阵: {input_dir}")
    print(f"[特征提取] 发现 {len(npy_files)} 条染色体矩阵")

    raw_tables: Dict[str, pd.DataFrame] = {}

    for npy_path, chrom_id in npy_files:
        df = extract_raw_hic_features_for_chromosome(
            npy_path=npy_path,
            chrom_id=chrom_id,
            resolution=resolution,
            insulation_mode=insulation_mode,
            insulation_ignore_zero=insulation_ignore_zero,
            use_memmap=use_memmap,
        )
        if df is None or df.empty:
            print(f"  ⚠️ 跳过: {Path(npy_path).name}")
            continue

        chrom_name = str(df.iloc[0]["chrom"])
        if chrom_name in raw_tables:
            print(f"  ⚠️ 检测到重复染色体 {chrom_name}，跳过后续文件")
            continue
        raw_tables[chrom_name] = df

    if not raw_tables:
        raise RuntimeError("没有成功提取任何染色体特征")

    # 全染色体统一标准化
    scaled_tables, scaler = apply_global_robust_scaler(raw_tables)
    del raw_tables
    gc.collect()

    feature_dir.mkdir(parents=True, exist_ok=True)
    save_scaler_parameters(scaler=scaler, output_dir=feature_dir)

    for chrom_name, df in scaled_tables.items():
        out_file = feature_dir / f"{chrom_name}_hic_features.csv"
        df.to_csv(out_file, index=False)
        print(f"  ✓ 已保存: {out_file}")

    print(f"[特征提取] 完成，共 {len(scaled_tables)} 条染色体 -> {feature_dir}")
    return scaled_tables

