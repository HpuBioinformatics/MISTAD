#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
从 .hic 文件提取 KR 归一化矩阵并转换为 .npy 格式
"""

import os
import subprocess
import numpy as np
from pathlib import Path
from typing import Dict, List

# 配置参数
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
JUICER_TOOLS_JAR = str(REPO_ROOT / "data" / "juicer_tools.jar")
HIC_FILE = "/public_data/liupinyuan/4DN12878/4DNFI1UEG1HD.hic"
OUTPUT_DIR = str(REPO_ROOT / "data" / "raw" / "GM12878_25kb")
RESOLUTION = 25000  # 25kb

# hg38 染色体长度
CHROM_LENGTHS = {
    "1": 248_956_422, "2": 242_193_529, "3": 198_295_559, "4": 190_214_555,
    "5": 181_538_259, "6": 170_805_979, "7": 159_345_973, "8": 145_138_636,
    "9": 138_394_717, "10": 133_797_422, "11": 135_086_622, "12": 133_275_309,
    "13": 114_364_328, "14": 107_043_718, "15": 101_991_189, "16": 90_338_345,
    "17": 83_257_441, "18": 80_373_285, "19": 58_617_616, "20": 64_444_167,
    "21": 46_709_983, "22": 50_818_468, "X": 156_040_895, "Y": 57_227_415,
}


def extract_matrix_from_hic(chrom: str, output_file: str) -> bool:
    """使用 juicer_tools 提取单个染色体的 KR 归一化矩阵"""
    cmd = [
        "java", "-jar", JUICER_TOOLS_JAR,
        "dump", "observed", "KR",
        HIC_FILE, chrom, chrom,
        "BP", str(RESOLUTION),
        output_file
    ]

    print(f"提取染色体 {chrom}...")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"  ✓ 已生成: {output_file}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ✗ 提取失败: {e.stderr}")
        return False


def sparse_to_dense_matrix(sparse_file: str, chrom: str) -> np.ndarray:
    """将稀疏矩阵文件转换为稠密矩阵"""
    print(f"  转换 {sparse_file} 为稠密矩阵...")

    # 计算矩阵大小
    chrom_len = CHROM_LENGTHS.get(chrom, 250_000_000)
    n_bins = (chrom_len + RESOLUTION - 1) // RESOLUTION

    print(f"  染色体 {chrom} 长度: {chrom_len:,} bp, bins: {n_bins}")

    # 读取稀疏矩阵数据
    try:
        data = np.loadtxt(sparse_file, dtype=np.float64)
    except Exception as e:
        print(f"  ✗ 读取文件失败: {e}")
        return None

    if data.size == 0:
        print(f"  ✗ 文件为空")
        return None

    # 创建稠密矩阵
    dense_matrix = np.zeros((n_bins, n_bins), dtype=np.float64)

    # 填充数据
    print(f"  填充 {len(data)} 个非零元素...")
    for row in data:
        i = int(row[0] // RESOLUTION)
        j = int(row[1] // RESOLUTION)
        value = row[2]

        if 0 <= i < n_bins and 0 <= j < n_bins:
            dense_matrix[i, j] = value
            dense_matrix[j, i] = value  # 对称矩阵

    # 处理 NaN 值
    dense_matrix = np.nan_to_num(dense_matrix, nan=0.0)

    print(f"  ✓ 矩阵大小: {dense_matrix.shape}, 非零元素: {np.count_nonzero(dense_matrix):,}")

    return dense_matrix


def main():
    """主函数：提取所有染色体并转换为 npy"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 要处理的染色体列表
    chromosomes = [str(i) for i in range(1, 23)] + ["X", "Y"]

    print(f"开始处理 {len(chromosomes)} 个染色体...")
    print(f"输入文件: {HIC_FILE}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"分辨率: {RESOLUTION:,} bp")
    print("-" * 60)

    success_count = 0

    for chrom in chromosomes:
        print(f"\n{'='*60}")
        print(f"处理染色体 {chrom}")
        print(f"{'='*60}")

        # 临时稀疏矩阵文件
        sparse_file = os.path.join(OUTPUT_DIR, f"chr{chrom}_sparse.txt")

        # 输出 npy 文件
        npy_file = os.path.join(OUTPUT_DIR, f"chr{chrom}_25kb_hicKR.npy")

        # 如果 npy 文件已存在，跳过
        if os.path.exists(npy_file):
            print(f"  ✓ 已存在: {npy_file}，跳过")
            success_count += 1
            continue

        # 提取稀疏矩阵
        if not extract_matrix_from_hic(chrom, sparse_file):
            continue

        # 转换为稠密矩阵
        dense_matrix = sparse_to_dense_matrix(sparse_file, chrom)
        if dense_matrix is None:
            continue

        # 保存为 npy
        np.save(npy_file, dense_matrix)
        print(f"  ✓ 已保存: {npy_file}")

        # 删除临时稀疏矩阵文件（如果存在）
        if os.path.exists(sparse_file):
            os.remove(sparse_file)
            print(f"  ✓ 已删除临时文件: {sparse_file}")

        success_count += 1

    print(f"\n{'='*60}")
    print(f"处理完成！成功: {success_count}/{len(chromosomes)}")
    print(f"{'='*60}")

    # 列出所有生成的 npy 文件
    print(f"\n生成的 .npy 文件:")
    for chrom in chromosomes:
        npy_file = os.path.join(OUTPUT_DIR, f"chr{chrom}_25kb_hicKR.npy")
        if os.path.exists(npy_file):
            size_mb = os.path.getsize(npy_file) / (1024 * 1024)
            print(f"  chr{chrom}: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()