#!/usr/bin/env python3
"""
对预选 TAD 边界做蛋白覆盖二次筛选。

输入 (--candidates)，制表符分隔，列固定：
    chrom  start  end  support_tool_count  support_tool_names

输出两个文件（列 = 输入原有列 + 新增标识列，格式保持一致）：
    <out_prefix>.annotated.tsv   全部候选 + 蛋白覆盖标识，不过滤
    <out_prefix>.filtered.tsv    只保留 HIGH / VERY_HIGH 的边界

置信度规则（只看 CTCF / RAD21 / SMC3 命中 + support_tool_count）：
    VERY_HIGH: 支持工具 >=3 且 CTCF、RAD21、SMC3 都命中
    HIGH:      (支持工具>=2 且 CTCF 命中 且 RAD21 或 SMC3 命中)
               或 (支持工具>=3 且 RAD21、SMC3 都命中)
    其余为 LOW

H3K4me3/H3K27ac/POLR2A、H3K4me1/H3K36me3：只加覆盖标识，不参与置信度判断。
H3K9me3/EZH2：额外算边界两侧覆盖差异（transition），同样只做标识，不参与判断。
"""
import argparse
import os

import numpy as np
import pandas as pd

PEAK_DIR = 'data/proteins'  # 固定，不做成参数
CORE = ['CTCF', 'RAD21', 'SMC3']
AUX = ['H3K4me3', 'H3K27ac', 'POLR2A', 'H3K4me1', 'H3K36me3']
BROAD = ['H3K9me3', 'EZH2']
PROTEINS = CORE + AUX + BROAD


def load_peaks(protein):
    path = f'{PEAK_DIR}/{protein}.bed'
    if not os.path.exists(path):
        print(f'⚠️ 缺少 {protein}: {path}')
        return {}
    peaks = pd.read_csv(path, sep='\t', header=None, usecols=[0, 1, 2],
                         names=['chrom', 'start', 'end'])
    return {c: g[['start', 'end']].to_numpy() for c, g in peaks.groupby('chrom')}


def covered(peaks_by_chrom, chrom, start, end):
    """[start, end) 是否与该蛋白任意一个 peak 重叠。"""
    peaks = peaks_by_chrom.get(chrom)
    if peaks is None or len(peaks) == 0:
        return False
    return bool(((peaks[:, 0] < end) & (peaks[:, 1] > start)).any())


def coverage_frac(peaks_by_chrom, chrom, start, end):
    """[start, end) 被该蛋白 peak 覆盖的比例（粗略估计，不做区间合并）。"""
    peaks = peaks_by_chrom.get(chrom)
    if peaks is None or len(peaks) == 0 or end <= start:
        return 0.0
    lo = np.maximum(peaks[:, 0], start)
    hi = np.minimum(peaks[:, 1], end)
    covered_bp = np.clip(hi - lo, 0, None).sum()
    return float(min(covered_bp, end - start) / (end - start))


def annotate(df, peak_index):
    for p in PROTEINS:
        df[f'{p}_hit'] = [int(covered(peak_index[p], r.chrom, r.start, r.end)) for r in df.itertuples()]

    df['protein_hit_names'] = df[[f'{p}_hit' for p in PROTEINS]].apply(
        lambda row: ','.join(p for p, v in zip(PROTEINS, row) if v), axis=1)

    for p in BROAD:
        width = df['end'] - df['start']
        left = [coverage_frac(peak_index[p], r.chrom, r.start - w, r.start) for r, w in zip(df.itertuples(), width)]
        right = [coverage_frac(peak_index[p], r.chrom, r.end, r.end + w) for r, w in zip(df.itertuples(), width)]
        df[f'{p}_transition'] = np.round(np.abs(np.array(left) - np.array(right)), 3)
    return df


def assign_confidence(df):
    n = df['support_tool_count']
    ctcf, rad21, smc3 = df['CTCF_hit'] == 1, df['RAD21_hit'] == 1, df['SMC3_hit'] == 1

    very_high = (n >= 3) & ctcf & rad21 & smc3
    high = ((n >= 2) & ctcf & (rad21 | smc3)) | ((n >= 3) & rad21 & smc3)

    df['confidence'] = 'LOW'
    df.loc[high, 'confidence'] = 'HIGH'
    df.loc[very_high, 'confidence'] = 'VERY_HIGH'
    return df


def main(candidates_path, out_prefix):
    df = pd.read_csv(candidates_path, sep='\t')
    peak_index = {p: load_peaks(p) for p in PROTEINS}

    df = assign_confidence(annotate(df, peak_index))

    annotated_path = f'{out_prefix}.annotated.tsv'
    filtered_path = f'{out_prefix}.filtered.tsv'
    df.to_csv(annotated_path, sep='\t', index=False)
    df[df['confidence'].isin(['HIGH', 'VERY_HIGH'])].to_csv(filtered_path, sep='\t', index=False)

    print(f'✅ 全部候选 {len(df)} 个，标识结果 -> {annotated_path}')
    kept = df["confidence"].isin(["HIGH", "VERY_HIGH"]).sum()
    print(f'✅ 高置信 {kept} 个，过滤结果 -> {filtered_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidates', required=True)
    parser.add_argument('--out-prefix', default='boundary_protein')
    args = parser.parse_args()
    main(args.candidates, args.out_prefix)

