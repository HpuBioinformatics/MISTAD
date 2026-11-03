#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
5维 MISTAD regression 模型批量预测 + Wilcoxon TAD过滤。
"""

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 配置
# ============================================================

FEATURE_NAMES = [
    "insulation_100kb",
    "insulation_150kb",
    "directionality_index",
    "up_mean",
    "down_mean",
]
INPUT_DIM = 5
VALID_TASK_MODES = {"multitask", "classification", "regression"}

# 仓库根目录（脚本位于 <repo>/prediction/ 下），默认路径均为仓库内相对路径
REPO_ROOT = Path(__file__).resolve().parents[1]

# 默认指向仓库内训练输出；可用 --training-root / --output-root / --hic-dir 覆盖
TRAINING_ROOT = REPO_ROOT / "outputs" / "model_reg_v1"
OUTPUT_ROOT = REPO_ROOT / "prediction_outputs" / "TADs_reg"
HIC_DIR = REPO_ROOT / "prediction_outputs" / "hic"

DEFAULT_CHROMS = ["chr20", "chr21", "chr22"]

PROFILES = ["reg_base"]

# 只检查能从实验名明确确定的字段
EXPECTED = {
    "reg_base": {"task_mode": "regression"},
}


# ============================================================
# 模型
# ============================================================

def gn_groups(channels, preferred=8):
    for g in (preferred, 4, 2, 1):
        if channels % g == 0:
            return g
    return 1


def freeze_if(module, trainable):
    if not trainable:
        for p in module.parameters():
            p.requires_grad = False
    return module


class SEBlock1D(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, hidden, 1),
            nn.GELU(),
            nn.Conv1d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.net(x)


class DilatedResidualBlock(nn.Module):
    def __init__(self, channels, dropout=0.25, dilation=1, use_se=True):
        super().__init__()
        g = gn_groups(channels)
        self.use_se = use_se

        self.conv1 = nn.Conv1d(
            channels, channels, 3,
            padding=dilation,
            dilation=dilation,
        )
        self.norm1 = nn.GroupNorm(g, channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(channels, channels, 1)
        self.norm2 = nn.GroupNorm(g, channels)
        self.se = freeze_if(SEBlock1D(channels), use_se)

    def forward(self, x):
        h = self.norm1(self.conv1(x))
        h = self.dropout(F.gelu(h))
        h = self.norm2(self.conv2(h))
        if self.use_se:
            h = self.se(h)
        return F.gelu(x + h)


class WindowPositionalEncoding(nn.Module):
    def __init__(self, channels, max_len=64):
        super().__init__()

        pe = torch.zeros(max_len, channels)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)

        div = torch.exp(
            torch.arange(0, channels, 2, dtype=torch.float32)
            * (-math.log(10000.0) / channels)
        )

        pe[:, 0::2] = torch.sin(pos * div)
        ncos = pe[:, 1::2].shape[1]
        pe[:, 1::2] = torch.cos(pos * div[:ncos])

        self.register_buffer(
            "pe",
            pe.transpose(0, 1).unsqueeze(0),
        )

    def forward(self, x):
        if x.shape[-1] > self.pe.shape[-1]:
            raise ValueError("window超过positional encoding长度")
        return x + self.pe[:, :, :x.shape[-1]]


class WindowSelfAttention(nn.Module):
    def __init__(self, channels, num_heads=4, dropout=0.1):
        super().__init__()

        heads = num_heads if channels % num_heads == 0 else 1

        self.norm = nn.GroupNorm(
            gn_groups(channels),
            channels,
        )
        self.attn = nn.MultiheadAttention(
            channels,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.norm(x).transpose(1, 2)
        h, _ = self.attn(h, h, h, need_weights=False)
        return x + self.dropout(h.transpose(1, 2))


def make_prediction_head(hidden_dim, dropout):
    return nn.Sequential(
        nn.Conv1d(hidden_dim, hidden_dim // 2, 1),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Conv1d(hidden_dim // 2, 1, 1),
    )


class MultiTaskDilatedCNN(nn.Module):
    def __init__(
        self,
        input_dim=5,
        hidden_dim=64,
        dropout=0.25,
        attn_heads=4,
        use_pos=True,
        use_attn=True,
        use_se=True,
        dilations=(1, 2, 4),
        use_refine=True,
        task_mode="regression",
        align_stochastic_calls=False,
    ):
        super().__init__()

        if task_mode not in VALID_TASK_MODES:
            raise ValueError(task_mode)

        self.use_pos = bool(use_pos)
        self.use_attn = bool(use_attn)
        self.use_refine = bool(use_refine)
        self.task_mode = task_mode
        self.align_stochastic_calls = bool(align_stochastic_calls)

        self.has_cls_task = task_mode in {
            "multitask", "classification"
        }
        self.has_reg_task = task_mode in {
            "multitask", "regression"
        }

        g = gn_groups(hidden_dim)

        self.stem = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(g, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.blocks = nn.ModuleList([
            DilatedResidualBlock(
                hidden_dim,
                dropout,
                int(d),
                use_se,
            )
            for d in dilations
        ])

        self.pos_enc = WindowPositionalEncoding(hidden_dim)

        self.window_attn = WindowSelfAttention(
            hidden_dim,
            attn_heads,
            dropout,
        )

        self.cls_refine = DilatedResidualBlock(
            hidden_dim, dropout, 1, use_se
        )
        self.reg_refine = DilatedResidualBlock(
            hidden_dim, dropout, 1, use_se
        )

        self.cls_head = make_prediction_head(hidden_dim, dropout)
        self.reg_head = make_prediction_head(hidden_dim, dropout)

        freeze_if(self.window_attn, self.use_attn)
        freeze_if(
            self.cls_refine,
            self.use_refine and self.has_cls_task,
        )
        freeze_if(
            self.reg_refine,
            self.use_refine and self.has_reg_task,
        )
        freeze_if(self.cls_head, self.has_cls_task)
        freeze_if(self.reg_head, self.has_reg_task)

    def forward(self, x):
        h = self.stem(x.transpose(1, 2))

        for block in self.blocks:
            h = block(h)

        h = self.pos_enc(h) if self.use_pos else h

        if self.use_attn:
            h = self.window_attn(h)

        h_cls = (
            self.cls_refine(h)
            if self.has_cls_task and self.use_refine
            else h
        )

        h_reg = (
            self.reg_refine(h)
            if self.has_reg_task and self.use_refine
            else h
        )

        zeros = h.new_zeros(
            (h.shape[0], h.shape[-1])
        )

        cls_logits = (
            self.cls_head(h_cls).squeeze(1)
            if self.has_cls_task
            else zeros
        )

        reg_logits = (
            self.reg_head(h_reg).squeeze(1)
            if self.has_reg_task
            else zeros
        )

        return {
            "cls": cls_logits,
            "reg": reg_logits,
        }


def build_model(cfg, input_dim):
    return MultiTaskDilatedCNN(
        input_dim=input_dim,
        hidden_dim=cfg["hidden_dim"],
        dropout=cfg["dropout"],
        attn_heads=cfg["attn_heads"],
        use_pos=cfg["use_pos"],
        use_attn=cfg["use_attn"],
        use_se=cfg["use_se"],
        dilations=cfg["dilations"],
        use_refine=cfg["use_refine"],
        task_mode=cfg["task_mode"],
        align_stochastic_calls=cfg.get(
            "align_stochastic_calls", False
        ),
    )


# ============================================================
# checkpoint
# ============================================================

def torch_load(path, device="cpu"):
    try:
        return torch.load(
            path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        return torch.load(path, map_location=device)


def state_dict_hash(state_dict):
    h = hashlib.sha256()

    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()

        h.update(name.encode())
        h.update(str(tensor.dtype).encode())
        h.update(str(tuple(tensor.shape)).encode())
        h.update(tensor.numpy().tobytes())

    return h.hexdigest()


def same_value(a, b):
    if isinstance(a, float) or isinstance(b, float):
        try:
            return abs(float(a) - float(b)) <= 1e-12
        except Exception:
            return False
    return a == b


def validate_checkpoint(profile, path):
    ckpt = torch_load(path, "cpu")

    if not isinstance(ckpt, dict):
        raise RuntimeError("checkpoint不是dict")

    errors = []

    saved_profile = ckpt.get("profile_name")
    if saved_profile is not None and saved_profile != profile:
        errors.append(
            f"profile_name={saved_profile!r}"
        )

    cfg = ckpt.get("config")
    if not isinstance(cfg, dict):
        errors.append("缺少config")
        cfg = {}

    input_dim = int(ckpt.get("input_dim", -1))
    if input_dim != INPUT_DIM:
        errors.append(
            f"input_dim={input_dim}, 期望={INPUT_DIM}"
        )

    names = ckpt.get("feature_names")
    if list(names or []) != FEATURE_NAMES:
        errors.append(
            f"feature_names={names!r}"
        )

    for key, expected in EXPECTED[profile].items():
        if key not in cfg:
            errors.append(f"config缺少{key}")
        elif not same_value(cfg[key], expected):
            errors.append(
                f"{key}={cfg[key]!r}, 期望={expected!r}"
            )

    best = ckpt.get("best_record")
    if not isinstance(best, dict):
        errors.append("缺少best_record")
    elif "threshold" not in best:
        errors.append("best_record缺少threshold")

    state = ckpt.get("model_state_dict")
    if not isinstance(state, dict):
        errors.append("缺少model_state_dict")

    saved_hash = ckpt.get("best_state_hash")
    if saved_hash and isinstance(state, dict):
        if state_dict_hash(state) != saved_hash:
            errors.append("best_state_hash校验失败")

    if not errors:
        try:
            model = build_model(cfg, INPUT_DIM)
            model.load_state_dict(state, strict=True)
        except Exception as exc:
            errors.append(
                f"strict=True加载失败: {exc}"
            )

    if errors:
        raise RuntimeError(
            "\n  - ".join(
                [f"{profile}: checkpoint校验失败"] + errors
            )
        )

    return ckpt


def discover_checkpoint(profile, training_root, seed):
    seed_dir = (
        Path(training_root)
        / "runs"
        / profile
        / f"seed_{seed}"
    )

    candidates = list(
        seed_dir.glob("*/best_model.pth")
    )

    direct = seed_dir / "best_model.pth"
    if direct.exists():
        candidates.append(direct)

    candidates = sorted(
        set(candidates),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not candidates:
        raise FileNotFoundError(
            f"{profile}: 未找到checkpoint: {seed_dir}"
        )

    valid = []
    invalid = []

    for path in candidates:
        try:
            ckpt = validate_checkpoint(profile, path)
            valid.append((path, ckpt))
        except Exception as exc:
            invalid.append((path, str(exc)))

    if not valid:
        details = "\n".join(
            f"{p}: {e}" for p, e in invalid
        )
        raise RuntimeError(
            f"{profile}: 无有效checkpoint\n{details}"
        )

    path, ckpt = valid[0]

    if len(valid) > 1:
        print(
            f"[WARN] {profile}: "
            f"{len(valid)}个有效run，使用最新"
        )

    return path, ckpt


def validate_all(profiles, training_root, seed):
    loaded = {}

    print("=" * 80)
    print("校验全部5维 regression checkpoint")
    print("=" * 80)

    for profile in profiles:
        path, ckpt = discover_checkpoint(
            profile,
            training_root,
            seed,
        )

        loaded[profile] = {
            "path": path,
            "checkpoint": ckpt,
        }

        print(
            f"[OK] {profile:<35} "
            f"th={float(ckpt['best_record']['threshold']):.4f}"
        )
        print(f"     {path}")

    print("=" * 80)

    return loaded


# ============================================================
# Hi-C / Wilcoxon
# ============================================================

def center_logits(values, center_idx):
    return (
        values[:, center_idx]
        if values.dim() > 1
        else values
    )


def upper_triangle_values(block):
    block = np.asarray(block, dtype=np.float64)
    x = block[np.triu_indices_from(block, k=1)]
    return x[np.isfinite(x)]


def scale_matrix_diagonals(hic, max_distance, eps=1e-6):
    matrix = np.asarray(hic, dtype=np.float64)
    n = matrix.shape[0]

    scaled = np.full(
        (n, n),
        np.nan,
        dtype=np.float64,
    )

    for offset in range(
        min(int(max_distance), n - 1) + 1
    ):
        idx = np.arange(n - offset)
        values = matrix[idx, idx + offset]

        finite = values[np.isfinite(values)]
        positive = finite[finite > 0]
        ref = positive if positive.size >= 8 else finite

        if ref.size == 0:
            continue

        median = float(np.median(ref))
        mad = float(
            np.median(np.abs(ref - median))
        )

        scale = 1.4826 * mad

        if not np.isfinite(scale) or scale < eps:
            scale = float(np.std(ref))

        if not np.isfinite(scale) or scale < eps:
            scale = 1.0

        z = (values - median) / scale

        scaled[idx, idx + offset] = z
        scaled[idx + offset, idx] = z

    np.fill_diagonal(scaled, 0.0)
    return scaled


def find_gap_regions(hic, window, min_valid_fraction):
    matrix = np.asarray(hic)
    n = matrix.shape[0]
    gaps = set()

    for idx in range(n):
        start = max(0, idx - window)
        end = min(n, idx + window + 1)

        row = matrix[idx, start:end]

        valid = int(
            (
                np.isfinite(row)
                & (row > 0)
            ).sum()
        )

        fraction = valid / max(len(row) - 1, 1)

        if fraction < min_valid_fraction:
            gaps.add(idx)

    return gaps


def compute_pvalue_for_bin(
    scaled,
    boundary_bin,
    window,
    min_values,
):
    from scipy.stats import norm, ranksums

    n = scaled.shape[0]
    b = int(boundary_bin)

    left = np.arange(max(0, b - window), b)
    right = np.arange(
        b + 1,
        min(n, b + window + 1),
    )

    if len(left) < 2 or len(right) < 2:
        return {
            "pvalue": 1.0,
            "effect": 0.0,
            "cross_median": np.nan,
            "within_median": np.nan,
        }

    cross = scaled[
        np.ix_(left, right)
    ].reshape(-1)

    cross = cross[np.isfinite(cross)]

    within = np.concatenate([
        upper_triangle_values(
            scaled[np.ix_(left, left)]
        ),
        upper_triangle_values(
            scaled[np.ix_(right, right)]
        ),
    ])

    cross_med = (
        float(np.median(cross))
        if cross.size else np.nan
    )
    within_med = (
        float(np.median(within))
        if within.size else np.nan
    )

    if (
        cross.size < min_values
        or within.size < min_values
    ):
        return {
            "pvalue": 1.0,
            "effect": 0.0,
            "cross_median": cross_med,
            "within_median": within_med,
        }

    try:
        result = ranksums(
            cross,
            within,
            alternative="less",
        )
        pvalue = float(result.pvalue)
    except TypeError:
        result = ranksums(cross, within)
        pvalue = float(norm.cdf(result.statistic))

    return {
        "pvalue": pvalue,
        "effect": float(within_med - cross_med),
        "cross_median": cross_med,
        "within_median": within_med,
    }


def prepare_hic_cache(chroms, hic_dir, args):
    cache = {}

    for chrom in chroms:
        path = Path(hic_dir) / f"{chrom}_dense.npy"

        if not path.exists():
            raise FileNotFoundError(path)

        hic = np.load(path)

        if hic.ndim != 2 or hic.shape[0] != hic.shape[1]:
            raise ValueError(
                f"{path}: Hi-C必须为方阵"
            )

        cache[chrom] = {
            "hic": hic,
            "scaled": scale_matrix_diagonals(
                hic,
                2 * args.wilcoxon_window + 2,
            ),
            "gaps": find_gap_regions(
                hic,
                args.wilcoxon_window,
                args.gap_min_valid_fraction,
            ),
            "wilcoxon": {},
        }

        print(
            f"{chrom}: bins={hic.shape[0]:,}"
        )

    return cache


def get_wilcoxon_result(cache, boundary_bin, args):
    b = int(boundary_bin)

    if b not in cache["wilcoxon"]:
        cache["wilcoxon"][b] = compute_pvalue_for_bin(
            cache["scaled"],
            b,
            args.wilcoxon_window,
            args.wilcoxon_min_values,
        )

    return cache["wilcoxon"][b]


def choose_representative(cluster):
    return min(
        cluster,
        key=lambda r: (
            -r["reg_prob"],
            -r["eval_prob"],
            r["pvalue"]
            if np.isfinite(r["pvalue"])
            else 1.0,
            r["center_bin"],
        ),
    )


# ============================================================
# 输出
# ============================================================

def clear_prediction_outputs(base):
    for name in ("predictions", "25KB_TAD"):
        path = base / name
        if path.exists():
            shutil.rmtree(path)

        path.mkdir(parents=True, exist_ok=True)


def write_prediction_outputs(
    profile,
    predictions,
    args,
    chroms,
    hic_cache,
    output_root,
):
    base = Path(output_root) / profile
    pred_dir = base / "predictions"
    tad_dir = base / "25KB_TAD"

    pred_lines = [
        "chrom\tcenter_bin\tstart_bp\tcls_prob\treg_prob\t"
        "eval_prob\tthreshold\tcandidate\n"
    ]

    boundary_lines = [
        "chrom\tcenter_bin\tstart_bp\teval_prob\tcls_prob\t"
        "reg_prob\tpvalue\teffect\tcross_median\twithin_median\t"
        "wilcoxon_pass\tselected_boundary\n"
    ]

    summary = {}
    total_tads = 0

    for chrom in chroms:
        pred = predictions[chrom]
        rows = []

        for i, prob in enumerate(pred["eval_probs"]):
            if pred["starts"] is not None:
                start_bp = int(pred["starts"][i])
            else:
                start_bp = (
                    i + pred["window_size"] // 2
                ) * args.resolution_kb * 1000

            center_bin = (
                start_bp
                // (args.resolution_kb * 1000)
            )

            row = {
                "center_bin": center_bin,
                "start_bp": start_bp,
                "cls_prob": float(pred["cls_probs"][i]),
                "reg_prob": float(pred["reg_probs"][i]),
                "eval_prob": float(prob),
                "threshold": float(pred["threshold"]),
                "pvalue": np.nan,
                "effect": np.nan,
                "cross_median": np.nan,
                "within_median": np.nan,
                "wilcoxon_pass": False,
                "selected_boundary": False,
            }

            rows.append(row)

            pred_lines.append(
                f"{chrom}\t{center_bin}\t{start_bp}\t"
                f"{row['cls_prob']:.9g}\t"
                f"{row['reg_prob']:.9g}\t"
                f"{row['eval_prob']:.9g}\t"
                f"{row['threshold']:.9g}\t"
                f"{int(row['eval_prob'] > row['threshold'])}\n"
            )

        positives = sorted(
            [
                r for r in rows
                if r["eval_prob"] > r["threshold"]
            ],
            key=lambda r: r["center_bin"],
        )

        clusters = []

        for row in positives:
            if (
                not clusters
                or row["center_bin"]
                - clusters[-1][-1]["center_bin"]
                > args.cluster_max_gap
            ):
                clusters.append([row])
            else:
                clusters[-1].append(row)

        chrom_hic = hic_cache[chrom]
        filtered_clusters = []

        for cluster in clusters:
            kept = []

            for row in cluster:
                b = row["center_bin"]

                if (
                    b < 0
                    or b >= chrom_hic["hic"].shape[0]
                    or b in chrom_hic["gaps"]
                ):
                    row["pvalue"] = 1.0
                    row["effect"] = 0.0
                    continue

                test = get_wilcoxon_result(
                    chrom_hic,
                    b,
                    args,
                )

                row.update(test)

                passed = (
                    row["pvalue"] < args.pvalue_cutoff
                )

                if args.require_positive_effect:
                    passed &= row["effect"] > 0

                row["wilcoxon_pass"] = bool(passed)

                if passed:
                    kept.append(row)

            if kept:
                filtered_clusters.append(kept)

        representatives = []

        for cluster in filtered_clusters:
            rep = choose_representative(cluster)
            rep["selected_boundary"] = True
            representatives.append(rep)

        boundaries = sorted({
            r["center_bin"]
            for r in representatives
        })

        tads = list(
            zip(boundaries[:-1], boundaries[1:])
        )

        total_tads += len(tads)

        with open(
            tad_dir / f"{chrom}.txt",
            "w",
            encoding="utf-8",
        ) as f:
            for start, end in tads:
                f.write(f"{start + 1} {end}\n")

        for row in positives:
            boundary_lines.append(
                f"{chrom}\t"
                f"{row['center_bin']}\t"
                f"{row['start_bp']}\t"
                f"{row['eval_prob']:.9g}\t"
                f"{row['cls_prob']:.9g}\t"
                f"{row['reg_prob']:.9g}\t"
                f"{row['pvalue']:.9g}\t"
                f"{row['effect']:.9g}\t"
                f"{row['cross_median']:.9g}\t"
                f"{row['within_median']:.9g}\t"
                f"{int(row['wilcoxon_pass'])}\t"
                f"{int(row['selected_boundary'])}\n"
            )

        passed = sum(
            r["wilcoxon_pass"]
            for r in positives
        )

        summary[chrom] = {
            "bins": len(rows),
            "candidates": len(positives),
            "clusters": len(clusters),
            "wilcoxon_pass": passed,
            "boundaries": len(boundaries),
            "tads": len(tads),
        }

        print(
            f"  {chrom}: candidates={len(positives)}, "
            f"pass={passed}, boundaries={len(boundaries)}, "
            f"TADs={len(tads)}"
        )

    (pred_dir / "predictions.txt").write_text(
        "".join(pred_lines),
        encoding="utf-8",
    )

    (
        pred_dir / "boundaries_wilcoxon.txt"
    ).write_text(
        "".join(boundary_lines),
        encoding="utf-8",
    )

    return summary, total_tads


# ============================================================
# 预测
# ============================================================

@torch.no_grad()
def predict_one(
    profile,
    checkpoint_path,
    checkpoint,
    args,
    chroms,
    hic_cache,
    output_root,
):
    cfg = checkpoint["config"]

    input_dim = int(checkpoint["input_dim"])
    center_idx = int(checkpoint["center_idx"])

    threshold = float(
        checkpoint["best_record"]["threshold"]
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = build_model(
        cfg,
        input_dim,
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model.eval()

    data_dir = (
        Path(args.data_dir)
        if args.data_dir
        else Path(cfg["data_base"]) / cfg["dataset"]
    )

    window_size = int(cfg["window_size"])
    predictions = {}

    print(
        f"\n[{profile}] threshold={threshold:.4f}, "
        f"device={device}"
    )
    print(f"checkpoint: {checkpoint_path}")

    for chrom in chroms:
        npz_path = (
            data_dir
            / f"{chrom}_ws{window_size}.npz"
        )

        if not npz_path.exists():
            raise FileNotFoundError(npz_path)

        with np.load(
            npz_path,
            allow_pickle=True,
        ) as data:
            features = data["features"].astype(
                np.float32
            )

            starts = (
                data["starts"].astype(np.int64)
                if "starts" in data.files
                else None
            )

        if (
            features.ndim != 3
            or features.shape[1:]
            != (window_size, INPUT_DIM)
        ):
            raise ValueError(
                f"{chrom}: features={features.shape}"
            )

        cls_batches = []
        reg_batches = []

        for start in range(
            0,
            len(features),
            args.batch_size,
        ):
            batch = torch.from_numpy(
                features[
                    start:start + args.batch_size
                ]
            ).to(device)

            outputs = model(batch)

            cls_batches.append(
                torch.sigmoid(
                    center_logits(
                        outputs["cls"],
                        center_idx,
                    )
                )
            )

            reg_batches.append(
                torch.sigmoid(
                    center_logits(
                        outputs["reg"],
                        center_idx,
                    )
                )
            )

        cls_probs = (
            torch.cat(cls_batches)
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        reg_probs = (
            torch.cat(reg_batches)
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        # 新版全部为 regression
        eval_probs = reg_probs

        predictions[chrom] = {
            "cls_probs": cls_probs,
            "reg_probs": reg_probs,
            "eval_probs": eval_probs,
            "threshold": threshold,
            "starts": starts,
            "window_size": window_size,
        }

        print(
            f"  {chrom}: samples={len(features):,}, "
            f"candidates="
            f"{int((eval_probs > threshold).sum()):,}"
        )

    base = Path(output_root) / profile
    clear_prediction_outputs(base)

    summary, total_tads = write_prediction_outputs(
        profile,
        predictions,
        args,
        chroms,
        hic_cache,
        output_root,
    )

    return {
        "profile": profile,
        "checkpoint": str(checkpoint_path),
        "output": str(base),
        "threshold": threshold,
        "summary": summary,
        "total_tads": total_tads,
        "status": "ok",
    }


# ============================================================
# CLI
# ============================================================

def parse_list(text):
    return [
        x.strip()
        for x in str(text).split(",")
        if x.strip()
    ]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--training-root",
        default=str(TRAINING_ROOT),
    )
    parser.add_argument(
        "--output-root",
        default=str(OUTPUT_ROOT),
    )
    parser.add_argument(
        "--hic-dir",
        default=str(HIC_DIR),
    )
    parser.add_argument(
        "--data-dir",
        default=None,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1001,
    )
    parser.add_argument(
        "--profiles",
        default=",".join(PROFILES),
    )
    parser.add_argument(
        "--chroms",
        default=",".join(DEFAULT_CHROMS),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--resolution-kb",
        type=int,
        default=25,
    )
    parser.add_argument(
        "--cluster-max-gap",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--wilcoxon-window",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--pvalue-cutoff",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--wilcoxon-min-values",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--gap-min-valid-fraction",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--no-positive-effect",
        dest="require_positive_effect",
        action="store_false",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
    )

    parser.set_defaults(
        require_positive_effect=True
    )

    return parser.parse_args()


def main():
    args = parse_args()

    profiles = parse_list(args.profiles)
    chroms = parse_list(args.chroms)

    unknown = [
        p for p in profiles
        if p not in EXPECTED
    ]

    if unknown:
        raise ValueError(
            f"未知profile: {unknown}"
        )

    loaded = validate_all(
        profiles,
        args.training_root,
        args.seed,
    )

    if args.validate_only:
        print("checkpoint校验完成")
        return

    hic_cache = prepare_hic_cache(
        chroms,
        args.hic_dir,
        args,
    )

    output_root = Path(args.output_root)
    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = []

    for profile in profiles:
        item = loaded[profile]

        report.append(
            predict_one(
                profile,
                item["path"],
                item["checkpoint"],
                args,
                chroms,
                hic_cache,
                output_root,
            )
        )

    report_path = (
        output_root
        / "batch_prediction_report.json"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\n" + "=" * 80)
    print("全部5维 regression 预测完成")
    print(f"报告: {report_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
