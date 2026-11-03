#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MISTAD 5维纯回归模型训练。

模型设定（reg_base）：
    use_pos=False, use_attn=False, task_mode="regression",
    ambiguous_weight=0.35，5维Hi-C输入。

注意：
1) 只使用 regression head 做验证、最佳epoch选择、阈值搜索和测试。
2) 训练损失、5维输入、数据划分、sampler、canonical initialization、
   optimizer 和速度优化逻辑保持原代码风格。
"""

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse, copy, csv, hashlib, json, math, platform, random, statistics, sys, time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

MODEL_SCHEMA_VERSION = "mistad_5feat_fixed_registration_v1"
TRAINING_SCHEMA_VERSION = "mistad_5feat_reg_twostage_v1"

# 仓库根目录（脚本位于 <repo>/training/ 下），数据与输出默认放在仓库内相对路径
REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_HEAD_POLICY = "regression_only"
VALID_TASK_MODES = {"multitask", "classification", "regression"}

FEATURE_NAMES = [
    "insulation_100kb", "insulation_150kb", "directionality_index",
    "up_mean", "down_mean",
]
LOSS_KEYS = (
    "loss", "cls_loss", "reg_loss", "cls_center_loss",
    "cls_aux_loss", "reg_center_loss", "reg_aux_loss", "peak_loss",
)

CFG = {
    "dataset": "ws21",
    "data_base": REPO_ROOT / "datasets",
    "output_root": REPO_ROOT / "outputs" / "model_reg_v1",
    "train_chrs": [f"chr{i}" for i in range(1, 18)],
    "val_chrs": ["chr18", "chr19"],
    "test_chrs": ["chr20", "chr21", "chr22"],
    "excluded_train_chrs": ["chr7", "chr10"],
    "window_size": 21, "seed": 1001, "epochs": 80, "batch_size": 256,
    "num_workers": 0, "lr": 1e-4, "weight_decay": 5e-4,
    "patience": 8, "scheduler_patience": 2, "grad_clip": 1.0,
    "deterministic": False, "align_stochastic_calls": False,
    "finite_check_interval": 100,
    "hidden_dim": 64, "dropout": 0.25, "attn_heads": 4, "dilations": [1, 2, 4],
    "use_pos": True, "use_attn": True, "use_se": True, "use_refine": True,
    "task_mode": "multitask", "eval_head_policy": EVAL_HEAD_POLICY,
    "use_peak_loss": True,
    "cls_weight": 1.0, "reg_weight": 0.25,
    "focal_alpha": 0.75, "focal_gamma": 1.5,
    "tversky_alpha": 0.25, "tversky_beta": 0.70,
    "tversky_gamma": 0.75, "tversky_weight": 0.70,
    "weighted_alpha": 1.0, "peak_emphasis_power": 1.5,
    "huber_beta": 0.2, "ambiguous_weight": 0.35, "ambiguous_low": 0.30,
    "aux_cls_weight": 0.5, "aux_reg_weight": 0.2,
    "peak_weight": 0.2, "peak_temperature": 1.0,
    "sigma": 2.5, "eval_positive_radius": 1, "eval_label_threshold": None,
    "hard_recall_soft_threshold": 0.99,
    "sampler_type": "tempered", "sampler_gamma": 0.35,
    "sampler_min_weight": 0.1, "sampler_max_weight": 10.0,
    "threshold_min": 0.05, "threshold_max": 0.95, "threshold_step": 0.01,
    "best_metric": "auprc", "tie_tolerance": 1e-4,
    "hard_recall_drop_tolerance": 0.05,
    "evaluate_test": True, "save": True,
}

PROFILE_SPECS = OrderedDict({
    "reg_base": {
        "baseline": "reg",
        "overrides": {},
        "allowed_changes": set(),
        "description": "纯回归基线：no-pos/no-attn + regression-only，ambiguous_weight=0.35。",
    },
})

NON_ABLATION_KEYS = {
    "data_base", "output_root", "save", "evaluate_test",
    "excluded_train_chrs", "finite_check_interval", "hard_recall_drop_tolerance",
    "model_schema_version", "training_schema_version", "code_sha256",
    "profile_name", "display_profile", "experiment_round", "experiment_batch", "description",
}

def to_builtin(x):
    if isinstance(x, dict): return {str(k): to_builtin(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)): return [to_builtin(v) for v in x]
    if isinstance(x, Path): return str(x)
    if isinstance(x, np.integer): return int(x)
    if isinstance(x, np.floating): return float(x)
    if isinstance(x, np.ndarray): return x.tolist()
    if torch.is_tensor(x): return x.detach().cpu().tolist()
    return x

def save_json(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_builtin(payload), f, indent=2, ensure_ascii=False, sort_keys=True)

def load_json(path):
    with open(path, "r", encoding="utf-8") as f: return json.load(f)

def write_csv(path, rows):
    if not rows: return
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fields, seen = [], set()
    for row in rows:
        for k in row:
            if k not in seen: seen.add(k); fields.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

def torch_load(path, map_location="cpu"):
    try: return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError: return torch.load(path, map_location=map_location)

def current_code_sha256():
    try: return hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()
    except Exception: return "unavailable"

def environment_fingerprint():
    return {
        "python": sys.version, "platform": platform.platform(),
        "torch": torch.__version__, "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(), "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
    }

def set_global_determinism(seed, deterministic=False):
    seed = int(seed)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True, warn_only=False)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.use_deterministic_algorithms(False)

def reset_training_rng(seed):
    seed = int(seed) + 1000
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def seed_worker(_):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed); np.random.seed(seed)

def state_dict_hash(state):
    d = hashlib.sha256()
    for name in sorted(state):
        t = state[name].detach().cpu().contiguous()
        d.update(name.encode()); d.update(str(t.dtype).encode())
        d.update(str(tuple(t.shape)).encode()); d.update(t.numpy().tobytes())
    return d.hexdigest()

def json_hash(payload):
    raw = json.dumps(to_builtin(payload), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()

def gaussian_value(distance, sigma):
    sigma = float(sigma)
    if sigma <= 0: raise ValueError("sigma must be positive")
    return math.exp(-(float(distance) ** 2) / (2 * sigma ** 2))

def compute_eval_label_threshold(sigma, radius=1):
    return (gaussian_value(radius, sigma) + gaussian_value(radius + 1, sigma)) / 2

def center_of(values, center_idx):
    if torch.is_tensor(values): return values[:, center_idx] if values.dim() > 1 else values
    values = np.asarray(values)
    return values[:, center_idx] if values.ndim > 1 else values

def parse_list(text, caster=str):
    return [caster(v.strip()) for v in str(text).split(",") if v.strip()]

def normalize_cfg(base):
    cfg = copy.deepcopy(base)
    cfg["data_base"], cfg["output_root"] = Path(cfg["data_base"]), Path(cfg["output_root"])
    cfg["task_mode"] = str(cfg["task_mode"]).lower()
    if cfg["task_mode"] not in VALID_TASK_MODES: raise ValueError(f"Bad task_mode: {cfg['task_mode']}")
    cfg["eval_head_policy"] = EVAL_HEAD_POLICY
    cfg["dilations"] = [int(v) for v in cfg["dilations"]]
    excluded = set(cfg.get("excluded_train_chrs", []))
    cfg["train_chrs"] = [c for c in cfg["train_chrs"] if c not in excluded]
    if cfg.get("eval_label_threshold") is None:
        cfg["eval_label_threshold"] = compute_eval_label_threshold(cfg["sigma"], cfg["eval_positive_radius"])
    cfg["eval_label_threshold"] = float(cfg["eval_label_threshold"])
    cfg["finite_check_interval"] = max(1, int(cfg.get("finite_check_interval", 100)))
    cfg["model_schema_version"], cfg["training_schema_version"] = MODEL_SCHEMA_VERSION, TRAINING_SCHEMA_VERSION
    cfg["code_sha256"] = current_code_sha256()
    return cfg

def signature_payload(cfg):
    ignore = {"output_root", "save", "profile_name", "display_profile", "experiment_round", "description"}
    return {k: to_builtin(v) for k, v in cfg.items() if k not in ignore}

def config_signature(cfg): return json_hash(signature_payload(cfg))[:12]

def config_diff(base, candidate):
    keys = (set(base) | set(candidate)) - NON_ABLATION_KEYS
    return {k: {"baseline": to_builtin(base.get(k)), "candidate": to_builtin(candidate.get(k))}
            for k in sorted(keys) if to_builtin(base.get(k)) != to_builtin(candidate.get(k))}

def reg_baseline_cfg(base):
    """
    reg_base:
        use_pos=False
        use_attn=False
        task_mode="regression"
        ambiguous_weight=0.35
    """
    cfg = copy.deepcopy(base)
    cfg.update({
        "use_pos": False,
        "use_attn": False,
        "use_se": True,
        "use_refine": True,
        "task_mode": "regression",
        "use_peak_loss": True,
        "aux_reg_weight": 0.2,
        "weighted_alpha": 1.0,
        "ambiguous_weight": 0.35,
    })
    return normalize_cfg(cfg)


def resolve_profile(name, base):
    if name not in PROFILE_SPECS:
        raise ValueError(f"Unknown profile: {name}")

    spec = PROFILE_SPECS[name]

    if "alias_of" in spec:
        return resolve_profile(spec["alias_of"], base)

    baseline_name = spec["baseline"]

    if baseline_name == "reg":
        baseline = reg_baseline_cfg(base)
    else:
        raise ValueError(f"{name}: unsupported baseline={baseline_name}")

    candidate = copy.deepcopy(baseline)
    candidate.update(copy.deepcopy(spec["overrides"]))
    candidate = normalize_cfg(candidate)

    actual = set(config_diff(baseline, candidate))
    expected = set(spec["allowed_changes"])

    if actual != expected:
        raise RuntimeError(
            f"{name} not clean: expected={sorted(expected)}, "
            f"actual={sorted(actual)}"
        )

    if candidate["task_mode"] != "regression":
        raise RuntimeError(
            f"{name}: task_mode must be regression, "
            f"got {candidate['task_mode']}"
        )

    return candidate, name


def validate_experiment_definitions(base):
    for name in PROFILE_SPECS:
        resolve_profile(name, base)


def find_first_npz(data_dir):
    files = sorted(Path(data_dir).glob("chr*_ws*.npz"))
    if not files: raise FileNotFoundError(f"No NPZ in {data_dir}")
    return files[0]

def infer_input_dim(data_dir):
    sample = find_first_npz(data_dir)
    with np.load(sample, allow_pickle=True) as d: shape = d["features"].shape
    if len(shape) != 3 or int(shape[-1]) != len(FEATURE_NAMES):
        raise ValueError(f"{sample}: features={shape}, expected [N,W,5]")
    return int(shape[-1])

class HICFeatureDataset(Dataset):
    def __init__(self, data_dir, chroms, window, input_dim):
        xs, ys, chrom_labels, files_used = [], [], [], []
        for chrom in chroms:
            files = sorted(Path(data_dir).glob(f"{chrom}_ws*.npz"))
            if not files: raise FileNotFoundError(f"Missing NPZ for {chrom}")
            for path in files:
                with np.load(path, allow_pickle=True) as d:
                    x = d["features"].astype(np.float32); y = d["soft_labels"].astype(np.float32)
                if x.ndim != 3 or x.shape[1:] != (window, input_dim):
                    raise ValueError(f"{path}: features={x.shape}, expected [N,{window},{input_dim}]")
                if len(x) != len(y): raise ValueError(f"{path}: length mismatch")
                xs.append(x); ys.append(y); chrom_labels.extend([chrom] * len(x)); files_used.append(str(path))
        self.features = np.concatenate(xs).astype(np.float32)
        self.soft_labels = np.concatenate(ys).astype(np.float32)
        self.sample_chroms = np.asarray(chrom_labels)
        self.source_files = files_used
        self.features_tensor = torch.from_numpy(self.features)
        self.soft_labels_tensor = torch.from_numpy(self.soft_labels)
    def __len__(self): return len(self.soft_labels)
    def __getitem__(self, i): return self.features_tensor[i], self.soft_labels_tensor[i]


# ============================================================
# Model
# ============================================================

def gn_groups(channels, preferred=8):
    for g in (preferred, 4, 2, 1):
        if channels % g == 0: return g
    return 1

def freeze_if(module, keep_trainable):
    if not keep_trainable:
        for p in module.parameters(): p.requires_grad = False
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
    def forward(self, x): return x * self.net(x)

class DilatedResidualBlock(nn.Module):
    def __init__(self, channels, dropout=0.25, dilation=1, use_se=True):
        super().__init__()
        self.use_se = bool(use_se); g = gn_groups(channels)
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.norm1 = nn.GroupNorm(g, channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(channels, channels, 1)
        self.norm2 = nn.GroupNorm(g, channels)
        self.se = freeze_if(SEBlock1D(channels), self.use_se)
    def forward(self, x):
        h = self.norm1(self.conv1(x))
        h = self.dropout(F.gelu(h))
        h = self.norm2(self.conv2(h))
        if self.use_se: h = self.se(h)
        return F.gelu(x + h)

class WindowPositionalEncoding(nn.Module):
    def __init__(self, channels, max_len=64):
        super().__init__()
        pe = torch.zeros(max_len, channels)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, channels, 2, dtype=torch.float32) * (-math.log(10000.0) / channels))
        pe[:, 0::2] = torch.sin(pos * div)
        n = pe[:, 1::2].shape[1]
        pe[:, 1::2] = torch.cos(pos * div[:n])
        self.register_buffer("pe", pe.transpose(0, 1).unsqueeze(0))
    def forward(self, x):
        if x.shape[-1] > self.pe.shape[-1]: raise ValueError("Window too long")
        return x + self.pe[:, :, :x.shape[-1]]

class WindowSelfAttention(nn.Module):
    def __init__(self, channels, num_heads=4, dropout=0.1):
        super().__init__()
        heads = num_heads if channels % num_heads == 0 else 1
        self.norm = nn.GroupNorm(gn_groups(channels), channels)
        self.attn = nn.MultiheadAttention(channels, heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        h = self.norm(x).transpose(1, 2)
        h, _ = self.attn(h, h, h, need_weights=False)
        return x + self.dropout(h.transpose(1, 2))

def make_prediction_head(hidden, dropout):
    return nn.Sequential(
        nn.Conv1d(hidden, hidden // 2, 1),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Conv1d(hidden // 2, 1, 1),
    )

class MultiTaskDilatedCNN(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64, dropout=0.25, attn_heads=4,
                 use_pos=True, use_attn=True, use_se=True, dilations=(1, 2, 4),
                 use_refine=True, task_mode="multitask", align_stochastic_calls=True):
        super().__init__()
        if task_mode not in VALID_TASK_MODES: raise ValueError(f"Unsupported task_mode: {task_mode}")
        self.use_pos, self.use_attn, self.use_refine = bool(use_pos), bool(use_attn), bool(use_refine)
        self.task_mode, self.align_stochastic_calls = task_mode, bool(align_stochastic_calls)
        self.has_cls_task = task_mode in {"multitask", "classification"}
        self.has_reg_task = task_mode in {"multitask", "regression"}
        g = gn_groups(hidden_dim)
        self.stem = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(g, hidden_dim), nn.GELU(), nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList([
            DilatedResidualBlock(hidden_dim, dropout, int(d), use_se) for d in dilations
        ])
        self.pos_enc = WindowPositionalEncoding(hidden_dim)
        self.window_attn = WindowSelfAttention(hidden_dim, attn_heads, dropout)
        self.cls_refine = DilatedResidualBlock(hidden_dim, dropout, 1, use_se)
        self.reg_refine = DilatedResidualBlock(hidden_dim, dropout, 1, use_se)
        self.cls_head = make_prediction_head(hidden_dim, dropout)
        self.reg_head = make_prediction_head(hidden_dim, dropout)

        freeze_if(self.window_attn, self.use_attn)
        freeze_if(self.cls_refine, self.use_refine and self.has_cls_task)
        freeze_if(self.reg_refine, self.use_refine and self.has_reg_task)
        freeze_if(self.cls_head, self.has_cls_task)
        freeze_if(self.reg_head, self.has_reg_task)

    def forward(self, x):
        h = self.stem(x.transpose(1, 2))
        for block in self.blocks: h = block(h)

        positioned = self.pos_enc(h) if self.use_pos else h

        if self.use_attn:
            h = self.window_attn(positioned)
        else:
            if self.training and self.align_stochastic_calls:
                _ = self.window_attn(self.pos_enc(h))
            h = positioned

        if self.has_cls_task:
            h_cls = self.cls_refine(h) if self.use_refine else h
        else:
            if self.training and self.align_stochastic_calls: _ = self.cls_refine(h)
            h_cls = h

        if self.has_reg_task:
            h_reg = self.reg_refine(h) if self.use_refine else h
        else:
            if self.training and self.align_stochastic_calls: _ = self.reg_refine(h)
            h_reg = h

        if not self.use_refine and self.training and self.align_stochastic_calls:
            if self.has_cls_task: _ = self.cls_refine(h)
            if self.has_reg_task: _ = self.reg_refine(h)

        zero_logits = h.new_zeros((h.shape[0], h.shape[-1]))

        if self.has_cls_task:
            cls_logits = self.cls_head(h_cls).squeeze(1)
        else:
            if self.training and self.align_stochastic_calls: _ = self.cls_head(h_cls)
            cls_logits = zero_logits

        if self.has_reg_task:
            reg_logits = self.reg_head(h_reg).squeeze(1)
        else:
            if self.training and self.align_stochastic_calls: _ = self.reg_head(h_reg)
            reg_logits = zero_logits

        return {"cls": cls_logits, "reg": reg_logits}

def count_parameters(model):
    return {
        "total": int(sum(p.numel() for p in model.parameters())),
        "trainable": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
    }

# ============================================================
# Loss
# ============================================================

class FocalBCEWithLogits(nn.Module):
    def __init__(self, alpha=0.75, gamma=1.5):
        super().__init__(); self.alpha, self.gamma = float(alpha), float(gamma)
    def forward(self, logits, target):
        target = target.float()
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        p = torch.sigmoid(logits)
        pt = p * target + (1 - p) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        return (alpha_t * (1 - pt).pow(self.gamma) * bce).mean()

class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.25, beta=0.70, gamma=0.75, eps=1e-6):
        super().__init__()
        self.alpha, self.beta, self.gamma, self.eps = map(float, (alpha, beta, gamma, eps))
    def forward(self, logits, target):
        p = torch.sigmoid(logits).reshape(-1)
        target = target.float().reshape(-1)
        tp = (p * target).sum()
        fp = (p * (1 - target)).sum()
        fn = ((1 - p) * target).sum()
        score = (tp + self.eps) / (tp + self.alpha * fp + self.beta * fn + self.eps)
        return (1 - score).clamp_min(0).pow(self.gamma)

def soft_argmax_position(logits, temperature=1.0):
    w = torch.softmax(logits / max(float(temperature), 1e-6), dim=1)
    pos = torch.arange(logits.shape[1], device=logits.device, dtype=logits.dtype)
    return (w * pos.unsqueeze(0)).sum(dim=1)

class PeakLocalizationLoss(nn.Module):
    def __init__(self, eval_label_threshold, temperature=1.0):
        super().__init__()
        self.eval_label_threshold, self.temperature = float(eval_label_threshold), float(temperature)
    def forward(self, reg_logits_full, soft_target_full):
        max_soft, true_peak_index = soft_target_full.max(dim=1)
        valid = (max_soft > self.eval_label_threshold).to(reg_logits_full.dtype)
        predicted = soft_argmax_position(reg_logits_full, self.temperature)
        per_sample = F.smooth_l1_loss(
            predicted, true_peak_index.float(), beta=1.0, reduction="none"
        )
        return (per_sample * valid).sum() / valid.sum().clamp_min(1.0)

class MultiTaskBoundaryLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.task_mode = cfg["task_mode"]
        self.has_cls_task = self.task_mode in {"multitask", "classification"}
        self.has_reg_task = self.task_mode in {"multitask", "regression"}
        self.eval_label_threshold = float(cfg["eval_label_threshold"])
        self.cls_weight, self.reg_weight = float(cfg["cls_weight"]), float(cfg["reg_weight"])
        self.weighted_alpha = float(cfg["weighted_alpha"])
        self.peak_emphasis_power = float(cfg["peak_emphasis_power"])
        self.huber_beta = float(cfg["huber_beta"])
        self.ambiguous_weight, self.ambiguous_low = float(cfg["ambiguous_weight"]), float(cfg["ambiguous_low"])
        self.tversky_weight = float(cfg["tversky_weight"])
        self.aux_cls_weight, self.aux_reg_weight = float(cfg["aux_cls_weight"]), float(cfg["aux_reg_weight"])
        self.peak_weight = float(cfg["peak_weight"])
        self.use_peak_loss = bool(cfg["use_peak_loss"]) and self.has_reg_task
        self.focal_cls = FocalBCEWithLogits(cfg["focal_alpha"], cfg["focal_gamma"])
        self.tversky_cls = FocalTverskyLoss(cfg["tversky_alpha"], cfg["tversky_beta"], cfg["tversky_gamma"])
        self.peak_loss_fn = PeakLocalizationLoss(cfg["eval_label_threshold"], cfg["peak_temperature"])

    def _classification_term(self, logits, target):
        loss = self.focal_cls(logits, target)
        if self.tversky_weight > 0:
            loss = loss + self.tversky_weight * self.tversky_cls(logits, target)
        return loss

    def _regression_weights(self, target):
        weights = 1 + self.weighted_alpha * target.clamp_min(0).pow(self.peak_emphasis_power)
        ambiguous = ((target > self.ambiguous_low) & (target <= self.eval_label_threshold)).float()
        return weights * (1 - ambiguous + ambiguous * self.ambiguous_weight)

    def _regression_term(self, logits, target):
        base = F.smooth_l1_loss(
            torch.sigmoid(logits), target, reduction="none", beta=self.huber_beta
        )
        return (self._regression_weights(target) * base).mean()

    def forward(self, outputs, soft_target_full, center_idx):
        cls_full, reg_full = outputs["cls"], outputs["reg"]
        soft_target_full = soft_target_full.float()
        has_window = soft_target_full.dim() > 1 and soft_target_full.shape[1] > 1
        soft_center = soft_target_full[:, center_idx] if has_window else soft_target_full.reshape(-1)
        zero = cls_full.new_zeros(())

        cls_center_loss = cls_aux_loss = cls_loss = zero
        if self.has_cls_task:
            cls_center = cls_full[:, center_idx] if cls_full.dim() > 1 else cls_full
            binary_center = (soft_center > self.eval_label_threshold).float()
            cls_center_loss = self._classification_term(cls_center, binary_center)
            if has_window and self.aux_cls_weight > 0:
                binary_full = (soft_target_full > self.eval_label_threshold).float()
                cls_aux_loss = self._classification_term(cls_full, binary_full)
            cls_loss = cls_center_loss + self.aux_cls_weight * cls_aux_loss

        reg_center_loss = reg_aux_loss = peak_loss = reg_loss = zero
        if self.has_reg_task:
            reg_center = reg_full[:, center_idx] if reg_full.dim() > 1 else reg_full
            reg_center_loss = self._regression_term(reg_center, soft_center)
            if has_window and self.aux_reg_weight > 0:
                reg_aux_loss = self._regression_term(reg_full, soft_target_full)
            if has_window and self.use_peak_loss and self.peak_weight > 0:
                peak_loss = self.peak_loss_fn(reg_full, soft_target_full)
            reg_loss = reg_center_loss + self.aux_reg_weight * reg_aux_loss + self.peak_weight * peak_loss

        total = (
            self.cls_weight * cls_loss + self.reg_weight * reg_loss
            if self.task_mode == "multitask"
            else cls_loss if self.task_mode == "classification"
            else reg_loss
        )

        values = {
            "loss": total, "cls_loss": cls_loss, "reg_loss": reg_loss,
            "cls_center_loss": cls_center_loss, "cls_aux_loss": cls_aux_loss,
            "reg_center_loss": reg_center_loss, "reg_aux_loss": reg_aux_loss,
            "peak_loss": peak_loss,
        }
        metrics = torch.stack([values[k].detach() for k in LOSS_KEYS])
        return total, metrics

def reduce_loss_metrics(metric_batches):
    if not metric_batches: raise RuntimeError("No loss metrics collected")
    rows = torch.stack(metric_batches).cpu().tolist()
    sums = [0.0] * len(LOSS_KEYS)
    for row in rows:
        for i, value in enumerate(row): sums[i] += value
    n = len(rows)
    return {k: sums[i] / n for i, k in enumerate(LOSS_KEYS)}


# ============================================================
# DataLoader / metrics
# ============================================================

def make_sampler(soft_labels, center_idx, gamma, min_weight, max_weight, seed):
    y = center_of(soft_labels, center_idx)
    groups = (y > .70, (y > .30) & (y <= .70), (y > .05) & (y <= .30), y <= .05)
    weights = np.ones(len(y), dtype=np.float64)
    for mask in groups:
        count = int(mask.sum())
        if count: weights[mask] = (1.0 / count) ** float(gamma)
    weights /= max(float(weights.mean()), 1e-12)
    weights = np.clip(weights, float(min_weight), float(max_weight))
    g = torch.Generator().manual_seed(int(seed) + 101)
    sampler = WeightedRandomSampler(
        torch.from_numpy(weights), len(weights), replacement=True, generator=g
    )
    return sampler, {
        "weights_sha256": hashlib.sha256(weights.tobytes()).hexdigest(),
        "weight_min": float(weights.min()), "weight_max": float(weights.max()),
        "weight_mean": float(weights.mean()), "generator_seed": int(seed) + 101,
    }

def create_dataloaders(cfg, train_set, val_set, test_set, center_idx):
    if cfg["sampler_type"] == "tempered":
        sampler, sampler_info = make_sampler(
            train_set.soft_labels, center_idx, cfg["sampler_gamma"],
            cfg["sampler_min_weight"], cfg["sampler_max_weight"], cfg["seed"]
        )
        shuffle = False
    elif cfg["sampler_type"] == "none":
        sampler, sampler_info, shuffle = None, {"type": "none"}, True
    else:
        raise ValueError(f"Unsupported sampler_type: {cfg['sampler_type']}")

    common = dict(
        batch_size=int(cfg["batch_size"]),
        num_workers=int(cfg["num_workers"]),
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
    )
    if cfg["num_workers"] > 0:
        common.update(persistent_workers=True, prefetch_factor=4)

    train_loader = DataLoader(
        train_set, sampler=sampler, shuffle=shuffle,
        generator=torch.Generator().manual_seed(int(cfg["seed"]) + 201), **common
    )
    val_loader = DataLoader(
        val_set, shuffle=False,
        generator=torch.Generator().manual_seed(int(cfg["seed"]) + 301), **common
    )
    test_loader = DataLoader(
        test_set, shuffle=False,
        generator=torch.Generator().manual_seed(int(cfg["seed"]) + 401), **common
    ) if test_set is not None else None

    return train_loader, val_loader, test_loader, {
        "sampler": sampler_info,
        "train_loader_seed": int(cfg["seed"]) + 201,
        "val_loader_seed": int(cfg["seed"]) + 301,
        "test_loader_seed": int(cfg["seed"]) + 401,
    }

def safe_corr(x, y):
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    if len(x) <= 1 or np.std(x) <= 0 or np.std(y) <= 0: return 0.0
    return float(np.corrcoef(x, y)[0, 1])

def _binary_static_metrics(probabilities, labels):
    p = np.asarray(probabilities, dtype=np.float64)
    y = (np.asarray(labels, dtype=np.float64) > 0.5).astype(np.int64)
    if y.sum() > 0:
        order = np.argsort(-p, kind="mergesort")
        ys = y[order]
        tp_cum = np.cumsum(ys)
        fp_cum = np.cumsum(1 - ys)
        precision_curve = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
        recall_curve = tp_cum / max(y.sum(), 1)
        recall_prev = np.concatenate([[0.0], recall_curve[:-1]])
        auprc = float(np.sum((recall_curve - recall_prev) * precision_curve))
    else:
        auprc = 0.0
    return {
        "auprc": auprc,
        "pearson_binary": safe_corr(p, y),
        "binary_mae": float(np.mean(np.abs(p - y))),
        "binary_mse": float(np.mean((p - y) ** 2)),
    }

def _threshold_metrics(probabilities, labels, threshold):
    p = np.asarray(probabilities, dtype=np.float64)
    y = (np.asarray(labels, dtype=np.float64) > 0.5).astype(np.int64)
    pred = (p > float(threshold)).astype(np.int64)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "f1": float(f1), "precision": float(precision), "recall": float(recall),
        "specificity": float(specificity), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }

def binary_metrics(probabilities, labels, threshold):
    return {
        **_threshold_metrics(probabilities, labels, threshold),
        **_binary_static_metrics(probabilities, labels),
    }

def continuous_metrics(p, y):
    p, y = np.asarray(p, dtype=np.float64), np.asarray(y, dtype=np.float64)
    return {
        "soft_pearson": safe_corr(p, y),
        "soft_mae": float(np.mean(np.abs(p - y))),
        "soft_mse": float(np.mean((p - y) ** 2)),
    }

def find_best_threshold(probabilities, labels, cfg):
    static = _binary_static_metrics(probabilities, labels)
    best_threshold, best_threshold_metrics, best_f1 = 0.5, None, float("-inf")
    thresholds = np.arange(
        cfg["threshold_min"], cfg["threshold_max"] + 1e-12, cfg["threshold_step"]
    )
    for threshold in thresholds:
        current = _threshold_metrics(probabilities, labels, float(threshold))
        if current["f1"] > best_f1 + 1e-15:
            best_threshold = float(threshold)
            best_threshold_metrics = current
            best_f1 = current["f1"]
        elif abs(current["f1"] - best_f1) <= 1e-15 and float(threshold) < best_threshold:
            best_threshold = float(threshold)
            best_threshold_metrics = current
    return best_threshold, {**best_threshold_metrics, **static}

def hard_recall_from_soft(p, y, threshold, hard_threshold):
    p, y = np.asarray(p, np.float64), np.asarray(y, np.float64)
    mask = y > hard_threshold
    return float((p[mask] > threshold).mean()) if mask.any() else 0.0

def expected_eval_head(task_mode):
    """
    本次训练为纯回归模型，统一使用regression head评价。
    """
    if task_mode != "regression":
        raise ValueError(
            "This training script only supports "
            f"task_mode='regression', got {task_mode!r}"
        )
    return "regression"


def select_eval_logits(outputs, task_mode):
    expected_eval_head(task_mode)
    return outputs["reg"], "regression"


# ============================================================
# Train / evaluate
# ============================================================

def run_train_epoch(model, loader, criterion, optimizer, device, center_idx, cfg):
    model.train()
    metric_batches = []
    interval, n_batches = int(cfg["finite_check_interval"]), len(loader)

    for batch_idx, (features, target) in enumerate(loader, 1):
        features = features.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        outputs = model(features)
        loss, metrics = criterion(outputs, target, center_idx)

        if batch_idx % interval == 0 or batch_idx == n_batches:
            if not torch.isfinite(loss).item():
                raise FloatingPointError("Non-finite training loss")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if cfg["grad_clip"] > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=float(cfg["grad_clip"])
            )

        optimizer.step()
        metric_batches.append(metrics)

    avg = reduce_loss_metrics(metric_batches)
    return {f"train_{k}": v for k, v in avg.items()}

@torch.no_grad()
def evaluate(model, loader, criterion, device, center_idx, cfg,
             fixed_threshold=None, return_arrays=False):
    model.eval()
    metric_batches, prob_batches, soft_batches = [], [], []
    peak_error_sums, peak_valid_counts = [], []
    eval_head = None
    interval, n_batches = int(cfg["finite_check_interval"]), len(loader)

    for batch_idx, (features, target) in enumerate(loader, 1):
        features = features.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        outputs = model(features)
        loss, metrics = criterion(outputs, target, center_idx)

        if batch_idx % interval == 0 or batch_idx == n_batches:
            if not torch.isfinite(loss).item():
                raise FloatingPointError("Non-finite evaluation loss")

        metric_batches.append(metrics)

        eval_logits, eval_head = select_eval_logits(outputs, cfg["task_mode"])
        center_logits = eval_logits[:, center_idx] if eval_logits.dim() > 1 else eval_logits

        prob_batches.append(torch.sigmoid(center_logits).detach())
        soft_batches.append(center_of(target, center_idx).detach())

        if cfg["task_mode"] in {"multitask", "regression"} and target.dim() > 1:
            max_soft, true_peak_index = target.max(dim=1)
            valid = max_soft > cfg["eval_label_threshold"]
            predicted = soft_argmax_position(outputs["reg"], cfg["peak_temperature"])
            errors = torch.abs(predicted - true_peak_index.float())
            valid_float = valid.to(errors.dtype)
            peak_error_sums.append((errors * valid_float).sum().detach())
            peak_valid_counts.append(valid_float.sum().detach())

    loss_metrics = reduce_loss_metrics(metric_batches)
    probabilities = torch.cat(prob_batches).cpu().numpy().astype(np.float32)
    soft_targets = torch.cat(soft_batches).cpu().numpy().astype(np.float32)

    if peak_error_sums:
        peak_sum = torch.stack(peak_error_sums).sum()
        peak_count = torch.stack(peak_valid_counts).sum().clamp_min(1.0)
        peak_mae = float((peak_sum / peak_count).cpu().item())
    else:
        peak_mae = 0.0

    labels = (soft_targets > cfg["eval_label_threshold"]).astype(np.float32)

    if fixed_threshold is None:
        threshold, metrics = find_best_threshold(probabilities, labels, cfg)
    else:
        threshold = float(fixed_threshold)
        metrics = binary_metrics(probabilities, labels, threshold)

    result = {
        **loss_metrics,
        "eval_head": eval_head,
        "eval_head_policy": cfg["eval_head_policy"],
        "threshold": float(threshold),
        **metrics,
        **continuous_metrics(probabilities, soft_targets),
        "hard_recall": hard_recall_from_soft(
            probabilities, soft_targets, threshold, cfg["hard_recall_soft_threshold"]
        ),
        "peak_position_mae": peak_mae,
    }

    if return_arrays:
        return result, {
            "probabilities": probabilities,
            "soft_targets": soft_targets,
            "binary_labels": labels,
        }
    return result

def record_is_better(candidate, incumbent, cfg):
    if incumbent is None: return True

    drop_tol = float(cfg.get("hard_recall_drop_tolerance", 0.05))
    if float(candidate["hard_recall"]) < float(incumbent["hard_recall"]) * (1.0 - drop_tol):
        return False

    metric = cfg["best_metric"]
    tolerance = float(cfg["tie_tolerance"])
    candidate_score = float(candidate[metric])
    incumbent_score = float(incumbent[metric])

    if not np.isfinite(candidate_score): return False
    if candidate_score > incumbent_score + tolerance: return True
    if abs(candidate_score - incumbent_score) <= tolerance:
        return float(candidate["val_loss"]) < float(incumbent["val_loss"])
    return False

# ============================================================
# Canonical initialization / paths
# ============================================================

def architecture_payload(cfg, input_dim):
    return {
        "model_schema_version": MODEL_SCHEMA_VERSION,
        "code_sha256": cfg["code_sha256"],
        "seed": int(cfg["seed"]),
        "input_dim": int(input_dim),
        "feature_names": FEATURE_NAMES,
        "hidden_dim": int(cfg["hidden_dim"]),
        "dropout": float(cfg["dropout"]),
        "attn_heads": int(cfg["attn_heads"]),
        "dilations": list(cfg["dilations"]),
        "window_size": int(cfg["window_size"]),
    }

def canonical_init_path(cfg, input_dim):
    sig = json_hash(architecture_payload(cfg, input_dim))[:12]
    return (
        Path(cfg["output_root"]) / "canonical_initialization" /
        f"seed_{cfg['seed']}" / f"canonical_{sig}.pth"
    )

def build_model_from_cfg(cfg, input_dim):
    return MultiTaskDilatedCNN(
        input_dim=input_dim, hidden_dim=cfg["hidden_dim"], dropout=cfg["dropout"],
        attn_heads=cfg["attn_heads"], use_pos=cfg["use_pos"], use_attn=cfg["use_attn"],
        use_se=cfg["use_se"], dilations=cfg["dilations"], use_refine=cfg["use_refine"],
        task_mode=cfg["task_mode"], align_stochastic_calls=cfg["align_stochastic_calls"],
    )

def get_or_create_canonical_state(cfg, input_dim):
    path = canonical_init_path(cfg, input_dim)
    expected = architecture_payload(cfg, input_dim)

    if path.exists():
        payload = torch_load(path, "cpu")
        if payload.get("architecture") != expected:
            raise RuntimeError(f"Canonical metadata mismatch: {path}")
        state = payload["state_dict"]
        if state_dict_hash(state) != payload["state_hash"]:
            raise RuntimeError(f"Canonical hash mismatch: {path}")
        return state, payload["state_hash"], path

    path.parent.mkdir(parents=True, exist_ok=True)
    canonical_cfg = copy.deepcopy(cfg)
    canonical_cfg.update({
        "use_pos": True, "use_attn": True, "use_se": True,
        "use_refine": True, "task_mode": "multitask"
    })
    set_global_determinism(cfg["seed"], cfg["deterministic"])
    model = build_model_from_cfg(canonical_cfg, input_dim).cpu()
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    state_hash = state_dict_hash(state)
    torch.save({
        "architecture": expected, "state_dict": state, "state_hash": state_hash,
        "environment": environment_fingerprint(),
    }, path)
    return state, state_hash, path

def make_run_paths(cfg, profile):
    sig = config_signature(cfg)
    out = Path(cfg["output_root"]) / "runs" / profile / f"seed_{cfg['seed']}" / sig
    out.mkdir(parents=True, exist_ok=True)
    return {
        "signature": sig,
        "output_dir": out,
        "checkpoint": out / "best_model.pth",
        "summary": out / "summary.json",
        "history": out / "history.csv",
        "config": out / "config.json",
        "test_predictions": out / "test_predictions.npz",
    }


# ============================================================
# Experiment
# ============================================================

def train_one_experiment(cfg, trained_profile, force=False):
    cfg = normalize_cfg(cfg)
    cfg["profile_name"] = trained_profile
    paths = make_run_paths(cfg, trained_profile)

    if paths["summary"].exists() and not force:
        print(f"\n[SKIP] Existing run preserved: {paths['summary']}")
        return load_json(paths["summary"])

    set_global_determinism(cfg["seed"], cfg["deterministic"])

    if cfg["deterministic"] and os.environ.get("PYTHONHASHSEED") is None:
        print("[WARN] 建议启动时设置 PYTHONHASHSEED=<seed> CUBLAS_WORKSPACE_CONFIG=:4096:8")

    data_dir = Path(cfg["data_base"]) / cfg["dataset"]
    center_idx = cfg["window_size"] // 2
    input_dim = infer_input_dim(data_dir)
    planned_eval_head = expected_eval_head(cfg["task_mode"])

    print("\n" + "=" * 100)
    print(f"Profile={trained_profile} seed={cfg['seed']}")
    print(
        f"pos={cfg['use_pos']} attn={cfg['use_attn']} se={cfg['use_se']} "
        f"refine={cfg['use_refine']} task={cfg['task_mode']}"
    )
    print(f"eval_head={planned_eval_head} | policy={cfg['eval_head_policy']}")
    print(
        f"peak={cfg['use_peak_loss']} aux_cls={cfg['aux_cls_weight']} "
        f"aux_reg={cfg['aux_reg_weight']} tversky={cfg['tversky_weight']} "
        f"ambiguous={cfg['ambiguous_weight']}"
    )
    print(f"train={cfg['train_chrs']} | features={FEATURE_NAMES}")
    print(f"output={paths['output_dir']}")
    print("=" * 100)

    train_set = HICFeatureDataset(data_dir, cfg["train_chrs"], cfg["window_size"], input_dim)
    val_set = HICFeatureDataset(data_dir, cfg["val_chrs"], cfg["window_size"], input_dim)
    test_set = HICFeatureDataset(
        data_dir, cfg["test_chrs"], cfg["window_size"], input_dim
    ) if cfg["evaluate_test"] else None

    train_loader, val_loader, test_loader, loader_info = create_dataloaders(
        cfg, train_set, val_set, test_set, center_idx
    )

    canonical_state, canonical_hash, canonical_path = get_or_create_canonical_state(cfg, input_dim)
    model = build_model_from_cfg(cfg, input_dim).cpu()
    model.load_state_dict(canonical_state, strict=True)

    initial_hash = state_dict_hash(model.state_dict())
    if initial_hash != canonical_hash:
        raise RuntimeError("Initial state != canonical state")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    reset_training_rng(cfg["seed"])

    criterion = MultiTaskBoundaryLoss(cfg)
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable: raise RuntimeError("No trainable parameters")

    optimizer = optim.AdamW(trainable, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=.5,
        patience=cfg["scheduler_patience"], min_lr=1e-6
    )

    parameter_counts, model_architecture = count_parameters(model), str(model)

    save_json(paths["config"], {
        "config": cfg,
        "profile": trained_profile,
        "input_dim": input_dim,
        "feature_names": FEATURE_NAMES,
        "eval_head_policy": EVAL_HEAD_POLICY,
        "planned_eval_head": planned_eval_head,
        "model_architecture": model_architecture,
        "parameter_counts": parameter_counts,
        "canonical_initialization_path": str(canonical_path),
        "canonical_initialization_hash": canonical_hash,
        "initial_state_hash": initial_hash,
        "loader_info": loader_info,
        "environment": environment_fingerprint(),
        "config_signature": paths["signature"],
    })

    print(
        f"Device={device} | train/val/test={len(train_set):,}/"
        f"{len(val_set):,}/{len(test_set) if test_set else 0:,}"
    )
    print(
        f"Params total/trainable={parameter_counts['total']:,}/"
        f"{parameter_counts['trainable']:,}"
    )

    header = (
        f"{'Ep':>3} | {'Head':>4} | {'Th':>5} | {'TrLoss':>8} | {'TrCls':>8} | {'TrReg':>8} | "
        f"{'VaLoss':>8} | {'VaCls':>8} | {'VaReg':>8} | {'AUPRC':>7} | {'F1':>6} | "
        f"{'Prec':>6} | {'Rec':>6} | {'Spec':>6} | {'HRec':>6} | "
        f"{'SoftMAE':>8} | {'PeakMAE':>8} | {'LR':>9} | {'Mark':>5}"
    )
    print("\n" + "-" * len(header))
    print(header)
    print("-" * len(header))

    best_record = best_state = None
    best_state_hash = None
    history, patience_counter, start_time = [], 0, time.time()

    for epoch in range(1, int(cfg["epochs"]) + 1):
        train_metrics = run_train_epoch(
            model, train_loader, criterion, optimizer, device, center_idx, cfg
        )
        valid = evaluate(model, val_loader, criterion, device, center_idx, cfg)

        expected_head = expected_eval_head(cfg["task_mode"])
        if valid["eval_head"] != expected_head:
            raise RuntimeError(
                f"Eval-head policy violation: task={cfg['task_mode']} "
                f"got={valid['eval_head']} expected={expected_head}"
            )

        lr = float(optimizer.param_groups[0]["lr"])
        record = {
            "epoch": epoch,
            **{k: float(v) for k, v in train_metrics.items()},
            **{f"val_{k}": float(valid[k]) for k in LOSS_KEYS},
            "eval_head": valid["eval_head"],
            "eval_head_policy": valid["eval_head_policy"],
            "threshold": float(valid["threshold"]),
            "auprc": float(valid["auprc"]),
            "f1": float(valid["f1"]),
            "precision": float(valid["precision"]),
            "recall": float(valid["recall"]),
            "specificity": float(valid["specificity"]),
            "hard_recall": float(valid["hard_recall"]),
            "pearson_binary": float(valid["pearson_binary"]),
            "binary_mae": float(valid["binary_mae"]),
            "binary_mse": float(valid["binary_mse"]),
            "soft_pearson": float(valid["soft_pearson"]),
            "soft_mae": float(valid["soft_mae"]),
            "soft_mse": float(valid["soft_mse"]),
            "peak_position_mae": float(valid["peak_position_mae"]),
            "tp": int(valid["tp"]), "fp": int(valid["fp"]),
            "fn": int(valid["fn"]), "tn": int(valid["tn"]),
            "lr": lr,
            "time_sec": float(time.time() - start_time),
        }

        history.append(record)
        scheduler.step(record[cfg["best_metric"]])

        if record_is_better(record, best_record, cfg):
            best_record = copy.deepcopy(record)
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            patience_counter, mark = 0, "*"
            best_state_hash = state_dict_hash(best_state)

            torch.save({
                "model_state_dict": best_state,
                "optimizer_state_dict": optimizer.state_dict(),
                "config": to_builtin(cfg),
                "profile_name": trained_profile,
                "feature_names": FEATURE_NAMES,
                "model_architecture": model_architecture,
                "parameter_counts": parameter_counts,
                "input_dim": input_dim,
                "center_idx": center_idx,
                "canonical_initialization_hash": canonical_hash,
                "initial_state_hash": initial_hash,
                "best_state_hash": best_state_hash,
                "best_record": best_record,
                "history": history,
                "config_signature": paths["signature"],
                "code_sha256": cfg["code_sha256"],
                "eval_head_policy": EVAL_HEAD_POLICY,
            }, paths["checkpoint"])
        else:
            patience_counter += 1
            mark = f"P{patience_counter}"

        write_csv(paths["history"], history)

        print(
            f"{epoch:3d} | {record['eval_head'][:4]:>4} | {record['threshold']:5.3f} | "
            f"{record['train_loss']:8.5f} | {record['train_cls_loss']:8.5f} | "
            f"{record['train_reg_loss']:8.5f} | {record['val_loss']:8.5f} | "
            f"{record['val_cls_loss']:8.5f} | {record['val_reg_loss']:8.5f} | "
            f"{record['auprc']:7.4f} | {record['f1']:6.4f} | "
            f"{record['precision']:6.4f} | {record['recall']:6.4f} | "
            f"{record['specificity']:6.4f} | {record['hard_recall']:6.4f} | "
            f"{record['soft_mae']:8.5f} | {record['peak_position_mae']:8.4f} | "
            f"{lr:9.2e} | {mark:>5}"
        )

        if patience_counter >= cfg["patience"]:
            print(f"Early stopping at epoch {epoch}")
            break

    if best_state is None: raise RuntimeError("No valid checkpoint produced")

    model.load_state_dict(best_state, strict=True)
    final_best_hash = state_dict_hash(model.state_dict())

    test_record = test_arrays = None
    if test_loader is not None:
        test_record, test_arrays = evaluate(
            model, test_loader, criterion, device, center_idx, cfg,
            fixed_threshold=best_record["threshold"], return_arrays=True
        )
        if test_record["eval_head"] != expected_eval_head(cfg["task_mode"]):
            raise RuntimeError("Test eval-head policy violation")

    summary = {
        "profile_name": trained_profile,
        "config": cfg,
        "config_signature": paths["signature"],
        "eval_head_policy": EVAL_HEAD_POLICY,
        "planned_eval_head": planned_eval_head,
        "data_dir": str(data_dir),
        "input_dim": input_dim,
        "feature_names": FEATURE_NAMES,
        "parameter_counts": parameter_counts,
        "model_architecture": model_architecture,
        "canonical_initialization_path": str(canonical_path),
        "canonical_initialization_hash": canonical_hash,
        "initial_state_hash": initial_hash,
        "best_state_hash": final_best_hash,
        "loader_info": loader_info,
        "best_val_record": best_record,
        "test_record": test_record,
        "history": history,
        "total_time_seconds": float(time.time() - start_time),
        "checkpoint_path": str(paths["checkpoint"]),
        "summary_path": str(paths["summary"]),
        "history_path": str(paths["history"]),
        "config_path": str(paths["config"]),
        "test_predictions_path": str(paths["test_predictions"]) if test_arrays is not None else None,
        "environment": environment_fingerprint(),
    }

    if test_arrays is not None:
        np.savez_compressed(
            paths["test_predictions"],
            probabilities=test_arrays["probabilities"],
            soft_targets=test_arrays["soft_targets"],
            binary_labels=test_arrays["binary_labels"],
            chromosomes=test_set.sample_chroms,
            threshold=np.asarray(best_record["threshold"], dtype=np.float32),
            eval_head=np.asarray(test_record["eval_head"]),
            eval_head_policy=np.asarray(EVAL_HEAD_POLICY),
        )

    save_json(paths["summary"], summary)

    print("\nBest Validation:")
    print(
        f" Epoch={best_record['epoch']} | Head={best_record['eval_head']} | "
        f"Policy={best_record['eval_head_policy']} | "
        f"Th={best_record['threshold']:.3f} | Loss={best_record['val_loss']:.5f} | "
        f"Cls={best_record['val_cls_loss']:.5f} | Reg={best_record['val_reg_loss']:.5f} | "
        f"AUPRC={best_record['auprc']:.5f} | F1={best_record['f1']:.5f} | "
        f"P={best_record['precision']:.5f} | R={best_record['recall']:.5f} | "
        f"Spec={best_record['specificity']:.5f} | HRec={best_record['hard_recall']:.5f} | "
        f"SoftPearson={best_record['soft_pearson']:.5f} | "
        f"SoftMAE={best_record['soft_mae']:.5f} | "
        f"PeakMAE={best_record['peak_position_mae']:.4f}"
    )

    if test_record is not None:
        print("\nFinal Test using best validation threshold:")
        print(
            f" Head={test_record['eval_head']} | Policy={test_record['eval_head_policy']} | "
            f"Th={test_record['threshold']:.3f} | Loss={test_record['loss']:.5f} | "
            f"Cls={test_record['cls_loss']:.5f} | Reg={test_record['reg_loss']:.5f} | "
            f"AUPRC={test_record['auprc']:.5f} | F1={test_record['f1']:.5f} | "
            f"P={test_record['precision']:.5f} | R={test_record['recall']:.5f} | "
            f"Spec={test_record['specificity']:.5f} | HRec={test_record['hard_recall']:.5f} | "
            f"SoftPearson={test_record['soft_pearson']:.5f} | "
            f"SoftMAE={test_record['soft_mae']:.5f} | "
            f"PeakMAE={test_record['peak_position_mae']:.4f}"
        )

    print(f"Saved summary: {paths['summary']}")
    return summary

def create_alias_record(display_profile, trained_profile, summary, base_cfg):
    path = (
        Path(base_cfg["output_root"]) / "aliases" / display_profile /
        f"seed_{summary['config']['seed']}" / "alias_summary.json"
    )
    save_json(path, {
        "display_profile": display_profile,
        "trained_profile": trained_profile,
        "source_summary_path": summary["summary_path"],
        "source_checkpoint_path": summary["checkpoint_path"],
        "best_val_record": summary["best_val_record"],
        "test_record": summary["test_record"],
        "eval_head_policy": EVAL_HEAD_POLICY,
        "canonical_initialization_hash": summary["canonical_initialization_hash"],
        "best_state_hash": summary["best_state_hash"],
    })
    return path

def run_profile(base_cfg, display_profile, force=False):
    cfg, trained_profile = resolve_profile(display_profile, base_cfg)
    summary = train_one_experiment(cfg, trained_profile, force)
    result = copy.deepcopy(summary)
    result.update({
        "display_profile": display_profile,
        "trained_profile": trained_profile,
        "is_alias": display_profile != trained_profile,
        "description": PROFILE_SPECS[display_profile]["description"],
    })
    result["alias_summary_path"] = str(create_alias_record(
        display_profile, trained_profile, summary, base_cfg
    )) if result["is_alias"] else None
    return result

def comparison_row(result):
    val, test, cfg = result["best_val_record"], result.get("test_record"), result["config"]
    return {
        "display_profile": result["display_profile"],
        "trained_profile": result["trained_profile"],
        "is_alias": int(result["is_alias"]),
        "seed": int(cfg["seed"]),
        "task_mode": cfg["task_mode"],
        "eval_head": val["eval_head"],
        "eval_head_policy": cfg["eval_head_policy"],
        "use_pos": cfg["use_pos"],
        "use_attn": cfg["use_attn"],
        "use_se": cfg["use_se"],
        "use_peak_loss": cfg["use_peak_loss"],
        "aux_cls_weight": cfg["aux_cls_weight"],
        "aux_reg_weight": cfg["aux_reg_weight"],
        "tversky_weight": cfg["tversky_weight"],
        "ambiguous_weight": cfg["ambiguous_weight"],
        "trainable_parameters": result["parameter_counts"]["trainable"],
        "canonical_hash": result["canonical_initialization_hash"],
        "initial_state_hash": result["initial_state_hash"],
        "best_state_hash": result["best_state_hash"],
        "val_auprc": val["auprc"],
        "val_f1": val["f1"],
        "val_precision": val["precision"],
        "val_recall": val["recall"],
        "val_specificity": val["specificity"],
        "val_hard_recall": val["hard_recall"],
        "val_soft_mae": val["soft_mae"],
        "val_soft_pearson": val["soft_pearson"],
        "val_peak_position_mae": val["peak_position_mae"],
        "val_threshold": val["threshold"],
        "best_epoch": val["epoch"],
        "val_loss": val["val_loss"],
        "test_auprc": test["auprc"] if test else None,
        "test_f1": test["f1"] if test else None,
        "test_precision": test["precision"] if test else None,
        "test_recall": test["recall"] if test else None,
        "test_specificity": test["specificity"] if test else None,
        "test_hard_recall": test["hard_recall"] if test else None,
        "test_soft_mae": test["soft_mae"] if test else None,
        "test_peak_position_mae": test["peak_position_mae"] if test else None,
        "summary_path": result["summary_path"],
        "checkpoint_path": result["checkpoint_path"],
        "alias_summary_path": result["alias_summary_path"],
        "description": result["description"],
    }


def run_reproducibility_check(base_cfg, profile, seed, epochs=3):
    base_cfg = copy.deepcopy(base_cfg)
    base_cfg["deterministic"] = True
    root = Path(base_cfg["output_root"]) / "reproducibility_checks" / profile / f"seed_{seed}"
    runs = []

    for i in (1, 2):
        cfg = copy.deepcopy(base_cfg)
        cfg.update({
            "seed": int(seed),
            "epochs": int(epochs),
            "patience": int(epochs) + 1,
            "output_root": root / f"repeat_{i}",
        })
        runs.append(run_profile(cfg, profile, force=True))

    first, second = runs

    def strip_time(x):
        if isinstance(x, dict):
            return {
                k: strip_time(v)
                for k, v in x.items()
                if k not in {"time_sec", "total_time_seconds"}
            }
        if isinstance(x, list): return [strip_time(v) for v in x]
        return x

    report = {
        "profile": profile,
        "seed": int(seed),
        "epochs": int(epochs),
        "eval_head_policy": EVAL_HEAD_POLICY,
        "initial_state_hash_equal": first["initial_state_hash"] == second["initial_state_hash"],
        "best_state_hash_equal": first["best_state_hash"] == second["best_state_hash"],
        "history_equal": strip_time(to_builtin(first["history"])) == strip_time(to_builtin(second["history"])),
        "best_val_equal": strip_time(to_builtin(first["best_val_record"])) == strip_time(to_builtin(second["best_val_record"])),
    }
    report["passed"] = all(v for k, v in report.items() if k.endswith("_equal"))
    save_json(root / "reproducibility_report.json", report)

    if not report["passed"]:
        raise RuntimeError("Reproducibility check failed")

    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report

# ============================================================
# CLI
# ============================================================

def print_profiles():
    print(f"Features ({len(FEATURE_NAMES)}d): {FEATURE_NAMES}")
    print(f"Eval policy: {EVAL_HEAD_POLICY}")
    print("Formal profile: task_mode=regression, eval_head=regression")
    for name in PROFILE_SPECS:
        print(f"  {name:<40}{PROFILE_SPECS[name]['description']}")


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    p.add_argument(
        "--mode",
        choices=["list", "single", "repro"],
        default="list",
    )
    p.add_argument("--profile")
    p.add_argument("--seeds", default="1001")
    p.add_argument(
        "--force",
        action="store_true",
        help="显式重跑同一config signature；默认已有结果SKIP，不覆盖。",
    )
    p.add_argument("--data-base")
    p.add_argument("--output-root")
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--num-workers", type=int)
    p.add_argument("--exclude-train-chrs")
    p.add_argument("--no-test", action="store_true")
    p.add_argument(
        "--allow-nondeterministic",
        action="store_true",
        help="兼容旧命令；当前默认已是快速非严格确定性模式。",
    )
    p.add_argument(
        "--strict-deterministic",
        action="store_true",
        help="启用严格确定性kernel；会明显变慢。",
    )
    p.add_argument(
        "--align-stochastic-calls",
        action="store_true",
        help=(
            "未使用的模块仍执行后丢弃，仅用于极严格随机调用对齐；"
            "会明显变慢。"
        ),
    )
    p.add_argument("--repro-epochs", type=int, default=3)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = copy.deepcopy(CFG)

    if args.data_base:
        cfg["data_base"] = Path(args.data_base)
    if args.output_root:
        cfg["output_root"] = Path(args.output_root)
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.num_workers is not None:
        cfg["num_workers"] = args.num_workers
    if args.exclude_train_chrs is not None:
        cfg["excluded_train_chrs"] = parse_list(args.exclude_train_chrs)

    cfg["evaluate_test"] = not args.no_test
    cfg["deterministic"] = bool(args.strict_deterministic)
    cfg["align_stochastic_calls"] = bool(args.align_stochastic_calls)
    cfg["eval_head_policy"] = EVAL_HEAD_POLICY

    seeds = parse_list(args.seeds, int)
    if not seeds:
        raise ValueError("--seeds must contain at least one seed")

    validate_experiment_definitions(cfg)

    print("\n" + "=" * 100)
    print("MISTAD 5D pure-regression training")
    print(f"Training schema: {TRAINING_SCHEMA_VERSION}")
    print(f"Eval policy: {EVAL_HEAD_POLICY}")
    print(f"Profile: {', '.join(PROFILE_SPECS)}")
    print(f"Output root: {Path(cfg['output_root']).resolve()}")
    print("Existing matching runs are preserved unless --force is used.")
    print("=" * 100)

    if args.mode == "list":
        print_profiles()
    elif args.mode == "single":
        if not args.profile or len(seeds) != 1:
            raise ValueError("single requires --profile and one seed")
        single_cfg = copy.deepcopy(cfg)
        single_cfg["seed"] = seeds[0]
        print(json.dumps(
            comparison_row(
                run_profile(single_cfg, args.profile, args.force)
            ),
            indent=2,
            ensure_ascii=False,
        ))
    elif args.mode == "repro":
        if not args.profile or len(seeds) != 1:
            raise ValueError("repro requires --profile and one seed")
        run_reproducibility_check(
            cfg, args.profile, seeds[0], args.repro_epochs
        )


if __name__ == "__main__":
    main()
