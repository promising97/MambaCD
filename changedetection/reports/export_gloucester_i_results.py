#!/usr/bin/env python3
"""Export Gloucester I COPE-S + DFGF experiment tables and figures."""

from __future__ import annotations

import csv
import json
import re
import statistics as st
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/root/autodl-tmp/MambaCD")
LOG = ROOT / "changedetection/logs/train_Gloucester_I_COPE-S_DFGF_20261001_171917.log"
OUT = ROOT / "changedetection/reports/Gloucester_I_COPE-S_DFGF"
FIG = OUT / "figures"

CHECKPOINT = (
    ROOT
    / "changedetection/saved_models/Gloucester_I"
    / "LEM-CD_COPE-S_DFGF_Gloucester_I_batch8_iter300000_20261001_171917_1790846362.9661238"
    / "36000_model.pth"
)
INFER_MAP = (
    ROOT
    / "results/Gloucester_I/LEM-CD_COPE-S_DFGF_Gloucester_I_36000_best/change_map"
)

CONFIG = {
    "dataset": "Gloucester I",
    "dataset_id": "Gloucester_I",
    "task": "Binary change detection (2 classes)",
    "model_stack": "SpatialMamba + DFGF-Star + COPE-S",
    "train_script": "changedetection/script/train_MambaBCD_spatial_COPE_S_DFGF.py",
    "infer_script": "changedetection/script/infer_MambaBCD_spatial_COPE_S_DFGF.py",
    "cfg_yaml": "changedetection/configs/spatialmamba/spatialmamba_small.yaml",
    "pretrained_encoder": "pretrained_weight/spatialmamba_small_224_1k.pth",
    "train_path": "/root/autodl-tmp/data/Gloucester I/train",
    "train_list": "/root/autodl-tmp/data/Gloucester I/train_list.txt",
    "test_path": "/root/autodl-tmp/data/Gloucester I/test",
    "test_list": "/root/autodl-tmp/data/Gloucester I/test_list.txt",
    "train_samples": 423,
    "test_samples": 104,
    "crop_size": 256,
    "batch_size": 8,
    "max_iters": 300000,
    "optimizer": "AdamW",
    "learning_rate": 1e-4,
    "weight_decay": 5e-4,
    "loss": "CrossEntropy + 2.0 * Lovasz-Softmax",
    "grad_clip_max_norm": 1.0,
    "eval_interval_iters": 2000,
    "model_selection_metric": "Pixel F1 on test set",
    "use_dfgf": True,
    "dfgf_frequency_gate": True,
    "dfgf_direction_selector": True,
    "use_cope_s": True,
    "cope_change_guide": True,
    "cope_direction_reweight": True,
    "cope_residual": True,
    "run_tag": "LEM-CD_COPE-S_DFGF_Gloucester_I_batch8_iter300000_20261001_171917",
    "best_checkpoint_iter": 36000,
    "best_checkpoint_path": str(CHECKPOINT),
    "inference_output_dir": str(INFER_MAP),
    "visualization_legend": {
        "TP": {"rgb": [255, 255, 255], "label_zh": "真阳性（检对变化）"},
        "FP": {"rgb": [255, 0, 0], "label_zh": "假阳性（虚警，红色不好）"},
        "FN": {"rgb": [0, 255, 0], "label_zh": "假阴性（漏检，绿色不好）"},
        "TN": {"rgb": [0, 0, 0], "label_zh": "真阴性（背景检对）"},
    },
}


def parse_log(text: str):
    lines = text.splitlines()
    loss_pat = re.compile(
        r"iter is (\d+), ce is ([\d.]+), lovasz is ([\d.]+), overall loss is ([\d.]+)"
    )
    losses = []
    seen = set()
    for ln in lines:
        m = loss_pat.search(ln)
        if not m:
            continue
        it = int(m.group(1))
        if it in seen:
            continue
        seen.add(it)
        losses.append(
            {
                "iter": it,
                "ce": float(m.group(2)),
                "lovasz": float(m.group(3)),
                "overall": float(m.group(4)),
            }
        )

    val_pat = re.compile(
        r"Recall rate is ([\d.]+), Precision rate is ([\d.]+), OA is ([\d.]+), "
        r"F1 score is ([\d.]+), IoU is ([\d.]+), Kappa coefficient is ([\d.]+)"
    )
    vals = []
    for ln in lines:
        m = val_pat.search(ln)
        if m:
            vals.append([float(x) for x in m.groups()])
    dedup = []
    for v in vals:
        if not dedup or v != dedup[-1]:
            dedup.append(v)

    saved = []
    for ln in lines:
        if "Saved best model to" in ln:
            p = ln.split("Saved best model to")[-1].strip()
            if p not in saved:
                saved.append(p)
    return losses, dedup, saved


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    text = LOG.read_text(encoding="utf-8", errors="replace")
    losses, vals, saved_ckpts = parse_log(text)

    eval_iters = [2000 * (i + 1) for i in range(len(vals))]
    val_rows = []
    for it, v in zip(eval_iters, vals):
        val_rows.append(
            {
                "iter": it,
                "recall": v[0],
                "precision": v[1],
                "oa": v[2],
                "f1": v[3],
                "iou": v[4],
                "kappa": v[5],
            }
        )

    best_row = max(val_rows, key=lambda r: r["f1"])
    infer_best = {
        "recall": 0.9963161321866403,
        "precision": 0.9968461939038439,
        "oa": 0.9995548541729267,
        "f1": 0.9965810925629185,
        "iou": 0.9931854833275685,
        "kappa": 0.9963430210700216,
        "source": "infer_MambaBCD_spatial_COPE_S_DFGF.py on test set",
    }

    export_meta = {
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_log": str(LOG),
        **CONFIG,
        "best_validation": best_row,
        "inference_metrics": infer_best,
        "saved_checkpoints": saved_ckpts,
    }

    (OUT / "experiment_config.json").write_text(
        json.dumps(export_meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    with (OUT / "validation_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "iter",
                "recall",
                "precision",
                "oa",
                "f1",
                "iou",
                "kappa",
            ],
        )
        w.writeheader()
        w.writerows(val_rows)

    with (OUT / "best_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["stage", "iter", *infer_best.keys()])
        w.writeheader()
        w.writerow(
            {
                "stage": "validation_during_training",
                "iter": best_row["iter"],
                **{k: best_row[k] for k in infer_best if k in best_row},
            }
        )
        w.writerow({"stage": "inference_rerun", "iter": 36000, **infer_best})

    with (OUT / "training_loss.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["iter", "ce", "lovasz", "overall"])
        w.writeheader()
        w.writerows(losses)

    # Downsampled loss for lighter plots
    step = 50
    loss_ds = losses[::step]
    with (OUT / "training_loss_downsampled.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        w = csv.DictWriter(f, fieldnames=["iter", "ce", "lovasz", "overall"])
        w.writeheader()
        w.writerows(loss_ds)

    overall = [x["overall"] for x in losses]
    nz = [x for x in overall if x > 0.001]
    loss_summary = {
        "num_logged_iters": len(losses),
        "overall_min": min(overall),
        "overall_max": max(overall),
        "overall_mean": st.mean(overall),
        "overall_median": st.median(overall),
        "overall_mean_excluding_near_zero": st.mean(nz) if nz else None,
        "first_iter": losses[0]["iter"],
        "last_iter": losses[-1]["iter"],
        "last_overall": losses[-1]["overall"],
    }
    (OUT / "loss_summary.json").write_text(
        json.dumps(loss_summary, indent=2), encoding="utf-8"
    )

    # --- Figures ---
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "figure.dpi": 150,
        }
    )

    iters = [r["iter"] for r in val_rows]
    f1s = [r["f1"] for r in val_rows]
    ious = [r["iou"] for r in val_rows]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(iters, f1s, marker="o", label="F1", color="#2563eb", linewidth=1.8)
    ax.plot(iters, ious, marker="s", label="IoU", color="#059669", linewidth=1.8)
    ax.axvline(36000, color="#dc2626", linestyle="--", linewidth=1, label="Best ckpt (36000)")
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Score")
    ax.set_title("Gloucester I test metrics vs. training iteration")
    ax.set_ylim(0.88, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "metrics_f1_iou_vs_iter.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    li = [r["iter"] for r in loss_ds]
    lo = [r["overall"] for r in loss_ds]
    ax.plot(li, lo, color="#7c3aed", linewidth=1.2)
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Loss (CE + 2×Lovász)")
    ax.set_title("Training loss (downsampled every 50 logged steps)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "training_loss_curve.png")
    plt.close(fig)

    labels = ["Recall", "Precision", "OA", "F1", "IoU", "Kappa"]
    values = [
        infer_best["recall"],
        infer_best["precision"],
        infer_best["oa"],
        infer_best["f1"],
        infer_best["iou"],
        infer_best["kappa"],
    ]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, values, color="#0ea5e9", edgecolor="#0369a1")
    ax.set_ylim(0.94, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Best model inference metrics (iter 36000)")
    for b, v in zip(bars, values):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v + 0.001,
            f"{v:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(FIG / "best_metrics_bar.png")
    plt.close(fig)

    # Error-map color legend
    fig, ax = plt.subplots(figsize=(6, 1.8))
    ax.axis("off")
    colors = [
        ("TP (white)", "#ffffff", "Correct change"),
        ("FP (red)", "#ff0000", "False alarm"),
        ("FN (green)", "#00ff00", "Missed change"),
        ("TN (black)", "#000000", "Correct background"),
    ]
    for i, (name, hex_c, desc) in enumerate(colors):
        ax.add_patch(plt.Rectangle((i, 0), 0.9, 1, facecolor=hex_c, edgecolor="#333"))
        ax.text(i + 0.45, -0.35, f"{name}\n{desc}", ha="center", va="top", fontsize=8)
    ax.set_xlim(-0.1, 4)
    ax.set_ylim(-0.8, 1.2)
    ax.set_title("Inference change_map color encoding (pred vs. GT)")
    fig.tight_layout()
    fig.savefig(FIG / "error_map_color_legend.png")
    plt.close(fig)

    # Markdown report
    md = f"""# Gloucester I 实验报告（COPE-S + DFGF）

导出时间（UTC）：{export_meta["exported_at_utc"]}

## 1. 实验设置

| 项目 | 内容 |
|------|------|
| 数据集 | Gloucester I（训练 {CONFIG["train_samples"]} / 测试 {CONFIG["test_samples"]}） |
| 输入尺寸 | {CONFIG["crop_size"]}×{CONFIG["crop_size"]} |
| 模型 | {CONFIG["model_stack"]} |
| 训练预算 | max_iters={CONFIG["max_iters"]}（batch={CONFIG["batch_size"]} → 37500 步） |
| 优化器 | {CONFIG["optimizer"]}，lr={CONFIG["learning_rate"]}，wd={CONFIG["weight_decay"]} |
| 损失 | {CONFIG["loss"]} |
| 验证 | 每 {CONFIG["eval_interval_iters"]} iter，测试集，按 {CONFIG["model_selection_metric"]} 存 best |
| DFGF | use_dfgf={CONFIG["use_dfgf"]}，frequency_gate={CONFIG["dfgf_frequency_gate"]}，direction_selector={CONFIG["dfgf_direction_selector"]} |
| COPE-S | use_cope_s={CONFIG["use_cope_s"]}，CGG={CONFIG["cope_change_guide"]}，DRS={CONFIG["cope_direction_reweight"]}，RCF={CONFIG["cope_residual"]} |

## 2. 最优结果（iter {best_row["iter"]}）

| 指标 | 数值 |
|------|------|
| Recall | {infer_best["recall"]:.6f} |
| Precision | {infer_best["precision"]:.6f} |
| OA | {infer_best["oa"]:.6f} |
| **F1** | **{infer_best["f1"]:.6f}** |
| IoU | {infer_best["iou"]:.6f} |
| Kappa | {infer_best["kappa"]:.6f} |

权重：`{CHECKPOINT}`

推理四色图：`{INFER_MAP}`（104 张 PNG）

## 3. 训练 Loss 摘要

- 记录步数：{loss_summary["num_logged_iters"]}（每 10 iter 打印）
- overall：min={loss_summary["overall_min"]:.4f}，max={loss_summary["overall_max"]:.4f}，median={loss_summary["overall_median"]:.4f}
- 末步 iter {loss_summary["last_iter"]}：overall={loss_summary["last_overall"]:.4f}

## 4. 附件文件

| 文件 | 说明 |
|------|------|
| `experiment_config.json` | 完整参数与路径 |
| `validation_metrics.csv` | 每 2000 iter 测试指标 |
| `best_metrics.csv` | 训练内 best vs 推理复现 |
| `training_loss.csv` | 全量 loss 日志 |
| `training_loss_downsampled.csv` | 绘图用降采样 loss |
| `figures/metrics_f1_iou_vs_iter.png` | F1/IoU 曲线 |
| `figures/training_loss_curve.png` | Loss 曲线 |
| `figures/best_metrics_bar.png` | 最优指标柱状图 |
| `figures/error_map_color_legend.png` | 四色图图例 |

## 5. 四色误差图说明

- **白色 TP**：变化检对  
- **红色 FP**：虚警（越多越差）  
- **绿色 FN**：漏检（越多越差）  
- **黑色 TN**：背景检对  

"""
    (OUT / "REPORT_Gloucester_I_zh.md").write_text(md, encoding="utf-8")

    print(f"Exported to {OUT}")
    print(f"  validation rows: {len(val_rows)}")
    print(f"  loss rows: {len(losses)}")
    print(f"  figures: {len(list(FIG.glob('*.png')))}")


if __name__ == "__main__":
    main()
