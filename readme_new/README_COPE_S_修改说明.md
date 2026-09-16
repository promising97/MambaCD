# COPE-S 代码接入与修改说明

本包所有文件均使用新名称，不覆盖原来的 GaborFPU/DFGF 实验文件。

## 文件放置位置

| 新文件 | 放置目录 | 用途 |
|---|---|---|
| `COPE_S.py` | `MambaCD/changedetection/models/` | COPE-S 核心模块 |
| `ChangeDecoder_Star_COPE_S_DFGF.py` | `MambaCD/changedetection/models/` | 四级 COPE-S 解码器 |
| `MambaBCD_spatial_COPE_S_DFGF.py` | `MambaCD/changedetection/models/` | 新模型入口 |
| `StarIR_gabor_DFGF_COPE_S.py` | `MambaCD/changedetection/models/` | DFGF-Star/DFFN 隔离副本 |
| `train_MambaBCD_spatial_COPE_S_DFGF.py` | `MambaCD/changedetection/script/` | 训练与消融入口 |
| `infer_MambaBCD_spatial_COPE_S_DFGF.py` | `MambaCD/changedetection/script/` | 严格推理入口 |

旧文件全部保留，不要删除或覆盖。

## 核心结构变化

旧 GaborFPU：

```text
OPE特征 → 通道分组 → 共享Gabor处理 → Concat → 1×1 Conv
```

新 COPE-S：

```text
双时相同尺度编码特征
  → 独立1×1投影 + Channel-LN
  → 绝对差异
  ├→ CGG空间变化引导图
  └→ DRS四方向样本权重

OPE特征
  → 四组通道
  → 0°/45°/90°/135°方向专属Gabor专家（OEB）
  → 空间引导 × 方向权重
  → Channel Shuffle + 1×1投影
  → 小尺度残差融合（RCF）
```

四级解码均执行：

```text
DFGF-Star(z_k) ∥ COPE-S(z_k, F_pre_k, F_post_k)
→ Concat → 1×1 Fusion → 自顶向下DFFN
```

COPE-S 不增加 FFT；频率可靠性仍由 DFGF 负责。

## 服务器上需要修改的位置

1. 如果项目不在 `/root/autodl-tmp/MambaCD`，修改新训练脚本顶部的 `PROJECT_ROOT`。
2. 同步修改新推理脚本开头的 `sys.path.append(...)`。
3. 在两个新脚本的参数区修改训练集、测试集、预训练权重、保存目录和 `--resume` 路径，或通过命令行覆盖。
4. 六个 `.py` 文件必须按上表放置，文件名不要再次改变。

## 完整训练

```bash
python MambaCD/changedetection/script/train_MambaBCD_spatial_COPE_S_DFGF.py \
  --use-cope-s \
  --cope-change-guide \
  --cope-direction-reweight \
  --cope-residual
```

日志应出现：

```text
COPE-S configuration: use_cope_s=True, change_guide=True, direction_reweight=True, residual=True
```

## 推理

```bash
python MambaCD/changedetection/script/infer_MambaBCD_spatial_COPE_S_DFGF.py \
  --use-cope-s \
  --cope-change-guide \
  --cope-direction-reweight \
  --cope-residual \
  --resume /实际路径/COPE-S模型.pth
```

训练和推理的 COPE-S 开关必须完全一致。正式推理必须使用新结构训练出的 checkpoint。

## 消融开关

| 实验 | 开关 |
|---|---|
| 原 GaborFPU 基线 | 使用旧训练脚本 |
| 移除 COPE-S 分支 | `--no-use-cope-s` |
| OEB + RCF | `--no-cope-change-guide --no-cope-direction-reweight --cope-residual` |
| OEB + CGG + RCF | `--cope-change-guide --no-cope-direction-reweight --cope-residual` |
| 完整 COPE-S | `--cope-change-guide --cope-direction-reweight --cope-residual` |
| 去除残差 | `--cope-change-guide --cope-direction-reweight --no-cope-residual` |

## 适用数据

COPE-S 适用于伪SAR—SAR、SAR—SAR或已经对齐到近似同一模态空间的数据。如果输入是未经转换的原始光学—真实SAR，应改用 COPE-X。

## 验证说明

代码已进行 Python 语法和模块导入名称检查。当前执行环境没有 PyTorch，因此正式训练前请先在服务器上用 batch size 1、256×256 输入完成一次前向测试。
