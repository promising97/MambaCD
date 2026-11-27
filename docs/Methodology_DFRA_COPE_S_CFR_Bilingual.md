# III. Methodology / 方法论

## A. Problem Formulation and Network Overview / 问题定义与网络概述

Given a pair of co-registered remote-sensing images acquired at two times, denoted by \(\mathbf{X}^{\mathrm{pre}},\mathbf{X}^{\mathrm{post}}\in\mathbb{R}^{3\times H\times W}\), the objective of binary change detection is to predict a pixel-wise label map \(\mathbf{Y}\in\{0,1\}^{H\times W}\). The proposed network follows an encoder–decoder architecture. A weight-shared SpatialMamba encoder first extracts hierarchical representations from the two images:

给定一对分别在两个时相获取且已配准的遥感影像，记为 \(\mathbf{X}^{\mathrm{pre}},\mathbf{X}^{\mathrm{post}}\in\mathbb{R}^{3\times H\times W}\)，二值变化检测的目标是预测像素级标签图 \(\mathbf{Y}\in\{0,1\}^{H\times W}\)。所提出的网络采用编码器–解码器架构。首先，权重共享的 SpatialMamba 编码器从两幅影像中提取层次化表征：

\[
\left\{\mathbf{F}^{t}_{l}\right\}_{l=1}^{4}
=\mathcal{E}\!\left(\mathbf{X}^{t}\right),\qquad
t\in\{\mathrm{pre},\mathrm{post}\},
\tag{1}
\]

where \(l=1,\ldots,4\) indexes feature levels from high to low spatial resolution. Sharing \(\mathcal{E}\) constrains the two temporal streams to use the same representation space, so that their differences are less affected by branch-specific parameters.

其中，\(l=1,\ldots,4\) 表示由高到低空间分辨率的特征层级。共享 \(\mathcal{E}\) 可约束两个时相分支使用相同的表征空间，从而减弱其差异受到分支特异性参数的影响。

As illustrated in Fig. X, the decoder proceeds from the coarsest level to the finest level. At level \(l\), the bi-temporal features are concatenated and projected to a common width \(C=128\) using a \(3\times3\) convolution:

如图 X 所示，解码器从最粗层级逐步处理至最细层级。在第 \(l\) 层，将双时相特征拼接，并通过 \(3\times3\) 卷积将其投影至统一通道宽度 \(C=128\)：

\[
\mathbf{z}_{l}
=\phi_l\!\left(
\left[\mathbf{F}^{\mathrm{pre}}_{l},
\mathbf{F}^{\mathrm{post}}_{l}\right]
\right),
\tag{2}
\]

where \([\cdot,\cdot]\) denotes channel-wise concatenation. Each \(\mathbf{z}_{l}\) is processed by two complementary branches: DFRA models directional structures with content-adaptive frequency reliability, whereas COPE-S enhances directional details under explicit bi-temporal change guidance. Their outputs are fused by a \(1\times1\) convolution:

其中，\([\cdot,\cdot]\) 表示沿通道维拼接。每个 \(\mathbf{z}_{l}\) 均由两个互补分支处理：DFRA 利用内容自适应的频率可靠性对方向结构进行建模，而 COPE-S 则在显式双时相变化引导下增强方向细节。两个分支的输出通过 \(1\times1\) 卷积进行融合：

\[
\widetilde{\mathbf{p}}_{l}
=\rho_l\!\left(
\left[
\mathcal{D}_{l}(\mathbf{z}_{l}),
\mathcal{C}_{l}\!\left(
\mathbf{z}_{l},
\mathbf{F}^{\mathrm{pre}}_{l},
\mathbf{F}^{\mathrm{post}}_{l}
\right)
\right]
\right),
\tag{3}
\]

where \(\mathcal{D}_{l}\), \(\mathcal{C}_{l}\), and \(\rho_l\) denote DFRA, COPE-S, and the branch-fusion projection, respectively. The coarsest output initializes the feature pyramid, \(\mathbf{p}_{4}=\widetilde{\mathbf{p}}_{4}\). The remaining levels progressively combine the current-scale representation with the upsampled coarser output and refine the result using CFR:

其中，\(\mathcal{D}_{l}\)、\(\mathcal{C}_{l}\) 和 \(\rho_l\) 分别表示 DFRA、COPE-S 和分支融合投影。最粗层级的输出用于初始化特征金字塔，即 \(\mathbf{p}_{4}=\widetilde{\mathbf{p}}_{4}\)。其余层级逐步将当前尺度表征与上采样后的粗尺度输出相结合，并利用 CFR 对结果进行细化：

\[
\mathbf{p}_{l}
=\mathcal{R}_{l}\!\left(
\widetilde{\mathbf{p}}_{l}
+\mathcal{U}\!\left(\mathbf{p}_{l+1}\right)
\right),\qquad l=3,2,1,
\tag{4}
\]

where \(\mathcal{U}\) is bilinear interpolation to the current spatial size and \(\mathcal{R}_{l}\) is CFR. Finally, a \(1\times1\) classifier maps \(\mathbf{p}_{1}\) to two-class logits, which are bilinearly resized to \(H\times W\).

其中，\(\mathcal{U}\) 表示将特征双线性插值至当前空间尺寸，\(\mathcal{R}_{l}\) 表示 CFR。最后，\(1\times1\) 分类器将 \(\mathbf{p}_{1}\) 映射为两类 logits，并通过双线性插值将其尺寸调整为 \(H\times W\)。

## B. Directional Frequency-Reliability Aggregation / 方向频率可靠性聚合

Remote-sensing changes are often expressed through boundaries, elongated objects, and texture variations with strong directional and frequency characteristics. Applying a single spatial filter treats all directions and spectral components uniformly, whereas unconstrained frequency modulation can amplify acquisition noise and irrelevant high-frequency responses. DFRA addresses this limitation by combining fixed directional priors, a content-adaptive frequency reliability mask, and learned direction weighting.

遥感影像中的变化通常表现为边界、狭长目标以及具有显著方向和频率特性的纹理变化。单一空间滤波器会对所有方向和频谱分量进行同等处理，而无约束的频率调制则可能放大采集噪声和无关的高频响应。DFRA 通过结合固定方向先验、内容自适应频率可靠性掩码以及学习式方向加权来克服这一局限。

Given \(\mathbf{z}_{l}\), a \(1\times1\) projection followed by a depth-wise \(3\times3\) convolution produces two features, \(\mathbf{q}_{l}\) and \(\mathbf{v}_{l}\):

给定 \(\mathbf{z}_{l}\)，首先通过 \(1\times1\) 投影和逐通道 \(3\times3\) 卷积生成两个特征 \(\mathbf{q}_{l}\) 与 \(\mathbf{v}_{l}\)：

\[
[\mathbf{q}_{l},\mathbf{v}_{l}]
=\operatorname{DWConv}_{3\times3}
\left(\operatorname{Conv}_{1\times1}(\mathbf{z}_{l})\right).
\tag{5}
\]

The \(\mathbf{q}_{l}\) branch extracts direction-frequency responses, while \(\mathbf{v}_{l}\) supplies spatial modulation. We divide \(\mathbf{q}_{l}\) into \(O\) channel groups, \(\{\mathbf{q}_{l,o}\}_{o=1}^{O}\), and assign one Gabor kernel to each orientation. For orientation \(\theta_o\), the fixed kernel is

\(\mathbf{q}_{l}\) 分支提取方向–频率响应，\(\mathbf{v}_{l}\) 则提供空间调制。我们将 \(\mathbf{q}_{l}\) 划分为 \(O\) 个通道组 \(\{\mathbf{q}_{l,o}\}_{o=1}^{O}\)，并为每个方向分配一个 Gabor 核。对于方向 \(\theta_o\)，固定卷积核定义为

\[
g_o(x,y)
=\exp\!\left[
-\frac{x_o^2+\gamma^2y_o^2}{2\sigma^2}
\right]
\cos\!\left(\frac{2\pi x_o}{\lambda}\right),
\tag{6}
\]

where \(x_o=x\cos\theta_o+y\sin\theta_o\), \(y_o=-x\sin\theta_o+y\cos\theta_o\), and \(\sigma\), \(\lambda\), and \(\gamma\) control the Gaussian scale, wavelength, and aspect ratio. The zero-mean, \(L_1\)-normalized kernel yields the directional subband

其中，\(x_o=x\cos\theta_o+y\sin\theta_o\)，\(y_o=-x\sin\theta_o+y\cos\theta_o\)，\(\sigma\)、\(\lambda\) 和 \(\gamma\) 分别控制高斯尺度、波长和纵横比。经零均值化和 \(L_1\) 归一化后的卷积核生成方向子带：

\[
\mathbf{s}_{l,o}=g_o * \mathbf{q}_{l,o}.
\tag{7}
\]

### 1) Frequency reliability gate / 频率可靠性门控

The designed gate estimates which Fourier coefficients of each directional subband are reliable for the current sample. We first compute its orthonormal real-valued two-dimensional Fourier transform:

所设计的门控用于估计当前样本中每个方向子带的哪些傅里叶系数是可靠的。首先，计算其正交归一的实值二维傅里叶变换：

\[
\mathbf{S}_{l,o}=\operatorname{rFFT2}(\mathbf{s}_{l,o}).
\tag{8}
\]

To reduce sensitivity to activation scale, the channel-averaged log magnitude is standardized over the frequency grid:

为降低对激活尺度的敏感性，首先计算通道平均对数幅值，然后在频率网格上进行标准化：

\[
\widehat{\mathbf{A}}_{l,o}
=\operatorname{Std}\!\left[
\frac{1}{C_o}
\sum_{c=1}^{C_o}
\log\!\left(1+\left|\mathbf{S}^{(c)}_{l,o}\right|\right)
\right].
\tag{9}
\]

A lightweight convolutional refiner \(\psi_o(\cdot)\) predicts content-dependent logits from \(\widehat{\mathbf{A}}_{l,o}\). In parallel, a learnable orientation-specific prior \(\mathbf{P}_o\) is interpolated to the current rFFT grid. The reliability mask is

轻量级卷积细化器 \(\psi_o(\cdot)\) 根据 \(\widehat{\mathbf{A}}_{l,o}\) 预测内容相关的 logits。与此同时，将可学习的方向特异性先验 \(\mathbf{P}_o\) 插值至当前 rFFT 网格。可靠性掩码定义为

\[
\mathbf{M}_{l,o}
=\operatorname{sigmoid}\!\left(
\operatorname{Resize}(\mathbf{P}_o)
+\psi_o(\widehat{\mathbf{A}}_{l,o})
\right).
\tag{10}
\]

Because \(\mathbf{M}_{l,o}\) is real valued, multiplying it with the complex spectrum changes the magnitude while retaining the original phase:

由于 \(\mathbf{M}_{l,o}\) 为实值，将其与复数频谱相乘仅改变幅值，同时保留原始相位：

\[
\widehat{\mathbf{s}}_{l,o}
=\operatorname{irFFT2}
\left(\mathbf{S}_{l,o}\odot\mathbf{M}_{l,o}\right).
\tag{11}
\]

This formulation separates two roles: \(\mathbf{P}_o\) captures a resolution-independent frequency preference for each orientation, whereas \(\psi_o\) adapts that preference to the observed content.

该形式将两个作用解耦：\(\mathbf{P}_o\) 描述各方向上与分辨率无关的频率偏好，而 \(\psi_o\) 则根据观测内容对该偏好进行自适应调整。

### 2) Direction selection and spatial modulation / 方向选择与空间调制

The directional responses are further weighted according to the input. Global average pooling and a two-layer multilayer perceptron generate normalized orientation coefficients:

根据输入对方向响应进一步加权。全局平均池化和两层多层感知机生成归一化的方向系数：

\[
\boldsymbol{\omega}_{l}
=O\cdot\operatorname{softmax}
\left(
\operatorname{MLP}(\operatorname{GAP}(\mathbf{q}_{l}))
\right).
\tag{12}
\]

The factor \(O\) keeps the initial mean gain of each direction close to one. The direction-frequency representation is then obtained through residual aggregation:

系数 \(O\) 使每个方向的初始平均增益接近 1。随后，通过残差聚合获得方向–频率表征：

\[
\mathbf{q}'_{l}
=\mathbf{q}_{l}
+\alpha
\left[
\omega_{l,1}\widehat{\mathbf{s}}_{l,1},
\ldots,
\omega_{l,O}\widehat{\mathbf{s}}_{l,O}
\right],
\tag{13}
\]

where \(\alpha\) is learnable. For the complementary branch, a depth-wise convolution followed by a sigmoid produces a spatial gate:

其中，\(\alpha\) 为可学习参数。对于互补分支，使用逐通道卷积及 sigmoid 函数生成空间门控：

\[
\mathbf{v}'_{l}
=\mathbf{v}_{l}\odot
\operatorname{sigmoid}
\left(\operatorname{DWConv}_{3\times3}(\mathbf{v}_{l})\right).
\tag{14}
\]

After normalization, \(\mathbf{q}'_{l}\) is modulated by \(\mathbf{v}'_{l}\), recalibrated using a pooled channel response, and projected to \(C\) channels. This coupling preserves the direction-frequency evidence selected by DFRA while suppressing spatially irrelevant responses.

归一化后，\(\mathbf{q}'_{l}\) 由 \(\mathbf{v}'_{l}\) 调制，再利用池化后的通道响应进行重校准，最后投影至 \(C\) 个通道。该耦合机制在保留 DFRA 所筛选方向–频率证据的同时，抑制空间无关响应。

## C. Change-Oriented Pyramid Enhancement for Near-Homogeneous Data / 面向近同质数据的变化导向金字塔增强

DFRA operates on the fused decoder feature and therefore does not explicitly distinguish whether a directional response is supported by the original temporal pair. COPE-S complements it by deriving change cues directly from the two encoder features and using these cues to regulate directional experts. This design is intended for same-domain or near-homogeneous bi-temporal observations, for which aligned temporal differences provide useful spatial guidance.

DFRA 作用于融合后的解码器特征，因此无法显式判别某一方向响应是否得到原始双时相影像对的支持。作为补充，COPE-S 直接从两个编码器特征中提取变化线索，并利用这些线索调控方向专家。该设计面向同域或近同质双时相观测；对于此类数据，对齐后的时相差异能够提供有效的空间引导。

At level \(l\), separate \(1\times1\) projections and channel-wise layer normalization align the two encoder features:

在第 \(l\) 层，分别使用 \(1\times1\) 投影和沿通道维的层归一化对齐两个编码器特征：

\[
\overline{\mathbf{F}}^{t}_{l}
=\operatorname{LN}
\left(
\operatorname{Conv}_{1\times1}
(\mathbf{F}^{t}_{l})
\right),
\quad t\in\{\mathrm{pre},\mathrm{post}\}.
\tag{15}
\]

Their absolute difference,

二者的绝对差

\[
\boldsymbol{\Delta}_{l}
=\left|
\overline{\mathbf{F}}^{\mathrm{pre}}_{l}
-\overline{\mathbf{F}}^{\mathrm{post}}_{l}
\right|,
\tag{16}
\]

drives two complementary controls. The change-guidance generator produces a pixel-wise change mask

用于驱动两种互补控制机制。变化引导生成器产生像素级变化掩码

\[
\mathbf{m}_{l}
=\operatorname{sigmoid}
\left(
\operatorname{Conv}_{3\times3}(\boldsymbol{\Delta}_{l})
\right),
\tag{17}
\]

whereas the direction-reweighting selector produces sample-wise orientation weights

而方向重加权选择器则产生样本级方向权重

\[
\mathbf{a}_{l}
=\operatorname{softmax}
\left(
\operatorname{MLP}(\operatorname{GAP}(\boldsymbol{\Delta}_{l}))
\right).
\tag{18}
\]

The decoder feature \(\mathbf{z}_{l}\) is split into \(O\) channel groups. Each oriented expert applies a fixed Gabor filter followed by an independent depth-wise \(3\times3\) refiner:

将解码器特征 \(\mathbf{z}_{l}\) 划分为 \(O\) 个通道组。每个方向专家首先应用固定 Gabor 滤波器，随后使用独立的逐通道 \(3\times3\) 细化器：

\[
\mathbf{e}_{l,o}
=g_o * \mathbf{z}_{l,o}
+\eta_o\!\left(g_o * \mathbf{z}_{l,o}\right),
\tag{19}
\]

where \(\eta_o\) consists of depth-wise convolution, batch normalization, and SiLU activation. Each expert response is jointly modulated by the spatial change mask and the corresponding direction coefficient:

其中，\(\eta_o\) 由逐通道卷积、批归一化和 SiLU 激活函数构成。每个专家响应均由空间变化掩码和相应方向系数联合调制：

\[
\widetilde{\mathbf{e}}_{l,o}
=a_{l,o}\,
\mathbf{m}_{l}\odot\mathbf{e}_{l,o}.
\tag{20}
\]

The modulated responses are concatenated, channel shuffled to promote cross-orientation interaction, and projected back to \(C\) channels. Residual change fusion then gives

调制后的响应经拼接和通道混洗以促进跨方向交互，随后投影回 \(C\) 个通道。残差变化融合结果为

\[
\mathcal{C}_{l}
=\mathbf{z}_{l}
+\beta\,
\pi_l\!\left(
\operatorname{Shuffle}
\left[
\widetilde{\mathbf{e}}_{l,1},\ldots,
\widetilde{\mathbf{e}}_{l,O}
\right]
\right),
\tag{21}
\]

where \(\pi_l\) is a \(1\times1\) projection with batch normalization and \(\beta\) is learnable. Thus, the spatial mask identifies where directional enhancement is needed, while the orientation weights determine which directional experts should dominate for the current image pair.

其中，\(\pi_l\) 表示带批归一化的 \(1\times1\) 投影，\(\beta\) 为可学习参数。因此，空间掩码用于确定需要方向增强的位置，而方向权重则决定当前影像对中应占主导地位的方向专家。

## D. Cross-Scale Fusion Refinement / 跨尺度融合细化

Directly adding adjacent pyramid levels can introduce interpolation artifacts and mix coarse semantics with fine spatial responses without further selection. CFR is therefore applied after each upsample-and-add operation in the last three decoder stages. For an aggregated feature \(\mathbf{h}_{l}=\widetilde{\mathbf{p}}_{l}+\mathcal{U}(\mathbf{p}_{l+1})\), CFR first expands its channels using a \(1\times1\) projection and performs depth-wise spatial mixing:

直接相加相邻金字塔层级可能引入插值伪影，并在缺乏进一步筛选的情况下混合粗粒度语义与精细空间响应。因此，在解码器最后三个阶段中，每次执行上采样与相加操作后均应用 CFR。对于聚合特征 \(\mathbf{h}_{l}=\widetilde{\mathbf{p}}_{l}+\mathcal{U}(\mathbf{p}_{l+1})\)，CFR 首先通过 \(1\times1\) 投影扩展其通道，再进行逐通道空间混合：

\[
[\mathbf{h}^{(1)}_{l},\mathbf{h}^{(2)}_{l}]
=\operatorname{Split}
\left(
\operatorname{DWConv}_{3\times3}
\left(
\operatorname{Conv}_{1\times1}(\mathbf{h}_{l})
\right)
\right).
\tag{22}
\]

The two branches form a gated feed-forward interaction,

两个分支构成门控前馈交互：

\[
\mathbf{h}'_{l}
=\operatorname{GELU}(\mathbf{h}^{(1)}_{l})
\odot\mathbf{h}^{(2)}_{l},
\tag{23}
\]

followed by channel projection and a Gabor-based frequency transform:

随后执行通道投影和基于 Gabor 的频率变换：

\[
\mathbf{p}_{l}
=\mathcal{G}
\left(
\operatorname{Conv}_{1\times1}(\mathbf{h}'_{l})
\right),
\tag{24}
\]

where \(\mathcal{G}\) denotes the frequency-aware transform used by CFR. Unlike DFRA, CFR does not use the proposed rFFT reliability mask; its role is to refine the feature generated by cross-scale addition before it is propagated to the next, finer decoder level.

其中，\(\mathcal{G}\) 表示 CFR 所使用的频率感知变换。不同于 DFRA，CFR 不使用所提出的 rFFT 可靠性掩码；其作用是在跨尺度相加产生的特征传播至下一个更精细的解码器层级之前对其进行细化。

## E. Optimization Objective and Implementation / 优化目标与实现细节

The network is optimized using a combination of pixel-wise cross-entropy and Lovász–Softmax losses:

网络采用像素级交叉熵损失与 Lovász–Softmax 损失的组合进行优化：

\[
\mathcal{L}
=\mathcal{L}_{\mathrm{CE}}
+2\mathcal{L}_{\mathrm{Lov\acute{a}sz}},
\tag{25}
\]

where pixels labeled 255 are ignored. The current implementation uses AdamW with an initial learning rate of \(1\times10^{-4}\), a weight decay of \(5\times10^{-4}\), and gradient clipping with a maximum norm of 1.0. Input crops are \(256\times256\), and the default batch size is 8.

其中，忽略标签为 255 的像素。当前实现采用 AdamW，初始学习率为 \(1\times10^{-4}\)，权重衰减为 \(5\times10^{-4}\)，并使用最大范数为 1.0 的梯度裁剪。输入裁块尺寸为 \(256\times256\)，默认批大小为 8。

For both DFRA and COPE-S, \(O=4\) orientations (\(0^\circ\), \(45^\circ\), \(90^\circ\), and \(135^\circ\)) and \(5\times5\) Gabor kernels are used. The default Gabor parameters are \(\sigma=2.0\) and \(\lambda=3.0\). DFRA additionally uses \(\gamma=0.5\), a learnable frequency-prior grid of size \(16\times(16/2+1)\), and an initial residual scale of 0.1. COPE-S uses an aspect ratio of 0.5 and also initializes its residual scale to 0.1. CFR uses an expansion factor of 3. All decoder levels operate at \(C=128\) channels.

DFRA 和 COPE-S 均采用 \(O=4\) 个方向（\(0^\circ\)、\(45^\circ\)、\(90^\circ\) 和 \(135^\circ\)）以及 \(5\times5\) Gabor 核。默认 Gabor 参数为 \(\sigma=2.0\) 和 \(\lambda=3.0\)。DFRA 还采用 \(\gamma=0.5\)、大小为 \(16\times(16/2+1)\) 的可学习频率先验网格，以及初始值为 0.1 的残差缩放系数。COPE-S 使用 0.5 的纵横比，其残差缩放系数同样初始化为 0.1。CFR 的扩展因子为 3。解码器各层级的通道数均为 \(C=128\)。

---

## Terminology Ledger / 术语表

| Canonical term / 规范术语 | Definition used in this draft / 本稿所用定义 | Deprecated/source-code variant / 已弃用术语或源代码变体 |
|---|---|---|
| DFRA | Directional frequency-reliability aggregation module / 方向频率可靠性聚合模块 | `StarModule`, DFGF-Star / `StarModule`、DFGF-Star |
| frequency reliability gate / 频率可靠性门控 | The designed rFFT magnitude mask inside DFRA / DFRA 内所设计的 rFFT 幅值掩码 | DFGF |
| COPE-S | Change-oriented pyramid enhancement for same-/near-domain data / 面向同域或近同质数据的变化导向金字塔增强 | modified GaborFPU |
| CFR | Cross-scale fusion refinement / 跨尺度融合细化 | `DFFN`, `smooth_layer_*` / `DFFN`、`smooth_layer_*` |
| SpatialMamba encoder / SpatialMamba 编码器 | Weight-shared hierarchical encoder / 权重共享的层次化编码器 | `Backbone_SpatialMamba` |
| OEB | Oriented expert bank / 方向专家库 | `OrientedGaborExpert` |
| CGG | Change-guidance generator / 变化引导生成器 | `change_mask` |
| DRS | Direction-reweighting selector / 方向重加权选择器 | `direction_selector` |
| RCF | Residual change fusion / 残差变化融合 | residual output in `COPES` / `COPES` 中的残差输出 |

## Assumptions or Missing Inputs / 假设或缺失输入

1. The expansions of **DFRA**, **CFR**, **CGG**, **DRS**, **OEB**, and **RCF** must be confirmed by the authors. The prose above assigns expansions that are consistent with the implemented operations, but these names are not declared in the current source code.

   **DFRA**、**CFR**、**CGG**、**DRS**、**OEB** 和 **RCF** 的全称须由作者确认。上述正文给出的全称与已实现的操作一致，但当前源代码中并未声明这些名称。

2. Replace “Fig. X” with the final framework-figure number.

   将“图 X”替换为最终框架图编号。

3. Add citations for SpatialMamba, Gabor filtering, Lovász–Softmax, and any inherited frequency-transform implementation after checking the original papers.

   在核查原始论文后，为 SpatialMamba、Gabor 滤波、Lovász–Softmax 以及任何沿用的频率变换实现补充引文。

4. Dataset-specific preprocessing, augmentation, train/validation/test splits, hardware, random seeds, and learning-rate scheduling are not fully established by the inspected model code and should be documented in the Experiments section.

   根据所检查的模型代码，尚无法完全确定特定于数据集的预处理、数据增强、训练/验证/测试集划分、硬件、随机种子和学习率调度；应在实验部分中予以说明。

## Claim–Evidence Map / 论断–证据映射

| Claim / 论断 | Evidence / 证据 | Status / 状态 |
|---|---|---|
| The encoder is weight shared / 编码器权重共享 | The same `self.encoder` is called for pre- and post-images / 对变化前和变化后影像调用同一个 `self.encoder` | Supported / 已支持 |
| DFRA preserves Fourier phase / DFRA 保留傅里叶相位 | A real mask multiplies the complex rFFT spectrum before `irfft2` / 在执行 `irfft2` 之前，以实值掩码乘以复数 rFFT 频谱 | Supported / 已支持 |
| COPE-S uses explicit bi-temporal change guidance / COPE-S 使用显式双时相变化引导 | Its mask and direction weights are computed from projected absolute temporal differences / 其掩码和方向权重根据投影后的时相绝对差计算得到 | Supported / 已支持 |
| Decoder fusion is coarse-to-fine / 解码器采用由粗到细的融合方式 | Stages II–IV use bilinear upsampling, addition, and CFR / 第 II–IV 阶段使用双线性上采样、相加和 CFR | Supported / 已支持 |
| DFRA, COPE-S, and CFR improve accuracy / DFRA、COPE-S 和 CFR 可提高精度 | Requires controlled ablations and benchmark results / 需要受控消融实验和基准测试结果 | Evidence needed / 尚需证据 |
