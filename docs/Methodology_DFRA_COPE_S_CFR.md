# III. Methodology

## A. Problem Formulation and Network Overview

Given a pair of co-registered remote-sensing images acquired at two times, denoted by \(\mathbf{X}^{\mathrm{pre}},\mathbf{X}^{\mathrm{post}}\in\mathbb{R}^{3\times H\times W}\), the objective of binary change detection is to predict a pixel-wise label map \(\mathbf{Y}\in\{0,1\}^{H\times W}\). The proposed network follows an encoder–decoder architecture. A weight-shared SpatialMamba encoder first extracts hierarchical representations from the two images:

\[
\left\{\mathbf{F}^{t}_{l}\right\}_{l=1}^{4}
=\mathcal{E}\!\left(\mathbf{X}^{t}\right),\qquad
t\in\{\mathrm{pre},\mathrm{post}\},
\tag{1}
\]

where \(l=1,\ldots,4\) indexes feature levels from high to low spatial resolution. Sharing \(\mathcal{E}\) constrains the two temporal streams to use the same representation space, so that their differences are less affected by branch-specific parameters.

As illustrated in Fig. X, the decoder proceeds from the coarsest level to the finest level. At level \(l\), the bi-temporal features are concatenated and projected to a common width \(C=128\) using a \(3\times3\) convolution:

\[
\mathbf{z}_{l}
=\phi_l\!\left(
\left[\mathbf{F}^{\mathrm{pre}}_{l},
\mathbf{F}^{\mathrm{post}}_{l}\right]
\right),
\tag{2}
\]

where \([\cdot,\cdot]\) denotes channel-wise concatenation. Each \(\mathbf{z}_{l}\) is processed by two complementary branches: DFRA models directional structures with content-adaptive frequency reliability, whereas COPE-S enhances directional details under explicit bi-temporal change guidance. Their outputs are fused by a \(1\times1\) convolution:

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

\[
\mathbf{p}_{l}
=\mathcal{R}_{l}\!\left(
\widetilde{\mathbf{p}}_{l}
+\mathcal{U}\!\left(\mathbf{p}_{l+1}\right)
\right),\qquad l=3,2,1,
\tag{4}
\]

where \(\mathcal{U}\) is bilinear interpolation to the current spatial size and \(\mathcal{R}_{l}\) is CFR. Finally, a \(1\times1\) classifier maps \(\mathbf{p}_{1}\) to two-class logits, which are bilinearly resized to \(H\times W\).

## B. Directional Frequency-Reliability Aggregation

Remote-sensing changes are often expressed through boundaries, elongated objects, and texture variations with strong directional and frequency characteristics. Applying a single spatial filter treats all directions and spectral components uniformly, whereas unconstrained frequency modulation can amplify acquisition noise and irrelevant high-frequency responses. DFRA addresses this limitation by combining fixed directional priors, a content-adaptive frequency reliability mask, and learned direction weighting.

Given \(\mathbf{z}_{l}\), a \(1\times1\) projection followed by a depth-wise \(3\times3\) convolution produces two features, \(\mathbf{q}_{l}\) and \(\mathbf{v}_{l}\):

\[
[\mathbf{q}_{l},\mathbf{v}_{l}]
=\operatorname{DWConv}_{3\times3}
\left(\operatorname{Conv}_{1\times1}(\mathbf{z}_{l})\right).
\tag{5}
\]

The \(\mathbf{q}_{l}\) branch extracts direction-frequency responses, while \(\mathbf{v}_{l}\) supplies spatial modulation. We divide \(\mathbf{q}_{l}\) into \(O\) channel groups, \(\{\mathbf{q}_{l,o}\}_{o=1}^{O}\), and assign one Gabor kernel to each orientation. For orientation \(\theta_o\), the fixed kernel is

\[
g_o(x,y)
=\exp\!\left[
-\frac{x_o^2+\gamma^2y_o^2}{2\sigma^2}
\right]
\cos\!\left(\frac{2\pi x_o}{\lambda}\right),
\tag{6}
\]

where \(x_o=x\cos\theta_o+y\sin\theta_o\), \(y_o=-x\sin\theta_o+y\cos\theta_o\), and \(\sigma\), \(\lambda\), and \(\gamma\) control the Gaussian scale, wavelength, and aspect ratio. The zero-mean, \(L_1\)-normalized kernel yields the directional subband

\[
\mathbf{s}_{l,o}=g_o * \mathbf{q}_{l,o}.
\tag{7}
\]

### 1) Frequency reliability gate

The designed gate estimates which Fourier coefficients of each directional subband are reliable for the current sample. We first compute its orthonormal real-valued two-dimensional Fourier transform:

\[
\mathbf{S}_{l,o}=\operatorname{rFFT2}(\mathbf{s}_{l,o}).
\tag{8}
\]

To reduce sensitivity to activation scale, the channel-averaged log magnitude is standardized over the frequency grid:

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

\[
\mathbf{M}_{l,o}
=\operatorname{sigmoid}\!\left(
\operatorname{Resize}(\mathbf{P}_o)
+\psi_o(\widehat{\mathbf{A}}_{l,o})
\right).
\tag{10}
\]

Because \(\mathbf{M}_{l,o}\) is real valued, multiplying it with the complex spectrum changes the magnitude while retaining the original phase:

\[
\widehat{\mathbf{s}}_{l,o}
=\operatorname{irFFT2}
\left(\mathbf{S}_{l,o}\odot\mathbf{M}_{l,o}\right).
\tag{11}
\]

This formulation separates two roles: \(\mathbf{P}_o\) captures a resolution-independent frequency preference for each orientation, whereas \(\psi_o\) adapts that preference to the observed content.

### 2) Direction selection and spatial modulation

The directional responses are further weighted according to the input. Global average pooling and a two-layer multilayer perceptron generate normalized orientation coefficients:

\[
\boldsymbol{\omega}_{l}
=O\cdot\operatorname{softmax}
\left(
\operatorname{MLP}(\operatorname{GAP}(\mathbf{q}_{l}))
\right).
\tag{12}
\]

The factor \(O\) keeps the initial mean gain of each direction close to one. The direction-frequency representation is then obtained through residual aggregation:

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

\[
\mathbf{v}'_{l}
=\mathbf{v}_{l}\odot
\operatorname{sigmoid}
\left(\operatorname{DWConv}_{3\times3}(\mathbf{v}_{l})\right).
\tag{14}
\]

After normalization, \(\mathbf{q}'_{l}\) is modulated by \(\mathbf{v}'_{l}\), recalibrated using a pooled channel response, and projected to \(C\) channels. This coupling preserves the direction-frequency evidence selected by DFRA while suppressing spatially irrelevant responses.

## C. Change-Oriented Pyramid Enhancement for Near-Homogeneous Data

DFRA operates on the fused decoder feature and therefore does not explicitly distinguish whether a directional response is supported by the original temporal pair. COPE-S complements it by deriving change cues directly from the two encoder features and using these cues to regulate directional experts. This design is intended for same-domain or near-homogeneous bi-temporal observations, for which aligned temporal differences provide useful spatial guidance.

At level \(l\), separate \(1\times1\) projections and channel-wise layer normalization align the two encoder features:

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

\[
\boldsymbol{\Delta}_{l}
=\left|
\overline{\mathbf{F}}^{\mathrm{pre}}_{l}
-\overline{\mathbf{F}}^{\mathrm{post}}_{l}
\right|,
\tag{16}
\]

drives two complementary controls. The change-guidance generator produces a pixel-wise change mask

\[
\mathbf{m}_{l}
=\operatorname{sigmoid}
\left(
\operatorname{Conv}_{3\times3}(\boldsymbol{\Delta}_{l})
\right),
\tag{17}
\]

whereas the direction-reweighting selector produces sample-wise orientation weights

\[
\mathbf{a}_{l}
=\operatorname{softmax}
\left(
\operatorname{MLP}(\operatorname{GAP}(\boldsymbol{\Delta}_{l}))
\right).
\tag{18}
\]

The decoder feature \(\mathbf{z}_{l}\) is split into \(O\) channel groups. Each oriented expert applies a fixed Gabor filter followed by an independent depth-wise \(3\times3\) refiner:

\[
\mathbf{e}_{l,o}
=g_o * \mathbf{z}_{l,o}
+\eta_o\!\left(g_o * \mathbf{z}_{l,o}\right),
\tag{19}
\]

where \(\eta_o\) consists of depth-wise convolution, batch normalization, and SiLU activation. Each expert response is jointly modulated by the spatial change mask and the corresponding direction coefficient:

\[
\widetilde{\mathbf{e}}_{l,o}
=a_{l,o}\,
\mathbf{m}_{l}\odot\mathbf{e}_{l,o}.
\tag{20}
\]

The modulated responses are concatenated, channel shuffled to promote cross-orientation interaction, and projected back to \(C\) channels. Residual change fusion then gives

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

## D. Cross-Scale Fusion Refinement

Directly adding adjacent pyramid levels can introduce interpolation artifacts and mix coarse semantics with fine spatial responses without further selection. CFR is therefore applied after each upsample-and-add operation in the last three decoder stages. For an aggregated feature \(\mathbf{h}_{l}=\widetilde{\mathbf{p}}_{l}+\mathcal{U}(\mathbf{p}_{l+1})\), CFR first expands its channels using a \(1\times1\) projection and performs depth-wise spatial mixing:

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

\[
\mathbf{h}'_{l}
=\operatorname{GELU}(\mathbf{h}^{(1)}_{l})
\odot\mathbf{h}^{(2)}_{l},
\tag{23}
\]

followed by channel projection and a Gabor-based frequency transform:

\[
\mathbf{p}_{l}
=\mathcal{G}
\left(
\operatorname{Conv}_{1\times1}(\mathbf{h}'_{l})
\right),
\tag{24}
\]

where \(\mathcal{G}\) denotes the frequency-aware transform used by CFR. Unlike DFRA, CFR does not use the proposed rFFT reliability mask; its role is to refine the feature generated by cross-scale addition before it is propagated to the next, finer decoder level.

## E. Optimization Objective and Implementation

The network is optimized using a combination of pixel-wise cross-entropy and Lovász–Softmax losses:

\[
\mathcal{L}
=\mathcal{L}_{\mathrm{CE}}
+2\mathcal{L}_{\mathrm{Lov\acute{a}sz}},
\tag{25}
\]

where pixels labeled 255 are ignored. The current implementation uses AdamW with an initial learning rate of \(1\times10^{-4}\), a weight decay of \(5\times10^{-4}\), and gradient clipping with a maximum norm of 1.0. Input crops are \(256\times256\), and the default batch size is 8.

For both DFRA and COPE-S, \(O=4\) orientations (\(0^\circ\), \(45^\circ\), \(90^\circ\), and \(135^\circ\)) and \(5\times5\) Gabor kernels are used. The default Gabor parameters are \(\sigma=2.0\) and \(\lambda=3.0\). DFRA additionally uses \(\gamma=0.5\), a learnable frequency-prior grid of size \(16\times(16/2+1)\), and an initial residual scale of 0.1. COPE-S uses an aspect ratio of 0.5 and also initializes its residual scale to 0.1. CFR uses an expansion factor of 3. All decoder levels operate at \(C=128\) channels.

---

## Terminology Ledger

| Canonical term | Definition used in this draft | Deprecated/source-code variant |
|---|---|---|
| DFRA | Directional frequency-reliability aggregation module | `StarModule`, DFGF-Star |
| frequency reliability gate | The designed rFFT magnitude mask inside DFRA | DFGF |
| COPE-S | Change-oriented pyramid enhancement for same-/near-domain data | modified GaborFPU |
| CFR | Cross-scale fusion refinement | `DFFN`, `smooth_layer_*` |
| SpatialMamba encoder | Weight-shared hierarchical encoder | `Backbone_SpatialMamba` |
| OEB | Oriented expert bank | `OrientedGaborExpert` |
| CGG | Change-guidance generator | `change_mask` |
| DRS | Direction-reweighting selector | `direction_selector` |
| RCF | Residual change fusion | residual output in `COPES` |

## Assumptions or Missing Inputs

1. The expansions of **DFRA**, **CFR**, **CGG**, **DRS**, **OEB**, and **RCF** must be confirmed by the authors. The prose above assigns expansions that are consistent with the implemented operations, but these names are not declared in the current source code.
2. Replace “Fig. X” with the final framework-figure number.
3. Add citations for SpatialMamba, Gabor filtering, Lovász–Softmax, and any inherited frequency-transform implementation after checking the original papers.
4. Dataset-specific preprocessing, augmentation, train/validation/test splits, hardware, random seeds, and learning-rate scheduling are not fully established by the inspected model code and should be documented in the Experiments section.

## Claim–Evidence Map

| Claim | Evidence | Status |
|---|---|---|
| The encoder is weight shared | The same `self.encoder` is called for pre- and post-images | Supported |
| DFRA preserves Fourier phase | A real mask multiplies the complex rFFT spectrum before `irfft2` | Supported |
| COPE-S uses explicit bi-temporal change guidance | Its mask and direction weights are computed from projected absolute temporal differences | Supported |
| Decoder fusion is coarse-to-fine | Stages II–IV use bilinear upsampling, addition, and CFR | Supported |
| DFRA, COPE-S, and CFR improve accuracy | Requires controlled ablations and benchmark results | Evidence needed |
