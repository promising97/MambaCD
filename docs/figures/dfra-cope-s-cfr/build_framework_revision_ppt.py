#!/usr/bin/env python3
"""PPT: framework figure revision guide + TGRS legend layout."""

from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

OUT_DIR = Path("/root/autodl-tmp/MambaCD/docs/figures/dfra-cope-s-cfr")
HANDDRAWN_PNG = OUT_DIR / "user_framework_handdrawn.png"
COPIES = [
    Path("/root/autodl-tmp/MambaCD/docs"),
    Path("/root/autodl-tmp/论文绘图"),
]

# TGRS / Okabe–Ito semantic palette
C = {
    "ink": RGBColor(0x1A, 0x1A, 0x1A),
    "ieee": RGBColor(0x00, 0x52, 0x93),
    "input_fc": RGBColor(0xD6, 0xEA, 0xF8),
    "input_ec": RGBColor(0x00, 0x72, 0xB2),
    "feat_fc": RGBColor(0xFD, 0xEB, 0xD0),
    "feat_ec": RGBColor(0xE6, 0x9F, 0x00),
    "dfra_fc": RGBColor(0xFA, 0xD7, 0xA0),
    "dfra_ec": RGBColor(0xD5, 0x5E, 0x00),
    "cope_fc": RGBColor(0xB7, 0xE4, 0xD2),
    "cope_ec": RGBColor(0x00, 0x9E, 0x73),
    "cfr_fc": RGBColor(0xFF, 0xF9, 0xC4),
    "cfr_ec": RGBColor(0xF3, 0x9C, 0x12),
    "out_fc": RGBColor(0xE8, 0xF5, 0xE9),
    "out_ec": RGBColor(0x2E, 0x7D, 0x32),
    "warn_fc": RGBColor(0xFD, 0xED, 0xEC),
    "warn_ec": RGBColor(0xC0, 0x39, 0x2B),
    "ok_fc": RGBColor(0xEA, 0xFA, 0xF1),
    "ok_ec": RGBColor(0x27, 0xAE, 0x60),
    "gray": RGBColor(0x7F, 0x8C, 0x8D),
    "white": RGBColor(0xFF, 0xFF, 0xFF),
}


def rgb(hex6: str) -> RGBColor:
    h = hex6.lstrip("#")
    return RGBColor(int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def set_slide_bg(slide, color: RGBColor = C["white"]) -> None:
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_title(slide, title: str, subtitle: str = "") -> None:
    box = slide.shapes.add_textbox(Inches(0.45), Inches(0.28), Inches(12.4), Inches(0.55))
    tf = box.text_frame
    tf.text = title
    p = tf.paragraphs[0]
    p.font.size = Pt(22)
    p.font.bold = True
    p.font.color.rgb = C["ieee"]
    p.font.name = "Arial"
    if subtitle:
        sub = slide.shapes.add_textbox(Inches(0.45), Inches(0.82), Inches(12.4), Inches(0.4))
        stf = sub.text_frame
        stf.text = subtitle
        sp = stf.paragraphs[0]
        sp.font.size = Pt(11)
        sp.font.color.rgb = C["gray"]
        sp.font.name = "Arial"


def round_box(slide, x, y, w, h, text, fc, ec, *, fs=10, bold=False, dash=False):
    sh = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h)
    sh.fill.solid()
    sh.fill.fore_color.rgb = fc
    sh.line.color.rgb = ec
    sh.line.width = Pt(1.25 if bold else 0.9)
    if dash:
        sh.line.dash_style = 2
    tf = sh.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.text = text
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    p.font.size = Pt(fs)
    p.font.bold = bold
    p.font.name = "Arial"
    p.font.color.rgb = C["ink"]
    return sh


def label(slide, x, y, text, *, fs=9, color=None, bold=False, w=2.5):
    box = slide.shapes.add_textbox(x, y, Inches(w), Inches(0.35))
    tf = box.text_frame
    tf.text = text
    p = tf.paragraphs[0]
    p.font.size = Pt(fs)
    p.font.bold = bold
    p.font.name = "Arial"
    p.font.color.rgb = color or C["ink"]
    return box


def arrow_conn(slide, x1, y1, x2, y2, *, dashed=False):
    conn = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, x1, y1, x2, y2)
    conn.line.color.rgb = C["ieee"]
    conn.line.width = Pt(1.1)
    if dashed:
        conn.line.dash_style = 2
    return conn


def badge(slide, x, y, text, fc, ec):
    return round_box(slide, x, y, Inches(0.35), Inches(0.35), text, fc, ec, fs=11, bold=True)


def slide_cover(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)
    label(
        slide,
        Inches(0.8),
        Inches(2.2),
        "MambaCD 整体框架图修订指南",
        fs=28,
        bold=True,
        color=C["ieee"],
        w=11,
    )
    label(
        slide,
        Inches(0.8),
        Inches(3.05),
        "改进位置示意图 · TGRS 图例展示 · 与 ChangeDecoder_Star_COPE_S_DFGF 对齐",
        fs=14,
        color=C["gray"],
        w=11,
    )
    label(slide, Inches(0.8), Inches(6.5), "DFRA  |  COPE-S  |  CFR  |  IEEE TGRS 双栏审美", fs=11, color=C["gray"], w=11)


def slide_checklist(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)
    add_title(slide, "必须修改项（核对代码）", "按优先级排列；改完后 (a)(b)(c) 与实现一一对应")
    items = [
        ("P0", "Stage I (D4, H/32)：仅 fuse 输出，无 bilinear×2 ⊕，无 CFR"),
        ("P0", "每级 OPE 前：Concat(T1_feat, T2_feat) → 3×3 overlap embed → 128 ch"),
        ("P0", "COPE-S 侧向输入：同尺度 encoder 特征 F_pre^k, F_post^k（非仅 z_k）"),
        ("P1", "(b) 模块名 DFGF → DFRA；q 路含 rFFT gate；v 路 spatial；channel gate + project"),
        ("P1", "(c) 残差 z_k + α·enhanced；DRS 为 softmax；|Δ| 在 1×1+LN 对齐特征上"),
        ("P2", "D1 后补 1×1 cls (128→2) + 双线性至原图；encoder 共享权重（❄仅 freeze 实验时）"),
        ("P2", "Stage 编号：I↔D4(H/32) … IV↔D1(H/4)；CFR 脚注 = DFFN (smooth_layer)"),
    ]
    y = Inches(1.35)
    for pri, txt in items:
        fc, ec = (C["warn_fc"], C["warn_ec"]) if pri == "P0" else (C["feat_fc"], C["feat_ec"])
        if pri == "P2":
            fc, ec = C["ok_fc"], C["ok_ec"]
        round_box(slide, Inches(0.55), y, Inches(0.55), Inches(0.38), pri, fc, ec, fs=10, bold=True)
        label(slide, Inches(1.2), y + Inches(0.04), txt, fs=11, w=11)
        y += Inches(0.72)


def slide_decoder_rail(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)
    add_title(slide, "示意图 1：四阶 Decoder 数据流（(a) 主图）", "左：易错画法  →  右：与代码一致")

    # Wrong column
    label(slide, Inches(0.5), Inches(1.15), "❌ 易错", fs=12, bold=True, color=C["warn_ec"])
    wx = Inches(0.5)
    round_box(slide, wx, Inches(1.55), Inches(1.1), Inches(0.45), "D4", C["feat_fc"], C["feat_ec"], bold=True)
    round_box(slide, wx + Inches(1.25), Inches(1.55), Inches(1.3), Inches(0.45), "DFRA∥COPE", C["dfra_fc"], C["dfra_ec"], fs=9)
    round_box(slide, wx + Inches(2.7), Inches(1.55), Inches(0.9), Inches(0.45), "CFR?", C["cfr_fc"], C["cfr_ec"], fs=9)
    label(slide, wx, Inches(2.15), "D4 不应出现 CFR / skip-add", fs=9, color=C["warn_ec"], w=5.5)

    wy = Inches(2.55)
    for name in ("D3", "D2", "D1"):
        round_box(slide, wx, wy, Inches(0.7), Inches(0.4), name, C["feat_fc"], C["feat_ec"], fs=9)
        round_box(slide, wx + Inches(0.85), wy, Inches(1.1), Inches(0.4), "fuse", C["feat_fc"], C["feat_ec"], fs=8)
        round_box(slide, wx + Inches(2.05), wy, Inches(0.75), Inches(0.4), "⊕", C["warn_fc"], C["warn_ec"], fs=10, bold=True)
        round_box(slide, wx + Inches(2.9), wy, Inches(0.75), Inches(0.4), "CFR", C["cfr_fc"], C["cfr_ec"], fs=8)
        wy += Inches(0.55)

    # Correct column
    label(slide, Inches(6.85), Inches(1.15), "✓ 推荐", fs=12, bold=True, color=C["ok_ec"])
    cx = Inches(6.85)
    round_box(slide, cx, Inches(1.55), Inches(0.55), Inches(0.35), "C", C["feat_fc"], C["feat_ec"], fs=9)
    round_box(slide, cx + Inches(0.65), Inches(1.55), Inches(1.35), Inches(0.35), "3×3 OPE→128", C["feat_fc"], C["feat_ec"], fs=8)
    round_box(slide, cx + Inches(2.1), Inches(1.45), Inches(0.85), Inches(0.55), "DFRA", C["dfra_fc"], C["dfra_ec"], fs=9, bold=True)
    round_box(slide, cx + Inches(2.1), Inches(2.05), Inches(0.85), Inches(0.55), "COPE-S", C["cope_fc"], C["cope_ec"], fs=9, bold=True)
    round_box(slide, cx + Inches(3.05), Inches(1.75), Inches(1.0), Inches(0.45), "1×1 fuse", C["feat_fc"], C["feat_ec"], fs=8)
    round_box(slide, cx + Inches(4.15), Inches(1.75), Inches(1.05), Inches(0.45), "p₄ out", C["out_fc"], C["out_ec"], fs=9, bold=True)
    label(slide, cx, Inches(2.65), "Stage I：无 ↑×2、无 CFR", fs=9, color=C["ok_ec"], w=5)

    cy = Inches(3.05)
    stages = [("D3", "II"), ("D2", "III"), ("D1", "IV")]
    for dname, st in stages:
        round_box(slide, cx, cy, Inches(0.45), Inches(0.32), "C", C["feat_fc"], C["feat_ec"], fs=8)
        round_box(slide, cx + Inches(0.52), cy, Inches(1.0), Inches(0.32), "OPE+∥", C["feat_fc"], C["feat_ec"], fs=7)
        round_box(slide, cx + Inches(1.58), cy, Inches(0.85), Inches(0.32), "1×1", C["feat_fc"], C["feat_ec"], fs=8)
        round_box(slide, cx + Inches(2.48), cy, Inches(0.95), Inches(0.32), "↑×2⊕", C["ok_fc"], C["ok_ec"], fs=8, bold=True)
        round_box(slide, cx + Inches(3.48), cy, Inches(0.75), Inches(0.32), "CFR", C["cfr_fc"], C["cfr_ec"], fs=8, bold=True)
        label(slide, cx + Inches(4.3), cy + Inches(0.02), f"{dname} Stage {st}", fs=8, color=C["gray"], w=1.2)
        cy += Inches(0.48)

    round_box(slide, cx + Inches(3.48), cy + Inches(0.05), Inches(1.55), Inches(0.38), "1×1 cls → map", C["out_fc"], C["out_ec"], fs=9)
    label(
        slide,
        Inches(0.5),
        Inches(5.35),
        "⊕ 表示与上一级输出 skip-add；bilinear ×2 在上采样支路标注（与 ↑×2⊕ 框并列即可）。",
        fs=10,
        color=C["gray"],
        w=12,
    )


def slide_cope_side(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)
    add_title(slide, "示意图 2：COPE-S 侧向输入（(a) + (c)）", "变化线索来自 encoder 同尺度特征，不是原图像素 |Δ|")

    enc = round_box(slide, Inches(0.6), Inches(1.6), Inches(1.4), Inches(2.2), "Encoder\nStage k", C["input_fc"], C["input_ec"], fs=10)
    label(slide, Inches(0.65), Inches(3.9), "F_pre^k", fs=9, color=C["input_ec"])
    label(slide, Inches(0.65), Inches(4.15), "F_post^k", fs=9, color=rgb("#CC79A0"))

    round_box(slide, Inches(2.35), Inches(1.85), Inches(0.45), Inches(0.35), "C", C["feat_fc"], C["feat_ec"])
    z = round_box(slide, Inches(3.0), Inches(1.75), Inches(1.15), Inches(0.55), "z_k\n(OPE 128)", C["feat_fc"], C["feat_ec"], fs=9)

    dfra = round_box(slide, Inches(4.45), Inches(1.55), Inches(1.05), Inches(0.55), "DFRA", C["dfra_fc"], C["dfra_ec"], bold=True)
    cope = round_box(slide, Inches(4.45), Inches(2.25), Inches(1.05), Inches(0.55), "COPE-S", C["cope_fc"], C["cope_ec"], bold=True)
    fuse = round_box(slide, Inches(5.75), Inches(1.85), Inches(1.0), Inches(0.55), "1×1\n256→128", C["feat_fc"], C["feat_ec"], fs=9)

    arrow_conn(slide, Inches(2.0), Inches(2.1), Inches(2.35), Inches(2.0))
    arrow_conn(slide, Inches(2.8), Inches(2.0), Inches(3.0), Inches(2.0))
    arrow_conn(slide, Inches(4.15), Inches(2.0), Inches(4.45), Inches(1.85))
    arrow_conn(slide, Inches(4.15), Inches(2.0), Inches(4.45), Inches(2.5))
    arrow_conn(slide, Inches(5.5), Inches(1.85), Inches(5.75), Inches(2.1))
    arrow_conn(slide, Inches(5.5), Inches(2.5), Inches(5.75), Inches(2.1))

    # Dashed side branch to COPE
    arrow_conn(slide, Inches(2.0), Inches(2.35), Inches(4.45), Inches(2.55), dashed=True)
    arrow_conn(slide, Inches(2.0), Inches(2.65), Inches(4.45), Inches(2.55), dashed=True)

    inset = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.1), Inches(1.5), Inches(5.5), Inches(2.35)
    )
    inset.fill.solid()
    inset.fill.fore_color.rgb = C["white"]
    inset.line.color.rgb = C["cope_ec"]
    inset.line.width = Pt(1)
    inset.line.dash_style = 2
    label(slide, Inches(7.25), Inches(1.58), "(c) COPE-S 内部（推荐）", fs=11, bold=True, color=C["cope_ec"], w=5)
    round_box(slide, Inches(7.3), Inches(1.95), Inches(1.0), Inches(0.4), "1×1+LN", C["feat_fc"], C["feat_ec"], fs=8)
    round_box(slide, Inches(8.4), Inches(1.95), Inches(0.55), Inches(0.4), "|Δ|", C["feat_fc"], C["feat_ec"], fs=9)
    round_box(slide, Inches(9.05), Inches(1.85), Inches(1.05), Inches(0.35), "CGG", C["cope_fc"], C["cope_ec"], fs=8)
    round_box(slide, Inches(9.05), Inches(2.25), Inches(1.05), Inches(0.35), "DRS↓", C["cope_fc"], C["cope_ec"], fs=8)
    round_box(slide, Inches(7.3), Inches(2.75), Inches(2.2), Inches(0.45), "4× OEB (0/45/90/135°)", C["cope_fc"], C["cope_ec"], fs=8)
    round_box(slide, Inches(9.65), Inches(2.75), Inches(1.1), Inches(0.45), "⊙ CGG/DRS", C["feat_fc"], C["feat_ec"], fs=8)
    round_box(slide, Inches(10.85), Inches(2.75), Inches(1.35), Inches(0.45), "Shuffle+RCF", C["cope_fc"], C["cope_ec"], fs=8)
    label(slide, Inches(7.25), Inches(3.35), "output = z_k + α·enhanced", fs=10, color=C["ieee"], w=5)

    label(
        slide,
        Inches(0.6),
        Inches(4.55),
        "图注建议：虚线 = change cues / 条件支路；实线 = 主特征流。",
        fs=10,
        color=C["gray"],
        w=12,
    )


def slide_dfra_fix(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)
    add_title(slide, "示意图 3：(b) DFRA 子图修订", "删除 DFGF / 多余 S；与 StarModule forward 一致")

    label(slide, Inches(0.55), Inches(1.2), "删除或替换", fs=11, bold=True, color=C["warn_ec"])
    for t, x in [("DFGF 块名", 0.55), ("split 后 S 圆", 2.1), ("× gate CMU 含糊", 3.65)]:
        round_box(slide, Inches(x), Inches(1.55), Inches(1.35), Inches(0.42), t, C["warn_fc"], C["warn_ec"], fs=9)

    label(slide, Inches(0.55), Inches(2.35), "推荐数据流", fs=11, bold=True, color=C["ok_ec"])
    x0 = Inches(0.55)
    y0 = Inches(2.75)
    boxes = [
        (x0, "z_k", "input"),
        (x0 + Inches(1.05), "1×1+DW\nsplit q,v", "feat"),
        (x0 + Inches(2.35), "4-dir Gabor\n+rFFT gate", "dfra"),
        (x0 + Inches(3.75), "Spatial\ngate (v)", "input"),
        (x0 + Inches(4.95), "q ⊙ v", "feat"),
        (x0 + Inches(5.85), "channel\ngate", "feat"),
        (x0 + Inches(6.75), "project", "feat"),
        (x0 + Inches(7.65), "DFRA", "out"),
    ]
    pal = {
        "input": (C["input_fc"], C["input_ec"]),
        "feat": (C["feat_fc"], C["feat_ec"]),
        "dfra": (C["dfra_fc"], C["dfra_ec"]),
        "out": (C["out_fc"], C["out_ec"]),
    }
    prev_x = None
    for i, (x, txt, kind) in enumerate(boxes):
        fc, ec = pal[kind]
        if i == 3:
            round_box(slide, x, y0 + Inches(0.55), Inches(0.95), Inches(0.55), txt, fc, ec, fs=8)
        else:
            round_box(slide, x, y0, Inches(0.95), Inches(0.55), txt, fc, ec, fs=8, bold=(kind == "dfra"))
        if prev_x is not None and i not in (3, 4):
            arrow_conn(slide, prev_x + Inches(0.95), y0 + Inches(0.28), x, y0 + Inches(0.28))
        if i == 2:
            arrow_conn(slide, x + Inches(0.48), y0 + Inches(0.55), x + Inches(0.48), y0 + Inches(0.55))
        prev_x = x if i != 3 else prev_x
    arrow_conn(slide, x0 + Inches(2.35) + Inches(0.48), y0 + Inches(0.55), x0 + Inches(4.95), y0 + Inches(0.55))
    arrow_conn(slide, x0 + Inches(3.75) + Inches(0.48), y0 + Inches(0.82), x0 + Inches(4.95), y0 + Inches(0.55))

    dash = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.55), Inches(4.05), Inches(7.8), Inches(0.95)
    )
    dash.fill.solid()
    dash.fill.fore_color.rgb = rgb("#FBFCFC")
    dash.line.color.rgb = C["dfra_ec"]
    dash.line.dash_style = 2
    label(slide, Inches(0.7), Inches(4.12), "Designed rFFT gate (phase preserved):  rFFT → log-mag → P → σ(P+ψ) → irFFT", fs=9, color=C["dfra_ec"], w=11)


def slide_legend(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)
    add_title(slide, "图例展示方式（建议放在 (a) 全图底部）", "双栏宽 7.16 in：一行符号 + 一行模块色 + 一行缩写（8 pt Arial）")

    # Row 1: symbols
    label(slide, Inches(0.5), Inches(1.2), "A. 运算符号（左对齐，间距均匀）", fs=11, bold=True, color=C["ieee"])
    syms = [
        ("C", "通道拼接\nConcat"),
        ("∥", "并行支路"),
        ("⊙", "逐元素乘\n(门控)"),
        ("⊕", "逐元素加\n(skip)"),
        ("↑×2", "bilinear\n上采样"),
        ("---", "虚线：条件/变化线索"),
    ]
    sx = Inches(0.5)
    for sym, desc in syms:
        badge(slide, sx, Inches(1.55), sym, C["feat_fc"], C["feat_ec"])
        label(slide, sx - Inches(0.05), Inches(1.98), desc, fs=8, w=1.05)
        sx += Inches(1.95)

    # Row 2: module colors
    label(slide, Inches(0.5), Inches(2.65), "B. 模块语义色（与 DFRA/COPE-S 小图一致）", fs=11, bold=True, color=C["ieee"])
    mods = [
        ("Input / skip", "input"),
        ("Transform", "feat"),
        ("DFRA / gate", "dfra"),
        ("COPE-S", "cope"),
        ("CFR (DFFN)", "cfr"),
        ("Output", "out"),
    ]
    pal_map = {
        "input": (C["input_fc"], C["input_ec"]),
        "feat": (C["feat_fc"], C["feat_ec"]),
        "dfra": (C["dfra_fc"], C["dfra_ec"]),
        "cope": (C["cope_fc"], C["cope_ec"]),
        "cfr": (C["cfr_fc"], C["cfr_ec"]),
        "out": (C["out_fc"], C["out_ec"]),
    }
    mx = Inches(0.5)
    for lab, key in mods:
        fc, ec = pal_map[key]
        round_box(slide, mx, Inches(3.0), Inches(1.85), Inches(0.42), lab, fc, ec, fs=9, bold=True)
        mx += Inches(2.0)

    # Row 3: abbreviations panel
    label(slide, Inches(0.5), Inches(3.65), "C. 缩略语（可置于图注或 legend 下方 1 行）", fs=11, bold=True, color=C["ieee"])
    panel = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.5), Inches(4.0), Inches(12.3), Inches(1.55))
    panel.fill.solid()
    panel.fill.fore_color.rgb = rgb("#F8F9FA")
    panel.line.color.rgb = C["gray"]
    panel.line.width = Pt(0.75)
    lines = [
        "DFRA — Directional Frequency-Reliability Attention (StarModule): 4-dir Gabor + rFFT gate on q; spatial gate on v.",
        "COPE-S — Change-Oriented Pyramid Enhancement (same/near-domain): CGG spatial mask, DRS softmax weights, 4 OEB, RCF residual.",
        "CFR — Change Feature Refinement (implementation: DFFN after skip-add, Stages II–IV only).  OPE — 3×3 overlap embed, 2C→128.",
    ]
    ty = Inches(4.12)
    for line in lines:
        label(slide, Inches(0.65), ty, line, fs=9, w=12)
        ty += Inches(0.38)

    label(
        slide,
        Inches(0.5),
        Inches(5.75),
        "排版：legend 总高度 ≤ 全图 12%；灰度打印时依赖线型（实/虚）+ 形状边框，不仅靠色相。",
        fs=9,
        color=C["gray"],
        w=12,
    )


def _fit_picture_size(iw: int, ih: int, max_w_in: float, max_h_in: float) -> tuple[float, float]:
    aspect = iw / ih
    box_aspect = max_w_in / max_h_in
    if aspect >= box_aspect:
        w = max_w_in
        h = max_w_in / aspect
    else:
        h = max_h_in
        w = max_h_in * aspect
    return w, h


def callout_badge(slide, x, y, num: str, *, kind: str = "warn"):
    fc, ec = (C["warn_fc"], C["warn_ec"]) if kind == "warn" else (C["ok_fc"], C["ok_ec"])
    sh = slide.shapes.add_shape(MSO_SHAPE.OVAL, x, y, Inches(0.28), Inches(0.28))
    sh.fill.solid()
    sh.fill.fore_color.rgb = fc
    sh.line.color.rgb = ec
    sh.line.width = Pt(1.2)
    tf = sh.text_frame
    tf.text = num
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    p.font.size = Pt(10)
    p.font.bold = True
    p.font.name = "Arial"
    p.font.color.rgb = ec
    return sh


def callout_note(slide, x, y, w, text: str, *, kind: str = "warn", fs: int = 8):
    fc, ec = (C["warn_fc"], C["warn_ec"]) if kind == "warn" else (C["ok_fc"], C["ok_ec"])
    return round_box(slide, x, y, w, Inches(0.55), text, fc, ec, fs=fs, bold=False)


def highlight_rect(slide, x, y, w, h, *, kind: str = "warn"):
    ec = C["warn_ec"] if kind == "warn" else C["ok_ec"]
    sh = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, h)
    sh.fill.background()
    sh.line.color.rgb = ec
    sh.line.width = Pt(2.0)
    sh.line.dash_style = 2 if kind == "warn" else 1
    return sh


def slide_handdrawn_overlay(prs, image_path: Path):
    """User hand-drawn framework as picture + editable annotation layer."""
    from PIL import Image

    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)
    add_title(
        slide,
        "对照改：手绘整体框架 + 标注层（可拖动）",
        "底层为你的 PNG；红圈/红框 = 必改；绿框 = 建议新增",
    )

    if not image_path.is_file():
        label(slide, Inches(0.6), Inches(2.5), f"缺少底图：{image_path}", fs=14, color=C["warn_ec"], w=10)
        return

    iw, ih = Image.open(image_path).size
    pic_left = Inches(0.35)
    pic_top = Inches(1.02)
    pic_w_in, pic_h_in = _fit_picture_size(iw, ih, 12.55, 5.78)
    pic_w, pic_h = Inches(pic_w_in), Inches(pic_h_in)
    slide.shapes.add_picture(str(image_path), pic_left, pic_top, width=pic_w)

    # Normalized anchors (x, y, w, h) relative to picture box — tune to your layout
    def rx(nx: float):
        return pic_left + pic_w * nx

    def ry(ny: float):
        return pic_top + pic_h * ny

    def rw(nw: float):
        return pic_w * nw

    def rh(nh: float):
        return pic_h * nh

    # (a) D4 / deepest stage — CFR should not appear here
    highlight_rect(slide, rx(0.06), ry(0.52), rw(0.22), rh(0.12), kind="warn")
    callout_badge(slide, rx(0.27), ry(0.50), "1", kind="warn")
    callout_note(
        slide,
        rx(0.01),
        ry(0.64),
        Inches(2.05),
        "D4(Stage I)：删 CFR\n仅 fuse → p₄",
        kind="warn",
    )

    # Concat before OPE at each stage row
    highlight_rect(slide, rx(0.30), ry(0.38), rw(0.08), rh(0.06), kind="warn")
    callout_badge(slide, rx(0.38), ry(0.36), "2", kind="warn")
    callout_note(
        slide,
        rx(0.30),
        ry(0.28),
        Inches(2.15),
        "OPE 前加 C：\nConcat(T1,T2) feat",
        kind="warn",
    )

    # COPE-S needs encoder side branch
    highlight_rect(slide, rx(0.42), ry(0.44), rw(0.12), rh(0.08), kind="ok")
    callout_badge(slide, rx(0.54), ry(0.42), "3", kind="ok")
    arrow_conn(slide, rx(0.12), ry(0.35), rx(0.42), ry(0.48), dashed=True)
    callout_note(
        slide,
        rx(0.56),
        ry(0.44),
        Inches(2.0),
        "虚线：F_pre^k,\nF_post^k → COPE-S",
        kind="ok",
    )

    # Output head missing
    highlight_rect(slide, rx(0.88), ry(0.40), rw(0.10), rh(0.14), kind="ok")
    callout_badge(slide, rx(0.86), ry(0.38), "4", kind="ok")
    callout_note(
        slide,
        rx(0.72),
        ry(0.22),
        Inches(2.05),
        "补 1×1 cls\n128→2 + ↑ 原图",
        kind="ok",
    )

    # (b) DFGF rename
    highlight_rect(slide, rx(0.05), ry(0.78), rw(0.28), rh(0.18), kind="warn")
    callout_badge(slide, rx(0.32), ry(0.76), "5", kind="warn")
    callout_note(
        slide,
        rx(0.02),
        ry(0.88),
        Inches(2.35),
        "(b) DFGF→DFRA\n删 S；加 rFFT gate",
        kind="warn",
        fs=7,
    )

    # (c) COPE formula / DRS
    highlight_rect(slide, rx(0.52), ry(0.78), rw(0.42), rh(0.18), kind="warn")
    callout_badge(slide, rx(0.92), ry(0.76), "6", kind="warn")
    callout_note(
        slide,
        rx(0.52),
        ry(0.88),
        Inches(2.5),
        "(c) z_k+α·enhanced\nDRS=softmax；|Δ|在 LN 后",
        kind="warn",
        fs=7,
    )

    # Encoder snowflake
    callout_badge(slide, rx(0.08), ry(0.08), "7", kind="warn")
    callout_note(
        slide,
        rx(0.10),
        ry(0.05),
        Inches(2.3),
        "❄ 改为 weight-tied\nsiamese（除非 freeze）",
        kind="warn",
        fs=7,
    )

    label(
        slide,
        Inches(0.35),
        Inches(6.88),
        "提示：在 PPT 中选中底图可替换为你最新导出的 PNG；标注框/编号可自由拖动对齐。",
        fs=8,
        color=C["gray"],
        w=12.5,
    )


def slide_layout_mock(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)
    add_title(slide, "全图布局 mock（legend 位置）", "(a) 占宽 100%；(b)(c) 可并排或另页；legend 贴 (a) 底边")

    frame = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.45), Inches(1.15), Inches(12.4), Inches(3.35))
    frame.fill.solid()
    frame.fill.fore_color.rgb = C["white"]
    frame.line.color.rgb = C["ieee"]
    frame.line.width = Pt(1.5)
    label(slide, Inches(0.55), Inches(1.22), "(a) Four-stage decoder + shared encoder  ······  change map", fs=10, bold=True, color=C["ieee"], w=11)

    leg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.45), Inches(4.55), Inches(12.4), Inches(0.85))
    leg.fill.solid()
    leg.fill.fore_color.rgb = rgb("#F4F6F7")
    leg.line.color.rgb = C["gray"]
    label(slide, Inches(0.55), Inches(4.62), "[ Symbols: C  ∥  ⊙  ⊕  ↑×2  ]    [ Color swatches ×6 ]    [ DFRA | COPE-S | CFR ]", fs=9, w=12)

    sub = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.45), Inches(5.55), Inches(6.0), Inches(1.35))
    sub.fill.solid()
    sub.fill.fore_color.rgb = C["dfra_fc"]
    sub.fill.transparency = 0.3
    sub.line.color.rgb = C["dfra_ec"]
    label(slide, Inches(0.55), Inches(5.62), "(b) DFRA", fs=10, bold=True, color=C["dfra_ec"], w=2)

    sub2 = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(6.85), Inches(5.55), Inches(6.0), Inches(1.35))
    sub2.fill.solid()
    sub2.fill.fore_color.rgb = C["cope_fc"]
    sub2.fill.transparency = 0.3
    sub2.line.color.rgb = C["cope_ec"]
    label(slide, Inches(6.95), Inches(5.62), "(c) COPE-S", fs=10, bold=True, color=C["cope_ec"], w=2)


def build_ppt(path: Path) -> None:
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    slide_cover(prs)
    slide_checklist(prs)
    slide_handdrawn_overlay(prs, HANDDRAWN_PNG)
    slide_decoder_rail(prs)
    slide_cope_side(prs)
    slide_dfra_fix(prs)
    slide_legend(prs)
    slide_layout_mock(prs)

    path.parent.mkdir(parents=True, exist_ok=True)
    prs.save(path)


def main() -> None:
    out = OUT_DIR / "fig_framework_revision_guide.pptx"
    build_ppt(out)
    for dest in COPIES:
        dest.mkdir(parents=True, exist_ok=True)
        (dest / out.name).write_bytes(out.read_bytes())
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
