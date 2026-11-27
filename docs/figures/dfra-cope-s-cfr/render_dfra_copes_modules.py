#!/usr/bin/env python3
"""DFRA and COPE-S module schematics.

scipilot-figure-skill covers data plots, not architecture diagrams. This script
keeps its IEEE/TGRS contract: 7.16 in width, 8 pt floor, Arial in-figure,
Okabe–Ito / TGRS semantic colors, (a)(b) labels, vector export, grayscale
preview, and the visual-QA loop.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, "/tmp/scipilot-figure-skill/scripts")
from export_figure import export_figure  # noqa: E402
from setup_style import setup_style  # noqa: E402
from visual_qa import audit_layout, print_report, render_preview  # noqa: E402

OUT = Path("/root/autodl-tmp/MambaCD/docs/figures/dfra-cope-s-cfr")
COPIES = [
    Path("/root/autodl-tmp/MambaCD/docs"),
    Path("/root/autodl-tmp/论文绘图"),
]

# Okabe–Ito + TGRS architecture mapping (colorblind-safe, grayscale-distinct)
PAL = {
    "input": ("#D6EAF8", "#0072B2"),  # sky / IEEE blue
    "std": ("#D4E6F1", "#0072B2"),
    "feat": ("#FDEBD0", "#E69F00"),  # gold transform
    "dfra": ("#FAD7A0", "#D55E00"),  # vermillion — DFRA / frequency gate
    "cope": ("#B7E4D2", "#009E73"),  # bluish green — COPE-S
    "out": ("#E8F5E9", "#2E7D32"),
}

IEEE_BLUE = "#005293"
INK = "#1A1A1A"
ARROW = "#1B4F72"
DASH = "#5D6D7E"


def box(ax, x, y, w, h, text, kind, *, fs=8.0, weight="normal"):
    fc, ec = PAL[kind]
    lw = 1.15 if kind in ("dfra", "cope") else 0.8
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.008,rounding_size=0.06",
            linewidth=lw,
            facecolor=fc,
            edgecolor=ec,
            zorder=2,
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        fontweight=weight,
        color=INK,
        zorder=3,
        linespacing=1.12,
        clip_on=False,
    )
    return (x, y, w, h)


def C(r, side):
    x, y, w, h = r
    return {
        "l": (x, y + h / 2),
        "r": (x + w, y + h / 2),
        "b": (x + w / 2, y),
        "t": (x + w / 2, y + h),
    }[side]


def arrow(ax, p0, p1, *, dashed=False):
    ax.add_patch(
        FancyArrowPatch(
            p0,
            p1,
            arrowstyle="-|>",
            mutation_scale=7.5,
            lw=0.9,
            color=DASH if dashed else ARROW,
            linestyle=(0, (2.4, 1.6)) if dashed else "solid",
            shrinkA=2.5,
            shrinkB=5.5,
            clip_on=False,
            zorder=1.5,
        )
    )


def poly_arrow(ax, pts, *, dashed=False):
    color = DASH if dashed else ARROW
    ls = (0, (2.4, 1.6)) if dashed else "solid"
    xs, ys = zip(*pts)
    ax.plot(
        xs,
        ys,
        color=color,
        lw=0.9,
        linestyle=ls,
        solid_capstyle="round",
        zorder=1.4,
        clip_on=False,
    )
    arrow(ax, pts[-2], pts[-1], dashed=dashed)


def dashed_group(ax, x, y, w, h, title):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.01,rounding_size=0.08",
            linewidth=0.7,
            linestyle=(0, (2.6, 1.6)),
            facecolor="#FBFCFC",
            edgecolor="#7F8C8D",
            zorder=0.8,
        )
    )
    ax.text(
        x + 0.10,
        y + h - 0.20,
        title,
        ha="left",
        va="center",
        fontsize=8.0,
        fontweight="bold",
        color=IEEE_BLUE,
        zorder=3,
        clip_on=False,
    )


def note(ax, x, y, text, *, ha="left", fs=7.2):
    ax.text(
        x,
        y,
        text,
        ha=ha,
        va="center",
        fontsize=fs,
        fontstyle="italic",
        color="#4B5563",
        zorder=3,
        clip_on=False,
        linespacing=1.25,
    )


def tag(ax, x, y, text):
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=7.0,
        fontweight="bold",
        color=IEEE_BLUE,
        zorder=3,
        clip_on=False,
    )


def draw_dfra(ax):
    ax.set_xlim(0.00, 7.60)
    ax.set_ylim(0.02, 7.55)
    ax.set_aspect("auto")
    ax.axis("off")
    ax.set_title(
        "(a)  DFRA",
        fontsize=9,
        fontweight="bold",
        color=IEEE_BLUE,
        pad=5,
        loc="left",
    )

    zk = box(ax, 0.18, 4.48, 0.88, 0.82, r"$z_k$", "input", fs=9, weight="bold")
    sp = box(ax, 1.20, 4.48, 1.10, 0.82, "split\nq, v", "feat", fs=7.6)

    gb = box(ax, 2.52, 5.88, 1.32, 0.82, "4-dir\nGabor", "feat", fs=8.0)
    gt = box(
        ax,
        4.02,
        5.82,
        1.72,
        0.94,
        "Frequency\ngate",
        "dfra",
        fs=8.0,
        weight="bold",
    )
    dw = box(ax, 5.92, 5.88, 1.48, 0.82, "Dir.\nreweight", "feat", fs=7.6)

    vp = box(ax, 2.55, 3.18, 1.70, 0.82, "Spatial\ngate (v)", "std", fs=7.6)
    mul = box(ax, 4.50, 3.18, 1.18, 0.82, r"q $\odot$ v", "feat", fs=8.0, weight="bold")
    out = box(ax, 5.90, 3.18, 1.50, 0.82, "DFRA", "out", fs=8.4, weight="bold")

    arrow(ax, C(zk, "r"), C(sp, "l"))
    # q-path rises from split, v-path drops — no diagonal through boxes
    poly_arrow(ax, [C(sp, "t"), (C(sp, "t")[0], C(gb, "l")[1]), C(gb, "l")])
    arrow(ax, C(gb, "r"), C(gt, "l"))
    arrow(ax, C(gt, "r"), C(dw, "l"))
    poly_arrow(ax, [C(sp, "b"), (C(sp, "b")[0], C(vp, "l")[1]), C(vp, "l")])
    arrow(ax, C(vp, "r"), C(mul, "l"))
    poly_arrow(
        ax,
        [
            C(dw, "b"),
            (C(dw, "b")[0], 4.42),
            (C(mul, "t")[0], 4.42),
            C(mul, "t"),
        ],
    )
    arrow(ax, C(mul, "r"), C(out, "l"))
    tag(ax, 2.28, 5.55, "q")
    tag(ax, 2.28, 3.95, "v")

    note(
        ax,
        0.18,
        2.58,
        "q-path: 4-dir Gabor + rFFT reliability gate.\nv-path: spatial gating only.",
    )

    dashed_group(ax, 0.14, 0.38, 7.30, 1.95, "Designed rFFT gate  (phase preserved)")
    r1 = box(ax, 0.32, 0.58, 1.18, 0.72, "rFFT", "feat", fs=8.0)
    r2 = box(ax, 1.68, 0.58, 1.28, 0.72, "log-mag", "feat", fs=7.6)
    r3 = box(ax, 3.14, 0.58, 1.18, 0.72, "prior P", "dfra", fs=8.0, weight="bold")
    r4 = box(ax, 4.50, 0.58, 1.55, 0.72, r"$M=\sigma(P+\psi)$", "dfra", fs=7.8, weight="bold")
    r5 = box(ax, 6.23, 0.58, 1.05, 0.72, "irFFT", "feat", fs=8.0)
    for a, b in ((r1, r2), (r2, r3), (r3, r4), (r4, r5)):
        arrow(ax, C(a, "r"), C(b, "l"))

    note(ax, 0.18, 0.16, r"Inner residual  $q\leftarrow q+\alpha\cdot\tilde{q}$.  LN / SE omitted.")


def draw_cope(ax):
    ax.set_xlim(0.00, 7.60)
    ax.set_ylim(0.02, 7.55)
    ax.set_aspect("auto")
    ax.axis("off")
    ax.set_title(
        "(b)  COPE-S",
        fontsize=9,
        fontweight="bold",
        color=IEEE_BLUE,
        pad=5,
        loc="left",
    )

    pre = box(ax, 0.18, 6.38, 1.05, 0.70, r"$F_{\mathrm{pre}}$", "input", fs=8.0, weight="bold")
    post = box(ax, 0.18, 5.48, 1.05, 0.70, r"$F_{\mathrm{post}}$", "input", fs=8.0, weight="bold")
    aln = box(ax, 1.42, 5.88, 1.18, 0.80, r"1$\times$1 align", "feat", fs=7.5)
    abd = box(ax, 2.78, 5.88, 0.78, 0.80, r"$|\Delta|$", "feat", fs=9.0, weight="bold")
    cgg = box(ax, 3.74, 6.38, 1.92, 0.70, "CGG  spatial", "cope", fs=8.0, weight="bold")
    drs = box(ax, 3.74, 5.48, 1.92, 0.70, "DRS  softmax", "cope", fs=8.0, weight="bold")

    zk = box(ax, 0.18, 3.12, 0.92, 0.82, r"$z_k$", "input", fs=9.0, weight="bold")
    oeb = box(ax, 1.28, 3.04, 1.50, 0.98, "4-dir OEB", "cope", fs=8.0, weight="bold")
    mod = box(ax, 2.96, 3.12, 1.70, 0.82, r"$\odot$ CGG/DRS", "feat", fs=7.6, weight="bold")
    shu = box(ax, 4.84, 3.12, 1.12, 0.82, "Shuffle", "feat", fs=8.0)
    rcf = box(ax, 6.14, 3.12, 1.26, 0.82, "RCF", "cope", fs=8.2, weight="bold")
    out = box(ax, 6.14, 2.02, 1.26, 0.74, "COPE-S", "out", fs=8.2, weight="bold")

    arrow(ax, C(pre, "r"), (aln[0], aln[1] + aln[3] * 0.72))
    arrow(ax, C(post, "r"), (aln[0], aln[1] + aln[3] * 0.28))
    arrow(ax, C(aln, "r"), C(abd, "l"))
    poly_arrow(ax, [C(abd, "r"), (cgg[0], C(cgg, "l")[1])])
    poly_arrow(ax, [C(abd, "r"), (drs[0], C(drs, "l")[1])])

    # Right-side cue bus: CGG/DRS → ⊙, stays in a corridor (no text)
    bus_x = 5.88
    join = (C(mod, "t")[0], 4.22)
    ax.plot(
        [C(cgg, "r")[0], bus_x, bus_x, join[0], C(mod, "t")[0]],
        [C(cgg, "r")[1], C(cgg, "r")[1], 4.22, 4.22, C(mod, "t")[1]],
        color=DASH,
        lw=0.9,
        linestyle=(0, (2.4, 1.6)),
        solid_capstyle="round",
        zorder=1.4,
        clip_on=False,
    )
    ax.plot(
        [C(drs, "r")[0], bus_x],
        [C(drs, "r")[1], C(drs, "r")[1]],
        color=DASH,
        lw=0.9,
        linestyle=(0, (2.4, 1.6)),
        solid_capstyle="round",
        zorder=1.4,
        clip_on=False,
    )
    arrow(ax, join, C(mod, "t"), dashed=True)

    arrow(ax, C(zk, "r"), C(oeb, "l"))
    arrow(ax, C(oeb, "r"), C(mod, "l"))
    arrow(ax, C(mod, "r"), C(shu, "l"))
    arrow(ax, C(shu, "r"), C(rcf, "l"))
    arrow(ax, C(rcf, "b"), C(out, "t"))

    note(
        ax,
        0.18,
        4.55,
        "T1/T2 features build change cues\nthat steer four orientation experts.",
    )
    note(
        ax,
        0.18,
        1.42,
        "CGG: spatial mask.  DRS: orientation weights.\n"
        "OEB: 0° / 45° / 90° / 135°.  RCF: residual $z_k+\\alpha\\cdot\\delta$.",
    )
    note(ax, 0.18, 0.55, "No extra FFT in COPE-S.  The rFFT gate lives only in DFRA.")


def draw_legend(fig):
    items = [
        ("Input / skip", "input"),
        ("Transform", "feat"),
        ("DFRA / gate", "dfra"),
        ("COPE-S", "cope"),
        ("Output", "out"),
    ]
    x0, y, w, h, gap = 0.08, 0.028, 0.155, 0.036, 0.016
    for i, (lab, kind) in enumerate(items):
        x = x0 + i * (w + gap)
        fc, ec = PAL[kind]
        fig.add_artist(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.006,rounding_size=0.03",
                linewidth=0.8,
                facecolor=fc,
                edgecolor=ec,
                transform=fig.transFigure,
                clip_on=False,
            )
        )
        fig.text(
            x + w / 2,
            y + h / 2,
            lab,
            ha="center",
            va="center",
            fontsize=7.2,
            fontweight="bold",
            transform=fig.transFigure,
            clip_on=False,
        )


def main() -> None:
    setup_style(journal="ieee", lang="en", use_sciplots=False)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "figure.constrained_layout.use": False,
            "axes.unicode_minus": False,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 5.35))
    fig.patch.set_facecolor("white")
    for ax in axes:
        ax.set_facecolor("white")
        ax.spines[:].set_visible(False)

    draw_dfra(axes[0])
    draw_cope(axes[1])
    fig.subplots_adjust(left=0.035, right=0.985, top=0.915, bottom=0.105, wspace=0.12)
    draw_legend(fig)

    OUT.mkdir(parents=True, exist_ok=True)
    preview = OUT / "_preview_modules.png"
    render_preview(fig, str(preview), dpi=160)
    print_report(audit_layout(fig))

    stem = str(OUT / "fig_dfra_copes_modules")
    export_figure(
        fig,
        stem,
        formats=["pdf", "svg", "png"],
        dpi=600,
        size_inches=(7.16, 5.35),
        grayscale_preview=True,
        tight=True,
        pad_inches=0.05,
    )
    src_png = OUT / "fig_dfra_copes_modules.png"
    src_pdf = OUT / "fig_dfra_copes_modules.pdf"
    src_svg = OUT / "fig_dfra_copes_modules.svg"
    gray = OUT / "fig_dfra_copes_modules_grayscale.png"
    for dest in COPIES:
        dest.mkdir(parents=True, exist_ok=True)
        for src in (src_png, src_pdf, src_svg, gray, preview):
            if src.exists():
                (dest / src.name).write_bytes(src.read_bytes())
    plt.close(fig)
    print(f"Wrote {stem}.pdf / .svg / .png")


if __name__ == "__main__":
    main()
