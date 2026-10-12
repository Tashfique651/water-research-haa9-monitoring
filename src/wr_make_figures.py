#!/usr/bin/env python3
"""Generate the complete Water Research static figure package.

The script creates one graphical abstract, nine main-text figures, and ten
supplementary figures. SVG is the editable master, PDF is the vector upload
format, and PNG is the 600 dpi review derivative. Every plotted value is
copied to one tidy panel-specific CSV under ``wr_figures/source_data``.

All artwork is code-native.  No external logos, clip art, or generated images
are used.  The script intentionally treats 60 ug/L as a harmonized analytical
benchmark for HAA9, not as an HAA9 regulatory limit.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, PathPatch, Polygon, Rectangle
from matplotlib.path import Path as MplPath


PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "wr_figures"
SOURCE_OUT = OUT / "source_data"

MM = 1.0 / 25.4
FULL_W = 190 * MM
GA_W = 190 * MM

NAVY = "#173B57"
TEAL = "#008C95"
ORANGE = "#E67E22"
MAGENTA = "#B23A73"
SKY = "#5FA8D3"
GREEN = "#3B8C6E"
PURPLE = "#6C5AA7"
GOLD = "#D4A72C"
INK = "#263238"
MID = "#65737E"
LIGHT = "#E8EEF2"
PALE_TEAL = "#DDF2F1"
PALE_ORANGE = "#FAE9D8"
PALE_MAGENTA = "#F5DFE9"
WHITE = "#FFFFFF"
CB = [NAVY, TEAL, ORANGE, MAGENTA, SKY, GREEN, PURPLE, GOLD]

ASSET_ORDER = [
    "Graphical_Abstract",
    "Fig1",
    "Fig2",
    "Fig3",
    "Fig4",
    "Fig5",
    "Fig6",
    "Fig7",
    "Fig8",
    "Fig9",
    "FigS1",
    "FigS2",
    "FigS3",
    "FigS4",
    "FigS5",
    "FigS6",
    "FigS7",
    "FigS8",
    "FigS9",
    "FigS10",
]

LEGACY_ASSET_ORDER = [
    "Graphical_Abstract",
    *[f"Fig{i}" for i in range(1, 9)],
    *[f"FigS{i}" for i in range(1, 11)],
]

BUILD_EPOCH = datetime.fromtimestamp(int(os.environ.get("SOURCE_DATE_EPOCH", "0")), tz=timezone.utc)


def require(condition: bool, message: str) -> None:
    """Raise a production-safe validation error when a required condition fails."""
    if not condition:
        raise RuntimeError(message)


@dataclass
class FigureBundle:
    name: str
    fig: mpl.figure.Figure
    caption: str
    alt_text: str
    sources: list[tuple[str, pd.DataFrame, str]]


def configure_style() -> dict[str, str]:
    """Set a journal-safe, deterministic visual style and resolve Arial Narrow."""
    font_dir = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
    regular_candidates = [font_dir / "ARIALN.TTF", font_dir / "arialn.ttf"]
    bold_candidates = [font_dir / "ARIALNB.TTF", font_dir / "arialnb.ttf"]
    regular_path = next((path for path in regular_candidates if path.is_file()), None)
    bold_path = next((path for path in bold_candidates if path.is_file()), None)
    require(regular_path is not None, "Arial Narrow regular font is required but was not found.")
    require(bold_path is not None, "Arial Narrow bold font is required but was not found.")
    font_manager.fontManager.addfont(str(regular_path))
    font_manager.fontManager.addfont(str(bold_path))
    regular_name = "Arial Narrow"
    bold_name = "Arial Narrow"
    resolved_regular = Path(font_manager.findfont(font_manager.FontProperties(family=regular_name, weight="normal"), fallback_to_default=False)).resolve()
    resolved_bold = Path(font_manager.findfont(font_manager.FontProperties(family=bold_name, weight="bold"), fallback_to_default=False)).resolve()
    require(resolved_regular.name.upper() in {"ARIALN.TTF", "ARIALNI.TTF"}, f"Arial Narrow regular resolved unexpectedly: {resolved_regular}")
    require(resolved_bold.name.upper() in {"ARIALNB.TTF", "ARIALNBI.TTF"}, f"Arial Narrow bold resolved unexpectedly: {resolved_bold}")

    mpl.rcParams.update(
        {
            "font.family": regular_name,
            "font.sans-serif": [regular_name, "Arial Narrow", "Arial"],
            "font.size": 8.5,
            "axes.titlesize": 10.0,
            "axes.titleweight": "bold",
            "axes.labelsize": 8.5,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 8.0,
            "figure.titlesize": 10.0,
            "axes.linewidth": 0.75,
            "lines.linewidth": 1.35,
            "lines.markersize": 4.5,
            "patch.linewidth": 0.75,
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "figure.facecolor": WHITE,
            "axes.facecolor": WHITE,
            "savefig.facecolor": WHITE,
            "savefig.transparent": False,
            "svg.fonttype": "none",
            "svg.hashsalt": "dbp-water-research-figures-v2",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "pdf.use14corefonts": False,
            "axes.unicode_minus": False,
            "legend.frameon": False,
        }
    )
    return {
        "family": regular_name,
        "regular_path": str(regular_path.resolve()),
        "bold_path": str(bold_path.resolve()),
    }


def load_csv(rel: str, required: Sequence[str] = ()) -> pd.DataFrame:
    path = PROJECT / rel
    require(path.exists(), f"Required input is missing: {path}")
    frame = pd.read_csv(path)
    missing = sorted(set(required) - set(frame.columns))
    require(not missing, f"{rel} is missing required columns: {missing}")
    return frame


def load_parquet(rel: str, required: Sequence[str] = ()) -> pd.DataFrame:
    path = PROJECT / rel
    require(path.exists(), f"Required input is missing: {path}")
    frame = pd.read_parquet(path)
    missing = sorted(set(required) - set(frame.columns))
    require(not missing, f"{rel} is missing required columns: {missing}")
    return frame


def panel_title(ax: mpl.axes.Axes, letter: str, text: str, *, pad: float = 7.0) -> None:
    # The description is retained in the caller as semantic documentation only.
    # Requested publication labels contain the panel letter, never a subtitle.
    label = f"({letter})"
    require(re.fullmatch(r"\([a-z]\)",label) is not None, f"Invalid panel label {label}")
    ax.set_title(label, loc="left", pad=pad, fontsize=10.0, fontweight="bold")


def clean_axis(ax: mpl.axes.Axes, *, grid: str | None = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=LIGHT, lw=0.7, zorder=0)
    ax.tick_params(axis="x", labelrotation=0)
    ax.tick_params(axis="y", labelrotation=0)


def percent_axis(ax: mpl.axes.Axes, axis: str = "y", decimals: int = 0) -> None:
    fmt = mpl.ticker.PercentFormatter(1.0, decimals=decimals)
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def vector_colorbar(fig: mpl.figure.Figure, mappable: mpl.cm.ScalarMappable, **kwargs) -> mpl.colorbar.Colorbar:
    """Create a colorbar while keeping its solids as editable PDF/SVG vectors."""
    cb = fig.colorbar(mappable, **kwargs)
    if cb.solids is not None:
        cb.solids.set_rasterized(False)
    return cb


def colored_forest(
    ax: mpl.axes.Axes,
    estimates: Sequence[float],
    y_positions: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
    colors: Sequence[str],
    *,
    size: float = 42,
) -> None:
    """Draw one errorbar at a time so every row can retain its own color."""
    for x, y, lo, hi, color in zip(estimates, y_positions, lower, upper, colors):
        ax.errorbar(
            [x],
            [y],
            xerr=[[x - lo], [hi - x]],
            fmt="none",
            ecolor=color,
            elinewidth=1.4,
            capsize=3,
            zorder=2,
        )
        ax.scatter([x], [y], s=size, color=color, edgecolor=WHITE, lw=0.8, zorder=3)


def pct(x: float, digits: int = 1) -> str:
    return f"{100.0 * x:.{digits}f}%"


def straight_arrow(ax: mpl.axes.Axes, xy0: tuple[float, float], xy1: tuple[float, float], **kwargs) -> None:
    opts = dict(
        arrowstyle="-|>",
        mutation_scale=11,
        color=NAVY,
        lw=1.2,
        transform=ax.transAxes,
        clip_on=False,
        connectionstyle="arc3,rad=0",
    )
    opts.update(kwargs)
    ax.add_patch(FancyArrowPatch(xy0, xy1, **opts))


def round_box(
    ax: mpl.axes.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    *,
    facecolor: str,
    edgecolor: str = NAVY,
    lw: float = 1.0,
    radius: float = 0.025,
    transform=None,
    zorder: int = 2,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=lw,
        transform=transform or ax.transAxes,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def draw_drop(ax: mpl.axes.Axes, center: tuple[float, float], scale: float, color: str, *, alpha: float = 1.0) -> None:
    cx, cy = center
    # A circle plus a short triangular crown is robust under non-square axes
    # transforms and remains a fully editable vector mark.
    ax.add_patch(Circle((cx, cy - 0.18 * scale), 0.72 * scale, facecolor=color, edgecolor=NAVY, lw=0.9, alpha=alpha, transform=ax.transAxes, zorder=4))
    ax.add_patch(Polygon([(cx, cy + 1.12 * scale), (cx - 0.58 * scale, cy + 0.10 * scale), (cx + 0.58 * scale, cy + 0.10 * scale)], closed=True, facecolor=color, edgecolor=NAVY, lw=0.9, alpha=alpha, transform=ax.transAxes, zorder=4))


def rectangle_heatmap(
    ax: mpl.axes.Axes,
    matrix: np.ndarray,
    xlabels: Sequence[str],
    ylabels: Sequence[str],
    *,
    cmap: str | mcolors.Colormap,
    vmin: float,
    vmax: float,
    annotations: set[tuple[int, int]] | None = None,
    fmt: str = ".0%",
    hatch_nan: bool = True,
) -> mpl.cm.ScalarMappable:
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cm = mpl.colormaps[cmap] if isinstance(cmap, str) else cmap
    annotations = annotations or set()
    for iy in range(matrix.shape[0]):
        for ix in range(matrix.shape[1]):
            value = matrix[iy, ix]
            if np.isfinite(value):
                fc = cm(norm(value))
                hatch = None
            else:
                fc = WHITE
                hatch = "///" if hatch_nan else None
            ax.add_patch(Rectangle((ix, iy), 1, 1, facecolor=fc, edgecolor=WHITE, lw=0.9, hatch=hatch))
            if (iy, ix) in annotations and np.isfinite(value):
                rgb = np.asarray(mcolors.to_rgb(fc))
                luminance = float(np.dot(rgb, [0.2126, 0.7152, 0.0722]))
                ax.text(ix + 0.5, iy + 0.5, format(value, fmt), ha="center", va="center", color=WHITE if luminance < 0.48 else INK, fontsize=8.0, fontweight="bold")
    ax.set_xlim(0, matrix.shape[1])
    ax.set_ylim(matrix.shape[0], 0)
    ax.set_xticks(np.arange(matrix.shape[1]) + 0.5, xlabels)
    ax.set_yticks(np.arange(matrix.shape[0]) + 0.5, ylabels)
    ax.tick_params(length=0, labelrotation=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return mpl.cm.ScalarMappable(norm=norm, cmap=cm)


def binned_density(x: np.ndarray, y: np.ndarray, *, xmax: float, ymax: float, bins: int = 42) -> pd.DataFrame:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x = np.clip(x[ok], 0, xmax)
    y = np.clip(y[ok], 0, ymax)
    hist, xe, ye = np.histogram2d(x, y, bins=[bins, bins], range=[[0, xmax], [0, ymax]])
    rows = []
    for ix in range(bins):
        for iy in range(bins):
            if hist[ix, iy] > 0:
                rows.append(
                    {
                        "x_left": xe[ix],
                        "x_right": xe[ix + 1],
                        "x_mid": (xe[ix] + xe[ix + 1]) / 2,
                        "y_left": ye[iy],
                        "y_right": ye[iy + 1],
                        "y_mid": (ye[iy] + ye[iy + 1]) / 2,
                        "count": int(hist[ix, iy]),
                    }
                )
    return pd.DataFrame(rows)


def plot_density_rects(ax: mpl.axes.Axes, bins: pd.DataFrame, *, cmap: str = "Blues", zmax: float | None = None) -> mpl.cm.ScalarMappable:
    vmax = zmax or float(np.log10(bins["count"].max() + 1))
    norm = mcolors.Normalize(vmin=0, vmax=vmax)
    cm = mpl.colormaps[cmap]
    for row in bins.itertuples(index=False):
        z = math.log10(row.count + 1)
        ax.add_patch(
            Rectangle(
                (row.x_left, row.y_left),
                row.x_right - row.x_left,
                row.y_right - row.y_left,
                facecolor=cm(norm(z)),
                edgecolor="none",
                lw=0,
            )
        )
    return mpl.cm.ScalarMappable(norm=norm, cmap=cm)


def source_frame(frame: pd.DataFrame, panel: str, relpath: str) -> pd.DataFrame:
    out = frame.copy()
    out.insert(0, "source_input_path", relpath)
    out.insert(0, "source_panel", panel)
    return out


def ga() -> FigureBundle:
    pair = load_parquet(
        "data/processed/ucmr4/paired_haa_events.parquet",
        ["PWSID", "HAA5", "HAA9", "finite_grid_system_eligible"],
    )
    main = load_csv("results/ucmr4/four_cell_primary_B60.csv", ["split", "population", "y00", "y10", "y01", "y11"])
    chrono = load_csv("wr_results/core/chronological_disjoint_B60_summary.csv", ["direction", "interaction", "interaction_ci_lower", "interaction_ci_upper"])
    row = main[(main["split"] == "all") & (main["population"] == "reference_positive")].iloc[0]
    eligible = pair[pair["finite_grid_system_eligible"].astype(bool)]
    n_events = len(eligible)
    n_systems = eligible["PWSID"].nunique()
    require(n_systems == 4904 and n_events == 63105, "Unexpected UCMR4 finite-grid cohort size")
    summary = pd.DataFrame(
        {
            "metric": ["eligible_events", "eligible_systems", "reference_positive_systems", "y00", "y10", "y01", "y11", "chronological_interaction_min", "chronological_interaction_max"],
            "value": [n_events, n_systems, int(row["n_reference_positive_systems"]), row.y00, row.y10, row.y01, row.y11, chrono.interaction.min(), chrono.interaction.max()],
        }
    )

    fig, ax = plt.subplots(figsize=(GA_W, GA_W / 2.5), layout="constrained")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.02, 0.93, "OBSERVED COVERAGE DEFINES MONITORING SIGNAL CAPTURE", fontsize=13.0, fontweight="bold", color=NAVY, va="top")
    ax.text(0.02, 0.84, "Public archives  |  finite observed grids  |  exact design replay", fontsize=8.8, color=MID, va="top")

    # Archive field.
    round_box(ax, (0.025, 0.21), 0.20, 0.50, facecolor="#F3F7FA", edgecolor=NAVY)
    draw_drop(ax, (0.075, 0.57), 0.040, SKY)
    for i, (dx, dy, c) in enumerate([(0.13, 0.59, TEAL), (0.18, 0.53, ORANGE), (0.125, 0.43, MAGENTA), (0.18, 0.39, NAVY)]):
        ax.add_patch(Circle((dx, dy), 0.018 + 0.002 * i, facecolor=c, edgecolor=WHITE, lw=0.8, transform=ax.transAxes, zorder=4))
    ax.text(0.125, 0.665, "PUBLIC ARCHIVE", ha="center", va="center", fontsize=8.2, fontweight="bold", color=NAVY)
    ax.text(0.125, 0.28, f"{n_events:,} matched events\n{n_systems:,} systems", ha="center", va="center", fontweight="bold", fontsize=9.4)
    ax.text(0.125, 0.18, "UCMR4 observed records", ha="center", va="top", fontsize=8.2, color=MID)

    straight_arrow(ax, (0.235, 0.46), (0.285, 0.46), color=TEAL)

    # Nested panels.
    round_box(ax, (0.295, 0.21), 0.20, 0.50, facecolor=PALE_TEAL, edgecolor=TEAL)
    ax.add_patch(Circle((0.385, 0.49), 0.108, facecolor=WHITE, edgecolor=TEAL, lw=1.6, transform=ax.transAxes, zorder=3))
    ax.add_patch(Circle((0.385, 0.49), 0.066, facecolor=PALE_ORANGE, edgecolor=ORANGE, lw=1.6, transform=ax.transAxes, zorder=4))
    for angle in np.linspace(0, 2 * np.pi, 4, endpoint=False):
        ax.add_patch(Circle((0.385 + 0.086 * np.cos(angle), 0.49 + 0.086 * np.sin(angle)), 0.010, facecolor=TEAL, edgecolor=WHITE, lw=0.6, transform=ax.transAxes, zorder=5))
    ax.text(0.385, 0.50, "HAA5", ha="center", va="center", fontsize=10.0, fontweight="bold", color=NAVY, zorder=6)
    ax.text(0.385, 0.635, "HAA9 = 5 + 4", ha="center", fontsize=9.2, fontweight="bold", color=TEAL)
    ax.text(0.395, 0.28, "Panel scope", ha="center", fontsize=8.5, color=MID)

    straight_arrow(ax, (0.505, 0.46), (0.555, 0.46), color=TEAL)

    # Observation grid.
    round_box(ax, (0.565, 0.21), 0.20, 0.50, facecolor="#F3F7FA", edgecolor=NAVY)
    for r in range(3):
        for c in range(4):
            fc = TEAL if (r, c) in {(0, 0), (0, 2), (1, 1), (2, 0), (2, 3)} else LIGHT
            ax.add_patch(Rectangle((0.598 + c * 0.034, 0.37 + r * 0.064), 0.026, 0.045, facecolor=fc, edgecolor=WHITE, lw=0.9, transform=ax.transAxes, zorder=4))
    for c in range(4):
        ax.add_patch(Circle((0.611 + c * 0.034, 0.558), 0.006, facecolor=NAVY, edgecolor=WHITE, lw=0.5, transform=ax.transAxes, zorder=5))
    for r in range(3):
        ax.add_patch(Rectangle((0.582, 0.382 + r * 0.064), 0.009, 0.020, facecolor=ORANGE, edgecolor=WHITE, lw=0.5, transform=ax.transAxes, zorder=5))
    ax.text(0.665, 0.62, "Sites x groups", ha="center", fontweight="bold", fontsize=9.5, zorder=6)
    ax.text(0.665, 0.28, "Observation support", ha="center", fontsize=8.5, color=MID)

    straight_arrow(ax, (0.775, 0.46), (0.815, 0.46), color=TEAL)

    # Result field.
    round_box(ax, (0.825, 0.21), 0.15, 0.50, facecolor=PALE_ORANGE, edgecolor=ORANGE)
    values = [("HAA5 / one group", row.y00, NAVY), ("HAA9 / one group", row.y10, TEAL), ("HAA5 / full grid", row.y01, ORANGE), ("HAA9 / full grid", row.y11, MAGENTA)]
    for i, (label, value, color) in enumerate(values):
        y = 0.625 - i * 0.102
        ax.add_patch(Circle((0.855, y), 0.021, facecolor=color, edgecolor=WHITE, lw=0.8, transform=ax.transAxes, zorder=4))
        ax.text(0.855, y, f"{100*value:.0f}", ha="center", va="center", color=WHITE, fontsize=8.0, fontweight="bold", zorder=5)
        ax.text(0.883, y, label, ha="left", va="center", fontsize=8.0, color=INK)
    ax.text(0.90, 0.235, "Conditional capture (%)", ha="center", fontsize=8.2, color=MID)

    chrono_text = "/".join(f"{100*x:.1f}" for x in sorted(chrono.interaction))
    ax.text(0.50, 0.095, f"Chronological two-round transfer: +{chrono_text} percentage points", ha="center", va="center", fontsize=9.2, fontweight="bold", color=NAVY)
    ax.text(0.50, 0.035, "60 µg/L is a harmonized analytical HAA9 benchmark, not an HAA9 regulatory limit", ha="center", va="center", fontsize=8.0, color=MID)

    caption = (
        "Graphical abstract. Public UCMR4 records were restricted to matched HAA5-HAA9 observations and replayed on each system's finite observed site-by-group grid. "
        "Conditional capture among 769 systems above the 60 µg/L HAA9 analytical benchmark rose from 20.9% for one HAA5 group to 100% for the full observed HAA9 grid; disjoint chronological evaluation retained a positive 4.4-4.6 percentage-point interaction."
    )
    alt = (
        "A left-to-right vector graphic links 63,105 matched records from 4,904 systems to nested HAA5 and HAA9 panels, then to a site-by-group observation grid. "
        "Four colored markers report conditional capture of 21%, 42%, 58%, and 100%. A footer reports positive chronological interaction of 4.4 to 4.6 percentage points and states that 60 micrograms per liter is an analytical HAA9 benchmark, not a regulatory limit."
    )
    return FigureBundle("Graphical_Abstract", fig, caption, alt, [("summary", summary, "derived from UCMR4 primary and chronological tables")])


def fig1() -> FigureBundle:
    pair = load_parquet("data/processed/ucmr4/paired_haa_events.parquet", ["PWSID", "SampleEventCode", "site_id", "CollectionDate", "HAA9", "finite_grid_system_eligible"])
    sem = load_csv("wr_results/core/round_semantics_summary.csv", ["metric", "count", "denominator", "fraction"])
    audit = load_csv("wr_results/core/joint_round_site_grid_audit.csv", ["PWSID", "observed_rounds", "observed_sites", "missing_cells", "rectangular_observed_grid", "reference_positive_B60"])
    b60 = load_csv("wr_results/core/threshold_factorial_with_intervals.csv", ["analysis", "threshold_ug_l", "p00", "p10", "p01", "p11"])
    strict = load_csv("wr_results/validation/syr4_strict_site_and_overlap_sensitivity_metrics.csv", ["cohort", "metric", "estimate", "n_pairs", "n_systems"])
    icr = load_csv("wr_results/validation/icr_temporal_support_comparison.csv", ["analysis_version", "scope", "analyte", "time_scale", "full_six_quarter_systems"])
    caqa = load_csv("wr_results/california/ca_data_quality_audit.csv", ["section", "metric", "value"])

    selected = audit[(~audit["rectangular_observed_grid"].astype(bool)) & audit["reference_positive_B60"].astype(bool) & (audit["observed_rounds"] <= 4) & (audit["observed_sites"] <= 8)].sort_values(["missing_cells", "observed_sites"], ascending=[False, False]).iloc[0]
    ex = pair[(pair["PWSID"].astype(str) == str(selected.PWSID)) & pair["finite_grid_system_eligible"].astype(bool)].copy()
    ex["CollectionDate"] = pd.to_datetime(ex["CollectionDate"])
    round_order = ex.groupby("SampleEventCode")["CollectionDate"].median().sort_values().index.tolist()
    site_order = sorted(ex["site_id"].astype(str).unique())
    pivot = ex.pivot_table(index="SampleEventCode", columns="site_id", values="HAA9", aggfunc="max").reindex(index=round_order, columns=site_order)
    ex_plot = (
        pivot.rename_axis(index="SampleEventCode", columns="site_id")
        .reset_index()
        .melt(id_vars="SampleEventCode", var_name="site_id", value_name="HAA9_ug_l")
    )
    ex_plot["example_label"] = "anonymized irregular observed grid"

    sem_keep = sem[sem.metric.isin(["multi_date", "multi_facility", "multi_site", "crosses_calendar_year"])].copy()
    sem_order = ["multi_date", "multi_facility", "multi_site", "crosses_calendar_year"]
    sem_keep["metric"] = pd.Categorical(sem_keep.metric, sem_order, ordered=True)
    sem_keep = sem_keep.sort_values("metric")
    b = b60[(b60["analysis"] == "B60")].iloc[0]
    cell = pd.DataFrame({"panel_scope": ["HAA5", "HAA9", "HAA5", "HAA9"], "observation_scope": ["one group", "one group", "full grid", "full grid"], "capture": [b.p00, b.p10, b.p01, b.p11]})

    syr = strict[(strict.cohort == "strict_site_primary") & (strict.metric == "all_linked_pairs")].iloc[0]
    icrrow = icr[(icr.analysis_version == "mrl_bounded_recovery") & (icr.scope == "primary_distribution") & (icr.analyte == "HAA5") & (icr.time_scale == "quarterly")].iloc[0]
    ca_records = float(caqa.loc[caqa.metric.str.contains("valid primary analysis records"), "value"].iloc[0])
    archive = pd.DataFrame(
        {
            "archive": ["UCMR4", "SYR4", "ICR", "California"],
            "role": ["primary exact replay", "federal external archive", "historical temporal check", "current state context"],
            "plotted_count": [pair[pair.finite_grid_system_eligible.astype(bool)].shape[0], syr.estimate, icrrow.full_six_quarter_systems, ca_records],
            "unit": ["matched events", "strict linked pairs", "six-quarter HAA5 systems", "routine records"],
        }
    )

    fig = plt.figure(figsize=(FULL_W, 154 * MM))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.86, 1.14], left=0.075, right=0.975, bottom=0.10, top=0.945, wspace=0.48, hspace=0.58)
    axa, axb, axc, axd = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])

    panel_title(axa, "a", "Archive-specific evidence tiers")
    axa.set_xlim(0, 1); axa.set_ylim(0, 1); axa.axis("off")
    xs = [0.11, 0.37, 0.63, 0.89]
    sizes = [0.067, 0.079, 0.052, 0.069]
    labels = ["63,105", "527,172", "167", "84,304"]
    roles = ["PRIMARY", "EXTERNAL", "HISTORICAL", "CONTEXT"]
    for i, row in archive.iterrows():
        x = xs[i]
        axa.add_patch(Circle((x, 0.59), sizes[i], facecolor=CB[i], edgecolor=WHITE, lw=1.0, transform=axa.transAxes))
        axa.text(x, 0.59, labels[i], ha="center", va="center", color=WHITE, fontsize=8.0, fontweight="bold")
        axa.text(x, 0.38, row.archive, ha="center", va="center", fontsize=8.5, fontweight="bold", color=INK)
        axa.text(x, 0.23, roles[i], ha="center", va="center", fontsize=8.0, color=CB[i], fontweight="bold")
        if i < 3:
            axa.plot([x + sizes[i] + 0.018, xs[i + 1] - sizes[i + 1] - 0.018], [0.59, 0.59], color=LIGHT, lw=2.0, transform=axa.transAxes, clip_on=False)
    axa.text(0.50, 0.055, "Separate cohorts, periods, and linkage rules", ha="center", fontsize=8.0, color=MID)

    panel_title(axb, "b", "Administrative-group semantics")
    labels_b = ["Multiple dates", "Multiple facilities", "Multiple sites", "Cross-year"]
    y = np.arange(len(sem_keep))
    axb.barh(y, sem_keep.fraction, color=[TEAL, ORANGE, NAVY, MAGENTA], edgecolor=WHITE, zorder=2)
    for yi, v, n in zip(y, sem_keep.fraction, sem_keep["count"]):
        if v > 0.80:
            axb.text(v - 0.018, yi, f"{100*v:.1f}%  (n={int(n):,})", ha="right", va="center", fontsize=8.0, color=WHITE, fontweight="bold")
        else:
            axb.text(v + 0.014, yi, f"{100*v:.1f}%  (n={int(n):,})", va="center", fontsize=8.0)
    axb.set_yticks(y, labels_b)
    axb.invert_yaxis(); axb.set_xlim(0, 1.05); axb.set_xlabel("Fraction of 15,285 event-code groups")
    percent_axis(axb, "x"); clean_axis(axb, grid="x")

    panel_title(axc, "c", "An irregular observed grid")
    matrix = pivot.to_numpy(float)
    vmax = max(60, float(np.nanpercentile(matrix, 95)))
    sm = rectangle_heatmap(axc, matrix, [f"S{i+1}" for i in range(len(site_order))], [f"G{i+1}" for i in range(len(round_order))], cmap=mpl.colors.LinearSegmentedColormap.from_list("haa", [WHITE, SKY, NAVY]), vmin=0, vmax=vmax, annotations=set(), hatch_nan=True)
    for iy in range(matrix.shape[0]):
        for ix in range(matrix.shape[1]):
            if np.isfinite(matrix[iy, ix]) and matrix[iy, ix] > 60:
                axc.scatter(ix + 0.5, iy + 0.5, marker="*", s=58, color=MAGENTA, edgecolor=WHITE, lw=0.6, zorder=5)
    axc.set_xlabel("Observed site"); axc.set_ylabel("Administrative group")
    axc.set_aspect("auto")
    cax = axc.inset_axes([0.82, 0.60, 0.035, 0.30])
    cbar = vector_colorbar(fig, sm, cax=cax)
    cbar.set_label("HAA9 (µg/L)"); cbar.ax.tick_params(labelsize=8)
    axc.text(0.02, 0.96, "Asterisk: above 60 µg/L", transform=axc.transAxes, va="top", fontsize=8.0, color=MAGENTA, fontweight="bold")
    axc.text(0.00, -0.19, f"Example: {int(selected.observed_rounds)} groups x {int(selected.observed_sites)} sites; {int(selected.missing_cells)} cells absent", transform=axc.transAxes, fontsize=8.0, color=MID)

    panel_title(axd, "d", "Panel x observation replay")
    m = np.array([[b.p00, b.p10], [b.p01, b.p11]])
    sm2 = rectangle_heatmap(axd, m, ["HAA5", "HAA9"], ["One group", "Full grid"], cmap=mpl.colors.LinearSegmentedColormap.from_list("capture", [WHITE, PALE_TEAL, TEAL, NAVY]), vmin=0, vmax=1, annotations={(0, 0), (0, 1), (1, 0), (1, 1)})
    axd.set_xlabel("Panel scope"); axd.set_ylabel("Observation scope")
    axd.add_patch(Rectangle((1.03, 1.03), 0.94, 0.94, fill=False, edgecolor=MAGENTA, lw=2.0))
    axd.text(0.5, -0.18, "Outlined cell defines the reference", transform=axd.transAxes, ha="center", fontsize=8.0, color=MAGENTA)

    caption = (
        "Figure 1. Finite-grid replay and evidence boundaries. (a) Four public archives contribute distinct evidence tiers and are not pooled as exchangeable observations. "
        "(b) UCMR4 SampleEventCode groups can span dates, facilities, or sites. (c) An observed site-by-group grid illustrates explicit missing cells; labels are anonymized and colored cells show HAA9 concentrations. "
        "(d) The 2 x 2 replay changes panel scope and observation scope while holding the reference-positive cohort fixed."
    )
    alt = (
        "Four-panel methods figure. Panel a shows UCMR4, SYR4, ICR, and California as separate evidence tiers. Panel b shows that 6.5% of UCMR4 event-code groups span multiple dates, 17.9% multiple facilities, 93.7% multiple sites, and 0.5% cross calendar years. "
        "Panel c is a heatmap of one irregular observed site-by-group grid with hatched missing cells. Panel d is a 2 by 2 matrix with conditional capture values 20.9%, 41.9%, 57.7%, and 100%."
    )
    return FigureBundle(
        "Fig1", fig, caption, alt,
        [
            ("a", archive, "derived from four archive audit tables"),
            ("b", sem_keep, "wr_results/core/round_semantics_summary.csv"),
            ("c", ex_plot, "data/processed/ucmr4/paired_haa_events.parquet"),
            ("d", cell, "wr_results/core/threshold_factorial_with_intervals.csv"),
        ],
    )


def fig2() -> FigureBundle:
    pair = load_parquet("data/processed/ucmr4/paired_haa_events.parquet", ["HAA5", "HAA9", "finite_grid_system_eligible"])
    screen = load_csv("wr_results/core/screening_curve.csv", ["haa5_action_threshold_ug_l", "sensitivity", "followup_fraction", "positive_predictive_value"])
    folds = load_csv("wr_results/core/screening_crossfit_folds.csv", ["fold", "selected_HAA5_action_threshold_ug_l", "evaluation_sensitivity", "evaluation_followup_fraction"])
    ops = load_csv("wr_results/core/screening_operating_points.csv", ["policy", "sensitivity", "sensitivity_ci_lower", "sensitivity_ci_upper", "followup_fraction"])
    chem = pd.DataFrame(
        {
            "panel": ["HAA5"] * 5 + ["HAA9 additional four"] * 4,
            "species": ["MCAA", "DCAA", "TCAA", "MBAA", "DBAA", "BCAA", "BDCAA", "CDBAA", "TBAA"],
            "position": np.arange(9) + 1,
        }
    )
    eligible = pair[pair.finite_grid_system_eligible.astype(bool)]
    xmax = float(np.ceil(np.nanpercentile(np.r_[eligible.HAA5, eligible.HAA9], 99.5) / 20) * 20)
    xmax = max(120.0, min(xmax, 260.0))
    dens = binned_density(eligible.HAA5.to_numpy(), eligible.HAA9.to_numpy(), xmax=xmax, ymax=xmax, bins=46)

    fig = plt.figure(figsize=(FULL_W, 145 * MM))
    gs = fig.add_gridspec(2, 2, width_ratios=[0.92, 1.08], left=0.09, right=0.975, bottom=0.11, top=0.945, hspace=0.52, wspace=0.48)
    axa, axb, axc, axd = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]

    panel_title(axa, "a", "HAA9 nests four additional acids")
    axa.set_xlim(0, 10); axa.set_ylim(0, 3.1); axa.axis("off")
    axa.add_patch(FancyBboxPatch((0.42, 0.74), 5.05, 1.72, boxstyle="round,pad=0.12,rounding_size=0.18", fc=PALE_ORANGE, ec=ORANGE, lw=1.2))
    axa.add_patch(FancyBboxPatch((0.15, 0.40), 9.55, 2.45, boxstyle="round,pad=0.12,rounding_size=0.18", fc="none", ec=TEAL, lw=1.4))
    positions = [
        (1.18, 1.88), (2.94, 1.88), (4.70, 1.88), (2.05, 1.18), (3.82, 1.18),
        (6.55, 1.78), (8.40, 1.78), (6.55, 0.92), (8.40, 0.92),
    ]
    for i, row in chem.iterrows():
        x, y = positions[i]
        color = ORANGE if i < 5 else TEAL
        axa.add_patch(FancyBboxPatch((x-0.65, y-0.25), 1.30, 0.50, boxstyle="round,pad=0.04,rounding_size=0.12", facecolor=color, edgecolor=WHITE, lw=1.0))
        axa.text(x, y, row.species, ha="center", va="center", color=WHITE, fontsize=8.0, fontweight="bold")
    axa.text(2.95, 2.63, "HAA5", color=ORANGE, ha="center", fontweight="bold", fontsize=9.0)
    axa.text(7.85, 2.63, "HAA9 scope", color=TEAL, ha="center", fontweight="bold", fontsize=9.0)
    axa.text(7.48, 2.28, "+4 species", color=TEAL, ha="center", fontsize=8.2)

    panel_title(axb, "b", "Matched panels diverge in the upper tail")
    sm = plot_density_rects(axb, dens, cmap="Blues")
    axb.plot([0, xmax], [0, xmax], color=MID, ls="--", lw=1.0, zorder=4)
    axb.axhline(60, color=MAGENTA, lw=1.0, ls=":")
    axb.axvline(60, color=ORANGE, lw=1.0, ls=":")
    axb.set(xlim=(0, xmax), ylim=(0, xmax), xlabel="HAA5 (µg/L)", ylabel="HAA9 (µg/L)")
    clean_axis(axb, grid=None)
    cb = vector_colorbar(fig, sm, ax=axb, fraction=0.046, pad=0.02)
    cb.set_label("log10(events + 1)")
    axb.text(0.97, 0.04, "clipped at 99.5th percentile", transform=axb.transAxes, ha="right", fontsize=8.0, color=MID)

    panel_title(axc, "c", "Calibrated HAA5 trigger")
    axc.plot(screen.followup_fraction, screen.sensitivity, color=NAVY, lw=1.8)
    axc.fill_between(screen.followup_fraction, 0, screen.sensitivity, color=SKY, alpha=0.10)
    op = ops[ops.policy == "cross_fitted_HAA5_trigger"].iloc[0]
    direct = ops[ops.policy == "HAA9_at_60_one_event_code_group"].iloc[0]
    axc.scatter([op.followup_fraction], [op.sensitivity], s=55, color=ORANGE, edgecolor=WHITE, lw=1.0, zorder=4, label="Cross-fitted trigger")
    axc.scatter([direct.followup_fraction], [direct.sensitivity], s=55, marker="s", color=TEAL, edgecolor=WHITE, lw=1.0, zorder=4, label="Direct one-group HAA9")
    axc.annotate(f"{pct(op.sensitivity)} sensitivity\n{pct(op.followup_fraction)} follow-up", xy=(op.followup_fraction, op.sensitivity), xytext=(0.23, 0.56), textcoords="axes fraction", fontsize=8.0, arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=0.9))
    axc.set(xlim=(0, 0.55), ylim=(0, 1.02), xlabel="Systems sent to HAA9 follow-up", ylabel="HAA9 sensitivity")
    percent_axis(axc, "x"); percent_axis(axc, "y"); clean_axis(axc, grid="both")
    axc.legend(loc="lower right")

    panel_title(axd, "d", "Five-fold evaluation")
    fold_offsets = {0: (8, 8), 1: (8, 8), 2: (8, -14), 3: (8, 8), 4: (8, 8)}
    for r in folds.itertuples(index=False):
        axd.scatter(r.evaluation_followup_fraction, r.evaluation_sensitivity, s=48, color=CB[int(r.fold) % len(CB)], edgecolor=WHITE, lw=0.8, zorder=3)
        axd.annotate(f"F{int(r.fold)+1}", (r.evaluation_followup_fraction, r.evaluation_sensitivity), xytext=fold_offsets[int(r.fold)], textcoords="offset points", fontsize=8.0, color=INK)
    axd.scatter(op.followup_fraction, op.sensitivity, s=90, facecolor=WHITE, edgecolor=NAVY, lw=1.5, zorder=4, label="Pooled cross-fit")
    axd.scatter(direct.followup_fraction, direct.sensitivity, s=65, marker="s", facecolor=WHITE, edgecolor=TEAL, lw=1.3, zorder=4, label="Direct HAA9")
    axd.set(xlim=(0.055, 0.105), ylim=(0.36, 0.49), xlabel="Evaluation follow-up fraction", ylabel="Evaluation sensitivity")
    percent_axis(axd, "x"); percent_axis(axd, "y"); clean_axis(axd, grid="both"); axd.legend(loc="lower right")

    caption = (
        "Figure 2. Panel chemistry and internally calibrated screening. (a) HAA9 contains the five HAA5 species plus four additional haloacetic acids. "
        "(b) Matched UCMR4 panel totals are shown as a vector binned-density field; values above the 99.5th percentile are clipped only for display. "
        "(c) The HAA5 trigger curve relates HAA9 follow-up workload to sensitivity for the observed-grid HAA9 benchmark. (d) Five system-level folds show the selected operating point and are compared with direct one-group HAA9 measurement."
    )
    alt = (
        "Four panels show nine nested haloacetic-acid species, a density plot of HAA5 versus HAA9, a screening tradeoff curve, and five cross-validation fold points. "
        "The cross-fitted HAA5 trigger uses 7.7% follow-up for 42.2% sensitivity, similar to 41.9% for direct one-group HAA9 measurement."
    )
    return FigureBundle(
        "Fig2", fig, caption, alt,
        [
            ("a", chem, "author-defined nested panel membership"),
            ("b", dens, "data/processed/ucmr4/paired_haa_events.parquet"),
            ("c", screen, "wr_results/core/screening_curve.csv"),
            ("c", ops, "wr_results/core/screening_operating_points.csv"),
            ("d", folds, "wr_results/core/screening_crossfit_folds.csv"),
        ],
    )


def fig3() -> FigureBundle:
    boot = load_csv("results/ucmr4/bootstrap_system_cluster_B60.csv", ["split", "metric", "estimate", "ci_lower_2_5pct", "ci_upper_97_5pct"])
    primary = load_csv("results/ucmr4/four_cell_primary_B60.csv", ["split", "population", "y00", "y10", "y01", "y11"])
    pathways = load_csv("results/ucmr4/repair_categories_B60.csv", ["design", "split", "category", "system_weighted_fraction"])
    b = primary[(primary.split == "all") & (primary.population == "reference_positive")].iloc[0]
    cells = pd.DataFrame({"cell": ["p00", "p10", "p01", "p11"], "capture": [b.y00, b.y10, b.y01, b.y11], "panel": ["HAA5", "HAA9", "HAA5", "HAA9"], "observations": ["one group", "one group", "full grid", "full grid"]})
    contrast_names = ["conditional_panel_gain_at_M0", "conditional_design_gain_at_HAA5", "conditional_combined_gain", "conditional_interaction"]
    contrasts = boot[(boot.split == "all") & boot.metric.isin(contrast_names)].copy()
    order = contrast_names
    contrasts["metric"] = pd.Categorical(contrasts.metric, order, ordered=True)
    contrasts = contrasts.sort_values("metric")
    paths = pathways[(pathways.design == "one_round") & (pathways.split == "all")].copy()
    path_order = ["already_detected", "panel_only", "design_only", "either_change", "both_required"]
    paths["category"] = pd.Categorical(paths.category, path_order, ordered=True)
    paths = paths.sort_values("category")
    denominator = boot[(boot.split == "all") & boot.metric.isin(["conditional_y00", "conditional_y10", "conditional_y01", "conditional_y11", "unconditional_y00", "unconditional_y10", "unconditional_y01", "unconditional_y11"])].copy()
    denominator["denominator"] = np.where(denominator.metric.str.startswith("conditional"), "Reference-positive cohort", "All eligible systems")
    denominator["cell"] = denominator.metric.str.extract(r"(y\d\d)", expand=False)

    fig = plt.figure(figsize=(FULL_W, 160 * MM))
    gs = fig.add_gridspec(2, 2, left=0.08, right=0.975, bottom=0.13, top=0.945, hspace=0.56, wspace=0.47)
    axa, axb, axc, axd = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]

    panel_title(axa, "a", "Four-cell replay at 60 µg/L")
    matrix = np.array([[b.y00, b.y10], [b.y01, b.y11]])
    sm = rectangle_heatmap(axa, matrix, ["HAA5", "HAA9"], ["One group", "Full grid"], cmap=mpl.colors.LinearSegmentedColormap.from_list("cap", [WHITE, SKY, TEAL, NAVY]), vmin=0, vmax=1, annotations={(0, 0), (0, 1), (1, 0), (1, 1)})
    axa.set_xlabel("Panel scope"); axa.set_ylabel("Observation scope")
    vector_colorbar(fig, sm, ax=axa, fraction=0.046, pad=0.02, label="Conditional capture")

    panel_title(axb, "b", "Gains are non-additive")
    lab = ["Panel gain\nat one group", "Observation gain\nunder HAA5", "Combined gain", "Interaction"]
    y = np.arange(len(contrasts))[::-1]
    x = contrasts.estimate.to_numpy()
    lo = contrasts.ci_lower_2_5pct.to_numpy(); hi = contrasts.ci_upper_97_5pct.to_numpy()
    colors = [TEAL, ORANGE, NAVY, MAGENTA]
    colored_forest(axb, x, y, lo, hi, colors, size=45)
    for xi, yi in zip(x, y): axb.text(xi + 0.025, yi, pct(xi), va="center", fontsize=8.0)
    axb.set_yticks(y, lab); axb.set_xlim(0, 0.88); axb.set_xlabel("Conditional gain")
    percent_axis(axb, "x"); clean_axis(axb, grid="x")

    panel_title(axc, "c", "Selected-round pathway composition")
    left = 0.0
    path_labels = ["Already detected", "Panel only", "Observation only", "Either change", "Both required"]
    hatches = ["", "//", "..", "xx", "\\\\"]
    for i, r in enumerate(paths.itertuples(index=False)):
        axc.barh([0], [r.system_weighted_fraction], left=[left], color=CB[i], hatch=hatches[i], edgecolor=WHITE, height=0.55, label=path_labels[i])
        if r.system_weighted_fraction >= 0.18:
            label_color = WHITE if i in [0, 1, 2, 4] else INK
            axc.text(
                left + r.system_weighted_fraction / 2,
                0,
                f"{100*r.system_weighted_fraction:.1f}%",
                ha="center",
                va="center",
                color=label_color,
                fontsize=8.0,
                fontweight="bold",
                bbox=dict(facecolor=CB[i], edgecolor="none", pad=1.4),
            )
        left += r.system_weighted_fraction
    axc.set(xlim=(0, 1), yticks=[], xlabel="System-weighted realization share")
    percent_axis(axc, "x"); clean_axis(axc, grid=None)
    axc.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, handlelength=1.5, columnspacing=0.9)

    panel_title(axd, "d", "Conditional versus achieved yield")
    cell_order = ["y00", "y10", "y01", "y11"]
    labels_d = ["HAA5 / one", "HAA9 / one", "HAA5 / full", "HAA9 / full"]
    cond = denominator[denominator.denominator == "Reference-positive cohort"].set_index("cell").loc[cell_order]
    uncond = denominator[denominator.denominator == "All eligible systems"].set_index("cell").loc[cell_order]
    y = np.arange(4)
    axd.hlines(y, uncond.estimate, cond.estimate, color=LIGHT, lw=3, zorder=1)
    axd.scatter(cond.estimate, y, s=45, color=NAVY, label="Conditional capture", zorder=3)
    axd.scatter(uncond.estimate, y, s=45, marker="s", color=ORANGE, label="Achieved-cohort yield", zorder=3)
    for yi, xc, xu in zip(y, cond.estimate, uncond.estimate):
        if xc >= 0.98:
            axd.text(xc - 0.018, yi - 0.20, pct(xc, 0), ha="right", va="center", fontsize=8.0, color=NAVY)
        else:
            axd.text(xc, yi + 0.18, pct(xc, 0), ha="center", va="center", fontsize=8.0, color=NAVY)
        achieved_y = yi + 0.30 if yi == 0 else yi - 0.18
        axd.text(xu, achieved_y, pct(xu, 1), ha="center", va="center", fontsize=8.0, color=ORANGE)
    axd.set_yticks(y, labels_d); axd.invert_yaxis(); axd.set_xlim(0, 1.06); axd.set_xlabel("System-weighted fraction")
    percent_axis(axd, "x"); clean_axis(axd, grid="x"); axd.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)

    caption = (
        "Figure 3. Primary factorial replay at the 60 µg/L harmonized analytical benchmark. (a) Four conditional capture cells vary panel scope and observation scope. "
        "(b) System-cluster bootstrap intervals show panel, observation, combined, and interaction contrasts. (c) Five mutually exclusive pathways partition system-weighted selected-round realizations within the 769-system reference-positive cohort. "
        "(d) Conditional capture and achieved-cohort yield use different denominators and are shown separately."
    )
    alt = (
        "Four panels summarize the primary factorial analysis. The four conditional capture cells are 20.9%, 41.9%, 57.7%, and 100%. "
        "A forest plot shows gains of 21.0%, 36.9%, 79.1%, and a 21.2 percentage-point interaction. A stacked bar partitions system-weighted selected-round realizations into five pathways. A dumbbell plot contrasts conditional capture with smaller all-eligible-system yields."
    )
    return FigureBundle(
        "Fig3", fig, caption, alt,
        [
            ("a", cells, "results/ucmr4/four_cell_primary_B60.csv"),
            ("b", contrasts, "results/ucmr4/bootstrap_system_cluster_B60.csv"),
            ("c", paths, "results/ucmr4/repair_categories_B60.csv"),
            ("d", denominator, "results/ucmr4/bootstrap_system_cluster_B60.csv"),
        ],
    )


def _surface_matrix(surface: pd.DataFrame, panel: str) -> np.ndarray:
    sub = surface[surface.panel == panel]
    return sub.pivot(index="requested_rounds", columns="requested_sites", values="conditional_capture").sort_index().sort_index(axis=1).to_numpy()


def fig4() -> FigureBundle:
    surface = load_csv("wr_results/core/joint_round_site_capture_surface.csv", ["panel", "requested_rounds", "requested_sites", "conditional_capture", "ci_lower", "ci_upper"])
    resource = load_csv("wr_results/core/resource_capture_curve.csv", ["panel", "selected_physical_samples", "conditional_capture", "ci_lower", "ci_upper"])
    h5 = _surface_matrix(surface, "HAA5")
    h9 = _surface_matrix(surface, "HAA9")
    diff = h9 - h5
    annotations = {(0, 0), (1, 1), (3, 3), (3, 7)}
    diff_source = surface[surface.panel == "HAA9"].merge(surface[surface.panel == "HAA5"], on=["requested_rounds", "requested_sites"], suffixes=("_HAA9", "_HAA5"))
    diff_source["HAA9_minus_HAA5_capture"] = diff_source.conditional_capture_HAA9 - diff_source.conditional_capture_HAA5

    fig = plt.figure(figsize=(FULL_W, 142 * MM), layout="constrained")
    gs = fig.add_gridspec(2, 2, hspace=0.23, wspace=0.20)
    axa, axb, axc, axd = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    cmap = mpl.colors.LinearSegmentedColormap.from_list("capture", [WHITE, SKY, TEAL, NAVY])
    sm = rectangle_heatmap(axa, h5, [str(i) for i in range(1, 9)], [str(i) for i in range(1, 5)], cmap=cmap, vmin=0, vmax=1, annotations=annotations)
    panel_title(axa, "a", "HAA5 joint capture")
    axa.set(xlabel="Requested sites", ylabel="Requested groups")
    sm2 = rectangle_heatmap(axb, h9, [str(i) for i in range(1, 9)], [str(i) for i in range(1, 5)], cmap=cmap, vmin=0, vmax=1, annotations=annotations)
    panel_title(axb, "b", "HAA9 joint capture")
    axb.set(xlabel="Requested sites", ylabel="Requested groups")
    dcmap = mpl.colors.LinearSegmentedColormap.from_list("gain", [WHITE, PALE_ORANGE, ORANGE, MAGENTA])
    sm3 = rectangle_heatmap(axc, diff, [str(i) for i in range(1, 9)], [str(i) for i in range(1, 5)], cmap=dcmap, vmin=0, vmax=max(0.42, float(np.nanmax(diff))), annotations=annotations)
    panel_title(axc, "c", "Panel-expansion gain")
    axc.set(xlabel="Requested sites", ylabel="Requested groups")
    cbar1 = vector_colorbar(fig, sm, ax=[axa, axb], fraction=0.028, pad=0.02); cbar1.set_label("Conditional capture")
    cbar2 = vector_colorbar(fig, sm3, ax=axc, fraction=0.046, pad=0.02); cbar2.set_label("HAA9 - HAA5")

    panel_title(axd, "d", "Physical-sample capture")
    for panel, color, marker in [("HAA5", ORANGE, "o"), ("HAA9", TEAL, "s")]:
        sub = resource[resource.panel == panel].sort_values("selected_physical_samples")
        axd.plot(sub.selected_physical_samples, sub.conditional_capture, color=color, marker=marker, ms=3.8, label=panel)
        axd.fill_between(sub.selected_physical_samples, sub.ci_lower, sub.ci_upper, color=color, alpha=0.13, linewidth=0)
    axd.set(xlim=(1, 12), ylim=(0, 1.02), xlabel="Physical samples selected", ylabel="Conditional capture")
    axd.set_xticks([1, 2, 4, 6, 8, 10, 12]); percent_axis(axd, "y"); clean_axis(axd, grid="both"); axd.legend(loc="lower right")
    axd.text(0.02, 0.92, "Uniform selection without replacement;\nsaturation at observed support", transform=axd.transAxes, fontsize=8.0, color=MID, va="top")

    caption = (
        "Figure 4. Joint observation support and panel scope. Exact without-replacement replay on 697 rectangular reference-positive grids estimates capture for requested groups and sites under (a) HAA5 and (b) HAA9. "
        "(c) Their difference isolates panel-expansion gain at each joint support level. (d) A separate physical-sample replay shows conditional capture with system-cluster bootstrap intervals; requested support saturates at what was observed."
    )
    alt = (
        "Three four-by-eight heatmaps show HAA5 capture, HAA9 capture, and their difference as requested groups and sites increase. HAA9 is higher throughout. "
        "A fourth panel shows rising HAA5 and HAA9 conditional capture across one to twelve selected physical samples, with HAA9 above HAA5 and shaded confidence intervals."
    )
    return FigureBundle(
        "Fig4", fig, caption, alt,
        [
            ("a-b", surface, "wr_results/core/joint_round_site_capture_surface.csv"),
            ("c", diff_source, "derived from wr_results/core/joint_round_site_capture_surface.csv"),
            ("d", resource, "wr_results/core/resource_capture_curve.csv"),
        ],
    )


def fig5() -> FigureBundle:
    band = load_csv("wr_results/core/threshold_interaction_simultaneous_band.csv", ["threshold_ug_l", "interaction", "pointwise_ci_lower", "pointwise_ci_upper", "simultaneous_ci_lower", "simultaneous_ci_upper"])
    chrono = load_csv("wr_results/core/chronological_disjoint_threshold_curve.csv", ["direction", "threshold_ug_l", "interaction", "interaction_ci_lower", "interaction_ci_upper"])
    b60chrono = load_csv("wr_results/core/chronological_disjoint_B60_summary.csv", ["direction", "interaction", "interaction_ci_lower", "interaction_ci_upper"])
    fixed = load_csv("wr_results/core/fixed_support_factorial.csv", ["analysis", "interaction", "interaction_ci_lower", "interaction_ci_upper"])
    cohort = load_csv("wr_results/core/cohort_operator_margin_sensitivity.csv", ["analysis", "interaction", "interaction_ci_lower", "interaction_ci_upper", "comparison_type"])
    boot = load_csv("results/ucmr4/bootstrap_system_cluster_B60.csv", ["split", "metric", "estimate", "ci_lower_2_5pct", "ci_upper_97_5pct"])
    cont = load_csv("wr_results/core/continuous_interaction_systems.csv", ["continuous_interaction_ug_l"])
    contsum = load_csv("wr_results/core/continuous_interaction_summary.csv", ["mean_continuous_interaction_ug_l", "median_continuous_interaction_ug_l", "fraction_positive", "winsorized_1pct_mean_ug_l"])
    sev = load_csv("wr_results/core/severity_persistence_systems.csv", ["HAA9_margin_above_B60_ug_l", "HAA9_positive_round_fraction", "HAA5_panel_ceiling_miss"])

    hist_count, edges = np.histogram(cont.continuous_interaction_ug_l.clip(-5, 15), bins=np.linspace(-5, 15, 41))
    hist = pd.DataFrame({"bin_left": edges[:-1], "bin_right": edges[1:], "count": hist_count})
    forest_rows = []
    for split, label in [("all", "Primary observed grid"), ("development", "Development split"), ("holdout", "Hashed holdout")]:
        r = boot[(boot.split == split) & (boot.metric == "conditional_interaction")].iloc[0]
        forest_rows.append((label, r.estimate, r.ci_lower_2_5pct, r.ci_upper_97_5pct, "same-window"))
    for _, r in fixed.iloc[1:].iterrows():
        label = {"source_aligned_minimum_rounds": "Source-aligned support", "at_least_four_event_code_groups": "At least four groups", "four_groups_with_repeated_site": "Four groups + repeated site"}[r.analysis]
        forest_rows.append((label, r.interaction, r.interaction_ci_lower, r.interaction_ci_upper, "same-window"))
    for _, r in b60chrono.iterrows():
        label = "Late 2 to early 2" if r.direction.startswith("late2") else "Early 2 to late 2"
        forest_rows.append((label, r.interaction, r.interaction_ci_lower, r.interaction_ci_upper, "chronological"))
    forest = pd.DataFrame(forest_rows, columns=["analysis", "estimate", "ci_lower", "ci_upper", "evidence_stage"])

    fig = plt.figure(figsize=(FULL_W, 178 * MM), layout="constrained")
    outer = fig.add_gridspec(2, 2, height_ratios=[0.87, 1.13], hspace=0.23, wspace=0.22)
    axa = fig.add_subplot(outer[0, 0]); axb = fig.add_subplot(outer[0, 1]); axc = fig.add_subplot(outer[1, 0])
    right = outer[1, 1].subgridspec(2, 1, hspace=0.48)
    axd = fig.add_subplot(right[0, 0]); axe = fig.add_subplot(right[1, 0])

    panel_title(axa, "a", "Threshold-dependent interaction")
    axa.fill_between(band.threshold_ug_l, band.simultaneous_ci_lower, band.simultaneous_ci_upper, color=SKY, alpha=0.16, label="95% simultaneous band")
    axa.fill_between(band.threshold_ug_l, band.pointwise_ci_lower, band.pointwise_ci_upper, color=NAVY, alpha=0.18, label="95% pointwise interval")
    axa.plot(band.threshold_ug_l, band.interaction, color=NAVY, marker="o", ms=3.2)
    axa.axhline(0, color=MID, lw=0.9); axa.axvline(60, color=MAGENTA, ls=":", lw=1.1)
    axa.scatter([60], band.loc[band.threshold_ug_l == 60, "interaction"], s=50, color=MAGENTA, edgecolor=WHITE, zorder=4)
    axa.set(xlabel="Harmonized threshold (µg/L)", ylabel="Interaction", xlim=(20, 100))
    percent_axis(axa, "y"); clean_axis(axa, grid="both"); axa.legend(loc="upper left")

    panel_title(axb, "b", "Disjoint chronological transfer")
    for direction, color, label in [("late2_conditions_early2_evaluates", ORANGE, "Condition late; evaluate early"), ("early2_conditions_late2_evaluates", TEAL, "Condition early; evaluate late")]:
        sub = chrono[chrono.direction == direction].sort_values("threshold_ug_l")
        axb.plot(sub.threshold_ug_l, sub.interaction, color=color, marker="o", ms=3.0, label=label)
        axb.fill_between(sub.threshold_ug_l, sub.interaction_ci_lower, sub.interaction_ci_upper, color=color, alpha=0.12)
    axb.axhline(0, color=MID, lw=0.9); axb.axvline(60, color=MAGENTA, ls=":", lw=1.1)
    axb.set(xlabel="Conditioning threshold (µg/L)", ylabel="Evaluation interaction", xlim=(20, 100))
    percent_axis(axb, "y"); clean_axis(axb, grid="both"); axb.legend(loc="upper left")

    panel_title(axc, "c", "B60 robustness hierarchy")
    y = np.arange(len(forest))[::-1]
    colors = [NAVY if s == "same-window" else ORANGE for s in forest.evidence_stage]
    colored_forest(axc, forest.estimate, y, forest.ci_lower, forest.ci_upper, colors, size=40)
    axc.axvline(0, color=MID, lw=0.9)
    axc.set_yticks(y, forest.analysis); axc.set_xlim(-0.02, 0.27); axc.set_xlabel("Interaction")
    percent_axis(axc, "x"); clean_axis(axc, grid="x")
    axc.axhspan(-0.5, 1.5, color=PALE_ORANGE, zorder=-2)
    axc.text(0.98, 0.02, "disjoint chronological", transform=axc.transAxes, ha="right", fontsize=8.0, color=ORANGE)

    panel_title(axd, "d", "Continuous interaction distribution")
    mids = (hist.bin_left + hist.bin_right) / 2
    axd.bar(mids, hist["count"], width=np.diff(edges)*0.94, color=SKY, edgecolor=WHITE)
    s = contsum.iloc[0]
    axd.axvline(s.median_continuous_interaction_ug_l, color=NAVY, lw=1.4, label=f"Median {s.median_continuous_interaction_ug_l:.2f}")
    axd.axvline(s.winsorized_1pct_mean_ug_l, color=ORANGE, lw=1.4, ls="--", label=f"1% winsorized mean {s.winsorized_1pct_mean_ug_l:.2f}")
    axd.set(xlabel="Continuous interaction (µg/L)", ylabel="Systems", xlim=(-5, 15))
    clean_axis(axd, grid="y"); axd.legend(loc="upper right", fontsize=8.0)
    axd.text(0.02, 0.88, f"{pct(s.fraction_positive)} > 0", transform=axd.transAxes, color=NAVY, fontweight="bold")

    panel_title(axe, "e", "Margin and observed recurrence")
    miss = sev.HAA5_panel_ceiling_miss.astype(bool)
    axe.scatter(sev.loc[~miss, "HAA9_margin_above_B60_ug_l"], sev.loc[~miss, "HAA9_positive_round_fraction"], s=12, alpha=0.35, color=NAVY, label="HAA5 full grid positive")
    axe.scatter(sev.loc[miss, "HAA9_margin_above_B60_ug_l"], sev.loc[miss, "HAA9_positive_round_fraction"], s=15, alpha=0.48, marker="^", color=ORANGE, label="HAA5 ceiling miss")
    xmax_e = float(np.nanpercentile(sev.HAA9_margin_above_B60_ug_l, 98))
    axe.set(xlabel="HAA9 margin above B60 (µg/L)", ylabel="Positive group fraction", xlim=(0, xmax_e * 1.38))
    percent_axis(axe, "y"); clean_axis(axe, grid="both")
    axe.text(xmax_e * 1.04, 0.83, "● HAA5 full-grid\n   positive", color=NAVY, fontsize=8.0, va="center")
    axe.text(xmax_e * 1.04, 0.48, "▲ HAA5 ceiling\n   miss", color=ORANGE, fontsize=8.0, va="center")

    caption = (
        "Figure 5. Threshold dependence, disjoint evaluation, and concentration-scale behavior. (a) The primary interaction is shown with pointwise and simultaneous 95% bootstrap intervals across 20-100 µg/L. "
        "(b) Two chronological directions condition on two groups and evaluate the other two. (c) Same-window stability checks and chronological transfer estimates are distinguished. "
        "(d) The continuous interaction distribution is truncated only for display, with robust summaries marked. (e) HAA9 recurrence is plotted against margin above the B60 benchmark, distinguishing systems missed by the full HAA5 observed grid."
    )
    alt = (
        "Five panels show that the interaction changes from negative at low thresholds to positive at higher thresholds. Two chronological curves follow the same pattern and are about 4.4 to 4.6 percentage points at 60. "
        "A forest plot shows roughly 21 percentage points for same-window checks but smaller disjoint chronological values. A histogram shows a right-skewed continuous interaction, and a scatter plot relates HAA9 margin to recurrence."
    )
    return FigureBundle(
        "Fig5", fig, caption, alt,
        [
            ("a", band, "wr_results/core/threshold_interaction_simultaneous_band.csv"),
            ("b", chrono, "wr_results/core/chronological_disjoint_threshold_curve.csv"),
            ("c", forest, "derived from bootstrap, fixed-support, and chronological tables"),
            ("d", hist, "derived from wr_results/core/continuous_interaction_systems.csv"),
            ("d", contsum, "wr_results/core/continuous_interaction_summary.csv"),
            ("e", sev, "wr_results/core/severity_persistence_systems.csv"),
        ],
    )


def fig6() -> FigureBundle:
    strict = load_csv("wr_results/validation/syr4_strict_site_and_overlap_sensitivity_metrics.csv", ["cohort", "metric", "estimate", "cluster_bootstrap_ci_low", "cluster_bootstrap_ci_high"])
    quarter = load_csv("wr_results/validation/syr4_quarter_capture_curves_extended.csv", ["cohort", "analyte", "n_available_quarters", "quarters_sampled_k", "mean_exact_capture_probability", "cluster_bootstrap_ci_low", "cluster_bootstrap_ci_high"])
    icr = load_csv("wr_results/validation/icr_temporal_support_comparison.csv", ["analysis_version", "scope", "analyte", "time_scale", "full_six_quarter_systems", "full_window_positive_systems"])
    ca = load_parquet("wr_results/california/ca_exact_haa5_tthm_pairs_2023_2025.parquet", ["haa5_value_midpoint_ug_l", "tthm_value_midpoint_ug_l", "haa5_any_quantified", "tthm_any_quantified", "pws_id"])
    casens = load_csv("wr_results/california/ca_paired_concordance_sensitivity.csv", ["analysis", "spearman_rho", "ci95_low", "ci95_high"])
    caagreement = load_csv("wr_results/california/ca_reference_threshold_agreement.csv", ["metric", "estimate", "ci95_low", "ci95_high"])

    strict_keep = strict[strict.cohort.isin(["same_date_all_pairs", "strict_site_primary"]) & strict.metric.isin(["spearman_rho_both_detected", "threshold_status_agreement"])].copy()
    qkeep = quarter[(quarter.n_available_quarters == 4) & quarter.cohort.isin(["all_valid_single_analyte_records", "paired_strict_site_sensitivity"])].copy()
    ikeep = icr[(icr.scope == "primary_distribution") & (icr.time_scale == "quarterly")].copy()
    ca_detected = ca[ca.haa5_any_quantified.astype(bool) & ca.tthm_any_quantified.astype(bool)].copy()
    require(len(ca_detected) == 25528 and ca_detected.pws_id.nunique() == 1407, "Unexpected California paired cohort size")
    ca_xmax = float(np.ceil(np.nanpercentile(ca_detected.haa5_value_midpoint_ug_l, 99.5) / 20) * 20)
    ca_ymax = float(np.ceil(np.nanpercentile(ca_detected.tthm_value_midpoint_ug_l, 99.5) / 20) * 20)
    ca_xmax = max(100, min(ca_xmax, 240)); ca_ymax = max(120, min(ca_ymax, 260))
    cadens = binned_density(ca_detected.haa5_value_midpoint_ug_l.to_numpy(), ca_detected.tthm_value_midpoint_ug_l.to_numpy(), xmax=ca_xmax, ymax=ca_ymax, bins=44)

    fig = plt.figure(figsize=(FULL_W, 160 * MM))
    gs = fig.add_gridspec(2, 2, left=0.13, right=0.90, bottom=0.10, top=0.945, hspace=0.52, wspace=0.50)
    axa, axb, axc, axd = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]

    panel_title(axa, "a", "Strict-site SYR4 linkage")
    metrics = ["spearman_rho_both_detected", "threshold_status_agreement"]
    names = ["Spearman ρ", "Status agreement"]
    y = np.array([1, 0])
    for yi, metric in zip(y, metrics):
        rows = strict_keep[strict_keep.metric == metric].set_index("cohort")
        x0 = rows.loc["same_date_all_pairs", "estimate"]; x1 = rows.loc["strict_site_primary", "estimate"]
        axa.plot([x0, x1], [yi, yi], color=LIGHT, lw=4)
        axa.scatter([x0], [yi], color=NAVY, s=46, label="All same-date links" if yi == 1 else None, zorder=3)
        axa.scatter([x1], [yi], color=ORANGE, marker="s", s=46, label="Strict site" if yi == 1 else None, zorder=3)
        axa.text(max(x0, x1) + 0.008, yi, f"Δ={x1-x0:+.5f}", va="center", fontsize=8.0)
    axa.set_yticks(y, names); axa.set_xlim(0.55, 0.97); axa.set_xlabel("Estimate")
    clean_axis(axa, grid="x"); axa.legend(loc="lower right")

    panel_title(axb, "b", "Full-record SYR4 quarter curves")
    styles = {
        ("all_valid_single_analyte_records", "HAA5"): (ORANGE, "o", "-", "HAA5, all valid"),
        ("all_valid_single_analyte_records", "TTHM"): (TEAL, "s", "-", "TTHM, all valid"),
        ("paired_strict_site_sensitivity", "HAA5"): (ORANGE, "o", "--", "HAA5, paired strict-site"),
        ("paired_strict_site_sensitivity", "TTHM"): (TEAL, "s", "--", "TTHM, paired strict-site"),
    }
    for (coh, analyte), (color, marker, ls, label) in styles.items():
        sub = qkeep[(qkeep.cohort == coh) & (qkeep.analyte == analyte)].sort_values("quarters_sampled_k")
        axb.plot(sub.quarters_sampled_k, sub.mean_exact_capture_probability, color=color, marker=marker, ls=ls, label=label)
        if coh == "all_valid_single_analyte_records":
            axb.fill_between(sub.quarters_sampled_k, sub.cluster_bootstrap_ci_low, sub.cluster_bootstrap_ci_high, color=color, alpha=0.10)
    axb.set(xlim=(0.8, 4.2), ylim=(0.3, 1.02), xticks=[1, 2, 3, 4], xlabel="Quarters sampled", ylabel="Conditional capture")
    percent_axis(axb, "y"); clean_axis(axb, grid="both"); axb.legend(loc="lower right", fontsize=8.0)

    panel_title(axc, "c", "ICR reporting-limit recovery")
    versions = ["original_minus999_excluded", "mrl_bounded_recovery"]
    x = np.arange(2); width = 0.34
    for j, analyte in enumerate(["HAA5", "HAA9"]):
        sub = ikeep[ikeep.analyte == analyte].set_index("analysis_version").loc[versions]
        vals = sub.full_six_quarter_systems.to_numpy()
        pos = sub.full_window_positive_systems.to_numpy()
        color = ORANGE if analyte == "HAA5" else TEAL
        bars = axc.bar(x + (j-0.5)*width, vals, width=width, color=color, hatch="//" if j == 0 else "", edgecolor=WHITE, label=analyte)
        for bar, p in zip(bars, pos):
            label = f"{int(bar.get_height())} total\n{int(p)} ref+"
            if bar.get_height() >= 40:
                axc.text(bar.get_x()+bar.get_width()/2, bar.get_height()-7, label, ha="center", va="top", fontsize=8.0, color=WHITE, fontweight="bold", linespacing=0.92)
            else:
                axc.text(bar.get_x()+bar.get_width()/2, bar.get_height()+3, label, ha="center", va="bottom", fontsize=8.0, color=INK, linespacing=0.92)
    axc.set_xticks(x, ["Exclude -999", "MRL-bounded"])
    axc.set_ylabel("Six-quarter systems"); axc.set_ylim(0, 200); clean_axis(axc, grid="y"); axc.legend(loc="upper right", ncol=2)

    panel_title(axd, "d", "California HAA5-TTHM context")
    sm = plot_density_rects(axd, cadens, cmap="PuBuGn")
    axd.axvline(60, color=ORANGE, ls=":", lw=1.0); axd.axhline(80, color=TEAL, ls=":", lw=1.0)
    axd.set(xlim=(0, ca_xmax), ylim=(0, ca_ymax), xlabel="HAA5 (µg/L)", ylabel="TTHM (µg/L)")
    clean_axis(axd, grid=None)
    rho = casens[casens.analysis.str.contains("both analytes quantified", case=False, na=False)].iloc[0]
    require(np.isclose(rho.spearman_rho, 0.602518073399094), "California concordance estimate changed unexpectedly")
    agreement = caagreement[caagreement.metric == "overall_agreement"].iloc[0]
    axd.text(0.04, 0.94, f"ρ = {rho.spearman_rho:.3f}\n95% CI {rho.ci95_low:.3f}-{rho.ci95_high:.3f}\n25,528 quantified pairs", transform=axd.transAxes, va="top", fontsize=8.0, fontweight="bold", color=NAVY, bbox=dict(facecolor=WHITE, edgecolor="none", alpha=0.82, pad=2.5))
    axd.text(0.96, 0.04, f"60/80 status agreement: {agreement.estimate:.1%}", transform=axd.transAxes, ha="right", va="bottom", fontsize=8.0, color=INK, bbox=dict(facecolor=WHITE, edgecolor="none", alpha=0.82, pad=2.0))
    cax = fig.add_axes([0.915, 0.10, 0.013, 0.335])
    cb = vector_colorbar(fig, sm, cax=cax); cb.set_label("log10(pairs + 1)")

    caption = (
        "Figure 6. Validation hierarchy across distinct public archives. (a) Requiring the same SYR4 sampling point changes rank correlation and threshold-status agreement negligibly. "
        "(b) Quarter capture curves rebuilt from all valid single-analyte records are compared with the strict paired subset. (c) ICR -999 totals whose component-specific reporting limits sum below the benchmark are classified only as definitely below, expanding six-quarter support without concentration imputation. "
        "(d) California routine distribution-system records from 2023-2025 show exact-time, jointly quantified HAA5-TTHM concordance as contextual validation."
    )
    alt = (
        "Four validation panels. A dumbbell plot shows virtually identical SYR4 correlation and agreement before and after strict-site linkage. Four-quarter capture curves reach 100% at four quarters for HAA5 and TTHM. "
        "ICR support rises from 151 to 167 systems for HAA5 and from 8 to 11 for HAA9 after reporting-limit-bounded negative recovery. A California density plot of 25,528 jointly quantified pairs reports Spearman rho 0.603."
    )
    return FigureBundle(
        "Fig6", fig, caption, alt,
        [
            ("a", strict_keep, "wr_results/validation/syr4_strict_site_and_overlap_sensitivity_metrics.csv"),
            ("b", qkeep, "wr_results/validation/syr4_quarter_capture_curves_extended.csv"),
            ("c", ikeep, "wr_results/validation/icr_temporal_support_comparison.csv"),
            ("d", cadens, "wr_results/california/ca_exact_haa5_tthm_pairs_2023_2025.parquet"),
            ("d", casens, "wr_results/california/ca_paired_concordance_sensitivity.csv"),
            ("d", caagreement, "wr_results/california/ca_reference_threshold_agreement.csv"),
        ],
    )


def fig7() -> FigureBundle:
    thresholds = load_csv("wr_results/core/threshold_factorial_with_intervals.csv", ["analysis", "threshold_ug_l", "n_reference_positive_systems", "p00", "p10", "p01"])
    t = thresholds[thresholds.cohort_definition == "threshold_specific_reference_positive"].copy()
    t["panel_gain"] = t.p10 - t.p00
    t["observation_gain"] = t.p01 - t.p00
    boot = load_csv("results/ucmr4/bootstrap_system_cluster_B60.csv", ["split", "metric", "estimate", "ci_lower_2_5pct", "ci_upper_97_5pct"])
    fixed = load_csv("wr_results/core/fixed_support_factorial.csv", ["analysis", "interaction", "interaction_ci_lower", "interaction_ci_upper"])
    cohort = load_csv("wr_results/core/cohort_operator_margin_sensitivity.csv", ["analysis", "interaction", "interaction_ci_lower", "interaction_ci_upper", "comparison_type"])
    chrono = load_csv("wr_results/core/chronological_disjoint_B60_summary.csv", ["direction", "interaction", "interaction_ci_lower", "interaction_ci_upper"])
    robust = []
    for split, label in [("all", "Primary replay"), ("development", "Development split"), ("holdout", "Hashed holdout")]:
        r = boot[(boot.split == split) & (boot.metric == "conditional_interaction")].iloc[0]
        robust.append((label, r.estimate, r.ci_lower_2_5pct, r.ci_upper_97_5pct, "same-window robustness"))
    for _, r in fixed.iloc[1:].iterrows():
        lab = {"source_aligned_minimum_rounds": "Source-aligned", "at_least_four_event_code_groups": "Four-group support", "four_groups_with_repeated_site": "Repeated-site support"}[r.analysis]
        robust.append((lab, r.interaction, r.interaction_ci_lower, r.interaction_ci_upper, "same-window robustness"))
    sens = cohort[cohort.analysis.isin(["exclude_2021_complete_records", "include_systems_with_any_available_complete_triplets"])]
    for _, r in sens.iterrows():
        lab = "Exclude 2021" if r.analysis.startswith("exclude") else "Include incomplete grids"
        robust.append((lab, r.interaction, r.interaction_ci_lower, r.interaction_ci_upper, "same-window robustness"))
    for _, r in chrono.iterrows():
        lab = "Late to early" if r.direction.startswith("late2") else "Early to late"
        robust.append((lab, r.interaction, r.interaction_ci_lower, r.interaction_ci_upper, "disjoint chronological transfer"))
    robust = pd.DataFrame(robust, columns=["analysis", "estimate", "ci_lower", "ci_upper", "evidence_stage"])

    fig = plt.figure(figsize=(FULL_W, 122 * MM))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.94, 1.06], left=0.09, right=0.975, bottom=0.19, top=0.90, wspace=0.58)
    axa, axb = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])

    panel_title(axa, "a", "Monitoring-priority plane")
    lim = max(float(t.panel_gain.max()), float(t.observation_gain.max())) * 1.10
    axa.fill_between([0, lim], [0, lim], [lim, lim], color=PALE_ORANGE, alpha=0.55, zorder=0)
    axa.fill_between([0, lim], [0, 0], [0, lim], color=PALE_TEAL, alpha=0.55, zorder=0)
    axa.plot([0, lim], [0, lim], color=MID, ls="--", lw=1.0)
    norm = mcolors.Normalize(vmin=t.threshold_ug_l.min(), vmax=t.threshold_ug_l.max())
    cm = mpl.colormaps["viridis"]
    sizes = 28 + 115 * (t.n_reference_positive_systems / t.n_reference_positive_systems.max())
    for i in range(len(t)-1):
        axa.plot(
            [t.panel_gain.iloc[i], t.panel_gain.iloc[i + 1]],
            [t.observation_gain.iloc[i], t.observation_gain.iloc[i + 1]],
            color=cm(norm(t.threshold_ug_l.iloc[i + 1])),
            lw=1.0,
            solid_capstyle="round",
            zorder=2,
        )
    sc = axa.scatter(t.panel_gain, t.observation_gain, c=t.threshold_ug_l, cmap=cm, norm=norm, s=sizes, edgecolor=WHITE, lw=0.7, zorder=3)
    for B in [20, 40, 60, 80, 100]:
        r = t.loc[t.threshold_ug_l == B]
        if not r.empty:
            rr = r.iloc[0]; axa.text(rr.panel_gain + 0.006, rr.observation_gain + 0.006, str(B), fontsize=8.0, color=INK)
    axa.text(0.54, 0.88, "Observation gain\n> panel gain", transform=axa.transAxes, color=ORANGE, fontweight="bold", fontsize=8.0, bbox=dict(facecolor=WHITE, edgecolor="none", alpha=0.78, pad=2))
    axa.text(0.54, 0.10, "Panel gain\n> observation gain", transform=axa.transAxes, color=TEAL, fontweight="bold", fontsize=8.0, bbox=dict(facecolor=WHITE, edgecolor="none", alpha=0.78, pad=2))
    axa.set(xlim=(0, lim), ylim=(0, lim), xlabel="Panel gain at one group", ylabel="Observation gain under HAA5")
    percent_axis(axa, "x"); percent_axis(axa, "y"); clean_axis(axa, grid="both")
    axa.text(0.98, 0.02, "Point labels: threshold (µg/L)\nPoint area: reference support", transform=axa.transAxes, ha="right", va="bottom", fontsize=8.0, color=MID)

    panel_title(axb, "b", "Robustness and inference boundary")
    y = np.arange(len(robust))[::-1]
    colors = [NAVY if s.startswith("same") else ORANGE for s in robust.evidence_stage]
    colored_forest(axb, robust.estimate, y, robust.ci_lower, robust.ci_upper, colors, size=38)
    axb.axvline(0, color=MID, lw=0.9)
    axb.set_yticks(y, robust.analysis); axb.set_xlim(-0.015, 0.27); axb.set_xlabel("B60 interaction")
    percent_axis(axb, "x"); clean_axis(axb, grid="x")
    axb.axhspan(-0.5, 1.5, color=PALE_ORANGE, zorder=-2)
    axb.text(0.98, 0.04, "Orange: disjoint rounds", transform=axb.transAxes, ha="right", va="bottom", fontsize=8.0, color=ORANGE)
    fig.text(0.53, 0.055, "Inference boundary: observed public-data signal capture only.\nNo causal, prevalence, compliance, or health-risk inference.", ha="center", va="center", fontsize=8.2, color=MID)

    caption = (
        "Figure 7. Monitoring-design interpretation and evidence boundary. (a) Threshold-specific panel gain and observation gain trace a data-derived priority plane; point size represents the number of reference-positive systems and color represents the analytical threshold. "
        "(b) Same-window robustness checks cluster near the primary B60 interaction, whereas disjoint chronological transfer is smaller but remains positive. The supported inference is monitoring signal capture within observed public-data frames, not causal, regulatory, prevalence, or health-risk inference."
    )
    alt = (
        "Two-panel discussion figure. A trajectory across thresholds compares gain from expanding the chemical panel with gain from expanding observed groups and sites; low thresholds favor observation expansion and the balance changes across the path. "
        "A forest plot shows same-window interaction estimates near 21 percentage points and disjoint chronological estimates near 4 to 5 percentage points, followed by an explicit inference boundary."
    )
    return FigureBundle("Fig7", fig, caption, alt, [("a", t, "wr_results/core/threshold_factorial_with_intervals.csv"), ("b", robust, "derived from primary robustness and chronological tables")])


def figs1() -> FigureBundle:
    sem = load_csv("wr_results/core/round_semantics_summary.csv", ["metric", "fraction", "count", "denominator"])
    rounds = load_csv("wr_results/core/round_definition_sensitivity.csv", ["analysis", "interaction", "interaction_ci_lower", "interaction_ci_upper", "total_units"])
    cohort = load_csv("wr_results/core/cohort_operator_margin_sensitivity.csv", ["analysis", "interaction", "interaction_ci_lower", "interaction_ci_upper", "threshold_ug_l", "comparison_type"])
    sem_keep = sem[sem.metric.isin(["multi_date", "span_gt_7_days", "span_gt_30_days", "multi_facility", "multi_site", "crosses_calendar_year"])].copy()
    sem_order = ["multi_date", "span_gt_7_days", "span_gt_30_days", "multi_facility", "multi_site", "crosses_calendar_year"]
    sem_keep["metric"] = pd.Categorical(sem_keep.metric, sem_order, ordered=True)
    sem_keep = sem_keep.sort_values("metric")
    local = cohort[cohort.comparison_type == "local_threshold_margin"].copy()
    cohort_keep = cohort[cohort.comparison_type == "cohort_or_date_scope"].copy()

    fig = plt.figure(figsize=(FULL_W, 158 * MM))
    gs = fig.add_gridspec(2, 2, left=0.13, right=0.975, bottom=0.11, top=0.945, hspace=0.55, wspace=0.50)
    axa, axb, axc, axd = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    panel_title(axa, "a", "Administrative-group QA")
    labels = ["Multi-date", ">7 d span", ">30 d span", "Multi-facility", "Multi-site", "Cross-year"]
    y = np.arange(len(sem_keep)); axa.barh(y, sem_keep.fraction, color=CB[:len(y)], edgecolor=WHITE)
    axa.set_yticks(y, labels); axa.invert_yaxis(); axa.set_xlabel("Fraction of groups"); percent_axis(axa, "x"); clean_axis(axa, grid="x")
    for yi, v in zip(y, sem_keep.fraction): axa.text(v+0.012, yi, pct(v), va="center", fontsize=8.0)

    panel_title(axb, "b", "Round-definition sensitivity")
    y = np.arange(len(rounds))[::-1]
    labels_b = ["Event code", "Event code + year", "Date", "Facility x event", "Physical sample"]
    colored_forest(axb, rounds.interaction, y, rounds.interaction_ci_lower, rounds.interaction_ci_upper, CB[:len(y)], size=42)
    axb.set_yticks(y, labels_b); axb.set_xlim(0.17, 0.40); axb.set_xlabel("B60 interaction"); percent_axis(axb, "x"); clean_axis(axb, grid="x")
    axb2 = axb.twiny(); axb2.set_xlim(axb.get_xlim()); axb2.set_xticks([]); axb2.spines[:].set_visible(False)
    for yi, units in zip(y, rounds.total_units): axb.text(0.395, yi, f"{int(units):,} units", ha="right", va="center", fontsize=8.0, color=MID, bbox=dict(facecolor=WHITE, edgecolor="none", alpha=0.86, pad=1.2))

    panel_title(axc, "c", "Cohort and date-scope checks")
    y = np.arange(len(cohort_keep))[::-1]
    labels_c = ["Primary", "Exclude 2021", "Incomplete grids"]
    colored_forest(axc, cohort_keep.interaction, y, cohort_keep.interaction_ci_lower, cohort_keep.interaction_ci_upper, [NAVY, ORANGE, TEAL], size=44)
    axc.set_yticks(y, labels_c); axc.set_xlabel("B60 interaction"); axc.set_xlim(0.15, 0.25); percent_axis(axc, "x"); clean_axis(axc, grid="x")

    panel_title(axd, "d", "Local benchmark sensitivity")
    axd.errorbar(local.threshold_ug_l, local.interaction, yerr=[local.interaction-local.interaction_ci_lower, local.interaction_ci_upper-local.interaction], color=MAGENTA, marker="o", capsize=3)
    axd.axvline(60, color=NAVY, ls=":", lw=1.0)
    axd.set(xlabel="Benchmark (µg/L)", ylabel="Interaction", xlim=(58.8, 61.2)); percent_axis(axd, "y"); clean_axis(axd, grid="both")

    caption = (
        "Figure S1. Round semantics and primary analysis QA. (a) Administrative event-code groups frequently contain multiple sites and sometimes multiple dates or facilities. "
        "(b) Alternative unit definitions change the resource interpretation and interaction. (c) Date-scope and incomplete-grid cohort checks leave the primary interaction nearly unchanged. "
        "(d) Strict greater-than comparisons from 59 to 61 µg/L show local benchmark stability."
    )
    alt = "Four panels show event-group quality checks, interaction estimates under five unit definitions, stable cohort sensitivity estimates, and a local 59 to 61 microgram-per-liter benchmark sensitivity curve."
    return FigureBundle("FigS1", fig, caption, alt, [("a", sem_keep, "wr_results/core/round_semantics_summary.csv"), ("b", rounds, "wr_results/core/round_definition_sensitivity.csv"), ("c-d", cohort, "wr_results/core/cohort_operator_margin_sensitivity.csv")])


def figs2() -> FigureBundle:
    support = load_csv("wr_results/core/support_systems.csv", ["facility_water_type_profile", "round_count", "stable_site_count", "source_aligned_support", "four_code_groups", "reference_positive_B60"])
    grid = load_csv("wr_results/core/joint_round_site_grid_audit.csv", ["rectangular_observed_grid", "missing_cells", "observed_rounds", "observed_sites", "reference_positive_B60"])
    fixed = load_csv("wr_results/core/fixed_support_factorial.csv", ["analysis", "interaction", "interaction_ci_lower", "interaction_ci_upper", "n_reference_positive_systems"])
    src_groups = support.assign(source=support.facility_water_type_profile.replace({"GW":"Groundwater", "SW":"Surface water", "MX":"Mixed", "MULTIPLE":"Multiple types", "GU":"Groundwater under influence"})).groupby(["source", "round_count"], observed=True).size().rename("systems").reset_index()
    stable_hist = support.groupby("stable_site_count").size().rename("systems").reset_index()
    missing = grid.groupby("missing_cells").size().rename("systems").reset_index().sort_values("missing_cells")

    fig = plt.figure(figsize=(FULL_W, 224 * MM))
    gs = fig.add_gridspec(3, 2, left=0.08, right=0.975, bottom=0.075, top=0.955, hspace=0.63, wspace=0.50)
    axa, axb, axc, axd, axe, axf = [fig.add_subplot(gs[i, j]) for i in range(3) for j in range(2)]
    panel_title(axa, "a", "Round support by source profile")
    source_order = [s for s in ["Groundwater", "Surface water", "Mixed", "Multiple types", "Groundwater under influence"] if s in src_groups.source.unique()]
    bottom = np.zeros(len(source_order))
    for rc, color, hatch in [(1, NAVY, ""), (2, TEAL, "//"), (3, ORANGE, ".."), (4, MAGENTA, "xx")]:
        vals = np.array([src_groups[(src_groups.source == s) & (src_groups.round_count == rc)].systems.sum() for s in source_order], dtype=float)
        denom = np.array([src_groups[src_groups.source == s].systems.sum() for s in source_order], dtype=float)
        frac = np.divide(vals, denom, out=np.zeros_like(vals), where=denom>0)
        axa.bar(np.arange(len(source_order)), frac, bottom=bottom, color=color, hatch=hatch, edgecolor=WHITE, label=f"{rc} group{'s' if rc>1 else ''}")
        bottom += frac
    source_labels = {
        "Groundwater": "Ground-\nwater",
        "Surface water": "Surface\nwater",
        "Mixed": "Mixed",
        "Multiple types": "Multiple\ntypes",
        "Groundwater under influence": "GW under\ninfluence",
    }
    axa.set_xticks(np.arange(len(source_order)), [source_labels[s] for s in source_order])
    axa.set_ylabel("Fraction of systems"); percent_axis(axa, "y"); clean_axis(axa, grid="y"); axa.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.24))

    panel_title(axb, "b", "Stable-site support")
    cut = stable_hist[stable_hist.stable_site_count <= 12]
    axb.bar(cut.stable_site_count, cut.systems, color=TEAL, edgecolor=WHITE)
    axb.set(xlabel="Sites represented in every group", ylabel="Systems", xticks=range(1, 13)); clean_axis(axb, grid="y")
    tail = int(stable_hist.loc[stable_hist.stable_site_count > 12, "systems"].sum())
    axb.text(0.98, 0.92, f">12 sites: {tail:,} systems", transform=axb.transAxes, ha="right", fontsize=8.0, color=MID)

    panel_title(axc, "c", "Observed-grid completeness")
    shown = missing[missing.missing_cells <= 10]
    axc.bar(shown.missing_cells, shown.systems, color=NAVY, edgecolor=WHITE)
    axc.set(xlabel="Missing site-by-group cells", ylabel="Systems", xticks=range(0, 11, 2)); clean_axis(axc, grid="y")
    axc.text(0.98, 0.92, f"{grid.rectangular_observed_grid.astype(bool).sum():,}/{len(grid):,} have no missing cells", transform=axc.transAxes, ha="right", fontsize=8.0, color=NAVY, fontweight="bold")

    panel_title(axd, "d", "Fixed-support sensitivity")
    labels = ["All complete grids", "Source-aligned", "≥4 groups", "4 groups +\nrepeated site"]
    y = np.arange(len(fixed))[::-1]
    colored_forest(axd, fixed.interaction, y, fixed.interaction_ci_lower, fixed.interaction_ci_upper, CB[:4], size=44)
    axd.set_yticks(y, labels); axd.set_xlabel("B60 interaction"); axd.set_xlim(0.16, 0.26); percent_axis(axd, "x"); clean_axis(axd, grid="x")

    # Retain the displaced main Fig7 evidence without changing its frozen data.
    threshold = load_csv("wr_results/playbook_extension/fixed_B60_and_changing_cohort_thresholds.csv")
    gains = threshold[threshold.cohort == "threshold_specific_HAA9_positive"].copy().sort_values("threshold_ug_l")
    gains["panel_gain"] = gains.p10 - gains.p00
    gains["observation_gain"] = gains.p01 - gains.p00
    panel_title(axe, "e", "")
    lim = max(float(gains.panel_gain.max()), float(gains.observation_gain.max())) * 1.08
    axe.plot([0, lim], [0, lim], color=MID, ls="--", lw=.9)
    axe.plot(gains.panel_gain, gains.observation_gain, color=INK, lw=1)
    axe.scatter(gains.panel_gain, gains.observation_gain, c=gains.threshold_ug_l, cmap="viridis", s=42+90*gains.systems/gains.systems.max(), edgecolor=WHITE, lw=.7)
    for benchmark, dx, dy in [(20,.006,-.020),(60,-.020,.012),(100,-.010,.014)]:
        row = gains[np.isclose(gains.threshold_ug_l, benchmark)].iloc[0]
        if benchmark == 60:
            axe.annotate("60",xy=(row.panel_gain,row.observation_gain),xytext=(.16,.31),ha="center",va="center",fontsize=8,arrowprops=dict(arrowstyle="-",connectionstyle="arc3,rad=0",color=MID,lw=.8,shrinkA=2,shrinkB=3))
        else:
            axe.text(row.panel_gain+dx, row.observation_gain+dy, str(benchmark), fontsize=8, ha="right" if dx<0 else "left", va="top" if dy<0 else "bottom")
    axe.set(xlim=(0,lim), ylim=(0,lim), xlabel="Panel gain at one round", ylabel="Observation gain under HAA5")
    percent_axis(axe,"x"); percent_axis(axe,"y"); clean_axis(axe,grid="both")
    axe.text(.96,.05,"Unadjusted gains\nnot a cost boundary",transform=axe.transAxes,ha="right",va="bottom",fontsize=8,color=MID)

    loso = load_csv("wr_results/core/leave_one_state_out.csv")
    ordered = loso.sort_values("difference_from_overall").copy()
    panel_title(axf, "f", "")
    colors = [ORANGE if v<0 else TEAL for v in ordered.difference_from_overall]
    axf.vlines(np.arange(len(ordered)),0,100*ordered.difference_from_overall,color=colors,lw=1.1)
    axf.scatter(np.arange(len(ordered)),100*ordered.difference_from_overall,c=colors,s=24,marker="D",edgecolor=WHITE,lw=.5)
    axf.axhline(0,color=MID,lw=.9)
    axf.set(xlabel="Excluded jurisdiction code (ordered)",ylabel="Interaction change (pp)")
    clean_axis(axf)
    for idx,text_x,dy in [(0,4,-.15),(len(ordered)-1,len(ordered)-5,.15)]:
        value = 100*float(ordered.difference_from_overall.iloc[idx])
        axf.annotate(ordered.excluded_state_code.iloc[idx],xy=(idx,value),xytext=(text_x,value+dy),ha="center",va="top" if dy<0 else "bottom",fontsize=8,arrowprops=dict(arrowstyle="-",connectionstyle="arc3,rad=0",color=MID,lw=.8,shrinkA=1,shrinkB=1))
    axf.margins(y=.22)

    caption = (
        "Figure S2. Observation support and complementary sensitivity checks. (a) Administrative-round counts by facility-water-type profile. (b) Sites recurring in every round. (c) Rectangular-grid frequency and missing cells. "
        "(d) B60 interaction in source-aligned, four-round and repeated-site cohorts. (e) Threshold-specific panel versus observation-support gains; point size denotes reference-positive support and colour the benchmark. The equality line is not a cost boundary. (f) Change in interaction after omitting each represented jurisdiction code."
    )
    alt = "Six letter-only panels show group-count composition, stable sites, missing cells, fixed-support interactions, the unadjusted panel-versus-observation gain plane, and leave-one-jurisdiction influence. The gain plane has no economic priority regions."
    return FigureBundle("FigS2", fig, caption, alt, [("a", src_groups, "wr_results/core/support_systems.csv"), ("b", stable_hist, "wr_results/core/support_systems.csv"), ("c", missing, "wr_results/core/joint_round_site_grid_audit.csv"), ("d", fixed, "wr_results/core/fixed_support_factorial.csv"), ("e", gains, "wr_results/playbook_extension/fixed_B60_and_changing_cohort_thresholds.csv"), ("f", ordered, "wr_results/core/leave_one_state_out.csv")])


def figs3() -> FigureBundle:
    full = load_csv("wr_results/core/threshold_factorial_with_intervals.csv", ["threshold_ug_l", "cohort_definition", "n_reference_positive_systems", "p00", "p10", "p01", "p11"])
    band = load_csv("wr_results/core/threshold_interaction_simultaneous_band.csv", ["threshold_ug_l", "interaction", "pointwise_ci_lower", "pointwise_ci_upper", "simultaneous_ci_lower", "simultaneous_ci_upper"])
    spec = full[full.cohort_definition == "threshold_specific_reference_positive"].copy()
    fixed = full[full.cohort_definition == "fixed_B60_reference_positive_systems"].copy()
    fig = plt.figure(figsize=(FULL_W, 156 * MM))
    gs = fig.add_gridspec(2, 2, left=0.08, right=0.975, bottom=0.11, top=0.945, hspace=0.52, wspace=0.46)
    axa, axb, axc, axd = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    panel_title(axa, "a", "All four conditional cells")
    for metric, color, label in [("p00", NAVY, "HAA5 / one"), ("p10", TEAL, "HAA9 / one"), ("p01", ORANGE, "HAA5 / full"), ("p11", MAGENTA, "HAA9 / full")]:
        axa.plot(spec.threshold_ug_l, spec[metric], color=color, marker="o", ms=2.8, label=label)
        lo = spec[f"{metric}_ci_lower"]; hi = spec[f"{metric}_ci_upper"]
        axa.fill_between(spec.threshold_ug_l, lo, hi, color=color, alpha=0.08)
    axa.set(xlabel="Threshold (µg/L)", ylabel="Conditional capture", xlim=(20, 100), ylim=(0, 1.03)); percent_axis(axa, "y"); clean_axis(axa, grid="both"); axa.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    panel_title(axb, "b", "Interaction uncertainty")
    axb.fill_between(band.threshold_ug_l, band.simultaneous_ci_lower, band.simultaneous_ci_upper, color=SKY, alpha=0.18, label="Simultaneous")
    axb.fill_between(band.threshold_ug_l, band.pointwise_ci_lower, band.pointwise_ci_upper, color=NAVY, alpha=0.18, label="Pointwise")
    axb.plot(band.threshold_ug_l, band.interaction, color=NAVY, marker="o", ms=3.0)
    axb.axhline(0, color=MID, lw=0.9); axb.set(xlabel="Threshold (µg/L)", ylabel="Interaction", xlim=(20, 100)); percent_axis(axb, "y"); clean_axis(axb, grid="both"); axb.legend()
    panel_title(axc, "c", "Threshold-specific support")
    axc.plot(spec.threshold_ug_l, spec.n_reference_positive_systems, color=MAGENTA, marker="o", ms=3.2)
    axc.fill_between(spec.threshold_ug_l, 0, spec.n_reference_positive_systems, color=PALE_MAGENTA, alpha=0.65)
    axc.set(xlabel="Threshold (µg/L)", ylabel="Reference-positive systems", xlim=(20, 100)); clean_axis(axc, grid="both")
    panel_title(axd, "d", "Cohort definition changes the curve")
    axd.plot(spec.threshold_ug_l, spec.p00, color=NAVY, marker="o", label="Threshold-specific cohort")
    axd.plot(fixed.threshold_ug_l, fixed.p00, color=ORANGE, marker="s", label="Fixed B60 cohort")
    axd.axvline(60, ymax=.74, color=MAGENTA, ls=":", lw=1.0)
    axd.set(xlabel="Threshold (µg/L)", ylabel="One-group HAA5 capture", xlim=(20, 100), ylim=(0, 1)); percent_axis(axd, "y"); clean_axis(axd, grid="both"); axd.legend(loc="upper right")
    caption = (
        "Figure S3. Complete benchmark-dependence audit. (a) Conditional factorial capture with pointwise 95% whole-PWS bootstrap intervals. (b) Additive interaction with pointwise intervals and a simultaneous 95% bootstrap band across benchmarks. "
        "(c) Threshold-specific reference-positive counts. (d) One-round HAA5 capture under threshold-specific and fixed-B60 estimands. Panels (a), (b) and (d) are summarized in main Fig. 7; the complete audit remains here."
    )
    alt = "Four line charts show all factorial cell probabilities, interaction uncertainty, declining reference support, and the difference between threshold-specific and fixed-B60 cohort estimates across 20 to 100 micrograms per liter."
    return FigureBundle("FigS3", fig, caption, alt, [("a,c,d", full, "wr_results/core/threshold_factorial_with_intervals.csv"), ("b", band, "wr_results/core/threshold_interaction_simultaneous_band.csv")])


def figs4() -> FigureBundle:
    screen = load_csv("wr_results/core/screening_curve.csv", ["haa5_action_threshold_ug_l", "sensitivity", "followup_fraction", "false_positive_rate", "positive_predictive_value"])
    folds = load_csv("wr_results/core/screening_crossfit_folds.csv", ["fold", "selected_HAA5_action_threshold_ug_l", "evaluation_sensitivity", "evaluation_followup_fraction", "evaluation_positive_predictive_value"])
    ops = load_csv("wr_results/core/screening_operating_points.csv", ["policy", "sensitivity", "sensitivity_ci_lower", "sensitivity_ci_upper", "followup_fraction", "followup_fraction_ci_lower", "followup_fraction_ci_upper", "positive_predictive_value"])
    fig = plt.figure(figsize=(FULL_W, 156 * MM))
    gs = fig.add_gridspec(2, 2, left=0.10, right=0.975, bottom=0.12, top=0.945, hspace=0.56, wspace=0.60)
    axa, axb, axc, axd = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    panel_title(axa, "a", "Screening frontier")
    axa.plot(screen.followup_fraction, screen.sensitivity, color=NAVY)
    axa.set(xlabel="Follow-up fraction", ylabel="Sensitivity", xlim=(0, 1), ylim=(0, 1.02)); percent_axis(axa, "x"); percent_axis(axa, "y"); clean_axis(axa, grid="both")
    panel_title(axb, "b", "Precision frontier")
    axb.plot(screen.followup_fraction, screen.positive_predictive_value, color=TEAL)
    axb.set(xlabel="Follow-up fraction", ylabel="Positive predictive value", xlim=(0, 1), ylim=(0, 1.02)); percent_axis(axb, "x"); percent_axis(axb, "y"); clean_axis(axb, grid="both")
    panel_title(axc, "c", "Fold-selected triggers")
    y = np.arange(len(folds)); axc.hlines(y, 45, folds.selected_HAA5_action_threshold_ug_l, color=LIGHT, lw=3)
    axc.scatter(folds.selected_HAA5_action_threshold_ug_l, y, c=CB[:len(folds)], s=48, edgecolor=WHITE)
    axc.set_yticks(y, [f"Fold {int(i)+1}" for i in folds.fold]); axc.set_xlim(45, 49); axc.set_xlabel("Selected HAA5 trigger (µg/L)"); clean_axis(axc, grid="x")
    panel_title(axd, "d", "Fold evaluation")
    fold_offsets = {0: (8, 8), 1: (8, 8), 2: (8, -14), 3: (8, 8), 4: (8, 8)}
    for r in folds.itertuples(index=False):
        axd.scatter(r.evaluation_followup_fraction, r.evaluation_sensitivity, s=45, color=CB[int(r.fold)], edgecolor=WHITE)
        axd.annotate(str(int(r.fold)+1), (r.evaluation_followup_fraction, r.evaluation_sensitivity), xytext=fold_offsets[int(r.fold)], textcoords="offset points", fontsize=8.0)
    pooled = ops[ops.policy == "cross_fitted_HAA5_trigger"].iloc[0]
    axd.errorbar([pooled.followup_fraction], [pooled.sensitivity], xerr=[[pooled.followup_fraction-pooled.followup_fraction_ci_lower], [pooled.followup_fraction_ci_upper-pooled.followup_fraction]], yerr=[[pooled.sensitivity-pooled.sensitivity_ci_lower], [pooled.sensitivity_ci_upper-pooled.sensitivity]], fmt="o", color=NAVY, capsize=3, label="Pooled")
    axd.set(xlabel="Follow-up fraction", ylabel="Sensitivity", xlim=(0.055, 0.105), ylim=(0.36, 0.49)); percent_axis(axd, "x"); percent_axis(axd, "y"); clean_axis(axd, grid="both"); axd.legend()
    caption = (
        "Figure S4. Screening calibration details. (a) The complete HAA5-trigger sensitivity-workload curve and (b) positive-predictive-value curve are shown. "
        "(c) Five training folds selected closely grouped HAA5 trigger values. (d) Evaluation-fold workload and sensitivity are shown with the pooled cross-fitted operating point and bootstrap intervals."
    )
    alt = "Four panels show full screening frontiers, five selected HAA5 thresholds between 46.75 and 47.75 micrograms per liter, and evaluation folds centered near 42% sensitivity at 7.7% follow-up workload."
    return FigureBundle("FigS4", fig, caption, alt, [("a-b", screen, "wr_results/core/screening_curve.csv"), ("c-d", folds, "wr_results/core/screening_crossfit_folds.csv"), ("d", ops, "wr_results/core/screening_operating_points.csv")])


def figs5() -> FigureBundle:
    scenario = load_csv("wr_results/core/resource_scenario_surface.csv", ["budget_HAA5_sample_equivalents", "incremental_HAA9_reporting_cost_per_sample", "HAA9_minus_HAA5_capture", "preferred_design"])
    breakeven = load_csv("wr_results/core/resource_break_even.csv", ["budget_HAA5_sample_equivalents", "max_incremental_HAA9_cost_ratio"])
    units = load_csv("wr_results/core/resource_unit_systems.csv", ["expected_samples_in_one_selected_group", "expected_dates_in_one_selected_group", "full_grid_physical_samples", "full_grid_collection_dates"])
    budgets = sorted(scenario.budget_HAA5_sample_equivalents.unique())
    costs = sorted(scenario.incremental_HAA9_reporting_cost_per_sample.unique())
    pref = scenario.pivot(index="incremental_HAA9_reporting_cost_per_sample", columns="budget_HAA5_sample_equivalents", values="preferred_design").reindex(index=costs, columns=budgets)
    gain = scenario.pivot(index="incremental_HAA9_reporting_cost_per_sample", columns="budget_HAA5_sample_equivalents", values="HAA9_minus_HAA5_capture").reindex(index=costs, columns=budgets)
    pref_num = (pref == "HAA9_reporting").astype(float).to_numpy()
    fig = plt.figure(figsize=(FULL_W, 158 * MM))
    gs = fig.add_gridspec(2, 2, left=0.08, right=0.91, bottom=0.11, top=0.945, hspace=0.55, wspace=0.58)
    axa, axb, axc, axd = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    panel_title(axa, "a", "Highest capture under stated scenario")
    rectangle_heatmap(axa, pref_num, [str(int(x)) for x in budgets], [f"{x:.1f}" for x in costs], cmap=mpl.colors.ListedColormap([PALE_ORANGE, PALE_TEAL]), vmin=0, vmax=1)
    axa.set(xlabel="Budget (HAA5 sample equivalents)", ylabel="Incremental HAA9 reporting cost ratio")
    axa.set_yticks(np.arange(0, len(costs), 4)+0.5, [f"{costs[i]:.1f}" for i in range(0, len(costs), 4)])
    axa.legend(handles=[Rectangle((0,0),1,1,fc=PALE_TEAL,ec=TEAL,label="Higher HAA9 capture"), Rectangle((0,0),1,1,fc=PALE_ORANGE,ec=ORANGE,label="Higher HAA5 capture")], loc="upper right", fontsize=8.0)
    panel_title(axb, "b", "Scenario capture difference")
    sm = rectangle_heatmap(axb, gain.to_numpy(), [str(int(x)) for x in budgets], [f"{x:.1f}" for x in costs], cmap=mpl.colors.LinearSegmentedColormap.from_list("div", [ORANGE, WHITE, TEAL]), vmin=-max(abs(gain.min().min()), abs(gain.max().max())), vmax=max(abs(gain.min().min()), abs(gain.max().max())))
    axb.set(xlabel="Budget (HAA5 sample equivalents)", ylabel="Incremental HAA9 reporting cost ratio")
    axb.set_yticks(np.arange(0, len(costs), 4)+0.5, [f"{costs[i]:.1f}" for i in range(0, len(costs), 4)])
    vector_colorbar(fig, sm, ax=axb, fraction=0.046, pad=0.02, label="HAA9 - HAA5 capture")
    panel_title(axc, "c", "Exact integer scenario frontier")
    axc.plot(breakeven.budget_HAA5_sample_equivalents, breakeven.max_incremental_HAA9_cost_ratio, color=NAVY, marker="o", drawstyle="steps-mid")
    axc.fill_between(breakeven.budget_HAA5_sample_equivalents, 0, breakeven.max_incremental_HAA9_cost_ratio, color=PALE_TEAL, alpha=0.65, step="mid")
    axc.set(xlabel="Budget (HAA5 sample equivalents)", ylabel="Maximum incremental cost ratio", xticks=budgets); clean_axis(axc, grid="both")
    axc.text(0.02, 0.93, "Dimensionless scenario only;\nnot a dollar cost-effectiveness analysis", transform=axc.transAxes, va="top", fontsize=8.0, color=MID, bbox=dict(facecolor=WHITE, edgecolor="none", alpha=0.78, pad=2))
    panel_title(axd, "d", "Physical-unit audit")
    vals = [units.expected_samples_in_one_selected_group, units.full_grid_physical_samples, units.expected_dates_in_one_selected_group, units.full_grid_collection_dates]
    positions = [0,1,2.3,3.3]
    bp = axd.boxplot(vals, positions=positions, widths=0.55, showfliers=False, patch_artist=True, medianprops=dict(color=INK, lw=1.2), whiskerprops=dict(color=MID), capprops=dict(color=MID))
    for patch, color, hatch in zip(bp["boxes"], [ORANGE, NAVY, ORANGE, NAVY], ["//", "", "//", ""]): patch.set(facecolor=color, edgecolor=WHITE, hatch=hatch, alpha=0.85)
    axd.set_xticks(positions, ["One group\nsamples", "Full grid\nsamples", "One group\ndates", "Full grid\ndates"])
    axd.set_ylabel("Physical resource units"); clean_axis(axd, grid="y")
    caption = (
        "Figure S5. Resource scenarios and physical-unit audit. (a) The design with the higher modeled capture and (b) the HAA9-minus-HAA5 capture difference vary across integer budgets and assumed incremental HAA9 reporting costs. "
        "(c) The stepped exact-integer frontier summarizes only the stated scenario surface. No policy preference, universal smooth break-even function, empirical price, or dollar cost-effectiveness inference is supported. (d) Physical samples and collection dates contained in one administrative group and the full observed grid are shown."
    )
    alt = "Four panels show the higher-capture design under stated scenarios, a capture-difference heatmap, a stepped exact-integer scenario frontier, and boxplots of physical samples and dates represented by one group and a full grid."
    return FigureBundle("FigS5", fig, caption, alt, [("a-b", scenario, "wr_results/core/resource_scenario_surface.csv"), ("c", breakeven, "wr_results/core/resource_break_even.csv"), ("d", units, "wr_results/core/resource_unit_systems.csv")])


def figs6() -> FigureBundle:
    chrono = load_csv("wr_results/core/chronological_disjoint_threshold_curve.csv", ["direction", "threshold_ug_l", "p00", "p10", "p01", "p11", "interaction", "interaction_ci_lower", "interaction_ci_upper"])
    altfold = load_csv("wr_results/core/disjoint_grid_summary.csv", ["analysis", "interaction", "interaction_ci_lower", "interaction_ci_upper"])
    fig = plt.figure(figsize=(FULL_W, 148 * MM), layout="constrained")
    gs = fig.add_gridspec(2, 2, hspace=0.24, wspace=0.24)
    axa, axb, axc, axd = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    cells = [("p00", NAVY, "HAA5 / one"), ("p10", TEAL, "HAA9 / one"), ("p01", ORANGE, "HAA5 / two"), ("p11", MAGENTA, "HAA9 / two")]
    for ax, direction, letter, title in [(axa, "late2_conditions_early2_evaluates", "a", "Condition late; evaluate early"), (axb, "early2_conditions_late2_evaluates", "b", "Condition early; evaluate late")]:
        panel_title(ax, letter, title)
        sub = chrono[chrono.direction == direction]
        for metric, color, label in cells: ax.plot(sub.threshold_ug_l, sub[metric], color=color, marker="o", ms=2.7, label=label)
        ax.set(xlabel="Conditioning threshold (µg/L)", ylabel="Evaluation capture", xlim=(20,100), ylim=(0,1.02)); percent_axis(ax, "y"); clean_axis(ax, grid="both"); ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=8.0)
    panel_title(axc, "c", "Chronological interaction by threshold")
    for direction, color, label in [("late2_conditions_early2_evaluates", ORANGE, "Late to early"), ("early2_conditions_late2_evaluates", TEAL, "Early to late")]:
        sub = chrono[chrono.direction == direction]
        axc.plot(sub.threshold_ug_l, sub.interaction, color=color, marker="o", ms=2.8, label=label)
        axc.fill_between(sub.threshold_ug_l, sub.interaction_ci_lower, sub.interaction_ci_upper, color=color, alpha=0.12)
    axc.axhline(0, color=MID, lw=0.9); axc.set(xlabel="Threshold (µg/L)", ylabel="Interaction", xlim=(20,100)); percent_axis(axc, "y"); clean_axis(axc, grid="both"); axc.legend()
    panel_title(axd, "d", "Alternating-fold sensitivity")
    y = np.arange(len(altfold))[::-1]
    labels = ["Pooled directions", "Reference fold 0", "Reference fold 1"]
    colored_forest(axd, altfold.interaction, y, altfold.interaction_ci_lower, altfold.interaction_ci_upper, [NAVY, ORANGE, TEAL], size=44)
    axd.axvline(0, color=MID, lw=0.9); axd.set_yticks(y, labels); axd.set_xlabel("B60 interaction"); percent_axis(axd, "x"); clean_axis(axd, grid="x")
    caption = (
        "Figure S6. Disjoint-round evaluation details. (a) Two late groups define reference positivity and two early groups evaluate the four cells; (b) the direction is reversed. "
        "(c) Chronological interactions and bootstrap intervals are shown across thresholds. (d) Alternating observed-group folds provide a separate split-round sensitivity analysis."
    )
    alt_text = "Four panels show full chronological cell curves in each direction, two threshold-dependent interaction curves, and three positive alternating-fold interaction estimates near five percentage points at B60."
    return FigureBundle("FigS6", fig, caption, alt_text, [("a-c", chrono, "wr_results/core/chronological_disjoint_threshold_curve.csv"), ("d", altfold, "wr_results/core/disjoint_grid_summary.csv")])


def figs7() -> FigureBundle:
    detect = load_csv("results/additional/species_coverage_censoring.csv", ["abbreviation", "detected_event_fraction", "mrl_ug_L"])
    contrib = load_csv("results/additional/species_group_contributions.csv", ["target_group", "abbreviation", "weighting", "primary_detect_only_share_official_denominator", "primary_cluster_bootstrap_ci_low", "primary_cluster_bootstrap_ci_high"])
    tox = load_csv("wr_results/toxicity_sensitivity/weighted_composition_summary.csv", ["coefficient_source", "nondetect_substitution", "system_equal_mean_additional_four_weighted_fraction", "cluster_bootstrap_ci_lower", "cluster_bootstrap_ci_upper", "event_mean_mass_fraction_additional_four"])
    coeff = load_csv("wr_results/toxicity_sensitivity/coefficient_set_comparison.csv", ["metric", "spearman_rank_correlation", "median_source_1", "median_source_2"])
    extra = contrib[(contrib.target_group == "Extra4") & (contrib.weighting == "equal_system_mass_weighted")].copy()
    fig = plt.figure(figsize=(FULL_W, 166 * MM))
    gs = fig.add_gridspec(2, 2, left=0.08, right=0.975, bottom=0.15, top=0.945, hspace=0.60, wspace=0.54)
    axa, axb, axc, axd = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    panel_title(axa, "a", "Species detection and MRLs")
    d = detect.sort_values("detected_event_fraction")
    y = np.arange(len(d)); axa.barh(y, d.detected_event_fraction, color=[ORANGE if x in ["MCAA","DCAA","TCAA","MBAA","DBAA"] else TEAL for x in d.abbreviation], edgecolor=WHITE)
    axa.set_yticks(y, d.abbreviation); axa.set_xlim(0, 1.18); axa.set_xlabel("Detected event fraction"); percent_axis(axa, "x"); clean_axis(axa, grid="x")
    for yi, v, mrl in zip(y, d.detected_event_fraction, d.mrl_ug_L): axa.text(v+0.018, yi, f"MRL {mrl:g}", va="center", fontsize=8.0, color=MID)
    panel_title(axb, "b", "Additional-four mass shares")
    e = extra.sort_values("primary_detect_only_share_official_denominator")
    y = np.arange(len(e)); x=e.primary_detect_only_share_official_denominator.to_numpy(); lo=e.primary_cluster_bootstrap_ci_low.to_numpy(); hi=e.primary_cluster_bootstrap_ci_high.to_numpy()
    axb.errorbar(x, y, xerr=[x-lo,hi-x], fmt="none", ecolor=TEAL, capsize=3); axb.scatter(x,y,color=TEAL,s=45,edgecolor=WHITE)
    axb.set_yticks(y,e.abbreviation); axb.set_xlabel("Detected-only equal-system mass share"); percent_axis(axb,"x"); clean_axis(axb,grid="x")
    panel_title(axc, "c", "Censor and coefficient sensitivity")
    sources = list(tox.coefficient_source.unique()); subs=["lower","mid","upper"]; x=np.arange(len(subs)); width=0.34
    for j, src in enumerate(sources):
        s=tox[tox.coefficient_source==src].set_index("nondetect_substitution").loc[subs]
        label="R support table" if "R_supporting" in src else "Dimension table"
        axc.bar(x+(j-.5)*width,s.system_equal_mean_additional_four_weighted_fraction,width=width,color=NAVY if j==0 else ORANGE,hatch="//" if j==1 else "",edgecolor=WHITE,label=label)
    axc.axhline(float(tox.event_mean_mass_fraction_additional_four.iloc[0]),color=TEAL,ls="--",lw=1.2,label="Mass fraction")
    axc.set_xticks(x,["Lower","Midpoint","Upper"]); axc.set_ylabel("Additional-four weighted fraction"); percent_axis(axc,"y"); clean_axis(axc,grid="y"); axc.legend(fontsize=8.0)
    panel_title(axd, "d", "Coefficient-set rank stability")
    y=np.arange(len(coeff))[::-1]; axd.barh(y,coeff.spearman_rank_correlation,color=[NAVY,TEAL],edgecolor=WHITE)
    labels=["HAA9 weighted\ncomposition","Additional-four\nweighted fraction"]
    axd.set_yticks(y,labels); axd.set_xlim(0,1); axd.set_xlabel("Event-level Spearman rank correlation"); clean_axis(axd,grid="x")
    for yi,v in zip(y,coeff.spearman_rank_correlation): axd.text(v-0.02,yi,f"{v:.3f}",ha="right",va="center",color=WHITE,fontweight="bold")
    fig.text(0.53,0.045,"In-vitro CHO-cell cytotoxicity-weighted composition index; not a health-risk metric",ha="center",fontsize=8.2,color=MAGENTA,fontweight="bold")
    caption = (
        "Figure S7. Species-level censoring and composition sensitivity. (a) Detection fractions and minimum reporting limits vary across nine haloacetic acids. (b) Detected-only equal-system mass shares partition the four species added in HAA9. "
        "(c) The additional-four fraction of an in-vitro CHO-cell cytotoxicity-weighted composition index varies by nondetect substitution and public coefficient source; the mass fraction is shown for context. (d) Event ranks are more stable than index magnitudes across coefficient sets. The index is compositional and is not a health-risk estimate."
    )
    alt = "Four panels show species detection fractions, mass shares of the four additional acids, weighted-composition fractions under two coefficient sources and three censoring substitutions, and high but imperfect rank correlations between coefficient sets. An explicit note states that the index is not a health-risk metric."
    return FigureBundle("FigS7", fig, caption, alt, [("a",detect,"results/additional/species_coverage_censoring.csv"),("b",extra,"results/additional/species_group_contributions.csv"),("c",tox,"wr_results/toxicity_sensitivity/weighted_composition_summary.csv"),("d",coeff,"wr_results/toxicity_sensitivity/coefficient_set_comparison.csv")])


def figs8() -> FigureBundle:
    overlap=load_csv("wr_results/validation/ucmr4_syr4_overlap_tolerance_summary.csv",["tolerance_ug_l","one_to_one_matches","matched_systems","fraction_of_same_date_ucmr_events"])
    strict=load_csv("wr_results/validation/syr4_strict_site_and_overlap_sensitivity_metrics.csv",["cohort","metric","estimate","n_pairs","n_systems"])
    recovery=load_csv("wr_results/validation/icr_minus999_recovery_audit.csv",["scope","analyte","documented_component_mrl_sum_ug_l","reported_minus999_events","recovered_definitely_below_events"])
    caqa=load_csv("wr_results/california/ca_data_quality_audit.csv",["section","metric","value"])
    strict_keep=strict[strict.cohort.isin(["same_date_all_pairs","strict_site_primary"]) & strict.metric.isin(["spearman_rho_both_detected","threshold_status_agreement"])].copy()
    ca_metrics=["event-analyte keys with multiple source rows","event-analyte keys with discordant values","exact PWS-site-date-time HAA5/TTHM pairs","date-only paired keys"]
    ca_keep=caqa[caqa.metric.isin(ca_metrics)].copy()
    fig=plt.figure(figsize=(FULL_W,156*MM))
    gs=fig.add_gridspec(2,2,left=.08,right=.975,bottom=.11,top=.945,hspace=.55,wspace=.48); axa,axb,axc,axd=[fig.add_subplot(gs[i,j]) for i in range(2) for j in range(2)]
    panel_title(axa,"a","Archive-overlap tolerance")
    axa.plot(overlap.tolerance_ug_l,overlap.fraction_of_same_date_ucmr_events,color=NAVY,marker="o")
    axa.axvline(.05,color=ORANGE,ls=":",lw=1.1); axa.set(xlabel="HAA5 tolerance (µg/L)",ylabel="Same-date UCMR4 events matched"); percent_axis(axa,"y"); clean_axis(axa,grid="both")
    axa.text(.12,.20,"0.05 µg/L audit",transform=axa.transAxes,ha="left",fontsize=8.0,color=ORANGE)
    panel_title(axb,"b","Strict-site linkage")
    metrics=["spearman_rho_both_detected","threshold_status_agreement"]; labels=["Spearman ρ","Status agreement"]
    for yi,(m,l) in enumerate(zip(metrics,labels)):
        s=strict_keep[strict_keep.metric==m].set_index("cohort"); x0=s.loc["same_date_all_pairs","estimate"];x1=s.loc["strict_site_primary","estimate"]
        axb.plot([x0,x1],[yi,yi],color=LIGHT,lw=4);axb.scatter(x0,yi,color=NAVY,s=42);axb.scatter(x1,yi,color=ORANGE,marker="s",s=42);axb.text(max(x0,x1)+.006,yi,f"Δ {x1-x0:+.5f}",va="center",fontsize=8.0)
    axb.set_yticks([0,1],labels);axb.set_xlim(.55,1.04);axb.set_xlabel("Estimate");clean_axis(axb,grid="x")
    panel_title(axc,"c","Reporting-limit-bounded ICR recovery")
    r=recovery[recovery.scope=="primary_distribution"].set_index("analyte").loc[["HAA5","HAA9"]];x=np.arange(2)
    bars=axc.bar(x,r.recovered_definitely_below_events,color=[ORANGE,TEAL],hatch=["//",""] if False else None,edgecolor=WHITE)
    axc.set_xticks(x,["HAA5","HAA9"]);axc.set_ylabel("Recovered definitely-below events");clean_axis(axc,grid="y")
    for bar,mrl in zip(bars,r.documented_component_mrl_sum_ug_l): axc.text(bar.get_x()+bar.get_width()/2,bar.get_height()+25,f"MRL sum {mrl:g}",ha="center",fontsize=8.0)
    axc.text(.98,.90,"No concentration\nimputation",transform=axc.transAxes,ha="right",va="top",color=MAGENTA,fontweight="bold",fontsize=8.0,bbox=dict(facecolor=WHITE,edgecolor="none",alpha=.80,pad=2))
    panel_title(axd,"d","California record-linkage QA")
    display=ca_keep.copy();display["label"]=display.metric.map({ca_metrics[0]:"Duplicate keys",ca_metrics[1]:"Discordant keys",ca_metrics[2]:"Exact-time pairs",ca_metrics[3]:"Date-only pairs"})
    y=np.arange(len(display));baseline=200.0
    for yi,v,color,marker in zip(y,display.value,[ORANGE,MAGENTA,TEAL,NAVY],["o","s","^","D"]):
        axd.hlines(yi,baseline,v,color=color,lw=3.0)
        axd.scatter(v,yi,color=color,marker=marker,s=54,edgecolor=WHITE,linewidth=.8,zorder=3)
    axd.set_yticks(y,display.label);axd.invert_yaxis();axd.set_xlabel("Count (log scale)");axd.set_xscale("log");axd.set_xlim(baseline,80000);clean_axis(axd,grid="x")
    axd.set_xticks([300,1000,3000,10000,30000],["300","1k","3k","10k","30k"]);axd.xaxis.set_minor_locator(mpl.ticker.NullLocator());axd.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    for yi,v in zip(y,display.value): axd.text(v*1.08,yi,f"{int(v):,}",va="center",fontsize=8.0)
    caption=("Figure S8. Cross-archive and record-linkage QA. (a) Candidate UCMR4-SYR4 overlap increases with the allowed HAA5 concentration tolerance and is treated as a sensitivity audit, not proof of duplicate origin. "
             "(b) Same-date and strict-site SYR4 linkage produce nearly identical metrics. (c) ICR -999 totals are recovered only when every named component is -999 and the documented component reporting-limit sum is below the analytical benchmark. "
             "(d) California duplicate collapse and exact-time versus date-only pairing counts are shown.")
    alt="Four QA panels show tolerance-dependent candidate archive overlap, negligible strict-site linkage changes, reporting-limit-bounded recovery of ICR below-benchmark events, and California duplicate and pairing counts on a log scale."
    return FigureBundle("FigS8",fig,caption,alt,[("a",overlap,"wr_results/validation/ucmr4_syr4_overlap_tolerance_summary.csv"),("b",strict_keep,"wr_results/validation/syr4_strict_site_and_overlap_sensitivity_metrics.csv"),("c",recovery,"wr_results/validation/icr_minus999_recovery_audit.csv"),("d",display,"wr_results/california/ca_data_quality_audit.csv")])


# ---------------------------------------------------------------------------
# Water Research revision-playbook figures (version 2)
# ---------------------------------------------------------------------------


def _mm_figure(height_mm: float, *, rows: int = 1, cols: int = 1, **kwargs):
    """Return a full-journal-width figure and axes with generous outer margins."""
    return plt.subplots(
        rows,
        cols,
        figsize=(FULL_W, height_mm * MM),
        layout="constrained",
        **kwargs,
    )


def _errorbar_vertical(
    ax: mpl.axes.Axes,
    x: Sequence[float],
    estimate: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
    *,
    colors: Sequence[str],
    markers: Sequence[str] | None = None,
) -> None:
    markers = markers or ["o"] * len(x)
    for xi, value, lo, hi, color, marker in zip(x, estimate, lower, upper, colors, markers):
        ax.errorbar(
            xi,
            value,
            yerr=[[value - lo], [hi - value]],
            fmt=marker,
            ms=5.4,
            color=color,
            mfc=color,
            mec=WHITE,
            mew=0.8,
            capsize=3,
            lw=1.25,
            zorder=4,
        )


def _short_policy(value: str) -> str:
    return {
        "random_event_code_group": "Random group",
        "warm_season_May_to_September_else_random": "Warm season",
        "first_chronological_event_code_group": "First group",
        "best_observed_HAA9_round_oracle": "Outcome oracle",
        "highest_TOC_midpoint": "Highest TOC",
        "highest_TOC_lower_bound": "Highest TOC, low",
        "highest_TOC_upper_bound": "Highest TOC, high",
        "random_event_code_group_TOC_complete_case": "Random, TOC subset",
    }.get(value, value.replace("_", " "))


def _four_cell_frame(row: pd.Series, *, label: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "analysis": label,
            "panel_scope": ["HAA5", "HAA9", "HAA5", "HAA9"],
            "observation_scope": ["one selected group", "one selected group", "full observed grid", "full observed grid"],
            "conditional_capture": [row.p00, row.p10, row.p01, row.p11],
            "ci_lower": [row.p00_ci_lower, row.p10_ci_lower, row.p01_ci_lower, row.p11_ci_lower],
            "ci_upper": [row.p00_ci_upper, row.p10_ci_upper, row.p01_ci_upper, row.p11_ci_upper],
            "systems": int(row.systems),
        }
    )


def wr2_ga() -> FigureBundle:
    metric = load_csv(
        "wr_results/playbook_extension/interaction_scale_metrics.csv",
        ["analysis", "systems", "p00", "p10", "p01", "p11", "additive_interaction"],
    )
    primary = metric.loc[metric.analysis == "primary_complete_observed_grid_B60"].iloc[0]
    equivalence = load_csv(
        "wr_results/playbook_extension/physical_sample_equivalence_curve.csv",
        ["HAA9_samples_selected", "HAA9_conditional_capture", "full_HAA5_record_capture"],
    )
    equivalence_systems = load_csv(
        "wr_results/playbook_extension/physical_sample_equivalence_systems.csv",
        ["available_samples"],
    )
    saturated_below_four = int((equivalence_systems.available_samples < 4).sum())
    require(len(equivalence_systems) == 769 and saturated_below_four == 3, "Unexpected physical-sample support below four samples.")
    equivalence["systems_saturating_below_four_samples"] = saturated_below_four
    equivalence["prespecified_comparison_margin"] = False
    ga_cells = _four_cell_frame(primary, label="primary complete observed grid")
    ga_scope = pd.DataFrame(
        {
            "panel": ["HAA5", "HAA9"],
            "species": [5, 9],
            "additional_species": [0, 4],
            "benchmark_ug_l": [60.0, 60.0],
        }
    )

    fig = plt.figure(figsize=(GA_W, GA_W / 2.5), layout="constrained")
    gs = fig.add_gridspec(1, 3, width_ratios=[0.86, 1.03, 1.11], wspace=0.10)
    axa, axb, axc = [fig.add_subplot(gs[0, i]) for i in range(3)]

    # Scope is shown with nested droplets, not a flowchart.
    panel_title(axa, "a", "Panel scope", pad=5)
    axa.set_xlim(0, 1); axa.set_ylim(0, 1); axa.axis("off")
    draw_drop(axa, (0.29, 0.62), 0.15, PALE_TEAL)
    draw_drop(axa, (0.72, 0.62), 0.18, TEAL)
    axa.text(0.29, 0.59, "HAA5", ha="center", va="center", fontsize=13.0, fontweight="bold", color=NAVY, zorder=6)
    axa.text(0.72, 0.59, "HAA9", ha="center", va="center", fontsize=13.0, fontweight="bold", color=WHITE, zorder=6)
    axa.text(0.23, 0.30, "5\nspecies", ha="center", va="center", fontsize=11.5, fontweight="bold", color=NAVY, linespacing=.88)
    axa.text(0.78, 0.30, "9\nspecies", ha="center", va="center", fontsize=11.5, fontweight="bold", color=TEAL, linespacing=.88)
    axa.text(0.50, 0.08, "same 60 µg/L analytical benchmark", ha="center", fontsize=11.5, color=INK)

    panel_title(axb, "b", "Scope x observation", pad=5)
    matrix = np.array([[primary.p00, primary.p10], [primary.p01, primary.p11]])
    rectangle_heatmap(
        axb,
        matrix,
        ["HAA5", "HAA9"],
        ["One group", "Full grid"],
        cmap=mpl.colors.LinearSegmentedColormap.from_list("ga_capture", [WHITE, PALE_TEAL, TEAL, NAVY]),
        vmin=0,
        vmax=1,
        annotations={(0, 0), (0, 1), (1, 0), (1, 1)},
    )
    axb.set_xlabel("Panel", fontsize=12.5); axb.set_ylabel("Observation", fontsize=12.5)
    axb.tick_params(labelsize=12.5)

    panel_title(axc, "c", "Capture by sample count", pad=5)
    axc.fill_between(
        equivalence.HAA9_samples_selected,
        equivalence.HAA9_capture_ci_lower,
        equivalence.HAA9_capture_ci_upper,
        color=PALE_TEAL,
        lw=0,
    )
    axc.plot(equivalence.HAA9_samples_selected, equivalence.HAA9_conditional_capture, color=TEAL, marker="o", markevery=2, lw=2.1, label="HAA9")
    benchmark = float(equivalence.full_HAA5_record_capture.iloc[0])
    axc.axhline(benchmark, color=NAVY, lw=2.0, ls="--", label="Full-record HAA5")
    match = equivalence.iloc[(equivalence.HAA9_samples_selected - 4).abs().argsort()[:1]].iloc[0]
    axc.scatter([4], [match.HAA9_conditional_capture], s=72, color=ORANGE, marker="D", edgecolor=WHITE, zorder=6)
    axc.text(4.5, match.HAA9_conditional_capture + 0.055, "Up to 4 HAA9 samples\nmean Δ = +0.14 pp", fontsize=12.5, color=ORANGE, fontweight="bold", linespacing=.92,bbox=dict(facecolor=WHITE,edgecolor="none",alpha=.90,pad=1.5))
    axc.set(xlim=(0, equivalence.HAA9_samples_selected.max()), ylim=(0, 1.02), xlabel="Physical samples selected", ylabel="Conditional capture")
    axc.tick_params(labelsize=12.5); axc.xaxis.label.set_size(12.5); axc.yaxis.label.set_size(12.5)
    percent_axis(axc); clean_axis(axc)
    axc.legend(loc="lower right", fontsize=11.5, handlelength=2.4)

    caption = (
        "Graphical abstract. (a) HAA5 is nested within HAA9 and both panels are evaluated at the same 60 µg/L analytical benchmark. "
        "(b) Detection changes jointly with panel scope and observation scope among 769 HAA9-benchmark-positive systems. "
        "(c) With a sampling budget of up to four randomly selected HAA9 physical samples, the mean point estimate is near-identical to the complete observed HAA5 record; three of 769 systems saturate below four samples and no comparison margin was prespecified."
    )
    alt = "Nested HAA5 and HAA9 droplets lead to a four-cell capture matrix and sample-count curve; up to four HAA9 samples and the full observed HAA5 record have near-identical mean point estimates without a prespecified comparison margin."
    return FigureBundle(
        "Graphical_Abstract",
        fig,
        caption,
        alt,
        [
            ("a", ga_scope, "study panel definitions"),
            ("b", ga_cells, "wr_results/playbook_extension/interaction_scale_metrics.csv"),
            ("c", equivalence, "wr_results/playbook_extension/physical_sample_equivalence_curve.csv"),
        ],
    )


def wr2_fig1() -> FigureBundle:
    pair = load_parquet(
        "data/processed/ucmr4/paired_haa_events.parquet",
        ["PWSID", "SampleEventCode", "site_id", "CollectionDate", "HAA9", "finite_grid_system_eligible"],
    )
    sem = load_csv("wr_results/core/round_semantics_summary.csv", ["metric", "count", "denominator", "fraction"])
    audit = load_csv(
        "wr_results/core/joint_round_site_grid_audit.csv",
        ["PWSID", "observed_rounds", "observed_sites", "missing_cells", "rectangular_observed_grid", "reference_positive_B60"],
    )
    metrics = load_csv("wr_results/playbook_extension/interaction_scale_metrics.csv", ["analysis", "systems", "p00", "p10", "p01", "p11"])
    primary = metrics.loc[metrics.analysis == "primary_complete_observed_grid_B60"].iloc[0]

    archive = pd.DataFrame(
        {
            "evidence_tier": ["Primary replay", "External replication", "Historical temporal check", "Current state context"],
            "archive": ["UCMR4", "SYR4", "ICR", "California"],
            "exchangeable_with_primary": [False, False, False, False],
        }
    )
    sem_order = ["multi_date", "multi_facility", "multi_site", "crosses_calendar_year"]
    sem_keep = sem.loc[sem.metric.isin(sem_order)].copy()
    sem_keep["metric"] = pd.Categorical(sem_keep.metric, sem_order, ordered=True)
    sem_keep = sem_keep.sort_values("metric")

    candidates = audit[(~audit.rectangular_observed_grid.astype(bool)) & audit.reference_positive_B60.astype(bool)]
    require(not candidates.empty, "No irregular reference-positive grid is available for Figure 1.")
    selected = candidates.sort_values(["missing_cells", "observed_sites"], ascending=[False, False]).iloc[0]
    ex = pair[(pair.PWSID.astype(str) == str(selected.PWSID)) & pair.finite_grid_system_eligible.astype(bool)].copy()
    ex.CollectionDate = pd.to_datetime(ex.CollectionDate)
    round_order = ex.groupby("SampleEventCode").CollectionDate.median().sort_values().index.tolist()
    site_order = sorted(ex.site_id.astype(str).unique())
    pivot = ex.pivot_table(index="SampleEventCode", columns="site_id", values="HAA9", aggfunc="max").reindex(index=round_order, columns=site_order)
    ex_source = pivot.rename_axis(index="group", columns="site").reset_index().melt(id_vars="group", var_name="site", value_name="HAA9_ug_l")
    cells = _four_cell_frame(primary, label="primary complete observed grid")

    fig = plt.figure(figsize=(FULL_W, 146 * MM), layout="constrained")
    gs = fig.add_gridspec(2, 2, height_ratios=[0.77, 1.23], wspace=0.12, hspace=0.14)
    axa, axb, axc, axd = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]

    panel_title(axa, "a", "Evidence tiers")
    axa.axis("off"); axa.set_xlim(0, 1); axa.set_ylim(0, 1)
    xs = np.linspace(0.12, 0.88, 4)
    markers = ["o", "s", "D", "^"]
    for x, row, marker, color in zip(xs, archive.itertuples(), markers, [NAVY, TEAL, ORANGE, MAGENTA]):
        axa.scatter(x, 0.62, s=310, marker=marker, color=color, edgecolor=WHITE, lw=1.0)
        axa.text(x, 0.33, row.archive, ha="center", fontweight="bold")
        axa.text(x, 0.16, row.evidence_tier.replace(" ", "\n", 1), ha="center", va="center", fontsize=7.5, color=MID)
    axa.text(0.50, 0.92, "Archives remain separate", ha="center", color=MAGENTA, fontweight="bold")

    panel_title(axb, "b", "Administrative-group semantics")
    labels = ["Multiple dates", "Multiple facilities", "Multiple sites", "Cross-year"]
    y = np.arange(4)
    bars = axb.barh(y, sem_keep.fraction, color=[TEAL, ORANGE, NAVY, MAGENTA], hatch=["//", "xx", "", ".."], edgecolor=WHITE)
    axb.set_yticks(y, labels); axb.invert_yaxis(); axb.set_xlim(0, 1.05); percent_axis(axb, "x"); clean_axis(axb, grid="x")
    axb.set_xlabel("Fraction of event-code groups")
    for bar, value in zip(bars, sem_keep.fraction):
        axb.text(min(value + 0.02, 0.98), bar.get_y() + bar.get_height() / 2, pct(value), va="center", ha="left" if value < .85 else "right", color=WHITE if value >= .85 else INK, fontweight="bold")

    panel_title(axc, "c", "Irregular observed grid")
    matrix = pivot.to_numpy(float)
    cmap = mpl.colors.LinearSegmentedColormap.from_list("grid_haa", [WHITE, SKY, NAVY])
    sm = rectangle_heatmap(axc, matrix, [f"S{i+1}" for i in range(matrix.shape[1])], [f"G{i+1}" for i in range(matrix.shape[0])], cmap=cmap, vmin=0, vmax=max(60, float(np.nanpercentile(matrix, 95))))
    for iy, ix in np.argwhere(np.isfinite(matrix) & (matrix > 60)):
        axc.scatter(ix + 0.5, iy + 0.5, marker="*", s=50, color=MAGENTA, edgecolor=WHITE, lw=.5)
    axc.set_xlabel("Observed site"); axc.set_ylabel("Administrative group")
    cbar = vector_colorbar(fig, sm, ax=axc, fraction=.046, pad=.03)
    cbar.set_label("HAA9 (µg/L)")
    axc.text(.02, .98, "star: >60 µg/L", transform=axc.transAxes, va="top", color=MAGENTA, fontweight="bold")

    panel_title(axd, "d", "Finite-grid replay")
    m = np.array([[primary.p00, primary.p10], [primary.p01, primary.p11]])
    rectangle_heatmap(axd, m, ["HAA5", "HAA9"], ["One group", "Full grid"], cmap=mpl.colors.LinearSegmentedColormap.from_list("capture_v2", [WHITE, PALE_TEAL, TEAL, NAVY]), vmin=0, vmax=1, annotations={(0,0),(0,1),(1,0),(1,1)})
    axd.set_xlabel("Panel scope"); axd.set_ylabel("Observation scope")
    axd.add_patch(Rectangle((1.04, 1.04), .92, .92, fill=False, edgecolor=MAGENTA, lw=2))

    caption = (
        "Figure 1. Study design and evidence boundaries. (a) Four of the five source archives are displayed with non-exchangeable roles; SYR3 is reported separately. (b) UCMR4 administrative event-code groups can span dates, facilities, and sites. "
        "(c) Finite-grid replay retains the observed site-by-group geometry and missing cells. (d) The prespecified 2 x 2 contrast changes panel and observation scope within 769 HAA9-benchmark-positive systems."
    )
    alt = "Four panels distinguish evidence tiers, quantify event-code grouping semantics, show one irregular observed sampling grid, and define the panel-by-observation factorial replay."
    return FigureBundle("Fig1", fig, caption, alt, [("a", archive, "archive-specific analysis protocols"), ("b", sem_keep, "wr_results/core/round_semantics_summary.csv"), ("c", ex_source, "data/processed/ucmr4/paired_haa_events.parquet"), ("d", cells, "wr_results/playbook_extension/interaction_scale_metrics.csv")])


def wr2_fig2() -> FigureBundle:
    pair = load_parquet("data/processed/ucmr4/paired_haa_events.parquet", ["HAA5", "HAA9", "extra4_ug_l", "finite_grid_system_eligible"])
    endpoint = load_csv("wr_results/playbook_extension/regulation_shaped_endpoint_summary.csv", ["endpoint", "panel", "supported_systems", "capture_among_HAA9_positive_systems"])
    eligible = pair[pair.finite_grid_system_eligible.astype(bool)].copy()
    density = binned_density(eligible.HAA5.to_numpy(), eligible.HAA9.to_numpy(), xmax=180, ymax=240, bins=46)

    scope = pd.DataFrame({"species_group": ["HAA5 core", "Additional four"], "species_count": [5, 4], "relationship": ["nested in HAA9", "HAA9 minus HAA5"]})
    ladder = endpoint[endpoint.endpoint == "single_sample_maximum"].copy()
    ladder["panel"] = pd.Categorical(ladder.panel, ["HAA5", "HAA5_plus_BCAA", "HAA6Br", "HAA9"], ordered=True)
    ladder = ladder.sort_values("panel")
    extra = eligible.loc[eligible.HAA9 > 0, ["HAA5", "HAA9", "extra4_ug_l"]].copy()
    require(len(extra) == 61193, f"Expected 61,193 HAA9-positive eligible events, found {len(extra):,}.")
    extra["extra4_fraction_of_HAA9"] = extra.extra4_ug_l / extra.HAA9
    quant = extra.extra4_fraction_of_HAA9.quantile([.1,.25,.5,.75,.9]).rename_axis("quantile").reset_index(name="fraction")
    quant["HAA9_positive_events"] = len(extra)

    fig, axes = _mm_figure(146, rows=2, cols=2)
    axa, axb, axc, axd = axes.ravel()
    panel_title(axa, "a", "Nested analyte scope")
    axa.axis("off"); axa.set_xlim(0,1); axa.set_ylim(0,1)
    for angle in np.linspace(0, 2*np.pi, 9, endpoint=False):
        x=.50+.29*np.cos(angle); y=.51+.29*np.sin(angle)
        core = angle < (5/9)*2*np.pi
        axa.scatter(x,y,s=250,marker="o" if core else "s",color=NAVY if core else ORANGE,edgecolor=WHITE,lw=1)
    axa.text(.50,.51,"HAA9",ha="center",va="center",fontsize=15,fontweight="bold")
    axa.text(.50,.13,"5 core species + 4 additional species",ha="center",color=MID)
    axa.legend(handles=[Line2D([0],[0],marker="o",color="none",markerfacecolor=NAVY,label="HAA5 core",markersize=8),Line2D([0],[0],marker="s",color="none",markerfacecolor=ORANGE,label="Additional four",markersize=8)],loc="upper center",ncol=2)

    panel_title(axb, "b", "Event-level concentration geometry")
    sm = plot_density_rects(axb, density, cmap="Blues")
    axb.plot([0,180],[0,180],ls="--",color=MID,lw=1.0,label="HAA5 = HAA9")
    axb.axvline(60,color=NAVY,ls=":",lw=1.4); axb.axhline(60,color=MAGENTA,ls="--",lw=1.4)
    axb.fill_between([0,60],60,240,color=PALE_ORANGE,alpha=.55,zorder=-1)
    axb.text(5,220,"HAA5-invisible\nHAA9 signal",color=MAGENTA,fontweight="bold")
    axb.set(xlim=(0,180),ylim=(0,240),xlabel="HAA5 (µg/L)",ylabel="HAA9 (µg/L)")
    clean_axis(axb)
    cbar=vector_colorbar(fig,sm,ax=axb,fraction=.046,pad=.03); cbar.set_label("log10(event count + 1)")

    panel_title(axc, "c", "Research endpoint ladder")
    x=np.arange(len(ladder)); colors=[NAVY,TEAL,ORANGE,MAGENTA]; hatches=["//","xx","..",""]
    bars=axc.bar(x,ladder.capture_among_HAA9_positive_systems,color=colors,hatch=hatches,edgecolor=WHITE)
    axc.set_xticks(x,["HAA5","HAA5 +\nBCAA","HAA6Br","HAA9"]); axc.set_ylim(0,1.08); axc.set_ylabel("Capture among HAA9-positive systems"); percent_axis(axc); clean_axis(axc)
    for bar,v in zip(bars,ladder.capture_among_HAA9_positive_systems): axc.text(bar.get_x()+bar.get_width()/2,v+.025,pct(v),ha="center",fontweight="bold")
    axc.text(.02,.97,"full-record individual-sample maximum",transform=axc.transAxes,va="top",color=MID)

    panel_title(axd, "d", "Additional-species contribution")
    axd.plot(quant["quantile"], quant.fraction, color=ORANGE, marker="D", lw=1.8)
    axd.fill_between(quant["quantile"], 0, quant.fraction, color=PALE_ORANGE, alpha=.65)
    axd.axhline(.5,color=MID,ls="--",lw=1)
    axd.set(xlabel="Event quantile",ylabel="Additional-four fraction of HAA9",ylim=(0,max(.72,float(quant.fraction.max()+.08))))
    percent_axis(axd,"x"); percent_axis(axd,"y"); clean_axis(axd)
    axd.text(.98,.95,f"{len(extra):,} HAA9-positive events\nzero-HAA9 events excluded",transform=axd.transAxes,ha="right",va="top",color=PURPLE,fontweight="bold")

    caption=("Figure 2. Chemical scope and endpoint context. (a) HAA5 is a five-species subset of HAA9. (b) Event concentrations identify the region where HAA9 exceeds 60 µg/L while HAA5 does not. "
             "(c) A full-record individual-sample maximum compares candidate panels; it is a research endpoint and not an HAA9 regulatory limit. (d) Among 61,193 HAA9-positive eligible events, the additional four species contribute a variable fraction of HAA9; zero-HAA9 events are excluded from the ratio.")
    alt="Four panels show nested panel chemistry, binned HAA5-versus-HAA9 event concentrations, a full-record individual-sample maximum panel comparison, and additional-four fraction quantiles among HAA9-positive events."
    return FigureBundle("Fig2",fig,caption,alt,[("a",scope,"study panel definitions"),("b",density,"data/processed/ucmr4/paired_haa_events.parquet"),("c",ladder,"wr_results/playbook_extension/regulation_shaped_endpoint_summary.csv"),("d",quant,"data/processed/ucmr4/paired_haa_events.parquet")])


def wr2_fig3() -> FigureBundle:
    metric=load_csv("wr_results/playbook_extension/interaction_scale_metrics.csv",["analysis","systems","p00","p10","p01","p11","additive_interaction","multiplicative_interaction_ratio","relative_excess_risk_due_to_interaction","synergy_index"])
    primary=metric.loc[metric.analysis=="primary_complete_observed_grid_B60"].iloc[0]
    mechanism=load_parquet("wr_results/playbook_extension/mechanism_systems.parquet",["cat_already_detected","cat_panel_only","cat_design_only","cat_either_change","cat_both_required","haa5_invisible"])
    wild=load_csv("wr_results/playbook_extension/interaction_wild_sign_flip_tests.csv",["analysis","observed_additive_interaction","systems","replicates","two_sided_p_value"])

    cells=_four_cell_frame(primary,label="primary complete observed grid")
    scale=pd.DataFrame({"measure":["Additive interaction","Multiplicative ratio","RERI","Synergy index"],"estimate":[primary.additive_interaction,primary.multiplicative_interaction_ratio,primary.relative_excess_risk_due_to_interaction,primary.synergy_index],"ci_lower":[primary.additive_interaction_ci_lower,primary.multiplicative_interaction_ratio_ci_lower,primary.relative_excess_risk_due_to_interaction_ci_lower,primary.synergy_index_ci_lower],"ci_upper":[primary.additive_interaction_ci_upper,primary.multiplicative_interaction_ratio_ci_upper,primary.relative_excess_risk_due_to_interaction_ci_upper,primary.synergy_index_ci_upper],"null":[0,1,0,1]})
    cats=["cat_already_detected","cat_panel_only","cat_design_only","cat_either_change","cat_both_required"]
    repair=pd.DataFrame({"class":["Already detected","Panel only","Design only","Either alone suffices","Both required"],"system_equivalents":[float(mechanism[c].sum()) for c in cats]})
    repair["fraction"]=repair.system_equivalents/len(mechanism)
    infer=metric.merge(wild[["analysis","two_sided_p_value","replicates"]],on="analysis",how="inner")
    infer["label"]=infer.analysis.map({"primary_complete_observed_grid_B60":"Primary replay","late2_condition_early2_evaluation":"Late → early transfer","early2_condition_late2_evaluation":"Early → late transfer"})

    fig,axes=_mm_figure(148,rows=2,cols=2);axa,axb,axc,axd=axes.ravel()
    panel_title(axa,"a","Conditional capture")
    x=np.arange(4); vals=cells.conditional_capture.to_numpy(); los=cells.ci_lower.to_numpy(); his=cells.ci_upper.to_numpy()
    _errorbar_vertical(axa,x,vals,los,his,colors=[NAVY,TEAL,ORANGE,MAGENTA],markers=["o","s","^","D"])
    axa.plot(x,vals,color=MID,lw=.8,zorder=1)
    axa.set_xticks(x,["HAA5\none group","HAA9\none group","HAA5\nfull grid","HAA9\nfull grid"]);axa.set_ylim(0,1.08);axa.set_ylabel("Conditional capture");percent_axis(axa);clean_axis(axa)
    for xi,v in zip(x,vals): axa.text(xi,v+.035,pct(v),ha="center",fontweight="bold")

    panel_title(axb,"b","Interaction across scales")
    y=np.arange(len(scale)); est=scale.estimate.to_numpy(); lo=scale.ci_lower.to_numpy(); hi=scale.ci_upper.to_numpy()
    colored_forest(axb,est,y,lo,hi,[TEAL,NAVY,ORANGE,PURPLE],size=46)
    for yi,nul in zip(y,scale.null): axb.scatter(nul,yi,marker="|",s=180,color=MID,zorder=1)
    axb.set_yticks(y,scale.measure);axb.invert_yaxis();axb.set_xlabel("Estimate; vertical tick is scale-specific null");clean_axis(axb,grid="x")
    axb.set_xlim(min(-.05,float(lo.min()-.08)),max(1.62,float(hi.max()+.08)))

    panel_title(axc,"c","What repairs missed systems")
    y=np.arange(5);bars=axc.barh(y,repair.fraction,color=[NAVY,TEAL,ORANGE,PURPLE,MAGENTA],hatch=["//","xx","..","++",""] ,edgecolor=WHITE)
    axc.set_yticks(y,repair["class"]);axc.invert_yaxis();axc.set_xlim(0,max(.42,float(repair.fraction.max()+.08)));axc.set_xlabel("Equal-PWS mean class probability");percent_axis(axc,"x");clean_axis(axc,grid="x")
    for bar,v in zip(bars,repair.fraction):axc.text(v+.012,bar.get_y()+bar.get_height()/2,pct(v),va="center",fontweight="bold")

    panel_title(axd,"d","Temporal transfer and inference")
    y=np.arange(len(infer));colored_forest(axd,infer.additive_interaction,y,infer.additive_interaction_ci_lower,infer.additive_interaction_ci_upper,[NAVY,TEAL,ORANGE],size=48)
    axd.axvline(0,color=MID,ls="--",lw=1);axd.set_yticks(y,infer.label);axd.invert_yaxis();axd.set_xlabel("Additive interaction");clean_axis(axd,grid="x");axd.set_xlim(-.01,.27)
    for yi,row in enumerate(infer.itertuples()):
        axd.text(row.additive_interaction_ci_upper+.007,yi,f"p={row.two_sided_p_value:.2g}",va="center",fontsize=7.2)
    axd.text(.98,.04,"100,000 cluster sign flips\nsecondary assumption-dependent test",transform=axd.transAxes,ha="right",va="bottom",fontsize=7.2,color=MID)

    caption=("Figure 3. The joint loss of panel and observation scope. (a) Conditional capture rises from 20.9% with one HAA5 group to 100% for the complete HAA9 grid. (b) Interaction is reported on four prespecified scales with scale-specific nulls. "
             "(c) Five exhaustive, mutually exclusive repair classes are equal-PWS averages over enumerated group choices; the additive interaction equals the both-required probability minus the either-alone-suffices probability. (d) Additive interaction remains positive in both disjoint chronological transfer directions; sign-flip tests are secondary and depend on system-level sign exchangeability.")
    alt="Four panels display factorial capture with intervals, four scale-dependent interaction measures, mutually exclusive repair classes, and temporal-transfer estimates with secondary sign-flip tests."
    return FigureBundle("Fig3",fig,caption,alt,[("a",cells,"wr_results/playbook_extension/interaction_scale_metrics.csv"),("b",scale,"wr_results/playbook_extension/interaction_scale_metrics.csv"),("c",repair,"wr_results/playbook_extension/mechanism_systems.parquet"),("d",infer,"wr_results/playbook_extension/interaction_scale_metrics.csv; interaction_wild_sign_flip_tests.csv")])


def wr2_fig4() -> FigureBundle:
    curve=load_csv("wr_results/core/resource_capture_curve.csv",["panel","selected_physical_samples","conditional_capture","ci_lower","ci_upper"])
    eq=load_csv("wr_results/playbook_extension/physical_sample_equivalence_curve.csv",["HAA9_samples_selected","HAA9_conditional_capture","HAA9_capture_ci_lower","HAA9_capture_ci_upper","full_HAA5_record_capture","paired_difference_vs_full_HAA5","paired_difference_ci_lower","paired_difference_ci_upper"])
    eq_systems=load_csv("wr_results/playbook_extension/physical_sample_equivalence_systems.csv",["available_samples"])
    saturated_below_four=int((eq_systems.available_samples<4).sum())
    require(len(eq_systems)==769 and saturated_below_four==3,"Unexpected physical-sample support below four samples in Figure 4.")
    frontier=load_csv("wr_results/playbook_extension/cost_ratio_frontier.csv",["HAA9_to_HAA5_per_sample_cost_ratio","equal_expected_budget_HAA9_capture","full_HAA5_record_capture","difference"])
    surface=load_csv("wr_results/core/joint_round_site_capture_surface.csv")
    require({"panel","requested_rounds","requested_sites","conditional_capture"}.issubset(surface.columns),"Joint round-site surface schema is incompatible with Figure 4.")
    selected=curve[curve.selected_physical_samples<=24].copy()
    eqplot=eq[eq.HAA9_samples_selected<=16].copy();eqplot["systems_saturating_below_four_samples"]=saturated_below_four;eqplot["prespecified_comparison_margin"]=False
    equivalence_summary=json.loads((PROJECT/"wr_results/playbook_extension/equivalence_summary.json").read_text(encoding="utf-8"))
    critical_ratio=float(equivalence_summary["critical_HAA9_to_HAA5_cost_ratio"])
    critical_lower=float(equivalence_summary["critical_cost_ratio_ci_lower"])
    critical_upper=float(equivalence_summary["critical_cost_ratio_ci_upper"])
    critical_capture=float(np.interp(critical_ratio,frontier.HAA9_to_HAA5_per_sample_cost_ratio,frontier.equal_expected_budget_HAA9_capture))
    frontier_source=frontier.assign(critical_cost_ratio=critical_ratio,critical_cost_ratio_ci_lower=critical_lower,critical_cost_ratio_ci_upper=critical_upper)

    fig,axes=_mm_figure(150,rows=2,cols=2);axa,axb,axc,axd=axes.ravel()
    panel_title(axa,"a","Random physical-sample capture")
    styles={"HAA5":(NAVY,"o","--"),"HAA9":(TEAL,"s","-")}
    for name,g in selected.groupby("panel"):
        color,marker,ls=styles.get(name,(MID,"D","-"));g=g.sort_values("selected_physical_samples")
        axa.fill_between(g.selected_physical_samples,g.ci_lower,g.ci_upper,color=color,alpha=.12,lw=0)
        axa.plot(g.selected_physical_samples,g.conditional_capture,color=color,marker=marker,markevery=3,ls=ls,label=name)
    axa.set(xlabel="Physical samples selected",ylabel="Conditional capture",ylim=(0,1.03));percent_axis(axa);clean_axis(axa);axa.legend()

    panel_title(axb,"b","Full-record HAA5 comparison")
    axb.fill_between(eqplot.HAA9_samples_selected,eqplot.HAA9_capture_ci_lower,eqplot.HAA9_capture_ci_upper,color=PALE_TEAL)
    axb.plot(eqplot.HAA9_samples_selected,eqplot.HAA9_conditional_capture,color=TEAL,marker="s",markevery=2,label="Selected HAA9")
    benchmark=float(eqplot.full_HAA5_record_capture.iloc[0]);axb.axhline(benchmark,color=NAVY,ls="--",label="Full observed HAA5")
    row4=eqplot.loc[eqplot.HAA9_samples_selected==4].iloc[0];axb.errorbar(4,row4.HAA9_conditional_capture,yerr=[[row4.HAA9_conditional_capture-row4.HAA9_capture_ci_lower],[row4.HAA9_capture_ci_upper-row4.HAA9_conditional_capture]],fmt="D",color=ORANGE,capsize=3,zorder=5)
    axb.text(4.5,row4.HAA9_conditional_capture+.045,f"Up to 4 HAA9: Δ {100*row4.paired_difference_vs_full_HAA5:+.2f} pp",color=ORANGE,fontweight="bold",bbox=dict(facecolor=WHITE,edgecolor="none",alpha=.90,pad=1.2))
    axb.set(xlabel="HAA9 samples selected",ylabel="Conditional capture",ylim=(0,1.03));percent_axis(axb);clean_axis(axb);axb.legend(loc="lower right")

    panel_title(axc,"c","Round x site allocation")
    p=surface[surface.panel=="HAA9"].copy();rounds=sorted(p.requested_rounds.unique());sites=sorted(p.requested_sites.unique())
    mat=p.pivot(index="requested_rounds",columns="requested_sites",values="conditional_capture").reindex(index=rounds,columns=sites).to_numpy()
    annotations={(iy,ix) for iy in range(len(rounds)) for ix in range(len(sites)) if (iy+ix)%3==0}
    sm=rectangle_heatmap(axc,mat,[str(v) for v in sites],[str(v) for v in rounds],cmap=mpl.colors.LinearSegmentedColormap.from_list("alloc",[WHITE,PALE_TEAL,TEAL,NAVY]),vmin=0,vmax=1,annotations=annotations)
    axc.set_xlabel("Sites selected");axc.set_ylabel("Administrative groups selected")
    cbar=vector_colorbar(fig,sm,ax=axc,fraction=.046,pad=.03);cbar.set_label("Conditional capture")

    panel_title(axd,"d","Equal-budget cost frontier")
    x=frontier.HAA9_to_HAA5_per_sample_cost_ratio
    axd.plot(x,frontier.equal_expected_budget_HAA9_capture,color=TEAL,lw=2,label="HAA9, equal budget")
    axd.axhline(float(frontier.full_HAA5_record_capture.iloc[0]),color=NAVY,ls="--",label="Full observed HAA5")
    axd.axvspan(critical_lower,critical_upper,color=PALE_ORANGE,alpha=.55,lw=0,label="Bootstrap interval")
    axd.axvline(critical_ratio,color=ORANGE,ls=":",lw=1.8)
    axd.scatter([critical_ratio],[critical_capture],marker="D",s=58,color=ORANGE,edgecolor=WHITE,zorder=5)
    axd.text(critical_ratio+.08,critical_capture+.035,f"break-even ≈ {critical_ratio:.2f}",color=ORANGE,fontweight="bold")
    axd.set(xlabel="HAA9:HAA5 per-sample cost ratio",ylabel="Conditional capture",ylim=(0,1.03));percent_axis(axd);clean_axis(axd);axd.legend(loc="upper right")

    caption=("Figure 4. Detection yield under finite sampling resources. (a) Expected conditional capture increases with random physical-sample selection for each panel. (b) With a sampling budget of up to four HAA9 samples, the mean point estimate is near-identical to the complete observed HAA5 record (paired difference 0.14 percentage points, 95% CI -3.04 to 3.56); three of 769 systems saturate below four samples and no comparison margin was prespecified. "
             "(c) HAA9 capture depends on the allocation of effort across administrative groups and sites. (d) Under the observed system-specific sample-count distribution and fractional randomization, the mean equal-budget break-even cost ratio is approximately 3.74; this is a scenario threshold, not a procurement estimate.")
    alt="Resource analysis showing panel-specific capture curves, a paired comparison of up to four HAA9 samples with full-record HAA5, a rounds-by-sites allocation heatmap, and an equal-budget cost-ratio frontier."
    return FigureBundle("Fig4",fig,caption,alt,[("a",selected,"wr_results/core/resource_capture_curve.csv"),("b",eqplot,"wr_results/playbook_extension/physical_sample_equivalence_curve.csv"),("c",p,"wr_results/core/joint_round_site_capture_surface.csv"),("d",frontier_source,"wr_results/playbook_extension/cost_ratio_frontier.csv; equivalence_summary.json")])


def _wilson_interval(events: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    require(total > 0, "A positive denominator is required for a Wilson interval.")
    p = events / total
    den = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / den
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def _mean_bootstrap_interval(values: Sequence[float], *, seed: int, replicates: int = 5000) -> tuple[float, float]:
    """Deterministic percentile interval for an equal-system mean."""
    array=np.asarray(values,dtype=float)
    array=array[np.isfinite(array)]
    require(len(array)>1,"At least two finite values are required for a mean bootstrap interval.")
    rng=np.random.default_rng(seed)
    draws=rng.choice(array,size=(replicates,len(array)),replace=True).mean(axis=1)
    lo,hi=np.quantile(draws,[.025,.975])
    return float(lo),float(hi)


def wr2_fig5() -> FigureBundle:
    systems = load_parquet(
        "wr_results/playbook_extension/mechanism_systems.parquet",
        ["bromide_midpoint_ug_l", "toc_midpoint_ug_l", "median_extra4_fraction", "haa5_invisible", "both_required_any", "cat_both_required", "haa9_round_cv", "haa9_round_cv_eligible", "bromide_quartile", "toc_quartile"],
    )
    models = load_csv(
        "wr_results/playbook_extension/mechanism_logistic_models.csv",
        ["outcome", "model", "term", "analysis_systems", "adjusted_odds_ratio", "odds_ratio_ci_lower", "odds_ratio_ci_upper", "p_value"],
    )
    hetero = load_csv(
        "wr_results/playbook_extension/heterogeneity_summary.csv",
        ["stratifier", "level", "systems", "HAA5_invisible_fraction", "HAA5_invisible_ci_lower", "HAA5_invisible_ci_upper", "both_required_probability", "both_required_ci_lower", "both_required_ci_upper"],
    )

    complete = systems.dropna(subset=["bromide_midpoint_ug_l", "toc_midpoint_ug_l"]).copy()
    require(len(complete) == 541, f"Expected 541 bromide/TOC-complete systems, found {len(complete)}.")
    complete["bromide_bin"] = complete.bromide_quartile.astype(str)
    complete["toc_bin"] = complete.toc_quartile.astype(str)
    require(set(complete.bromide_bin) == {"Q1", "Q2", "Q3"}, "Expected three tied-value-preserving bromide bins.")
    require(set(complete.toc_bin) == {"Q1", "Q2", "Q3", "Q4"}, "Expected four TOC quartiles.")
    landscape = complete.groupby(["toc_bin", "bromide_bin"], observed=True).agg(
        systems=("haa5_invisible", "size"),
        haa5_invisible_fraction=("haa5_invisible", "mean"),
        both_required_probability=("both_required_any", "mean"),
        bromide_median_ug_l=("bromide_midpoint_ug_l", "median"),
        toc_median_ug_l=("toc_midpoint_ug_l", "median"),
    ).reset_index()

    term_order = [
        ("haa5_invisible", "environmental_indicators", "bromide_midpoint_ug_l_per_IQR", "Invisible: bromide/IQR"),
        ("haa5_invisible", "environmental_indicators", "toc_midpoint_ug_l_per_IQR", "Invisible: TOC/IQR"),
        ("haa5_invisible", "composition_descriptor", "median_extra4_fraction_per_IQR", "Invisible: Extra4 fraction/IQR"),
        ("both_required_any", "environmental_indicators", "bromide_midpoint_ug_l_per_IQR", "Both required: bromide/IQR"),
        ("both_required_any", "environmental_indicators", "toc_midpoint_ug_l_per_IQR", "Both required: TOC/IQR"),
        ("both_required_any", "composition_descriptor", "median_extra4_fraction_per_IQR", "Both required: Extra4 fraction/IQR"),
    ]
    rows = []
    for outcome, model, term, label in term_order:
        row = models[(models.outcome == outcome) & (models.model == model) & (models.term == term)]
        require(len(row) == 1, f"Figure 5 model term is missing or duplicated: {outcome}/{model}/{term}")
        item = row.iloc[0].to_dict(); item["label"] = label; rows.append(item)
    forest = pd.DataFrame(rows)

    comp = hetero[hetero.stratifier == "extra4_fraction_quartile"].copy()
    comp["quartile_order"] = comp.level.str.extract(r"(\d+)").astype(float)
    comp = comp.sort_values("quartile_order")

    cv = systems[systems.haa9_round_cv_eligible.astype(bool) & systems.haa9_round_cv.notna()].copy()
    require(not cv.empty, "No temporal-CV-eligible systems are available for Figure 5.")
    cv["cv_quartile"] = pd.qcut(cv.haa9_round_cv.rank(method="first"), 4, labels=["Q1", "Q2", "Q3", "Q4"])
    cv_rows=[]
    for qi,(q,g) in enumerate(cv.groupby("cv_quartile",observed=True),start=1):
        n=len(g);values=g.cat_both_required.to_numpy(float);lo,hi=_mean_bootstrap_interval(values,seed=20260914+qi)
        cv_rows.append({"cv_quartile":str(q),"systems":n,"mean_both_required_probability":float(values.mean()),"ci_lower":lo,"ci_upper":hi,"bootstrap_replicates":5000,"median_HAA9_round_CV":float(g.haa9_round_cv.median())})
    cv_summary=pd.DataFrame(cv_rows)

    fig,axes=_mm_figure(152,rows=2,cols=2);axa,axb,axc,axd=axes.ravel()
    panel_title(axa,"a","Bromide x TOC landscape")
    mat=landscape.pivot(index="toc_bin",columns="bromide_bin",values="haa5_invisible_fraction").reindex(index=["Q1","Q2","Q3","Q4"],columns=["Q1","Q2","Q3"]).to_numpy()
    sm=rectangle_heatmap(axa,mat,["Q1","Q2","Q3"],["Q1","Q2","Q3","Q4"],cmap=mpl.colors.LinearSegmentedColormap.from_list("mechanism",[WHITE,PALE_ORANGE,ORANGE,MAGENTA]),vmin=0,vmax=.80,annotations={(i,j) for i in range(4) for j in range(3)})
    axa.set_xlabel("Bromide midpoint bin");axa.set_ylabel("TOC midpoint quartile")
    cbar=vector_colorbar(fig,sm,ax=axa,fraction=.046,pad=.03);cbar.set_label("HAA5-invisible fraction")

    panel_title(axb,"b","Adjusted associations")
    y=np.arange(len(forest))[::-1];colors=[ORANGE,NAVY,PURPLE,ORANGE,NAVY,PURPLE]
    colored_forest(axb,forest.adjusted_odds_ratio,y,forest.odds_ratio_ci_lower,forest.odds_ratio_ci_upper,colors,size=44)
    axb.axvline(1,color=MID,ls="--",lw=1);axb.set_xscale("log");axb.set_xticks([.5,1,2,4],["0.5","1","2","4"]);axb.xaxis.set_minor_locator(mpl.ticker.NullLocator());axb.set_yticks(y,forest.label);axb.set_xlabel("Adjusted odds ratio (log scale)");clean_axis(axb,grid="x")

    panel_title(axc,"c","Composition gradient")
    x=np.arange(len(comp))
    _errorbar_vertical(axc,x-.07,comp.HAA5_invisible_fraction,comp.HAA5_invisible_ci_lower,comp.HAA5_invisible_ci_upper,colors=[ORANGE]*len(comp),markers=["o"]*len(comp))
    _errorbar_vertical(axc,x+.07,comp.both_required_probability,comp.both_required_ci_lower,comp.both_required_ci_upper,colors=[MAGENTA]*len(comp),markers=["s"]*len(comp))
    axc.plot(x-.07,comp.HAA5_invisible_fraction,color=ORANGE,ls="--",lw=1);axc.plot(x+.07,comp.both_required_probability,color=MAGENTA,lw=1)
    axc.set_xticks(x,comp.level);axc.set_ylim(0,.78);axc.set_xlabel("Median Extra4-fraction quartile");axc.set_ylabel("System fraction");percent_axis(axc);clean_axis(axc)
    axc.legend(handles=[Line2D([0],[0],color=ORANGE,marker="o",ls="--",label="HAA5-invisible"),Line2D([0],[0],color=MAGENTA,marker="s",label="Both required")],loc="upper left")

    panel_title(axd,"d","Temporal-variability gradient")
    x=np.arange(len(cv_summary));_errorbar_vertical(axd,x,cv_summary.mean_both_required_probability,cv_summary.ci_lower,cv_summary.ci_upper,colors=[NAVY,TEAL,ORANGE,MAGENTA],markers=["o","s","^","D"])
    axd.plot(x,cv_summary.mean_both_required_probability,color=MID,lw=1)
    axd.set_xticks(x,[f"{q}\nCV {v:.2f}" for q,v in zip(cv_summary.cv_quartile,cv_summary.median_HAA9_round_CV)]);axd.set_ylim(0,max(.45,float(cv_summary.ci_upper.max()+.06)));axd.set_xlabel("HAA9 round-CV quartile");axd.set_ylabel("Mean both-required probability");percent_axis(axd);clean_axis(axd)

    caption=("Figure 5. Descriptive correlates of HAA5-invisible detection. (a) The system fraction missed by HAA5 is shown across three tied-value-preserving bromide bins and four TOC quartiles among 541 complete cases. (b) Logistic models with HC0 heteroskedasticity-robust intervals summarize adjusted associations, not causal effects. "
             "(c) HAA5-invisible and both-required outcomes increase across the additional-four composition gradient. (d) Equal-PWS mean both-required repair probability is summarized across temporal-variability quartiles among systems with at least two HAA9 rounds and a positive overall HAA9 round mean; bars are 5,000-replicate PWS bootstrap intervals.")
    alt="Mechanism figure with an annotated bromide-by-TOC heatmap, an adjusted odds-ratio forest, and outcome gradients across additional-species composition and HAA9 temporal variability."
    return FigureBundle("Fig5",fig,caption,alt,[("a",landscape,"wr_results/playbook_extension/mechanism_systems.parquet"),("b",forest,"wr_results/playbook_extension/mechanism_logistic_models.csv"),("c",comp,"wr_results/playbook_extension/heterogeneity_summary.csv"),("d",cv_summary,"wr_results/playbook_extension/mechanism_systems.parquet")])


def wr2_fig6() -> FigureBundle:
    policy=load_csv("wr_results/playbook_extension/round_policy_summary.csv",["policy","systems","selection_is_outcome_informed","p00","p10","p01","p11","additive_interaction"])
    contrasts=load_csv("wr_results/playbook_extension/round_policy_contrasts.csv",["policy","baseline_policy","paired_systems","HAA5_detection_difference","HAA5_difference_ci_lower","HAA5_difference_ci_upper","HAA9_detection_difference","HAA9_difference_ci_lower","HAA9_difference_ci_upper","additive_interaction_difference","interaction_difference_ci_lower","interaction_difference_ci_upper"])
    endpoint=load_csv("wr_results/playbook_extension/regulation_shaped_endpoint_summary.csv",["endpoint","panel","supported_systems","positive_systems","capture_among_HAA9_positive_systems"])
    selected=policy[policy.policy.isin(["random_event_code_group","warm_season_May_to_September_else_random"])].copy()
    require(len(selected)==2,"Random and warm-season policy rows are required for Figure 6.")
    warm=contrasts[contrasts.policy=="warm_season_May_to_September_else_random"].iloc[0]
    contrast=pd.DataFrame({"measure":["HAA5 detection","HAA9 detection","Additive interaction"],"difference":[warm.HAA5_detection_difference,warm.HAA9_detection_difference,warm.additive_interaction_difference],"ci_lower":[warm.HAA5_difference_ci_lower,warm.HAA9_difference_ci_lower,warm.interaction_difference_ci_lower],"ci_upper":[warm.HAA5_difference_ci_upper,warm.HAA9_difference_ci_upper,warm.interaction_difference_ci_upper],"paired_systems":int(warm.paired_systems)})
    endpoint_order=["single_sample_maximum","OEL_analogue","LRAA_analogue"];panel_order=["HAA5","HAA5_plus_BCAA","HAA6Br","HAA9"]
    endpoint["endpoint_order"]=pd.Categorical(endpoint.endpoint,endpoint_order,ordered=True);endpoint["panel_order"]=pd.Categorical(endpoint.panel,panel_order,ordered=True)
    endpoint=endpoint.sort_values(["endpoint_order","panel_order"])
    support=endpoint.groupby("endpoint",sort=False).agg(supported_systems=("supported_systems","max"),HAA9_positive_systems=("positive_systems",lambda s:int(endpoint.loc[s.index].loc[endpoint.loc[s.index,"panel"]=="HAA9","positive_systems"].iloc[0]))).reset_index()
    support["endpoint_label"]=support.endpoint.map({"single_sample_maximum":"Full-record sample maximum","OEL_analogue":"OEL research analogue","LRAA_analogue":"LRAA research analogue"})

    fig,axes=_mm_figure(148,rows=2,cols=2);axa,axb,axc,axd=axes.ravel()
    panel_title(axa,"a","Random versus warm-season round")
    x=np.arange(4);width=.34;cell_names=["HAA5\none group","HAA9\none group","HAA5\nfull grid","HAA9\nfull grid"]
    for j,row in enumerate(selected.itertuples()):
        values=[row.p00,row.p10,row.p01,row.p11];color=NAVY if row.policy.startswith("random") else ORANGE;hatch="//" if row.policy.startswith("random") else ""
        axa.bar(x+(j-.5)*width,values,width,color=color,hatch=hatch,edgecolor=WHITE,label=_short_policy(row.policy))
    axa.set_xticks(x,cell_names);axa.set_ylim(0,1.08);axa.set_ylabel("Conditional capture");percent_axis(axa);clean_axis(axa);axa.legend(loc="upper left")

    panel_title(axb,"b","Warm-season paired contrasts")
    y=np.arange(3)[::-1];colored_forest(axb,contrast.difference,y,contrast.ci_lower,contrast.ci_upper,[NAVY,TEAL,MAGENTA],size=48)
    axb.axvline(0,color=MID,ls="--",lw=1);axb.set_yticks(y,contrast.measure);axb.set_xlabel("Warm-season minus random probability");percent_axis(axb,"x");clean_axis(axb,grid="x");axb.set_xlim(-.11,.20)
    axb.text(.98,.04,"May-September when available\notherwise matched random baseline",transform=axb.transAxes,ha="right",va="bottom",color=MID)

    panel_title(axc,"c","Research endpoint analogues")
    mat=endpoint.pivot(index="endpoint",columns="panel",values="capture_among_HAA9_positive_systems").reindex(index=endpoint_order,columns=panel_order).to_numpy()
    sm=rectangle_heatmap(axc,mat,["HAA5","HAA5 +\nBCAA","HAA6Br","HAA9"],["Individual max\n(full record)","OEL analogue","LRAA analogue"],cmap=mpl.colors.LinearSegmentedColormap.from_list("endpoint",[WHITE,PALE_TEAL,TEAL,NAVY]),vmin=0,vmax=1,annotations={(i,j) for i in range(3) for j in range(4)})
    axc.set_xlabel("Analyte panel");axc.set_ylabel("Research endpoint")
    cbar=vector_colorbar(fig,sm,ax=axc,fraction=.046,pad=.03);cbar.set_label("Capture among endpoint-specific HAA9 positives")

    panel_title(axd,"d","Endpoint support")
    y=np.arange(len(support));baseline=1800.0
    markers=["o","s","D"]
    for yi,n,color,marker in zip(y,support.supported_systems,[NAVY,TEAL,ORANGE],markers):
        axd.hlines(yi,baseline,n,color=color,lw=3.0)
        axd.scatter(n,yi,color=color,marker=marker,s=62,edgecolor=WHITE,linewidth=.8,zorder=3)
    axd.set_yticks(y,support.endpoint_label);axd.invert_yaxis();axd.set_xscale("log");axd.set_xlabel("Supported systems (log scale)");clean_axis(axd,grid="x")
    axd.set_xlim(baseline,6500);axd.set_xticks([2000,3000,4000,5000,6000],["2,000","3,000","4,000","5,000","6,000"])
    axd.xaxis.set_minor_locator(mpl.ticker.NullLocator());axd.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    for yi,n,pos in zip(y,support.supported_systems,support.HAA9_positive_systems):axd.text(n*1.035,yi,f"{int(n):,} supported; {int(pos):,} HAA9+",va="center")

    caption=("Figure 6. Operational round selection and regulation-shaped research endpoints. (a) A prespecified warm-season selection increases one-round detection relative to random event-code-group selection while full-grid cells are unchanged. (b) Paired contrasts quantify the detection changes and the corresponding reduction in additive interaction. "
             "(c) Candidate-panel capture is compared for a full-record individual-sample maximum and OEL and LRAA research analogues. (d) Endpoint support differs substantially; OEL and LRAA calculations are analytical analogues and do not establish regulatory compliance.")
    alt="Operational figure comparing random and warm-season group selection, paired differences, candidate-panel capture under three research endpoints, and endpoint-specific system support."
    return FigureBundle("Fig6",fig,caption,alt,[("a",selected,"wr_results/playbook_extension/round_policy_summary.csv"),("b",contrast,"wr_results/playbook_extension/round_policy_contrasts.csv"),("c",endpoint,"wr_results/playbook_extension/regulation_shaped_endpoint_summary.csv"),("d",support,"wr_results/playbook_extension/regulation_shaped_endpoint_summary.csv")])


def wr2_fig7() -> FigureBundle:
    threshold=load_csv("wr_results/playbook_extension/fixed_B60_and_changing_cohort_thresholds.csv",["threshold_ug_l","cohort","systems","p00","p10","p01","additive_interaction","additive_interaction_ci_lower","additive_interaction_ci_upper"])
    loso=load_csv("wr_results/core/leave_one_state_out.csv",["excluded_state_code","excluded_reference_positive_systems","interaction_without_state","difference_from_overall","overall_interaction"])
    rounds=load_csv("wr_results/core/round_definition_sensitivity.csv",["analysis","n_analysis_systems","interaction","interaction_ci_lower","interaction_ci_upper","total_units"])
    fixed=load_csv("wr_results/core/fixed_support_factorial.csv",["analysis","n_analysis_systems","interaction","interaction_ci_lower","interaction_ci_upper"])
    t_specific=threshold[threshold.cohort=="threshold_specific_HAA9_positive"].copy();t_fixed=threshold[threshold.cohort=="fixed_B60_HAA9_positive"].copy()
    t_specific["panel_gain"]=t_specific.p10-t_specific.p00;t_specific["observation_gain"]=t_specific.p01-t_specific.p00
    robust=fixed.copy();robust["label"]=robust.analysis.map({"all_panel_complete_observed_grids":"Primary complete grid","source_aligned_minimum_rounds":"Source-aligned minimum","at_least_four_event_code_groups":"At least four groups","four_groups_with_repeated_site":"Repeated-site support"})
    round_plot=rounds.copy();round_plot["label"]=round_plot.analysis.map({"event_code":"Event code","event_code_year":"Event code + year","collection_date":"Collection date","facility_event_code_year":"Facility + event + year","physical_sample":"Physical sample"})

    fig,axes=_mm_figure(152,rows=2,cols=2);axa,axb,axc,axd=axes.ravel()
    panel_title(axa,"a","Threshold and cohort definition")
    for frame,label,color,marker,ls in [(t_fixed,"Fixed B60 cohort",NAVY,"o","--"),(t_specific,"Threshold-specific cohort",ORANGE,"s","-")]:
        axa.fill_between(frame.threshold_ug_l,frame.additive_interaction_ci_lower,frame.additive_interaction_ci_upper,color=color,alpha=.10)
        axa.plot(frame.threshold_ug_l,frame.additive_interaction,color=color,marker=marker,markevery=2,ls=ls,label=label)
    axa.axhline(0,color=MID,lw=1);axa.set(xlabel="Analytical threshold (µg/L)",ylabel="Additive interaction");percent_axis(axa,"y");clean_axis(axa);axa.legend(loc="lower right")

    panel_title(axb,"b","Monitoring-priority plane")
    lim=max(float(t_specific.panel_gain.max()),float(t_specific.observation_gain.max()))*1.08
    axb.fill_between([0,lim],[0,lim],[lim,lim],color=PALE_ORANGE,alpha=.65);axb.fill_between([0,lim],[0,0],[0,lim],color=PALE_TEAL,alpha=.65);axb.plot([0,lim],[0,lim],color=MID,ls="--")
    order=t_specific.sort_values("threshold_ug_l");axb.plot(order.panel_gain,order.observation_gain,color=INK,lw=1)
    axb.scatter(order.panel_gain,order.observation_gain,c=order.threshold_ug_l,cmap="viridis",s=42+90*order.systems/order.systems.max(),marker="o",edgecolor=WHITE,lw=.7)
    for b in [20,60,100]:
        row=order[np.isclose(order.threshold_ug_l,b)]
        if not row.empty:
            dx,dy={20:(.006,-.020),60:(-.020,.012),100:(-.010,.014)}[b]
            axb.text(row.panel_gain.iloc[0]+dx,row.observation_gain.iloc[0]+dy,str(b),fontsize=7.2,ha="right" if dx<0 else "left",va="top" if dy<0 else "bottom")
    axb.set(xlim=(0,lim),ylim=(0,lim),xlabel="Panel gain at one group",ylabel="Observation gain under HAA5");percent_axis(axb,"x");percent_axis(axb,"y");clean_axis(axb,grid="both")
    axb.text(.96,.92,"above line: observation priority",transform=axb.transAxes,ha="right",color=ORANGE,fontweight="bold")

    panel_title(axc,"c","Leave-one-code-out influence")
    ordered=loso.sort_values("difference_from_overall");colors=[ORANGE if v<0 else TEAL for v in ordered.difference_from_overall]
    axc.vlines(np.arange(len(ordered)),0,ordered.difference_from_overall,color=colors,lw=1.1);axc.scatter(np.arange(len(ordered)),ordered.difference_from_overall,c=colors,s=24,marker="D",edgecolor=WHITE,lw=.5)
    axc.axhline(0,color=MID,lw=1);axc.set(xlabel="Excluded represented jurisdiction code",ylabel="Change in additive interaction");percent_axis(axc,"y",decimals=1);clean_axis(axc)
    extremes=[(0,4,-.0015),(len(ordered)-1,len(ordered)-5,.0015)]
    for idx,text_x,dy in extremes:
        value=float(ordered.difference_from_overall.iloc[idx])
        axc.annotate(ordered.excluded_state_code.iloc[idx],xy=(idx,value),xytext=(text_x,value+dy),ha="center",va="top" if dy<0 else "bottom",fontsize=7,arrowprops=dict(arrowstyle="-",color=MID,lw=.8,shrinkA=1,shrinkB=1))
    axc.margins(y=.22)

    panel_title(axd,"d","Round-definition sensitivity")
    y=np.arange(len(round_plot))[::-1];colored_forest(axd,round_plot.interaction,y,round_plot.interaction_ci_lower,round_plot.interaction_ci_upper,[NAVY,TEAL,ORANGE,MAGENTA,PURPLE],size=44)
    axd.axvline(0,color=MID,ls="--");axd.set_yticks(y,round_plot.label);axd.set_xlabel("Additive interaction");percent_axis(axd,"x");clean_axis(axd,grid="x")
    axd.set_xlim(.07,.36)

    caption=("Figure 7. Robustness, estimand drift, and monitoring interpretation. (a) Fixed-B60 and threshold-specific cohorts yield different threshold trajectories, separating performance change from cohort change. (b) Threshold-specific panel and observation gains define a descriptive monitoring-priority plane. "
             "(c) Leave-one-represented-jurisdiction-code-out influence is small relative to the overall additive interaction. (d) Alternative administrative and physical-sample round definitions preserve a positive interaction, with magnitude depending on the observation unit.")
    alt="Discussion figure contrasting fixed and changing threshold cohorts, tracing panel versus observation gains, quantifying leave-one-represented-jurisdiction-code-out influence, and comparing round-definition sensitivity estimates."
    return FigureBundle("Fig7",fig,caption,alt,[("a",threshold,"wr_results/playbook_extension/fixed_B60_and_changing_cohort_thresholds.csv"),("b",t_specific,"wr_results/playbook_extension/fixed_B60_and_changing_cohort_thresholds.csv"),("c",ordered,"wr_results/core/leave_one_state_out.csv"),("d",round_plot,"wr_results/core/round_definition_sensitivity.csv")])


def wr2_fig8() -> FigureBundle:
    policy=load_csv("wr_results/playbook_extension/round_policy_summary.csv",["policy","systems","selection_is_outcome_informed","p00","p00_ci_lower","p00_ci_upper","p10","p10_ci_lower","p10_ci_upper","p01","p01_ci_lower","p01_ci_upper","p11","p11_ci_lower","p11_ci_upper"])
    endpoint=load_csv("wr_results/playbook_extension/regulation_shaped_endpoint_summary.csv",["endpoint","panel","supported_systems","positive_systems"])
    eq=load_csv("wr_results/playbook_extension/physical_sample_equivalence_curve.csv",["HAA9_samples_selected","HAA9_conditional_capture","HAA9_capture_ci_lower","HAA9_capture_ci_upper","full_HAA5_record_capture"])
    prospective=policy[policy.policy.isin(["random_event_code_group","warm_season_May_to_September_else_random","first_chronological_event_code_group"])].copy()
    prospective["label"]=prospective.policy.map(_short_policy)
    prospective["panel_gain"]=prospective.p10-prospective.p00
    prospective["observation_gain"]=prospective.p01-prospective.p00
    random=prospective[prospective.policy=="random_event_code_group"].iloc[0]
    warm=prospective[prospective.policy=="warm_season_May_to_September_else_random"].iloc[0]
    row4=eq[eq.HAA9_samples_selected==4].iloc[0]
    configs=pd.DataFrame({
        "configuration":["HAA5, one random group","HAA9, one random group","HAA9, one warm-season group","HAA5, full observed grid","HAA9, four random samples","HAA9, full observed grid"],
        "conditional_capture":[random.p00,random.p10,warm.p10,random.p01,row4.HAA9_conditional_capture,random.p11],
        "ci_lower":[random.p00_ci_lower,random.p10_ci_lower,warm.p10_ci_lower,random.p01_ci_lower,row4.HAA9_capture_ci_lower,random.p11_ci_lower],
        "ci_upper":[random.p00_ci_upper,random.p10_ci_upper,warm.p10_ci_upper,random.p01_ci_upper,row4.HAA9_capture_ci_upper,random.p11_ci_upper],
        "selection_role":["prospective random","prospective random","prespecified calendar rule","complete observed record","prospective random physical samples","reference"],
    })
    e=endpoint[endpoint.panel=="HAA9"].copy()
    e["label"]=e.endpoint.map({"single_sample_maximum":"Full-record sample maximum","OEL_analogue":"OEL research analogue","LRAA_analogue":"LRAA research analogue"})
    endpoint_order=["single_sample_maximum","OEL_analogue","LRAA_analogue"]
    e["order"]=pd.Categorical(e.endpoint,endpoint_order,ordered=True);e=e.sort_values("order")
    sequence=pd.DataFrame({
        "stage":[1,2,3,4],
        "design_question":["Define estimand","Replay observed support","Test transportability","State evidence boundary"],
        "evidence":["Panel x observation scope; 60 µg/L benchmark","Finite event-code-group x site grids; n=769 reference-positive","Threshold, round, chronology and external-archive checks","Detection capture only; incomplete population and state linkage"],
        "status":["prespecified","observed geometry","sensitivity","qualified inference"],
    })

    fig,axes=_mm_figure(148,rows=2,cols=2);axa,axb,axc,axd=axes.ravel()
    panel_title(axa,"a","Retrospective rule-gain field")
    lim=max(float(prospective.panel_gain.max()),float(prospective.observation_gain.max()))*1.12
    axa.fill_between([0,lim],[0,lim],[lim,lim],color=PALE_ORANGE,alpha=.65)
    axa.fill_between([0,lim],[0,0],[0,lim],color=PALE_TEAL,alpha=.65)
    axa.plot([0,lim],[0,lim],ls="--",color=MID)
    for row,marker,color,offset in zip(prospective.itertuples(),["o","s","D"],[NAVY,ORANGE,TEAL],[(.006,.006),(.006,-.022),(.006,.006)]):
        axa.scatter(row.panel_gain,row.observation_gain,s=75,marker=marker,color=color,edgecolor=WHITE,zorder=4)
        axa.text(row.panel_gain+offset[0],row.observation_gain+offset[1],row.label,fontsize=7.2,va="bottom" if offset[1]>=0 else "top")
    axa.set(xlim=(0,lim),ylim=(0,lim),xlabel="Panel gain at one group",ylabel="Observation gain under HAA5")
    percent_axis(axa,"x");percent_axis(axa,"y");clean_axis(axa,grid="both")
    axa.text(.97,.93,"above line: larger observation gain",transform=axa.transAxes,ha="right",color=ORANGE,fontweight="bold")

    panel_title(axb,"b","Observed detection configurations")
    y=np.arange(len(configs))[::-1];colors=[NAVY,TEAL,ORANGE,NAVY,TEAL,MAGENTA];markers=["o","s","D","o","s","D"]
    colored_forest(axb,configs.conditional_capture,y,configs.ci_lower,configs.ci_upper,colors,size=45)
    for yi,m,c,v in zip(y,markers,colors,configs.conditional_capture):axb.scatter(v,yi,marker=m,s=46,color=c,edgecolor=WHITE,zorder=5)
    axb.set_yticks(y,configs.configuration);axb.set_xlim(.12,1.04);axb.set_xlabel("Conditional capture")
    percent_axis(axb,"x");clean_axis(axb,grid="x")

    panel_title(axc,"c","Endpoint-specific support")
    y=np.arange(len(e));bars=axc.barh(y,e.supported_systems,color=[NAVY,TEAL,ORANGE],hatch=["//","xx",".."],edgecolor=WHITE,label="Supported")
    axc.scatter(e.positive_systems,y,marker="D",s=52,color=MAGENTA,edgecolor=WHITE,label="HAA9-positive",zorder=4)
    axc.set_yticks(y,e.label);axc.invert_yaxis();axc.set_xscale("log");axc.set_xlabel("Systems (log scale)")
    clean_axis(axc,grid="x");axc.legend(loc="lower right")
    for yi,n in zip(y,e.positive_systems):axc.text(n*1.08,yi,f"{int(n):,}",va="center",fontsize=7.2,color=MAGENTA)

    panel_title(axd,"d","Evidence-led design sequence")
    axd.axis("off");axd.set_xlim(0,1);axd.set_ylim(0,1)
    xs=[.12,.38,.64,.88];colors=[NAVY,TEAL,ORANGE,MAGENTA];markers=["o","s","D","^"]
    for i,(row,x,color,marker) in enumerate(zip(sequence.itertuples(),xs,colors,markers)):
        axd.scatter(x,.62,s=260,marker=marker,color=color,edgecolor=WHITE,lw=1,zorder=4)
        axd.text(x,.62,str(row.stage),ha="center",va="center",color=WHITE,fontweight="bold",fontsize=10)
        axd.text(x,.38,row.design_question.replace(" ","\n",1),ha="center",va="center",fontweight="bold",fontsize=7.2)
        axd.text(x,.14,row.status.replace(" ","\n"),ha="center",va="center",color=MID,fontsize=7)
        if i<3: straight_arrow(axd,(x+.075,.62),(xs[i+1]-.075,.62),color=MID,lw=1)
    axd.text(.50,.91,"Interpretive sequence, not a policy recommendation",ha="center",color=MAGENTA,fontweight="bold",fontsize=7.4)

    caption=("Figure 8. Descriptive design synthesis. (a) Prospectively implementable random, first-group, and warm-season rules were evaluated retrospectively and occupy different positions in the panel-gain versus observation-gain field. (b) Selected illustrative observed configurations show that similar mean capture can arise from different combinations of chemical and physical sampling scope; no comparison margin was prespecified. "
             "(c) A full-record individual-sample maximum and OEL and LRAA research analogues have different support and HAA9-positive counts. (d) The evidence sequence separates estimand definition, observed-grid replay, transportability checks, and qualified inference. This synthesis organizes the evidence and is not a policy recommendation.")
    alt="Discussion-stage synthesis showing retrospectively evaluated rule gains, selected illustrative detection configurations, endpoint-specific support, and a four-stage evidence sequence with an explicit non-policy boundary."
    return FigureBundle("Fig8",fig,caption,alt,[("a",prospective,"wr_results/playbook_extension/round_policy_summary.csv"),("b",configs,"wr_results/playbook_extension/round_policy_summary.csv; physical_sample_equivalence_curve.csv"),("c",e,"wr_results/playbook_extension/regulation_shaped_endpoint_summary.csv"),("d",sequence,"figure synthesis of prespecified study evidence")])


def wr2_fig8_clean() -> FigureBundle:
    """Whitespace-led discussion synthesis with three non-overlapping panels."""
    policy=load_csv("wr_results/playbook_extension/round_policy_summary.csv",["policy","systems","p00","p00_ci_lower","p00_ci_upper","p10","p10_ci_lower","p10_ci_upper","p01","p01_ci_lower","p01_ci_upper","p11","p11_ci_lower","p11_ci_upper"])
    eq=load_csv("wr_results/playbook_extension/physical_sample_equivalence_curve.csv",["HAA9_samples_selected","HAA9_conditional_capture","HAA9_capture_ci_lower","HAA9_capture_ci_upper"])
    eq_systems=load_csv("wr_results/playbook_extension/physical_sample_equivalence_systems.csv",["available_samples"])
    saturated_below_four=int((eq_systems.available_samples<4).sum())
    require(len(eq_systems)==769 and saturated_below_four==3,"Unexpected physical-sample support below four samples in Figure 8.")
    prospective=policy[policy.policy.isin(["random_event_code_group","warm_season_May_to_September_else_random","first_chronological_event_code_group"])].copy()
    prospective["label"]=prospective.policy.map(_short_policy);prospective["panel_gain"]=prospective.p10-prospective.p00;prospective["observation_gain"]=prospective.p01-prospective.p00
    random=prospective[prospective.policy=="random_event_code_group"].iloc[0];warm=prospective[prospective.policy=="warm_season_May_to_September_else_random"].iloc[0];row4=eq[eq.HAA9_samples_selected==4].iloc[0]
    configs=pd.DataFrame({
        "configuration":["HAA5 | random group","HAA9 | random group","HAA9 | warm group","HAA5 | full record","HAA9 | up to 4 samples","HAA9 | full record"],
        "conditional_capture":[random.p00,random.p10,warm.p10,random.p01,row4.HAA9_conditional_capture,random.p11],
        "ci_lower":[random.p00_ci_lower,random.p10_ci_lower,warm.p10_ci_lower,random.p01_ci_lower,row4.HAA9_capture_ci_lower,random.p11_ci_lower],
        "ci_upper":[random.p00_ci_upper,random.p10_ci_upper,warm.p10_ci_upper,random.p01_ci_upper,row4.HAA9_capture_ci_upper,random.p11_ci_upper],
        "scope_note":["one administrative group","one administrative group","May-September if available","complete observed HAA5 grid","uniform selection; observed-support saturation","complete observed HAA9 grid"],
    })
    configs["systems_saturating_below_four_samples"]=saturated_below_four
    configs["prespecified_comparison_margin"]=False
    sequence=pd.DataFrame({
        "stage":[1,2,3,4],
        "stage_label":["Define\nestimand","Replay\nobserved grid","Test\ntransportability","Qualify\ninference"],
        "evidence_line":["Panel x observation","4,904 eligible | 769 HAA9+","Chronology | SYR4 | ICR | CA","587 population-linked | 709 mapped"],
        "boundary":["prespecified benchmark","finite support retained","archives remain separate","detection capture; not national"],
    })

    fig=plt.figure(figsize=(FULL_W,132*MM),layout="constrained")
    gs=fig.add_gridspec(2,2,height_ratios=[1.18,.82],wspace=.12,hspace=.16)
    axa,axb=fig.add_subplot(gs[0,0]),fig.add_subplot(gs[0,1]);axc=fig.add_subplot(gs[1,:])

    panel_title(axa,"a","Retrospective rule-gain field")
    lim=max(float(prospective.panel_gain.max()),float(prospective.observation_gain.max()))*1.14
    axa.fill_between([0,lim],[0,lim],[lim,lim],color=PALE_ORANGE,alpha=.62);axa.fill_between([0,lim],[0,0],[0,lim],color=PALE_TEAL,alpha=.62);axa.plot([0,lim],[0,lim],ls="--",color=MID)
    handles=[]
    for row,marker,color in zip(prospective.itertuples(),["o","s","D"],[NAVY,ORANGE,TEAL]):
        axa.scatter(row.panel_gain,row.observation_gain,s=76,marker=marker,color=color,edgecolor=WHITE,zorder=4)
        handles.append(Line2D([0],[0],marker=marker,color="none",markerfacecolor=color,markeredgecolor=WHITE,label=row.label,markersize=7))
    axa.set(xlim=(0,lim),ylim=(0,lim),xlabel="Panel gain at one group",ylabel="Observation gain under HAA5");percent_axis(axa,"x");percent_axis(axa,"y");clean_axis(axa,grid="both")
    axa.legend(handles=handles,loc="upper left",fontsize=7.2)
    axa.text(.97,.94,"observation gain > panel gain",transform=axa.transAxes,ha="right",va="top",color=ORANGE,fontweight="bold",fontsize=7.2)

    panel_title(axb,"b","Selected illustrative configurations")
    y=np.arange(len(configs))[::-1];colors=[NAVY,TEAL,ORANGE,NAVY,TEAL,MAGENTA];markers=["o","s","D","o","s","D"]
    colored_forest(axb,configs.conditional_capture,y,configs.ci_lower,configs.ci_upper,colors,size=42)
    for yi,m,c,v in zip(y,markers,colors,configs.conditional_capture):axb.scatter(v,yi,marker=m,s=44,color=c,edgecolor=WHITE,zorder=5)
    axb.set_yticks(y,configs.configuration);axb.set_xlim(.12,1.04);axb.set_xlabel("Conditional capture");percent_axis(axb,"x");clean_axis(axb,grid="x")
    axb.axhspan(.55,2.45,color=PALE_TEAL,alpha=.55,zorder=-2)

    panel_title(axc,"c","Evidence-led interpretation sequence")
    axc.axis("off");axc.set_xlim(0,1);axc.set_ylim(0,1)
    xs=[.10,.365,.635,.90];colors=[NAVY,TEAL,ORANGE,MAGENTA];markers=["o","s","D","^"]
    axc.plot([xs[0],xs[-1]],[.57,.57],color=LIGHT,lw=5,zorder=1)
    for i,(row,x,color,marker) in enumerate(zip(sequence.itertuples(),xs,colors,markers)):
        axc.scatter(x,.57,s=300,marker=marker,color=color,edgecolor=WHITE,lw=1,zorder=4)
        axc.text(x,.57,str(row.stage),ha="center",va="center",color=WHITE,fontweight="bold",fontsize=10)
        axc.text(x,.31,row.stage_label,ha="center",va="center",fontweight="bold",fontsize=8.2,linespacing=.94)
        axc.text(x,.82,row.evidence_line,ha="center",va="center",fontsize=7.4,color=INK)
        axc.text(x,.08,row.boundary,ha="center",va="center",fontsize=7,color=MID)
        if i<3: straight_arrow(axc,(x+.07,.57),(xs[i+1]-.07,.57),color=MID,lw=1)
    axc.text(.50,.97,"Descriptive synthesis, not a policy recommendation",ha="center",va="top",color=MAGENTA,fontweight="bold",fontsize=7.6)

    caption=("Figure 8. Descriptive design synthesis. (a) Prospectively implementable random, first-group, and warm-season rules were evaluated retrospectively and occupy different positions in the panel-gain versus observation-gain field. (b) Selected illustrative configurations show that similar mean capture can arise from different chemical and physical sampling scopes; the up-to-four-HAA9 and complete-HAA5 mean point estimates are near-identical, three of 769 systems saturate below four samples, and no comparison margin was prespecified. "
             "(c) Interpretation proceeds from a defined estimand through finite-grid replay and separate transportability checks to an explicit evidence boundary. This synthesis organizes the observed evidence and is not a policy recommendation.")
    alt="Three-panel discussion synthesis with a retrospective rule-gain field, conditional capture for six selected illustrative configurations, and a spacious four-stage evidence interpretation sequence."
    return FigureBundle("Fig8",fig,caption,alt,[("a",prospective,"wr_results/playbook_extension/round_policy_summary.csv"),("b",configs,"wr_results/playbook_extension/round_policy_summary.csv; physical_sample_equivalence_curve.csv"),("c",sequence,"figure synthesis of prespecified study evidence")])


def wr2_figs9() -> FigureBundle:
    models=load_csv("wr_results/playbook_extension/mechanism_logistic_models.csv",["outcome","model","term","analysis_systems","adjusted_odds_ratio","odds_ratio_ci_lower","odds_ratio_ci_upper","p_value"])
    policy=load_csv("wr_results/playbook_extension/round_policy_summary.csv",["policy","systems","selection_is_outcome_informed","additive_interaction","additive_interaction_ci_lower","additive_interaction_ci_upper"])
    exact=load_csv("wr_results/playbook_extension/exact_five_species_panel_sensitivity.csv",["five_species_panel","is_regulatory_HAA5","additive_interaction","descending_interaction_rank"])
    negative=load_csv("wr_results/playbook_extension/negative_controls.csv")

    env=models[(models.model=="environmental_indicators") & (models.term!="const")].copy()
    comp=models[(models.model=="composition_descriptor") & (models.term!="const")].copy()
    label_map={
        "bromide_midpoint_ug_l_per_IQR":"Bromide midpoint / IQR",
        "toc_midpoint_ug_l_per_IQR":"TOC midpoint / IQR",
        "median_extra4_fraction_per_IQR":"Median Extra4 fraction / IQR",
        "observed_round_count_per_doubling":"Observed groups / doubling",
        "observed_site_count_per_doubling":"Observed sites / doubling",
        "source_group_groundwater_or_under_influence":"Groundwater or under influence",
        "source_group_mixed_or_multiple":"Mixed or multiple source",
        "size_profile_S":"Small system profile",
    }
    env["label"]=env.term.map(label_map);comp["label"]=comp.term.map(label_map)
    policy["label"]=policy.policy.map(_short_policy)
    policy["evidence_role"]=np.where(policy.selection_is_outcome_informed.astype(bool),"outcome-informed diagnostic",np.where(policy.policy.str.contains("TOC"),"complete-case diagnostic","prospective rule"))
    exact["is_regulatory_HAA5"]=exact.is_regulatory_HAA5.astype(bool)
    haa5=exact[exact.is_regulatory_HAA5]
    require(len(haa5)==1,"The exact-five-species sensitivity must contain one regulatory HAA5 row.")

    fig,axes=_mm_figure(168,rows=2,cols=2);axa,axb,axc,axd=axes.ravel()
    for ax,frame,letter,title in [(axa,env,"a","Environmental models"),(axb,comp,"b","Composition models")]:
        frame=frame.copy();frame["row_label"]=frame.label+" | "+frame.outcome.map({"haa5_invisible":"invisible","both_required_any":"both required"})
        y=np.arange(len(frame))[::-1];colors=[ORANGE if o=="haa5_invisible" else MAGENTA for o in frame.outcome]
        colored_forest(ax,frame.adjusted_odds_ratio,y,frame.odds_ratio_ci_lower,frame.odds_ratio_ci_upper,colors,size=35)
        ax.axvline(1,color=MID,ls="--",lw=1);ax.set_xscale("log");ax.set_xlim(.14,5.2);ax.set_xticks([.2,.5,1,2,5],["0.2","0.5","1","2","5"]);ax.xaxis.set_minor_locator(mpl.ticker.NullLocator());ax.set_yticks(y,frame.row_label);ax.set_xlabel("Adjusted odds ratio (log scale)");clean_axis(ax,grid="x");panel_title(ax,letter,title)

    panel_title(axc,"c","Round-policy diagnostics")
    p=policy.sort_values("additive_interaction",ascending=False);y=np.arange(len(p))[::-1]
    markers=["D" if r=="outcome-informed diagnostic" else "s" if r=="complete-case diagnostic" else "o" for r in p.evidence_role]
    colors=[MAGENTA if r=="outcome-informed diagnostic" else ORANGE if r=="complete-case diagnostic" else TEAL for r in p.evidence_role]
    colored_forest(axc,p.additive_interaction,y,p.additive_interaction_ci_lower,p.additive_interaction_ci_upper,colors,size=40)
    for yi,marker,color,val in zip(y,markers,colors,p.additive_interaction):axc.scatter(val,yi,marker=marker,s=44,color=color,edgecolor=WHITE,zorder=5)
    axc.axvline(0,color=MID,ls="--");axc.set_yticks(y,p.label);axc.set_xlabel("Additive interaction");percent_axis(axc,"x");clean_axis(axc,grid="x")
    axc.legend(handles=[Line2D([0],[0],marker="o",color="none",markerfacecolor=TEAL,label="Prospective rule"),Line2D([0],[0],marker="s",color="none",markerfacecolor=ORANGE,label="Complete-case diagnostic"),Line2D([0],[0],marker="D",color="none",markerfacecolor=MAGENTA,label="Outcome-informed oracle")],loc="lower left",fontsize=7)

    panel_title(axd,"d","Panel negative controls")
    bins=np.linspace(float(exact.additive_interaction.min()),float(exact.additive_interaction.max()),18)
    axd.hist(exact.loc[~exact.is_regulatory_HAA5,"additive_interaction"],bins=bins,color=LIGHT,edgecolor=MID,hatch="//",label="Other exact five-species panels")
    hv=float(haa5.additive_interaction.iloc[0]);rank=int(haa5.descending_interaction_rank.iloc[0])
    axd.axvline(hv,color=MAGENTA,lw=2,label=f"Regulatory HAA5: rank {rank}/126")
    null_value=float(negative["additive_interaction"].iloc[0]) if "additive_interaction" in negative.columns else 0.0
    axd.axvline(null_value,color=NAVY,ls="--",lw=1.6,label="Duplicated-panel exact null")
    axd.set(xlabel="Additive interaction",ylabel="Exact five-species panels",xlim=(-.03,.64));axd.set_xticks([0,.2,.4,.6]);percent_axis(axd,"x");clean_axis(axd);axd.legend(loc="upper left")

    caption=("Figure S9. Mechanism models and diagnostic sensitivities. (a,b) Full non-intercept coefficients from prespecified environmental and composition logistic models are associations with HC0 heteroskedasticity-robust intervals. (c) TOC-based and outcome-oracle selection rules are nonprospective diagnostics and are separated from prospective random, first-group, and warm-season rules. "
             "(d) The 126 exact five-species panel enumeration is exploratory rather than a formal randomization test; duplicating HAA5 as the expanded panel returns the expected zero interaction.")
    alt="Supplementary figure with full adjusted-model coefficient forests, round-policy diagnostic estimates, and exact-size panel and duplicated-panel negative controls."
    exact_source=exact.assign(duplicated_panel_null_additive_interaction=null_value)
    return FigureBundle("FigS9",fig,caption,alt,[("a",env,"wr_results/playbook_extension/mechanism_logistic_models.csv"),("b",comp,"wr_results/playbook_extension/mechanism_logistic_models.csv"),("c",policy,"wr_results/playbook_extension/round_policy_summary.csv"),("d",exact_source,"wr_results/playbook_extension/exact_five_species_panel_sensitivity.csv; negative_controls.csv")])


_STATE_TILE = {
    "WA":(0,0),"MT":(2,0),"ND":(3,0),"MN":(4,0),"WI":(5,0),"MI":(6,0),"VT":(8,0),"NH":(9,0),"ME":(10,0),
    "OR":(0,1),"ID":(1,1),"WY":(2,1),"SD":(3,1),"IA":(4,1),"IL":(5,1),"IN":(6,1),"OH":(7,1),"PA":(8,1),"NY":(9,1),"MA":(10,1),
    "CA":(0,2),"NV":(1,2),"UT":(2,2),"CO":(3,2),"NE":(4,2),"MO":(5,2),"KY":(6,2),"WV":(7,2),"VA":(8,2),"MD":(9,2),"NJ":(10,2),"CT":(11,2),"RI":(12,2),
    "AZ":(1,3),"NM":(2,3),"KS":(3,3),"OK":(4,3),"AR":(5,3),"TN":(6,3),"NC":(7,3),"SC":(8,3),"DE":(9,3),"DC":(10,3),
    "TX":(3,4),"LA":(5,4),"MS":(6,4),"AL":(7,4),"GA":(8,4),"FL":(9,4),
    "AK":(0,6),"HI":(1,6),
}


def wr2_figs10() -> FigureBundle:
    pop=load_csv("wr_results/playbook_extension/population_weighting_summary.csv",["weighting_scheme","systems","covered_population","effective_sample_size","largest_system_weight_share","p00","p10","p01","p11","additive_interaction","additive_interaction_ci_lower","additive_interaction_ci_upper"])
    states=load_csv("wr_results/playbook_extension/state_map_values.csv",["state_profile","systems","additive_interaction","HAA5_invisible_fraction","both_required_probability","adjusted_population_covered"])
    pop["label"]=pop.weighting_scheme.map({"system_equal_complete_case":"Equal, linked subset","adjusted_population":"Adjusted population","retail_population":"Retail population","adjusted_population_capped_at_complete_case_p99":"Adjusted, 99th-pct cap"})
    cells=[]
    for row in pop.itertuples():
        for cell,value in zip(["HAA5 one group","HAA9 one group","HAA5 full grid","HAA9 full grid"],[row.p00,row.p10,row.p01,row.p11]):cells.append({"weighting_scheme":row.weighting_scheme,"label":row.label,"cell":cell,"conditional_capture":value})
    cell_frame=pd.DataFrame(cells)
    tile=pd.DataFrame([{"state_profile":state,"tile_x":xy[0],"tile_y":xy[1]} for state,xy in _STATE_TILE.items()]).merge(states,on="state_profile",how="left")
    tile["support_class"]=np.select([tile.systems.isna(),tile.systems.fillna(0)<10],["no supported systems","fewer than 10 systems"],default="10 or more systems")

    fig,axes=_mm_figure(168,rows=2,cols=2);axa,axb,axc,axd=axes.ravel()
    panel_title(axa,"a","Population-weighted interaction")
    y=np.arange(len(pop));colored_forest(axa,pop.additive_interaction,y,pop.additive_interaction_ci_lower,pop.additive_interaction_ci_upper,[NAVY,TEAL,ORANGE,PURPLE],size=46)
    axa.axvline(0,color=MID,ls="--");axa.set_yticks(y,pop.label);axa.invert_yaxis();axa.set_xlabel("Additive interaction");percent_axis(axa,"x");clean_axis(axa,grid="x");axa.set_xlim(0,.36)
    axa.text(.98,.04,"587/769 reference-positive systems linked\nsecondary complete-case sensitivity",transform=axa.transAxes,ha="right",va="bottom",color=MID)

    panel_title(axb,"b","Weighted factorial cells")
    schemes=["system_equal_complete_case","adjusted_population","adjusted_population_capped_at_complete_case_p99"]
    styles=[(NAVY,"o","--"),(TEAL,"s","-"),(PURPLE,"D",":")]
    for scheme,(color,marker,ls) in zip(schemes,styles):
        g=cell_frame[cell_frame.weighting_scheme==scheme];axb.plot(np.arange(4),g.conditional_capture,color=color,marker=marker,ls=ls,label=g.label.iloc[0])
    axb.set_xticks(np.arange(4),["HAA5\none","HAA9\none","HAA5\nfull","HAA9\nfull"]);axb.set_ylim(0,1.04);axb.set_ylabel("Conditional capture");percent_axis(axb);clean_axis(axb);axb.legend(loc="lower right")

    panel_title(axc,"c","State tile sensitivity")
    axc.set_xlim(-.3,13.1);axc.set_ylim(6.9,-.4);axc.set_aspect("equal");axc.axis("off")
    norm=mcolors.Normalize(vmin=float(states.additive_interaction.quantile(.05)),vmax=float(states.additive_interaction.quantile(.95)));cm=mpl.colormaps["PuBuGn"]
    for row in tile.itertuples():
        if pd.isna(row.additive_interaction):fc=LIGHT;hatch="xx"
        else:fc=cm(norm(row.additive_interaction));hatch="///" if row.systems<10 else None
        axc.add_patch(Rectangle((row.tile_x,row.tile_y),.86,.78,facecolor=fc,edgecolor=WHITE,lw=.8,hatch=hatch))
        axc.text(row.tile_x+.43,row.tile_y+.39,row.state_profile,ha="center",va="center",fontsize=7,fontweight="bold",color=INK if pd.isna(row.additive_interaction) or row.additive_interaction<states.additive_interaction.quantile(.70) else WHITE)
    sm=mpl.cm.ScalarMappable(norm=norm,cmap=cm);cbar=vector_colorbar(fig,sm,ax=axc,fraction=.046,pad=.02);cbar.set_label("Additive interaction")
    axc.text(.01,.02,"hatched: <10 systems; cross-hatched: no supported systems",transform=axc.transAxes,fontsize=7,color=MID)

    panel_title(axd,"d","State support and heterogeneity")
    sizes=22+110*states.adjusted_population_covered.fillna(0)/max(1,float(states.adjusted_population_covered.max()))
    axd.scatter(states.systems,states.additive_interaction,s=sizes,c=states.HAA5_invisible_fraction,cmap="viridis",marker="o",edgecolor=WHITE,lw=.6)
    axd.set_xscale("log");axd.set(xlabel="Reference-positive systems (log scale)",ylabel="Additive interaction");percent_axis(axd,"y");clean_axis(axd,grid="both")
    for row in states.sort_values("systems",ascending=False).head(6).itertuples():axd.text(row.systems*1.04,row.additive_interaction,row.state_profile,va="center",fontsize=7)
    axd.text(.98,.04,"Color: HAA5-invisible fraction\nPoint area: linked adjusted population",transform=axd.transAxes,ha="right",va="bottom",color=MID)

    caption=("Figure S10. Incomplete population and geographic sensitivity analyses. (a,b) Population-weighted estimates use 587 of 769 reference-positive systems linked to EPA SYR4 population fields and are not nationally representative. (c) The tile map retains only verified 50-state or District of Columbia codes; states with fewer than 10 systems are hatched and territories or numeric regional codes are not reassigned. "
             "(d) State estimates are descriptive and become unstable at low support; point area reflects linked adjusted population rather than a sampling weight for national inference.")
    alt="Supplementary figure with population-weighted interaction and factorial estimates, a support-aware US state tile map, and a state-level support-versus-interaction scatterplot."
    return FigureBundle("FigS10",fig,caption,alt,[("a",pop,"wr_results/playbook_extension/population_weighting_summary.csv"),("b",cell_frame,"wr_results/playbook_extension/population_weighting_summary.csv"),("c",tile,"wr_results/playbook_extension/state_map_values.csv; state_map_coverage.json"),("d",states,"wr_results/playbook_extension/state_map_values.csv")])


def wr2_figs10_final() -> FigureBundle:
    """Consolidate population/geography and cross-archive validation in SI."""
    pop=load_csv("wr_results/playbook_extension/population_weighting_summary.csv",["weighting_scheme","systems","covered_population","effective_sample_size","largest_system_weight_share","p00","p10","p01","p11","additive_interaction","additive_interaction_ci_lower","additive_interaction_ci_upper"])
    states=load_csv("wr_results/playbook_extension/state_map_values.csv",["state_profile","systems","additive_interaction","HAA5_invisible_fraction","both_required_probability","adjusted_population_covered"])
    strict=load_csv("wr_results/validation/syr4_strict_site_and_overlap_sensitivity_metrics.csv",["cohort","metric","estimate","cluster_bootstrap_ci_low","cluster_bootstrap_ci_high"])
    quarter=load_csv("wr_results/validation/syr4_quarter_capture_curves_extended.csv",["cohort","analyte","n_available_quarters","quarters_sampled_k","mean_exact_capture_probability","cluster_bootstrap_ci_low","cluster_bootstrap_ci_high"])
    icr=load_csv("wr_results/validation/icr_temporal_support_comparison.csv",["analysis_version","scope","analyte","time_scale","full_six_quarter_systems","full_window_positive_systems"])
    ca=load_parquet("wr_results/california/ca_exact_haa5_tthm_pairs_2023_2025.parquet",["haa5_value_midpoint_ug_l","tthm_value_midpoint_ug_l","haa5_any_quantified","tthm_any_quantified","pws_id"])
    casens=load_csv("wr_results/california/ca_paired_concordance_sensitivity.csv",["analysis","spearman_rho","ci95_low","ci95_high"])
    caagreement=load_csv("wr_results/california/ca_reference_threshold_agreement.csv",["metric","estimate","ci95_low","ci95_high"])

    pop["label"]=pop.weighting_scheme.map({"system_equal_complete_case":"Equal, linked subset","adjusted_population":"Adjusted population","retail_population":"Retail population","adjusted_population_capped_at_complete_case_p99":"Adjusted, 99th-pct cap"})
    cells=[]
    for row in pop.itertuples():
        for cell,value in zip(["HAA5 one group","HAA9 one group","HAA5 full grid","HAA9 full grid"],[row.p00,row.p10,row.p01,row.p11]):
            cells.append({"weighting_scheme":row.weighting_scheme,"label":row.label,"cell":cell,"conditional_capture":value})
    cell_frame=pd.DataFrame(cells)
    tile=pd.DataFrame([{"state_profile":state,"tile_x":xy[0],"tile_y":xy[1]} for state,xy in _STATE_TILE.items()]).merge(states,on="state_profile",how="left")
    tile["support_class"]=np.select([tile.systems.isna(),tile.systems.fillna(0)<10],["no supported systems","fewer than 10 systems"],default="10 or more systems")
    strict_keep=strict[strict.cohort.isin(["same_date_all_pairs","strict_site_primary"]) & strict.metric.isin(["spearman_rho_both_detected","threshold_status_agreement"])].copy()
    qkeep=quarter[(quarter.n_available_quarters==4) & quarter.cohort.isin(["all_valid_single_analyte_records","paired_strict_site_sensitivity"])].copy()
    ikeep=icr[(icr.scope=="primary_distribution") & (icr.time_scale=="quarterly")].copy()
    ca_detected=ca[ca.haa5_any_quantified.astype(bool) & ca.tthm_any_quantified.astype(bool)].copy()
    ca_xmax=max(100,min(float(np.ceil(np.nanpercentile(ca_detected.haa5_value_midpoint_ug_l,99.5)/20)*20),240))
    ca_ymax=max(120,min(float(np.ceil(np.nanpercentile(ca_detected.tthm_value_midpoint_ug_l,99.5)/20)*20),260))
    cadens=binned_density(ca_detected.haa5_value_midpoint_ug_l.to_numpy(),ca_detected.tthm_value_midpoint_ug_l.to_numpy(),xmax=ca_xmax,ymax=ca_ymax,bins=44)
    rho=casens[casens.analysis.str.contains("both analytes quantified",case=False,na=False)].iloc[0]
    agreement=caagreement[caagreement.metric=="overall_agreement"].iloc[0]
    cadens["spearman_rho"]=float(rho.spearman_rho);cadens["spearman_ci_low"]=float(rho.ci95_low);cadens["spearman_ci_high"]=float(rho.ci95_high);cadens["status_agreement"]=float(agreement.estimate);cadens["quantified_pairs"]=len(ca_detected)

    fig=plt.figure(figsize=(FULL_W,244*MM),layout="constrained")
    gs=fig.add_gridspec(4,2,height_ratios=[.75,1.35,.85,.85],wspace=.10,hspace=.12)
    axa,axb=fig.add_subplot(gs[0,0]),fig.add_subplot(gs[0,1])
    axc=fig.add_subplot(gs[1,:])
    axd,axe=fig.add_subplot(gs[2,0]),fig.add_subplot(gs[2,1])
    axf,axg=fig.add_subplot(gs[3,0]),fig.add_subplot(gs[3,1])

    panel_title(axa,"a","Population-weighted interaction")
    y=np.arange(len(pop));colored_forest(axa,pop.additive_interaction,y,pop.additive_interaction_ci_lower,pop.additive_interaction_ci_upper,[NAVY,TEAL,ORANGE,PURPLE],size=40)
    axa.axvline(0,color=MID,ls="--");axa.set_yticks(y,pop.label);axa.invert_yaxis();axa.set_xlabel("Additive interaction");percent_axis(axa,"x");clean_axis(axa,grid="x");axa.set_xlim(0,.36)

    panel_title(axb,"b","Weighted factorial cells")
    schemes=["system_equal_complete_case","adjusted_population","adjusted_population_capped_at_complete_case_p99"]
    styles=[(NAVY,"o","--"),(TEAL,"s","-"),(PURPLE,"D",":")]
    for scheme,(color,marker,ls) in zip(schemes,styles):
        g=cell_frame[cell_frame.weighting_scheme==scheme]
        axb.plot(np.arange(4),g.conditional_capture,color=color,marker=marker,ls=ls,label=g.label.iloc[0])
    axb.set_xticks(np.arange(4),["HAA5\none","HAA9\none","HAA5\nfull","HAA9\nfull"]);axb.set_ylim(0,1.04);axb.set_ylabel("Conditional capture");percent_axis(axb);clean_axis(axb);axb.legend(loc="lower right",fontsize=7)

    panel_title(axc,"c","Support-aware state tile sensitivity")
    axc.set_xlim(-.3,13.1);axc.set_ylim(6.9,-.4);axc.set_aspect("equal");axc.axis("off")
    norm=mcolors.Normalize(vmin=float(states.additive_interaction.quantile(.05)),vmax=float(states.additive_interaction.quantile(.95)));cm=mpl.colormaps["PuBuGn"]
    threshold_color=float(states.additive_interaction.quantile(.70))
    for row in tile.itertuples():
        if pd.isna(row.additive_interaction):fc=LIGHT;hatch="xx"
        else:fc=cm(norm(row.additive_interaction));hatch="///" if row.systems<10 else None
        axc.add_patch(Rectangle((row.tile_x,row.tile_y),.86,.78,facecolor=fc,edgecolor=WHITE,lw=.8,hatch=hatch))
        axc.text(row.tile_x+.43,row.tile_y+.39,row.state_profile,ha="center",va="center",fontsize=7.5,fontweight="bold",color=INK if pd.isna(row.additive_interaction) or row.additive_interaction<threshold_color else WHITE)
    sm=mpl.cm.ScalarMappable(norm=norm,cmap=cm);cbar=vector_colorbar(fig,sm,ax=axc,fraction=.026,pad=.015);cbar.set_label("Additive interaction")
    axc.text(.01,.02,"hatched: <10 systems; cross-hatched: no supported systems; territories and numeric regional codes excluded",transform=axc.transAxes,fontsize=7,color=MID)

    panel_title(axd,"d","Strict-site SYR4 linkage")
    metric_order=["spearman_rho_both_detected","threshold_status_agreement"];names=["Spearman rho","Status agreement"]
    for yi,metric in enumerate(metric_order):
        rows=strict_keep[strict_keep.metric==metric].set_index("cohort");x0=rows.loc["same_date_all_pairs","estimate"];x1=rows.loc["strict_site_primary","estimate"]
        axd.plot([x0,x1],[yi,yi],color=LIGHT,lw=4);axd.scatter(x0,yi,color=NAVY,marker="o",s=30,zorder=3);axd.scatter(x1,yi,facecolors="none",edgecolors=ORANGE,linewidths=1.4,marker="s",s=58,zorder=4);axd.text(max(x0,x1)+.009,yi,f"Δ {x1-x0:+.5f}",va="center",fontsize=7)
    axd.set_yticks([0,1],names);axd.invert_yaxis();axd.set_xlim(.55,.98);axd.set_xlabel("Estimate");clean_axis(axd,grid="x")

    panel_title(axe,"e","SYR4 quarter curves")
    styles_q={("all_valid_single_analyte_records","HAA5"):(ORANGE,"o","-","HAA5, all valid"),("all_valid_single_analyte_records","TTHM"):(TEAL,"s","-","TTHM, all valid"),("paired_strict_site_sensitivity","HAA5"):(ORANGE,"o","--","HAA5, strict-site"),("paired_strict_site_sensitivity","TTHM"):(TEAL,"s","--","TTHM, strict-site")}
    for (cohort,analyte),(color,marker,ls,label) in styles_q.items():
        g=qkeep[(qkeep.cohort==cohort)&(qkeep.analyte==analyte)].sort_values("quarters_sampled_k");axe.plot(g.quarters_sampled_k,g.mean_exact_capture_probability,color=color,marker=marker,ls=ls,label=label)
    axe.set(xlim=(.8,4.2),ylim=(.3,1.02),xticks=[1,2,3,4],xlabel="Quarters sampled",ylabel="Conditional capture");percent_axis(axe);clean_axis(axe,grid="both");axe.legend(loc="lower right",fontsize=7)

    panel_title(axf,"f","ICR reporting-limit recovery")
    versions=["original_minus999_excluded","mrl_bounded_recovery"];x=np.arange(2);width=.34
    for j,analyte in enumerate(["HAA5","HAA9"]):
        g=ikeep[ikeep.analyte==analyte].set_index("analysis_version").loc[versions];bars=axf.bar(x+(j-.5)*width,g.full_six_quarter_systems,width,color=ORANGE if analyte=="HAA5" else TEAL,hatch="//" if analyte=="HAA5" else "",edgecolor=WHITE,label=analyte)
        for bar,pos in zip(bars,g.full_window_positive_systems):axf.text(bar.get_x()+bar.get_width()/2,bar.get_height()+3,f"{int(bar.get_height())}; {int(pos)} ref+",ha="center",va="bottom",fontsize=7)
    axf.set_xticks(x,["Exclude -999","MRL-bounded"]);axf.set_ylabel("Six-quarter systems");axf.set_ylim(0,205);clean_axis(axf);axf.legend(loc="upper left",ncol=2,fontsize=7)

    panel_title(axg,"g","California HAA5-TTHM context")
    sm2=plot_density_rects(axg,cadens,cmap="PuBuGn");axg.axvline(60,color=ORANGE,ls=":");axg.axhline(80,color=TEAL,ls=":")
    axg.set(xlim=(0,ca_xmax),ylim=(0,ca_ymax),xlabel="HAA5 (µg/L)",ylabel="TTHM (µg/L)");clean_axis(axg,grid=None)
    axg.text(.04,.94,f"rho={rho.spearman_rho:.3f} ({rho.ci95_low:.3f}-{rho.ci95_high:.3f})\n{len(ca_detected):,} quantified pairs",transform=axg.transAxes,va="top",fontweight="bold",fontsize=7.2,bbox=dict(fc=WHITE,ec="none",alpha=.8,pad=2))
    axg.text(.96,.05,f"60/80 agreement {agreement.estimate:.1%}",transform=axg.transAxes,ha="right",va="bottom",fontsize=7,color=INK,bbox=dict(fc=WHITE,ec="none",alpha=.8,pad=2))
    cbar2=vector_colorbar(fig,sm2,ax=axg,fraction=.046,pad=.02);cbar2.set_label("log10(pairs + 1)")

    caption=("Figure S10. Population, geography, and cross-archive validation. (a,b) Population-weighted estimates use 587 of 769 reference-positive systems linked to EPA SYR4 population fields and are not nationally representative. (c) The tile map retains only verified 50-state or District of Columbia codes; states with fewer than 10 systems are hatched. "
             "(d) Strict-site linkage changes SYR4 concordance negligibly. (e) SYR4 quarter curves compare all-valid and strict paired-site cohorts. (f) ICR reporting-limit-bounded recovery expands support without concentration imputation. (g) California 2023-2025 HAA5-TTHM pairs provide current contextual validation; external archives remain separate and are not pooled with UCMR4.")
    alt="Seven-panel supplementary figure combining incomplete population-weighted estimates, a support-aware state tile map, SYR4 linkage and quarter checks, ICR reporting-limit recovery, and California HAA5-TTHM contextual validation."
    return FigureBundle("FigS10",fig,caption,alt,[("a",pop,"wr_results/playbook_extension/population_weighting_summary.csv"),("b",cell_frame,"wr_results/playbook_extension/population_weighting_summary.csv"),("c",tile,"wr_results/playbook_extension/state_map_values.csv; state_map_coverage.json"),("d",strict_keep,"wr_results/validation/syr4_strict_site_and_overlap_sensitivity_metrics.csv"),("e",qkeep,"wr_results/validation/syr4_quarter_capture_curves_extended.csv"),("f",ikeep,"wr_results/validation/icr_temporal_support_comparison.csv"),("g",cadens,"wr_results/california/ca_exact_haa5_tthm_pairs_2023_2025.parquet; concordance and agreement summaries")])


def _expand_panel_selector(selector: str) -> list[str]:
    """Expand compact source selectors such as ``a-c`` or ``a,c,d``."""
    output: list[str] = []
    for token in selector.replace(" ", "").split(","):
        if "-" in token:
            start, end = token.split("-", 1)
            require(len(start) == len(end) == 1 and start.isalpha() and end.isalpha(), f"Invalid panel selector: {selector}")
            output.extend(chr(code) for code in range(ord(start.lower()), ord(end.lower()) + 1))
        else:
            require(len(token) == 1 and token.isalpha(), f"Invalid panel selector: {selector}")
            output.append(token.lower())
    require(bool(output), f"Panel selector expands to no panels: {selector}")
    return sorted(set(output))


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    frame.to_csv(tmp, index=False, encoding="utf-8-sig", lineterminator="\n")
    require(tmp.exists() and tmp.stat().st_size > 10, f"Failed to stage source data: {path}")
    os.replace(tmp, path)


def _atomic_write_text(path: Path, content: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(content, encoding="utf-8", newline="\n")
    require(tmp.exists() and tmp.stat().st_size > 0, f"Failed to stage text output: {path}")
    os.replace(tmp, path)


def _serialize_sources(bundle: FigureBundle) -> tuple[list[Path], list[str]]:
    """Write exactly one source table for every rendered panel."""
    SOURCE_OUT.mkdir(parents=True, exist_ok=True)
    by_panel: dict[str, list[tuple[pd.DataFrame, str]]] = {}
    inputs: list[str] = []
    for selector, frame, source in bundle.sources:
        require(isinstance(frame, pd.DataFrame) and not frame.empty, f"{bundle.name} panel {selector} has empty source data")
        inputs.append(source)
        for panel in _expand_panel_selector(selector):
            by_panel.setdefault(panel, []).append((frame.copy(), source))
    require(bool(by_panel), f"{bundle.name} has no source-data panels.")

    paths: list[Path] = []
    for panel, components in sorted(by_panel.items()):
        if len(components) == 1:
            frame, source = components[0]
            table = source_frame(frame, panel, source)
        else:
            # Heterogeneous annotation inputs are normalized to one long,
            # panel-level table: one value per row, one variable-name column.
            long_parts = []
            for component_index, (frame, source) in enumerate(components, start=1):
                wide = frame.reset_index(drop=True).copy()
                wide.insert(0, "source_row", np.arange(len(wide), dtype=int))
                long = wide.melt(id_vars="source_row", var_name="variable", value_name="value")
                long.insert(0, "source_input_path", source)
                long.insert(0, "source_component", component_index)
                long.insert(0, "source_panel", panel)
                long_parts.append(long)
            table = pd.concat(long_parts, ignore_index=True)
        path = SOURCE_OUT / f"{bundle.name}{panel}_source.csv"
        _atomic_write_csv(table, path)
        paths.append(path)
    return paths, sorted(set(inputs))


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b""):h.update(chunk)
    return h.hexdigest()


def asset_output_dir(name: str) -> Path:
    if name == "Graphical_Abstract":
        return OUT / "graphical_abstract"
    if name.startswith("FigS"):
        return OUT / "supplementary"
    return OUT / "main"


def _validate_figure_design(bundle: FigureBundle) -> None:
    """Fail explicitly on journal-size, typography, and canvas-boundary defects."""
    require(bundle.name in ASSET_ORDER, f"Unexpected asset name: {bundle.name}")
    width_in, height_in = bundle.fig.get_size_inches()
    require(abs(width_in / MM - 190.0) < 0.02, f"{bundle.name} is not 190 mm wide: {width_in / MM:.3f} mm")
    require(height_in > 1.0, f"{bundle.name} has an implausible height.")
    bundle.fig.canvas.draw()
    if bundle.name == "Graphical_Abstract":
        for artist in bundle.fig.findobj(mpl.text.Text):
            if artist.get_visible() and artist.get_text().strip() and artist.get_fontsize() < 11.0:
                artist.set_fontsize(11.0)
        bundle.fig.canvas.draw()
    texts = [artist for artist in bundle.fig.findobj(mpl.text.Text) if artist.get_visible() and artist.get_text().strip()]
    require(bool(texts), f"{bundle.name} has no visible text.")
    named_panels=[artist.get_text() for artist in texts if re.match(r"^\([a-z]\)(?:\s|$)",artist.get_text()) and re.fullmatch(r"\([a-z]\)",artist.get_text()) is None]
    require(not named_panels,f"{bundle.name} contains descriptive panel headings: {named_panels}")
    too_small = [(artist.get_text()[:40], artist.get_fontsize()) for artist in texts if artist.get_fontsize() < 7.0 - 1e-6]
    require(not too_small, f"{bundle.name} contains text below 7 pt: {too_small[:5]}")
    require(all("Arial Narrow" in artist.get_fontfamily() for artist in texts), f"{bundle.name} contains text not assigned to Arial Narrow.")
    renderer = bundle.fig.canvas.get_renderer()
    tight_inches = bundle.fig.get_tightbbox(renderer)
    require(tight_inches is not None, f"{bundle.name} has no measurable tight bounding box.")
    tight_px = tight_inches.transformed(bundle.fig.dpi_scale_trans)
    canvas_w, canvas_h = bundle.fig.canvas.get_width_height()
    tolerance_px = 4.0
    require(tight_px.x0 >= -tolerance_px and tight_px.y0 >= -tolerance_px and tight_px.x1 <= canvas_w + tolerance_px and tight_px.y1 <= canvas_h + tolerance_px, f"{bundle.name} contains clipped or off-canvas artists: bbox={tight_px.bounds}, canvas={(canvas_w, canvas_h)}")


def export_bundle(bundle: FigureBundle, font_info: dict[str, str]) -> dict:
    _validate_figure_design(bundle)
    source_paths,inputs=_serialize_sources(bundle)
    canvas=bundle.fig.canvas
    canvas.draw()
    width_in,height_in=bundle.fig.get_size_inches()
    outputs={}
    asset_dir = asset_output_dir(bundle.name)
    asset_dir.mkdir(parents=True, exist_ok=True)
    for ext in ["svg","png","pdf"]:
        path=asset_dir/f"{bundle.name}.{ext}"
        tmp=path.with_name(path.name+".tmp")
        if ext=="pdf":
            metadata={"Creator":"Reproducible Python/matplotlib figure system","Title":bundle.name,"CreationDate":BUILD_EPOCH,"ModDate":BUILD_EPOCH}
        elif ext=="svg":
            metadata={"Creator":"Reproducible Python/matplotlib figure system","Title":bundle.name,"Date":BUILD_EPOCH.isoformat()}
        else:
            metadata={"Software":"Reproducible Python/matplotlib figure system","Title":bundle.name,"Creation Time":BUILD_EPOCH.isoformat()}
        kwargs={"metadata":metadata}
        if ext=="png": kwargs["dpi"]=600
        bundle.fig.savefig(tmp,format=ext,**kwargs)
        require(tmp.exists() and tmp.stat().st_size>1000,f"Failed to stage {path}")
        os.replace(tmp,path)
        require(path.exists() and path.stat().st_size>1000,f"Failed to export {path}")
        outputs[ext]={"path":str(path.relative_to(PROJECT)).replace("\\","/"),"bytes":path.stat().st_size,"sha256":sha256(path)}
    png_w=int(round(width_in*600));png_h=int(round(height_in*600))
    if bundle.name=="Graphical_Abstract":
        require(png_w>=2400 and png_h>=960 and 2.40<=png_w/png_h<=2.60,"Graphical abstract does not meet the 2.5:1 high-resolution requirement.")
    plt.close(bundle.fig)
    return {
        "asset":bundle.name,
        "width_mm":round(width_in*25.4,2),
        "height_mm":round(height_in*25.4,2),
        "png_width_px_nominal":png_w,
        "png_height_px_nominal":png_h,
        "dpi":600,
        "font_requested":"Arial Narrow",
        "font_resolved_family":font_info["family"],
        "font_resolved_path":font_info["regular_path"],
        "source_data":[str(path.relative_to(PROJECT)).replace("\\","/") for path in source_paths],
        "source_inputs":inputs,
        "outputs":outputs,
    }


def validate_outputs(manifest: list[dict]) -> None:
    require([m["asset"] for m in manifest]==ASSET_ORDER,"Asset order or count is wrong")
    require(len(manifest)==20,"The package must contain one graphical abstract, nine main figures, and ten supplementary figures.")
    for item in manifest:
        for ext in ["svg","png","pdf"]:
            path=PROJECT/item["outputs"][ext]["path"]
            require(path.exists() and path.stat().st_size==item["outputs"][ext]["bytes"],f"Output size check failed: {path}")
        svg=(PROJECT/item["outputs"]["svg"]["path"]).read_text(encoding="utf-8")
        require("Arial Narrow" in svg, f"Arial Narrow is not declared in {item['asset']} SVG")
        require(re.search(r"<image\b", svg, flags=re.IGNORECASE) is None, f"Raster image element found in {item['asset']} SVG")
        pdf_bytes=(PROJECT/item["outputs"]["pdf"]["path"]).read_bytes()
        require(re.search(rb"/Subtype\s*/Image\b", pdf_bytes) is None, f"Raster image object found in {item['asset']} PDF")
        for source_rel in item["source_data"]:
            source_path=PROJECT/source_rel
            require(source_path.exists() and source_path.stat().st_size>10,f"Missing panel source table: {source_path}")
    expected={str((asset_output_dir(name)/f"{name}.{ext}").relative_to(OUT)).replace("\\","/") for name in ASSET_ORDER for ext in ["svg","png","pdf"]}
    actual={str(p.relative_to(OUT)).replace("\\","/") for p in OUT.rglob("*") if p.is_file() and p.suffix.lower() in {".svg",".png",".pdf"}}
    require(actual==expected,f"Unexpected or missing scientific assets: expected={sorted(expected)}, actual={sorted(actual)}")


def write_documentation(bundles: list[FigureBundle],manifest: list[dict]) -> None:
    lines=["# Water Research figure captions and alt text","", "All figures are original, code-native vector artwork. Static SVG/PDF figures are authoritative; PNG files are 600 dpi review derivatives.",""]
    for b in bundles:
        lines.extend([f"## {b.name}","",b.caption,"",f"**Alt text.** {b.alt_text}",""])
    _atomic_write_text(OUT/"FIGURE_CAPTIONS_AND_ALT_TEXT.md","\n".join(lines)+"\n")
    payload={
        "reproducibility_epoch_utc":BUILD_EPOCH.isoformat(),
        "generator":"src/wr_make_figures.py",
        "python":sys.version,
        "platform":platform.platform(),
        "matplotlib":mpl.__version__,
        "pandas":pd.__version__,
        "numpy":np.__version__,
        "asset_count":len(manifest),
        "static_figures_authoritative":True,
        "panel_label_style":"letter_only",
        "assets":manifest,
    }
    _atomic_write_text(OUT/"generation_manifest.json",json.dumps(payload,indent=2,ensure_ascii=False)+"\n")
    rows=[]
    for m in manifest:
        for ext,meta in m["outputs"].items():
            rows.append({"asset":m["asset"],"format":ext,"path":meta["path"],"bytes":meta["bytes"],"sha256":meta["sha256"],"width_mm":m["width_mm"],"height_mm":m["height_mm"],"png_width_px_nominal":m["png_width_px_nominal"],"png_height_px_nominal":m["png_height_px_nominal"],"dpi_for_png":m["dpi"],"font":"Arial Narrow","source_data":" | ".join(m["source_data"]),"source_inputs":" | ".join(m["source_inputs"])})
    _atomic_write_csv(pd.DataFrame(rows),OUT/"generation_manifest.csv")
    readme=(
        "# Figure source data\n\n"
        "Each CSV contains the exact tabular values used to draw one named asset. The `source_panel` column identifies the panel and `source_input_path` records provenance. "
        "Some panels use deterministic summaries, such as rectangular density bins, instead of plotting identifiable row-level data.\n"
    )
    _atomic_write_text(SOURCE_OUT/"README.md",readme)


def _clean_known_outputs() -> None:
    """Remove only files this generator owns; never remove a directory tree."""
    for name in LEGACY_ASSET_ORDER:
        directory=asset_output_dir(name)
        for ext in ["svg","png","pdf"]:
            path=directory/f"{name}.{ext}"
            if path.exists():
                require(path.is_file() or path.is_symlink(),f"Refusing to replace non-file output: {path}")
                path.unlink()
        old_combined=SOURCE_OUT/f"{name}_source_data.csv"
        if old_combined.exists():
            require(old_combined.is_file() or old_combined.is_symlink(),f"Refusing to replace non-file source output: {old_combined}")
            old_combined.unlink()
        for panel in "abcdefghij":
            source=SOURCE_OUT/f"{name}{panel}_source.csv"
            if source.exists():
                require(source.is_file() or source.is_symlink(),f"Refusing to replace non-file source output: {source}")
                source.unlink()
    for path in [OUT/"FIGURE_CAPTIONS_AND_ALT_TEXT.md",OUT/"generation_manifest.json",OUT/"generation_manifest.csv",SOURCE_OUT/"README.md"]:
        if path.exists():
            require(path.is_file() or path.is_symlink(),f"Refusing to replace non-file documentation output: {path}")
            path.unlink()


def main() -> None:
    from wr_haa9_revision_figures import main as revision_main
    revision_main()


if __name__=="__main__":
    main()
