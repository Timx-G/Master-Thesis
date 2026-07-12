#!/usr/bin/env python3
"""Render the six deterministic visual assets for the 2026 midterm defense."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
import textwrap
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from defense_visual_content import (
    ALGO_NAMES,
    ALGORITHM_SELECTION_COLUMNS,
    ALGORITHM_SELECTION_ROWS,
    ASSET_SPECS,
    BACKGROUND_CONTENT,
    CANONICAL_EVIDENCE,
    DD_MCAA_FLOW,
    EAD_JSRA_FRAMEWORK,
    IEA_CAP_FLOW,
    PROGRESS_EVIDENCE,
    PROGRESS_WORK_PACKAGES,
)
from matplotlib import font_manager
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

WIDTH_IN = 16
HEIGHT_IN = 9
DPI = 150
WIDTH_PX = WIDTH_IN * DPI
HEIGHT_PX = HEIGHT_IN * DPI

WHITE = "#FFFFFF"
TEAL = "#173F42"
TEAL_MID = "#0F5B5D"
GREEN = "#2F7D58"
BRICK = "#8F2633"
INK = "#243437"
GRAY = "#697477"
GRAY_MID = "#AAB4B5"
GRAY_LIGHT = "#E8ECEC"
TEAL_LIGHT = "#E5F0EF"
GREEN_LIGHT = "#E7F1EB"
BRICK_LIGHT = "#F3E6E8"

SCRIPT_DIR = Path(__file__).resolve().parent
DELIVERABLE_DIR = SCRIPT_DIR.parent


def _font_family(font) -> str:
    for record in font["name"].names:
        if record.nameID == 1:
            try:
                return record.toUnicode()
            except UnicodeDecodeError:
                continue
    return ""


def _extract_sc_face(collection_path: Path, weight: str) -> Path:
    """Extract the SC face because Matplotlib otherwise selects TTC face zero."""
    from fontTools.ttLib import TTCollection

    cache_dir = Path(tempfile.gettempdir()) / "master-thesis-defense-fonts"
    cache_dir.mkdir(parents=True, exist_ok=True)
    source_digest = hashlib.sha256()
    source_digest.update(str(collection_path.resolve()).encode("utf-8"))
    with collection_path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            source_digest.update(chunk)
    cache_key = source_digest.hexdigest()[:16]
    extracted = cache_dir / f"NotoSansCJKSC-{weight}-{cache_key}.otf"
    if extracted.exists():
        return extracted

    collection = TTCollection(collection_path, lazy=False)
    try:
        for font in collection.fonts:
            if _font_family(font) == "Noto Sans CJK SC":
                font.save(extracted)
                return extracted
    finally:
        collection.close()
    raise RuntimeError(f"Noto Sans CJK SC face not found in {collection_path}")


def _fontconfig_match(weight: str) -> Path | None:
    style = "Bold" if weight == "Bold" else "Regular"
    try:
        result = subprocess.run(
            [
                "fc-match",
                "-f",
                "%{file}\n",
                f"Noto Sans CJK SC:style={style}",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    candidate = Path(result.stdout.splitlines()[0]) if result.stdout.strip() else None
    if candidate and candidate.is_file() and "noto" in candidate.name.lower():
        return candidate
    return None


def _locate_noto_font(weight: str) -> Path:
    filename = f"NotoSansCJK-{weight}.ttc"
    candidates = (
        DELIVERABLE_DIR / "font_deps/usr/share/fonts/opentype/noto" / filename,
        Path("/usr/share/fonts/opentype/noto") / filename,
        Path("/usr/local/share/fonts/opentype/noto") / filename,
        Path.home() / ".local/share/fonts/opentype/noto" / filename,
        Path.home() / ".fonts" / filename,
    )
    font_path = next((path for path in candidates if path.is_file()), None)
    if font_path is None:
        font_path = _fontconfig_match(weight)
    if font_path is None:
        raise RuntimeError(
            "Noto Sans CJK SC is required; install the font or restore defense_2026/font_deps"
        )
    if font_path.suffix.lower() == ".ttc":
        return _extract_sc_face(font_path, weight)
    return font_path


REGULAR_FONT_PATH = _locate_noto_font("Regular")
BOLD_FONT_PATH = _locate_noto_font("Bold")
font_manager.fontManager.addfont(REGULAR_FONT_PATH)
font_manager.fontManager.addfont(BOLD_FONT_PATH)

matplotlib.rcParams.update(
    {
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": WHITE,
        "savefig.edgecolor": WHITE,
    }
)


@lru_cache(maxsize=None)
def font(size: float, bold: bool = False) -> font_manager.FontProperties:
    path = BOLD_FONT_PATH if bold else REGULAR_FONT_PATH
    return font_manager.FontProperties(fname=path, size=size)


def wrap_text(value: str, width: int) -> str:
    return "\n".join(
        textwrap.wrap(
            value,
            width=width,
            break_long_words=True,
            break_on_hyphens=False,
        )
    )


def format_formulas(value: str) -> str:
    replacements = (
        ("(f_i^min, task_id)", r"$(f_i^{\min},\ \mathrm{task\_id})$"),
        ("T_i=d_i-a_i", "$T_i = d_i - a_i$"),
        ("alpha_i^min", r"$\alpha_i^{\min}$"),
        ("f_i^min", r"$f_i^{\min}$"),
    )
    for source, rendered in replacements:
        value = value.replace(source, rendered)
    return value


def new_canvas(title: str):
    fig, ax = plt.subplots(figsize=(WIDTH_IN, HEIGHT_IN), dpi=DPI)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor(WHITE)
    ax.set_xlim(0, WIDTH_IN)
    ax.set_ylim(0, HEIGHT_IN)
    ax.axis("off")
    ax.text(
        0.62,
        8.48,
        title,
        fontproperties=font(25, True),
        color=TEAL,
        ha="left",
        va="center",
    )
    ax.add_patch(Rectangle((0.62, 8.08), 1.15, 0.07, color=TEAL_MID, linewidth=0))
    ax.add_patch(Rectangle((1.77, 8.08), 13.61, 0.025, color=GRAY_LIGHT, linewidth=0))
    return fig, ax


def panel(ax, x, y, width, height, *, fill=WHITE, edge=GRAY_MID, radius=0.12, lw=1.3):
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.018,rounding_size={radius}",
        facecolor=fill,
        edgecolor=edge,
        linewidth=lw,
    )
    ax.add_patch(patch)
    return patch


def arrow(ax, start, end, *, color=TEAL_MID, lw=2.1, style="-|>", connection="arc3"):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=14,
        linewidth=lw,
        color=color,
        connectionstyle=connection,
        shrinkA=2,
        shrinkB=2,
        zorder=5,
    )
    ax.add_patch(patch)
    return patch


def numbered_step(
    ax,
    x,
    y,
    width,
    height,
    number,
    value,
    *,
    accent=TEAL_MID,
    fill=WHITE,
    text_size=14,
    wrap_width=18,
):
    frame = panel(ax, x, y, width, height, fill=fill, edge=accent, radius=0.1, lw=1.5)
    ax.add_patch(Circle((x + 0.37, y + height - 0.38), 0.21, color=accent, linewidth=0))
    ax.text(
        x + 0.37,
        y + height - 0.38,
        f"{number:02d}",
        fontproperties=font(10.5, True),
        color=WHITE,
        ha="center",
        va="center",
    )
    text_artist = ax.text(
        x + 0.28,
        y + height / 2 - 0.08,
        format_formulas(wrap_text(value, wrap_width)),
        fontproperties=font(text_size, False),
        color=INK,
        ha="left",
        va="center",
        linespacing=1.35,
    )
    if not hasattr(ax, "_numbered_steps"):
        ax._numbered_steps = []
    ax._numbered_steps.append((frame, text_artist))
    return frame, text_artist


def footer_note(ax, value: str, *, color=GRAY, fill="#F6F8F8"):
    panel(ax, 0.62, 0.34, 14.76, 0.72, fill=fill, edge=GRAY_LIGHT, radius=0.08, lw=1)
    ax.add_patch(Rectangle((0.62, 0.34), 0.08, 0.72, color=color, linewidth=0))
    ax.text(
        0.92,
        0.70,
        value,
        fontproperties=font(13.3, False),
        color=INK,
        ha="left",
        va="center",
    )


def render_background():
    spec = ASSET_SPECS["01_background"]
    fig, ax = new_canvas(spec["title"])
    columns = BACKGROUND_CONTENT["columns"]
    xs = (0.62, 5.72, 10.82)
    width = 4.55
    fills = (TEAL_LIGHT, BRICK_LIGHT, GREEN_LIGHT)
    accents = (TEAL_MID, BRICK, GREEN)

    for index, (column, x, fill, accent) in enumerate(zip(columns, xs, fills, accents)):
        panel(ax, x, 1.42, width, 6.2, fill=fill, edge=accent, radius=0.12, lw=1.5)
        ax.add_patch(Rectangle((x, 6.72), width, 0.9, color=accent, linewidth=0))
        ax.text(
            x + 0.34,
            7.17,
            column["heading"],
            fontproperties=font(19, True),
            color=WHITE,
            ha="left",
            va="center",
        )
        if index == 1:
            item_ys = (6.17, 5.56, 4.95, 4.34)
        elif index == 2:
            item_ys = (5.84, 3.18)
        else:
            item_ys = (6.05, 5.18, 4.31, 3.44)
        for item, y in zip(column["items"], item_ys):
            ax.add_patch(Circle((x + 0.42, y), 0.07, color=accent, linewidth=0))
            ax.text(
                x + 0.67,
                y,
                wrap_text(item, 17 if index != 2 else 16),
                fontproperties=font(15.2 if index != 2 else 16, index == 2),
                color=INK,
                ha="left",
                va="center",
                linespacing=1.25,
            )

    # A static association/interference schematic: links indicate coupling, not flight paths.
    for drone_x in (6.72, 9.26):
        ax.add_patch(Rectangle((drone_x - 0.18, 2.75), 0.36, 0.18, color=TEAL, linewidth=0))
        ax.plot([drone_x - 0.65, drone_x + 0.65], [3.03, 3.03], color=TEAL, lw=2)
        ax.plot([drone_x - 0.42, drone_x - 0.18], [2.82, 2.92], color=TEAL, lw=2)
        ax.plot([drone_x + 0.18, drone_x + 0.42], [2.92, 2.82], color=TEAL, lw=2)
        for rotor_x in (drone_x - 0.65, drone_x + 0.65):
            ax.add_patch(Circle((rotor_x, 3.03), 0.12, fill=False, edgecolor=TEAL, lw=1.5))
    user_points = ((6.32, 1.92), (7.55, 1.75), (8.63, 1.8), (9.66, 2.02))
    for user_x, user_y in user_points:
        ax.add_patch(Circle((user_x, user_y + 0.16), 0.12, color=INK, linewidth=0))
        ax.plot([user_x, user_x], [user_y - 0.18, user_y + 0.03], color=INK, lw=1.6)
        ax.plot([user_x - 0.17, user_x + 0.17], [user_y - 0.02, user_y - 0.02], color=INK, lw=1.4)
    ax.plot([6.72, 6.32], [2.7, 2.18], color=TEAL_MID, lw=1.5)
    ax.plot([6.72, 7.55], [2.7, 2.04], color=TEAL_MID, lw=1.5)
    ax.plot([9.26, 8.63], [2.7, 2.08], color=TEAL_MID, lw=1.5)
    ax.plot([9.26, 9.66], [2.7, 2.27], color=TEAL_MID, lw=1.5)
    ax.plot([6.98, 8.98], [2.91, 2.91], color=BRICK, lw=2.2, linestyle=(0, (4, 3)))

    arrow(ax, (5.29, 4.47), (5.62, 4.47), color=TEAL_MID, lw=2.4)
    arrow(ax, (10.39, 4.47), (10.72, 4.47), color=TEAL_MID, lw=2.4)
    footer_note(ax, BACKGROUND_CONTENT["boundary"], color=BRICK)
    return fig


def render_ead_jsra_framework():
    spec = ASSET_SPECS["02_ead_jsra_framework"]
    fig, ax = new_canvas(spec["title"])
    steps = EAD_JSRA_FRAMEWORK["steps"]

    numbered_step(ax, 0.50, 4.25, 2.1, 2.2, 1, steps[0], fill=TEAL_LIGHT, wrap_width=11)
    numbered_step(
        ax,
        2.95,
        4.25,
        2.10,
        2.2,
        2,
        steps[1],
        fill=TEAL_LIGHT,
        text_size=12.8,
        wrap_width=12,
    )
    arrow(ax, (2.62, 5.35), (2.91, 5.35))

    panel(ax, 5.22, 3.70, 6.18, 3.38, fill="#F7FAFA", edge=TEAL_MID, radius=0.13, lw=2)
    ax.add_patch(Rectangle((5.22, 6.48), 6.18, 0.60, color=TEAL, linewidth=0))
    ax.text(
        5.52,
        6.78,
        ALGO_NAMES["full"]["display"],
        fontproperties=font(15, True),
        color=WHITE,
        ha="left",
        va="center",
    )
    inner_xs = (5.48, 7.43, 9.38)
    inner_fills = (TEAL_LIGHT, BRICK_LIGHT, GREEN_LIGHT)
    inner_accents = (TEAL_MID, BRICK, GREEN)
    for number, step, x, fill, accent in zip(
        (3, 4, 5), steps[2:5], inner_xs, inner_fills, inner_accents
    ):
        numbered_step(
            ax,
            x,
            4.18,
            1.76,
            1.85,
            number,
            step,
            fill=fill,
            accent=accent,
            text_size=12.6,
            wrap_width=10,
        )
    arrow(ax, (7.26, 5.10), (7.39, 5.10), color=BRICK)
    arrow(ax, (9.21, 5.10), (9.34, 5.10), color=GREEN)
    arrow(ax, (5.07, 5.35), (5.18, 5.35))

    numbered_step(
        ax,
        11.72,
        4.25,
        1.72,
        2.2,
        6,
        steps[5],
        fill=GREEN_LIGHT,
        accent=GREEN,
        text_size=11.8,
        wrap_width=8,
    )
    arrow(ax, (11.42, 5.35), (11.68, 5.35), color=GREEN)

    diamond = Polygon(
        ((14.74, 6.43), (15.94, 5.35), (14.74, 4.27), (13.54, 5.35)),
        closed=True,
        facecolor=WHITE,
        edgecolor=TEAL_MID,
        linewidth=1.8,
    )
    ax.add_patch(diamond)
    ax.text(
        14.74,
        5.35,
        wrap_text(steps[6], 7),
        fontproperties=font(11.5, True),
        color=INK,
        ha="center",
        va="center",
        linespacing=1.25,
    )
    arrow(ax, (13.46, 5.35), (13.67, 5.35))
    arrow(
        ax,
        (14.74, 4.22),
        (6.40, 3.62),
        color=TEAL_MID,
        lw=1.8,
        connection="arc3,rad=-0.08",
    )
    ax.text(
        10.25,
        3.23,
        EAD_JSRA_FRAMEWORK["loop"],
        fontproperties=font(12.2, False),
        color=TEAL_MID,
        ha="center",
        va="center",
    )

    evidence = EAD_JSRA_FRAMEWORK["evidence_chain"]
    ax.plot([1.25, 14.75], [1.86, 1.86], color=GRAY_MID, lw=1.4)
    for index, (x, value) in enumerate(zip((1.25, 4.62, 8.00, 11.38, 14.75), evidence), 1):
        ax.add_patch(Circle((x, 1.86), 0.20, color=TEAL if index < 4 else GREEN, linewidth=0))
        ax.text(
            x,
            1.39,
            wrap_text(value, 12),
            fontproperties=font(12.4, index >= 4),
            color=INK,
            ha="center",
            va="center",
            linespacing=1.15,
        )
    footer_note(ax, EAD_JSRA_FRAMEWORK["boundary"], color=BRICK, fill=BRICK_LIGHT)
    return fig


def render_iea_cap_flow():
    spec = ASSET_SPECS["03_iea_cap_flow"]
    fig, ax = new_canvas(spec["title"])
    steps = IEA_CAP_FLOW["steps"]

    top_xs = (0.48, 4.30, 8.12, 11.94)
    widths = (3.34, 3.34, 3.34, 3.58)
    for number, x, width, step in zip(range(1, 5), top_xs, widths, steps[:4]):
        accent = BRICK if number in (2, 4) else TEAL_MID
        fill = BRICK_LIGHT if number in (2, 4) else TEAL_LIGHT
        numbered_step(
            ax,
            x,
            5.40,
            width,
            2.02,
            number,
            step,
            fill=fill,
            accent=accent,
            text_size=13.1,
            wrap_width=30 if number == 3 else 18,
        )
        if number < 4:
            arrow(ax, (x + width + 0.03, 6.41), (top_xs[number] - 0.03, 6.41))

    numbered_step(
        ax,
        11.94,
        2.42,
        3.58,
        2.08,
        5,
        steps[4],
        fill=BRICK_LIGHT,
        accent=BRICK,
        text_size=12.8,
        wrap_width=19,
    )
    arrow(ax, (13.73, 5.35), (13.73, 4.54), color=BRICK)

    diamond = Polygon(
        ((9.80, 4.42), (11.20, 3.46), (9.80, 2.50), (8.40, 3.46)),
        closed=True,
        facecolor=WHITE,
        edgecolor=TEAL_MID,
        linewidth=1.8,
    )
    ax.add_patch(diamond)
    ax.text(
        9.80,
        3.46,
        wrap_text(IEA_CAP_FLOW["acceptance"], 11),
        fontproperties=font(13.2, True),
        color=INK,
        ha="center",
        va="center",
        linespacing=1.2,
    )
    arrow(ax, (11.88, 3.46), (11.24, 3.46), color=BRICK)

    numbered_step(
        ax,
        4.40,
        2.42,
        3.28,
        2.08,
        6,
        steps[5],
        fill=GREEN_LIGHT,
        accent=GREEN,
        text_size=14.0,
        wrap_width=16,
    )
    arrow(ax, (8.36, 3.46), (7.72, 3.46), color=GREEN)
    arrow(
        ax,
        (5.55, 4.54),
        (9.64, 5.34),
        color=GREEN,
        lw=1.8,
        connection="arc3,rad=-0.22",
    )

    numbered_step(
        ax,
        0.48,
        2.42,
        3.60,
        2.08,
        7,
        steps[6],
        fill="#F6F8F8",
        accent=GRAY,
        text_size=13.7,
        wrap_width=11,
    )
    arrow(ax, (8.95, 2.75), (3.80, 2.75), color=GRAY, lw=1.8, connection="arc3,rad=0.12")
    footer_note(ax, IEA_CAP_FLOW["boundary"], color=BRICK, fill=BRICK_LIGHT)
    return fig


def render_dd_mcaa_flow():
    spec = ASSET_SPECS["04_dd_mcaa_flow"]
    fig, ax = new_canvas(spec["title"])
    steps = DD_MCAA_FLOW["steps"]
    top_xs = (0.55, 4.37, 8.19, 12.01)
    bottom_xs = (12.01, 8.19, 4.37)

    for number, (x, step) in enumerate(zip(top_xs, steps[:4]), 1):
        accent = BRICK if number == 4 else TEAL_MID
        fill = BRICK_LIGHT if number == 4 else TEAL_LIGHT
        numbered_step(
            ax,
            x,
            5.42,
            3.44,
            1.92,
            number,
            step,
            fill=fill,
            accent=accent,
            text_size=13.5,
            wrap_width=16 if number == 4 else 18,
        )
        if number < 4:
            arrow(ax, (x + 3.47, 6.38), (top_xs[number] - 0.03, 6.38))
    arrow(ax, (13.73, 5.38), (13.73, 4.66), color=BRICK)

    for offset, (x, step) in enumerate(zip(bottom_xs, steps[4:]), 5):
        accent = GREEN if offset in (5, 6) else TEAL_MID
        fill = GREEN_LIGHT if offset in (5, 6) else TEAL_LIGHT
        numbered_step(
            ax,
            x,
            2.68,
            3.44,
            1.92,
            offset,
            step,
            fill=fill,
            accent=accent,
            text_size=13.5,
            wrap_width=30 if offset == 5 else 18,
        )
        if offset < 7:
            next_x = bottom_xs[offset - 4]
            arrow(ax, (x - 0.03, 3.64), (next_x + 3.47, 3.64), color=GREEN)

    panel(ax, 0.62, 1.46, 7.18, 0.70, fill=TEAL_LIGHT, edge=GRAY_LIGHT, radius=0.08, lw=1)
    ax.add_patch(Rectangle((0.62, 1.46), 0.08, 0.70, color=TEAL_MID, linewidth=0))
    ax.text(
        0.91,
        1.81,
        DD_MCAA_FLOW["theorem_boundary"],
        fontproperties=font(13.2, True),
        color=INK,
        ha="left",
        va="center",
    )
    panel(ax, 8.18, 1.46, 7.20, 0.70, fill=GREEN_LIGHT, edge=GRAY_LIGHT, radius=0.08, lw=1)
    ax.add_patch(Rectangle((8.18, 1.46), 0.08, 0.70, color=GREEN, linewidth=0))
    ax.text(
        8.47,
        1.81,
        DD_MCAA_FLOW["ownership"],
        fontproperties=font(13.2, True),
        color=INK,
        ha="left",
        va="center",
    )
    footer_note(ax, DD_MCAA_FLOW["excluded_scope"], color=BRICK, fill=BRICK_LIGHT)
    return fig


def render_algorithm_selection_table():
    spec = ASSET_SPECS["table_algorithm_selection"]
    fig, ax = new_canvas(spec["title"])
    x0 = 0.55
    widths = (2.35, 3.58, 4.18, 4.79)
    header_y = 7.18
    header_h = 0.72
    row_h = 1.17
    keys = ("candidate", "strength", "limitation", "role")

    x = x0
    for heading, width in zip(ALGORITHM_SELECTION_COLUMNS, widths):
        ax.add_patch(Rectangle((x, header_y), width, header_h, color=TEAL, linewidth=0))
        ax.text(
            x + 0.19,
            header_y + header_h / 2,
            heading,
            fontproperties=font(14.5, True),
            color=WHITE,
            ha="left",
            va="center",
        )
        x += width

    for row_index, row in enumerate(ALGORITHM_SELECTION_ROWS):
        y = header_y - (row_index + 1) * row_h
        highlighted = row["candidate"] == ALGO_NAMES["full"]["acronym"]
        fill = GREEN_LIGHT if highlighted else (WHITE if row_index % 2 == 0 else "#F6F8F8")
        x = x0
        for column_index, (key, width) in enumerate(zip(keys, widths)):
            edge = GREEN if highlighted else GRAY_LIGHT
            ax.add_patch(
                Rectangle((x, y), width, row_h, facecolor=fill, edgecolor=edge, linewidth=1.0)
            )
            value = row[key]
            wrap_widths = (10, 17, 20, 23)
            ax.text(
                x + 0.19,
                y + row_h / 2,
                wrap_text(value, wrap_widths[column_index]),
                fontproperties=font(13.0, column_index == 0 or highlighted),
                color=TEAL if highlighted else INK,
                ha="left",
                va="center",
                linespacing=1.25,
            )
            x += width

    ax.add_patch(Rectangle((x0, 0.73), sum(widths), 0.09, color=TEAL_MID, linewidth=0))
    ax.text(
        x0,
        0.42,
        CANONICAL_EVIDENCE["enumeration"],
        fontproperties=font(12.7, False),
        color=GRAY,
        ha="left",
        va="center",
    )
    return fig


def render_progress():
    spec = ASSET_SPECS["05_progress"]
    fig, ax = new_canvas(spec["title"])

    completed = PROGRESS_WORK_PACKAGES[:4]
    ongoing = PROGRESS_WORK_PACKAGES[4:]
    completed_xs = (0.62, 4.45, 8.28, 12.11)
    for index, (package, x) in enumerate(zip(completed, completed_xs), 1):
        panel(ax, x, 5.07, 3.27, 2.48, fill=GREEN_LIGHT, edge=GREEN, radius=0.10, lw=1.5)
        ax.add_patch(Rectangle((x, 6.92), 3.27, 0.63, color=GREEN, linewidth=0))
        ax.text(
            x + 0.22,
            7.235,
            package["status"],
            fontproperties=font(13.2, True),
            color=WHITE,
            ha="left",
            va="center",
        )
        ax.add_patch(Circle((x + 2.88, 7.235), 0.14, facecolor=WHITE, edgecolor=WHITE, lw=1))
        ax.text(
            x + 2.88,
            7.235,
            "✓",
            fontproperties=font(11, True),
            color=GREEN,
            ha="center",
            va="center",
        )
        ax.text(
            x + 0.25,
            6.42,
            package["name"],
            fontproperties=font(17, True),
            color=TEAL,
            ha="left",
            va="center",
        )
        ax.text(
            x + 0.25,
            5.72,
            wrap_text(package["detail"], 17),
            fontproperties=font(12.8, False),
            color=INK,
            ha="left",
            va="center",
            linespacing=1.25,
        )

    ongoing_xs = (0.62, 8.28)
    for package, x in zip(ongoing, ongoing_xs):
        panel(ax, x, 2.72, 7.10, 1.75, fill=TEAL_LIGHT, edge=TEAL_MID, radius=0.10, lw=1.5)
        ax.add_patch(Rectangle((x, 2.72), 0.12, 1.75, color=TEAL_MID, linewidth=0))
        ax.text(
            x + 0.38,
            4.03,
            package["status"],
            fontproperties=font(12.8, True),
            color=TEAL_MID,
            ha="left",
            va="center",
        )
        ax.text(
            x + 0.38,
            3.54,
            package["name"],
            fontproperties=font(18, True),
            color=TEAL,
            ha="left",
            va="center",
        )
        ax.text(
            x + 2.25,
            3.54,
            wrap_text(package["detail"], 24),
            fontproperties=font(13.2, False),
            color=INK,
            ha="left",
            va="center",
            linespacing=1.2,
        )

    evidence_xs = (0.72, 3.26, 5.80, 8.34, 10.88, 13.42)
    for value, x in zip(PROGRESS_EVIDENCE, evidence_xs):
        panel(ax, x, 0.76, 2.18, 1.28, fill="#F6F8F8", edge=GRAY_LIGHT, radius=0.08, lw=1)
        ax.add_patch(Rectangle((x, 0.76), 2.18, 0.08, color=TEAL_MID, linewidth=0))
        ax.text(
            x + 1.09,
            1.40,
            wrap_text(value, 11),
            fontproperties=font(12.5, True),
            color=INK,
            ha="center",
            va="center",
            linespacing=1.15,
        )
    return fig


RENDERERS = (
    ("01_background", render_background),
    ("02_ead_jsra_framework", render_ead_jsra_framework),
    ("03_iea_cap_flow", render_iea_cap_flow),
    ("04_dd_mcaa_flow", render_dd_mcaa_flow),
    ("05_progress", render_progress),
    ("table_algorithm_selection", render_algorithm_selection_table),
)


def save_asset(fig, stem: str, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.savefig(
        png_path,
        format="png",
        dpi=DPI,
        facecolor=WHITE,
        edgecolor=WHITE,
        metadata={"Software": "Master-Thesis defense visual renderer"},
    )
    fig.savefig(
        pdf_path,
        format="pdf",
        facecolor=WHITE,
        edgecolor=WHITE,
        metadata={
            "Creator": "Master-Thesis defense visual renderer",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(fig)
    return png_path, pdf_path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _render_release(release_dir: Path) -> dict:
    output_paths = []
    for stem, renderer in RENDERERS:
        output_paths.extend(save_asset(renderer(), stem, release_dir))
    filenames = [path.name for path in output_paths]
    manifest = {
        "schema_version": 1,
        "dimensions": {
            "width_px": WIDTH_PX,
            "height_px": HEIGHT_PX,
            "dpi": DPI,
        },
        "outputs": filenames,
        "algorithm_acronyms": [item["acronym"] for item in ALGO_NAMES.values()],
        "canonical_evidence_version": CANONICAL_EVIDENCE["version"],
        "sha256": {path.name: sha256(path) for path in output_paths},
    }
    manifest_path = release_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def generate_all(output_dir: Path) -> dict:
    output_dir = output_dir.resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}-staging-", dir=output_dir.parent)
    )
    release_dir = staging_root / "release"
    release_dir.mkdir()
    backup_root = None
    preserve_backup = False
    try:
        manifest = _render_release(release_dir)
        backup_root = Path(
            tempfile.mkdtemp(prefix=f".{output_dir.name}-backup-", dir=output_dir.parent)
        )
        backup_dir = backup_root / "previous"
        had_previous_release = output_dir.exists()
        if had_previous_release:
            output_dir.replace(backup_dir)
        try:
            release_dir.replace(output_dir)
        except BaseException:
            if had_previous_release and backup_dir.exists() and not output_dir.exists():
                try:
                    backup_dir.replace(output_dir)
                except BaseException:
                    preserve_backup = True
            raise
        return manifest
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)
        if backup_root is not None and not preserve_backup:
            shutil.rmtree(backup_root, ignore_errors=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DELIVERABLE_DIR / "visuals",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_all(args.output.resolve())


if __name__ == "__main__":
    main()
