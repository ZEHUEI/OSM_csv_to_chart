from __future__ import annotations

import io
import re

import matplotlib

# Required for servers such as Render where there is no GUI/display.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter


def clean_number(value: object) -> float:
    if pd.isna(value):
        return np.nan

    text = str(value).strip()

    if not text:
        return np.nan

    text = re.sub(r"[^0-9.\-]", "", text)

    try:
        return float(text)
    except ValueError:
        return np.nan


def find_column(
    columns: list[str],
    required_words: tuple[str, ...],
) -> str:
    for column in columns:
        normalised = re.sub(
            r"\s+",
            "",
            str(column).lower(),
        )

        if all(
            word.lower().replace(" ", "") in normalised
            for word in required_words
        ):
            return column

    raise ValueError(
        f"Could not find a column containing: "
        f"{', '.join(required_words)}. "
        f"Available columns: {columns}"
    )


def gradient_barh(
    ax,
    y,
    width,
    height,
    left,
    start_colour,
    end_colour,
):
    if pd.isna(width) or width <= 0:
        return

    rectangle = Rectangle(
        (left, y - height / 2),
        width,
        height,
        facecolor="none",
        edgecolor="none",
    )

    ax.add_patch(rectangle)

    gradient = np.linspace(0, 1, 256).reshape(1, -1)

    colour_map = LinearSegmentedColormap.from_list(
        f"gradient_{start_colour}_{end_colour}_{y}_{left}",
        [start_colour, end_colour],
    )

    image = ax.imshow(
        gradient,
        extent=(
            left,
            left + width,
            y - height / 2,
            y + height / 2,
        ),
        aspect="auto",
        cmap=colour_map,
        interpolation="bicubic",
        zorder=3,
    )

    image.set_clip_path(rectangle)


def add_header_inside_bar(
    ax,
    text,
    y,
    bar_width,
    maximum,
    colour="white",
):
    """
    Main bar headers always remain inside their own bar.
    The font shrinks for shorter bars.
    """
    ratio = bar_width / maximum if maximum > 0 else 0

    if ratio >= 0.35:
        fontsize = 11
    elif ratio >= 0.22:
        fontsize = 9
    elif ratio >= 0.14:
        fontsize = 7.5
    else:
        fontsize = 6

    x = min(
        maximum * 0.008,
        bar_width * 0.04,
    )

    ax.text(
        x,
        y,
        text,
        va="center",
        ha="left",
        fontsize=fontsize,
        fontweight="bold",
        color=colour,
        zorder=8,
        clip_on=True,
    )


def create_chart_from_csv_bytes(csv_bytes: bytes) -> io.BytesIO:
    """
    Read the uploaded CSV, create the chart, and return an in-memory PNG.
    """

    try:
        df = pd.read_csv(io.BytesIO(csv_bytes))
    except Exception as exc:
        raise ValueError(f"Could not read CSV: {exc}") from exc

    if df.empty:
        raise ValueError("The CSV contains no data rows.")

    if len(df.columns) < 4:
        raise ValueError(
            "CSV needs at least four columns: "
            "the title/category column plus 2025 actual, "
            "2026 actual, and 2026 target."
        )

    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    # Column 1 header is the chart title.
    # Values beneath it contain the main rows and E/W helper rows.
    label_column = df.columns[0]
    chart_title = label_column

    if not chart_title or chart_title.lower().startswith("unnamed:"):
        chart_title = "Membership"

    metric_columns = list(df.columns[1:])

    col_previous = find_column(
        metric_columns,
        ("2025h1", "actual"),
    )

    col_current = find_column(
        metric_columns,
        ("2026h1", "actual"),
    )

    col_target = find_column(
        metric_columns,
        ("2026h1", "target"),
    )

    for column in (
        col_previous,
        col_current,
        col_target,
    ):
        df[column] = df[column].apply(clean_number)

    df[label_column] = (
        df[label_column]
        .astype(str)
        .str.strip()
    )

    df = df.set_index(label_column)

    # Main category rows have all three values.
    # Helper rows such as O E / O W only supply the 2025 breakdown
    # and therefore do not appear as separate chart categories.
    summary_mask = (
        df[col_previous].notna()
        & df[col_current].notna()
        & df[col_target].notna()
    )

    categories = df.index[summary_mask].tolist()

    if not categories:
        raise ValueError(
            "No main category rows found. "
            "A main row needs 2025 Actual, 2026 Actual, and 2026 Target."
        )

    # Preserve the behavior from the original chart:
    # Label, Publishing, Overall in CSV -> Overall, Publishing, Label on chart.
    categories = categories[::-1]

    records = []

    for category in categories:
        category_name = str(category).strip()

        if not category_name:
            continue

        first_letter = category_name[0].upper()

        existing_row = f"{first_letter} E"
        withdrawn_row = f"{first_letter} W"

        previous = df.at[category, col_previous]
        current = df.at[category, col_current]
        target = df.at[category, col_target]

        if existing_row in df.index:
            existing = df.at[existing_row, col_previous]
        else:
            existing = np.nan

        if withdrawn_row in df.index:
            withdrawn = df.at[withdrawn_row, col_previous]
        else:
            withdrawn = np.nan

        if pd.isna(existing) and not pd.isna(withdrawn):
            existing = max(previous - withdrawn, 0)

        elif pd.isna(withdrawn) and not pd.isna(existing):
            withdrawn = max(previous - existing, 0)

        elif pd.isna(existing) and pd.isna(withdrawn):
            existing = previous
            withdrawn = 0

        records.append(
            {
                "category": category_name,
                "previous": float(previous),
                "current": float(current),
                "target": float(target),
                "existing": float(existing),
                "withdrawn": float(withdrawn),
            }
        )

    if not records:
        raise ValueError("No usable chart records were found.")

    maximum = max(
        max(
            record["previous"],
            record["current"],
            record["target"],
        )
        for record in records
    )

    if maximum <= 0:
        raise ValueError("Chart values must contain at least one positive number.")

    figure_height = max(
        7,
        len(records) * 2.4,
    )

    fig, ax = plt.subplots(
        figsize=(14, figure_height)
    )

    normal_bar_height = 0.36
    previous_bar_height = 0.48

    # The three bars in a category touch.
    stack_height = (
        normal_bar_height
        + normal_bar_height
        + previous_bar_height
    )

    # White gap appears only AFTER the green bar / full category group.
    white_gap_after_group = 0.65
    group_gap = stack_height + white_gap_after_group

    # Tiny overlap prevents anti-aliasing from showing a hairline seam
    # between red -> blue -> green.
    overlap = 0.012

    colours = {
        "target": (
            "#c40000",
            "#ff1a1a",
        ),
        "current": (
            "#2877bd",
            "#6db5eb",
        ),
        "existing": (
            "#11b711",
            "#36db36",
        ),
        "withdrawn": (
            "#74e574",
            "#b4ffb4",
        ),
    }

    y_tick_positions = []
    y_tick_labels = []

    for group_number, record in enumerate(records):
        group_top = group_number * group_gap

        # Exact touching layout:
        #
        # RED
        # BLUE
        # GREEN
        # <white gap>
        target_y = (
            group_top
            + normal_bar_height / 2
        )

        current_y = (
            group_top
            + normal_bar_height
            + normal_bar_height / 2
        )

        previous_y = (
            group_top
            + normal_bar_height * 2
            + previous_bar_height / 2
        )

        group_label_y = (
            group_top
            + stack_height / 2
        )

        category = record["category"]
        previous = record["previous"]
        current = record["current"]
        target = record["target"]
        existing = record["existing"]
        withdrawn = record["withdrawn"]

        # ====================================================
        # TARGET - RED
        # ====================================================

        gradient_barh(
            ax,
            target_y,
            target,
            normal_bar_height + overlap,
            0,
            colours["target"][0],
            colours["target"][1],
        )

        add_header_inside_bar(
            ax=ax,
            text=col_target,
            y=target_y,
            bar_width=target,
            maximum=maximum,
            colour="white",
        )

        # ====================================================
        # CURRENT - BLUE
        # ====================================================

        gradient_barh(
            ax,
            current_y,
            current,
            normal_bar_height + overlap,
            0,
            colours["current"][0],
            colours["current"][1],
        )

        add_header_inside_bar(
            ax=ax,
            text=col_current,
            y=current_y,
            bar_width=current,
            maximum=maximum,
            colour="white",
        )

        # ====================================================
        # PREVIOUS - GREEN / LIGHT GREEN
        # ====================================================

        gradient_barh(
            ax,
            previous_y,
            existing,
            previous_bar_height + overlap,
            0,
            colours["existing"][0],
            colours["existing"][1],
        )

        gradient_barh(
            ax,
            previous_y,
            withdrawn,
            previous_bar_height + overlap,
            existing,
            colours["withdrawn"][0],
            colours["withdrawn"][1],
        )

        # Main 2025 header ALWAYS stays inside.
        add_header_inside_bar(
            ax=ax,
            text=col_previous,
            y=previous_y,
            bar_width=previous,
            maximum=maximum,
            colour="black",
        )

        # ====================================================
        # EXISTING MEMBERS
        # ====================================================

        existing_ratio = existing / maximum

        if existing > 0:
            if existing_ratio >= 0.20:
                ax.text(
                    existing * 0.62,
                    previous_y,
                    f"Existing Members\nRM{existing:,.0f}",
                    va="center",
                    ha="center",
                    fontsize=8.5,
                    fontweight="bold",
                    color="black",
                    zorder=8,
                )
            else:
                ax.annotate(
                    f"Existing Members\nRM{existing:,.0f}",
                    xy=(
                        existing * 0.65,
                        previous_y,
                    ),
                    xytext=(
                        existing + maximum * 0.08,
                        previous_y - 0.55,
                    ),
                    va="center",
                    ha="left",
                    fontsize=8.5,
                    fontweight="bold",
                    arrowprops={
                        "arrowstyle": "->",
                        "linewidth": 1.2,
                        "color": "#24667a",
                    },
                    zorder=9,
                )

        # ====================================================
        # WITHDRAWN MEMBERS
        # ====================================================

        if withdrawn > 0:
            withdrawn_ratio = withdrawn / maximum

            if withdrawn_ratio >= 0.10:
                ax.text(
                    existing + withdrawn / 2,
                    previous_y,
                    f"Withdrawn Members\nRM{withdrawn:,.0f}",
                    va="center",
                    ha="center",
                    fontsize=8,
                    fontweight="bold",
                    color="black",
                    zorder=8,
                )
            else:
                ax.annotate(
                    f"Withdrawn Members\nRM{withdrawn:,.0f}",
                    xy=(
                        existing + withdrawn / 2,
                        previous_y,
                    ),
                    xytext=(
                        min(
                            previous + maximum * 0.07,
                            maximum * 1.08,
                        ),
                        previous_y + 0.48,
                    ),
                    va="center",
                    ha="left",
                    fontsize=8,
                    fontweight="bold",
                    arrowprops={
                        "arrowstyle": "->",
                        "linewidth": 1.2,
                        "color": "#24667a",
                    },
                    zorder=9,
                )

        # ====================================================
        # TOTAL VALUES AT BAR ENDS
        # ====================================================

        value_padding = maximum * 0.012

        ax.text(
            target + value_padding,
            target_y,
            f"{target:,.0f}",
            va="center",
            ha="left",
            fontsize=10,
            color="#444444",
        )

        ax.text(
            current + value_padding,
            current_y,
            f"{current:,.0f}",
            va="center",
            ha="left",
            fontsize=10,
            color="#444444",
        )

        ax.text(
            previous + value_padding,
            previous_y,
            f"{previous:,.0f}",
            va="center",
            ha="left",
            fontsize=10,
            color="#444444",
        )

        y_tick_positions.append(group_label_y)
        y_tick_labels.append(category)

    # ========================================================
    # Y AXIS
    # ========================================================

    ax.set_yticks(y_tick_positions)
    ax.set_yticklabels(
        y_tick_labels,
        fontsize=13,
    )

    ax.tick_params(
        axis="y",
        length=0,
        pad=12,
    )

    ax.set_ylabel("")

    # ========================================================
    # TITLE
    # ========================================================

    ax.set_title(
        chart_title,
        fontsize=17,
        fontweight="bold",
        pad=20,
    )

    # ========================================================
    # X AXIS
    # ========================================================

    ax.set_xlim(
        -maximum * 0.015,
        maximum * 1.18,
    )

    chart_bottom = (
        (len(records) - 1) * group_gap
        + stack_height
    )

    ax.set_ylim(
        -0.30,
        chart_bottom + 0.75,
    )

    ax.invert_yaxis()

    ax.xaxis.set_major_formatter(
        FuncFormatter(
            lambda x, _:
            f"{x:,.0f}"
            if x >= 0
            else ""
        )
    )

    ax.xaxis.grid(
        True,
        linewidth=0.8,
        alpha=0.35,
    )

    ax.set_axisbelow(True)

    ax.set_xlabel(
        "Membership value (RM)",
        fontsize=11,
    )

    for spine in (
        "top",
        "right",
        "left",
    ):
        ax.spines[spine].set_visible(False)

    ax.spines["bottom"].set_alpha(0.35)

    plt.tight_layout()

    image_buffer = io.BytesIO()

    fig.savefig(
        image_buffer,
        format="png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    image_buffer.seek(0)

    return image_buffer
