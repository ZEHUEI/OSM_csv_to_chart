from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter


# ============================================================
# CLEAN NUMBER
# ============================================================

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


# ============================================================
# FIND COLUMN
# ============================================================

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
        f"Could not find column containing: "
        f"{', '.join(required_words)}\n"
        f"Available columns: {columns}"
    )


# ============================================================
# GRADIENT BAR
# ============================================================

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

    gradient = np.linspace(
        0,
        1,
        256,
    ).reshape(1, -1)

    colour_map = LinearSegmentedColormap.from_list(
        "gradient",
        [
            start_colour,
            end_colour,
        ],
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


# ============================================================
# HEADER INSIDE BAR
# ============================================================

def add_header_inside_bar(
    ax,
    text,
    y,
    bar_width,
    maximum,
    colour="white",
):
    """
    Keep the main header text inside the bar.

    Font automatically becomes smaller when the bar
    is shorter.
    """

    ratio = bar_width / maximum

    if ratio >= 0.35:
        fontsize = 11

    elif ratio >= 0.22:
        fontsize = 9

    elif ratio >= 0.14:
        fontsize = 7.5

    else:
        fontsize = 6

    # Small left padding but always inside the bar.
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


# ============================================================
# MAIN CHART
# ============================================================

def create_chart(
    csv_path,
    output_path="membership_chart.png",
    show=False,
):

    csv_path = Path(csv_path)
    output_path = Path(output_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV file not found: {csv_path}"
        )

    # ========================================================
    # READ CSV
    # ========================================================

    df = pd.read_csv(csv_path)

    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    # ========================================================
    # FIRST COLUMN
    # ========================================================
    #
    # The HEADER of Column 1 becomes the chart title.
    #
    # Example:
    #
    # Membership,2025H1 (ACTUAL),2026H1 (Actual),2026H1 (Target)
    #
    # "Membership" becomes the title.
    #
    # Values underneath Column 1 remain the category/helper names:
    #
    # Label
    # Publishing
    # Overall
    # O E
    # O W
    # etc.
    # ========================================================

    label_column = df.columns[0]

    chart_title = label_column

    # ========================================================
    # FIND VALUE COLUMNS
    # ========================================================

    col_previous = find_column(
        list(df.columns),
        ("2025h1", "actual"),
    )

    col_current = find_column(
        list(df.columns),
        ("2026h1", "actual"),
    )

    col_target = find_column(
        list(df.columns),
        ("2026h1", "target"),
    )

    # ========================================================
    # CLEAN VALUES
    # ========================================================

    for column in (
        col_previous,
        col_current,
        col_target,
    ):
        df[column] = df[column].apply(
            clean_number
        )

    df[label_column] = (
        df[label_column]
        .astype(str)
        .str.strip()
    )

    df = df.set_index(label_column)

    # ========================================================
    # MAIN ROWS
    # ========================================================
    #
    # Main rows:
    #
    # Label
    # Publishing
    # Overall
    #
    # E/W rows remain helper rows only.
    # ========================================================

    summary_mask = (
        df[col_previous].notna()
        & df[col_current].notna()
        & df[col_target].notna()
    )

    categories = df.index[
        summary_mask
    ].tolist()

    if not categories:
        raise ValueError(
            "No main category rows found."
        )

    # Reverse CSV order.
    #
    # If CSV:
    #
    # Label
    # Publishing
    # Overall
    #
    # Chart:
    #
    # Overall
    # Publishing
    # Label

    categories = categories[::-1]

    # ========================================================
    # BUILD RECORDS
    # ========================================================

    records = []

    for category in categories:

        category_name = str(
            category
        ).strip()

        # Overall -> O
        # Publishing -> P
        # Label -> L

        first_letter = (
            category_name[0]
            .upper()
        )

        existing_row = f"{first_letter} E"
        withdrawn_row = f"{first_letter} W"

        previous = df.at[
            category,
            col_previous,
        ]

        current = df.at[
            category,
            col_current,
        ]

        target = df.at[
            category,
            col_target,
        ]

        # ----------------------------------------------------
        # EXISTING
        # ----------------------------------------------------

        if existing_row in df.index:

            existing = df.at[
                existing_row,
                col_previous,
            ]

        else:

            existing = np.nan

        # ----------------------------------------------------
        # WITHDRAWN
        # ----------------------------------------------------

        if withdrawn_row in df.index:

            withdrawn = df.at[
                withdrawn_row,
                col_previous,
            ]

        else:

            withdrawn = np.nan

        # ----------------------------------------------------
        # FALLBACK
        # ----------------------------------------------------

        if (
            pd.isna(existing)
            and not pd.isna(withdrawn)
        ):

            existing = max(
                previous - withdrawn,
                0,
            )

        elif (
            pd.isna(withdrawn)
            and not pd.isna(existing)
        ):

            withdrawn = max(
                previous - existing,
                0,
            )

        elif (
            pd.isna(existing)
            and pd.isna(withdrawn)
        ):

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

    # ========================================================
    # MAXIMUM
    # ========================================================

    maximum = max(
        max(
            record["previous"],
            record["current"],
            record["target"],
        )
        for record in records
    )

    # ========================================================
    # FIGURE
    # ========================================================

    figure_height = max(
        7,
        len(records) * 2.4,
    )

    fig, ax = plt.subplots(
        figsize=(
            14,
            figure_height,
        )
    )

    normal_bar_height = 0.36

    previous_bar_height = 0.48

    group_gap = 1.8

    # ========================================================
    # COLOURS
    # ========================================================

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

    # ========================================================
    # DRAW
    # ========================================================

    for group_number, record in enumerate(records):

        group_y = (
            group_number
            * group_gap
        )

        current_y = group_y

        target_y = (
                current_y
                - normal_bar_height
        )

        previous_y = (
                current_y
                + normal_bar_height / 2
                + previous_bar_height / 2
        )

        category = record[
            "category"
        ]

        previous = record[
            "previous"
        ]

        current = record[
            "current"
        ]

        target = record[
            "target"
        ]

        existing = record[
            "existing"
        ]

        withdrawn = record[
            "withdrawn"
        ]

        # ====================================================
        # TARGET BAR
        # ====================================================

        gradient_barh(
            ax,
            target_y,
            target,
            normal_bar_height,
            0,
            colours["target"][0],
            colours["target"][1],
        )

        # Main header ALWAYS inside.
        add_header_inside_bar(
            ax=ax,
            text=col_target,
            y=target_y,
            bar_width=target,
            maximum=maximum,
            colour="white",
        )

        # ====================================================
        # CURRENT BAR
        # ====================================================

        gradient_barh(
            ax,
            current_y,
            current,
            normal_bar_height,
            0,
            colours["current"][0],
            colours["current"][1],
        )

        # Main header ALWAYS inside.
        add_header_inside_bar(
            ax=ax,
            text=col_current,
            y=current_y,
            bar_width=current,
            maximum=maximum,
            colour="white",
        )

        # ====================================================
        # PREVIOUS - EXISTING
        # ====================================================

        gradient_barh(
            ax,
            previous_y,
            existing,
            previous_bar_height,
            0,
            colours["existing"][0],
            colours["existing"][1],
        )

        # ====================================================
        # PREVIOUS - WITHDRAWN
        # ====================================================

        gradient_barh(
            ax,
            previous_y,
            withdrawn,
            previous_bar_height,
            existing,
            colours["withdrawn"][0],
            colours["withdrawn"][1],
        )

        # ====================================================
        # PREVIOUS HEADER
        # ====================================================
        #
        # 2025 header is ALWAYS kept inside.
        # ====================================================

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

        existing_ratio = (
            existing / maximum
        )

        # Enough space -> inside.
        if existing_ratio >= 0.20:

            ax.text(
                existing * 0.62,
                previous_y,
                f"Existing Members\n"
                f"RM{existing:,.0f}",
                va="center",
                ha="center",
                fontsize=8.5,
                fontweight="bold",
                color="black",
                zorder=8,
            )

        # Not enough space -> arrow.
        else:

            ax.annotate(
                f"Existing Members\n"
                f"RM{existing:,.0f}",

                xy=(
                    existing * 0.65,
                    previous_y,
                ),

                xytext=(
                    existing
                    + maximum * 0.08,
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

            withdrawn_ratio = (
                withdrawn / maximum
            )

            # ------------------------------------------------
            # Enough space -> inside.
            # ------------------------------------------------

            if withdrawn_ratio >= 0.10:

                ax.text(
                    existing + withdrawn / 2,
                    previous_y,
                    f"Withdrawn Members\n"
                    f"RM{withdrawn:,.0f}",
                    va="center",
                    ha="center",
                    fontsize=8,
                    fontweight="bold",
                    color="black",
                    zorder=8,
                )

            # ------------------------------------------------
            # Too small -> arrow.
            # ------------------------------------------------

            else:

                ax.annotate(
                    f"Withdrawn Members\n"
                    f"RM{withdrawn:,.0f}",

                    xy=(
                        existing
                        + withdrawn / 2,
                        previous_y,
                    ),

                    xytext=(
                        min(
                            previous
                            + maximum * 0.07,
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
        # TOTAL VALUES
        # ====================================================

        value_padding = (
            maximum * 0.012
        )

        # Target total
        ax.text(
            target + value_padding,
            target_y,
            f"{target:,.0f}",
            va="center",
            ha="left",
            fontsize=10,
            color="#444444",
        )

        # Current total
        ax.text(
            current + value_padding,
            current_y,
            f"{current:,.0f}",
            va="center",
            ha="left",
            fontsize=10,
            color="#444444",
        )

        # Previous total
        ax.text(
            previous + value_padding,
            previous_y,
            f"{previous:,.0f}",
            va="center",
            ha="left",
            fontsize=10,
            color="#444444",
        )

        # ====================================================
        # CATEGORY
        # ====================================================

        y_tick_positions.append(
            group_y
        )

        y_tick_labels.append(
            category
        )

    # ========================================================
    # Y AXIS
    # ========================================================

    ax.set_yticks(
        y_tick_positions
    )

    ax.set_yticklabels(
        y_tick_labels,
        fontsize=13,
    )

    ax.tick_params(
        axis="y",
        length=0,
        pad=12,
    )

    # No separate Y-axis title.
    ax.set_ylabel("")

    # ========================================================
    # CHART TITLE
    # ========================================================
    #
    # Column 1 header becomes the title.
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

    # Extra room for arrows and total values.
    ax.set_xlim(
        -maximum * 0.015,
        maximum * 1.18,
    )

    ax.set_ylim(
        -1.0,
        (len(records) - 1)
        * group_gap
        + 1.15,
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

    # ========================================================
    # REMOVE BORDERS
    # ========================================================

    for spine in (
        "top",
        "right",
        "left",
    ):

        ax.spines[
            spine
        ].set_visible(False)

    ax.spines[
        "bottom"
    ].set_alpha(0.35)

    # ========================================================
    # SAVE
    # ========================================================

    plt.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    print(
        f"Chart saved to: "
        f"{output_path.resolve()}"
    )

    if show:
        plt.show()

    else:
        plt.close(fig)


# ============================================================
# ARGUMENTS
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "Create membership chart "
            "with Column 1 header used as the chart title."
        )
    )

    parser.add_argument(
        "csv_file",
    )

    parser.add_argument(
        "--output",
        default="membership_chart.png",
    )

    parser.add_argument(
        "--show",
        action="store_true",
    )

    return parser.parse_args()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    args = parse_arguments()

    create_chart(
        csv_path=args.csv_file,
        output_path=args.output,
        show=args.show,
    )