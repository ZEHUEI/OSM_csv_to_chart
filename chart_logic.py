from __future__ import annotations

import io
import re
import textwrap

import matplotlib

# Required for Render / servers with no GUI.
matplotlib.use("Agg")

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
    """
    Convert values such as:

    5,714,747
    RM 5,714,747
    $5,714,747

    into floats.
    """

    if pd.isna(value):
        return np.nan

    text = str(value).strip()

    if not text:
        return np.nan

    text = re.sub(
        r"[^0-9.\-]",
        "",
        text,
    )

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
    """
    Find a column containing all required words.
    """

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


# ============================================================
# NORMALISE ROW NAME
# ============================================================

def normalise_row_name(value: object) -> str:
    """
    Normalise helper-row names for safer matching.
    """

    text = str(value).strip().lower()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    text = re.sub(
        r"\s*-\s*",
        "-",
        text,
    )

    return text


# ============================================================
# IS HELPER ROW
# ============================================================

def is_helper_row(row_name: object) -> bool:
    """
    Detect rows that should NOT become chart categories.
    """

    text = str(row_name).strip()

    if not text:
        return False

    # Short style:
    # O E
    # O W
    if re.fullmatch(
        r"[A-Za-z]\s+[EW]",
        text,
        flags=re.IGNORECASE,
    ):
        return True

    normalised = normalise_row_name(
        text
    )

    if normalised.endswith(
        "-existing member"
    ):
        return True

    if normalised.endswith(
        "-withdrawn member"
    ):
        return True

    return False


# ============================================================
# FIND HELPER ROW
# ============================================================

def find_helper_row(
    index_values,
    category_name: str,
    helper_type: str,
) -> object | None:
    """
    Locate the Existing or Withdrawn row for a category.
    """

    if helper_type not in {
        "existing",
        "withdrawn",
    }:
        raise ValueError(
            "helper_type must be 'existing' or 'withdrawn'"
        )

    suffix = (
        "Existing Member"
        if helper_type == "existing"
        else "Withdrawn Member"
    )

    long_candidate = (
        f"{category_name}-{suffix}"
    )

    normalised_candidate = (
        normalise_row_name(
            long_candidate
        )
    )

    # --------------------------------------------------------
    # Try explicit long names.
    # --------------------------------------------------------

    for index_value in index_values:

        if (
            normalise_row_name(
                index_value
            )
            == normalised_candidate
        ):
            return index_value

    # --------------------------------------------------------
    # Fall back to:
    #
    # Overall -> O E / O W
    # Publishing -> P E / P W
    # Label -> L E / L W
    # --------------------------------------------------------

    category_name = (
        str(category_name)
        .strip()
    )

    if not category_name:
        return None

    first_letter = (
        category_name[0]
        .upper()
    )

    short_candidate = (
        f"{first_letter} E"
        if helper_type == "existing"
        else f"{first_letter} W"
    )

    for index_value in index_values:

        if (
            str(index_value)
            .strip()
            .upper()
            == short_candidate.upper()
        ):
            return index_value

    return None


# ============================================================
# BREAKDOWN VALIDATION
# ============================================================

def fill_breakdown(
    total: float,
    existing: float,
    withdrawn: float,
) -> tuple[float, float]:
    """
    Return a valid Existing / Withdrawn breakdown.
    """

    total = float(total)

    # --------------------------------------------------------
    # Both missing.
    # --------------------------------------------------------

    if (
        pd.isna(existing)
        and pd.isna(withdrawn)
    ):
        return total, 0.0

    # --------------------------------------------------------
    # Existing missing.
    # --------------------------------------------------------

    if pd.isna(existing):

        withdrawn = float(
            withdrawn
        )

        existing = max(
            total - withdrawn,
            0,
        )

    # --------------------------------------------------------
    # Withdrawn missing.
    # --------------------------------------------------------

    elif pd.isna(withdrawn):

        existing = float(
            existing
        )

        withdrawn = max(
            total - existing,
            0,
        )

    # --------------------------------------------------------
    # Both supplied.
    # --------------------------------------------------------

    else:

        existing = float(
            existing
        )

        withdrawn = float(
            withdrawn
        )

    # Prevent negative values.

    existing = max(
        existing,
        0,
    )

    withdrawn = max(
        withdrawn,
        0,
    )

    breakdown_total = (
        existing + withdrawn
    )

    tolerance = max(
        abs(total) * 0.001,
        1.0,
    )

    if (
        abs(
            breakdown_total - total
        )
        > tolerance
    ):
        return total, 0.0

    return (
        existing,
        withdrawn,
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
    """
    Draw one horizontal gradient segment.
    """

    if (
        pd.isna(width)
        or width <= 0
    ):
        return

    rectangle = Rectangle(
        (
            left,
            y - height / 2,
        ),
        width,
        height,
        facecolor="none",
        edgecolor="none",
    )

    ax.add_patch(
        rectangle
    )

    gradient = np.linspace(
        0,
        1,
        256,
    ).reshape(
        1,
        -1,
    )

    colour_map = (
        LinearSegmentedColormap.from_list(
            f"gradient_"
            f"{start_colour}_"
            f"{end_colour}_"
            f"{y}_"
            f"{left}",
            [
                start_colour,
                end_colour,
            ],
        )
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

    image.set_clip_path(
        rectangle
    )


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
    Keep main period headers inside the bar.
    """

    ratio = (
        bar_width / maximum
        if maximum > 0
        else 0
    )

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


# ============================================================
# BREAKDOWN LABELS
# ============================================================

def add_breakdown_labels(
    ax,
    y,
    total,
    existing,
    withdrawn,
    maximum,
    text_colour="black",
    arrow_direction="down",
):
    """
    Show Existing / Withdrawn labels.

    Large segment:
        label stays inside.

    Small segment:
        label moves outside with an arrow.
    """

    # ========================================================
    # EXISTING
    # ========================================================

    if existing > 0:

        existing_ratio = (
            existing / maximum
        )

        # ----------------------------------------------------
        # Keep inside when segment is >= 29%.
        #
        # This keeps the bottom blue bar's Existing label
        # inside as well.
        # ----------------------------------------------------

        if existing_ratio >= 0.29:

            # Push Existing Members slightly further right.
            existing_text_x = max(
                existing * 0.60,
                maximum * 0.21,
            )

            # Do not move beyond the segment.
            existing_text_x = min(
                existing_text_x,
                existing * 0.86,
            )

            ax.text(
                existing_text_x,
                y,
                f"Existing Members\n"
                f"RM{existing:,.0f}",
                va="center",
                ha="center",
                fontsize=9,
                fontweight="bold",
                color=text_colour,
                zorder=8,
            )

        # ----------------------------------------------------
        # Existing too small -> arrow outside.
        # ----------------------------------------------------

        else:

            # Green bars use black text_colour.
            is_green_bar = (
                text_colour == "black"
            )

            if is_green_bar:

                # Green Existing:
                # slightly lower and slightly to the right.
                vertical_offset = (
                    -0.58
                    if arrow_direction == "up"
                    else 0.45
                )

                x_offset = (
                    maximum * 0.045
                )

            else:

                vertical_offset = (
                    -0.48
                    if arrow_direction == "up"
                    else 0.48
                )

                x_offset = (
                    maximum * 0.08
                )

            ax.annotate(
                f"Existing Members\n"
                f"RM{existing:,.0f}",

                # Arrow begins from the Existing segment.
                xy=(
                    existing * 0.65,
                    y,
                ),

                # Text position.
                xytext=(
                    min(
                        existing
                        + x_offset,
                        maximum * 1.05,
                    ),
                    y + vertical_offset,
                ),

                va="center",
                ha="left",

                fontsize=9,
                fontweight="bold",
                color="black",

                arrowprops={
                    # Arrow points outward toward label.
                    "arrowstyle": "<-",
                    "linewidth": 1.2,
                    "color": "#24667a",
                },

                zorder=9,
            )

    # ========================================================
    # WITHDRAWN
    # ========================================================

    if withdrawn > 0:

        withdrawn_ratio = (
            withdrawn / maximum
        )

        # ----------------------------------------------------
        # Large Withdrawn section stays inside.
        # ----------------------------------------------------

        if withdrawn_ratio >= 0.20:

            ax.text(
                existing
                + withdrawn / 2,
                y,
                f"Withdrawn Members\n"
                f"RM{withdrawn:,.0f}",
                va="center",
                ha="center",
                fontsize=9,
                fontweight="bold",
                color=text_colour,
                zorder=8,
            )

        # ----------------------------------------------------
        # Small Withdrawn section goes outside.
        # ----------------------------------------------------

        else:

            is_green_bar = (
                text_colour == "black"
            )

            existing_ratio = (
                existing / maximum
            )

            if is_green_bar:

                # --------------------------------------------
                # If Existing is ALSO outside,
                # separate the two green labels more.
                # This mainly affects the final green bar.
                # --------------------------------------------

                if existing_ratio < 0.29:

                    withdrawn_vertical_offset = (
                        -1.00
                        if arrow_direction == "up"
                        else 1.20
                    )

                    withdrawn_x_extra = (
                        maximum * 0.14
                    )

                # --------------------------------------------
                # Existing is inside but Withdrawn is outside.
                # Example: middle green bar.
                # Keep label within the white gap.
                # --------------------------------------------

                else:

                    withdrawn_vertical_offset = (
                        -0.62
                        if arrow_direction == "up"
                        else 0.78
                    )

                    withdrawn_x_extra = (
                        maximum * 0.14
                    )

            else:

                withdrawn_vertical_offset = (
                    -0.62
                    if arrow_direction == "up"
                    else 0.78
                )

                withdrawn_x_extra = (
                    maximum * 0.10
                )

            ax.annotate(
                f"Withdrawn Members\n"
                f"RM{withdrawn:,.0f}",

                # Arrow starts from middle of light segment.
                xy=(
                    existing
                    + withdrawn / 2,
                    y,
                ),

                # Text sits outside/right.
                xytext=(
                    min(
                        total
                        + withdrawn_x_extra,
                        maximum * 1.12,
                    ),
                    y
                    + withdrawn_vertical_offset,
                ),

                va="center",
                ha="left",

                fontsize=9,
                fontweight="bold",
                color="black",

                arrowprops={
                    # Arrowhead points outward.
                    "arrowstyle": "<-",
                    "linewidth": 1.2,
                    "color": "#24667a",
                },

                zorder=9,
            )


# ============================================================
# MAIN FUNCTION
# ============================================================

def create_chart_from_csv_bytes(
    csv_bytes: bytes,
) -> io.BytesIO:
    """
    Read uploaded CSV bytes and return a PNG image buffer.
    """

    # ========================================================
    # READ CSV
    # ========================================================

    try:

        df = pd.read_csv(
            io.BytesIO(
                csv_bytes
            )
        )

    except Exception as exc:

        raise ValueError(
            f"Could not read CSV: {exc}"
        ) from exc

    if df.empty:

        raise ValueError(
            "The CSV contains no data rows."
        )

    if len(df.columns) < 4:

        raise ValueError(
            "CSV needs at least four columns: "
            "the title/category column plus "
            "2025 Actual, 2026 Actual and 2026 Target."
        )

    # ========================================================
    # CLEAN COLUMN HEADERS
    # ========================================================

    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    # ========================================================
    # COLUMN 1
    # ========================================================

    label_column = (
        df.columns[0]
    )

    chart_title = (
        label_column
    )

    if (
        not chart_title
        or chart_title
        .lower()
        .startswith("unnamed:")
    ):

        chart_title = (
            "Membership"
        )

    # ========================================================
    # FIND METRIC COLUMNS
    # ========================================================

    metric_columns = list(
        df.columns[1:]
    )

    col_previous = find_column(
        metric_columns,
        (
            "2025h1",
            "actual",
        ),
    )

    col_current = find_column(
        metric_columns,
        (
            "2026h1",
            "actual",
        ),
    )

    col_target = find_column(
        metric_columns,
        (
            "2026h1",
            "target",
        ),
    )

    # ========================================================
    # CLEAN NUMBERS
    # ========================================================

    for column in (
        col_previous,
        col_current,
        col_target,
    ):

        df[column] = (
            df[column]
            .apply(
                clean_number
            )
        )

    # ========================================================
    # CLEAN ROW NAMES
    # ========================================================

    df[label_column] = (
        df[label_column]
        .astype(str)
        .str.strip()
    )

    df = df.set_index(
        label_column
    )

    # ========================================================
    # FIND MAIN CATEGORIES
    # ========================================================

    categories = []

    for (
        index_value,
        row,
    ) in df.iterrows():

        if is_helper_row(
            index_value
        ):
            continue

        if (
            pd.notna(
                row[col_previous]
            )
            and pd.notna(
                row[col_current]
            )
            and pd.notna(
                row[col_target]
            )
        ):

            categories.append(
                index_value
            )

    if not categories:

        raise ValueError(
            "No main category rows found. "
            "A main row needs values for "
            "2025 Actual, 2026 Actual and 2026 Target."
        )

    categories = (
        categories[::-1]
    )

    # ========================================================
    # BUILD RECORDS
    # ========================================================

    records = []

    for category in categories:

        category_name = (
            str(category)
            .strip()
        )

        if not category_name:
            continue

        existing_row = (
            find_helper_row(
                df.index,
                category_name,
                "existing",
            )
        )

        withdrawn_row = (
            find_helper_row(
                df.index,
                category_name,
                "withdrawn",
            )
        )

        # ----------------------------------------------------
        # Official totals.
        # ----------------------------------------------------

        previous = float(
            df.at[
                category,
                col_previous,
            ]
        )

        current = float(
            df.at[
                category,
                col_current,
            ]
        )

        target = float(
            df.at[
                category,
                col_target,
            ]
        )

        # ====================================================
        # 2025 ACTUAL BREAKDOWN
        # ====================================================

        previous_existing = (
            df.at[
                existing_row,
                col_previous,
            ]
            if existing_row is not None
            else np.nan
        )

        previous_withdrawn = (
            df.at[
                withdrawn_row,
                col_previous,
            ]
            if withdrawn_row is not None
            else np.nan
        )

        # ====================================================
        # 2026 ACTUAL BREAKDOWN
        # ====================================================

        current_existing = (
            df.at[
                existing_row,
                col_current,
            ]
            if existing_row is not None
            else np.nan
        )

        current_withdrawn = (
            df.at[
                withdrawn_row,
                col_current,
            ]
            if withdrawn_row is not None
            else np.nan
        )

        # ====================================================
        # 2026 TARGET BREAKDOWN
        # ====================================================

        target_existing = (
            df.at[
                existing_row,
                col_target,
            ]
            if existing_row is not None
            else np.nan
        )

        target_withdrawn = (
            df.at[
                withdrawn_row,
                col_target,
            ]
            if withdrawn_row is not None
            else np.nan
        )

        # ====================================================
        # VALIDATE BREAKDOWNS
        # ====================================================

        (
            previous_existing,
            previous_withdrawn,
        ) = fill_breakdown(
            previous,
            previous_existing,
            previous_withdrawn,
        )

        (
            current_existing,
            current_withdrawn,
        ) = fill_breakdown(
            current,
            current_existing,
            current_withdrawn,
        )

        (
            target_existing,
            target_withdrawn,
        ) = fill_breakdown(
            target,
            target_existing,
            target_withdrawn,
        )

        records.append(
            {
                "category": category_name,

                "previous": previous,
                "previous_existing": (
                    previous_existing
                ),
                "previous_withdrawn": (
                    previous_withdrawn
                ),

                "current": current,
                "current_existing": (
                    current_existing
                ),
                "current_withdrawn": (
                    current_withdrawn
                ),

                "target": target,
                "target_existing": (
                    target_existing
                ),
                "target_withdrawn": (
                    target_withdrawn
                ),
            }
        )

    if not records:

        raise ValueError(
            "No usable chart records were found."
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

    if maximum <= 0:

        raise ValueError(
            "Chart values must contain "
            "at least one positive number."
        )

    # ========================================================
    # FIGURE SIZE
    # ========================================================

    figure_height = max(
        7,
        len(records) * 2.5,
    )

    fig, ax = plt.subplots(
        figsize=(
            14,
            figure_height,
        )
    )

    # ========================================================
    # BAR HEIGHTS
    # ========================================================

    normal_bar_height = (
        0.40
    )

    previous_bar_height = (
        0.50
    )

    stack_height = (
        normal_bar_height
        + normal_bar_height
        + previous_bar_height
    )

    white_gap_after_group = (
        0.72
    )

    group_gap = (
        stack_height
        + white_gap_after_group
    )

    overlap = (
        0.014
    )

    # ========================================================
    # COLOURS
    # ========================================================

    colours = {

        "target_existing": (
            "#c40000",
            "#ff1a1a",
        ),

        "target_withdrawn": (
            "#ff7777",
            "#ffb3b3",
        ),

        "current_existing": (
            "#2877bd",
            "#6db5eb",
        ),

        "current_withdrawn": (
            "#9ed2f5",
            "#d6edfc",
        ),

        "previous_existing": (
            "#11b711",
            "#11b711",
        ),

        "previous_withdrawn": (
            "#74e574",
            "#b4ffb4",
        ),
    }

    y_tick_positions = []
    y_tick_labels = []

    # ========================================================
    # DRAW
    # ========================================================

    for (
        group_number,
        record,
    ) in enumerate(records):

        group_top = (
            group_number
            * group_gap
        )

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

        # ====================================================
        # VALUES
        # ====================================================

        category = (
            record["category"]
        )

        target = (
            record["target"]
        )

        target_existing = (
            record[
                "target_existing"
            ]
        )

        target_withdrawn = (
            record[
                "target_withdrawn"
            ]
        )

        current = (
            record["current"]
        )

        current_existing = (
            record[
                "current_existing"
            ]
        )

        current_withdrawn = (
            record[
                "current_withdrawn"
            ]
        )

        previous = (
            record["previous"]
        )

        previous_existing = (
            record[
                "previous_existing"
            ]
        )

        previous_withdrawn = (
            record[
                "previous_withdrawn"
            ]
        )

        # ====================================================
        # TARGET - RED
        # ====================================================

        gradient_barh(
            ax=ax,
            y=target_y,
            width=target_existing,
            height=(
                normal_bar_height
                + overlap
            ),
            left=0,
            start_colour=(
                colours[
                    "target_existing"
                ][0]
            ),
            end_colour=(
                colours[
                    "target_existing"
                ][1]
            ),
        )

        gradient_barh(
            ax=ax,
            y=target_y,
            width=target_withdrawn,
            height=(
                normal_bar_height
                + overlap
            ),
            left=target_existing,
            start_colour=(
                colours[
                    "target_withdrawn"
                ][0]
            ),
            end_colour=(
                colours[
                    "target_withdrawn"
                ][1]
            ),
        )

        add_header_inside_bar(
            ax=ax,
            text=col_target,
            y=target_y,
            bar_width=target,
            maximum=maximum,
            colour="white",
        )

        add_breakdown_labels(
            ax=ax,
            y=target_y,
            total=target,
            existing=target_existing,
            withdrawn=target_withdrawn,
            maximum=maximum,
            text_colour="white",
            arrow_direction="up",
        )

        # ====================================================
        # 2026 ACTUAL - BLUE
        # ====================================================

        gradient_barh(
            ax=ax,
            y=current_y,
            width=current_existing,
            height=(
                normal_bar_height
                + overlap
            ),
            left=0,
            start_colour=(
                colours[
                    "current_existing"
                ][0]
            ),
            end_colour=(
                colours[
                    "current_existing"
                ][1]
            ),
        )

        gradient_barh(
            ax=ax,
            y=current_y,
            width=current_withdrawn,
            height=(
                normal_bar_height
                + overlap
            ),
            left=current_existing,
            start_colour=(
                colours[
                    "current_withdrawn"
                ][0]
            ),
            end_colour=(
                colours[
                    "current_withdrawn"
                ][1]
            ),
        )

        add_header_inside_bar(
            ax=ax,
            text=col_current,
            y=current_y,
            bar_width=current,
            maximum=maximum,
            colour="white",
        )

        add_breakdown_labels(
            ax=ax,
            y=current_y,
            total=current,
            existing=current_existing,
            withdrawn=current_withdrawn,
            maximum=maximum,
            text_colour="white",
            arrow_direction="down",
        )

        # ====================================================
        # 2025 ACTUAL - GREEN
        # ====================================================

        gradient_barh(
            ax=ax,
            y=previous_y,
            width=previous_existing,
            height=(
                previous_bar_height
                + overlap
            ),
            left=0,
            start_colour=(
                colours[
                    "previous_existing"
                ][0]
            ),
            end_colour=(
                colours[
                    "previous_existing"
                ][1]
            ),
        )

        gradient_barh(
            ax=ax,
            y=previous_y,
            width=previous_withdrawn,
            height=(
                previous_bar_height
                + overlap
            ),
            left=previous_existing,
            start_colour=(
                colours[
                    "previous_withdrawn"
                ][0]
            ),
            end_colour=(
                colours[
                    "previous_withdrawn"
                ][1]
            ),
        )

        add_header_inside_bar(
            ax=ax,
            text=col_previous,
            y=previous_y,
            bar_width=previous,
            maximum=maximum,
            colour="black",
        )

        add_breakdown_labels(
            ax=ax,
            y=previous_y,
            total=previous,
            existing=previous_existing,
            withdrawn=previous_withdrawn,
            maximum=maximum,
            text_colour="black",
            arrow_direction="down",
        )

        # ====================================================
        # TOTAL VALUES
        # ====================================================

        value_padding = (
            maximum * 0.012
        )

        ax.text(
            target
            + value_padding,
            target_y,
            f"{target:,.0f}",
            va="center",
            ha="left",
            fontsize=12,
            fontweight="bold",
            color="#444444",
        )

        ax.text(
            current
            + value_padding,
            current_y,
            f"{current:,.0f}",
            va="center",
            ha="left",
            fontsize=12,
            fontweight="bold",
            color="#444444",
        )

        ax.text(
            previous
            + value_padding,
            previous_y,
            f"{previous:,.0f}",
            va="center",
            ha="left",
            fontsize=12,
            fontweight="bold",
            color="#444444",
        )

        # ====================================================
        # CATEGORY
        # ====================================================

        y_tick_positions.append(
            group_label_y
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
        fontsize=12,
    )

    ax.tick_params(
        axis="y",
        length=0,
        pad=12,
    )

    ax.set_ylabel(
        ""
    )

    # ========================================================
    # TITLE
    # ========================================================

    wrapped_title = "\n".join(
        textwrap.wrap(
            chart_title,
            width=75,
        )
    )

    ax.set_title(
        wrapped_title,
        fontsize=17,
        fontweight="bold",
        pad=20,
    )

    # ========================================================
    # X AXIS
    # ========================================================

    ax.set_xlim(
        -maximum * 0.015,
        maximum * 1.20,
    )

    chart_bottom = (
        (
            len(records) - 1
        )
        * group_gap
        + stack_height
    )

    # Extra room is added at the bottom because
    # the last green labels may sit below the bar.
    ax.set_ylim(
        -0.45,
        chart_bottom + 1.45,
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

    ax.set_axisbelow(
        True
    )

    ax.set_xlabel(
        "Value (RM)",
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
        ].set_visible(
            False
        )

    ax.spines[
        "bottom"
    ].set_alpha(
        0.35
    )

    # ========================================================
    # SAVE TO MEMORY
    # ========================================================

    plt.tight_layout()

    image_buffer = (
        io.BytesIO()
    )

    fig.savefig(
        image_buffer,
        format="png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    image_buffer.seek(
        0
    )

    return image_buffer