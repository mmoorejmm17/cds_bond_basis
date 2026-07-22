"""
Plotting helpers for the CDS-bond basis timeseries tool.

Public API
----------
plot_bond_cds_basis(ticker, bond_history, start_date=None, end_date=None)
    → (IPython.display.HTML, bytes)
"""

import io
import base64

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
from IPython.display import HTML

from .config import (
    FIGSIZE,
    FIG_DPI,
    SUPTITLE_FONTSIZE,
    TITLE_FONTSIZE,
    LABEL_FONTSIZE,
    LEGEND_FONTSIZE,
    ANNOTATION_FONTSIZE,
    X_TICK_FONTSIZE,
)


# ── Low-level helpers ──────────────────────────────────────────────────────────

# All plotted lines (data series and mean/std reference lines) are drawn at
# 80% of their nominal alpha and linewidth.
_LINE_ALPHA = 0.8
_LINE_WIDTH_SCALE = 0.8

# Percentile thresholds beyond which a series' legend text is flagged.
_PERCENTILE_LOW = 20
_PERCENTILE_HIGH = 80

# Ensures the last-point annotation always draws above plotted lines/gridlines.
_ANNOTATION_ZORDER = 10


def _annotate_last_point(ax, dates, values, color="black", fmt="{:.1f}"):
    """Label the last valid point of a series directly below the line, in the
    line's own color, so the current value is readable at a glance. Always
    placed below (rather than alternating above/below by position) so every
    series is annotated the same way for visual consistency."""
    valid = values.notna()
    if not valid.any():
        return
    v = values[valid]
    last_date = dates[valid].iloc[-1]
    last_value = v.iloc[-1]
    ax.annotate(
        _escape_dollar(fmt.format(last_value)),
        xy=(last_date, last_value),
        xytext=(0, -8),
        textcoords="offset points",
        ha="center",
        va="top",
        color=color,
        fontsize=ANNOTATION_FONTSIZE,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, alpha=0.85),
        zorder=_ANNOTATION_ZORDER,
    )


def _percentile_color(percentile):
    """Flag color for a legend entry based on the last value's percentile rank:
    red at/below the 20th percentile, blue at/above the 80th, else default."""
    if percentile is None:
        return None
    if percentile <= _PERCENTILE_LOW:
        return "red"
    if percentile >= _PERCENTILE_HIGH:
        return "blue"
    return None


def _colorize_legend(legend, pct_by_label):
    """Recolor legend text entries per ``_percentile_color``, matched by label text."""
    for text in legend.get_texts():
        color = _percentile_color(pct_by_label.get(text.get_text()))
        if color is not None:
            text.set_color(color)


def _last_valid_value(values):
    """Return the last non-NaN value in a series, or None if all NaN."""
    valid = values.notna()
    if not valid.any():
        return None
    return values[valid].iloc[-1]


def _escape_dollar(text):
    """Escape literal '$' so matplotlib doesn't parse a pair of them as mathtext,
    which silently strips whitespace between them (e.g. '$97.6 to $101.0' -> '97.6to101.0')."""
    return text.replace("$", r"\$")


def _labeled(label, value, fmt="{:.1f}"):
    """Append a formatted latest/level value to a legend label, if available."""
    if value is None or pd.isna(value):
        return label
    return _escape_dollar(f"{label}: {fmt.format(value)}")


def _range_label(label, low, high, fmt="{:.1f}"):
    """Legend label for a symmetric mean +/- std band, e.g. '+/-1 std range: 17.7 to 42.2'."""
    return _escape_dollar(f"{label} range: {fmt.format(low)} to {fmt.format(high)}")


def _labeled_stats(label, values, fmt="{:.1f}"):
    """Build legend text for a series' last/max/min values, plus a separate
    percentile-rank line for the last value within the series' own history.

    Keeping last/max/min/percentile in the legend text (rather than drawing
    separate annotations on the chart) avoids reserving extra margin outside
    the axes for them — the legend's margin already exists, so this doesn't
    compress the plot area the way on-chart annotations did. The percentile
    is returned as its own label text (rather than folded into the same
    string as last/max/min) so its legend entry can be colored on its own —
    a single legend Text artist can't have just part of its string colored.

    Returns
    -------
    (main_text, percentile_text, percentile) : tuple[str, str | None, float | None]
        ``percentile_text`` and ``percentile`` are ``None`` when the series
        has no valid data.
    """
    valid = values.notna()
    if not valid.any():
        return label, None, None
    v = values[valid]
    last_value = v.iloc[-1]
    percentile = v.rank(pct=True).iloc[-1] * 100
    main_text = _escape_dollar(
        f"{label}: {fmt.format(last_value)}\nmax {fmt.format(v.max())}, min {fmt.format(v.min())}"
    )
    percentile_text = f"percentile: {percentile:.1f}%"
    return main_text, percentile_text, percentile


# ── Main plotting function ─────────────────────────────────────────────────────

def plot_bond_cds_basis(ticker, bond_history, start_date=None, end_date=None):
    """
    Generate the 3-panel CDS-vs-bond basis chart for a single ticker.

    Parameters
    ----------
    ticker : str
        CDS market ticker (e.g. ``"ORCL"``).
    bond_history : pd.DataFrame
        Merged bond/CDS history with columns ``ticker``, ``date``, ``cds_spread``,
        ``G_Spread``, ``basis_spread``, ``basis_price``, ``SECURITY_NAME``,
        ``target_cds_maturity``, ``Markit_ShortName``, and optionally
        ``cds_spread_5y`` (5Y CDS spread, plotted as a green dotted line if present).
    start_date, end_date : str or None
        Optional ISO-date strings to restrict the plotted time range.

    Returns
    -------
    html : IPython.display.HTML
        Inline base64-encoded PNG ready for ``display()`` in a notebook.
    img_bytes : bytes
        Raw PNG bytes — used by the slideshow runner to save marked charts.
    """
    plot_data = bond_history[bond_history["ticker"] == ticker].sort_values("date")
    if plot_data.empty:
        raise ValueError(f"No data found for ticker '{ticker}'")

    if start_date:
        plot_data = plot_data[plot_data["date"] >= pd.to_datetime(start_date)]
    if end_date:
        plot_data = plot_data[plot_data["date"] <= pd.to_datetime(end_date)]

    security_name = plot_data["SECURITY_NAME"].iloc[0]
    markit_shortname = plot_data["Markit_ShortName"].iloc[0]
    cds_maturity_used = pd.Timestamp(plot_data["target_cds_maturity"].iloc[0]).strftime("%Y-%m-%d")

    mean_basis_spread = plot_data["basis_spread"].mean()
    std_basis_spread = plot_data["basis_spread"].std()

    mean_basis_price = plot_data["basis_price"].mean()
    std_basis_price = plot_data["basis_price"].std()

    fig, axes = plt.subplots(3, 1, figsize=FIGSIZE, sharex=True)
    # x=0.37 centers the title over the axes (which occupy the left 0.74 of the
    # figure per the tight_layout rect below), not the whole figure — the
    # legends occupying the right 0.26 would otherwise pull a figure-centered
    # title visibly rightward.
    fig.suptitle(markit_shortname, fontsize=SUPTITLE_FONTSIZE, fontweight="bold", x=0.37)
    for _ax in axes:
        _ax.margins(y=0.15)
        _ax.set_axisbelow(True)
        _ax.yaxis.grid(True, color="black", alpha=0.15, linewidth=0.7)

    ax = axes[0]
    pct_by_label = {}
    label, pct_label, pct = _labeled_stats("CDS Spread", plot_data["cds_spread"])
    ax.plot(plot_data["date"], plot_data["cds_spread"], color="red", alpha=_LINE_ALPHA, linewidth=1.5 * _LINE_WIDTH_SCALE, label=label)
    _annotate_last_point(ax, plot_data["date"], plot_data["cds_spread"], color="red")
    if pct_label:
        ax.plot([], [], color="none", label=pct_label)
        pct_by_label[pct_label] = pct
    label, pct_label, pct = _labeled_stats("G-Spread", plot_data["G_Spread"])
    ax.plot(plot_data["date"], plot_data["G_Spread"], color="blue", alpha=_LINE_ALPHA, linewidth=1.5 * _LINE_WIDTH_SCALE, label=label)
    _annotate_last_point(ax, plot_data["date"], plot_data["G_Spread"], color="blue")
    if pct_label:
        ax.plot([], [], color="none", label=pct_label)
        pct_by_label[pct_label] = pct
    if "Benchmark_Spread" in plot_data.columns:
        last_benchmark_spread = _last_valid_value(plot_data["Benchmark_Spread"])
        if last_benchmark_spread is not None:
            ax.plot([], [], color="none", label=_labeled("Benchmark Spread", last_benchmark_spread))
    if "cds_spread_5y" in plot_data.columns:
        label, pct_label, pct = _labeled_stats("5Y CDS Spread", plot_data["cds_spread_5y"])
        ax.plot(
            plot_data["date"], plot_data["cds_spread_5y"], color="green", linestyle=":",
            alpha=_LINE_ALPHA, linewidth=1.5 * _LINE_WIDTH_SCALE, label=label,
        )
        _annotate_last_point(ax, plot_data["date"], plot_data["cds_spread_5y"], color="green")
        if pct_label:
            ax.plot([], [], color="none", label=pct_label)
            pct_by_label[pct_label] = pct
    if "LAST_PRICE" in plot_data.columns:
        last_bond_price = _last_valid_value(plot_data["LAST_PRICE"])
        if last_bond_price is not None:
            ax.plot([], [], color="none", label=_labeled("Bond Price", last_bond_price, fmt="${:,.2f}"))
    ax.set_title(f"{ticker} ({security_name}) CDS Spread (Mat: {cds_maturity_used}) vs G-Spread over time", fontsize=TITLE_FONTSIZE, fontweight="bold")
    ax.set_ylabel("Spread (bps)", fontsize=LABEL_FONTSIZE, fontweight="bold")
    legend = ax.legend(fontsize=LEGEND_FONTSIZE, loc="upper left", bbox_to_anchor=(1.02, 1))
    _colorize_legend(legend, pct_by_label)

    ax = axes[1]
    label, pct_label, pct = _labeled_stats("Spread Basis", plot_data["basis_spread"])
    pct_by_label = {}
    ax.plot(plot_data["date"], plot_data["basis_spread"], color="tab:blue", alpha=_LINE_ALPHA, linewidth=1.5 * _LINE_WIDTH_SCALE, label=label)
    _annotate_last_point(ax, plot_data["date"], plot_data["basis_spread"], color="tab:blue")
    if pct_label:
        ax.plot([], [], color="none", label=pct_label)
        pct_by_label[pct_label] = pct
    ax.axhline(mean_basis_spread, color="black", linestyle="--", linewidth=1 * _LINE_WIDTH_SCALE, alpha=_LINE_ALPHA, label=_labeled("Mean", mean_basis_spread))
    ax.axhline(mean_basis_spread + std_basis_spread, color="gold", linestyle="--", linewidth=1.2 * _LINE_WIDTH_SCALE, alpha=_LINE_ALPHA, label=_range_label("±1 std", mean_basis_spread - std_basis_spread, mean_basis_spread + std_basis_spread))
    ax.axhline(mean_basis_spread - std_basis_spread, color="gold", linestyle="--", linewidth=1.2 * _LINE_WIDTH_SCALE, alpha=_LINE_ALPHA)
    ax.axhline(mean_basis_spread + 2 * std_basis_spread, color="red", linestyle="--", linewidth=1.2 * _LINE_WIDTH_SCALE, alpha=_LINE_ALPHA, label=_range_label("±2 std", mean_basis_spread - 2 * std_basis_spread, mean_basis_spread + 2 * std_basis_spread))
    ax.axhline(mean_basis_spread - 2 * std_basis_spread, color="red", linestyle="--", linewidth=1.2 * _LINE_WIDTH_SCALE, alpha=_LINE_ALPHA)
    ax.set_title(f"{ticker} ({security_name}) Spread Basis (CDS Spread - GSpread) over time", fontsize=TITLE_FONTSIZE, fontweight="bold")
    ax.set_ylabel("Spread Basis (bps)", fontsize=LABEL_FONTSIZE, fontweight="bold")
    legend = ax.legend(fontsize=LEGEND_FONTSIZE, loc="upper left", bbox_to_anchor=(1.02, 1))
    _colorize_legend(legend, pct_by_label)

    ax = axes[2]
    label, pct_label, pct = _labeled_stats("Price Basis", plot_data["basis_price"], fmt="${:,.1f}")
    pct_by_label = {}
    ax.plot(plot_data["date"], plot_data["basis_price"], color="tab:blue", alpha=_LINE_ALPHA, linewidth=1.5 * _LINE_WIDTH_SCALE, label=label)
    _annotate_last_point(ax, plot_data["date"], plot_data["basis_price"], color="tab:blue", fmt="${:,.1f}")
    if pct_label:
        ax.plot([], [], color="none", label=pct_label)
        pct_by_label[pct_label] = pct
    ax.axhline(mean_basis_price, color="black", linestyle="--", linewidth=1 * _LINE_WIDTH_SCALE, alpha=_LINE_ALPHA, label=_labeled("Mean", mean_basis_price, fmt="${:,.1f}"))
    ax.axhline(mean_basis_price + std_basis_price, color="gold", linestyle="--", linewidth=1.2 * _LINE_WIDTH_SCALE, alpha=_LINE_ALPHA, label=_range_label("±1 std", mean_basis_price - std_basis_price, mean_basis_price + std_basis_price, fmt="${:,.1f}"))
    ax.axhline(mean_basis_price - std_basis_price, color="gold", linestyle="--", linewidth=1.2 * _LINE_WIDTH_SCALE, alpha=_LINE_ALPHA)
    ax.axhline(mean_basis_price + 2 * std_basis_price, color="red", linestyle="--", linewidth=1.2 * _LINE_WIDTH_SCALE, alpha=_LINE_ALPHA, label=_range_label("±2 std", mean_basis_price - 2 * std_basis_price, mean_basis_price + 2 * std_basis_price, fmt="${:,.1f}"))
    ax.axhline(mean_basis_price - 2 * std_basis_price, color="red", linestyle="--", linewidth=1.2 * _LINE_WIDTH_SCALE, alpha=_LINE_ALPHA)
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("${x:,.1f}"))
    ax.set_title(f"{ticker} ({security_name}) Price Basis (Upfront + Bond Price) over time", fontsize=TITLE_FONTSIZE, fontweight="bold")
    ax.set_ylabel("Price Basis", fontsize=LABEL_FONTSIZE, fontweight="bold")
    legend = ax.legend(fontsize=LEGEND_FONTSIZE, loc="upper left", bbox_to_anchor=(1.02, 1))
    _colorize_legend(legend, pct_by_label)

    date_formatter = mdates.DateFormatter("%d%b%y")
    for ax in axes:
        ax.xaxis.set_major_formatter(date_formatter)
        ax.tick_params(axis="x", labelrotation=0, labelsize=X_TICK_FONTSIZE, labelbottom=True)
        ax.set_xlabel("Date", fontsize=LABEL_FONTSIZE, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 0.74, 0.97])

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    img_bytes = buf.getvalue()
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)

    html = HTML(f'<img src="data:image/png;base64,{img_b64}" style="display:block; max-width:100%;"/>')
    return html, img_bytes
