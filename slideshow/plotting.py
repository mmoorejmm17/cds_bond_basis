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

def _last_valid_value(values):
    """Return the last non-NaN value in a series, or None if all NaN."""
    valid = values.notna()
    if not valid.any():
        return None
    return values[valid].iloc[-1]


def _annotate_last_point(ax, dates, values, color="black", fmt="{:.1f}"):
    """Annotate the last valid point of a series with an arrow + text box."""
    valid = values.notna()
    if not valid.any():
        return
    last_date = dates[valid].iloc[-1]
    last_value = values[valid].iloc[-1]
    ax.annotate(
        fmt.format(last_value),
        xy=(last_date, last_value),
        xytext=(15, 15),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->", color=color),
        color=color,
        fontsize=ANNOTATION_FONTSIZE,
    )


def _labeled(label, value, fmt="{:.1f}"):
    """Append a formatted latest/level value to a legend label, if available."""
    if value is None or pd.isna(value):
        return label
    return f"{label}: {fmt.format(value)}"


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
    fig.suptitle(markit_shortname, fontsize=SUPTITLE_FONTSIZE, fontweight="bold")

    ax = axes[0]
    last_cds_spread = _last_valid_value(plot_data["cds_spread"])
    last_g_spread = _last_valid_value(plot_data["G_Spread"])
    ax.plot(
        plot_data["date"], plot_data["cds_spread"], color="red",
        label=_labeled(f"CDS Spread (Mat: {cds_maturity_used})", last_cds_spread),
    )
    ax.plot(
        plot_data["date"], plot_data["G_Spread"], color="blue",
        label=_labeled("G-Spread", last_g_spread),
    )
    if "cds_spread_5y" in plot_data.columns:
        last_cds_spread_5y = _last_valid_value(plot_data["cds_spread_5y"])
        ax.plot(
            plot_data["date"], plot_data["cds_spread_5y"], color="green", linestyle=":",
            label=_labeled("5Y CDS Spread", last_cds_spread_5y),
        )
        _annotate_last_point(ax, plot_data["date"], plot_data["cds_spread_5y"], color="green")
    _annotate_last_point(ax, plot_data["date"], plot_data["cds_spread"], color="red")
    _annotate_last_point(ax, plot_data["date"], plot_data["G_Spread"], color="blue")
    ax.set_title(f"{ticker} ({security_name}) CDS Spread vs G-Spread over time", fontsize=TITLE_FONTSIZE)
    ax.set_ylabel("Spread (bps)", fontsize=LABEL_FONTSIZE)
    ax.legend(fontsize=LEGEND_FONTSIZE)

    ax = axes[1]
    ax.plot(plot_data["date"], plot_data["basis_spread"], color="tab:blue")
    _annotate_last_point(ax, plot_data["date"], plot_data["basis_spread"], color="tab:blue")
    ax.axhline(mean_basis_spread, color="black", linestyle="--", linewidth=1, alpha=1, label=_labeled("Mean", mean_basis_spread))
    ax.axhline(mean_basis_spread + std_basis_spread, color="gold", linestyle="--", linewidth=1.2, alpha=1, label=_labeled("+1 std", mean_basis_spread + std_basis_spread))
    ax.axhline(mean_basis_spread - std_basis_spread, color="gold", linestyle="--", linewidth=1.2, alpha=1, label=_labeled("-1 std", mean_basis_spread - std_basis_spread))
    ax.axhline(mean_basis_spread + 2 * std_basis_spread, color="red", linestyle="--", linewidth=1.2, alpha=1, label=_labeled("+2 std", mean_basis_spread + 2 * std_basis_spread))
    ax.axhline(mean_basis_spread - 2 * std_basis_spread, color="red", linestyle="--", linewidth=1.2, alpha=1, label=_labeled("-2 std", mean_basis_spread - 2 * std_basis_spread))
    ax.set_title(f"{ticker} ({security_name}) Spread Basis (CDS Spread - GSpread) over time", fontsize=TITLE_FONTSIZE)
    ax.set_ylabel("Spread Basis (bps)", fontsize=LABEL_FONTSIZE)
    ax.legend(fontsize=LEGEND_FONTSIZE)

    ax = axes[2]
    ax.plot(plot_data["date"], plot_data["basis_price"], color="tab:blue")
    _annotate_last_point(ax, plot_data["date"], plot_data["basis_price"], color="tab:blue", fmt="${:,.1f}")
    ax.axhline(mean_basis_price, color="black", linestyle="--", linewidth=1, alpha=1, label=_labeled("Mean", mean_basis_price, fmt="${:,.1f}"))
    ax.axhline(mean_basis_price + std_basis_price, color="gold", linestyle="--", linewidth=1.2, alpha=1, label=_labeled("+1 std", mean_basis_price + std_basis_price, fmt="${:,.1f}"))
    ax.axhline(mean_basis_price - std_basis_price, color="gold", linestyle="--", linewidth=1.2, alpha=1, label=_labeled("-1 std", mean_basis_price - std_basis_price, fmt="${:,.1f}"))
    ax.axhline(mean_basis_price + 2 * std_basis_price, color="red", linestyle="--", linewidth=1.2, alpha=1, label=_labeled("+2 std", mean_basis_price + 2 * std_basis_price, fmt="${:,.1f}"))
    ax.axhline(mean_basis_price - 2 * std_basis_price, color="red", linestyle="--", linewidth=1.2, alpha=1, label=_labeled("-2 std", mean_basis_price - 2 * std_basis_price, fmt="${:,.1f}"))
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("${x:,.1f}"))
    ax.set_title(f"{ticker} ({security_name}) Price Basis (Upfront + Bond Price) over time", fontsize=TITLE_FONTSIZE)
    ax.set_ylabel("Price Basis", fontsize=LABEL_FONTSIZE)
    ax.legend(fontsize=LEGEND_FONTSIZE)

    date_formatter = mdates.DateFormatter("%Y-%m-%d")
    for ax in axes:
        ax.xaxis.set_major_formatter(date_formatter)
        ax.tick_params(axis="x", labelrotation=45, labelsize=X_TICK_FONTSIZE)
        ax.set_xlabel("Date", fontsize=LABEL_FONTSIZE)

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    img_bytes = buf.getvalue()
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)

    html = HTML(f'<img src="data:image/png;base64,{img_b64}" style="display:block; max-width:100%;"/>')
    return html, img_bytes
