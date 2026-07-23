"""
Shared bond/CDS data pipeline for the cds-bond-basis notebooks.

Public API
----------
build_bond_history(...) -> (pd.DataFrame, pd.DataFrame)
    Pulls bond prices (Bloomberg) and CDS spreads (Databricks), merges them
    on the "Best_Bond_Proxy_ISIN" mapping, and computes maturity-matched and
    fixed-5Y-matched CDS-vs-bond basis columns.
"""

from datetime import date

import pandas as pd
from bbg_utils.bdh_mm import bdh_mm
from bbg_utils.bdp_mm import bdp_mm
from databricks_utils import get_delta_table_mm

DEFAULT_CDS_MAPPINGS_PATH = r"S:\Structured Credit\Matt Stuff\cds_mappings\cds_mappings_final.csv"
DEFAULT_CACHE_DIR = r"S:\Structured Credit\Matt Stuff\claude_repos\cds_bond_basis\cache"

CDS_TRANSFORM_COLUMNS = [
    "published_date",
    "market_cds_ticker",
    "tenor",
    "cds_maturity",
    "cds_index",
    "tier",
    "currency",
    "doc_clause",
    "running_coupon",
    "par_spread",
    "conv_spread",
    "upfront",
    "cds_assumed_recovery",
    "all_upfront",
]


def next_cds_maturity(maturity):
    """First standard semi-annual CDS maturity (20-Jun or 20-Dec) strictly after `maturity`."""
    maturity = pd.Timestamp(maturity).date()
    candidates = sorted(
        date(y, m, 20)
        for y in (maturity.year, maturity.year + 1)
        for m in (6, 12)
    )
    return next(c for c in candidates if c > maturity)


def build_bond_history(
    cds_mappings_path=DEFAULT_CDS_MAPPINGS_PATH,
    cache_dir=DEFAULT_CACHE_DIR,
    start_date="2025-01-01",
    verbose=True,
):
    """
    Build the merged bond/CDS history panel used across the cds-bond-basis
    notebooks.

    Parameters
    ----------
    cds_mappings_path : str
        Path to the cds_mappings_final.csv file.
    cache_dir : str
        Cache directory passed through to ``get_delta_table_mm``.
    start_date : str
        Earliest CDS/bond history date to pull.
    verbose : bool
        Print intermediate shapes, matching the old notebook cell output.

    Returns
    -------
    bond_history : pd.DataFrame
        One row per (bond security, date) with columns including ``ticker``,
        ``bbg_cds_ticker``, ``cds_index``, ``Markit_ShortName``, ``BOND_ISIN``,
        ``SECURITY_NAME``, ``MATURITY``, ``target_cds_maturity``,
        ``cds_5y_maturity``, ``LAST_PRICE``, ``Benchmark_Spread``, ``G_Spread``,
        ``cds_spread``, ``upfront``, ``cds_spread_5y``, ``cds_spread_5y_fixed``,
        ``basis_spread``, ``basis_spread_5y_fixed``, ``basis_price``.
    cds_mappings_filtered : pd.DataFrame
        The ``ticker`` / ``bbg_cds_ticker`` / ``cds_index`` / ``BOND_ISIN`` /
        ``Markit_ShortName`` mapping used to build ``bond_history``, one row
        per CDS ticker.
    """
    cds_mappings = pd.read_csv(cds_mappings_path, encoding="cp1252")

    cds_transform_interpolated = get_delta_table_mm(
        "teams.structured_credit_abhutra.cds_transform_interpolated",
        cache_dir=cache_dir,
        columns=CDS_TRANSFORM_COLUMNS,
        where=f"published_date >= '{start_date}'",
        filename="cds_transform_interpolated_subset_2025.parquet",
    )
    if verbose:
        print("cds_transform_interpolated", cds_transform_interpolated.shape)

    cds_mappings_filtered = (
        cds_mappings
        .dropna(subset=["Best_Bond_Proxy_ISIN"])
        [["CDS_Ticker", "CDS_Index", "Best_Bond_Proxy_ISIN", "Markit_ShortName", "BBG_5Y_CDS_Ticker"]]
        .rename(columns={
            "CDS_Ticker": "ticker",
            "CDS_Index": "cds_index",
            "Best_Bond_Proxy_ISIN": "BOND_ISIN",
            "BBG_5Y_CDS_Ticker": "bbg_cds_ticker",
        })
        .reset_index(drop=True)
    )

    bonds = cds_mappings_filtered.copy()
    bonds["security"] = bonds["BOND_ISIN"] + " Corp"

    bond_history = bdh_mm(
        bonds["security"],
        ["LAST_PRICE", "BLP_SPRD_TO_BENCH_MID", "BLOOMBERG_MID_G_SPREAD"],
        start_date=start_date,
    )
    if verbose:
        print("bond_history (raw)", bond_history.shape)

    bond_maturity = bdp_mm(bonds["security"], ["MATURITY", "SECURITY_NAME"])
    bond_maturity["MATURITY"] = pd.to_datetime(bond_maturity["MATURITY"])
    bond_maturity["target_cds_maturity"] = bond_maturity["MATURITY"].apply(next_cds_maturity)

    bond_history = bond_history.merge(bond_maturity, on="security", how="left")
    bond_history = bond_history.merge(
        bonds[["security", "ticker", "cds_index", "Markit_ShortName", "BOND_ISIN", "bbg_cds_ticker"]],
        on="security",
        how="left",
    )

    # Maturity-matched CDS spread: the interpolated curve point at the bond's
    # own target_cds_maturity, re-resolved at every date.
    bond_history = bond_history.merge(
        cds_transform_interpolated[["market_cds_ticker", "published_date", "cds_maturity", "conv_spread", "upfront"]],
        left_on=["ticker", "date", "target_cds_maturity"],
        right_on=["market_cds_ticker", "published_date", "cds_maturity"],
        how="left",
    ).drop(columns=["market_cds_ticker", "published_date", "cds_maturity"])

    # Rolling tenor='5Y' CDS spread — used only as a last-value "current 5Y"
    # reference stat, since the absolute maturity it points to drifts over time.
    cds_5y = (
        cds_transform_interpolated[cds_transform_interpolated["tenor"] == "5Y"]
        [["market_cds_ticker", "published_date", "conv_spread"]]
        .rename(columns={"conv_spread": "cds_spread_5y"})
    )
    bond_history = bond_history.merge(
        cds_5y,
        left_on=["ticker", "date"],
        right_on=["market_cds_ticker", "published_date"],
        how="left",
    ).drop(columns=["market_cds_ticker", "published_date"])

    # Fixed 5Y CDS maturity per ticker, taken from each ticker's most recent
    # available tenor='5Y' quote and held constant across the whole time
    # series — since only the 5Y point is reliably liquid, this gives a
    # single consistent maturity to compare against the bond's G-spread
    # throughout its history, rather than re-resolving tenor='5Y' at every
    # date (which drifts to a different absolute maturity as time passes).
    cds_5y_fixed_maturity = (
        cds_transform_interpolated[cds_transform_interpolated["tenor"] == "5Y"]
        .sort_values("published_date")
        .groupby("market_cds_ticker")["cds_maturity"]
        .last()
        .rename("cds_5y_maturity")
        .reset_index()
        .rename(columns={"market_cds_ticker": "ticker"})
    )
    bond_history = bond_history.merge(cds_5y_fixed_maturity, on="ticker", how="left")
    bond_history = bond_history.merge(
        cds_transform_interpolated[["market_cds_ticker", "published_date", "cds_maturity", "conv_spread"]]
            .rename(columns={"conv_spread": "cds_spread_5y_fixed"}),
        left_on=["ticker", "date", "cds_5y_maturity"],
        right_on=["market_cds_ticker", "published_date", "cds_maturity"],
        how="left",
    ).drop(columns=["market_cds_ticker", "published_date", "cds_maturity"])

    bond_history["basis_spread"] = bond_history["conv_spread"] - bond_history["BLOOMBERG_MID_G_SPREAD"]
    bond_history["basis_spread_5y_fixed"] = bond_history["cds_spread_5y_fixed"] - bond_history["BLOOMBERG_MID_G_SPREAD"]
    bond_history["basis_price"] = bond_history["upfront"] + bond_history["LAST_PRICE"]
    bond_history = bond_history.rename(columns={
        "BLP_SPRD_TO_BENCH_MID": "Benchmark_Spread",
        "BLOOMBERG_MID_G_SPREAD": "G_Spread",
        "conv_spread": "cds_spread",
    })

    if verbose:
        print("bond_history (final)", bond_history.shape)

    return bond_history, cds_mappings_filtered
