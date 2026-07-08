"""
CDS-Bond Basis chart rendering — importable package.

The slideshow engine itself lives in the shared ``slideshow_utils`` package.

Typical notebook usage::

    from slideshow.plotting import plot_bond_cds_basis
    from slideshow_utils    import run_slideshow
"""

from .plotting import plot_bond_cds_basis

__all__ = ["plot_bond_cds_basis"]
