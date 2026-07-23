from pathlib import Path

# ── Chart size config ──────────────────────────────────────────────────────────
_CHART_SCALE = 0.8  # overall 20%-smaller-chart pass; x-axis tick labels are excluded, see below
FIGSIZE  = (10.5 * _CHART_SCALE, 12 * _CHART_SCALE)  # (width, height) in inches — extra width reserves room for legends placed outside the axes
FIG_DPI  = 105

FONT_SCALE           = 0.75 * _CHART_SCALE  # matches the 25%-smaller-fonts pass applied to the notebook
SUPTITLE_FONTSIZE    = 10 * 1.5 * FONT_SCALE
TITLE_FONTSIZE       = 10 * 1.2 * FONT_SCALE
LABEL_FONTSIZE       = 10 * FONT_SCALE
LEGEND_FONTSIZE      = 10 * FONT_SCALE
ANNOTATION_FONTSIZE  = 9 * FONT_SCALE
X_TICK_FONTSIZE      = 10 * 0.7 * 0.75  # excluded from _CHART_SCALE so x-axis labels stay readable

# ── Paths ──────────────────────────────────────────────────────────────────────
# Resolves relative to wherever the notebook kernel's CWD is set (typically
# the directory containing the notebook, i.e. notebooks/).
INTERESTING_DIR = Path.cwd() / "interesting_charts"
INTERESTING_DIR.mkdir(exist_ok=True)
