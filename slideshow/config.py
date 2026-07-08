from pathlib import Path

# ── Chart size config ──────────────────────────────────────────────────────────
FIGSIZE  = (7, 10.5)  # (width, height) in inches
FIG_DPI  = 90

FONT_SCALE           = 0.75  # matches the 25%-smaller-fonts pass applied to the notebook
TITLE_FONTSIZE       = 10 * 1.2 * FONT_SCALE
LABEL_FONTSIZE       = 10 * FONT_SCALE
LEGEND_FONTSIZE      = 10 * FONT_SCALE
ANNOTATION_FONTSIZE  = 9 * FONT_SCALE
X_TICK_FONTSIZE      = 10 * 0.7 * FONT_SCALE

# ── Paths ──────────────────────────────────────────────────────────────────────
# Resolves relative to wherever the notebook kernel's CWD is set (typically
# the directory containing the notebook, i.e. notebooks/).
INTERESTING_DIR = Path.cwd() / "interesting_charts"
INTERESTING_DIR.mkdir(exist_ok=True)
