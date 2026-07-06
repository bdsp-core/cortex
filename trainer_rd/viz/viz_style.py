"""Shared plotting aesthetic for the trainer demos/figures (M10.1).

Codifies the conventions of `beautiful_figure_example_general.py` so every
figure and the demo video share one look:
  * sans-serif, large readable fonts (20 pt statics / 16 pt video panels);
  * major + minor grids at low alpha, always below the data;
  * the purple/grey/teal palette with darker edge variants;
  * neutral-grey reference lines, generous zoom-out margins;
  * vector (PDF/SVG) + high-dpi PNG export for statics.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── palette (https://www.color-hex.com/color-palette/106106 basis) ──
TASK_FILL = ['#9671bd', '#7e7e7e', '#77b5b6']        # purple / grey / teal
TASK_EDGE = ['#6a408d', '#4e4e4e', '#378d94']
NEUTRAL = '#8a8a8a'                                   # reference lines
ACCENT_DARK = '#3a3a3a'
# mode strip colours — muted companions of the palette
MODE_FILL = {'bias': '#bd8a71', 'skill': '#77b5b6', 'retention': '#9671bd',
             'eval': '#c9c9c9', 'recert': '#aab8ab'}
GOOD = '#5a9367'                                      # mastery / pass accents
BAD = '#b06060'                                       # fail accents

FIGDIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(FIGDIR, exist_ok=True)


def use_style(font_size=20):
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': font_size,
        'axes.titlesize': font_size,
        'axes.labelsize': font_size,
        'xtick.labelsize': font_size - 2,
        'ytick.labelsize': font_size - 2,
        'legend.fontsize': font_size - 2,
        'figure.titlesize': font_size + 2,
        'axes.edgecolor': ACCENT_DARK,
        'axes.linewidth': 1.2,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
    })


def style_ax(ax):
    """Grids + ordering per the reference example."""
    ax.grid(True, which='major', linestyle='-', linewidth=0.75, alpha=0.25)
    ax.minorticks_on()
    ax.grid(True, which='minor', linestyle='-', linewidth=0.25, alpha=0.15)
    ax.set_axisbelow(True)


def top_legend(ax, ncol, anchor=1.12, **kw):
    """Frameless horizontal legend centred above the axes."""
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, anchor),
              ncol=ncol, frameon=False, **kw)


def save_fig(fig, name, dpi=300):
    """PNG (raster preview) + PDF/SVG (vector, publication)."""
    paths = []
    for ext, kw in (("png", {"dpi": dpi}), ("pdf", {}), ("svg", {})):
        p = os.path.join(FIGDIR, f"{name}.{ext}")
        fig.savefig(p, bbox_inches='tight', **kw)
        paths.append(p)
    return paths
