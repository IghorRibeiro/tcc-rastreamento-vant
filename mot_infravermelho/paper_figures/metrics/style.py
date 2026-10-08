import matplotlib.pyplot as plt
from pathlib import Path

def ieee_style():
    """Aplica o estilo IEEE para figuras."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 9,
        'axes.labelsize': 9,
        'axes.titlesize': 10,
        'legend.fontsize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'lines.linewidth': 1.0,
        'lines.markersize': 4,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'axes.grid': False,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

def save_figure(fig, basename: Path, formats=('png', 'pdf')):
    """Salva a figura em PNG (300 dpi) e PDF vetorial."""
    for fmt in formats:
        ext = '.' + fmt
        if fmt == 'png':
            fig.savefig(str(basename) + ext, dpi=300, bbox_inches='tight', pad_inches=0.05)
        else:
            fig.savefig(str(basename) + ext, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)