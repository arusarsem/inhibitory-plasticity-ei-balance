import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXC = "#2a78d6"
INH = "#eb6834"
NET = "#52514e"
MUTED = "#8a8984"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": "#c9c8c3",
    "axes.labelcolor": "#0b0b0b",
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.labelsize": 9.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e8e7e3",
    "grid.linewidth": 0.6,
    "xtick.color": "#52514e",
    "ytick.color": "#52514e",
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.frameon": False,
    "legend.fontsize": 8.5,
    "lines.linewidth": 2.0,
    "font.family": "DejaVu Sans",
    "savefig.dpi": 160,
    "savefig.bbox": "tight",
})


def save(fig, path):
    fig.savefig(path)
    plt.close(fig)
