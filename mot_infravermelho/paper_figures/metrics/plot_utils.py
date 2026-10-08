import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .correlation import pearson_r, spearman_rho, r_squared, confidence_interval
from .style import ieee_style

def plot_scatter_metric(ax, df, x_col, y_col, label_x, label_y,
                        color='black', marker='o', label='',
                        add_regression=True, add_ci=True,
                        annotate_points=True, seq_col='sequence'):
    """
    Desenha scatter plot com regressão linear, IC (95%), anotações das sequências
    e estatísticas (Pearson r, Spearman ρ, R², p-valor).
    """
    ieee_style()

    mask = df[x_col].notna() & df[y_col].notna()
    x = df.loc[mask, x_col].values
    y = df.loc[mask, y_col].values
    if len(x) < 2:
        return

    ax.scatter(x, y, color=color, marker=marker, alpha=0.7, label=label)

    if add_regression and len(x) >= 2:
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(x), max(x), 50)
        ax.plot(x_line, p(x_line), color=color, linestyle='--', alpha=0.5)

        if add_ci and len(x) >= 3:
            ci_func = confidence_interval(x, y)
            if ci_func is not None:
                lower, upper = ci_func(x_line)
                ax.fill_between(x_line, lower, upper, color=color, alpha=0.15)

        r, pval = pearson_r(x, y)
        rho, pval_rho = spearman_rho(x, y)
        r2 = r_squared(x, y)
        stats_text = (f"Pearson r = {r:.3f} (p={pval:.3f})\n"
                      f"Spearman ρ = {rho:.3f} (p={pval_rho:.3f})\n"
                      f"R² = {r2:.3f}")
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                fontsize=6, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    if annotate_points:
        for _, row in df.iterrows():
            if pd.notna(row[x_col]) and pd.notna(row[y_col]):
                seq = row[seq_col]
                label = seq.replace('UAV', 'U') if 'UAV' in seq else seq
                ax.annotate(label, (row[x_col], row[y_col]),
                            xytext=(3, 3), textcoords='offset points',
                            fontsize=5, alpha=0.8)

    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)