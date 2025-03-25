import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from pathlib import Path
from sklearn.linear_model import LinearRegression

from matplotlib.colors import LinearSegmentedColormap

# Define your RGB colors (normalized to [0,1])
color1 = [91/255, 181/255, 172/255]   # 0%
color2 = [216/255, 179/255, 101/255]  # 50%
color3 = [222/255, 82/255, 108/255]   # 100%

# Create custom colormap
custom_cmap = LinearSegmentedColormap.from_list(
    'custom_gradient', [color1, color2, color3]
)


def plot_linearity(
        data: dict,
        truth: dict,
        column: str,
        weight_col: str,
        x_title: str,
        save_path: Path | None = None,
        fig_size=(8, 8),
        bins=100,
        quantiles=(0.1, 0.9),
):
    save_path.mkdir(parents=True, exist_ok=True)

    for sample in data:
        fig, ax = plt.subplots(figsize=fig_size)

        pred_vals = data[sample]['var'].to_ptphietamass()[column]
        true_vals = truth[sample]['var'].to_ptphietamass()[column]
        weights = np.ones_like(pred_vals)

        # Clip to quantile range
        x_min, x_max = np.percentile(true_vals, [q * 100 for q in quantiles])
        y_min, y_max = np.percentile(pred_vals, [q * 100 for q in quantiles])

        mask = (pred_vals >= y_min) & (pred_vals <= y_max) & (true_vals >= x_min) & (true_vals <= x_max)
        true_vals = true_vals[mask]
        pred_vals = pred_vals[mask]
        weights = weights[mask]

        # Correlation metrics
        model = LinearRegression().fit(true_vals.reshape(-1, 1), pred_vals, sample_weight=weights)
        slope = model.coef_[0]
        intercept = model.intercept_
        r2 = model.score(true_vals.reshape(-1, 1), pred_vals, sample_weight=weights)
        pearson_corr, _ = pearsonr(pred_vals, true_vals)

        # Heatmap
        h = ax.hist2d(true_vals, pred_vals, bins=bins, cmap=custom_cmap, density=True)
        cb = plt.colorbar(h[3], ax=ax)
        cb.set_label('Density')

        # Identity line
        x_line = np.linspace(min(true_vals), max(true_vals), 100)
        ax.plot(x_line, x_line, 'r--', linewidth=1.5, label='y = x')

        ax.set_xlabel(f"Truth {x_title}", fontsize=12)
        ax.set_ylabel(f"Reco {x_title}", fontsize=12)
        ax.set_title(f"{sample}\n$\\rho$={pearson_corr:.2f}, $R^2$={r2:.2f}, slope={slope:.2f}", fontsize=13)
        ax.grid(True)
        ax.legend(loc='upper left')

        if save_path is not None:
            fig.savefig(save_path / f"{sample}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)