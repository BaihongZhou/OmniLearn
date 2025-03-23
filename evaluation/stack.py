import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from pathlib import Path


def linear_interpolation(x1, y1, x2, y2, y_target):
    """Perform linear interpolation to find x at y_target."""
    if y1 == y2:
        return (x1 + x2) / 2  # Avoid division by zero
    return x1 + (y_target - y1) * (x2 - x1) / (y2 - y1)


def compute_fwhm(bin_centers, hist):
    """Compute the Full Width at Half Maximum (FWHM) with linear interpolation."""
    max_val = np.max(hist)  # Peak value
    half_max = max_val / 2  # Half of the peak value

    # Find indices where hist >= half_max
    indices = np.where(hist >= half_max)[0]

    if len(indices) < 2:
        raise ValueError("Not enough points above half-maximum to compute FWHM")

    # Find left and right crossing points using interpolation
    left_idx = indices[0] - 1  # One bin before crossing
    right_idx = indices[-1]  # First bin above crossing

    # Interpolate to find more accurate x1 and x2
    x1 = linear_interpolation(
        bin_centers[left_idx], hist[left_idx],
        bin_centers[left_idx + 1], hist[left_idx + 1], half_max)

    x2 = linear_interpolation(
        bin_centers[right_idx], hist[right_idx],
        bin_centers[right_idx - 1], hist[right_idx - 1], half_max)

    fwhm = x2 - x1  # Compute the FWHM
    return fwhm, x1, x2, half_max


def get_bin_edges_from_signal(
        data,
        column,
        bins=50,
        plot_sig_percentile=(5, 95),
):
    """Gets bin edges from the signal sample within a given percentile range."""
    signal_data = []
    for sample, config in data.items():
        if config.get('signal', False):
            signal_data.append(config['var'][column])

    if signal_data:
        signal_data_combined = np.concatenate(signal_data)

        # Compute percentile range
        lower_cutoff = np.percentile(signal_data_combined, plot_sig_percentile[0])
        upper_cutoff = np.percentile(signal_data_combined, plot_sig_percentile[1])

        # Filter data within percentile range
        filtered_data = signal_data_combined[
            (signal_data_combined >= lower_cutoff) & (signal_data_combined <= upper_cutoff)
            ]

        # Compute bin edges within this range
        bin_edges = np.histogram_bin_edges(filtered_data, bins=bins)
        return bin_edges

    return None


def plot_hist(
        data: dict,
        column_to_plot: str,
        weight_col: str,
        bins: int,
        x_range: tuple[float, float] | None = None,
        plot_sig_percentile: tuple[float, float] = (5, 95),
        fig_size=(10, 8),
        x_title: str = None,
        save_path: Path | None = None,
        delta_data: dict = None,
):
    bin_configs = {}
    # Get bin edges
    if plot_sig_percentile is not None and delta_data is None:
        bin_configs['bins'] = get_bin_edges_from_signal(data, column_to_plot, bins, plot_sig_percentile)
    else:
        bin_configs['bins'] = bins
        if x_range is not None:
            bin_configs['range'] = x_range

    fig, ax = plt.subplots(1, 1, figsize=fig_size)

    for sample in data.keys():

        values = data[sample]['var'][column_to_plot]
        if delta_data is not None:
            values /= (delta_data[sample]['var'][column_to_plot] + 1e-6)

        # print(f"Sample: {sample}, Num Values: {len(values)}, Sum Weights: {np.sum(data[sample][weight_col])}")

        hist_n, bin_edges, hist_obj = ax.hist(
            values,
            weights=data[sample][weight_col],
            histtype='step' if not data[sample]['signal'] else 'stepfilled',
            label=sample,
            color=data[sample].get('color', None),
            edgecolor=data[sample].get('color', None),
            alpha=0.75 if not data[sample]['signal'] else 0.5,
            linewidth=1.5,
            density=True,
            **bin_configs
        )

        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        try:
            fwhm_val, fwhm_x1, fwhm_x2, half_max = compute_fwhm(bin_centers, hist_n)
            # ax.axvline(fwhm_x1, color=data[sample].get('color', 'grey'), linestyle='--', linewidth=1.5, alpha=0.5)
            # ax.axvline(fwhm_x2, color=data[sample].get('color', 'grey'), linestyle='--', linewidth=1.5, alpha=0.5)

            # arrow_props = dict(arrowstyle="<->", color="gray", linewidth=1.5)
            # ax.annotate("", xy=(fwhm_x1, half_max), xytext=(fwhm_x2, half_max), arrowprops=arrow_props)
            # ax.text(
            #     (fwhm_x1 + fwhm_x2) / 2, half_max * 0.95,
            #     f"$\\Gamma = {fwhm_val:.2f}$",
            #     va='top', ha="center", fontsize=12,
            #     color="gray"
            # )

            if isinstance(hist_obj, list):  # stepfilled returns a list of patches
                hist_obj = hist_obj[0]  # Get the first patch object

            hist_obj.set_label(f"{sample} ($\\Gamma = {fwhm_val:.2f}$)")
        except ValueError:
            pass

    ax.set_ylabel('Yields', fontsize=12)
    ax.set_xlabel(x_title, fontsize=12)
    ax.legend(fontsize=12, ncol=2, loc='upper right')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Get the current y-axis limits
    ymin, ymax = ax.get_ylim()
    factor = 1.25
    ax.set_ylim(ymin, ymax * factor)

    # plt.show()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.close(fig)
