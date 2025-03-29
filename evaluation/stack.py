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
        plot_sig_percentile=(0.5, 0.95),
):
    """Gets bin edges from the signal sample within a given percentile range."""
    signal_data = []
    for sample, config in data.items():
        if config.get('signal', False):
            signal_data.append(getattr(config['var'], column))

    if signal_data:
        signal_data_combined = np.concatenate(signal_data)

        # Compute percentile range
        lower_cutoff = np.percentile(signal_data_combined, plot_sig_percentile[0] * 100)
        upper_cutoff = np.percentile(signal_data_combined, plot_sig_percentile[1] * 100)

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
        compare: dict = None,
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

        values = getattr(data[sample]['var'], column_to_plot)

        if compare is not None:
            values_compare = getattr(compare[sample]['var'], column_to_plot)

        if delta_data is not None:
            # values /= (delta_data[sample]['var'][column_to_plot] + 1e-6)
            values /= 1

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

        if compare is not None:
            ax.hist(
                values_compare,
                weights=data[sample][weight_col],
                histtype='step' if not data[sample]['signal'] else 'stepfilled',
                label=sample,
                # color=data[sample].get('color', None),
                # edgecolor=data[sample].get('color', None),
                linestyle='--',
                alpha=0.75 if not data[sample]['signal'] else 0.5,
                linewidth=1.5,
                density=True,
                **bin_configs
            )

        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        try:
            fwhm_val, fwhm_x1, fwhm_x2, half_max = compute_fwhm(bin_centers, hist_n)
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


def plot_reco_truth_histogram(
        data: dict,
        truth: dict,
        column: str,
        weight_col: str,
        x_title: str,
        save_path: Path | None = None,
        fig_size=(8, 8),
        bins=100,
        x_range: tuple[float, float] | None = None,
        quantiles=(0.01, 0.99),
        extra_data: dict | None = None,
        extra_label: str = "Custom",
):
    # Style config
    style_cfg = {
        "truth": {
            "label": "Truth", "color": "black", "linestyle": "-", "linewidth": 2,
        },
        "reco": {
            "label": "Reco", "color": "blue", "linestyle": "--", "linewidth": 2,
            "alpha": 0.75
        },
        "custom": {
            "label": extra_label, "color": "green", "linestyle": "-.", "linewidth": 2,
            "alpha": 0.75
        },
    }

    save_path.mkdir(parents=True, exist_ok=True)

    density = True
    for sample in data:
        if sample in ['hhttbbSM', 'Ztt', 'VBFhhttbbSM'] and column == "mass":
            density = False

        fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=fig_size, sharex=True, height_ratios=[3, 1])

        # Extract values
        true_vals = getattr(truth[sample]['var'], column)
        reco_vals = getattr(data[sample]['var'], column)
        custom_vals = getattr(extra_data[sample]['var'], column) if extra_data else None

        weights = data[sample][weight_col]
        # weights = np.ones_like(true_vals)

        # Apply quantile cut
        if x_range is not None:
            x_min, x_max = x_range
        else:
            x_min, x_max = np.percentile(reco_vals, [q * 100 for q in quantiles])
            mask = (reco_vals >= x_min) & (reco_vals <= x_max)
            true_vals, reco_vals, weights = true_vals[mask], reco_vals[mask], weights[mask]
            if custom_vals is not None:
                custom_vals = custom_vals[mask]

        # Histogram (density=True)
        hist_args = dict(bins=bins, range=(x_min, x_max), density=density)

        truth_hist, bin_edges = np.histogram(true_vals, weights=weights, **hist_args)
        reco_hist, _ = np.histogram(reco_vals, weights=weights, **hist_args)
        if custom_vals is not None:
            custom_hist, _ = np.histogram(custom_vals, weights=weights, **hist_args)

        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        # Plot distributions
        ax_top.plot(bin_centers, truth_hist, **style_cfg["truth"])
        ax_top.plot(bin_centers, reco_hist, **style_cfg["reco"])

        if custom_vals is not None:
            ax_top.plot(bin_centers, custom_hist, **style_cfg["custom"])

        ax_top.set_ylabel("Density", fontsize=12)
        ax_top.set_title(f"{sample} - {x_title}", fontsize=13)
        ax_top.legend()
        ax_top.grid(True)

        # Ratio and uncertainties (simplified)
        raw_hist_args = dict(bins=bins, range=(x_min, x_max), density=False)

        # Histogram with weights
        truth_raw, _ = np.histogram(true_vals, weights=weights, **raw_hist_args)
        reco_raw, _ = np.histogram(reco_vals, weights=weights, **raw_hist_args)
        if custom_vals is not None:
            custom_raw, _ = np.histogram(custom_vals, weights=weights, **raw_hist_args)

        # Histogram with weights^2 for error estimation
        truth_err2, _ = np.histogram(true_vals, weights=weights ** 2, **raw_hist_args)
        reco_err2, _ = np.histogram(reco_vals, weights=weights ** 2, **raw_hist_args)
        if custom_vals is not None:
            custom_err2, _ = np.histogram(custom_vals, weights=weights ** 2, **raw_hist_args)

        # Normalize
        bin_width = bin_edges[1] - bin_edges[0]
        total_truth = np.sum(truth_raw) * bin_width
        total_reco = np.sum(reco_raw) * bin_width
        if custom_vals is not None:
            total_custom = np.sum(custom_raw) * bin_width

        truth_hist = truth_raw / total_truth
        reco_hist = reco_raw / total_reco
        if custom_vals is not None:
            custom_hist = custom_raw / total_custom

        truth_err = np.sqrt(truth_err2) / total_truth
        reco_err = np.sqrt(reco_err2) / total_reco
        if custom_vals is not None:
            custom_err = np.sqrt(custom_err2) / total_custom

        # Ratio and propagated uncertainty
        def compute_ratio_and_error(num, denom, err_num, err_denom):
            ratio = np.divide(num, denom, out=np.ones_like(num), where=denom > 0)
            err = np.ones_like(ratio)
            mask = (denom > 0) & (num > 0)
            err[mask] = ratio[mask] * np.sqrt(
                (err_num[mask] / num[mask]) ** 2 + (err_denom[mask] / denom[mask]) ** 2
            )
            return ratio, err

        # Ratio and propagated uncertainty (as filled band with central line)
        def plot_ratio_with_band(ax, bin_centers, ratio, error, color, label):
            ax.plot(bin_centers, ratio, color=color, label=label)
            ax.fill_between(bin_centers, ratio - error, ratio + error,
                            color=color, alpha=0.3, linewidth=0)

        reco_ratio, reco_err = compute_ratio_and_error(reco_hist, truth_hist, reco_err, truth_err)
        plot_ratio_with_band(
            ax_bot, bin_centers, reco_ratio, reco_err, style_cfg["reco"]["color"], "Reco / Truth"
        )

        if custom_vals is not None:
            custom_ratio, custom_err = compute_ratio_and_error(custom_hist, truth_hist, custom_err, truth_err)
            plot_ratio_with_band(
                ax_bot, bin_centers, custom_ratio, custom_err, style_cfg["custom"]["color"], f"{extra_label} / Truth"
            )

        ax_bot.axhline(1.0, color="black", linestyle="--")
        ax_bot.set_ylim(0.5, 1.5)
        ax_bot.set_xlabel(x_title, fontsize=12)
        ax_bot.set_ylabel("Ratio", fontsize=11)
        ax_bot.legend()
        ax_bot.grid(True)

        if save_path is not None:
            fig.savefig(save_path / f"{sample}_hist_ratio.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_array_hist_ratio(
        x_truth: np.ndarray,
        x_list: dict[str, np.ndarray],
        label_map: dict[str, str] | None = None,
        weight: np.ndarray | None = None,
        x_title: str = "Variable",
        save_path: Path | None = None,
        fig_size=(8, 8),
        bins=100,
        quantiles=(0.01, 0.99),
        range: tuple[float, float] | None = None,
):
    assert isinstance(x_list, dict), "x_list must be a dictionary"

    # Quantile cut range
    if range is None:
        x_min, x_max = np.percentile(x_truth, [q * 100 for q in quantiles])
    # x_truth = x_truth[(x_truth >= x_min) & (x_truth <= x_max)]
    else:
        x_min, x_max = range

    # Histogram truth
    hist_args = dict(bins=bins, range=(x_min, x_max), density=False)
    truth_hist, bin_edges = np.histogram(x_truth, weights=weight, **hist_args)
    truth_err2, _ = np.histogram(x_truth, weights=(weight ** 2) if weight is not None else None,
                                 **hist_args)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    bin_width = bin_edges[1] - bin_edges[0]
    total_truth = np.sum(truth_hist) * bin_width
    truth_density = truth_hist / total_truth
    truth_err = np.sqrt(truth_err2) / total_truth

    # Plot
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=fig_size, sharex=True, height_ratios=[3, 1])

    ax_top.plot(bin_centers, truth_density, color="black", label="Truth", linewidth=2)

    for name, x_vals in x_list.items():
        label = label_map[name] if label_map else name
        mask = (x_vals >= x_min) & (x_vals <= x_max)
        x_vals = x_vals[mask]
        w = weight[mask] if weight is not None else None

        # Hist and error
        h, _ = np.histogram(x_vals, weights=w, **hist_args)
        h2, _ = np.histogram(x_vals, weights=w ** 2 if w is not None else None, **hist_args)
        total = np.sum(h) * bin_width
        h_density = h / total
        h_err = np.sqrt(h2) / total

        ax_top.plot(bin_centers, h_density, label=label, linewidth=2, linestyle="--")

        # Ratio
        ratio = np.divide(h_density, truth_density, out=np.ones_like(h_density), where=truth_density > 0)
        err = np.ones_like(ratio)
        mask_valid = (truth_density > 0) & (h_density > 0)
        err[mask_valid] = ratio[mask_valid] * np.sqrt(
            (h_err[mask_valid] / h_density[mask_valid]) ** 2 +
            (truth_err[mask_valid] / truth_density[mask_valid]) ** 2
        )

        ax_bot.plot(bin_centers, ratio, label=label)
        ax_bot.fill_between(bin_centers, ratio - err, ratio + err, alpha=0.3)

    ax_top.set_ylabel("Density")
    ax_top.set_title(f"{x_title}")
    ax_top.legend()
    ax_top.grid(True)

    ax_bot.axhline(1.0, color="black", linestyle="--")
    ax_bot.set_ylim(0.5, 1.5)
    ax_bot.set_xlabel(x_title)
    ax_bot.set_ylabel("Ratio")
    ax_bot.grid(True)
    ax_bot.legend()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
