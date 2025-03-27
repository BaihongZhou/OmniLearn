import numpy as np
import matplotlib.pyplot as plt


def plot_reweighting(
        mass: np.ndarray, weights: np.ndarray, num_bins: int = 50, range_min: float = 60.0, range_max: float = 200.0
):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.hist(mass, bins=num_bins, range=(range_min, range_max), color='skyblue', edgecolor='black')
    plt.title('Original Mass Distribution')
    plt.xlabel('Mass')
    plt.ylabel('Counts')

    plt.subplot(1, 2, 2)
    plt.hist(mass, bins=num_bins, weights=weights, range=(range_min, range_max), color='salmon', edgecolor='black')
    plt.title('Reweighted Mass Distribution')
    plt.xlabel('Mass')
    plt.ylabel('Weighted Counts')

    plt.tight_layout()
    plt.show()


def plot_weight_vs_mass(mass: np.ndarray, weights: np.ndarray, range_min: float = 60.0, range_max: float = 200.0):
    mask = (mass >= range_min) & (mass <= range_max)
    mass_filtered = mass[mask]
    weights_filtered = weights[mask]

    plt.figure(figsize=(6, 5))
    plt.scatter(mass_filtered, weights_filtered, s=1, alpha=0.5)
    plt.xlabel('Mass')
    plt.ylabel('Reweighting Factor')
    plt.title('Weight vs. Mass')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def reweight_flat_distribution(mass: np.ndarray, bins: int = 100, range_min: float = 60.0, range_max: float = 200.0):
    hist, bin_edges = np.histogram(mass, bins=bins, range=(range_min, range_max))

    # Avoid division by zero
    hist = np.maximum(hist, 1e-8)
    target = np.ones_like(hist)

    weights_per_bin = target / hist
    bin_indices = np.digitize(mass, bin_edges) - 1
    weights = np.zeros_like(mass)

    in_range_mask = (mass >= range_min) & (mass <= range_max)
    weights[in_range_mask] = weights_per_bin[bin_indices[in_range_mask]]

    # Normalize weights so the total weight in range matches the number of events in range
    total_entries_in_range = np.sum(in_range_mask)
    total_weight_in_range = np.sum(weights[in_range_mask])
    if total_weight_in_range > 0:
        weights[in_range_mask] *= total_entries_in_range / total_weight_in_range

    return weights
