import tensorflow as tf
import numpy as np
import wandb
import logging
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance, pearsonr
from scipy.spatial.distance import cdist


def compute_mmd_rbf(X, Y, gamma=1.0):
    """
    Maximum Mean Discrepancy with RBF kernel.
    Args:
        X, Y: arrays of shape (n_samples, n_features)
        gamma: 1 / (2 * sigma^2), bandwidth of RBF
    Returns:
        MMD^2(X, Y)
    """
    XX = np.exp(-cdist(X, X, 'sqeuclidean') * gamma)
    YY = np.exp(-cdist(Y, Y, 'sqeuclidean') * gamma)
    XY = np.exp(-cdist(X, Y, 'sqeuclidean') * gamma)

    return np.mean(XX) + np.mean(YY) - 2 * np.mean(XY)


def evaluate_distribution(pred_nu, truth_nu, epoch):
    results = {}

    for i in range(2):  # for nu1 and nu2
        prefix = f"nu{i + 1}"
        pred = pred_nu[:, i, :]  # shape (N, 3)
        truth = truth_nu[:, i, :]

        # MMD
        mmd = compute_mmd_rbf(pred, truth, gamma=1.0)
        results[f"{prefix}/MMD"] = mmd

        for j, name in enumerate(["pt", "eta", "phi"]):
            x = pred[:, j].ravel()
            y = truth[:, j].ravel()

            # Define bounded range (1.5× range of truth)
            min_truth, max_truth = np.min(y), np.max(y)
            truth_range = max_truth - min_truth
            buffer = 0.25 * truth_range  # extra 0.5× range split evenly on both sides
            low = min_truth - buffer
            high = max_truth + buffer

            # Create bins including underflow/overflow
            n_bins = 60
            bins = np.linspace(low, high, n_bins - 2)  # exclude 2 bins to add first/last manually
            bins = np.concatenate([[low - 1e10], bins, [high + 1e10]])  # add extreme edges for over/underflow

            # Compute histograms
            hist_pred, _ = np.histogram(x, bins=bins, density=True)
            hist_truth, _ = np.histogram(y, bins=bins, density=True)
            bin_centers = 0.5 * (bins[1:] + bins[:-1])

            # Log EMD & Pearson
            results[f"{prefix}/EMD_{name}"] = wasserstein_distance(x, y)
            results[f"{prefix}/Pearson_{name}"] = pearsonr(x, y)[0]

            # Plot
            fig, ax = plt.subplots()
            ax.step(bin_centers, hist_pred, where='mid', label='pred', linewidth=1.5)
            ax.step(bin_centers, hist_truth, where='mid', label='truth', linewidth=1.5)
            ax.set_xlim([low, high])
            ax.set_title(f"{prefix} {name} dist @ epoch {epoch}")
            ax.legend()
            ax.grid(True)

            wandb.log({f"{prefix}/dist_{name}": wandb.Image(fig)})
            plt.close(fig)

    return results


def unpack_tfdata(val_dataset, max_events=10000, logger=None):
    part_list, point_list, mask_list, met_list, truth_list = [], [], [], [], []
    total_events = 0

    for (x_batch, cond_batch) in val_dataset:
        num = x_batch['input_features'].shape[0]
        if total_events + num > max_events:
            num_to_add = max_events - total_events
        else:
            num_to_add = num

        if logger:
            logger.info(f"Adding {num_to_add} events to the validation set")

        part_list.append(x_batch['input_features'][:num_to_add].numpy())
        point_list.append(x_batch['input_points'][:num_to_add].numpy())
        mask_list.append(x_batch['input_mask'][:num_to_add].numpy())
        truth_list.append(x_batch['input_jet'][:num_to_add].numpy())
        met_list.append(cond_batch[:num_to_add].numpy())

        total_events += num_to_add
        if total_events >= max_events:
            break

    part = np.concatenate(part_list, axis=0)
    point = np.concatenate(point_list, axis=0)
    mask = np.concatenate(mask_list, axis=0)
    met = np.concatenate(met_list, axis=0)
    truth_nu = np.concatenate(truth_list, axis=0)

    return part, point, mask, met, truth_nu


class DiffusionValidationCallback(tf.keras.callbacks.Callback):
    def __init__(self, model, val_dataset, val_dataloader, eval_every=5, logger_name="val_logger"):
        super().__init__()
        self.model = model
        self.val_dataset = val_dataset
        self.val_dataloader = val_dataloader
        self.eval_every = eval_every
        self.logger = logging.getLogger(logger_name)

        try:
            import horovod.tensorflow.keras as hvd
            self.is_main = hvd.rank() == 0
        except ImportError:
            self.is_main = True

    def on_epoch_end(self, epoch, logs=None):
        if not self.is_main or (epoch % self.eval_every != 0):
            return

        part, point, mask, cond, truth_nu = unpack_tfdata(val_dataset=self.val_dataset, logger=self.logger)

        self.logger.info(f"[EvalCallback] Sampling at epoch {epoch}")
        gen_nu = self.model.generate(
            nsplit=50,
            cond=cond,
            particles=part,
            points=point,
            mask=mask,
            use_tqdm=True,
            candidate=1,
        )

        pred_nu = self.val_dataloader.revert_preprocess_neutrino(gen_nu[:, 0, :]).reshape(-1, 2, 3) # shape: (N, 6)
        truth_nu = self.val_dataloader.revert_preprocess_neutrino(truth_nu).reshape(-1, 2, 3)

        results = evaluate_distribution(pred_nu, truth_nu, epoch)

        # Log everything to Wandb
        wandb.log(results)
