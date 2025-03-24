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
            results[f"{prefix}/EMD_{name}"] = wasserstein_distance(x, y)
            results[f"{prefix}/Pearson_{name}"] = pearsonr(x, y)[0]

            # Plot and log to wandb
            fig, ax = plt.subplots()
            ax.hist(x, bins=60, alpha=0.5, label='pred', density=True)
            ax.hist(y, bins=60, alpha=0.5, label='truth', density=True)
            ax.set_title(f"{prefix} {name} dist @ epoch {epoch}")
            ax.legend()
            ax.grid()
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
        met_list.append(x_batch['input_jet'][:num_to_add].numpy())
        truth_list.append(cond_batch[:num_to_add].numpy())

        total_events += num_to_add
        if total_events >= max_events:
            break

    part = np.concatenate(part_list, axis=0)
    point = np.concatenate(point_list, axis=0)
    mask = np.concatenate(mask_list, axis=0)
    met = np.concatenate(met_list, axis=0)
    truth_nu = np.concatenate(truth_list, axis=0).reshape(-1, 2, 3)

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

        reco_nu = self.val_dataloader.revert_preprocess_neutrino(gen_nu[:, 0, :])  # shape: (N, 6)
        pred_nu = reco_nu.reshape(-1, 2, 3)  # shape: (N, 2, 3)
        truth_nu = truth_nu

        results = evaluate_distribution(pred_nu, truth_nu, epoch)

        # Log everything to Wandb
        wandb.log(results)
