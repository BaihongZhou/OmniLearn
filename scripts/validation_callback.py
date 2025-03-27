import tensorflow as tf
import numpy as np
import logging
import vector
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance, pearsonr
from scipy.spatial.distance import cdist

#### Horovod imports
try:
    import horovod.tensorflow.keras as hvd
except ImportError or ModuleNotFoundError:
    from dummy_hvd import hvd as hvd


def evaluate_distribution(pred_nu, truth_nu, epoch, save_plots, logger=None):
    if hvd.rank() == 0:
        import wandb

    results = {}

    logger.info("[Eval Distribution] Evaluating neutrino distributions")

    for i in range(2):  # for nu1 and nu2
        prefix = f"nu{i + 1}"
        pred = pred_nu[:, i, :]  # shape (N, 4)
        truth = truth_nu[:, i, :]

        logger.info(f"[Eval Distribution] nu {i} --> pred shape: {pred.shape}, truth shape: {truth.shape}")

        for j, name in enumerate(["pt", "eta", "phi"]):
            x = pred[:, j].ravel()
            y = truth[:, j].ravel()

            # Log EMD & Pearson
            results[f"{prefix}/EMD_{name}"] = wasserstein_distance(x, y)
            results[f"{prefix}/Pearson_{name}"] = pearsonr(x, y)[0]

            if save_plots:
                # Define bounded range (1.5× range of truth)
                min_truth, max_truth = np.min(y), np.max(y)
                truth_range = max_truth - min_truth
                buffer = 0.25 * truth_range  # extra 0.5× range split evenly on both sides
                low = min_truth - buffer
                high = max_truth + buffer

                # Create bins including underflow/overflow
                n_bins = 51
                bins = np.linspace(low, high, n_bins - 2)  # exclude 2 bins to add first/last manually
                bins = np.concatenate([[low - 1e10], bins, [high + 1e10]])  # add extreme edges for over/underflow

                # Compute histograms
                hist_pred, _ = np.histogram(x, bins=bins, density=True)
                hist_truth, _ = np.histogram(y, bins=bins, density=True)
                bin_centers = 0.5 * (bins[1:] + bins[:-1])

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

                if logger:
                    logger.info(f"[EvalCallback] --> Saved {prefix} {name} distribution plot")

    return results


def log_vector_distribution(pred_vec, truth_vec, name, epoch, raw_file=None, raw_file_label_map=None, logger=None,
                            weight=None):
    if hvd.rank() == 0:
        import wandb

    logger.info(f"[Log Vector Distribution] Evaluating {name} distribution")

    components = ["pt", "eta", "phi", "mass", "energy"]
    process_ids = np.unique(raw_file) if raw_file is not None else [None]

    for proc_id in process_ids:
        if raw_file is not None:
            mask = (raw_file == proc_id)
            pred = pred_vec[mask]
            truth = truth_vec[mask]
            weights = weight[mask]
            label = raw_file_label_map.get(
                proc_id, f"process_{proc_id}"
            ) if raw_file_label_map else f"process_{proc_id}"
            suffix = f"_{label}"
        else:
            pred = pred_vec
            truth = truth_vec
            suffix = ""

        for k in components:
            x_pred = getattr(pred, k)
            x_truth = getattr(truth, k)

            min_val, max_val = np.min(x_truth), np.max(x_truth)
            span = max_val - min_val
            low, high = min_val - 0.15 * span, max_val + 0.15 * span

            if k == "mass":
                low, high = 40, 250
            if k == "pt":
                low, high = 0, 250
            if k == "energy":
                low, high = 0, 1000

            bins = np.linspace(low, high, 101)
            hist_pred, _ = np.histogram(x_pred, bins=bins, density=True)
            hist_truth, _ = np.histogram(x_truth, bins=bins, density=True, weights=weights)
            bin_centers = 0.5 * (bins[1:] + bins[:-1])

            fig, ax = plt.subplots()
            ax.step(bin_centers, hist_pred, where='mid', label='pred', linewidth=1.5)
            ax.step(bin_centers, hist_truth, where='mid', label='truth', linewidth=1.5, linestyle='--')
            ax.set_title(f"{name} {k}{suffix} @ epoch {epoch}")
            ax.grid(True)
            ax.legend()
            wandb.log({f"{name}/dist_{k}{suffix}": wandb.Image(fig)})
            plt.close(fig)

            if logger:
                logger.info(f"[EvalCallback] --> Saved {name} {k}{suffix} distribution plot")


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
    def __init__(
            self, model, val_dataset, val_dataloader, eval_every=5, logger_name="val_logger",
            extra_list_name=None
    ):
        super().__init__()
        self.model = model
        self.val_dataset = val_dataset
        self.val_dataloader = val_dataloader
        self.eval_every = eval_every
        self.logger = logging.getLogger(logger_name)

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.eval_every != 0:
            return

        max_events = 10000

        part, point, mask, cond, truth_nu = unpack_tfdata(
            val_dataset=self.val_dataset, logger=self.logger, max_events=max_events
        )
        extra = self.val_dataloader.extra[:max_events]
        raw_file = self.val_dataloader.raw_file[:max_events]
        weight = self.val_dataloader.weight[:max_events]

        self.logger.info(f"[EvalCallback] Rank: {hvd.rank()} -- Sampling at epoch {epoch}")
        gen_nu = self.model.generate(
            nsplit=50,
            cond=cond,
            particles=part,
            points=point,
            mask=mask,
            use_tqdm=True,
            candidate=1,
        )

        num_nu = 3

        pred_nu = self.val_dataloader.revert_preprocess_neutrino(gen_nu[:, 0, :]).reshape(-1, num_nu, 4)
        truth_nu = self.val_dataloader.revert_preprocess_neutrino(truth_nu).reshape(-1, num_nu, 4)

        # Convert vector arrays to plain numpy before allgather
        tau1_array = np.stack([extra[:, 2], extra[:, 3], extra[:, 4], extra[:, 5]], axis=1)
        tau2_array = np.stack([extra[:, 6], extra[:, 7], extra[:, 8], extra[:, 9]], axis=1)
        truth_tautau_array = np.stack([extra[:, 10], extra[:, 11], extra[:, 12], extra[:, 13]], axis=1)

        # Gather predicted and truth neutrinos + taus
        pred_nu = hvd.allgather(tf.convert_to_tensor(pred_nu)).numpy()
        truth_nu = hvd.allgather(tf.convert_to_tensor(truth_nu)).numpy()
        tau1_array = hvd.allgather(tf.convert_to_tensor(tau1_array)).numpy()
        tau2_array = hvd.allgather(tf.convert_to_tensor(tau2_array)).numpy()
        truth_tautau_array = hvd.allgather(tf.convert_to_tensor(truth_tautau_array)).numpy()
        raw_file = hvd.allgather(tf.convert_to_tensor(raw_file)).numpy()
        weight = hvd.allgather(tf.convert_to_tensor(weight)).numpy()

        tau1 = vector.arr({
            "pt": tau1_array[:, 0],
            "eta": tau1_array[:, 1],
            "phi": tau1_array[:, 2],
            "mass": tau1_array[:, 3],
        })
        tau2 = vector.arr({
            "pt": tau2_array[:, 0],
            "eta": tau2_array[:, 1],
            "phi": tau2_array[:, 2],
            "mass": tau2_array[:, 3],
        })
        truth_tautau = vector.arr({
            "pt": truth_tautau_array[:, 0],
            "eta": truth_tautau_array[:, 1],
            "phi": truth_tautau_array[:, 2],
            "mass": truth_tautau_array[:, 3],
        })

        truth_nu1 = vector.arr({
            "pt": np.expm1(truth_nu[:, 0, 0]),
            "eta": truth_nu[:, 0, 1],
            "phi": truth_nu[:, 0, 2],
            "energy": np.expm1(truth_nu[:, 0, 3]),
        })

        truth_nu2 = vector.arr({
            "pt": np.expm1(truth_nu[:, 1, 0]),
            "eta": truth_nu[:, 1, 1],
            "phi": truth_nu[:, 1, 2],
            "energy": np.expm1(truth_nu[:, 1, 3]),
        })

        nu1 = vector.arr({
            "pt": np.expm1(pred_nu[:, 0, 0]),
            "eta": pred_nu[:, 0, 1],
            "phi": pred_nu[:, 0, 2],
            "energy": np.expm1(pred_nu[:, 0, 3]),
        })
        nu2 = vector.arr({
            "pt": np.expm1(pred_nu[:, 1, 0]),
            "eta": pred_nu[:, 1, 1],
            "phi": pred_nu[:, 1, 2],
            "energy": np.expm1(pred_nu[:, 1, 3]),
        })
        tautau_diff = vector.arr({
            "px": np.expm1(pred_nu[:, 2, 0]),
            "py": np.expm1(pred_nu[:, 2, 1]),
            "pz": np.expm1(pred_nu[:, 2, 2]),
            "energy": np.expm1(pred_nu[:, 2, 3]),
        })

        tau1_full = tau1 + nu1
        tau2_full = tau2 + nu2
        tautau_nu_pred = tau1_full + tau2_full
        tautau_direct_pred = (tau1 + tau2).to_pxpypzenergy() + tautau_diff
        # tautau_truth = tau1 + truth_nu1 + tau2 + truth_nu2
        tautau_truth = truth_tautau

        if hvd.rank() == 0:
            import wandb

            save_plots = (epoch % self.eval_every == 0)

            self.logger.info(f"[EvalCallback] Rank: {hvd.rank()} -- Total events: {len(tautau_truth.pt)}")

            unique_file_map = {v: k for k, v in self.val_dataloader.unique_file_map.items()}
            self.logger.info(f"[EvalCallback] Rank: {hvd.rank()} -- Unique files: {len(unique_file_map)}")

            results = evaluate_distribution(pred_nu, truth_nu, epoch, logger=self.logger, save_plots=save_plots)
            if save_plots:
                log_vector_distribution(
                    tautau_nu_pred, tautau_truth,
                    name="tautau", epoch=epoch,
                    weight=weight,
                    raw_file=raw_file, raw_file_label_map=unique_file_map,
                    logger=self.logger,
                )

                log_vector_distribution(
                    tautau_direct_pred, tautau_truth,
                    name="tautau_predict", epoch=epoch,
                    weight=weight,
                    raw_file=raw_file, raw_file_label_map=unique_file_map,
                    logger=self.logger,
                )

            # Log everything to Wandb
            wandb.log(results)

        _ = hvd.allreduce(tf.constant(0.0))
