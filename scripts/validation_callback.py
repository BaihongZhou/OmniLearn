import tensorflow as tf
import numpy as np
import logging
import vector
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance, pearsonr
from scipy.spatial.distance import cdist

from matplotlib.colors import LinearSegmentedColormap

# Define your RGB colors (normalized to [0,1])
color0 = [255 / 255, 255 / 255, 255 / 255]  # #ffffff
color1 = [91 / 255, 181 / 255, 172 / 255]  # #5bb5ac
color2 = [216 / 255, 179 / 255, 101 / 255]  # #d8b365
color3 = [222 / 255, 82 / 255, 108 / 255]  # #de526c

# Create custom colormap
custom_cmap = LinearSegmentedColormap.from_list(
    'custom_gradient', [color0, color1, color2, color3]
)

#### Horovod imports
try:
    import horovod.tensorflow.keras as hvd
except ImportError or ModuleNotFoundError:
    from dummy_hvd import hvd as hvd


def inverse_signed_log1p(y):
    return np.sign(y) * (np.expm1(np.abs(y)))


def evaluate_distribution(nu1, nu2, truth_nu1, truth_nu2, epoch, save_plots, logger=None, weights=None):
    if hvd.rank() == 0:
        import wandb

    results = {}
    logger.info("[Eval Distribution] Evaluating neutrino distributions")

    for i, (pred, truth) in enumerate([(nu1, truth_nu1), (nu2, truth_nu2)]):
        prefix = f"nu{i + 1}"

        logger.info(f"[Eval Distribution] {prefix} --> pred shape: {pred.shape}, truth shape: {truth.shape}")

        for name in ["pt", "eta", "phi", "energy"]:
            x = getattr(pred, name)
            y = getattr(truth, name)

            # Remove NaNs or Infs
            mask = np.isfinite(x) & np.isfinite(y)
            n_bad = len(x) - np.count_nonzero(mask)

            if n_bad > 0 and logger:
                logger.warning(f"[Eval] Removed {n_bad} non-finite values in {prefix} {name}")

            x = x[mask]
            y = y[mask]
            w = weights[mask] if weights is not None else None

            # Skip Pearson if variance is zero
            if len(x) == 0 or np.std(x) == 0 or np.std(y) == 0:
                if logger:
                    logger.warning(f"[Eval] Zero std or empty input in {prefix} {name}, skipping Pearson.")
                results[f"{prefix}/Pearson_{name}"] = np.nan
            else:
                results[f"{prefix}/Pearson_{name}"] = pearsonr(x, y)[0]

            try:
                results[f"{prefix}/EMD_{name}"] = wasserstein_distance(x, y)
            except Exception as e:
                if logger:
                    logger.warning(f"[Eval] Failed EMD for {prefix} {name}: {str(e)}")
                results[f"{prefix}/EMD_{name}"] = np.nan

            if save_plots:
                min_val, max_val = np.min(y), np.max(y)
                span = max_val - min_val
                low, high = min_val - 0.25 * span, max_val + 0.25 * span

                if name == "mass":
                    low, high = 40, 250
                if name == "pt":
                    low, high = 0, 250
                if name == "energy":
                    low, high = 0, 750

                bins = np.linspace(low, high, 51)
                hist_pred, _ = np.histogram(x, bins=bins, density=True)
                hist_truth, _ = np.histogram(y, bins=bins, density=True, weights=w)
                bin_centers = 0.5 * (bins[1:] + bins[:-1])

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


def log_vector_distribution(
        pred_vec, truth_vec, name, epoch, raw_file=None, raw_file_label_map=None, logger=None,
        weight=None
):
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
            weights = weight

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


def log_dR_distribution(dR_pred_1, dR_pred_2, dR_truth_1, dR_truth_2, epoch, logger=None, weights=None):
    if hvd.rank() == 0:
        import wandb

    results = {}
    for i, (dR_pred, dR_truth) in enumerate([
        (dR_pred_1, dR_truth_1),
        (dR_pred_2, dR_truth_2)
    ]):
        if dR_pred is None:
            continue

        dR_pred = np.ravel(dR_pred)
        dR_truth = np.ravel(dR_truth)

        name = f"nu_tau_{i + 1}"

        results[f"other/Pearson_{name}"] = pearsonr(dR_pred, dR_truth)[0]
        results[f"other/EMD_{name}"] = wasserstein_distance(dR_pred, dR_truth)

        min_val, max_val = np.min(dR_truth), np.max(dR_truth)
        span = max_val - min_val
        low, high = min_val - 0.15 * span, max_val + 0.15 * span
        # low = max(low, 0)

        bins = np.linspace(low, high, 101)
        hist_pred, _ = np.histogram(dR_pred, bins=bins, density=True)
        hist_truth, _ = np.histogram(dR_truth, bins=bins, density=True, weights=weights)
        bin_centers = 0.5 * (bins[1:] + bins[:-1])

        fig, ax = plt.subplots()
        ax.step(bin_centers, hist_pred, where='mid', label='pred', linewidth=1.5)
        ax.step(bin_centers, hist_truth, where='mid', label='truth', linewidth=1.5, linestyle='--')
        ax.set_title(f"{name} @ epoch {epoch}")
        ax.grid(True)
        ax.legend()
        wandb.log({f"other/{name}": wandb.Image(fig)})
        plt.close(fig)

        fig, ax = plt.subplots()
        h = ax.hist2d(dR_truth, dR_pred, bins=100, range=[[low, high], [low, high]], cmap=custom_cmap, density=True)
        fig.colorbar(h[3], ax=ax)
        ax.set_xlabel("Truth")
        ax.set_ylabel("Prediction")
        ax.set_title(f"{name} 2D Histogram @ epoch {epoch}")
        ax.grid(True)
        wandb.log({f"other/2Dhist_{name}": wandb.Image(fig)})
        plt.close(fig)

        if logger:
            logger.info(f"[EvalCallback] --> Saved {name} distribution plot")

    wandb.log(results)


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

        if self.val_dataloader.weight is not None:
            weight = self.val_dataloader.weight[:max_events]
        else:
            weight = np.ones_like(raw_file)

        self.logger.info(f"[EvalCallback] Rank: {hvd.rank()} -- Sampling at epoch {epoch}")
        gen_nu = self.model.generate(
            nsplit=50,
            cond=cond,
            particles=part,
            points=point,
            mask=mask,
            use_tqdm=True,
            candidate=1,
            data_loader_target_mean=self.val_dataloader.mean_jet,
            data_loader_target_std=self.val_dataloader.std_jet,
        )

        # num_nu = 2

        # pred_nu = self.val_dataloader.revert_preprocess_neutrino(gen_nu[:, 0, :]).reshape(-1, num_nu, 4)
        pred_nu = gen_nu[:, 0, :]
        truth_nu = self.val_dataloader.revert_preprocess_neutrino(truth_nu)

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

        nu1_start = 1
        nu2_start = 4
        truth_nu1 = vector.arr({
            "pt": np.expm1(truth_nu[:, nu1_start]),
            "eta": truth_nu[:, nu1_start + 1],
            "phi": truth_nu[:, nu1_start + 2],
            "mass": np.zeros_like(truth_nu[:, nu1_start]),
        })
        truth_nu2 = vector.arr({
            "pt": np.expm1(truth_nu[:, nu2_start]),
            "eta": truth_nu[:, nu2_start + 1],
            "phi": truth_nu[:, nu2_start + 2],
            "mass": np.zeros_like(truth_nu[:, 2]),
        })

        nu1 = vector.arr({
            "pt": np.expm1(pred_nu[:, nu1_start]),
            "eta": pred_nu[:, nu1_start + 1],
            "phi": pred_nu[:, nu1_start + 2],
            "mass": np.zeros_like(pred_nu[:, 2]),
        })
        nu2 = vector.arr({
            "pt": np.expm1(pred_nu[:, nu2_start]),
            "eta": pred_nu[:, nu2_start + 1],
            "phi": pred_nu[:, nu2_start + 2],
            "mass": np.zeros_like(pred_nu[:, 2]),
        })

        # dR_nu_tau_1 = pred_nu[:, 0]
        # dR_nu_tau_2 = pred_nu[:, 1]
        # truth_dR_nu_tau_1 = tau1.deltaR(truth_nu1)
        # truth_dR_nu_tau_2 = tau2.deltaR(truth_nu2)
        # truth_dR_nu_tau_1 = truth_nu[:, 0]
        # truth_dR_nu_tau_2 = truth_nu[:, 1]
        mass_pred = pred_nu[:, 0]
        mass_truth = truth_nu[:, 0]

        mass_raw_pred = self.val_dataloader.mass_transform.inverse_transform(mass_pred.reshape(-1, 1))
        mass_raw_truth = self.val_dataloader.mass_transform.inverse_transform(mass_truth.reshape(-1, 1))

        tau1_full = tau1 + nu1
        tau2_full = tau2 + nu2
        tautau_nu_pred = tau1_full + tau2_full
        # tautau_direct_pred = (tau1 + tau2).to_pxpypzenergy() + tautau_diff
        # tautau_truth = tau1 + truth_nu1 + tau2 + truth_nu2
        tautau_truth = truth_tautau

        if hvd.rank() == 0:
            import wandb

            save_plots = (epoch % self.eval_every == 0)

            self.logger.info(f"[EvalCallback] Rank: {hvd.rank()} -- Total events: {len(tautau_truth.pt)}")

            unique_file_map = {v: k for k, v in self.val_dataloader.unique_file_map.items()}
            self.logger.info(f"[EvalCallback] Rank: {hvd.rank()} -- Unique files: {len(unique_file_map)}")

            results = evaluate_distribution(
                nu1=nu1, nu2=nu2,
                truth_nu1=truth_nu1, truth_nu2=truth_nu2,
                epoch=epoch,
                save_plots=save_plots,
                logger=self.logger,
                weights=weight,
            )
            if save_plots:
                log_vector_distribution(
                    tautau_nu_pred, tautau_truth,
                    name="tautau", epoch=epoch,
                    weight=weight,
                    raw_file=raw_file, raw_file_label_map=unique_file_map,
                    logger=self.logger,
                )

                log_dR_distribution(
                    # dR_pred_1=dR_nu_tau_1,
                    # dR_pred_2=dR_nu_tau_2,
                    # dR_truth_1=truth_dR_nu_tau_1,
                    # dR_truth_2=truth_dR_nu_tau_2,
                    dR_pred_1=mass_pred,
                    dR_pred_2=mass_raw_pred,
                    dR_truth_1=mass_truth,
                    dR_truth_2=mass_raw_truth,
                    epoch=epoch,
                    logger=self.logger,
                    weights=weight,
                )

                # log_vector_distribution(
                #     tautau_direct_pred, tautau_truth,
                #     name="tautau_predict", epoch=epoch,
                #     weight=weight,
                #     raw_file=raw_file, raw_file_label_map=unique_file_map,
                #     logger=self.logger,
                # )

            # Log everything to Wandb
            wandb.log(results)

        _ = hvd.allreduce(tf.constant(0.0))
