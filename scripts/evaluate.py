import os
import sys
from pathlib import Path

current_file_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_file_path)

import numpy as np
import tensorflow as tf
import argparse
from PET_jetnet import PET_jetnet
import utils
import logging
import glob

#### Horovod imports
try:
    import horovod.tensorflow.keras as hvd
except ImportError or ModuleNotFoundError:
    from dummy_hvd import hvd as hvd

from configs.global_config import load_config
import configs.global_config as config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train the PET model on JetNet datasets.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the configuration file")

    return parser.parse_args()


def load_data_and_model(eval_config, sample_config, model_config):
    if not Path(eval_config['model_path']).exists():
        raise ValueError(f"Model path {eval_config['model_path']} does not exist.")

    eval_loader = utils.TauReconDataLoader(
        path=Path(eval_config['processed_folder']) / f"{sample_config['tag']}_evaluation.hdf5",
        sample_norm=sample_config['normalization'],
        batch_size=4096,
        rank=hvd.rank(),
        size=hvd.size(),
        nevts=sample_config['n_events'] if sample_config['n_events'] > 0 else None,
    )

    model_config = config.cfg['model']
    model_config.pop('ckpt_save_path')
    model = PET_jetnet(
        num_feat=eval_loader.num_feat,
        num_jet=eval_loader.num_jet,
        num_classes=eval_loader.num_classes,
        num_part=eval_loader.num_part,
        model_name=None,

        **model_config
    )

    model.load_weights(eval_config['model_path'])

    return eval_loader, model


def sample_data(eval_dataloader, model, sample_name, raw_particle_list, split: bool = False):
    """ Sample data using the model and save to file. """
    part, point, mask, met, truth_nu = eval_dataloader.make_eval_data(preprocess=True)
    extra_info = eval_dataloader.extra
    raw_file = eval_dataloader.raw_file

    nsplit = 50
    raw_nu_candidates = model.generate(
        nsplit,
        met, part, point, mask,
        use_tqdm=hvd.rank() == 0,
        candidate=1,
    )

    final_neutrinos = [
        eval_dataloader.revert_preprocess_neutrino(raw_nu_candidates[:, i]).reshape(-1, 1, 6)
        for i in range(raw_nu_candidates.shape[1])
    ]
    final_neutrinos = np.concatenate(final_neutrinos, axis=1)

    try:
        logger.info(f"[Rank {hvd.rank()}] Starting allgather with shape {final_neutrinos.shape}")
        final_neutrinos = hvd.allgather(final_neutrinos)
        extra_info = hvd.allgather(tf.constant(extra_info)).numpy()
        raw_file = hvd.allgather(tf.constant(raw_file)).numpy()
    except Exception as e:
        logger.error(f"[Rank {hvd.rank()}] Error during allgather: {e}")
        raise

    if hvd.rank() == 0:
        logger.info(f"final neutrinos shape: {final_neutrinos.shape}")
        logger.info(f"extra info shape: {extra_info.shape}")
        logger.info(f"raw file shape: {raw_file.shape}")

        if not split:
            data_dict = {
                'recon_nu1': final_neutrinos[:, :, :3],
                'recon_nu2': final_neutrinos[:, :, 3:],
            }
            data_dict.update({
                f"extra_{i}": extra_info[:, i]
                for i in range(extra_info.shape[1])
            })

            np.savez(sample_name, **data_dict)

            logger.info(f"Saved {sample_name}")
        else:
            file_list = eval_dataloader.unique_file_map
            for file in file_list:
                mask = raw_file == file_list[file]
                data_dict = {
                    'nu1': final_neutrinos[mask][:, :, :3],
                    'nu2': final_neutrinos[mask][:, :, 3:],
                }
                data_dict.update({
                    f"extra_{i}": extra_info[mask, i]
                    for i in range(extra_info.shape[1])
                })
                np.savez(sample_name.with_name(file + ".npz"), **data_dict)
                logger.info(f"Saved {sample_name.with_name(file + '.npz')}")


def main():
    utils.setup_gpus()
    if hvd.rank() == 0:
        logging.info("Horovod and GPUs initialized successfully.")
    flags = parse_arguments()

    load_config(flags.config)

    if hvd.rank() == 0:
        logging.info("Sampling the data without boost.")

    if "evaluation" in config.cfg:
        eval_config = config.cfg["evaluation"]
        eval_save_path = Path(eval_config['eval_save_path'])
        eval_loader, model = load_data_and_model(
            eval_config=eval_config,
            sample_config=config.cfg["sample"],
            model_config=config.cfg["model"],
        )

        if hvd.rank() == 0: logging.info("Sampling start!")

        eval_save_path.mkdir(parents=True, exist_ok=True)
        sample_data(
            eval_loader, model, eval_save_path / "merged.npz",
            raw_particle_list=[
                particle
                for cat, p_cfg in config.cfg["features"].items()
                if cat in ["tau_vis", "jet"]
                for particle in p_cfg["particles"]
            ],
            split=config.cfg["evaluation"].get("split_save", False)
        )
    else:
        logging.warning("Evaluation mode not implemented.")


if __name__ == '__main__':
    main()
