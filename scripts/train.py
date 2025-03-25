import os
import sys
from pathlib import Path

current_file_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_file_path)

import numpy as np

import argparse
import logging
import wandb
from wandb.integration.keras import WandbMetricsLogger

# Custom local imports
import utils
from PET_jetnet import PET_jetnet
from validation_callback import DiffusionValidationCallback

# Keras imports
from tensorflow.keras.optimizers import schedules, Lion
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from configs.global_config import load_config
import configs.global_config as config

#### Horovod imports
try:
    import horovod.tensorflow.keras as hvd
except ImportError or ModuleNotFoundError:
    from dummy_hvd import hvd as hvd

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train the PET model on JetNet datasets.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the configuration file")

    return parser.parse_args()


def get_data_loader():
    dataset = None
    if config.cfg['dataset'] == 'bbtautau':
        dataset = [
            utils.TauReconDataLoader(
                path=Path(config.cfg['sample']['base_folder']) / f"{config.cfg['sample']['tag']}_{dataset_type}.hdf5",
                sample_norm=config.cfg['sample']['normalization'],
                batch_size=config.cfg['training']['batch_size'],
                rank=hvd.rank(),
                size=hvd.size(),
                nevts=config.cfg['sample']['n_events'] if config.cfg['sample']['n_events'] > 0 else None,
                sample_weight_map=config.cfg['features']['weight']['sample'],
            )

            for dataset_type in ['train', 'test']
        ]

    return dataset[0], dataset[1]


def configure_optimizers(train_loader, lr_factor=1.0):
    scale_lr = config.cfg['training']['lr'] * np.sqrt(hvd.size())
    lr_schedule = schedules.CosineDecay(
        initial_learning_rate=config.cfg['training']['lr'] / lr_factor,
        warmup_target=scale_lr / lr_factor,
        warmup_steps=3 * train_loader.nevts // config.cfg['training']['batch_size'] // hvd.size(),
        decay_steps=config.cfg['training']['epoch'] * train_loader.nevts // config.cfg['training'][
            'batch_size'] // hvd.size(),
    )
    optimizer = Lion(
        learning_rate=lr_schedule,
        clipnorm=1.0,
        beta_1=0.95,
        beta_2=0.99
    )
    return hvd.DistributedOptimizer(optimizer)


def main():
    utils.setup_gpus()
    flags = parse_arguments()

    load_config(flags.config)

    if hvd.rank() == 0:
        wandb.login()

        run = wandb.init(
            project="bbtautau neutrino predcition",  # Specify your project
            config={  # Track hyperparameters and metadata
                **config.cfg['training'],
                **config.cfg['model'],
            },
        )

    train_loader, val_loader = get_data_loader()
    val_dataset = val_loader.make_tfdata(delete=True)

    model_config = config.cfg['model']
    ckpt_save_path = Path(model_config.pop('ckpt_save_path'))
    model = PET_jetnet(
        num_feat=train_loader.num_feat,
        num_jet=train_loader.num_jet,
        num_classes=train_loader.num_classes,
        num_part=train_loader.num_part,
        model_name=None,

        **model_config
    )

    optimizer_body = configure_optimizers(train_loader, lr_factor=1)
    optimizer_head = configure_optimizers(train_loader, lr_factor=1)
    model.compile(optimizer_body, optimizer_head)
    callbacks = [
        EarlyStopping(patience=45, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', patience=15, min_lr=1e-8, min_delta=1e-4),
    ]

    if hvd.rank() == 0:
        callbacks.append(WandbMetricsLogger())

    val_callback = DiffusionValidationCallback(
        model=model,
        val_dataset=val_dataset,
        val_dataloader=val_loader,
        eval_every=5,
        extra_list_name=[]
    )
    callbacks.append(val_callback)

    checkpoint_name = utils.get_model_name(config.cfg["dataset"], config.cfg["model"])
    checkpoint_path = ckpt_save_path / 'checkpoints' / checkpoint_name

    if hvd.rank() == 0:
        print("Checkpoint name: ", checkpoint_name)

        checkpoint_callback = ModelCheckpoint(
            checkpoint_path,
            save_best_only=True,
            mode='auto',
            save_weights_only=True,
            period=1
        )
        callbacks.append(checkpoint_callback)

    model.fit(
        train_loader.make_tfdata(),
        epochs=config.cfg['training']['epoch'],
        validation_data=val_dataset,
        batch_size=config.cfg['training']['batch_size'],
        callbacks=callbacks,
        steps_per_epoch=train_loader.steps_per_epoch,
        validation_steps=val_loader.steps_per_epoch,
        verbose=hvd.rank() == 0,
    )


if __name__ == "__main__":
    main()
