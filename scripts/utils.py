import logging

import numpy as np
import h5py as h5
from sklearn.utils import shuffle
import sys
import os
import tensorflow as tf
import gc
import random
import itertools
import pickle, copy
from scipy.stats import norm
import glob

#### Horovod imports
try:
    import horovod.tensorflow.keras as hvd
except ImportError or ModuleNotFoundError:
    from dummy_hvd import hvd as hvd


def setup_gpus():
    hvd.init()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank() % len(gpus)], 'GPU')


def get_model_name(dataset: str, model_config: dict, add_string=""):
    model_name = 'PET_{}_{}_{}_{}_{}_{}_{}{}.weights.h5'.format(
        dataset,
        model_config["num_layers"],
        'local' if model_config["local"] else 'nolocal',
        'layer_scale' if model_config["layer_scale"] else 'nolayer_scale',
        'simple' if model_config["simple"] else 'token',
        'fine_tune' if model_config["fine_tune"] else 'baseline',
        model_config["mode"],
        add_string,
    )
    return model_name


class DataLoader:
    """Base class for all data loaders with common preprocessing methods."""

    def __init__(self, path, batch_size=512, rank=0, size=1, **kwargs):
        self.path = path
        self.batch_size = batch_size
        self.rank = rank
        self.size = size

        self.mean_part = []
        self.std_part = []

        self.mean_jet = []
        self.std_jet = []

        self.part_names = []
        self.jet_names = []

    def pad(self, x, num_pad):
        return np.pad(
            x,
            pad_width=((0, 0), (0, 0), (0, num_pad)),
            mode='constant', constant_values=0
        )

    def make_eval_data(self, preprocess=False):
        pass

    def make_tfdata(self):
        pass

    def load_data(self, path, batch_size=512, rank=0, size=1, nevts=None):
        pass

    def preprocess(self, x, mask):
        num_feat = x.shape[-1]
        return mask[:, :, None] * (x[:, :, :num_feat] - self.mean_part[:num_feat]) / self.std_part[:num_feat]

    def preprocess_neutrino(self, x):
        return (x - self.mean_jet) / self.std_jet

    def revert_preprocess(self, x, mask):
        num_feat = x.shape[-1]
        new_part = mask[:, :, None] * (x[:, :, :num_feat] * self.std_part[:num_feat] + self.mean_part[:num_feat])
        return new_part

    def revert_preprocess_neutrino(self, x):
        new_x = self.std_jet * x + self.mean_jet
        # Convert multiplicity back into integers
        return new_x


def process_file_name(file_name):
    f = file_name.decode('utf-8')
    f = f.split('_')[:-1]
    return '_'.join(f)


class TauReconDataLoader(DataLoader):
    def __init__(
            self, path,
            sample_norm: dict,
            batch_size=512, rank=0, size=1, nevts=None,
            sample_weight_map: dict[str, float] = None,
    ):
        super().__init__(path, batch_size, rank, size)
        in_file = h5.File(self.path, 'r')
        self.logger = logging.getLogger('data_loader')

        self.X = in_file['X'][rank:nevts:size]
        self.global_cond = in_file['Condition'][rank:nevts:size]
        self.neutrino = in_file['nu'][rank:nevts:size]

        self.extra = in_file['Extra'][rank:nevts:size]
        if 'RawFile' in in_file:
            unique_file = np.unique(in_file['RawFile'][rank:nevts:size])
            self.unique_file_map = {
                process_file_name(file): idx for idx, file in enumerate(unique_file)
            }
            # convert raw file string to unique index
            self.raw_file = np.array(
                [self.unique_file_map[process_file_name(file)] for file in in_file['RawFile'][rank:nevts:size]])
        else:
            self.raw_file = None

        if 'Weight' in in_file:
            raw_weight = in_file['Weight'][rank:nevts:size]

            event_weight = raw_weight[:, 0]
            sample_weight = raw_weight[:, 1]

            if sample_weight_map and self.raw_file is not None:
                # Build inverse map: index -> filename
                index_to_filename = {v: k for k, v in self.unique_file_map.items()}
                # Assign sample weights using the map
                sample_weight = np.array([
                    sample_weight_map.get(index_to_filename[idx], 1.0)  # default to 1.0 if missing
                    for idx in self.raw_file
                ])

                if hvd.rank() == 0:
                    # Debug print: show the mapping from process index to file name
                    self.logger.info("Raw file index to filename mapping:")
                    for idx, fname in index_to_filename.items():
                        self.logger.info(f"  [{idx}] -> {fname}")

                    # Debug print: sample_weight_map entries
                    self.logger.info("Sample weight map:")
                    for fname, weight in sample_weight_map.items():
                        self.logger.info(f"  {fname}: {weight}")

                    # Debug print: actual weight assigned for each process index
                    self.logger.info("Assigned sample weights per process index:")
                    for idx in sorted(index_to_filename):
                        mask = self.raw_file == idx
                        assigned_weights = sample_weight[mask]
                        if len(assigned_weights) > 0:
                            avg_weight = np.mean(assigned_weights)
                            self.logger.info(
                                f"  [{idx}] {index_to_filename[idx]} -> mean weight: {avg_weight:.3f}, N = {len(assigned_weights)}")

            # self.weight = event_weight * sample_weight
            # self.weight = event_weight
            self.weight = sample_weight
        else:
            self.weight = None

        self.mask = self.X[:, :, 2] != 0
        self.nevts = in_file['X'].shape[0] if nevts is None else nevts
        self.num_part = self.X.shape[1]
        self.num_jet = self.neutrino.shape[1]
        self.num_classes = self.global_cond.shape[1]
        self.num_adv_classes = len(unique_file) if unique_file is not None else None
        self.mean_part = sample_norm['particle']['mean']
        self.std_part = sample_norm['particle']['std']
        self.mean_jet = sample_norm['nu']['mean']
        self.std_jet = sample_norm['nu']['std']

        self.num_pad = 0
        self.num_feat = self.X.shape[2] + self.num_pad  # missing inputs

        self.steps_per_epoch = None  # will pass none, otherwise needs to add repeat to tf data
        self.files = [path]

    def make_eval_data(self, preprocess=False):
        if preprocess:
            X = self.preprocess(self.X, self.mask).astype(np.float32)
            X = self.pad(X, num_pad=self.num_pad)
            neutrino = self.preprocess_neutrino(self.neutrino).astype(np.float32)
        else:
            X = self.X
            neutrino = self.neutrino

        return X, X[:, :, 1:3], self.mask.astype(np.float32), self.global_cond, neutrino

    def make_tfdata(self, delete: bool = True):
        X = self.preprocess(self.X, self.mask).astype(np.float32)
        X = self.pad(X, num_pad=self.num_pad)
        neutrino = self.preprocess_neutrino(self.neutrino).astype(np.float32)

        tf_zip = tf.data.Dataset.from_tensor_slices({
            'input_features': X,
            'input_points': X[:, :, 1:3],
            'input_mask': self.mask.astype(np.float32),
            'input_jet': neutrino,
            'input_weight': self.weight.astype(np.float32),
            # 'input_weight': None,
            'input_file': self.raw_file.astype(np.int32),
        })

        if hvd.rank() == 0:
            # print shape for input features, input points, input mask, input jet, input weight, input file
            self.logger.info(f"input_features shape: {X.shape}")
            self.logger.info(f"input_points shape: {X[:, :, 1:3].shape}")
            self.logger.info(f"input_mask shape: {self.mask.shape}")
            self.logger.info(f"target shape: {neutrino.shape}")
            self.logger.info(f"input_weight shape: {self.weight.shape}")
            self.logger.info(f"input_file shape: {self.raw_file.shape}")

        tf_global_cond = tf.data.Dataset.from_tensor_slices(self.global_cond)
        if delete:
            del self.X, self.global_cond, self.mask
        gc.collect()

        return tf.data.Dataset.zip((tf_zip, tf_global_cond)).cache().shuffle(self.batch_size * 100).batch(
            self.batch_size).prefetch(tf.data.AUTOTUNE)
