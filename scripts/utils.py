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


class TauReconDataLoader(DataLoader):
    def __init__(
            self, path,
            sample_norm: dict,
            batch_size=512, rank=0, size=1, nevts=None,
    ):
        super().__init__(path, batch_size, rank, size)
        in_file = h5.File(self.path, 'r')

        self.X = in_file['X'][rank:nevts:size]
        self.global_cond = in_file['MET'][rank:nevts:size]
        self.neutrino = in_file['nu'][rank:nevts:size]

        self.extra = in_file['Extra'][rank:nevts:size]
        self.raw_file = in_file['RawFile'][rank:nevts:size]

        # self.labels = np.ones((self.X.shape[0], self.X.shape[1], 1))
        #
        # # for padding particles, the label is 0
        # self.labels[self.X[:, :, 0] == 0] = 0
        # self.X = np.concatenate([self.X, self.labels], -1)
        # self.y[:, 0] = np.log(self.y[:, 0])
        self.mask = self.X[:, :, 2] != 0
        self.nevts = in_file['X'].shape[0] if nevts is None else nevts
        self.num_part = self.X.shape[1]
        self.num_jet = self.neutrino.shape[1]
        self.num_classes = self.global_cond.shape[1]

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

    def make_tfdata(self):
        X = self.preprocess(self.X, self.mask).astype(np.float32)
        X = self.pad(X, num_pad=self.num_pad)
        neutrino = self.preprocess_neutrino(self.neutrino).astype(np.float32)

        tf_zip = tf.data.Dataset.from_tensor_slices({
            'input_features': X,
            'input_points': X[:, :, 1:3],
            'input_mask': self.mask.astype(np.float32),
            'input_jet': neutrino
        })

        tf_global_cond = tf.data.Dataset.from_tensor_slices(self.global_cond)
        del self.X, self.global_cond, self.mask
        gc.collect()

        return tf.data.Dataset.zip((tf_zip, tf_global_cond)).cache().shuffle(self.batch_size * 100).batch(
            self.batch_size).prefetch(tf.data.AUTOTUNE)
