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


def load_pickle(folder, f):
    file_name = os.path.join(folder, 'histories', f.replace(".weights.h5", ".pkl"))
    with open(file_name, 'rb') as file_pi:
        history_dict = pickle.load(file_pi)
    return history_dict


def revert_npart(npart, name='30'):
    # Reverse the preprocessing to recover the particle multiplicity
    stats = {'30': (29.03636, 2.7629626),
             '49': (21.66242333, 8.86935969),
             '150': (49.398304, 20.772636),
             '279': (57.28675, 29.41252836)}
    mean, std = stats[name]
    return np.round(npart * std + mean).astype(np.int32)


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
        return np.pad(x, pad_width=((0, 0), (0, 0), (0, num_pad)),
                      mode='constant', constant_values=0)

    def make_eval_data(self, preprocess=False):
        if preprocess:
            pion = self.X
            X = self.preprocess(self.X, self.mask).astype(np.float32)
            X = self.pad(X, num_pad=self.num_pad)
            jet = self.preprocess_jet(self.jet).astype(np.float32)
        else:
            X = self.X
            pion = None
            jet = self.jet

        return X, X[:, :, 1:3], self.mask.astype(np.float32), jet, self.y

    def make_tfdata(self):
        X = self.preprocess(self.X, self.mask).astype(np.float32)
        X = self.pad(X, num_pad=self.num_pad)
        jet = self.preprocess_jet(self.jet).astype(np.float32)

        tf_zip = tf.data.Dataset.from_tensor_slices({
            'input_features': X,
            'input_points': X[:, :, 1:3],
            'input_mask': self.mask.astype(np.float32),
            'input_jet': jet
        })

        tf_y = tf.data.Dataset.from_tensor_slices(self.y)
        del self.X, self.y, self.mask
        gc.collect()

        return tf.data.Dataset.zip((tf_zip, tf_y)).cache().shuffle(self.batch_size * 100).batch(
            self.batch_size).prefetch(tf.data.AUTOTUNE)

    def load_data(self, path, batch_size=512, rank=0, size=1, nevts=None):
        pass

    def preprocess(self, x, mask):
        num_feat = x.shape[-1]
        return mask[:, :, None] * (x[:, :, :num_feat] - self.mean_part[:num_feat]) / self.std_part[:num_feat]

    def preprocess_jet(self, x):
        return (x - self.mean_jet) / self.std_jet

    def revert_preprocess(self, x, mask):
        num_feat = x.shape[-1]
        new_part = mask[:, :, None] * (x[:, :, :num_feat] * self.std_part[:num_feat] + self.mean_part[:num_feat])
        return new_part

    def revert_preprocess_jet(self, x):

        new_x = self.std_jet * x + self.mean_jet
        # Convert multiplicity back into integers
        return new_x


class RecoTauDataLoaderWithPKLForSample(DataLoader):
    def __init__(self, path, batch_size=1024, rank=0, size=1, nevts=None, samples_name='none'):
        super().__init__(path, batch_size, rank, size)
        self.samples_name = samples_name
        self.load_data(path, batch_size, rank, size, nevts)

        if samples_name == 'pi_pi':
            self.mean_part = [2.59163526e+01, 0.0, 0.0, 7.05392784e+01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            self.std_part = [17.0718089, 1.0, 1.0, 115.65273143, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            self.mean_jet = [
                0.01682312963810866, 0.016733380409220386, 0.08293679661612122, 0.021420429070607164,
                -0.0006042911085853564, -0.013209993791164207
            ]
            self.std_jet = [13.975173949266443, 13.981232817525068, 38.85618917630569, 14.059869543855184,
                            14.097959985975313, 40.51226330089419]
        elif samples_name == 'e_pi':
            self.mean_part = [2.470e+1, 0.0, 0.0, 6.949e+1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            self.std_part = [1.750e+1, 1.0, 1.0, 1.192e+2, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            self.mean_jet = [2.198e-2, 1.831e-2, -4.410e-3, -1.366e-2, 3.675e-2, 1.322e-1]
            self.std_jet = [1.569e+1, 1.582e+1, 4.252e+1, 1.825e+1, 1.823e+1, 5.234e+1]
        elif samples_name == 'e_rho':
            self.mean_part = [14.982864907864142, 0.0, 0.0, 42.79647942777018, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            self.std_part = [15.792115401062988, 1.0, 1.0, 97.37732861435599, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            self.mean_jet = [-0.0002678336552352207, 0.016736370466439304, 0.08861963610428333, 0.06419391577262668,
                             -0.0010769517545723172, 0.15892125628069512]
            self.std_jet = [14.050414410768337, 14.028565766125833, 39.01819593975496, 17.369162229933927,
                            17.322592331941568, 52.47183219751106]
        elif samples_name == 'mu_pi':
            self.mean_part = [2.449e+1, 0.0, 0.0, 7.038e+1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            self.std_part = [1.742e+1, 1.0, 1.0, 1.194e+2, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            self.mean_jet = [-3.981e-2, 2.796e-3, 1.387e-1, 3.514e-2, 3.003e-2, 9.278e-3]
            self.std_jet = [1.575e+1, 1.578e+1, 4.335e+1, 1.807e+1, 1.808e+1, 5.269e+1]
        elif samples_name == 'mu_rho':
            self.mean_part = [14.912302932315121, 0.0, 0.0, 42.950420353288834, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            self.std_part = [15.519213711111917, 1.0, 1.0, 96.64609240683967, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            self.mean_jet = [0.004521378634852432, 0.008900067295884653, -0.055627586346384726, -0.030963372592020283,
                             -0.017430794771537462, -0.05792407995355238]
            self.std_jet = [13.945134622193716, 13.947684551258607, 38.95641101985201, 17.409530564802242,
                            17.35883209984128, 53.38571112218628]
        elif samples_name == 'pi_rho':
            self.mean_part = [15.174936586596685, 0.0, 0.0, 41.85549317045239, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            self.std_part = [15.696616138696147, 1.0, 1.0, 92.55279442301007, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            self.mean_jet = [-0.02391130059056275, 0.00963103174019532, 0.0887821064592522, -4.879895705211167e-05,
                             0.002470162695203731, -0.04313502777777081]
            self.std_jet = [13.169428059607874, 13.224751262630551, 37.22644797694016, 14.779683471066075,
                            14.801913665250858, 44.11978535468495]
        elif samples_name == 'rho_rho':
            self.mean_part = [10.936752258785328, 0.0, 0.0, 30.6316162852822, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            self.std_part = [12.67041073494841, 1.0, 1.0, 80.66492595932489, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            self.mean_jet = [-0.005334533901221733, 0.012579453929881869, 0.08474348283190947, 0.0009670746789205529,
                             -0.011116983664680337, 0.001942205415118817]
            self.std_jet = [13.455550914459518, 13.466844645259748, 37.733618935444994, 13.549154909778169,
                            13.538312786991147, 39.80704566398323]
        else:
            raise ValueError('samples_name is not valid')
        self.num_pad = 0
        self.num_feat = self.X.shape[2] + self.num_pad  # missing inputs

        # self.y = np.identity(2)[self.y.astype(np.int32)]
        self.num_classes = self.y.shape[1]
        self.steps_per_epoch = None  # will pass none, otherwise needs to add repeat to tf data
        self.files = [path]

    def load_data(self, path, batch_size=512, rank=0, size=1, nevts=None):
        import vector
        import pickle
        self.path = path
        # Load all the data from the npz files
        with open(self.path, 'rb') as f:
            data = pickle.load(f)

        val_num = -1 if nevts is None else nevts
        pjet_1 = data['jet_1'][rank:nevts:size]
        pjet_2 = data['jet_2'][rank:nevts:size]
        pjet_3 = data['jet_3'][rank:nevts:size]
        pMET = data['MET'][rank:nevts:size]
        EventID = data['EventID'][rank:nevts:size]
        samples = data['sample'][rank:nevts:size]
        ptau_p_child1 = data['tau_p_child1'][rank:nevts:size]
        ptau_p_child2 = data['tau_p_child2'][rank:nevts:size]
        ptau_m_child1 = data['tau_m_child1'][rank:nevts:size]
        ptau_m_child2 = data['tau_m_child2'][rank:nevts:size]
        tau_p_child1_charge = data['tau_p_child1_charge'][rank:nevts:size]
        tau_p_child1_is_el = data['tau_p_child1_is_el'][rank:nevts:size]
        tau_p_child1_is_mu = data['tau_p_child1_is_mu'][rank:nevts:size]
        tau_p_child1_is_charged_pion = data['tau_p_child1_is_charged_pion'][rank:nevts:size]
        tau_p_child1_is_neutral_part = data['tau_p_child1_is_neutral_part'][rank:nevts:size]
        tau_p_child2_charge = data['tau_p_child2_charge'][rank:nevts:size]
        tau_p_child2_is_el = data['tau_p_child2_is_el'][rank:nevts:size]
        tau_p_child2_is_mu = data['tau_p_child2_is_mu'][rank:nevts:size]
        tau_p_child2_is_charged_pion = data['tau_p_child2_is_charged_pion'][rank:nevts:size]
        tau_p_child2_is_neutral_part = data['tau_p_child2_is_neutral_part'][rank:nevts:size]
        tau_m_child1_charge = data['tau_m_child1_charge'][rank:nevts:size]
        tau_m_child1_is_el = data['tau_m_child1_is_el'][rank:nevts:size]
        tau_m_child1_is_mu = data['tau_m_child1_is_mu'][rank:nevts:size]
        tau_m_child1_is_charged_pion = data['tau_m_child1_is_charged_pion'][rank:nevts:size]
        tau_m_child1_is_neutral_part = data['tau_m_child1_is_neutral_part'][rank:nevts:size]
        tau_m_child2_charge = data['tau_m_child2_charge'][rank:nevts:size]
        tau_m_child2_is_el = data['tau_m_child2_is_el'][rank:nevts:size]
        tau_m_child2_is_mu = data['tau_m_child2_is_mu'][rank:nevts:size]
        tau_m_child2_is_charged_pion = data['tau_m_child2_is_charged_pion'][rank:nevts:size]
        tau_m_child2_is_neutral_part = data['tau_m_child2_is_neutral_part'][rank:nevts:size]
        jet_1 = np.stack(
            [pjet_1.pt, pjet_1.eta, pjet_1.phi, pjet_1.E, np.zeros_like(pjet_1.pt), np.zeros_like(pjet_1.pt),
             np.zeros_like(pjet_1.pt), np.zeros_like(pjet_1.pt), np.zeros_like(pjet_1.pt)], -1)
        jet_2 = np.stack(
            [pjet_2.pt, pjet_2.eta, pjet_2.phi, pjet_2.E, np.zeros_like(pjet_2.pt), np.zeros_like(pjet_2.pt),
             np.zeros_like(pjet_2.pt), np.zeros_like(pjet_2.pt), np.zeros_like(pjet_2.pt)], -1)
        jet_3 = np.stack(
            [pjet_3.pt, pjet_3.eta, pjet_3.phi, pjet_3.E, np.zeros_like(pjet_3.pt), np.zeros_like(pjet_3.pt),
             np.zeros_like(pjet_3.pt), np.zeros_like(pjet_3.pt), np.zeros_like(pjet_3.pt)], -1)
        MET = np.stack([pMET.pt, pMET.phi], -1)
        tau_p_child1 = np.stack(
            [ptau_p_child1.pt, ptau_p_child1.eta, ptau_p_child1.phi, ptau_p_child1.E, tau_p_child1_charge,
             tau_p_child1_is_el, tau_p_child1_is_mu, tau_p_child1_is_charged_pion, tau_p_child1_is_neutral_part], -1)
        tau_p_child2 = np.stack(
            [ptau_p_child2.pt, ptau_p_child2.eta, ptau_p_child2.phi, ptau_p_child2.E, tau_p_child2_charge,
             tau_p_child2_is_el, tau_p_child2_is_mu, tau_p_child2_is_charged_pion, tau_p_child2_is_neutral_part], -1)
        tau_m_child1 = np.stack(
            [ptau_m_child1.pt, ptau_m_child1.eta, ptau_m_child1.phi, ptau_m_child1.E, tau_m_child1_charge,
             tau_m_child1_is_el, tau_m_child1_is_mu, tau_m_child1_is_charged_pion, tau_m_child1_is_neutral_part], -1)
        tau_m_child2 = np.stack(
            [ptau_m_child2.pt, ptau_m_child2.eta, ptau_m_child2.phi, ptau_m_child2.E, tau_m_child2_charge,
             tau_m_child2_is_el, tau_m_child2_is_mu, tau_m_child2_is_charged_pion, tau_m_child2_is_neutral_part], -1)
        del pjet_1, pjet_2, pjet_3, pMET, ptau_p_child1, ptau_p_child2, ptau_m_child1, ptau_m_child2

        self.EventID = EventID
        self.event_type = samples

        self.X = np.concatenate([tau_p_child1.reshape(tau_p_child1.shape[0], 1, tau_p_child1.shape[-1]),
                                 tau_p_child2.reshape(tau_p_child2.shape[0], 1, tau_p_child2.shape[-1]),
                                 tau_m_child1.reshape(tau_m_child1.shape[0], 1, tau_m_child1.shape[-1]),
                                 tau_m_child2.reshape(tau_m_child2.shape[0], 1, tau_m_child2.shape[-1])], axis=1)
        self.X = np.concatenate([self.X, jet_1.reshape(jet_1.shape[0], 1, jet_1.shape[-1]),
                                 jet_2.reshape(jet_2.shape[0], 1, jet_2.shape[-1]),
                                 jet_3.reshape(jet_3.shape[0], 1, jet_3.shape[-1])], axis=1)

        # add a one label to identify particles
        self.labels = np.ones((self.X.shape[0], self.X.shape[1], 1))
        if "rho" in self.samples_name:
            self.labels[:, 6:] = 0
        else:
            self.labels[:, 4:] = 2
        # for padding particles, the label is 0
        self.labels[self.X[:, :, 0] == 0] = 0
        self.X = np.concatenate([self.X, self.labels], -1)
        # For truth level study, self.jet are the truth 
        self.jet = np.zeros((self.X.shape[0], 6))
        # For truth level study, self.y are the MET
        self.y = MET  # met pT, met_phi
        # let's normalize the met pT
        self.y[:, 0] = np.log(self.y[:, 0])
        self.mask = self.X[:, :, 2] != 0

        # self.batch_size = batch_size
        self.nevts = self.X.shape[0]
        self.num_part = self.X.shape[1]
        self.num_jet = self.jet.shape[1]


class TauReconDataLoader(DataLoader):
    def __init__(
            self, path,
            sample_norm: dict,
            tau_visible_number: int = 2,
            batch_size=512, rank=0, size=1, nevts=None,
    ):
        super().__init__(path, batch_size, rank, size)
        in_file = h5.File(self.path, 'r')

        self.X = in_file['X'][rank:nevts:size]
        self.y = in_file['MET'][rank:nevts:size]
        self.jet = in_file['nu'][rank:nevts:size]

        self.labels = np.ones((self.X.shape[0], self.X.shape[1], 1))
        self.labels[:, tau_visible_number:] = 0

        # for padding particles, the label is 0
        self.labels[self.X[:, :, 0] == 0] = 0
        self.X = np.concatenate([self.X, self.labels], -1)
        self.y[:, 0] = np.log(self.y[:, 0])
        self.mask = self.X[:, :, 2] != 0
        self.nevts = in_file['X'].shape[0] if nevts is None else nevts
        self.num_part = self.X.shape[1]
        self.num_jet = self.jet.shape[1]
        self.num_classes = self.y.shape[1]

        self.mean_part = sample_norm['particle']['mean']
        self.std_part = sample_norm['particle']['std']
        self.mean_jet = sample_norm['nu']['mean']
        self.std_jet = sample_norm['nu']['std']

        self.num_pad = 0
        self.num_feat = self.X.shape[2] + self.num_pad  # missing inputs

        self.steps_per_epoch = None  # will pass none, otherwise needs to add repeat to tf data
        self.files = [path]

        self.tau_vis_num = tau_visible_number

    def load_data(self, path, batch_size=512, rank=0, size=1, nevts=None):
        pass
