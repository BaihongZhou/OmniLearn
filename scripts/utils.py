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


def get_model_name(flags,fine_tune=False,add_string=""):
    model_name = 'PET_{}_{}_{}_{}_{}_{}_{}{}.weights.h5'.format(
        flags.dataset,
        flags.num_layers,
        'local' if flags.local else 'nolocal',
        'layer_scale' if flags.layer_scale else 'nolayer_scale',
        'simple' if flags.simple else 'token',
        'fine_tune' if fine_tune else 'baseline',        
        flags.mode,
        add_string,
    )
    return model_name

def load_pickle(folder,f):
    file_name = os.path.join(folder,'histories',f.replace(".weights.h5",".pkl"))
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

        self.mean_part = [0.0, 0.0, -0.0278,
                          1.8999407,-0.027,2.244736, 0.0,
                          0.0, 0.0,  0.0,  0.0,  0.0, 0.0]
        self.std_part = [0.215, 0.215,  0.070, 
                         1.2212526, 0.069,1.2334691,1.0,
                         1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

        self.mean_jet =  [ 6.18224920e+02, 0.0, 1.2064709e+02,3.94133173e+01]
        self.std_jet  = [106.71761,0.88998157,40.196922,15.096386]

        self.part_names = ['$\eta_{rel}$', '$\phi_{rel}$', 'log($1 - p_{Trel}$)','log($p_{T}$)','log($1 - E_{rel}$)','log($E$)','$\Delta$R']
        self.jet_names = ['Jet p$_{T}$ [GeV]', 'Jet $\eta$', 'Jet Mass [GeV]','Multiplicity']
        
    
    def pad(self,x,num_pad):
        return np.pad(x, pad_width=((0, 0), (0, 0), (0, num_pad)),
                      mode='constant', constant_values=0)

    def data_from_file(self,file_path, preprocess=False):
        with h5.File(file_path, 'r') as file:
            data_chunk = file['data'][:]
            mask_chunk = data_chunk[:, :, 2] != 0
            
            jet_chunk = file['jet'][:]
            label_chunk = file['pid'][:]

            if preprocess:
                data_chunk = self.preprocess(data_chunk, mask_chunk)
                data_chunk = self.pad(data_chunk,num_pad=self.num_pad)
                jet_chunk = self.preprocess_jet(jet_chunk)
                
            points_chunk = data_chunk[:, :, 1:3]
            
        return [data_chunk,points_chunk,mask_chunk,jet_chunk],label_chunk

    def make_eval_data(self,preprocess=False):
        if preprocess:
            pion = self.X
            X = self.preprocess(self.X,self.mask).astype(np.float32)
            X = self.pad(X,num_pad=self.num_pad)
            jet = self.preprocess_jet(self.jet).astype(np.float32)
        else:
            X = self.X
            pion = None
            jet = self.jet

        if self.EventID is not None:
            return X,X[:,:,1:3],self.mask.astype(np.float32),jet,self.y, self.EventID, self.event_type,pion
        else:
            return X,X[:,:,1:3],self.mask.astype(np.float32),jet,self.y

    def make_tfdata(self):
        X = self.preprocess(self.X,self.mask).astype(np.float32)
        X = self.pad(X,num_pad=self.num_pad)
        jet = self.preprocess_jet(self.jet).astype(np.float32)

        tf_zip = tf.data.Dataset.from_tensor_slices(
            {'input_features':X,
             'input_points':X[:,:,1:3],
             'input_mask':self.mask.astype(np.float32),
             'input_jet':jet})
        

        tf_y = tf.data.Dataset.from_tensor_slices(self.y)
        del self.X, self.y,  self.mask
        gc.collect()
        
        return tf.data.Dataset.zip((tf_zip,tf_y)).cache().shuffle(self.batch_size*100).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)


    def load_data(self,path, batch_size=512,rank=0,size=1,nevts=None):
        # self.path = path

        self.X = h5.File(self.path,'r')['data'][rank:nevts:size]
        self.y = h5.File(self.path,'r')['pid'][rank:nevts:size]
        self.jet = h5.File(self.path,'r')['jet'][rank:nevts:size]
        self.mask = self.X[:,:,2]!=0

        # self.batch_size = batch_size
        self.nevts = h5.File(self.path,'r')['data'].shape[0] if nevts is None else nevts
        self.num_part = self.X.shape[1]
        self.num_jet = self.jet.shape[1]


    def preprocess(self,x,mask):                
        num_feat = x.shape[-1]
        return mask[:,:, None]*(x[:,:,:num_feat]-self.mean_part[:num_feat])/self.std_part[:num_feat]

    def preprocess_jet(self,x):        
        return (x-self.mean_jet)/self.std_jet
    
    def revert_preprocess(self,x,mask):                
        num_feat = x.shape[-1]        
        new_part = mask[:,:, None]*(x[:,:,:num_feat]*self.std_part[:num_feat] + self.mean_part[:num_feat])
        return  new_part

    def revert_preprocess_jet(self,x):

        new_x = self.std_jet*x+self.mean_jet
        #Convert multiplicity back into integers
        return new_x

        
class TruthTauDataLoader(DataLoader):
    def __init__(self, path, batch_size=512,rank=0,size=1, nevts=None, samples_name='none'):
        super().__init__(path, batch_size, rank, size)
        self.load_data(path, batch_size,rank,size,nevts)

        if samples_name == 'pipi':
            self.mean_part = [2.59163526e+01, 0.0, 0.0, 7.05392784e+01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            self.std_part = [17.0718089, 1.0, 1.0, 115.65273143, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            self.mean_jet = [0.01682312963810866, 0.016733380409220386, 0.08293679661612122, 0.021420429070607164, -0.0006042911085853564, -0.013209993791164207]
            self.std_jet  = [13.975173949266443, 13.981232817525068, 38.85618917630569, 14.059869543855184, 14.097959985975313, 40.51226330089419]
        
        self.num_pad = 0
        self.num_feat = self.X.shape[2] + self.num_pad #missing inputs
        self.samples_name = samples_name
        
        # self.y = np.identity(2)[self.y.astype(np.int32)]
        self.num_classes = self.y.shape[1]
        self.steps_per_epoch = None #will pass none, otherwise needs to add repeat to tf data
        self.files = [path]
    
    def load_data(self,path, batch_size=512,rank=0,size=1,nevts=None):
        self.path = path
        self.X = h5.File(self.path,'r')['X'][rank:nevts:size]
        self.y = h5.File(self.path,'r')['MET'][rank:nevts:size]
        self.jet = h5.File(self.path,'r')['nu'][rank:nevts:size]
        #let's normalize the met pT
        #add two different labels to identify particles
        self.labels = np.ones((self.X.shape[0],self.X.shape[1],1))
        if self.X.shape[1] > 4:
            self.labels[:,4:] = 0
        # for padding particles, the label is 0
        self.labels[self.X[:,:,0]==0] = 0
        self.X = np.concatenate([self.X,self.labels],-1)
        self.y[:,0] = np.log(self.y[:,0])
        self.mask = self.X[:,:,2]!=0
        # self.batch_size = batch_size
        self.nevts = h5.File(self.path,'r')['X'].shape[0] if nevts is None else nevts
        self.num_part = self.X.shape[1]
        self.num_jet = self.jet.shape[1]
    