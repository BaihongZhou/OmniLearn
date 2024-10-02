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
import horovod.tensorflow.keras as hvd
#from dummy_hvd import hvd as hvd

def setup_gpus():
    hvd.init()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')


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
            X = self.preprocess(self.X,self.mask).astype(np.float32)
            X = self.pad(X,num_pad=self.num_pad)
            jet = self.preprocess_jet(self.jet).astype(np.float32)
        else:
            X = self.X
            jet = self.jet

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




class TopDataLoader(DataLoader):    
    def __init__(self, path, batch_size=512,rank=0,size=1):
        super().__init__(path, batch_size, rank, size)

        self.load_data(path, batch_size,rank,size)
        self.num_pad = 6
        self.num_feat = self.X.shape[2] + self.num_pad #missing inputs
        
        # self.y = np.identity(2)[self.y.astype(np.int32)]
        self.num_classes = self.y.shape[1]
        self.steps_per_epoch = None #will pass none, otherwise needs to add repeat to tf data
        self.files = [path]

class TopDataLoaderWithGenerator(DataLoader):    
    def __init__(self, path, batch_size=512,rank=0,size=1):
        super().__init__(path, batch_size, rank, size)

        self.load_data(path, batch_size,rank,size)

        self.mean_part = [3.70747896e+01, 0.0,  0.0,  1.16020465e+01,  0.0]
        self.std_part =  [71.52596521,  1.0,  1.0, 55.33589808,  1.0]
        self.mean_jet = [-0.16206509,  0.05322177, -2.457497]
        self.std_jet  = [112.16817, 112.59142, 242.80714]
        
        self.num_pad = 0
        self.num_feat = self.X.shape[2] + self.num_pad #missing inputs
        
        # self.y = np.identity(2)[self.y.astype(np.int32)]
        self.num_classes = self.y.shape[1]
        self.steps_per_epoch = None #will pass none, otherwise needs to add repeat to tf data
        self.files = [path]

    def get_pxyz(self,arr):
        pT = arr[:,0]
        eta = arr[:,1]
        phi = arr[:,2]
        E = arr[:,3]
        px = pT*np.cos(phi)
        py = pT*np.sin(phi)
        pz = pT*np.sinh(eta)
        return np.stack([px,py,pz],-1)
    
    def load_data(self,path, batch_size=512,rank=0,size=1,nevts=None):
        self.path = path
        # self.y is what we are trying to predict
        self.X = h5.File(self.path,'r')['jets'][rank:nevts:size][:,:,:4] #jet 4vector
        #add a zero label to identify jets
        self.X = np.concatenate([self.X,np.zeros((self.X.shape[0],self.X.shape[1],1))],-1)
        self.y_t = h5.File(self.path,'r')['regress'][rank:nevts:size]
        self.p_t = h5.File(self.path,'r')['fjets'][rank:nevts:size]
        self.lep = self.p_t[:,4:8] #lepton 4vector
        #add a one for leptons
        self.lep = np.concatenate([self.lep,np.ones((self.X.shape[0],1))],-1)
        self.jet = self.y_t[:,10:] #neutrino 4vector
        self.jet = self.get_pxyz(self.jet)
        self.X = np.concatenate([self.lep[:,None],self.X],axis=1)
        
        self.y = np.concatenate([self.p_t[:,14].reshape(-1,1),self.p_t[:,17].reshape(-1,1)],axis=1) #met pT, met_phi
        #let's normalize the met pT
        self.y[:,0] = np.log(self.y[:,0])
        self.mask = self.X[:,:,2]!=0

        # self.batch_size = batch_size
        self.nevts = h5.File(self.path,'r')['jets'].shape[0] if nevts is None else nevts
        self.num_part = self.X.shape[1]
        self.num_jet = self.jet.shape[1]

        

class ToyDataLoader(DataLoader):    
    def __init__(self, nevts,batch_size=512,rank=0,size=1):
        super().__init__(nevts,batch_size, rank, size)

        self.nevts = nevts
        self.X = np.concatenate([
            np.random.normal(loc = 0.0,scale=1.0,size=(self.nevts,15,13)),
            np.random.normal(loc = 1.0,scale=1.0,size=(self.nevts,15,13))])
        self.jet = np.concatenate([
            np.random.normal(loc = 0.0,scale=1.0,size=(self.nevts,4)),
            np.random.normal(loc = 1.0,scale=1.0,size=(self.nevts,4))])
        self.mask = self.X[:,:,2]!=0
        self.y = np.concatenate([np.ones((self.nevts)),np.zeros((self.nevts))])        
        self.num_part = self.X.shape[1]
        self.num_jet = self.jet.shape[1]

        
        self.num_pad = 0
        self.num_feat = self.X.shape[2] + self.num_pad #missing inputs
        
        #one hot label
        self.y = np.identity(2)[self.y.astype(np.int32)]
        self.num_classes = self.y.shape[1]
        self.steps_per_epoch = None #will pass none, otherwise needs to add repeat to tf data
        self.files = None

        

class TauDataLoader(DataLoader):    
    def __init__(self, path, batch_size=512,rank=0,size=1,nevts=None):
        super().__init__(path, batch_size, rank, size)

        self.mean_part = [ 0.0, 0.0, -4.68198519e-02,  2.20178221e-01,
                                -7.48168704e-02,  2.56480441e-01,  0.0,
                                0.0, 0.0,  0.0,  0.0,  0.0, 0.0]
        self.std_part =  [0.03927566, 0.04606768, 0.25982114,
                               0.82466037, 0.7541279,  0.86455974,1.0,
                               1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        self.mean_jet = [6.16614813e+01, 2.05619964e-03, 3.52885518e+00, 4.28755680e+00]
        self.std_jet  = [34.22578952,  0.68952567,  4.54982729,  3.20547624]

        self.load_data(path, batch_size,rank,size,nevts = nevts)

        self.num_pad = 0
        self.num_feat = self.X.shape[2] + self.num_pad #missing inputs
        
        self.num_classes = self.y.shape[1]
        self.steps_per_epoch = None #will pass none, otherwise needs to add repeat to tf data
        self.files = [path]

        
class AtlasDataLoader(DataLoader):    
    def __init__(self, path, batch_size=512,rank=0,size=1,is_small=False):
        super().__init__(path, batch_size, rank, size)
        self.mean_jet =  [1.73933684e+03, 4.94380870e-04, 2.21667582e+02, 5.52376512e+01]
        self.std_jet  = [9.75164004e+02, 8.31232765e-01, 2.03672420e+02, 2.51242747e+01]
        
        self.path = path
        if is_small:
            self.nevts = int(4e6)
        else:
            self.nevts = h5.File(self.path,'r')['data'].shape[0]
            
        self.X = h5.File(self.path,'r')['data'][rank:self.nevts:size]
        self.y = h5.File(self.path,'r')['pid'][rank:self.nevts:size]
        self.w = h5.File(self.path,'r')['weights'][rank:self.nevts:size]
        self.jet = h5.File(self.path,'r')['jet'][rank:self.nevts:size]
        self.mask = self.X[:,:,2]!=0

        self.batch_size = batch_size
        
        self.num_part = self.X.shape[1]
        self.num_pad = 6

        self.num_feat = self.X.shape[2] + self.num_pad #missing inputs
        self.num_jet = self.jet.shape[1]
        self.num_classes = 1
        self.steps_per_epoch = None #will pass none, otherwise needs to add repeat to tf data
        self.files = [path]

    def make_tfdata(self):
        X = self.preprocess(self.X,self.mask).astype(np.float32)
        X = self.pad(X,num_pad=self.num_pad)
        jet = self.preprocess_jet(self.jet).astype(np.float32)
        tf_zip = tf.data.Dataset.from_tensor_slices(
            {'input_features':X,
             'input_points':X[:,:,1:3],
             'input_mask':self.mask.astype(np.float32),
             'input_jet':jet})

        tf_y = tf.data.Dataset.from_tensor_slices(np.stack([self.y,self.w],-1))

        del self.X, self.y,  self.mask
        gc.collect()
        
        return tf.data.Dataset.zip((tf_zip,tf_y)).cache().shuffle(self.batch_size*100).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)



class H1DataLoader(DataLoader):    
    def __init__(self, path, batch_size=512,rank=0,size=1):
        super().__init__(path, batch_size, rank, size)

        self.mean_part = [0.031, 0.0, -0.10,
                          -0.23,-0.10,0.27, 0.0,
                          0.0, 0.0,  0.0,  0.0,  0.0, 0.0]
        self.std_part = [0.35, 0.35,  0.178, 
                         1.2212526, 0.169,1.17,1.0,
                         1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

        self.mean_jet =  [ 19.15986358 , 0.57154217 , 6.00354102, 11.730992]
        self.std_jet  = [9.18613789, 0.80465287 ,2.99805704 ,5.14910232]
        
        self.load_data(path, batch_size,rank,size)
                
        self.y = np.identity(2)[self.y.astype(np.int32)]        
        self.num_pad = 5
        self.num_feat = self.X.shape[2] + self.num_pad #missing inputs        
        self.num_classes = self.y.shape[1]
        self.steps_per_epoch = None #will pass none, otherwise needs to add repeat to tf data
        self.files = [path]
        

class OmniDataLoader(DataLoader):
    def __init__(self, path, batch_size=512,rank=0,size=1):
        super().__init__(path, batch_size, rank, size)

        self.mean_jet =  [2.25826286e+02, 1.25739745e-03, 1.83963520e+01 ,1.88828832e+01]
        self.std_jet  = [90.39824296 , 1.34598289 ,10.73467645  ,8.45697634]

        
        self.path = path
        self.X = h5.File(self.path,'r')['reco'][rank::size]
        self.Y = h5.File(self.path,'r')['gen'][rank::size]        

        self.weight = np.ones(self.X.shape[0])
        
        self.nevts = h5.File(self.path,'r')['reco'].shape[0]
        self.num_part = self.X.shape[1]
        self.num_pad = 0

        self.num_feat = self.X.shape[2] + self.num_pad #missing inputs
        self.num_jet = 4
        self.num_classes = 1
        self.steps_per_epoch = None #will pass none, otherwise needs to add repeat to tf data
        self.files = [path]

        self.reco = self.get_inputs(self.X,h5.File(self.path,'r')['reco_jets'][rank::size])
        self.gen = self.get_inputs(self.Y,h5.File(self.path,'r')['gen_jets'][rank::size])
        self.high_level_reco = h5.File(self.path,'r')['reco_subs'][rank::size]
        self.high_level_gen = h5.File(self.path,'r')['gen_subs'][rank::size]

    def get_inputs(self,X,jet):
        mask = X[:,:,2]!=0
        
        time = np.zeros((mask.shape[0],1)) #classifier gets time always 0
        #Preprocess and pad
        X = self.preprocess(X,mask).astype(np.float32)
        X = self.pad(X,num_pad=self.num_pad)
        jet = self.preprocess_jet(jet).astype(np.float32)
        coord = X[:,:,0:2]
        return [X,coord,mask,jet,time]


    def data_from_file(self,file_path):
        with h5.File(file_path, 'r') as file:
            X = h5.File(file_path,'r')['reco'][:]
            reco = self.get_inputs(X,h5.File(file_path,'r')['reco_jets'][:])
            label_chunk = np.ones(X.shape[0])
                        
        return reco,label_chunk


class QGDataLoader(DataLoader):
    def __init__(self, path, batch_size=512,rank=0,size=1):
        super().__init__(path, batch_size, rank, size)

        self.load_data(path, batch_size,rank,size)        
        self.y = np.identity(2)[self.y.astype(np.int32)]
        self.num_pad = 0
        self.num_feat = self.X.shape[2] + self.num_pad #missing inputs        
        self.num_classes = self.y.shape[1]
        self.steps_per_epoch = None #will pass none, otherwise needs to add repeat to tf data
        self.files = [path]


class CMSQGDataLoader(DataLoader):
    def __init__(self, path, batch_size=512,rank=0,size=1):
        super().__init__(path, batch_size, rank, size)

        self.load_data(path, batch_size,rank,size)
        self.y = np.identity(2)[self.y.astype(np.int32)]
        self.num_pad = 0
        self.num_feat = self.X.shape[2] + self.num_pad #missing inputs
        self.num_classes = self.y.shape[1]
        self.steps_per_epoch = None #will pass none, otherwise needs to add repeat to tf data
        self.files = [path]

        
    
class JetClassDataLoader(DataLoader):
    def __init__(self, path,
                 batch_size=512,rank=0,size=1,chunk_size=5000, **kwargs):
        super().__init__(path, batch_size, rank, size)
        self.chunk_size = chunk_size

        all_files = [os.path.join(self.path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        self.files = np.array_split(all_files,self.size)[self.rank]

        self.get_stats(all_files)
        

    def get_stats(self,file_list):
        #Will assume each file is 100k long
        self.nevts = len(file_list)*100000//5
        self.num_part = h5.File(file_list[0],'r')['data'].shape[1]
        self.num_feat = h5.File(file_list[0],'r')['data'].shape[2]
        self.num_jet = 4 #hardcoded for convenience
        self.num_classes = h5.File(file_list[0],'r')['pid'].shape[1]
        self.steps_per_epoch = self.nevts//self.size//self.batch_size
        self.num_pad = 0
        
    def single_file_generator(self, file_path):
        with h5.File(file_path, 'r') as file:
            data_size = file['data'].shape[0]
            for start in range(0, data_size, self.chunk_size):
                end = min(start + self.chunk_size, data_size)
                jet_chunk = file['jet'][start:end]
                mask_particle = jet_chunk[:,-1] > 1
                jet_chunk = jet_chunk[mask_particle]
                data_chunk = file['data'][start:end].astype(np.float32)
                data_chunk = data_chunk[mask_particle]
                mask_chunk = data_chunk[:, :, 2] != 0  
                
                
                label_chunk = file['pid'][start:end]
                label_chunk = label_chunk[mask_particle]
                data_chunk = self.preprocess(data_chunk, mask_chunk).astype(np.float32)
                jet_chunk = self.preprocess_jet(jet_chunk).astype(np.float32)
                points_chunk = data_chunk[:, :, :2]
                for j in range(data_chunk.shape[0]):                        
                    yield ({
                        'input_features': data_chunk[j],
                        'input_points': points_chunk[j],
                        'input_mask': mask_chunk[j],
                        'input_jet':jet_chunk[j]},                           
                           label_chunk[j])
                    
                
    def interleaved_file_generator(self):
        random.shuffle(self.files)
        generators = [self.single_file_generator(fp) for fp in self.files]
        round_robin_generators = itertools.cycle(generators)

        while True:
            try:
                next_gen = next(round_robin_generators)
                yield next(next_gen)
            except StopIteration:
                break

    def make_tfdata(self):
        dataset = tf.data.Dataset.from_generator(
            self.interleaved_file_generator,
            output_signature=(
                {'input_features': tf.TensorSpec(shape=(self.num_part, self.num_feat), dtype=tf.float32),
                 'input_points': tf.TensorSpec(shape=(self.num_part, 2), dtype=tf.float32),
                 'input_mask': tf.TensorSpec(shape=(self.num_part), dtype=tf.float32),
                 'input_jet': tf.TensorSpec(shape=(self.num_jet), dtype=tf.float32)},
                tf.TensorSpec(shape=(self.num_classes), dtype=tf.int64)
            ))
        return dataset.shuffle(self.batch_size*50).repeat().batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        
