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
import horovod.tensorflow.keras as hvd
# from dummy_hvd import hvd as hvd

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

class TruthTotalTauDataLoaderWithNpz(DataLoader):
    def __init__(self, path, batch_size=512,rank=0,size=1, data_type='train'):
        super().__init__(path, batch_size, rank, size)

        self.load_data(path, batch_size,rank,size, data_type)

        self.mean_part = [9.53190298e+00, 0.0,  0.0, 1.12493561e+02, 0.0]
        self.std_part =  [9.40143725,  1.0,  1.0, 233.26712552,  1.0]
        self.mean_jet = [-1.0836015, 1.6860425, 2.312119, -0.8071776, 1.2596784, 1.8461846]
        self.std_jet  = [11.045118, 11.041079, 269.44342, 13.143577, 13.144216, 356.6083]
        
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
        px = pT*np.cos(phi)
        py = pT*np.sin(phi)
        pz = pT*np.sinh(eta)
        return np.stack([px,py,pz],-1)
    
    def get_ptetaphiE(self,arr):
        pT = arr[:,0]
        eta = arr[:,1]
        phi = arr[:,2]
        m = arr[:,3]
        E = np.sqrt(pT**2 + m**2 + (pT*np.sinh(eta))**2)
        return np.stack([pT,eta,phi,E],-1)
    
    def load_data(self,path, batch_size=512,rank=0,size=1,nevts=None,data_type='train'):
        self.path = path
        path_list = glob.glob(self.path + '**/**/*.npz')
        self.data_type = data_type
        # Load all the data from the npz files
        jet_1 = None
        jet_2 = None
        jet_3 = None
        MET = None
        Type = None
        EventID = None
        tau_p_child1 = None
        tau_p_child2 = None
        tau_m_child1 = None
        tau_m_child2 = None
        nu_p = None
        nu_m = None
        for i in range(len(path_list)):
            data = np.load(path_list[i])
            # print("Type: {}, Number: {}".format(data["Type"][0], data['jet_1'].shape[0]))
            if i == 0:
                jet_1 = data['jet_1']
                jet_2 = data['jet_2']
                jet_3 = data['jet_3']
                MET = data['MET']
                Type = data['Type']
                EventID = data['EventID']
                tau_p_child1 = self.get_ptetaphiE(data['tau_p_child1'])
                tau_p_child2 = self.get_ptetaphiE(data['tau_p_child2'])
                tau_m_child1 = self.get_ptetaphiE(data['tau_m_child1'])
                tau_m_child2 = self.get_ptetaphiE(data['tau_m_child2'])
                nu_p = self.get_pxyz(data['nu_p'])
                nu_m = self.get_pxyz(data['nu_m'])
            else:
                jet_1 = np.concatenate((jet_1, data['jet_1']))
                jet_2 = np.concatenate((jet_2, data['jet_2']))
                jet_3 = np.concatenate((jet_3, data['jet_3']))
                MET = np.concatenate((MET, data['MET']))
                Type = np.concatenate((Type, data['Type']))
                EventID = np.concatenate((EventID, data['EventID']))
                tau_p_child1 = np.concatenate((tau_p_child1, self.get_ptetaphiE(data['tau_p_child1'])))
                tau_p_child2 = np.concatenate((tau_p_child2, self.get_ptetaphiE(data['tau_p_child2'])))
                tau_m_child1 = np.concatenate((tau_m_child1, self.get_ptetaphiE(data['tau_m_child1'])))
                tau_m_child2 = np.concatenate((tau_m_child2, self.get_ptetaphiE(data['tau_m_child2'])) )  
                nu_p = np.concatenate((nu_p, self.get_pxyz(data['nu_p'])))
                nu_m = np.concatenate((nu_m, self.get_pxyz(data['nu_m'])))
        # For truth level study, self.X are all the truth tau children
        self.X = np.concatenate([tau_m_child1.reshape(tau_m_child1.shape[0], 1, tau_m_child1.shape[-1]), tau_m_child2.reshape(tau_m_child2.shape[0], 1, tau_m_child2.shape[-1]), tau_p_child1.reshape(tau_p_child1.shape[0], 1, tau_p_child1.shape[-1]), tau_p_child2.reshape(tau_p_child2.shape[0], 1, tau_p_child2.shape[-1])], axis=1)
        #add a one label to identify particles
        self.X = np.concatenate([self.X,np.ones((self.X.shape[0],self.X.shape[1],1))],-1)
        # For truth level study, self.jet are the truth 
        self.jet = np.concatenate([nu_m, nu_p], axis=1)
        # For truth level study, self.y are the MET
        self.y = MET #met pT, met_phi
        # Then we would shuffle the data for training and testing, then we would split the data into training and testing
        if self.data_type == 'train':
            shuffle_ix = np.random.permutation(np.arange(len(self.X)))
            np.save("shuffle_ix.npy", shuffle_ix)
            self.X = self.X[shuffle_ix]
            self.jet = self.jet[shuffle_ix]
            self.y = self.y[shuffle_ix]
            self.X = self.X[:int(0.7*self.X.shape[0])]
            self.jet = self.jet[:int(0.7*self.jet.shape[0])]
            self.y = self.y[:int(0.7*self.y.shape[0])]
        elif self.data_type == 'test':
            shuffle_ix = np.load("shuffle_ix.npy")
            self.X = self.X[shuffle_ix]
            self.jet = self.jet[shuffle_ix]
            self.y = self.y[shuffle_ix]
            self.X = self.X[int(0.7*self.X.shape[0]):int(0.9*self.X.shape[0])]
            self.jet = self.jet[int(0.7*self.jet.shape[0]):int(0.9*self.jet.shape[0])]
            self.y = self.y[int(0.7*self.y.shape[0]):int(0.9*self.y.shape[0])]    
        #let's normalize the met pT
        self.y[:,0] = np.log(self.y[:,0])
        self.mask = self.X[:,:,2]!=0

        # self.batch_size = batch_size
        self.nevts = self.X.shape[0] 
        self.num_part = self.X.shape[1]
        self.num_jet = self.jet.shape[1]

class TruthTotalTauDataLoaderWithNpzForSample(DataLoader):
    def __init__(self, path, batch_size=512,rank=0,size=1, nevts=None, data_type='val'):
        super().__init__(path, batch_size, rank, size)

        self.load_data(path, batch_size,rank,size,nevts, data_type)

        self.mean_part = [9.53190298e+00, 0.0,  0.0, 1.12493561e+02, 0.0]
        self.std_part =  [9.40143725,  1.0,  1.0, 233.26712552,  1.0]
        self.mean_jet = [-1.0836015, 1.6860425, 2.312119, -0.8071776, 1.2596784, 1.8461846]
        self.std_jet  = [11.045118, 11.041079, 269.44342, 13.143577, 13.144216, 356.6083]
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
        px = pT*np.cos(phi)
        py = pT*np.sin(phi)
        pz = pT*np.sinh(eta)
        return np.stack([px,py,pz],-1)
    
    def get_ptetaphiE(self,arr):
        pT = arr[:,0]
        eta = arr[:,1]
        phi = arr[:,2]
        m = arr[:,3]
        E = np.sqrt(pT**2 + m**2 + (pT*np.sinh(eta))**2)
        return np.stack([pT,eta,phi,E],-1)
    
    def load_data(self,path, batch_size=512,rank=0,size=1,nevts=None,data_type='val'):
        self.path = path
        self.data_type = data_type
        # Load all the data from the npz files
        data = np.load(path)
        val_num = -1 if nevts is None else nevts
        jet_1 = data['jet_1'][:val_num]
        jet_2 = data['jet_2'][:val_num]
        jet_3 = data['jet_3'][:val_num]
        MET = data['MET'][:val_num]
        Type = data['Type'][:val_num]
        EventID = data['EventID'][:val_num]
        tau_p_child1 = self.get_ptetaphiE(data['tau_p_child1'][:val_num])
        tau_p_child2 = self.get_ptetaphiE(data['tau_p_child2'][:val_num])
        tau_m_child1 = self.get_ptetaphiE(data['tau_m_child1'][:val_num])
        tau_m_child2 = self.get_ptetaphiE(data['tau_m_child2'][:val_num])
        nu_p = self.get_pxyz(data['nu_p'][:val_num])
        nu_m = self.get_pxyz(data['nu_m'][:val_num])
        self.EventID = EventID
        self.event_type = Type
        
        # For truth level study, self.X are all the truth tau children
        self.X = np.concatenate([tau_m_child1.reshape(tau_m_child1.shape[0], 1, tau_m_child1.shape[-1]), tau_m_child2.reshape(tau_m_child2.shape[0], 1, tau_m_child2.shape[-1]), tau_p_child1.reshape(tau_p_child1.shape[0], 1, tau_p_child1.shape[-1]), tau_p_child2.reshape(tau_p_child2.shape[0], 1, tau_p_child2.shape[-1])], axis=1)
        #add a one label to identify particles
        self.X = np.concatenate([self.X,np.ones((self.X.shape[0],self.X.shape[1],1))],-1)
        # For truth level study, self.jet are the truth 
        self.jet = np.concatenate([nu_m, nu_p], axis=1)
        # For truth level study, self.y are the MET
        self.y = MET #met pT, met_phi
        #let's normalize the met pT
        self.y[:,0] = np.log(self.y[:,0])
        self.mask = self.X[:,:,2]!=0

        # self.batch_size = batch_size
        self.nevts = self.X.shape[0] 
        self.num_part = self.X.shape[1]
        self.num_jet = self.jet.shape[1]

class RecoTauDataLoaderWithNpzForSample(DataLoader):
    def __init__(self, path, batch_size=512,rank=0,size=1, nevts=None, data_type='val'):
        super().__init__(path, batch_size, rank, size)

        self.load_data(path, batch_size,rank,size,nevts, data_type)

        self.mean_part = [25.932678, 0.0, 0.0, 70.54288, 0.0]
        self.std_part =  [17.061451, 1.0, 1.0, 115.62893, 1.0]
        if self.data_type == 'Lorentz':
            self.mean_jet = [5.766e-2, 0.0,  0.0, -1.848e-2, 0.0,  0.0]
            self.std_jet  = [1.400e+1, 1.0,  1.0, 1.409e+1, 1.0,  1.0]
        else:
            self.mean_jet = [0.05766123, 0.014943519, 0.084477596, -0.01847846, -0.0021721262, -0.016755389]
            self.std_jet  = [14.002326, 13.991539, 38.890766, 14.091634, 14.086069, 40.51254]
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
        px = pT*np.cos(phi)
        py = pT*np.sin(phi)
        pz = pT*np.sinh(eta)
        return np.stack([px,py,pz],-1)
    
    def get_ptetaphiE(self,arr):
        pT = arr[:,0]
        eta = arr[:,1]
        phi = arr[:,2]
        m = arr[:,3]
        E = np.sqrt(pT**2 + m**2 + (pT*np.sinh(eta))**2)
        return np.stack([pT,eta,phi,E],-1)
    
    def load_data(self,path, batch_size=512,rank=0,size=1,nevts=None,data_type='val'):
        self.path = path
        self.data_type = data_type
        # Load all the data from the npz files
        data = np.load(path)
        val_num = -1 if nevts is None else nevts
        jet_1 = self.get_ptetaphiE(data['jet_1'][rank:nevts:size])
        jet_2 = self.get_ptetaphiE(data['jet_2'][rank:nevts:size])
        jet_3 = self.get_ptetaphiE(data['jet_3'][rank:nevts:size])
        MET = data['MET'][rank:nevts:size]
        Type = data['Type'][rank:nevts:size]
        EventID = data['EventID'][rank:nevts:size]
        tau_p_child1 = self.get_ptetaphiE(data['tau_p_child1'][rank:nevts:size])
        tau_p_child2 = self.get_ptetaphiE(data['tau_p_child2'][rank:nevts:size])
        tau_m_child1 = self.get_ptetaphiE(data['tau_m_child1'][rank:nevts:size])
        tau_m_child2 = self.get_ptetaphiE(data['tau_m_child2'][rank:nevts:size])
        if self.data_type == 'Lorentz':
            nu_p = data['nu_p'][rank:nevts:size]
            nu_m = data['nu_m'][rank:nevts:size]
        else:
            nu_p = self.get_pxyz(data['nu_p'][rank:nevts:size])
            nu_m = self.get_pxyz(data['nu_m'][rank:nevts:size])
        # MET_eta = - np.log(np.tan(0.5 * np.arccos((nu_p[:,2] + nu_m[:,2])/ (np.sqrt((nu_p[:,0] + nu_m[:,0])**2 + (nu_p[:,1] + nu_m[:,1])**2 + (nu_p[:,2] + nu_m[:,2])**2)))))
        # MET = np.stack([MET[:,0], MET_eta, MET[:,1]], -1)
        self.EventID = EventID
        self.event_type = Type
        
        # For truth level study, self.X are all the truth tau children
        self.X = np.concatenate([tau_p_child1.reshape(tau_p_child1.shape[0], 1, tau_p_child1.shape[-1]), tau_p_child2.reshape(tau_p_child2.shape[0], 1, tau_p_child2.shape[-1]), tau_m_child1.reshape(tau_m_child1.shape[0], 1, tau_m_child1.shape[-1]), tau_m_child2.reshape(tau_m_child2.shape[0], 1, tau_m_child2.shape[-1])], axis=1)
        self.X = np.concatenate([self.X, jet_1.reshape(jet_1.shape[0], 1, jet_1.shape[-1]), jet_2.reshape(jet_2.shape[0], 1, jet_2.shape[-1]), jet_3.reshape(jet_3.shape[0], 1, jet_3.shape[-1])], axis=1)
        
        #add a one label to identify particles
        self.labels = np.ones((self.X.shape[0],self.X.shape[1],1))
        self.labels[:,2:] = 2
        if self.X.shape[1] > 4:
            self.labels[:,4:] = 0
        # for padding particles, the label is 0
        self.labels[self.X[:,:,0]==0] = 0
        self.X = np.concatenate([self.X,self.labels],-1)
        # For truth level study, self.jet are the truth 
        self.jet = np.concatenate([nu_m, nu_p], axis=1)
        # For truth level study, self.y are the MET
        self.y = MET #met pT, met_phi
        #let's normalize the met pT
        self.y[:,0] = np.log(self.y[:,0])
        self.mask = self.X[:,:,2]!=0

        # self.batch_size = batch_size
        self.nevts = self.X.shape[0] 
        self.num_part = self.X.shape[1]
        self.num_jet = self.jet.shape[1]

class RecoNuPionDataLoaderWithNpzForSample(DataLoader):
    def __init__(self, path, batch_size=512,rank=0,size=1, nevts=None, data_type='val'):
        super().__init__(path, batch_size, rank, size)

        self.load_data(path, batch_size,rank,size,nevts, data_type)

        self.mean_part = [25.916352631518997, 0.0, 0.0, 70.53927850296773, 0.0]
        self.std_part =  [17.071808899509808, 1.0, 1.0, 115.65273170170438, 1.0]
        self.mean_jet =  [0.016823129638108548, 0.01673338040921922, 0.0829367966161031, -0.002386277422033296, -0.003310371627332507, -0.009401800161693223, 0.13957080263477428, 0.02142042907060498, -0.000604291108585292, -0.013209993791161497, -0.002842133862874096, 0.0028744978853758978, -0.005383584413412035, 0.13957080263477428]
        self.std_jet = [13.975173949266962, 13.981232817524928, 38.85618917630144, 4.918797017211112, 5.1787534477350325, 4.7777959132970285, 0.000692955744935215, 14.059869543852479, 14.0979599859741, 40.51226330089551, 5.8971071903305115, 6.023000064742951, 4.284193133583911, 0.000692955744935215]
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
        px = pT*np.cos(phi)
        py = pT*np.sin(phi)
        pz = pT*np.sinh(eta)
        return np.stack([px,py,pz],-1)
    
    def get_ptetaphiE(self,arr):
        pT = arr[:,0]
        eta = arr[:,1]
        phi = arr[:,2]
        m = arr[:,3]
        E = np.sqrt(pT**2 + m**2 + (pT*np.sinh(eta))**2)
        return np.stack([pT,eta,phi,E],-1)
    
    def load_data(self,path, batch_size=512,rank=0,size=1,nevts=None,data_type='val'):
        self.path = path
        self.data_type = data_type
        # Load all the data from the npz files
        data = np.load(path)
        val_num = -1 if nevts is None else nevts
        jet_1 = self.get_ptetaphiE(data['jet_1'][rank:nevts:size])
        jet_2 = self.get_ptetaphiE(data['jet_2'][rank:nevts:size])
        jet_3 = self.get_ptetaphiE(data['jet_3'][rank:nevts:size])
        MET = data['MET'][rank:nevts:size]
        Type = data['Type'][rank:nevts:size]
        EventID = data['EventID'][rank:nevts:size]
        tau_p_child1 = self.get_ptetaphiE(data['tau_p_child1'][rank:nevts:size])
        tau_p_child2 = self.get_ptetaphiE(data['tau_p_child2'][rank:nevts:size])
        tau_m_child1 = self.get_ptetaphiE(data['tau_m_child1'][rank:nevts:size])
        tau_m_child2 = self.get_ptetaphiE(data['tau_m_child2'][rank:nevts:size])
        self.EventID = EventID
        self.event_type = Type
        
        # For truth level study, self.X are all the truth tau children
        self.X = np.concatenate([tau_p_child1.reshape(tau_p_child1.shape[0], 1, tau_p_child1.shape[-1]), tau_p_child2.reshape(tau_p_child2.shape[0], 1, tau_p_child2.shape[-1]), tau_m_child1.reshape(tau_m_child1.shape[0], 1, tau_m_child1.shape[-1]), tau_m_child2.reshape(tau_m_child2.shape[0], 1, tau_m_child2.shape[-1])], axis=1)
        self.X = np.concatenate([self.X, jet_1.reshape(jet_1.shape[0], 1, jet_1.shape[-1]), jet_2.reshape(jet_2.shape[0], 1, jet_2.shape[-1]), jet_3.reshape(jet_3.shape[0], 1, jet_3.shape[-1])], axis=1)
        
        #add a one label to identify particles
        self.labels = np.ones((self.X.shape[0],self.X.shape[1],1))
        self.labels[:,2:] = 2
        if self.X.shape[1] > 4:
            self.labels[:,4:] = 0
        # for padding particles, the label is 0
        self.labels[self.X[:,:,0]==0] = 0
        self.X = np.concatenate([self.X,self.labels],-1)
        # For truth level study, self.jet are the truth 
        self.jet = np.zeros((self.X.shape[0], 14))
        # For truth level study, self.y are the MET
        self.y = MET #met pT, met_phi
        #let's normalize the met pT
        self.y[:,0] = np.log(self.y[:,0])
        self.mask = self.X[:,:,2]!=0

        # self.batch_size = batch_size
        self.nevts = self.X.shape[0] 
        self.num_part = self.X.shape[1]
        self.num_jet = self.jet.shape[1]

class RecoTauDataLoaderWithPKLForSample(DataLoader):
    def __init__(self, path, batch_size=1024,rank=0,size=1, nevts=None, samples_name='none'):
        super().__init__(path, batch_size, rank, size)
        self.samples_name = samples_name
        self.load_data(path, batch_size,rank,size,nevts)

        if samples_name == 'pi_pi':
            self.mean_part = [2.59163526e+01, 0.0, 0.0, 7.05392784e+01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            self.std_part = [17.0718089, 1.0, 1.0, 115.65273143, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            self.mean_jet = [0.01682312963810866, 0.016733380409220386, 0.08293679661612122, 0.021420429070607164, -0.0006042911085853564, -0.013209993791164207]
            self.std_jet  = [13.975173949266443, 13.981232817525068, 38.85618917630569, 14.059869543855184, 14.097959985975313, 40.51226330089419]
        elif samples_name == 'e_pi':
            self.mean_part = [2.470e+1, 0.0, 0.0, 6.949e+1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            self.std_part = [1.750e+1, 1.0, 1.0, 1.192e+2, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            self.mean_jet = [2.198e-2, 1.831e-2, -4.410e-3, -1.366e-2, 3.675e-2, 1.322e-1]
            self.std_jet  = [1.569e+1, 1.582e+1, 4.252e+1, 1.825e+1, 1.823e+1, 5.234e+1]
        elif samples_name == 'e_rho':
            self.mean_part = [14.982864907864142, 0.0, 0.0, 42.79647942777018, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            self.std_part = [15.792115401062988, 1.0, 1.0, 97.37732861435599, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            self.mean_jet = [-0.0002678336552352207, 0.016736370466439304, 0.08861963610428333, 0.06419391577262668, -0.0010769517545723172, 0.15892125628069512]
            self.std_jet  = [14.050414410768337, 14.028565766125833, 39.01819593975496, 17.369162229933927, 17.322592331941568, 52.47183219751106]
        elif samples_name == 'mu_pi':
            self.mean_part = [2.449e+1, 0.0, 0.0, 7.038e+1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            self.std_part = [1.742e+1, 1.0, 1.0, 1.194e+2, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            self.mean_jet = [-3.981e-2, 2.796e-3, 1.387e-1, 3.514e-2, 3.003e-2, 9.278e-3]
            self.std_jet  = [1.575e+1, 1.578e+1, 4.335e+1, 1.807e+1, 1.808e+1, 5.269e+1]
        elif samples_name == 'mu_rho':
            self.mean_part = [14.912302932315121, 0.0, 0.0, 42.950420353288834, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            self.std_part = [15.519213711111917, 1.0, 1.0, 96.64609240683967, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            self.mean_jet = [0.004521378634852432, 0.008900067295884653, -0.055627586346384726, -0.030963372592020283, -0.017430794771537462, -0.05792407995355238]
            self.std_jet  = [13.945134622193716, 13.947684551258607, 38.95641101985201, 17.409530564802242, 17.35883209984128, 53.38571112218628]
        elif samples_name == 'pi_rho':
            self.mean_part = [15.174936586596685, 0.0, 0.0, 41.85549317045239, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            self.std_part = [15.696616138696147, 1.0, 1.0, 92.55279442301007, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            self.mean_jet = [-0.02391130059056275, 0.00963103174019532, 0.0887821064592522, -4.879895705211167e-05, 0.002470162695203731, -0.04313502777777081]
            self.std_jet  = [13.169428059607874, 13.224751262630551, 37.22644797694016, 14.779683471066075, 14.801913665250858, 44.11978535468495]
        elif samples_name == 'rho_rho':
            self.mean_part = [10.936752258785328, 0.0, 0.0, 30.6316162852822, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            self.std_part = [12.67041073494841, 1.0, 1.0, 80.66492595932489, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            self.mean_jet = [-0.005334533901221733, 0.012579453929881869, 0.08474348283190947, 0.0009670746789205529, -0.011116983664680337, 0.001942205415118817]
            self.std_jet  = [13.455550914459518, 13.466844645259748, 37.733618935444994, 13.549154909778169, 13.538312786991147, 39.80704566398323]
        else:
            raise ValueError('samples_name is not valid')
        self.num_pad = 0
        self.num_feat = self.X.shape[2] + self.num_pad #missing inputs
        
        # self.y = np.identity(2)[self.y.astype(np.int32)]
        self.num_classes = self.y.shape[1]
        self.steps_per_epoch = None #will pass none, otherwise needs to add repeat to tf data
        self.files = [path]
    
    def load_data(self,path, batch_size=512,rank=0,size=1,nevts=None):
        import vector
        import pickle
        self.path = path
        # Load all the data from the npz files
        with open(self.path, 'rb') as f:
            data = pickle.load(f)
        if "rho" in self.samples_name:
            val_num = -1 if nevts is None else nevts
            pjet_1 = data['jet_1'][rank:nevts:size]
            pjet_2 = data['jet_2'][rank:nevts:size]
            pjet_3 = data['jet_3'][rank:nevts:size]
            pMET = data['MET'][rank:nevts:size]
            EventID = data['EventID'][rank:nevts:size]
            samples = data['sample'][rank:nevts:size]
            ptau_p_child1 = data['tau_p_child1'][rank:nevts:size]
            ptau_p_child2 = data['tau_p_child2'][rank:nevts:size]
            ptau_p_child3 = data['tau_p_child3'][rank:nevts:size]
            ptau_m_child1 = data['tau_m_child1'][rank:nevts:size]
            ptau_m_child2 = data['tau_m_child2'][rank:nevts:size]
            ptau_m_child3 = data['tau_m_child3'][rank:nevts:size]
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
            tau_p_child3_charge = data['tau_p_child3_charge'][rank:nevts:size]
            tau_p_child3_is_el = data['tau_p_child3_is_el'][rank:nevts:size]
            tau_p_child3_is_mu = data['tau_p_child3_is_mu'][rank:nevts:size]
            tau_p_child3_is_charged_pion = data['tau_p_child3_is_charged_pion'][rank:nevts:size]
            tau_p_child3_is_neutral_part = data['tau_p_child3_is_neutral_part'][rank:nevts:size]
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
            tau_m_child3_charge = data['tau_m_child3_charge'][rank:nevts:size]
            tau_m_child3_is_el = data['tau_m_child3_is_el'][rank:nevts:size]
            tau_m_child3_is_mu = data['tau_m_child3_is_mu'][rank:nevts:size]
            tau_m_child3_is_charged_pion = data['tau_m_child3_is_charged_pion'][rank:nevts:size]
            tau_m_child3_is_neutral_part = data['tau_m_child3_is_neutral_part'][rank:nevts:size]
            jet_1 = np.stack([pjet_1.pt, pjet_1.eta, pjet_1.phi, pjet_1.E, np.zeros_like(pjet_1.pt), np.zeros_like(pjet_1.pt), np.zeros_like(pjet_1.pt), np.zeros_like(pjet_1.pt), np.zeros_like(pjet_1.pt)], -1)
            jet_2 = np.stack([pjet_2.pt, pjet_2.eta, pjet_2.phi, pjet_2.E, np.zeros_like(pjet_2.pt), np.zeros_like(pjet_2.pt), np.zeros_like(pjet_2.pt), np.zeros_like(pjet_2.pt), np.zeros_like(pjet_2.pt)], -1)
            jet_3 = np.stack([pjet_3.pt, pjet_3.eta, pjet_3.phi, pjet_3.E, np.zeros_like(pjet_3.pt), np.zeros_like(pjet_3.pt), np.zeros_like(pjet_3.pt), np.zeros_like(pjet_3.pt), np.zeros_like(pjet_3.pt)], -1)
            MET = np.stack([pMET.pt, pMET.phi], -1)
            tau_p_child1 = np.stack([ptau_p_child1.pt, ptau_p_child1.eta, ptau_p_child1.phi, ptau_p_child1.E, tau_p_child1_charge, tau_p_child1_is_el, tau_p_child1_is_mu, tau_p_child1_is_charged_pion, tau_p_child1_is_neutral_part], -1)
            tau_p_child2 = np.stack([ptau_p_child2.pt, ptau_p_child2.eta, ptau_p_child2.phi, ptau_p_child2.E, tau_p_child2_charge, tau_p_child2_is_el, tau_p_child2_is_mu, tau_p_child2_is_charged_pion, tau_p_child2_is_neutral_part], -1)
            tau_p_child3 = np.stack([ptau_p_child3.pt, ptau_p_child3.eta, ptau_p_child3.phi, ptau_p_child3.E, tau_p_child3_charge, tau_p_child3_is_el, tau_p_child3_is_mu, tau_p_child3_is_charged_pion, tau_p_child3_is_neutral_part], -1)
            tau_m_child1 = np.stack([ptau_m_child1.pt, ptau_m_child1.eta, ptau_m_child1.phi, ptau_m_child1.E, tau_m_child1_charge, tau_m_child1_is_el, tau_m_child1_is_mu, tau_m_child1_is_charged_pion, tau_m_child1_is_neutral_part], -1)
            tau_m_child2 = np.stack([ptau_m_child2.pt, ptau_m_child2.eta, ptau_m_child2.phi, ptau_m_child2.E, tau_m_child2_charge, tau_m_child2_is_el, tau_m_child2_is_mu, tau_m_child2_is_charged_pion, tau_m_child2_is_neutral_part], -1)
            tau_m_child3 = np.stack([ptau_m_child3.pt, ptau_m_child3.eta, ptau_m_child3.phi, ptau_m_child3.E, tau_m_child3_charge, tau_m_child3_is_el, tau_m_child3_is_mu, tau_m_child3_is_charged_pion, tau_m_child3_is_neutral_part], -1)
            del pjet_1, pjet_2, pjet_3, pMET, ptau_p_child1, ptau_p_child2, ptau_m_child1, ptau_m_child2, ptau_p_child3, ptau_m_child3
            
            self.EventID = EventID
            self.event_type = samples
            
            self.X = np.concatenate([tau_p_child1.reshape(tau_p_child1.shape[0], 1, tau_p_child1.shape[-1]), tau_p_child2.reshape(tau_p_child2.shape[0], 1, tau_p_child2.shape[-1]), tau_p_child3.reshape(tau_p_child3.shape[0], 1, tau_p_child3.shape[-1]), tau_m_child1.reshape(tau_m_child1.shape[0], 1, tau_m_child1.shape[-1]), tau_m_child2.reshape(tau_m_child2.shape[0], 1, tau_m_child2.shape[-1]), tau_m_child3.reshape(tau_m_child3.shape[0], 1, tau_m_child3.shape[-1])], axis=1)
            self.X = np.concatenate([self.X, jet_1.reshape(jet_1.shape[0], 1, jet_1.shape[-1]), jet_2.reshape(jet_2.shape[0], 1, jet_2.shape[-1]), jet_3.reshape(jet_3.shape[0], 1, jet_3.shape[-1])], axis=1)
        else:
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
            jet_1 = np.stack([pjet_1.pt, pjet_1.eta, pjet_1.phi, pjet_1.E, np.zeros_like(pjet_1.pt), np.zeros_like(pjet_1.pt), np.zeros_like(pjet_1.pt), np.zeros_like(pjet_1.pt), np.zeros_like(pjet_1.pt)], -1)
            jet_2 = np.stack([pjet_2.pt, pjet_2.eta, pjet_2.phi, pjet_2.E, np.zeros_like(pjet_2.pt), np.zeros_like(pjet_2.pt), np.zeros_like(pjet_2.pt), np.zeros_like(pjet_2.pt), np.zeros_like(pjet_2.pt)], -1)
            jet_3 = np.stack([pjet_3.pt, pjet_3.eta, pjet_3.phi, pjet_3.E, np.zeros_like(pjet_3.pt), np.zeros_like(pjet_3.pt), np.zeros_like(pjet_3.pt), np.zeros_like(pjet_3.pt), np.zeros_like(pjet_3.pt)], -1)
            MET = np.stack([pMET.pt, pMET.phi], -1)
            tau_p_child1 = np.stack([ptau_p_child1.pt, ptau_p_child1.eta, ptau_p_child1.phi, ptau_p_child1.E, tau_p_child1_charge, tau_p_child1_is_el, tau_p_child1_is_mu, tau_p_child1_is_charged_pion, tau_p_child1_is_neutral_part], -1)
            tau_p_child2 = np.stack([ptau_p_child2.pt, ptau_p_child2.eta, ptau_p_child2.phi, ptau_p_child2.E, tau_p_child2_charge, tau_p_child2_is_el, tau_p_child2_is_mu, tau_p_child2_is_charged_pion, tau_p_child2_is_neutral_part], -1)
            tau_m_child1 = np.stack([ptau_m_child1.pt, ptau_m_child1.eta, ptau_m_child1.phi, ptau_m_child1.E, tau_m_child1_charge, tau_m_child1_is_el, tau_m_child1_is_mu, tau_m_child1_is_charged_pion, tau_m_child1_is_neutral_part], -1)
            tau_m_child2 = np.stack([ptau_m_child2.pt, ptau_m_child2.eta, ptau_m_child2.phi, ptau_m_child2.E, tau_m_child2_charge, tau_m_child2_is_el, tau_m_child2_is_mu, tau_m_child2_is_charged_pion, tau_m_child2_is_neutral_part], -1)
            del pjet_1, pjet_2, pjet_3, pMET, ptau_p_child1, ptau_p_child2, ptau_m_child1, ptau_m_child2
            
            self.EventID = EventID
            self.event_type = samples
            
            self.X = np.concatenate([tau_p_child1.reshape(tau_p_child1.shape[0], 1, tau_p_child1.shape[-1]), tau_p_child2.reshape(tau_p_child2.shape[0], 1, tau_p_child2.shape[-1]), tau_m_child1.reshape(tau_m_child1.shape[0], 1, tau_m_child1.shape[-1]), tau_m_child2.reshape(tau_m_child2.shape[0], 1, tau_m_child2.shape[-1])], axis=1)
            self.X = np.concatenate([self.X, jet_1.reshape(jet_1.shape[0], 1, jet_1.shape[-1]), jet_2.reshape(jet_2.shape[0], 1, jet_2.shape[-1]), jet_3.reshape(jet_3.shape[0], 1, jet_3.shape[-1])], axis=1)
        
        #add a one label to identify particles
        self.labels = np.ones((self.X.shape[0],self.X.shape[1],1))
        if "rho" in self.samples_name:
            self.labels[:,6:] = 0
        else:
            self.labels[:,4:] = 0
        # for padding particles, the label is 0
        self.labels[self.X[:,:,0]==0] = 0
        self.X = np.concatenate([self.X,self.labels],-1)
        # For truth level study, self.jet are the truth 
        self.jet = np.zeros((self.X.shape[0], 6))
        # For truth level study, self.y are the MET
        self.y = MET #met pT, met_phi
        #let's normalize the met pT
        self.y[:,0] = np.log(self.y[:,0])
        self.mask = self.X[:,:,2]!=0

        # self.batch_size = batch_size
        self.nevts = self.X.shape[0] 
        self.num_part = self.X.shape[1]
        self.num_jet = self.jet.shape[1]
