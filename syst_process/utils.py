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
            self.mean_part = [2.496e+1, 0.0, 0.0, 6.744e+1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            self.std_part = [2.320e+1, 1.0, 1.0, 1.296e+2, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            self.mean_jet = [-3.918e-2, -8.923e-2, 8.916e-2, 1.457e-1, -2.469e-2, 4.867e-1]
            self.std_jet  = [1.556e+1, 1.586e+1, 3.483e+1, 2.030e+1, 2.000e+1, 5.815e+1]
        elif samples_name == 'mu_pi':
            self.mean_part = [2.449e+1, 0.0, 0.0, 7.038e+1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            self.std_part = [1.742e+1, 1.0, 1.0, 1.194e+2, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            self.mean_jet = [-3.981e-2, 2.796e-3, 1.387e-1, 3.514e-2, 3.003e-2, 9.278e-3]
            self.std_jet  = [1.575e+1, 1.578e+1, 4.335e+1, 1.807e+1, 1.808e+1, 5.269e+1]
        elif samples_name == 'mu_rho':
            self.mean_part = [2.477e+1, 0.0, 0.0, 6.809e+1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            self.std_part = [2.375e+1, 1.0, 1.0, 1.363e+2, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            self.mean_jet = [5.372e-2, 1.312e-1, -3.785e-2, 3.895e-2, -8.385e-2, -1.141e-1]
            self.std_jet  = [1.515e+1, 1.512e+1, 3.482e+1, 2.002e+1, 1.986e+1, 5.879e+1]
        elif samples_name == 'pi_rho':
            self.mean_part = [2.544e+1, 0.0, 0.0, 6.669e+1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            self.std_part = [2.407e+1, 1.0, 1.0, 1.257e+2, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            self.mean_jet = [3.180e-2, 7.701e-2, -4.538e-2, 3.208e-2, 6.013e-2, 1.542e-1]
            self.std_jet  = [1.498e+1, 1.492e+1, 3.683e+1, 1.816e+1, 1.772e+1, 5.284e+1]
        elif samples_name == 'rho_rho':
            self.mean_part = [2.580e+1, 0.0, 0.0, 6.578e+1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            self.std_part = [3.154e+1, 1.0, 1.0, 1.374e+2, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            self.mean_jet = [2.514e-1, 5.067e-1, -4.509e-1, -1.008e-1, 4.397e-1, 1.293e+0]
            self.std_jet  = [1.932e+1, 1.988e+1, 4.263e+1, 1.875e+1, 1.908e+1, 4.625e+1]
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
        if self.X.shape[1] > 4:
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

class ReconstrctTauDataLoaderWithNpzForSample(DataLoader):
    def __init__(self, path, batch_size=512,rank=0,size=1, nevts=None, data_type='val'):
        super().__init__(path, batch_size, rank, size)

        self.load_data(path, batch_size,rank,size,nevts, data_type)

        self.mean_part = [25.933203, 0.0, 0.0, 70.54197, 0.0]
        self.std_part =  [17.061562, 1.0, 1.0, 115.62808, 1.0]
        self.mean_jet = [0.15880026, -0.0023856324, 0.10423162, 1.7788663, -0.061762687, 0.022895612, -0.039945163, 1.7788663]
        self.std_jet = [30.99985, 30.968163, 89.45671, 0.0018644714, 30.74226, 30.776234, 87.33403, 0.0018644569]
        
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
        tau_p = data['tau_p'][rank:nevts:size]
        tau_m = data['tau_m'][rank:nevts:size]
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
        self.jet = np.concatenate([tau_p, tau_m], axis=1)
        # For truth level study, self.y are the MET
        self.y = MET #met pT, met_phi
        #let's normalize the met pT
        self.y[:,0] = np.log(self.y[:,0])
        self.mask = self.X[:,:,2]!=0

        # self.batch_size = batch_size
        self.nevts = self.X.shape[0] 
        self.num_part = self.X.shape[1]
        self.num_jet = self.jet.shape[1]
                
class TruthTauDataLoader(DataLoader):
    def __init__(self, path, batch_size=512,rank=0,size=1, nevts=None):
        super().__init__(path, batch_size, rank, size)
        self.load_data(path, batch_size,rank,size,nevts)

        self.mean_part = [25.932678, 0.0, 0.0, 70.54288, 0.0]
        self.std_part =  [17.061451, 1.0, 1.0, 115.62893, 1.0]
        if np.any(self.jet[:,2] > 4):
            # self.mean_jet = list(self.mean_nu)
            # self.std_jet  = list(self.std_nu)
            self.mean_jet = [0.05766123, 0.014943519, 0.084477596, -0.01847846, -0.0021721262, -0.016755389]
            self.std_jet = [14.002326, 13.991539, 38.890766, 14.091634, 14.086069, 40.51254]
        else:
            self.mean_jet = [self.mean_jet[0], 0.0, 0.0, self.mean_jet[3], 0.0, 0.0]
            self.std_jet  = [self.std_jet[0],  1.0, 1.0, self.std_jet[3],  1.0, 1.0]
        print("Mean part: {}".format(self.mean_part))
        print("Std part: {}".format(self.std_part))
        print("Mean jet: {}".format(self.mean_jet))
        print("Std jet: {}".format(self.std_jet))
        self.num_pad = 0
        self.num_feat = self.X.shape[2] + self.num_pad #missing inputs
        
        # self.y = np.identity(2)[self.y.astype(np.int32)]
        self.num_classes = self.y.shape[1]
        self.steps_per_epoch = None #will pass none, otherwise needs to add repeat to tf data
        self.files = [path]
    
    def multipl_sample(self, X, y, jet, mul=10):
        import vector
        final_X = []
        final_y = []
        final_jet = []
        for i in range(mul):
            if i == 0:
                final_X = X
                final_y = y
                final_jet = jet
            else:
                np.random.seed(i*128)
                random_beta = np.random.normal(0, (i+1)/2, size=(X.shape[0], ))
                random_beta[random_beta < -0.2] = -0.2
                random_beta[random_beta > 0.2] = 0.2
                temp_X = []
                temp_jet = []
                for j in range(X.shape[1]):
                    temp = vector.arr({
                        'pt': X[:, j, 0],
                        'eta': X[:, j, 1],
                        'phi': X[:, j, 2],
                        'E': X[:, j, 3],
                    })
                    temp = temp.boostZ(random_beta)
                    temp_arr = np.array([temp.pt, temp.eta, temp.phi, temp.E]).reshape(-1, 1, 4)
                    # temp_arr = np.nan_to_num(temp_arr)
                    if j == 0:
                        temp_X = temp_arr
                    else:
                        temp_X = np.concatenate((temp_X, temp_arr), axis=1)
                    del temp
                    if np.isnan(temp_X).any():
                        raise ValueError("temp X contains NaN values!")
                final_X = np.concatenate((final_X, temp_X), axis=0)
                del temp_X
                jet_t = jet.reshape(-1, 2, 3)
                for j in range(jet_t.shape[1]):
                    temp = vector.arr({
                        'px': jet_t[:, j, 0],
                        'py': jet_t[:, j, 1],
                        'pz': jet_t[:, j, 2],
                        'm': np.zeros(jet_t.shape[0]),
                    })
                    temp = temp.boostZ(random_beta)
                    if j == 0:
                        temp_jet = np.array([temp.px, temp.py, temp.pz]).reshape(-1, 3)
                    else:
                        temp_jet = np.concatenate((temp_jet, np.array([temp.px, temp.py, temp.pz]).reshape(-1, 3)), axis=1)
                    del temp
                    if np.isnan(temp_jet).any():
                        raise ValueError("temp jet contains NaN values!")
                final_jet = np.concatenate((final_jet, temp_jet), axis=0)
                del temp_jet, jet_t
                final_y = np.concatenate((final_y, y), axis=0)
                print("Finish multiple times: {}".format(i))
                print("Final X shape: {}".format(final_X.shape))
                print("Final y shape: {}".format(final_y.shape))
                print("Final jet shape: {}".format(final_jet.shape))
        return final_X, final_y, final_jet
    
    def load_data(self,path, batch_size=512,rank=0,size=1,nevts=None):
        self.path = path
        self.X = h5.File(self.path,'r')['X'][rank:nevts:size]
        self.y = h5.File(self.path,'r')['y'][rank:nevts:size]
        self.jet = h5.File(self.path,'r')['nu'][rank:nevts:size]
        #let's normalize the met pT
        # self.X, self.y, self.jet = self.multipl_sample(self.X, self.y, self.jet)
        #add two different labels to identify particles
        self.labels = np.ones((self.X.shape[0],self.X.shape[1],1))
        self.labels[:,2:] = 2
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
        self.mean_X = h5.File(self.path,'r')['mean_X'][:]
        self.std_X = h5.File(self.path,'r')['std_X'][:]
        self.mean_nu = h5.File(self.path,'r')['mean_nu'][:]
        self.std_nu = h5.File(self.path,'r')['std_nu'][:]
        if len(self.mean_X) > 4:
            self.mean_X = np.mean(np.array(self.mean_X).reshape(-1,4),axis=0)
            self.std_X = np.mean(np.array(self.std_X).reshape(-1,4),axis=0)
        if len(self.mean_nu) > 6:
            self.mean_nu = np.mean(np.array(self.mean_nu).reshape(-1,6),axis=0)
            self.std_nu = np.mean(np.array(self.std_nu).reshape(-1,6),axis=0)

class ReconNuPionDataLoader(DataLoader):
    def __init__(self, path, batch_size=512,rank=0,size=1, nevts=None):
        super().__init__(path, batch_size, rank, size)
        self.load_data(path, batch_size,rank,size,nevts)

        self.mean_part = [25.916352631518997, 0.0, 0.0, 70.53927850296773, 0.0]
        self.std_part =  [17.071808899509808, 1.0, 1.0, 115.65273170170438, 1.0]
        self.mean_jet =  [0.016823129638108548, 0.01673338040921922, 0.0829367966161031, -0.002386277422033296, -0.003310371627332507, -0.009401800161693223, 0.13957080263477428, 0.02142042907060498, -0.000604291108585292, -0.013209993791161497, -0.002842133862874096, 0.0028744978853758978, -0.005383584413412035, 0.13957080263477428]
        self.std_jet = [13.975173949266962, 13.981232817524928, 38.85618917630144, 4.918797017211112, 5.1787534477350325, 4.7777959132970285, 0.000692955744935215, 14.059869543852479, 14.0979599859741, 40.51226330089551, 5.8971071903305115, 6.023000064742951, 4.284193133583911, 0.000692955744935215]
        # print("Mean part: {}".format(self.mean_part))
        # print("Std part: {}".format(self.std_part))
        # print("Mean jet: {}".format(self.mean_jet))
        # print("Std jet: {}".format(self.std_jet))
        self.num_pad = 0
        self.num_feat = self.X.shape[2] + self.num_pad #missing inputs
        
        # self.y = np.identity(2)[self.y.astype(np.int32)]
        self.num_classes = self.y.shape[1]
        self.steps_per_epoch = None #will pass none, otherwise needs to add repeat to tf data
        self.files = [path]
    
    def load_data(self,path, batch_size=512,rank=0,size=1,nevts=None):
        self.path = path
        self.X = h5.File(self.path,'r')['X'][rank:nevts:size]
        self.y = h5.File(self.path,'r')['y'][rank:nevts:size]
        self.jet = h5.File(self.path,'r')['total_recon'][rank:nevts:size]
        #let's normalize the met pT
        # self.X, self.y, self.jet = self.multipl_sample(self.X, self.y, self.jet)
        #add two different labels to identify particles
        self.labels = np.ones((self.X.shape[0],self.X.shape[1],1))
        self.labels[:,2:] = 2
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

class ReconTauPredict(DataLoader):
    def __init__(self, path, batch_size=512,rank=0,size=1, nevts=None):
        super().__init__(path, batch_size, rank, size)
        self.load_data(path, batch_size,rank,size,nevts)

        self.mean_part = [25.933203, 0.0, 0.0, 70.54197, 0.0]
        self.std_part =  [17.061562, 1.0, 1.0, 115.62808, 1.0]
        self.mean_jet = [0.15880026, -0.0023856324, 0.10423162, 1.7788663, -0.061762687, 0.022895612, -0.039945163, 1.7788663]
        self.std_jet = [30.99985, 30.968163, 89.45671, 0.0018644714, 30.74226, 30.776234, 87.33403, 0.0018644569]

        self.num_pad = 0
        self.num_feat = self.X.shape[2] + self.num_pad #missing inputs
        
        # self.y = np.identity(2)[self.y.astype(np.int32)]
        self.num_classes = self.y.shape[1]
        self.steps_per_epoch = None #will pass none, otherwise needs to add repeat to tf data
        self.files = [path]

    
    def load_data(self,path, batch_size=512,rank=0,size=1,nevts=None):
        self.path = path
        self.X = h5.File(self.path,'r')['X'][rank:nevts:size]
        self.y = h5.File(self.path,'r')['y'][rank:nevts:size]
        self.jet = h5.File(self.path,'r')['tau'][rank:nevts:size]
        #let's normalize the met pT
        # self.X, self.y, self.jet = self.multipl_sample(self.X, self.y, self.jet)
        #add two different labels to identify particles
        self.labels = np.ones((self.X.shape[0],self.X.shape[1],1))
        self.labels[:,2:] = 2
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
        # self.mean_X = h5.File(self.path,'r')['mean_X'][:]
        # self.std_X = h5.File(self.path,'r')['std_X'][:]
        # self.mean_tau = h5.File(self.path,'r')['mean_tau'][:]
        # self.std_tau = h5.File(self.path,'r')['std_tau'][:]
        
class TauDataLoaderWithGenerator(DataLoader):    
    def __init__(self, path, batch_size=512,rank=0,size=1):
        super().__init__(path, batch_size, rank, size)

        self.load_data(path, batch_size,rank,size)

        self.mean_part = [1.8371206e+01, 0.0,  0.0, 3.9444611e+01, 0.0]
        self.std_part =  [13.570198,  1.0,  1.0, 39.287296,  1.0]
        self.mean_jet = [[0.3682232, -0.21265848, -0.54441476], [-0.2322103, 0.20450957, 0.47339877]]
        self.std_jet  = [[17.644945, 17.901054, 54.378494], [17.735344, 17.621115, 61.020504]]
        
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
        self.pion = self.p_t[:,0:8].reshape(self.p_t.shape[0], 2, -1) #pion 4vector
        #add a one for leptons
        self.pion = np.concatenate([self.pion,np.ones((self.X.shape[0], 2, 1))],-1)
        self.jet = self.y_t[:,:] #neutrino 4vector
        self.jet_0 = self.get_pxyz(self.jet[:,0:4])
        self.jet_1 = self.get_pxyz(self.jet[:,4:8])
        # self.jet = np.concatenate([self.jet_0.reshape(self.jet_0.shape[0], self.jet_0.shape[-1]), self.jet_1.reshape(self.jet_1.shape[0], self.jet_1.shape[-1])], axis=1)
        self.jet = np.concatenate([self.jet_0.reshape(self.jet_0.shape[0], 1, self.jet_0.shape[-1]), self.jet_1.reshape(self.jet_1.shape[0], 1, self.jet_1.shape[-1])], axis=1)
        self.X = np.concatenate([self.pion,self.X],axis=1)
        
        self.y = np.concatenate([self.p_t[:,16].reshape(-1,1),self.p_t[:,18].reshape(-1,1)],axis=1) #met pT, met_phi
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
    
class OmniDataLoaderFortautau(DataLoader):
    def __init__(self, path, batch_size=512,rank=0,size=1):
        super().__init__(path, batch_size, rank, size)
        
        self.mean_jet = [0.057661757, 0.014943423, 0.08447809, -0.018478455, -0.002172197, -0.016755478]
        self.std_jet = [14.002306, 13.991556, 38.890743, 14.091658, 14.086099, 40.51243]
        
        self.path = path
        self.X = h5.File(self.path,'r')['X'][rank::size]
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
