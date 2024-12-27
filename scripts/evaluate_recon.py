import os
import h5py as h5
import numpy as np
import tensorflow as tf
from tensorflow import keras
import horovod.tensorflow.keras as hvd
import argparse
import pickle
from PET_jetnet import PET_jetnet
import utils
import plot_utils
import matplotlib.pyplot as plt
import logging
import glob

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process jet data.")
    parser.add_argument("--dataset", default="tautotalwithNpz", help="Folder containing input files")
    # parser.add_argument("--folder", default="/global/cfs/cdirs/m2616/avencast/Quantum_Entanglement/workspace/results/pi_pi/ml_export/", help="Folder containing input files")
    parser.add_argument("--folder", default="/global/cfs/cdirs/m2616/avencast/Quantum_Entanglement/workspace.new/results/", help="Folder containing input files")
    # parser.add_argument("--folder", default="/global/cfs/cdirs/m2616/avencast/Quantum_Entanglement/workspace_12.09/results/pi_pi/ml_export/", help="Folder containing input files")
    # parser.add_argument("--folder", default="/global/cfs/cdirs/m2616/avencast/Quantum_Entanglement/workspace_12.05/results/pi_pi/ml_export/", help="Folder containing input files")
    # parser.add_argument("--folder", default="/pscratch/sd/b/baihong/data/NumpyData/reco/", help="Folder containing input files")
    parser.add_argument("--mode", default="generator", help="Loss type to train the model: [all/classifier/generator]")
    parser.add_argument("--fine_tune", action='store_true', help="Fine tune a model")
    parser.add_argument("--local", action='store_true', help="Use local embedding")
    parser.add_argument("--num_layers", type=int, default=8, help="Number of transformer layers")
    parser.add_argument("--drop_probability", type=float, default=0.0, help="Stochastic Depth drop probability")
    parser.add_argument("--simple", action='store_true', help="Use simplified head model")
    parser.add_argument("--talking_head", action='store_true', help="Use talking head attention")
    parser.add_argument("--layer_scale", action='store_true', help="Use layer scale in the residual connections")
    parser.add_argument("--sample", action='store_true', default=False, help="Sample from trained model")

    parser.add_argument("--plot_folder", default="../plots", help="Folder to save the outputs")
    return parser.parse_args()

def get_data_info(flags):
    if flags.dataset == 'jetnet150':
        test = utils.jetnetDataLoader(os.path.join(flags.folder,'jetnet','test_150.h5'),rank = hvd.rank(),size = hvd.size(),big=True)
        return test
        
    elif flags.dataset == 'tautotalwithNpz':
        truth_path = flags.folder
        truth_path_list = glob.glob(truth_path + '**/**/*.npz')
        test_loader_list = []
        for truth_path in truth_path_list:
            test_loader_list.append(utils.RecoTauDataLoaderWithNpzForSample(truth_path,hvd.rank(),hvd.size(),nevts=10000,data_type='val'))
        
        return test_path_list, test_loader_list


def load_data_and_model(flags):
    
    truth_path = flags.folder
    # truth_path_list = glob.glob(truth_path + '*_particles.pkl')
    # truth_path_list = glob.glob(truth_path + 'OmniLearn_pi_pi_recon_particles.pkl')
    # truth_path_list = glob.glob(truth_path + '*/ml_export/*_recon.npz')
    truth_path_list = glob.glob(truth_path + '**/ml_export/OmniLearn_pi_pi_recon.npz')
    test_loader_list = []
    for truth_path in truth_path_list:
        test_loader_list.append(utils.RecoTauDataLoaderWithNpzForSample(truth_path,rank=hvd.rank(),size=hvd.size(),data_type='val'))
        # test_loader_list.append(utils.RecoTauDataLoaderWithNpzForSample(truth_path,rank=hvd.rank(),size=hvd.size(),data_type='Lorentz'))
        # test_loader_list.append(utils.RecoTauDataLoaderWithPKLForSample(truth_path,rank=hvd.rank(),size=hvd.size(),data_type='val'))
    test = test_loader_list[0]
    model = PET_jetnet(num_feat=test.num_feat,
                       num_jet=test.num_jet,
                       num_classes=test.num_classes,
                       num_part=test.num_part,
                       local=flags.local,
                       num_layers=flags.num_layers,
                       drop_probability=flags.drop_probability,
                       simple=flags.simple, layer_scale=flags.layer_scale,
                       talking_head=flags.talking_head,
                       mode=flags.mode, fine_tune=False, model_name=None, use_mean=flags.fine_tune)
    
    # model_name = "/pscratch/sd/b/baihong/data/checkpoints/PET_pipi_recon_Lorentz_8_local_layer_scale_token_baseline_generator.weights.h5"
    # model_name = "/pscratch/sd/b/baihong/data/checkpoints/PET_pipi_recon_6_local_layer_scale_token_baseline_generator.weights.h5"
    model_name = "/pscratch/sd/b/baihong/data/checkpoints/PET_pipi_recon_8_local_layer_scale_token_baseline_generator.weights.h5"
    model.load_weights(model_name)
    return truth_path_list, test_loader_list, model


def sample_data(test, model, flags, sample_name):
    """ Sample data using the model and save to file. """
    part,point,mask,jet,met,EventID,event_type, pion = test.make_eval_data(preprocess=True)
    
    nsplit = 50
    total_j = model.generate(nsplit,
                       met,part,point,mask,
                       use_tqdm=hvd.rank()==0)
    
    total_jet = []
    for i in range (total_j.shape[1]):
        total_jet.append(test.revert_preprocess_jet(total_j[:,i]).reshape(-1,1,6))
    total_jet = np.concatenate(total_jet, axis=1)
    total_jet = hvd.allgather(total_jet)
    EventID = hvd.allgather(tf.constant(EventID)).numpy()
    event_type = hvd.allgather(tf.constant(event_type)).numpy()
    pion = hvd.allgather(tf.constant(pion)).numpy()
    
    # total_jet = np.array(total_jet).reshape(-1, 100, 6)
    # part = test.revert_preprocess(part, mask)
    # part = hvd.allgather(part)

    if hvd.rank() == 0:
        dict = {
            'nu_p':total_jet[:,:,:3],
            'nu_m':total_jet[:,:,3:],
            'EventID':EventID,
            'event_type':event_type,
        }
        np.savez(sample_name, **dict)
            
            
def get_generated_data(sample_name):
    with h5.File(sample_name,"r") as h5f:
        jets_gen = h5f['jet'][:]
        particles_gen = h5f['part'][:]
        
    return jets_gen, particles_gen



def plot(jet1,jet2,title,plot_folder):
    var_names = ['Neutrino p$_{x}$ [GeV]', 'Neutrino p$_{y}$ [GeV]','Neutrino p$_{z}$ [GeV]','M$_W$ [GeV]']
    
    for ivar in range(len(var_names)):                    
        feed_dict = {
            'nu_truth':jet1[:,ivar],
            'nu_gen':  jet2[:,ivar]
        }
        

        fig,gs,binning = plot_utils.HistRoutine(feed_dict,xlabel=var_names[ivar],
                                                plot_ratio=True,
                                                reference_name='nu_truth',
                                                ylabel= 'Normalized entries')
        ax0 = plt.subplot(gs[0])     
        fig.savefig('{}/Neutrino_{}_{}.pdf'.format(plot_folder,title,ivar),bbox_inches='tight')

            
def plot_results(jets, jets_gen, particles, particles_gen, flags):
    """ Plot the results using the utility functions. """
    plot(jets, jets_gen, title='Jet', plot_folder=flags.plot_folder)

def add_w(nu,part):
    def get_pxyz(arr):
        pT = arr[:,0]
        eta = arr[:,1]
        phi = arr[:,2]
        E = arr[:,3]
        px = pT*np.cos(phi)
        py = pT*np.sin(phi)
        pz = pT*np.sinh(eta)
        return np.stack([px,py,pz],-1)
    
    def get_ptetaphi(arr):
        px = arr[:,0]
        py = arr[:,1]
        pz = arr[:,2]
        pT = np.sqrt(px**2 + py**2)
        eta = 0.5*np.log((np.sqrt(px**2 + py**2 + pz**2) + pz)/(np.sqrt(px**2 + py**2 + pz**2) - pz))
        phi = np.arctan2(py,px)
        return np.stack([pT,eta,phi],-1)

    lepton = get_pxyz(part[:,0])
    e_lep = np.sqrt(np.sum(lepton**2,1,keepdims=True))
    e_nu = np.sqrt(np.sum(nu**2,1,keepdims=True))
    w = lepton + nu
    lepton_0 = get_ptetaphi(part[:,0])
    mw2 = (e_lep+e_nu)**2 - np.sum(w**2,1,keepdims=True)
    return np.concatenate([nu,np.sqrt(np.abs(mw2))],-1)
    

def main():
    plot_utils.SetStyle()
    utils.setup_gpus()
    if hvd.rank()==0:logging.info("Horovod and GPUs initialized successfully.")
    flags = parse_arguments()
    
    if flags.sample:
        # if hvd.rank()==0:logging.info("Sampling the data with Lorentz samples.")
        # if hvd.rank()==0:logging.info("Sampling the data with boost.")
        if hvd.rank()==0:logging.info("Sampling the data without boost.")
        test_path_list, test_loader_list, model = load_data_and_model(flags)
        for i in range(len(test_path_list)):
            # test_name = test_path_list[i].replace("_recon.npz", "_recon_eval_L.npz")
            # test_name = test_path_list[i].replace("_recon.npz", "_recon_eval_com.npz")
            test_name = test_path_list[i].replace("_recon.npz", "_recon_with_obs.npz")
            # test_name = test_path_list[i].replace(".pkl", "_eval.npz")
            if os.path.exists(test_name):
                continue
            else:
                if hvd.rank()==0:logging.info("Sampling the {}.".format(test_name))
                test = test_loader_list[i]
                sample_data(test, model, flags, test_name)
    else:
        if hvd.rank()==0:logging.info("Loading saved samples.")
        # Load and process data, generate plots, etc.        
        test = get_data_info(flags)
        particles,_,_,jets,_ = test.make_eval_data()
        jets_gen, particles_gen = get_generated_data(sample_name)
        jets = add_w(jets,particles)
        jets_gen = add_w(jets_gen,particles_gen)
        # Plot results
        plot_results(jets, jets_gen, particles, particles_gen, flags)

if __name__ == '__main__':
    main()

