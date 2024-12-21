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

name_map = {
    "pi_pi": 0,
    "pi_rho": 1,
    "lep_pi": 2,
    "lep_rho": 3,
    "QCD": 4,
    "rho_rho": 5,
    "tt": 6,
    "Wlnu": 7,
    "Wtaunu": 8,
    "Zll": 9
}

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process jet data.")
    parser.add_argument("--dataset", default="tautotalwithNpz", help="Folder containing input files")
    parser.add_argument("--folder", default="/global/cfs/cdirs/m2616/avencast/Quantum_Entanglement/workspace/results/pi_pi/systematics", help="Folder containing input files")
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
    truth_path_list = glob.glob(truth_path + '/variation.*/merged.pkl')
    # truth_path_list = glob.glob(truth_path + 'ml_export/*_3_recon.npz')
    test_loader_list = []
    for truth_path in truth_path_list:
        test_loader_list.append(utils.RecoTauDataLoaderWithPKLForSample(truth_path,rank=hvd.rank(),size=hvd.size(),data_type='val'))
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
    
    model_name = "/pscratch/sd/b/baihong/data/checkpoints/PET_pipi_recon_8_local_layer_scale_token_baseline_generator.weights.h5"
    model.load_weights(model_name)
    return truth_path_list, test_loader_list, model

def data_split(total_jet, EventID, event_type):
    """ Split the data according to the event type. """
    pipi_jet = total_jet[event_type == 0]
    pirho_jet = total_jet[event_type == 1]
    leppi_jet = total_jet[event_type == 2]
    leprho_jet = total_jet[event_type == 3]
    qcd_jet = total_jet[event_type == 4]
    rhorho_jet = total_jet[event_type == 5]
    tt_jet = total_jet[event_type == 6]
    wlnu_jet = total_jet[event_type == 7]
    wtaunu_jet = total_jet[event_type == 8]
    zll_jet = total_jet[event_type == 9]
    
    pipi_ID = EventID[event_type == 0]
    pirho_ID = EventID[event_type == 1]
    leppi_ID = EventID[event_type == 2]
    leprho_ID = EventID[event_type == 3]
    qcd_ID = EventID[event_type == 4]
    rhorho_ID = EventID[event_type == 5]
    tt_ID = EventID[event_type == 6]
    wlnu_ID = EventID[event_type == 7]
    wtaunu_ID = EventID[event_type == 8]
    zll_ID = EventID[event_type == 9]
    
    return pipi_jet, pirho_jet, leppi_jet, leprho_jet, qcd_jet, rhorho_jet, tt_jet, wlnu_jet, wtaunu_jet, zll_jet, pipi_ID, pirho_ID, leppi_ID, leprho_ID, qcd_ID, rhorho_ID, tt_ID, wlnu_ID, wtaunu_ID, zll_ID


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
    pipi_jet, pirho_jet, leppi_jet, leprho_jet, qcd_jet, rhorho_jet, tt_jet, wlnu_jet, wtaunu_jet, zll_jet, pipi_ID, pirho_ID, leppi_ID, leprho_ID, qcd_ID, rhorho_ID, tt_ID, wlnu_ID, wtaunu_ID, zll_ID = data_split(total_jet, EventID, event_type)

    if hvd.rank() == 0:
        def save_if_not_empty(jet, ID, sample_name, particle_type):
            if jet.shape[0] > 0:
                data_dict = {
                    'nu_p': jet[:, :, :3],
                    'nu_m': jet[:, :, 3:],
                    'EventID': ID,
                }
            else:
                data_dict = {
                    'nu_p': [],
                    'nu_m': [],
                    'EventID': [],
                }
            np.savez(sample_name.replace("merged", particle_type), **data_dict)

        save_if_not_empty(pipi_jet, pipi_ID, sample_name, "pi_pi_particle")
        save_if_not_empty(pirho_jet, pirho_ID, sample_name, "pi_rho_particle")
        save_if_not_empty(leppi_jet, leppi_ID, sample_name, "lep_pi_particle")
        save_if_not_empty(leprho_jet, leprho_ID, sample_name, "lep_rho_particle")
        save_if_not_empty(qcd_jet, qcd_ID, sample_name, "QCD_particle")
        save_if_not_empty(rhorho_jet, rhorho_ID, sample_name, "rho_rho_particle")
        save_if_not_empty(tt_jet, tt_ID, sample_name, "tt_particle")
        save_if_not_empty(wlnu_jet, wlnu_ID, sample_name, "Wlnu_particle")
        save_if_not_empty(wtaunu_jet, wtaunu_ID, sample_name, "Wtaunu_particle")
        save_if_not_empty(zll_jet, zll_ID, sample_name, "Zll_particle")
        
            
            
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
        if hvd.rank()==0:logging.info("Sampling the data without boost.")
        test_path_list, test_loader_list, model = load_data_and_model(flags)
        for i in range(len(test_path_list)):
            test_name = test_path_list[i].replace(".pkl", "_eval.npz")
            if os.path.exists(test_name.replace("merged", "pi_pi_particle")):
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

