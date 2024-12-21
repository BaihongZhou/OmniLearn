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
    parser.add_argument("--dataset", default="tauReconWithNpz", help="Folder containing input files")
    # parser.add_argument("--folder", default="/global/cfs/cdirs/m2616/avencast/Quantum_Entanglement/workspace/results/pi_pi/ml_export/", help="Folder containing input files")
    parser.add_argument("--folder", default="/global/cfs/cdirs/m2616/avencast/Quantum_Entanglement/workspace.new/results/", help="Folder containing input files")
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
        
    elif flags.dataset == 'tauReconWithNpz':
        truth_path = flags.folder
        truth_path_list = glob.glob(truth_path + '**/**/*.npz')
        test_loader_list = []
        for truth_path in truth_path_list:
            test_loader_list.append(utils.ReconstrctTauDataLoaderWithNpzForSample(truth_path,hvd.rank(),hvd.size(),nevts=10000,data_type='val'))
        
        return test_path_list, test_loader_list


def load_data_and_model(flags):
    
    truth_path = flags.folder
    # truth_path_list = glob.glob(truth_path + 'OmniLearn_pi_pi_recon_particles.pkl')
    # truth_path_list = glob.glob(truth_path + 'OmniLearn_pi_pi_recon_particles.pkl')
    truth_path_list = glob.glob(truth_path + '*/ml_export/OmniLearn_pi_pi_recon.npz')
    test_loader_list = []
    for truth_path in truth_path_list:
        test_loader_list.append(utils.ReconstrctTauDataLoaderWithNpzForSample(truth_path,rank=hvd.rank(),size=hvd.size(),data_type='val'))
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
    # model_name = "/pscratch/sd/b/baihong/data/checkpoints/PET_pipi_recon_boosted_total_8_local_layer_scale_token_baseline_generator_0.weights.h5"
    model_name = "/pscratch/sd/b/baihong/data/checkpoints/PET_pipi_tau_recon_8_local_layer_scale_token_baseline_generator.weights.h5"
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
        total_jet.append(test.revert_preprocess_jet(total_j[:,i]).reshape(-1,1,8))
    total_jet = np.concatenate(total_jet, axis=1)
    total_jet = hvd.allgather(total_jet).numpy()
    EventID = hvd.allgather(tf.constant(EventID)).numpy()
    event_type = hvd.allgather(tf.constant(event_type)).numpy()
    pion = hvd.allgather(tf.constant(pion)).numpy()
    
    tau_p, tau_m, nu_p, nu_m = reverse_tau(total_jet, pion)

    if hvd.rank() == 0:
        dict = {
            'nu_p':nu_p,
            'nu_m':nu_m,
            'tau_p':tau_p,
            'tau_m':tau_m,
            'EventID':EventID,
            'event_type':event_type
        }
        np.savez(sample_name, **dict)
            
            
def get_generated_data(sample_name):
    with h5.File(sample_name,"r") as h5f:
        jets_gen = h5f['jet'][:]
        particles_gen = h5f['part'][:]
        
    return jets_gen, particles_gen

def reverse_tau(tau, pion):
    import vector
    tau_p_child1 = pion[:,0,:4]
    tau_p_child2 = pion[:,1,:4]
    tau_p_child1 = vector.arr({
        'pt':tau_p_child1[:,0],
        'eta':tau_p_child1[:,1],
        'phi':tau_p_child1[:,2],
        'mass':tau_p_child1[:,3]
    })
    tau_p_child2 = vector.arr({
        'pt':tau_p_child2[:,0],
        'eta':tau_p_child2[:,1],
        'phi':tau_p_child2[:,2],
        'mass':tau_p_child2[:,3]
    })
    tau_p_child = tau_p_child1 + tau_p_child2
    tau_m_child1 = pion[:,2,:4]
    tau_m_child2 = pion[:,3,:4]
    tau_m_child1 = vector.arr({
        'pt':tau_m_child1[:,0],
        'eta':tau_m_child1[:,1],
        'phi':tau_m_child1[:,2],
        'mass':tau_m_child1[:,3]
    })
    tau_m_child2 = vector.arr({
        'pt':tau_m_child2[:,0],
        'eta':tau_m_child2[:,1],
        'phi':tau_m_child2[:,2],
        'mass':tau_m_child2[:,3]
    })
    tau_m_child = tau_m_child1 + tau_m_child2
    raw_tau_p = tau[:,:,:4]
    raw_tau_m = tau[:,:,4:]
    tau_p = []
    tau_m = []
    nu_p = []
    nu_m = []
    # for i in range(raw_tau_p.shape[1]):
    #     tau_p_temp = raw_tau_p[:,i]
    #     tau_m_temp = raw_tau_p[:,i]
    #     tau_p_final_pt = np.exp(tau_p_temp[:,0])
    #     tau_p_final_eta = tau_p_temp[:,1] + tau_p_child.eta
    #     tau_p_final_phi = tau_p_temp[:,2] + tau_p_child.phi
    #     tau_p_final_mass = tau_p_temp[:,3]
    #     tau_m_final_pt = np.exp(tau_m_temp[:,0])
    #     tau_m_final_eta = tau_m_temp[:,1] + tau_m_child.eta
    #     tau_m_final_phi = tau_m_temp[:,2] + tau_m_child.phi
    #     tau_m_final_mass = tau_m_temp[:,3]
    #     tau_p_temp = vector.arr({
    #         'pt':tau_p_final_pt,
    #         'eta':tau_p_final_eta,
    #         'phi':tau_p_final_phi,
    #         'mass':tau_p_final_mass
    #     })
    #     tau_m_temp = vector.arr({
    #         'pt':tau_m_final_pt,
    #         'eta':tau_m_final_eta,
    #         'phi':tau_m_final_phi,
    #         'mass':tau_m_final_mass
    #     })
    #     nu_p_temp = tau_p_temp - tau_p_child
    #     nu_m_temp = tau_m_temp - tau_m_child
    #     if len(tau_p) == 0:
    #         tau_p = np.stack([tau_p_final_pt, tau_p_final_eta, tau_p_final_phi, tau_p_final_mass],-1).reshape(-1,1,4)
    #         tau_m = np.stack([tau_m_final_pt, tau_m_final_eta, tau_m_final_phi, tau_m_final_mass],-1).reshape(-1,1,4)
    #         nu_p = np.stack([nu_p_temp.pt, nu_p_temp.eta, nu_p_temp.phi, nu_p_temp.mass],-1).reshape(-1,1,4)
    #         nu_m = np.stack([nu_m_temp.pt, nu_m_temp.eta, nu_m_temp.phi, nu_m_temp.mass],-1).reshape(-1,1,4)
    #     else:
    #         tau_p = np.concatenate([tau_p,np.stack([tau_p_final_pt, tau_p_final_eta, tau_p_final_phi, tau_p_final_mass],-1).reshape(-1,1,4)],1)
    #         tau_m = np.concatenate([tau_m,np.stack([tau_m_final_pt, tau_m_final_eta, tau_m_final_phi, tau_m_final_mass],-1).reshape(-1,1,4)],1)
    #         nu_p = np.concatenate([nu_p,np.stack([nu_p_temp.pt, nu_p_temp.eta, nu_p_temp.phi, nu_p_temp.mass],-1).reshape(-1,1,4)],1)
    #         nu_m = np.concatenate([nu_m,np.stack([nu_m_temp.pt, nu_m_temp.eta, nu_m_temp.phi, nu_m_temp.mass],-1).reshape(-1,1,4)],1)
    for i in range(raw_tau_p.shape[1]):
        tau_p_temp = raw_tau_p[:,i]
        tau_m_temp = raw_tau_p[:,i]
        tau_p_px = tau_p_temp[:,0]
        tau_p_py = tau_p_temp[:,1]
        tau_p_pz = tau_p_temp[:,2]
        tau_p_mass = tau_p_temp[:,3]
        tau_m_px = tau_m_temp[:,0]
        tau_m_py = tau_m_temp[:,1]
        tau_m_pz = tau_m_temp[:,2]
        tau_m_mass = tau_m_temp[:,3]
        tau_p_par = vector.arr({
            'px':tau_p_px,
            'py':tau_p_py,
            'pz':tau_p_pz,
            'mass':tau_p_mass
        })
        tau_m_par = vector.arr({
            'px':tau_m_px,
            'py':tau_m_py,
            'pz':tau_m_pz,
            'mass':tau_m_mass
        })
        nu_p_par = tau_p_par - tau_p_child
        nu_m_par = tau_m_par - tau_m_child
        if i == 0:
            tau_p = np.stack([tau_p_px, tau_p_py, tau_p_pz, tau_p_mass],-1).reshape(-1,1,4)
            tau_m = np.stack([tau_m_px, tau_m_py, tau_m_pz, tau_m_mass],-1).reshape(-1,1,4)
            nu_p = np.stack([nu_p_par.px, nu_p_par.py, nu_p_par.pz],-1).reshape(-1,1,3)
            nu_m = np.stack([nu_m_par.px, nu_m_par.py, nu_m_par.pz],-1).reshape(-1,1,3)
        else:
            tau_p = np.concatenate([tau_p,np.stack([tau_p_px, tau_p_py, tau_p_pz, tau_p_mass],-1).reshape(-1,1,4)],1)
            tau_m = np.concatenate([tau_m,np.stack([tau_m_px, tau_m_py, tau_m_pz, tau_m_mass],-1).reshape(-1,1,4)],1)
            nu_p = np.concatenate([nu_p,np.stack([nu_p_par.px, nu_p_par.py, nu_p_par.pz],-1).reshape(-1,1,3)],1)
            nu_m = np.concatenate([nu_m,np.stack([nu_m_par.px, nu_m_par.py, nu_m_par.pz],-1).reshape(-1,1,3)],1)
    return tau_p, tau_m, nu_p, nu_m

    
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
            # test_name = test_path_list[i].replace("_recon.npz", "_recon_eval_B.npz")
            test_name = test_path_list[i].replace("_recon.npz", "_recon_eval_T.npz")
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

