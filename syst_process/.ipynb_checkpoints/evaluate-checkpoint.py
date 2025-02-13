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
    "pi_pi_MM": 10,
    "pi_pi_PM": 11,
    "pi_pi_MP": 12,
    "pi_pi_PP": 13,
    "pi_rho": 1,
    "pi_rho_MM": 14,
    "pi_rho_PM": 15,
    "pi_rho_MP": 16,
    "pi_rho_PP": 17,
    "lep_pi": 2,
    "lep_pi_MM": 18,
    "lep_pi_PM": 19,
    "lep_pi_MP": 20,
    "lep_pi_PP": 21,
    "lep_rho": 3,
    "lep_rho_MM": 22,
    "lep_rho_PM": 23,
    "lep_rho_MP": 24,
    "lep_rho_PP": 25,
    "QCD": 4,
    "rho_rho": 5,
    "rho_rho_MM": 26,
    "rho_rho_PM": 27,
    "rho_rho_MP": 28,
    "rho_rho_PP": 29,
    "tt": 6,
    "Wlnu": 7,
    "Wtaunu": 8,
    "Zll": 9
}

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process jet data.")
    parser.add_argument("--dataset", default="tautotalwithNpz", help="Folder containing input files")
    # parser.add_argument("--folder", default="/global/cfs/cdirs/m2616/avencast/Quantum_Entanglement/workspace/results/pi_pi/systematics", help="Folder containing input files")
    parser.add_argument("--folder", default="/global/cfs/cdirs/m2616/avencast/Quantum_Entanglement/workspace/results/", help="Folder containing input files")
    parser.add_argument("--mode", default="generator", help="Loss type to train the model: [all/classifier/generator]")
    parser.add_argument("--train_channel", type=str, default="pi_pi", help="Batch size")
    parser.add_argument("--fine_tune", action='store_true', help="Fine tune a model")
    parser.add_argument("--local", action='store_true', help="Use local embedding")
    parser.add_argument("--num_layers", type=int, default=8, help="Number of transformer layers")
    parser.add_argument("--drop_probability", type=float, default=0.0, help="Stochastic Depth drop probability")
    parser.add_argument("--simple", action='store_true', help="Use simplified head model")
    parser.add_argument("--talking_head", action='store_true', help="Use talking head attention")
    parser.add_argument("--layer_scale", action='store_true', help="Use layer scale in the residual connections")
    parser.add_argument("--sample", action='store_true', default=True, help="Sample from trained model")
    parser.add_argument("--plot_folder", default="../plots", help="Folder to save the outputs")
    return parser.parse_args()


def load_data_and_model(flags):
    
    truth_path = flags.folder
    truth_path_list = [path for path in glob.glob(truth_path + flags.train_channel + '/systematics/variation.*/merged.pkl') if (path.split('/')[-2].split('.')[-1] == 'plus' and path.split('/')[-2].split('.')[-2] != 'soft_met')]
    test_loader_list = []
    for truth_path in truth_path_list:
        test_loader_list.append(utils.RecoTauDataLoaderWithPKLForSample(truth_path,rank=hvd.rank(),size=hvd.size(),samples_name=flags.train_channel))
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
    if flags.train_channel == "pi_pi":
        model_name = "/pscratch/sd/b/baihong/data/checkpoints/PET_pipi_8_local_layer_scale_token_baseline_generator.weights.h5"
    elif flags.train_channel == "pi_rho":
        model_name = "/pscratch/sd/b/baihong/data/checkpoints/PET_pirho_8_local_layer_scale_token_baseline_generator.weights.h5"
    elif flags.train_channel == "e_pi":
        model_name = "/pscratch/sd/b/baihong/data/checkpoints/PET_epi_8_local_layer_scale_token_baseline_generator.weights.h5"
    elif flags.train_channel == "e_rho":
        model_name = "/pscratch/sd/b/baihong/data/checkpoints/PET_erho_8_local_layer_scale_token_baseline_generator.weights.h5"
    elif flags.train_channel == "mu_pi":
        model_name = "/pscratch/sd/b/baihong/data/checkpoints/PET_mupi_8_local_layer_scale_token_baseline_generator.weights.h5"
    elif flags.train_channel == "mu_rho":
        model_name = "/pscratch/sd/b/baihong/data/checkpoints/PET_murho_8_local_layer_scale_token_baseline_generator.weights.h5"
    elif flags.train_channel == "rho_rho":
        model_name = "/pscratch/sd/b/baihong/data/checkpoints/PET_rhorho_8_local_layer_scale_token_baseline_generator.weights.h5"
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
    pipi_MM_jet = total_jet[event_type == 10]
    pipi_PM_jet = total_jet[event_type == 11]
    pipi_MP_jet = total_jet[event_type == 12]
    pipi_PP_jet = total_jet[event_type == 13]
    pirho_MM_jet = total_jet[event_type == 14]
    pirho_PM_jet = total_jet[event_type == 15]
    pirho_MP_jet = total_jet[event_type == 16]
    pirho_PP_jet = total_jet[event_type == 17]
    leppi_MM_jet = total_jet[event_type == 18]
    leppi_PM_jet = total_jet[event_type == 19]
    leppi_MP_jet = total_jet[event_type == 20]
    leppi_PP_jet = total_jet[event_type == 21]
    leprho_MM_jet = total_jet[event_type == 22]
    leprho_PM_jet = total_jet[event_type == 23]
    leprho_MP_jet = total_jet[event_type == 24]
    leprho_PP_jet = total_jet[event_type == 25]
    rhorho_MM_jet = total_jet[event_type == 26]
    rhorho_PM_jet = total_jet[event_type == 27]
    rhorho_MP_jet = total_jet[event_type == 28]
    rhorho_PP_jet = total_jet[event_type == 29]
    
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
    pipi_MM_ID = EventID[event_type == 10]
    pipi_PM_ID = EventID[event_type == 11]
    pipi_MP_ID = EventID[event_type == 12]
    pipi_PP_ID = EventID[event_type == 13]
    pirho_MM_ID = EventID[event_type == 14]
    pirho_PM_ID = EventID[event_type == 15]
    pirho_MP_ID = EventID[event_type == 16]
    pirho_PP_ID = EventID[event_type == 17]
    leppi_MM_ID = EventID[event_type == 18]
    leppi_PM_ID = EventID[event_type == 19]
    leppi_MP_ID = EventID[event_type == 20]
    leppi_PP_ID = EventID[event_type == 21]
    leprho_MM_ID = EventID[event_type == 22]
    leprho_PM_ID = EventID[event_type == 23]
    leprho_MP_ID = EventID[event_type == 24]
    leprho_PP_ID = EventID[event_type == 25]
    rhorho_MM_ID = EventID[event_type == 26]
    rhorho_PM_ID = EventID[event_type == 27]
    rhorho_MP_ID = EventID[event_type == 28]
    rhorho_PP_ID = EventID[event_type == 29]
    
    return pipi_jet, pirho_jet, leppi_jet, leprho_jet, qcd_jet, rhorho_jet, tt_jet, wlnu_jet, wtaunu_jet, zll_jet, pipi_MM_jet, pipi_PM_jet, pipi_MP_jet, pipi_PP_jet, pipi_ID, pirho_ID, leppi_ID, leprho_ID, qcd_ID, rhorho_ID, tt_ID, wlnu_ID, wtaunu_ID, zll_ID, pipi_MM_ID, pipi_PM_ID, pipi_MP_ID, pipi_PP_ID, pirho_MM_jet, pirho_PM_jet, pirho_MP_jet, pirho_PP_jet, pirho_MM_ID, pirho_PM_ID, pirho_MP_ID, pirho_PP_ID, leppi_MM_jet, leppi_PM_jet, leppi_MP_jet, leppi_PP_jet, leppi_MM_ID, leppi_PM_ID, leppi_MP_ID, leppi_PP_ID, leprho_MM_jet, leprho_PM_jet, leprho_MP_jet, leprho_PP_jet, leprho_MM_ID, leprho_PM_ID, leprho_MP_ID, leprho_PP_ID, rhorho_MM_jet, rhorho_PM_jet, rhorho_MP_jet, rhorho_PP_jet, rhorho_MM_ID, rhorho_PM_ID, rhorho_MP_ID, rhorho_PP_ID


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
    pipi_jet, pirho_jet, leppi_jet, leprho_jet, qcd_jet, rhorho_jet, tt_jet, wlnu_jet, wtaunu_jet, zll_jet, pipi_MM_jet, pipi_PM_jet, pipi_MP_jet, pipi_PP_jet, pipi_ID, pirho_ID, leppi_ID, leprho_ID, qcd_ID, rhorho_ID, tt_ID, wlnu_ID, wtaunu_ID, zll_ID, pipi_MM_ID, pipi_PM_ID, pipi_MP_ID, pipi_PP_ID, pirho_MM_jet, pirho_PM_jet, pirho_MP_jet, pirho_PP_jet, pirho_MM_ID, pirho_PM_ID, pirho_MP_ID, pirho_PP_ID, leppi_MM_jet, leppi_PM_jet, leppi_MP_jet, leppi_PP_jet, leppi_MM_ID, leppi_PM_ID, leppi_MP_ID, leppi_PP_ID, leprho_MM_jet, leprho_PM_jet, leprho_MP_jet, leprho_PP_jet, leprho_MM_ID, leprho_PM_ID, leprho_MP_ID, leprho_PP_ID, rhorho_MM_jet, rhorho_PM_jet, rhorho_MP_jet, rhorho_PP_jet, rhorho_MM_ID, rhorho_PM_ID, rhorho_MP_ID, rhorho_PP_ID = data_split(total_jet, EventID, event_type)

    if hvd.rank() == 0:
        def save_if_not_empty(jet, ID, sample_name, particle_type, save_empty=False):
            if jet.shape[0] > 0:
                data_dict = {
                    'nu_p': jet[:, :, :3],
                    'nu_m': jet[:, :, 3:],
                    'EventID': ID,
                }
                np.savez(sample_name.replace("merged", particle_type), **data_dict)
                
            else:
                if save_empty:
                    data_dict = {
                        'nu_p': [],
                        'nu_m': [],
                        'EventID': [],
                    }
                    np.savez(sample_name.replace("merged", particle_type), **data_dict)
                else:
                    pass

        save_if_not_empty(pipi_jet, pipi_ID, sample_name, "pi_pi_particles", save_empty=True)
        save_if_not_empty(pirho_jet, pirho_ID, sample_name, "pi_rho_particles")
        save_if_not_empty(leppi_jet, leppi_ID, sample_name, "lep_pi_particles")
        save_if_not_empty(leprho_jet, leprho_ID, sample_name, "lep_rho_particles")
        save_if_not_empty(qcd_jet, qcd_ID, sample_name, "QCD_particles")
        save_if_not_empty(rhorho_jet, rhorho_ID, sample_name, "rho_rho_particles")
        save_if_not_empty(tt_jet, tt_ID, sample_name, "tt_particles")
        save_if_not_empty(wlnu_jet, wlnu_ID, sample_name, "Wlnu_particles")
        save_if_not_empty(wtaunu_jet, wtaunu_ID, sample_name, "Wtaunu_particles")
        save_if_not_empty(zll_jet, zll_ID, sample_name, "Zll_particles")
        save_if_not_empty(pipi_MM_jet, pipi_MM_ID, sample_name, "pi_pi_MM_particles")
        save_if_not_empty(pipi_PM_jet, pipi_PM_ID, sample_name, "pi_pi_PM_particles")
        save_if_not_empty(pipi_MP_jet, pipi_MP_ID, sample_name, "pi_pi_MP_particles")
        save_if_not_empty(pipi_PP_jet, pipi_PP_ID, sample_name, "pi_pi_PP_particles")
        save_if_not_empty(pirho_MM_jet, pirho_MM_ID, sample_name, "pi_rho_MM_particles")
        save_if_not_empty(pirho_PM_jet, pirho_PM_ID, sample_name, "pi_rho_PM_particles")
        save_if_not_empty(pirho_MP_jet, pirho_MP_ID, sample_name, "pi_rho_MP_particles")
        save_if_not_empty(pirho_PP_jet, pirho_PP_ID, sample_name, "pi_rho_PP_particles")
        save_if_not_empty(leppi_MM_jet, leppi_MM_ID, sample_name, "lep_pi_MM_particles")
        save_if_not_empty(leppi_PM_jet, leppi_PM_ID, sample_name, "lep_pi_PM_particles")
        save_if_not_empty(leppi_MP_jet, leppi_MP_ID, sample_name, "lep_pi_MP_particles")
        save_if_not_empty(leppi_PP_jet, leppi_PP_ID, sample_name, "lep_pi_PP_particles")
        save_if_not_empty(leprho_MM_jet, leprho_MM_ID, sample_name, "lep_rho_MM_particles")
        save_if_not_empty(leprho_PM_jet, leprho_PM_ID, sample_name, "lep_rho_PM_particles")
        save_if_not_empty(leprho_MP_jet, leprho_MP_ID, sample_name, "lep_rho_MP_particles")
        save_if_not_empty(leprho_PP_jet, leprho_PP_ID, sample_name, "lep_rho_PP_particles")
        save_if_not_empty(rhorho_MM_jet, rhorho_MM_ID, sample_name, "rho_rho_MM_particles")
        save_if_not_empty(rhorho_PM_jet, rhorho_PM_ID, sample_name, "rho_rho_PM_particles")
        save_if_not_empty(rhorho_MP_jet, rhorho_MP_ID, sample_name, "rho_rho_MP_particles")
        save_if_not_empty(rhorho_PP_jet, rhorho_PP_ID, sample_name, "rho_rho_PP_particles")
        
        # os.system("chmod -R 777 /global/cfs/cdirs/m2616/avencast/Quantum_Entanglement/workspace/results/pi_pi/")
    

def main():
    plot_utils.SetStyle()
    utils.setup_gpus()
    if hvd.rank()==0:logging.info("Horovod and GPUs initialized successfully.")
    flags = parse_arguments()
    # if hvd.rank()==0:os.system("chmod -R 777 /global/cfs/cdirs/m2616/avencast/Quantum_Entanglement/workspace/results/pi_pi/")
    if flags.sample:
        if hvd.rank()==0:logging.info("Sampling the data without boost.")
        test_path_list, test_loader_list, model = load_data_and_model(flags)
        for i in range(len(test_path_list)):
            test_name = test_path_list[i].replace(".pkl", "_eval.npz")
            if os.path.exists(test_name.replace("merged", "pi_pi_particles")):
                continue
            else:
                if hvd.rank()==0:logging.info("Sampling the {}.".format(test_name))
                test = test_loader_list[i]
                sample_data(test, model, flags, test_name)

if __name__ == '__main__':
    main()

