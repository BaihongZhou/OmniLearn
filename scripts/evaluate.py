import os
import h5py as h5
import numpy as np
import tensorflow as tf
from tensorflow import keras
from dummy_hvd import hvd as hvd
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
    parser.add_argument("--dataset", default="pipi", help="Folder containing input files")
    parser.add_argument("--folder", default="/global/cfs/cdirs/m2616/avencast/Quantum_Entanglement/workspace_20250302_sig_apply/results/", help="Folder containing input files")
    parser.add_argument("--mode", default="generator", help="Loss type to train the model: [all/classifier/generator]")
    parser.add_argument("--fine_tune", action='store_true', help="Fine tune a model")
    parser.add_argument("--local", action='store_true', help="Use local embedding")
    parser.add_argument("--num_layers", type=int, default=8, help="Number of transformer layers")
    parser.add_argument("--drop_probability", type=float, default=0.0, help="Stochastic Depth drop probability")
    parser.add_argument("--simple", action='store_true', help="Use simplified head model")
    parser.add_argument("--batch", type=int, default=512, help="Batch size")
    parser.add_argument("--talking_head", action='store_true', help="Use talking head attention")
    parser.add_argument("--layer_scale", action='store_true', help="Use layer scale in the residual connections")
    parser.add_argument("--sample", action='store_true', default=False, help="Sample from trained model")
    parser.add_argument("--plot_folder", default="../plots", help="Folder to save the outputs")
    return parser.parse_args()

def get_data_info(flags):
    if flags.dataset == 'pipi':
        val = utils.TruthTauDataLoader(os.path.join(flags.folder, 'pi_pi_recon_total_train.hdf5'),flags.batch,hvd.rank(),hvd.size(),samples_name="pipi")
    return val
        
def load_data_and_model(flags):
    if flags.dataset == 'pipi':
        test = utils.TruthTauDataLoader(os.path.join(flags.folder, 'pi_pi_recon_total_train.hdf5'),flags.batch,hvd.rank(),hvd.size(),samples_name="pipi")
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

    model_name = "/global/cfs/projectdirs/m2616/baihong/shared/OmniLearn/PET_pipi_8_local_layer_scale_token_baseline_generator.weights.h5"
    model.load_weights(model_name)
    return test, model


def sample_data(test, model, flags, sample_name="pipi_recon"):
    """ Sample data using the model and save to file. """
    part,point,mask,jet,met = test.make_eval_data(preprocess=True)
    
    nsplit = 50
    total_j = model.generate(nsplit,
                       met,part,point,mask,
                       use_tqdm=hvd.rank()==0)
    
    total_jet = []
    for i in range (total_j.shape[1]):
        total_jet.append(test.revert_preprocess_jet(total_j[:,i]).reshape(-1,1,6))
    total_jet = np.concatenate(total_jet, axis=1)

    if hvd.rank() == 0:
        dict = {
            'nu_p':total_jet[:,:,:3],
            'nu_m':total_jet[:,:,3:]
        }
        np.savez(sample_name, **dict)

    

def main():
    plot_utils.SetStyle()
    utils.setup_gpus()
    if hvd.rank()==0:logging.info("Horovod and GPUs initialized successfully.")
    flags = parse_arguments()
    
    if flags.sample:
        if hvd.rank()==0:logging.info("Sampling the data with boost samples.")
        test, model = load_data_and_model(flags)
        sample_data(test, model, flags)

if __name__ == '__main__':
    main()

