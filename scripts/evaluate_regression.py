import os
import h5py as h5
import numpy as np
import tensorflow as tf
from tensorflow import keras
import horovod.tensorflow.keras as hvd
import argparse
import pickle
from PET_regression import PET
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
    parser.add_argument("--folder", default="/global/cfs/cdirs/m2616/avencast/Quantum_Entanglement/workspace.new/results/", help="Folder containing input files")
    # parser.add_argument("--folder", default="/pscratch/sd/b/baihong/data/NumpyData/reco/", help="Folder containing input files")
    parser.add_argument("--mode", default="regression", help="Loss type to train the model: [all/classifier/generator]")
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
            test_loader_list.append(utils.RecoTauRegression(truth_path,hvd.rank(),hvd.size(),nevts=10000))
        
        return test_path_list, test_loader_list


def load_data_and_model(flags):
    
    truth_path = flags.folder
    # truth_path_list = glob.glob(truth_path + '*_particles.pkl')
    # truth_path_list = glob.glob(truth_path + 'OmniLearn_pi_pi_recon_particles.pkl')
    # truth_path_list = glob.glob(truth_path + '*/ml_export/*_recon.npz')
    truth_path_list = glob.glob(truth_path + '**/ml_export/OmniLearn_pi_pi_recon.npz')
    test_loader_list = []
    for truth_path in truth_path_list:
        test_loader_list.append(utils.RecoTauRegression(truth_path,rank=hvd.rank(),size=hvd.size()))
    test = test_loader_list[0]
    model = PET(num_feat=test.num_feat,
                       num_jet=test.num_jet,
                       num_classes=test.num_classes,
                       local=flags.local,
                       num_layers=flags.num_layers,
                       drop_probability=flags.drop_probability,
                       simple=flags.simple, layer_scale=flags.layer_scale,
                       talking_head=flags.talking_head,
                       mode=flags.mode)
    
    model_name = "/pscratch/sd/b/baihong/data/checkpoints/PET_pipi_regression_8_local_layer_scale_token_baseline_regression.weights.h5"
    model.load_weights(model_name)
    return truth_path_list, test_loader_list, model


def sample_data(test, model, flags, sample_name):
    """ Sample data using the model and save to file. """
    part,point,mask,jet,y,EventID,event_type, pion = test.make_eval_data(preprocess=True)
    X = tf.data.Dataset.from_tensor_slices(
            {'input_features':part,
             'input_points':point,
             'input_mask':mask,
             'input_jet':jet,
             'input_time':tf.zeros(jet.shape[0],1)})
    X = X.cache().batch(512).prefetch(tf.data.AUTOTUNE)
    
    pred = tf.constant(model.predict(X, verbose=hvd.rank() == 0)[0]).numpy()
    y = tf.constant(y).numpy()
    y = hvd.allgather(y)
    pred = hvd.allgather(pred)
    

    if hvd.rank() == 0:
        dict = {
            'truth_label': y,
            'pred_label': pred
        }
        np.savez(sample_name, **dict)
            
            
def get_generated_data(sample_name):
    data = np.load(sample_name)
    gen = data['pred_label']
    truth = data['truth_label']
        
    return gen, truth



def plot(jet1,jet2,title,plot_folder):
    var_names = [r'$cos \theta_A^n $', r'$cos \theta_A^k $', r'$cos \theta_A^r$', r'$cos \theta_B^n $', r'$cos \theta_B^k $', r'$cos \theta_B^r$']
    
    for ivar in range(len(var_names)):                    
        feed_dict = {
            'true': jet2[:,ivar],
            'gen':jet1[:,ivar]}
        

        fig,gs,binning = plot_utils.HistRoutine(feed_dict,xlabel=var_names[ivar],
                                                plot_ratio=True,
                                                reference_name='true',
                                                ylabel= 'Normalized entries')
        ax0 = plt.subplot(gs[0])     
        fig.savefig('{}/Cos_{}_{}.pdf'.format(plot_folder,title,ivar),bbox_inches='tight')

            
def plot_results(jets, jets_truth, flags):
    """ Plot the results using the utility functions. """
    plot(jets, jets_truth, title='Cos', plot_folder=flags.plot_folder)
    

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
            test_name = test_path_list[i].replace("_recon.npz", "_recon_regression.npz")
            # test_name = test_path_list[i].replace(".pkl", "_eval.npz")
            if os.path.exists(test_name):
                continue
            else:
                if hvd.rank()==0:logging.info("Sampling the {}.".format(test_name))
                test = test_loader_list[i]
                sample_data(test, model, flags, test_name)
    else:
        if hvd.rank()==0:logging.info("Ploting the distributions.")
        # Load and process data, generate plots, etc.        
        test_path_list, test_loader_list, model = load_data_and_model(flags)
        for i in range(len(test_path_list)):
            sample_name = test_path_list[i].replace("_recon.npz", "_recon_regression.npz")
            gen, truth = get_generated_data(sample_name)
            # Plot results
            plot_results(gen, truth, flags)

if __name__ == '__main__':
    main()

