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
    parser.add_argument("--folder", default="/global/cfs/cdirs/m2616/avencast/bbtautau/tautau_reconstruction/out_20250204_eval", help="Folder containing input files")
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


def load_data_and_model(flags):
    
    truth_path = flags.folder

    truth_path_list = glob.glob(truth_path + '/*_eval.pkl')
    test_loader_list = []
    for truth_path in truth_path_list:
        test_loader_list.append(utils.ReconbbtautauDataLoader(truth_path,rank=hvd.rank(),size=hvd.size(), samples_name="hhttbbSM"))
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
    
    # model_name = "/pscratch/sd/b/baihong/data/checkpoints/PET_bbtautau_8_local_layer_scale_token_baseline_generator.weights.h5"
    model_name = "/pscratch/sd/b/baihong/data/checkpoints/PET_hhttbbSM_8_local_layer_scale_token_baseline_generator.weights.h5"
    model.load_weights(model_name)
    return truth_path_list, test_loader_list, model


def sample_data(test, model, flags, sample_name):
    """ Sample data using the model and save to file. """
    part,point,mask,jet,y, truth_TauTau,Tau1, Tau1_nProng, Tau1_decayMode, Tau1_charge, Tau2, Tau2_nProng, Tau2_decayMode, Tau2_charge, Jet_b1, Jet_b2, Jet_0, Jet_1, Jet_2, met, truth_nu1, truth_nu2, mmc, sum_metTauTau, weight_mc, eventNumber, runNumber, n_jets = test.make_eval_data(preprocess=True)
    
    print("part", part.shape)
    print("point", point.shape)
    print("mask", mask.shape)
    print("jet", jet.shape)
    print("met", y.shape)
    nsplit = 50
    total_j = model.generate(nsplit,
                       y,part,point,mask,
                       use_tqdm=hvd.rank()==0)
    
    total_jet = []
    for i in range (total_j.shape[1]):
        total_jet.append(test.revert_preprocess_jet(total_j[:,i]).reshape(-1,1,6))
    total_jet = np.concatenate(total_jet, axis=1)
    total_jet = hvd.allgather(total_jet).numpy()
    truth_TauTau = hvd.allgather(tf.constant(truth_TauTau)).numpy()
    Tau1 = hvd.allgather(tf.constant(Tau1)).numpy()
    Tau1_nProng = hvd.allgather(tf.constant(Tau1_nProng)).numpy()
    Tau1_decayMode = hvd.allgather(tf.constant(Tau1_decayMode)).numpy()
    Tau1_charge = hvd.allgather(tf.constant(Tau1_charge)).numpy()
    Tau2 = hvd.allgather(tf.constant(Tau2)).numpy()
    Tau2_nProng = hvd.allgather(tf.constant(Tau2_nProng)).numpy()
    Tau2_decayMode = hvd.allgather(tf.constant(Tau2_decayMode)).numpy()
    Tau2_charge = hvd.allgather(tf.constant(Tau2_charge)).numpy()
    Jet_b1 = hvd.allgather(tf.constant(Jet_b1)).numpy()
    Jet_b2 = hvd.allgather(tf.constant(Jet_b2)).numpy()
    Jet_0 = hvd.allgather(tf.constant(Jet_0)).numpy()
    Jet_1 = hvd.allgather(tf.constant(Jet_1)).numpy()
    Jet_2 = hvd.allgather(tf.constant(Jet_2)).numpy()
    met = hvd.allgather(tf.constant(met)).numpy()
    truth_nu1 = hvd.allgather(tf.constant(truth_nu1)).numpy()
    truth_nu2 = hvd.allgather(tf.constant(truth_nu2)).numpy()
    mmc = hvd.allgather(tf.constant(mmc)).numpy()
    sum_metTauTau = hvd.allgather(tf.constant(sum_metTauTau)).numpy()
    weight_mc = hvd.allgather(tf.constant(weight_mc)).numpy()
    eventNumber = hvd.allgather(tf.constant(eventNumber)).numpy()
    runNumber = hvd.allgather(tf.constant(runNumber)).numpy()
    n_jets = hvd.allgather(tf.constant(n_jets)).numpy()
    

    if hvd.rank() == 0:
        dict = {
            "truth_TauTau": truth_TauTau,
            "Tau1": Tau1,
            "Tau1_nProng": Tau1_nProng,
            "Tau1_decayMode": Tau1_decayMode,
            "Tau1_charge": Tau1_charge,
            "Tau2": Tau2,
            "Tau2_nProng": Tau2_nProng,
            "Tau2_decayMode": Tau2_decayMode,
            "Tau2_charge": Tau2_charge,
            "Jet_b1": Jet_b1,
            "Jet_b2": Jet_b2,
            "Jet_0": Jet_0,
            "Jet_1": Jet_1,
            "Jet_2": Jet_2,
            "met": met,
            "truth_nu1": truth_nu1,
            "truth_nu2": truth_nu2,
            "mmc": mmc,
            "sum_metTauTau": sum_metTauTau,
            "weight_mc": weight_mc,
            "eventNumber": eventNumber,
            "runNumber": runNumber,
            "n_jets": n_jets,
            "reco_nu1": total_jet[:, :, :3],
            "reco_nu2": total_jet[:, :, 3:]
        }
        with open(sample_name, 'wb') as f:
            pickle.dump(dict, f)
            
def main():
    plot_utils.SetStyle()
    utils.setup_gpus()
    if hvd.rank()==0:logging.info("Horovod and GPUs initialized successfully.")
    flags = parse_arguments()
    if flags.sample:
        if hvd.rank()==0:logging.info("Sampling the data.")
        test_path_list, test_loader_list, model = load_data_and_model(flags)
        for i in range(len(test_path_list)):
            test_name = test_path_list[i].replace("_eval.pkl", "_recon_hhttbb.pkl")
            if os.path.exists(test_name):
                # os.system("chmod -R 777 /global/cfs/cdirs/m2616/avencast/bbtautau/tautau_reconstruction/out_20250204_eval/")
                continue
            else:
                if hvd.rank()==0:logging.info("Sampling the {}.".format(test_name))
                test = test_loader_list[i]
                sample_data(test, model, flags, test_name)
                os.system("chmod -R 777 /global/cfs/cdirs/m2616/avencast/bbtautau/tautau_reconstruction/out_20250204_eval/")
                

if __name__ == '__main__':
    main()

