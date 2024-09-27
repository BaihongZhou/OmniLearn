import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn import metrics
from dummy_hvd import hvd as hvd
import argparse
import sys
import gc

import h5py
from PET import PET
import utils
# from omnifold import Classifier

def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate performance metrics for trained models on various datasets.")
    parser.add_argument("--dataset", type=str, default="top", help="Folder containing input files")
    parser.add_argument("--folder", type=str, default="/lustre/collider/zhoubaihong/QE_study/Leptonic_ML/data/HDF5/", help="Folder containing input files")
    parser.add_argument("--batch", type=int, default=250, help="Batch size")
    parser.add_argument("--load", action='store_true', help="Load pre-evaluated npy files")
    parser.add_argument("--save-pred", action='store_true', help="Save the prediction values to a separate file")
    parser.add_argument("--mode", type=str, default="regression", help="Loss type to train the model")
    parser.add_argument("--fine_tune", action='store_true', help="Fine tune a model")
    parser.add_argument("--nid", type=int, default=0, help="Training ID for multiple trainings")
    parser.add_argument("--local", action='store_true', default=True, help="Use local embedding")
    parser.add_argument("--num_layers", type=int, default=8, help="Number of transformer layers")
    parser.add_argument("--drop_probability", type=float, default=0.0, help="Stochastic Depth drop probability")
    parser.add_argument("--simple", action='store_true', help="Use simplified head model")
    parser.add_argument("--talking_head", action='store_true', default=True, help="Use talking head attention")
    parser.add_argument("--layer_scale", action='store_true', default=True, help="Use layer scale in the residual connections")
    args = parser.parse_args()
    return args

def print_metrics(y_pred, y, thresholds, multi_label=False):
    if multi_label:
        print("AUC: {}".format(metrics.roc_auc_score(y, y_pred,average='macro',multi_class='ovo')))
        
        one_hot_predictions = np.zeros_like(y_pred)
        one_hot_predictions[np.arange(len(y_pred)), y_pred.argmax(axis=-1)] = 1
        
        print('Acc: {}'.format(metrics.accuracy_score(y,one_hot_predictions)))    
        
        bkg_idx = 0
        
        for idx in range(np.shape(y)[-1]):
            if idx == bkg_idx:continue
            mask = (y[:,idx]==1) | (y[:,bkg_idx]==1) #only keep signal+bkg
            pred_sb = y_pred[mask,idx]/(y_pred[mask,idx] + y_pred[mask,bkg_idx])
            fpr, tpr, _ = metrics.roc_curve(y[mask,idx], pred_sb)
            
            for threshold in thresholds:
                bineff = np.argmax(tpr>threshold)
                print('Class {} effS at {} 1.0/effB = {}'.format(idx,tpr[bineff],1.0/fpr[bineff]))

    else:
        print("AUC: {}".format(metrics.roc_auc_score(y, y_pred)))
        print('Acc: {}'.format(metrics.accuracy_score(y,y_pred>0.5)))
        
        fpr, tpr, _ = metrics.roc_curve(y, y_pred)

        for threshold in thresholds:
            bineff = np.argmax(tpr>threshold)
            print('effS at {} 1.0/effB = {}'.format(tpr[bineff],1.0/fpr[bineff]))

        #Avoid statistical fluctuations
        tpr=tpr[fpr>1e-4]
        fpr=fpr[fpr>1e-4]
        sic = np.ma.divide(tpr,np.sqrt(fpr)).filled(0)
        print("Max SIC: {}".format(np.max(sic)))


def get_data_info(flags):
    multi_label = False
    if flags.dataset == 'top':
        test = utils.TopDataLoader(os.path.join(flags.folder,'newh5/', 'test.hdf5'),
                                   flags.batch,rank = hvd.rank(),size = hvd.size())
        threshold = []
        folder_name = 'TOP'
        
    if flags.dataset == 'opt':
        test = utils.TopDataLoader(os.path.join(flags.folder,'Opt', 'test_ttbar.h5'),
                                   flags.batch,rank = hvd.rank(),size = hvd.size())
        threshold = [0.3, 0.5]
        folder_name = 'Opt'

    elif flags.dataset == 'qg':
        test = utils.QGDataLoader(os.path.join(flags.folder,'QG', 'test_qg.h5'),
                                  flags.batch,rank = hvd.rank(),size = hvd.size())
        threshold = [0.3,0.5]
        folder_name = 'QG'

    elif flags.dataset == 'atlas':
        test = utils.AtlasDataLoader(os.path.join(flags.folder,'ATLASTOP', 'test_atlas.h5'),
                                     flags.batch,rank = hvd.rank(),size = hvd.size())
        threshold = [0.5,0.8]
        folder_name = 'ATLASTOP'
        multi_label = False
    elif flags.dataset == 'atlas_small':
        test = utils.AtlasDataLoader(os.path.join(flags.folder,'ATLASTOP', 'test_atlas.h5'),
                                     flags.batch,rank = hvd.rank(),size = hvd.size())
        threshold = [0.5,0.8]
        folder_name = 'ATLASTOP'
        multi_label = False
    elif flags.dataset == 'h1':
        test = utils.H1DataLoader(os.path.join(flags.folder,'H1', 'test.h5'),
                                  flags.batch,rank = hvd.rank(),size = hvd.size())
        threshold = [0.5,0.1]
        folder_name = 'H1'

    elif flags.dataset == 'cms':
        test = utils.CMSQGDataLoader(os.path.join(flags.folder,'CMSQG', 'test_qgcms_pid.h5'),
                                     flags.batch,rank = hvd.rank(),size = hvd.size())
        threshold = [0.5,0.8]
        folder_name = 'CMSQG'
        
    elif flags.dataset == 'jetclass':
        test = utils.JetClassDataLoader(os.path.join(flags.folder,'JetClass','test',
                                                     rank = hvd.rank(),size = hvd.size()),
                                        flags.batch)
        threshold = [0.5]
        folder_name = 'JetClass/test'

    return test,multi_label,threshold,folder_name

def get_model_function(dataset):
    if 'atlas' in dataset:
        return PET, ''
    else:
        return PET, ''
    
def configure_optimizers(flags, train_loader, lr_factor=1.0):
    scale_lr = 0.001 * np.sqrt(hvd.size())
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=0.001/lr_factor,
        warmup_target = scale_lr/lr_factor,
        warmup_steps= flags.warm_epoch*train_loader.nevts//flags.batch//hvd.size(),
        decay_steps=flags.epoch*train_loader.nevts//flags.batch//hvd.size(),
    )
    optimizer = keras.optimizers.Lion(
        learning_rate=lr_schedule,
        weight_decay=flags.wd*lr_factor,
        beta_1=flags.b1,
        beta_2=flags.b2)
    return hvd.DistributedOptimizer(optimizer)

def load_or_evaluate_model(flags, test,folder_name):
    npy_file = os.path.join(flags.folder,folder_name,'npy','{}'.format(
        utils.get_model_name(
            flags,fine_tune=flags.fine_tune,
            add_string = '_{}'.format(flags.nid) if flags.nid > 0 else '').replace('.h5','.npy')))
    
    if flags.load:
        print("Loading saved npy files")
        data = np.load(npy_file, allow_pickle=True)
        return data.item()['y'], data.item()['pred']
    else:
        model_function, activation = get_model_function(flags.dataset)
        model = model_function(num_feat=test.num_feat,
                               num_jet=test.num_jet, num_classes=test.num_classes,
                               local=flags.local, num_layers=flags.num_layers,
                               drop_probability=flags.drop_probability,
                               simple=flags.simple, layer_scale=flags.layer_scale,
                               talking_head=flags.talking_head,
                               mode="regression")
        # optimizer_head = configure_optimizers(flags, test)
        # optimizer_body = configure_optimizers(flags, test)
        # model.compile(optimizer_body,optimizer_head)
        # model.summary()
        # print("The model is:")
        # print(model)
        X, y = test.make_eval_data()
        # print("Input varis is")
        # print(X.shape)
        # print("Output varis is")
        # print(y.shape)
        if flags.nid>0:
            #Load alternative runs
            add_string = '_{}'.format(flags.nid)
        else:
            add_string = ''

        model.load_weights(os.path.join(
            flags.folder,'checkpoints',
            utils.get_model_name(flags,fine_tune=flags.fine_tune,add_string=add_string)),by_name=True,skip_mismatch=True)
        # Check if there is nan in the weights;
        for layer in model.layers:
            weights = layer.get_weights()
            for w in weights:
                if np.isnan(w).any():
                    print("Layer {} contains NaN values".format(layer.name))
        # model.load_weights('/lustre/collider/zhoubaihong/QE_study/Leptonic_ML/data/HDF5/checkpoints/PET_top_8_local_layer_scale_token_baseline_classifier.weights.h5',by_name=True,skip_mismatch=True)
        
        y = tf.constant(y).numpy()
        pred = tf.constant(model.predict(X, verbose=hvd.rank() == 0)[0]).numpy()
        if hvd.rank()==0:
            if not os.path.exists(os.path.join(flags.folder,folder_name,'npy')):
                os.makedirs(os.path.join(flags.folder,folder_name,'npy'))
            np.save(npy_file,{'y':y,'pred':pred})

        
        return y, pred

def main():
    utils.setup_gpus()
    flags = parse_arguments()

    test,multi_label,thresholds,folder_name = get_data_info(flags)

    y, pred = load_or_evaluate_model(flags, test,folder_name)
    trutuh_lep = test.jet[:,9:14]
    truth_b = test.jet[:,0:4]
    # if flags.save_pred:
    add_text = 'fine_tune' if flags.fine_tune else 'baseline'
    with h5py.File('{}_{}.h5'.format(flags.dataset,add_text), "w") as fh5:
        dset = fh5.create_dataset('y', data=y)
        dset = fh5.create_dataset('pred', data=pred)
        dset = fh5.create_dataset('truth_lep', data=trutuh_lep)
        dset = fh5.create_dataset('truth_b', data=truth_b)
            
    # Evaluate results
    # print_metrics(pred, y, thresholds, multi_label=multi_label)

if __name__ == '__main__':
    main()
