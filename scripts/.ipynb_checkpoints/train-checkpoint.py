import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import horovod.tensorflow.keras as hvd
import os
import argparse
import logging
import pickle
# Custom local imports
import utils
from PET_jetnet import PET_jetnet
from PET_regression import PET as PET_regression

# Keras imports
from tensorflow.keras.optimizers import schedules, Lion
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

# Initialize Horovod

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train the PET model on JetNet datasets.")
    parser.add_argument("--dataset", type=str, default="pipi", help="Dataset to use")
    parser.add_argument("--folder", type=str, default="/pscratch/sd/b/baihong/data/", help="Folder containing input files")
    parser.add_argument("--mode", type=str, default="generator", help="Loss type to train the model")
    parser.add_argument("--batch", type=int, default=2048, help="Batch size")
    parser.add_argument("--epoch", type=int, default=1000, help="Max epoch")
    parser.add_argument("--lr", type=float, default=1.2e-4, help="Learning rate")
    parser.add_argument("--lr_factor", type=float, default=1, help="Factor to adjust learning rate")
    parser.add_argument("--fine_tune", action='store_true', default=False, help='Fine tune a model')
    parser.add_argument("--local", action='store_true', default=True, help='Use local embedding')
    parser.add_argument("--num_layers", type=int, default=8, help="Number of transformer layers")
    parser.add_argument("--drop_probability", type=float, default=0.0, help="Drop probability")
    parser.add_argument("--simple", action='store_true', default=False, help='Use simplified head model')
    parser.add_argument("--talking_head", action='store_true', default=False, help='Use talking head attention')
    parser.add_argument("--layer_scale", action='store_true', default=False, help='Use layer scale in the residual connections')
    return parser.parse_args()

def get_data_loader(flags):
    if flags.dataset == 'pipi':
        train = utils.TruthTauDataLoader(os.path.join(flags.folder,'NumpyData/', 'baseline/pi_pi_recon_total_train.hdf5'),flags.batch,hvd.rank(),hvd.size(),samples_name="pipi")
        val = utils.TruthTauDataLoader(os.path.join(flags.folder,'NumpyData/', 'baseline/pi_pi_recon_total_test.hdf5'),flags.batch,hvd.rank(),hvd.size(),samples_name="pipi")
    elif flags.dataset == 'epi':
        train = utils.TruthTauDataLoader(os.path.join(flags.folder,'NumpyData/', 'baseline/e_pi_recon_total_train.hdf5'),flags.batch,hvd.rank(),hvd.size(),samples_name="epi")
        val = utils.TruthTauDataLoader(os.path.join(flags.folder,'NumpyData/', 'baseline/e_pi_recon_total_test.hdf5'),flags.batch,hvd.rank(),hvd.size(),samples_name="epi")
    elif flags.dataset == 'erho':
        train = utils.TruthTauDataLoader(os.path.join(flags.folder,'NumpyData/', 'baseline/e_rho_recon_total_train.hdf5'),flags.batch,hvd.rank(),hvd.size(),samples_name="erho")
        val = utils.TruthTauDataLoader(os.path.join(flags.folder,'NumpyData/', 'baseline/e_rho_recon_total_test.hdf5'),flags.batch,hvd.rank(),hvd.size(),samples_name="erho")
    elif flags.dataset == 'mupi':
        train = utils.TruthTauDataLoader(os.path.join(flags.folder,'NumpyData/', 'baseline/mu_pi_recon_total_train.hdf5'),flags.batch,hvd.rank(),hvd.size(),samples_name="mupi")
        val = utils.TruthTauDataLoader(os.path.join(flags.folder,'NumpyData/', 'baseline/mu_pi_recon_total_test.hdf5'),flags.batch,hvd.rank(),hvd.size(),samples_name="mupi")
    elif flags.dataset == 'murho':
        train = utils.TruthTauDataLoader(os.path.join(flags.folder,'NumpyData/', 'baseline/mu_rho_recon_total_train.hdf5'),flags.batch,hvd.rank(),hvd.size(),samples_name="murho")
        val = utils.TruthTauDataLoader(os.path.join(flags.folder,'NumpyData/', 'baseline/mu_rho_recon_total_test.hdf5'),flags.batch,hvd.rank(),hvd.size(),samples_name="murho")
    elif flags.dataset == 'pirho':
        train = utils.TruthTauDataLoader(os.path.join(flags.folder,'NumpyData/', 'baseline/pi_rho_recon_total_train.hdf5'),flags.batch,hvd.rank(),hvd.size(),samples_name="pirho")
        val = utils.TruthTauDataLoader(os.path.join(flags.folder,'NumpyData/', 'baseline/pi_rho_recon_total_test.hdf5'),flags.batch,hvd.rank(),hvd.size(),samples_name="pirho")
    elif flags.dataset == 'rhorho':
        train = utils.TruthTauDataLoader(os.path.join(flags.folder,'NumpyData/', 'baseline/rho_rho_recon_total_train.hdf5'),flags.batch,hvd.rank(),hvd.size(),samples_name="rhorho")
        val = utils.TruthTauDataLoader(os.path.join(flags.folder,'NumpyData/', 'baseline/rho_rho_recon_total_test.hdf5'),flags.batch,hvd.rank(),hvd.size(),samples_name="rhorho")
    elif flags.dataset == 'mix':
        train = utils.TruthTauDataLoader(os.path.join(flags.folder,'NumpyData/', 'baseline/Recon_train.hdf5'),flags.batch,hvd.rank(),hvd.size(),samples_name="mix")
        val = utils.TruthTauDataLoader(os.path.join(flags.folder,'NumpyData/', 'baseline/Recon_test.hdf5'),flags.batch,hvd.rank(),hvd.size(),samples_name="mix")
    elif flags.dataset == 'baseline':
        train = utils.TruthTauDataLoader(os.path.join(flags.folder,'NumpyData/', 'baseline/baseline_train.hdf5'),flags.batch,hvd.rank(),hvd.size(),samples_name="mix")
        val = utils.TruthTauDataLoader(os.path.join(flags.folder,'NumpyData/', 'baseline/baseline_test.hdf5'),flags.batch,hvd.rank(),hvd.size(),samples_name="mix")
    return train, val

def configure_optimizers(flags, train_loader,lr_factor = 1.0):
    scale_lr = flags.lr * np.sqrt(hvd.size())
    lr_schedule = schedules.CosineDecay(
        initial_learning_rate=flags.lr/lr_factor,
        warmup_target=scale_lr/lr_factor,
        warmup_steps=3*train_loader.nevts//flags.batch//hvd.size(),
        decay_steps=flags.epoch*train_loader.nevts//flags.batch//hvd.size(),
    )
    optimizer = Lion(
        learning_rate=lr_schedule,
        clipnorm=1.0,
        beta_1=0.95,
        beta_2=0.99
    )
    return hvd.DistributedOptimizer(optimizer)

class LossHistory(keras.callbacks.Callback):
    def __init__(self, dataset_name):
        super(LossHistory, self).__init__()
        self.dataset_name = dataset_name
        self.losses = []

    def on_train_begin(self, logs={}):
        self.losses = []


    def on_epoch_end(self, epoch, logs=None):
        print_loss = logs.get('loss')
        val_loss = logs.get('val_loss')
        print_loss = str(print_loss)
        val_loss = str(val_loss)
        with open("./logs_{}/loss.csv".format(self.dataset_name), 'a+') as f:
            f.write(print_loss + ',' + val_loss)
            f.write('\n')

def main():
    utils.setup_gpus()
    flags = parse_arguments()

    train_loader, val_loader = get_data_loader(flags)

    if flags.fine_tune:
        model_name = utils.get_model_name(flags,flags.fine_tune).replace(flags.dataset,'pipi_base').replace('fine_tune','baseline')
        model_name = os.path.join(flags.folder,'checkpoints',model_name)
    else:
        model_name = None

    if flags.dataset == 'pipi_regression':
        model = PET_regression(num_feat=train_loader.num_feat,
                        num_jet=train_loader.num_jet,
                        num_classes=train_loader.num_classes,
                        local=flags.local,
                        num_layers=flags.num_layers,
                        drop_probability=flags.drop_probability,
                        simple=flags.simple, layer_scale=flags.layer_scale,
                        talking_head=flags.talking_head,
                        mode=flags.mode
                        )
    else:
        model = PET_jetnet(num_feat=train_loader.num_feat,
                        num_jet=train_loader.num_jet,
                        num_classes=train_loader.num_classes,
                        num_part=train_loader.num_part,
                        local=flags.local,
                        num_layers=flags.num_layers,
                        drop_probability=flags.drop_probability,
                        simple=flags.simple, layer_scale=flags.layer_scale,
                        talking_head=flags.talking_head,
                        mode=flags.mode,
                        model_name = model_name,
                        fine_tune=flags.fine_tune
                        )

    optimizer_body = configure_optimizers(flags, train_loader, lr_factor=flags.lr_factor if flags.fine_tune else 1)
    optimizer_head = configure_optimizers(flags, train_loader, lr_factor=flags.lr_factor if flags.fine_tune else 1)
    model.compile(optimizer_body, optimizer_head)
    history = LossHistory(flags.dataset)
    callbacks = [
        hvd.callbacks.BroadcastGlobalVariablesCallback(0),
        hvd.callbacks.MetricAverageCallback(),
        EarlyStopping(patience=45, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', patience=15, min_lr=1e-8, epsilon = 1e-4)]

    checkpoint_name = utils.get_model_name(flags, flags.fine_tune)
    checkpoint_path = os.path.join(flags.folder, 'checkpoints', checkpoint_name)
    if os.path.exists(checkpoint_path):
        model.load_weights(checkpoint_path)
        print("Loaded weights from ", checkpoint_path)
    
    if hvd.rank() == 0:
        print("Checkpoint name: ", checkpoint_name)
        
        checkpoint_callback = ModelCheckpoint(checkpoint_path,
                                              save_best_only=True, mode='auto',
                                              save_weights_only=True,
                                              period=1)
        callbacks.append(checkpoint_callback)
        callbacks.append(history)
        
        
    hist = model.fit(train_loader.make_tfdata(),
                     epochs=flags.epoch,
                     validation_data=val_loader.make_tfdata(),
                     batch_size=flags.batch,
                     callbacks=callbacks,
                     steps_per_epoch=train_loader.steps_per_epoch,
                     validation_steps=val_loader.steps_per_epoch,
                     verbose=hvd.rank() == 0,
                     )
    
    if hvd.rank() ==0:
        with open(os.path.join(flags.folder,'histories',utils.get_model_name(flags,flags.fine_tune).replace(".weights.h5",".pkl")),"wb") as f:
            pickle.dump(hist.history, f)

if __name__ == "__main__":
    main()
